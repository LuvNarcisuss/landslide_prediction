"""
缺失值插补模块
对清洗后的数据做：
1. NDVI/NDWI 多变量插补（MICE + KNN融合）
2. 降雨数据插补（空间相似性 + IterativeImputer）
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor


# 步骤1 - NDVI 随机森林插补（主方案）
def impute_ndvi_rf(df, verbose=True):
    """
    用随机森林预测缺失的NDVI值。
    预测特征：elevation + tpi + distance_to_river + lon + lat + landuse
    原理：植被指数与地形（海拔、地形位置）和土地利用强相关
    """
    has_ndvi = df['ndvi'].notna()
    n_missing = (~has_ndvi).sum()

    if n_missing == 0:
        if verbose:
            print("NDVI 无缺失，跳过插补")
        return df

    # 选取与植被相关的特征
    feature_cols = ['elevation', 'tpi', 'distance_to_river', 'lon', 'lat', 'landuse']

    # 只使用有完整数据的行训练
    train_mask = has_ndvi & df[feature_cols].notna().all(axis=1)
    predict_mask = ~has_ndvi & df[feature_cols].notna().all(axis=1)

    n_train = train_mask.sum()
    n_predict = predict_mask.sum()

    if n_train < 50 or n_predict == 0:
        if verbose:
            print(f"  训练样本不足 ({n_train}) 或无需预测 ({n_predict})，用中位数填充")
        df.loc[~has_ndvi, 'ndvi'] = df.loc[has_ndvi, 'ndvi'].median()
        return df

    try:
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        rf.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, 'ndvi'])
        df.loc[predict_mask, 'ndvi'] = rf.predict(df.loc[predict_mask, feature_cols])

        # 对极端值裁剪
        lo, hi = df.loc[has_ndvi, 'ndvi'].quantile([0.01, 0.99])
        df['ndvi'] = df['ndvi'].clip(lo, hi)

        r2 = rf.score(df.loc[train_mask, feature_cols], df.loc[train_mask, 'ndvi'])
        if verbose:
            print(f"NDVI 随机森林插补: {n_missing} → 完成 (训练R²={r2:.3f})")

    except Exception as e:
        if verbose:
            print(f"  NDVI RF插补失败 ({e})，用中位数填充")
        df.loc[~has_ndvi, 'ndvi'] = df.loc[has_ndvi, 'ndvi'].median()

    return df


# 步骤2 - NDWI 利用 NDVI 线性回归插补
def impute_ndwi_from_ndvi(df, verbose=True):
    """
    NDWI 与 NDVI 高度负相关（植被覆盖越高，水体指数越低）。
    用已有 NDVI-NDWI 对拟合线性模型，再预测缺失的 NDWI。
    """
    has_both = df['ndvi'].notna() & df['ndwi'].notna()
    has_ndvi_no_ndwi = df['ndvi'].notna() & df['ndwi'].isna()

    n_missing = df['ndwi'].isna().sum()

    if n_missing == 0:
        if verbose:
            print("NDWI 无缺失，跳过插补")
        return df

    # 方案1: 有NDVI时用线性回归预测NDWI
    if has_both.sum() > 100 and has_ndvi_no_ndwi.sum() > 0:
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(df.loc[has_both, ['ndvi']], df.loc[has_both, 'ndwi'])
        df.loc[has_ndvi_no_ndwi, 'ndwi'] = lr.predict(
            df.loc[has_ndvi_no_ndwi, ['ndvi']]
        ).clip(-1, 1)

    # 方案2: 仍然缺失的用NDVI和地形做RF插补
    still_missing = df['ndwi'].isna()
    if still_missing.sum() > 0:
        feature_cols = ['elevation', 'tpi', 'lon', 'lat', 'landuse', 'ndvi']
        train_mask = ~still_missing & df[feature_cols].notna().all(axis=1)
        if train_mask.sum() > 50:
            try:
                rf = RandomForestRegressor(
                    n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
                )
                rf.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, 'ndwi'])
                df.loc[still_missing, 'ndwi'] = rf.predict(
                    df.loc[still_missing, feature_cols]
                ).clip(-1, 1)
            except Exception:
                df.loc[still_missing, 'ndwi'] = df.loc[~still_missing, 'ndwi'].median()
        else:
            df.loc[still_missing, 'ndwi'] = df.loc[~still_missing, 'ndwi'].median()

    if verbose:
        print(f"NDWI 插补: {n_missing} → 完成")

    return df


# 步骤3 - 降雨数据 MICE 多变量插补
def impute_rain_mice(df, verbose=True):
    """
    降雨（rain_3d/7d/30d/api）用 MICE 多变量插补。
    利用：高程+经纬度 与降雨的空间相关性
    """
    rain_cols = ['rain_3d', 'rain_7d', 'rain_30d', 'api']
    n_missing = df[rain_cols].isna().any(axis=1).sum()
    total = len(df)

    if n_missing == 0:
        if verbose:
            print("降雨数据无缺失，跳过插补")
        return df

    if verbose:
        print(f"降雨数据缺失: {n_missing}/{total} ({n_missing/total*100:.1f}%)")

    # 用于插补的辅助特征
    aux_cols = ['elevation', 'lon', 'lat']

    # 先用中位数填充让 MICE 更稳定
    for col in rain_cols:
        df[col] = df[col].fillna(df[col].median())

    # 如果缺失率 > 5%，尝试 MICE
    if n_missing / total > 0.05:
        try:
            mice_cols = rain_cols + aux_cols
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
                max_iter=30, random_state=42, sample_posterior=False,
                initial_strategy='median'
            )
            imputed = imputer.fit_transform(df[mice_cols])
            for i, col in enumerate(rain_cols):
                df[col] = imputed[:, i]

            if verbose:
                print(f"  MICE 多变量插补完成 ({len(rain_cols)}个降雨变量)")

        except Exception as e:
            if verbose:
                print(f"  MICE 插补失败 ({e})，保持中位数填充")
    else:
        if verbose:
            print(f"  缺失率低 ({(n_missing/total)*100:.1f}%)，使用中位数填充")

    return df


# 步骤4 - 综合插补入口
def impute_all(df, verbose=True):
    """执行全部缺失值插补"""
    if verbose:
        print("")
        print("缺失值插补:")

    # NDVI 插补（最重要，43%缺失）
    df = impute_ndvi_rf(df, verbose)

    # NDWI 插补
    df = impute_ndwi_from_ndvi(df, verbose)

    # 降雨数据插补
    df = impute_rain_mice(df, verbose)

    # 最后检查是否有任何残留空值
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        if verbose:
            print(f"\n残留空值填充（中位数）:")
        for col in remaining.index:
            if df[col].dtype.kind in 'fc':
                df[col] = df[col].fillna(df[col].median())
                if verbose:
                    print(f"  {col}: {int(remaining[col])} 个 → 中位数填充")

    if verbose:
        print(f"缺失值插补完成，最终形状: {df.shape}")
        print("")

    return df


if __name__ == "__main__":
    from data_cleaning import clean_data
    df = clean_data(verbose=False)
    df = impute_all(df, verbose=True)
    print(f"插补后空值总数: {df.isnull().sum().sum()}")
