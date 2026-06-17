"""
Pipeline — 完整滑坡易发性评价流程

方法论说明：
  本 pipeline 遵循滑坡风险评价的"二阶段"范式：
    阶段1 - 滑坡易发性（Susceptibility）：仅用地形+降雨+植被+土地利用特征分类
    阶段2 - 滑坡风险（Risk）：在易发性基础上叠加入口暴露特征

  这种分离避免暴露特征（威胁人口/财产）成为"捷径特征"，
  确保模型学习真实的地质-气象滑坡规律。

输出格式与现有 preprocess_data() 兼容：
    X_train_scaled, X_test_scaled, y_train, y_test

用法：
    from pre_process.pipeline import preprocess_data
    X_train, X_test, y_train, y_test, extra = preprocess_data()
    # extra['exposure_train'] 包含暴露度特征（不参与分类，用于风险分析）
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pre_process.data_cleaning import clean_data
from pre_process.imputation import impute_all
from pre_process.negative_sampling import generate_hybrid_negatives, validate_negative_samples, rf_quality_check

warnings.filterwarnings('ignore')


# 步骤1 - 特征工程
SUSCEPTIBILITY_FEATURES = [
    # 地形因子 (4)
    'elevation', 'aspect', 'tpi', 'distance_to_river',
    # 降雨因子 (4)
    'rain_3d', 'rain_7d', 'rain_30d', 'api',
    # 植被因子 (2)
    'ndvi', 'ndwi',
    # 坐标 (2)
    'lon', 'lat',
    # 土地利用
    'landuse',
    # 新增特征 (7)
    'rain_ratio_3d_30d', 'rain_ratio_7d_30d', 'rain_intensity',
    'valley_risk', 'elev_rain', 'aspect_rain', 'ndvi_elev',
    'log_dist_river',
]

EXPOSURE_FEATURES = [
    'threatened_people', 'threatened_households', 'threatened_property',
    'exposure_index',
]


def build_susceptibility_features(df, verbose=True):
    """
    构建滑坡易发性建模特征集（不含暴露特征）

    返回: (df_with_features, feature_list, exposure_dataframe)
    """
    if verbose:
        print("")
        print("特征工程:")

    eps = 1e-8

    # 1. 新建衍生特征
    # 1a. 降雨累积强度比
    df['rain_ratio_3d_30d'] = df['rain_3d'] / (df['rain_30d'] + eps)
    df['rain_ratio_7d_30d'] = df['rain_7d'] / (df['rain_30d'] + eps)
    df['rain_intensity'] = df['rain_3d'] / (df['rain_7d'] + eps)

    # 1b. 地形交互特征
    df['valley_risk'] = np.maximum(0, -df['tpi']) / (df['distance_to_river'] + 1)
    df['elev_rain'] = df['elevation'] * np.log1p(np.maximum(df['rain_3d'], 0))
    df['aspect_rain'] = np.cos(np.radians(df['aspect'])) * df['rain_3d']
    df['ndvi_elev'] = df['ndvi'] * df['elevation'] / 1000.0

    # 1c. 距河距离对数变换（保护负值）
    df['log_dist_river'] = np.log1p(np.maximum(df['distance_to_river'], 0))

    # 2. 综合暴露指数（仅用于风险分析，不加入分类特征）
    for col in ['threatened_people', 'threatened_households', 'threatened_property']:
        if col not in df.columns:
            df[col] = 0.0
    max_ppl = max(df['threatened_people'].max(), 1.0)
    max_hh = max(df['threatened_households'].max(), 1.0)
    max_prop = max(df['threatened_property'].max(), 1.0)
    df['exposure_index'] = (
        0.4 * df['threatened_people'] / max_ppl +
        0.3 * df['threatened_households'] / max_hh +
        0.3 * df['threatened_property'] / max_prop
    )

    # 3. landuse_type One-hot编码
    onehot_cols = []
    if 'landuse_type' in df.columns:
        landuse_dummies = pd.get_dummies(df['landuse_type'], prefix='lu', drop_first=True)
        df = pd.concat([df, landuse_dummies], axis=1)
        onehot_cols = landuse_dummies.columns.tolist()
        if verbose:
            print(f"  landuse_type One-hot: {len(onehot_cols)} 个哑变量")

    # 4. 最终特征列表 = 基础易发性特征 + One-hot
    final_features = [c for c in SUSCEPTIBILITY_FEATURES if c in df.columns]
    final_features.extend(onehot_cols)

    if verbose:
        terrain_cols = 4
        rain_cols = 4
        veg_cols = 2
        coord_cols = 2
        derived_cols = 7
        print(f"  特征总数: {len(final_features)}")
        print(f"  组成: 地形{terrain_cols} + 降雨{rain_cols} + 植被{veg_cols}"
              f" + 坐标{coord_cols} + landuse"
              f" + {derived_cols}个衍生特征 + {len(onehot_cols)}个One-hot")
        print(f"  (注: 暴露特征不参与分类，用于风险分析)")

    return df, final_features


# 步骤2 - 类别编码
def encode_categorical(df, verbose=True):
    """对元数据字段进行数值编码（不加入分类特征）"""
    if verbose:
        print("")
        print("类别编码:")

    if 'hazard_type' in df.columns:
        hazard_map = {'滑坡': 1, '不稳定斜坡': 2, '泥石流': 3}
        df['hazard_code'] = df['hazard_type'].map(hazard_map).fillna(0)

    if 'danger_grade' in df.columns:
        danger_map = {'小': 0, '中': 1, '大': 2, '特大': 3}
        df['danger_code'] = df['danger_grade'].map(danger_map).fillna(0)

    if 'triggering_factor' in df.columns:
        df['has_rain'] = df['triggering_factor'].str.contains('降雨', na=False).astype(int)
        df['has_earthquake'] = df['triggering_factor'].str.contains('地震', na=False).astype(int)
        df['has_human'] = df['triggering_factor'].str.contains('人为', na=False).astype(int)

    return df


# 步骤3 - 主预处理函数
def preprocess_data(neg_ratio=1.0, test_size=0.2, random_state=42,
                    min_dist_km=0.8, hybrid_ratio=0.3,
                    run_quality_check=True, verbose=True):
    """
    完整预处理流程

    参数:
    - neg_ratio: 负样本/正样本比例 (默认1.0)
    - test_size: 测试集比例
    - random_state: 随机种子
    - min_dist_km: 空间采样缓冲区距离km (默认0.8)
    - hybrid_ratio: 混合负样本中真实数据占比 (默认0.3)
    - run_quality_check: 是否运行RF质量检验 (默认True)

    返回:
    - X_train, X_test, y_train, y_test: 易发性分类特征
    - extra: 附加信息（含暴露特征/exposure_train/feature_names等）
    """
    if verbose:
        print("滑坡易发性评价 Pipeline")
        print("（易发性·暴露度二阶段分离）")

    # ── 阶段1: 数据清洗 ──
    if verbose:
        print("")
        print("阶段1: 数据清洗")

    df_pos = clean_data(verbose=verbose)

    # ── 阶段2: 缺失值插补 ──
    df_pos = impute_all(df_pos, verbose=verbose)

    # ── 阶段3: 易发性特征工程 ──
    df_pos, sus_features = build_susceptibility_features(df_pos, verbose=verbose)

    # ── 类别编码 ──
    df_pos = encode_categorical(df_pos, verbose=verbose)

    # 保存暴露特征（不参与分类）
    exposure_cols = [c for c in EXPOSURE_FEATURES if c in df_pos.columns]
    exposure_pos = df_pos[exposure_cols].copy()

    # 正样本标签
    df_pos['label'] = 1

    # 保存正样本（用于负样本继承）
    pos_for_sampling = df_pos.copy()

    if verbose:
        print("")
        n_pos = len(df_pos)
        print(f"正样本（灾害点）: {n_pos} 条")

    # ── 阶段4: 三源混合负样本生成 ──
    df_neg, neg_info = generate_hybrid_negatives(
        pos_for_sampling, ratio=neg_ratio,
        hybrid_ratio=hybrid_ratio,
        min_dist_km=min_dist_km, random_state=random_state,
        verbose=verbose
    )
    n_neg = len(df_neg)

    # 负样本同样经过特征工程
    df_neg, _ = build_susceptibility_features(df_neg, verbose=False)
    df_neg = encode_categorical(df_neg, verbose=False)

    # 负样本的暴露特征（全部为0）
    exposure_neg = pd.DataFrame(
        0, index=df_neg.index, columns=exposure_cols
    )

    # 检查负样本质量
    validate_negative_samples(df_pos, df_neg, verbose=verbose)

    # RF交叉验证质量检验（判断正负样本是否过度可分）
    if run_quality_check:
        rf_quality_check(df_pos, df_neg, verbose=verbose)

    # ── 阶段5: 合并正负样本 ──
    missing_in_neg = [c for c in sus_features if c not in df_neg.columns]
    for c in missing_in_neg:
        df_neg[c] = 0

    X_pos = df_pos[sus_features].copy()
    y_pos = df_pos['label'].values
    X_neg = df_neg[sus_features].copy()
    y_neg = df_neg['label'].values

    X = pd.concat([X_pos, X_neg], ignore_index=True)
    y = np.concatenate([y_pos, y_neg])

    # 合并暴露特征
    exposure_all = pd.concat([exposure_pos, exposure_neg], ignore_index=True)

    if verbose:
        print("")
        print("阶段5: 合并数据")
        print(f"  总样本: {len(X)} 条")
        print(f"  正样本(1): {y.sum()} 条")
        print(f"  负样本(0): {(y == 0).sum()} 条")
        print(f"  易发性特征: {len(sus_features)} 个")
        print(f"  暴露特征（不参与分类）: {len(exposure_cols)} 个")

    # ── 阶段6: 训练/测试划分 ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y
    )

    # 同步划分暴露特征
    exp_train, exp_test = train_test_split(
        exposure_all, test_size=test_size,
        random_state=random_state, stratify=y
    )

    if verbose:
        print("")
        print("阶段6: 数据划分")
        print(f"  训练集: {X_train.shape[0]} 条")
        print(f"  测试集: {X_test.shape[0]} 条")
        n1_train = y_train.sum()
        n0_train = len(y_train) - n1_train
        print(f"  训练集分布: 正={n1_train}, 负={n0_train}"
              f" (比例 1:{n0_train/max(n1_train,1):.2f})")

    # ── 阶段7: 标准化 ──
    if verbose:
        print("")
        print("阶段7: 特征标准化")

    # 只对连续特征标准化，跳过 One-hot 和地理坐标
    onehot_cols = [c for c in X.columns if c.startswith('lu_')]
    geo_cols = ['lon', 'lat']
    skip_cols = onehot_cols + geo_cols
    scale_cols = [c for c in X.columns if c not in skip_cols]

    scaler = StandardScaler()
    scaler.fit(X_train[scale_cols])

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    # 转为numpy数组
    X_train_arr = X_train_scaled.values.astype(np.float64)
    X_test_arr = X_test_scaled.values.astype(np.float64)

    # 安全检查：确保无NaN
    train_nan = np.isnan(X_train_arr).sum()
    test_nan = np.isnan(X_test_arr).sum()
    if train_nan > 0 or test_nan > 0:
        if verbose:
            print(f"  [!] 检测到 NaN: 训练集={train_nan}, 测试集={test_nan}，执行填充")
        X_train_arr = np.nan_to_num(X_train_arr, nan=0.0)
        X_test_arr = np.nan_to_num(X_test_arr, nan=0.0)

    if verbose:
        print(f"  标准化特征: {len(scale_cols)}, 跳过One-hot: {len(onehot_cols)}")
        print(f"  训练集形状: {X_train_arr.shape}")
        print(f"  测试集形状: {X_test_arr.shape}")
        print("")
        print("预处理完成!")
        print(f"  易发性特征: {len(sus_features)} 个")
        print(f"  暴露特征（风险分析用）: {len(exposure_cols)} 个")
        print("")

    # 返回附加信息
    extra = {
        'feature_names': sus_features,
        'scaler': scaler,
        'scale_cols': scale_cols,
        'onehot_cols': onehot_cols,
        'pos_df': df_pos,
        'neg_df': df_neg,
        'X_train_df': X_train_scaled,
        'X_test_df': X_test_scaled,
        'exposure_train': exp_train,
        'exposure_test': exp_test,
        'exposure_cols': exposure_cols,
    }

    return X_train_arr, X_test_arr, y_train, y_test, extra


# 快捷版本（兼容原接口，只返回4个值）
def preprocess_data_fast(neg_ratio=1.0, test_size=0.2, random_state=42,
                         min_dist_km=0.8, hybrid_ratio=0.3, verbose=True):
    """简化版，只返回 (X_train, X_test, y_train, y_test)"""
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        neg_ratio=neg_ratio, test_size=test_size,
        random_state=random_state, verbose=verbose,
        min_dist_km=min_dist_km, hybrid_ratio=hybrid_ratio
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, extra = preprocess_data(neg_ratio=0.5, verbose=True)
    print(f"易发性特征: {len(extra['feature_names'])} 个")
    print(f"暴露特征: {extra['exposure_cols']}")
    print(f"X_train: {X_train.shape}, y_train正样本: {y_train.sum()}")
