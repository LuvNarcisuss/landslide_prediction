"""
数据清洗模块
对 aba_disaster_distribution.csv 进行：
1. 灾害类型筛选（滑坡 + 不稳定斜坡 + 泥石流）
2. 删除冗余字段
3. 空值处理（行删除 + 标记）
4. 异常值过滤
5. 重复值删除
"""
import os
import pandas as pd
import numpy as np


# 步骤1 - 加载原始数据
def load_raw_data(verbose=True):
    """加载原始CSV数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'dataset', 'aba_disaster_distribution.csv')

    df = pd.read_csv(data_path, encoding='utf-8-sig')
    if verbose:
        print(f"原始数据: {df.shape[0]} 条 × {df.shape[1]} 列")
    return df


# 步骤2 - 灾害类型筛选
def filter_hazard_types(df, verbose=True):
    """只保留 滑坡 + 不稳定斜坡 + 泥石流"""
    keep_types = ['滑坡', '不稳定斜坡', '泥石流']
    before = len(df)
    df = df[df['hazard_type'].isin(keep_types)].copy()
    after = len(df)

    if verbose:
        removed = before - after
        print(f"灾害类型筛选: {before} → {after} (移除 {removed} 条非目标灾害)")
        for t in keep_types:
            print(f"  保留: {t} = {(df['hazard_type'] == t).sum()} 条")

    return df


# 步骤3 - 删除冗余/无用字段
def drop_redundant_columns(df, verbose=True):
    """
    删除对建模无用的字段：
    - id: 纯序号
    - discovery_date: 元数据，含异常值
    - location: 文本描述，几乎全部唯一
    - data_source: 仅标注Landsat版本号
    - development_trend: 84.7%缺失
    - industry / industry_transferred: 81%缺失
    - disaster_type: 与hazard_type冗余
    - disaster_scale / scale_grade: 文本描述，与danger_grade相关
    """
    drop_cols = [
        'id', 'discovery_date', 'location', 'data_source',
        'development_trend', 'industry', 'industry_transferred',
        'disaster_type', 'disaster_scale', 'scale_grade'
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)

    if verbose:
        print(f"删除 {len(existing_drops)} 个冗余字段: {existing_drops}")
        print(f"剩余字段数: {df.shape[1]}")

    return df


# 步骤4 - 空值行删除（少量缺失字段）
def drop_sparse_null_rows(df, verbose=True):
    """
    删除关键字段的空值行（缺失率 < 5%）
    - triggering_factor: 66条缺失(0.9%)
    - threat_target: 75条缺失(1.0%)
    """
    before = len(df)

    # triggering_factor
    n_tf = df['triggering_factor'].isnull().sum()
    if n_tf > 0:
        df = df.dropna(subset=['triggering_factor'])
        if verbose:
            print(f"删除 triggering_factor 空值: {n_tf} 条")

    # threat_target
    n_tt = df['threat_target'].isnull().sum()
    if n_tt > 0:
        df = df.dropna(subset=['threat_target'])
        if verbose:
            print(f"删除 threat_target 空值: {n_tt} 条")

    if verbose:
        print(f"空值行删除后: {before} → {len(df)} 条")

    return df


# 步骤5 - 异常值过滤
def filter_outliers(df, verbose=True):
    """
    过滤各特征的异常值，使用四分位距(IQR)法
    """
    before = len(df)
    mask = pd.Series(True, index=df.index)

    # elevation: 阿坝州合理范围 500~5500m
    mask &= df['elevation'].between(500, 5500)

    # aspect: 0~360度
    mask &= df['aspect'].between(0, 360)

    # rain: 非负
    for col in ['rain_3d', 'rain_7d', 'rain_30d', 'api']:
        mask &= df[col].fillna(0) >= 0

    # tpi: IQR法
    q1, q3 = df['tpi'].quantile([0.01, 0.99])
    mask &= df['tpi'].between(q1, q3)

    # distance_to_river: IQR法
    q1, q3 = df['distance_to_river'].quantile([0.01, 0.99])
    mask &= df['distance_to_river'].between(q1, q3)

    df = df[mask].copy()
    removed = before - len(df)

    if verbose:
        print(f"异常值过滤: {before} → {len(df)} (移除 {removed} 条)")

    return df


# 步骤6 - 删除重复值
def drop_duplicates(df, verbose=True):
    """删除经纬度完全相同的重复记录"""
    before = len(df)
    df = df.drop_duplicates(subset=['lon', 'lat'], keep='first')
    removed = before - len(df)

    if verbose:
        print(f"重复值删除: {before} → {len(df)} (移除 {removed} 条)")

    return df


# 步骤7 - 标记缺失情况（用于后续建模）
def add_missing_flags(df):
    """为高缺失率字段添加缺失指示器"""
    df['ndvi_missing'] = df['ndvi'].isnull().astype(int)
    df['ndwi_missing'] = df['ndwi'].isnull().astype(int)
    df['rain_missing'] = df['rain_3d'].isnull().astype(int)
    return df


# 主清洗流程
def clean_data(verbose=True):
    """执行完整数据清洗"""
    if verbose:
        print("")
        print("步骤1 - 加载原始数据")

    df = load_raw_data(verbose)

    if verbose:
        print("")
        print("步骤2 - 灾害类型筛选（滑坡 + 不稳定斜坡 + 泥石流）")

    df = filter_hazard_types(df, verbose)

    if verbose:
        print("")
        print("步骤3 - 删除冗余字段")

    df = drop_redundant_columns(df, verbose)

    if verbose:
        print("")
        print("步骤4 - 空值行删除")

    df = drop_sparse_null_rows(df, verbose)

    if verbose:
        print("")
        print("步骤5 - 异常值过滤")

    df = filter_outliers(df, verbose)

    if verbose:
        print("")
        print("步骤6 - 删除重复值")

    df = drop_duplicates(df, verbose)

    if verbose:
        print("")
        print("步骤7 - 标记缺失情况")

    df = add_missing_flags(df)

    if verbose:
        print("")
        print(f"清洗完成: 最终 {df.shape[0]} 条 × {df.shape[1]} 列")
        print("")

    return df


if __name__ == "__main__":
    df = clean_data(verbose=True)
    print(f"清洗后数据形状: {df.shape}")
    print(f"剩余字段: {list(df.columns)}")
