"""
负样本生成模块
为滑坡易发性评价生成非灾害点（负样本/标签0）。

方法：三源混合负样本策略
1. 真实非降雨灾害（来源A）：同一数据集中非降雨诱发的坡体灾害
2. 非目标灾害类型（来源B）：崩塌 + 地面塌陷
3. 空间随机采样（来源C）：缓冲区 > 800m + 特征继承 + 噪声 + 高程校正
4. 三源按比例混合，RF质量检验
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import QuantileTransformer


# 步骤1 - 最近邻特征继承
def _inherit_features_from_neighbors(pos_df, cand_lat, cand_lon, n_neighbors=3):
    """
    从最近的 n 个灾害点继承特征值，加小噪声
    避免生成的特征与真实分布脱节
    """
    coords = np.radians(pos_df[['lat', 'lon']].values)
    tree = BallTree(coords, metric='haversine')

    dist, indices = tree.query(
        np.radians([[cand_lat, cand_lon]]), k=n_neighbors
    )

    nearest = pos_df.iloc[indices[0]]

    weights = 1.0 / (dist[0] + 1e-6)
    weights = weights / weights.sum()

    candidate = {}

    numeric_noise = {
        'elevation': 150.0,
        'aspect': 30.0,
        'tpi': 15.0,
        'distance_to_river': 100.0,
        'rain_3d': 8.0,
        'rain_7d': 12.0,
        'rain_30d': 25.0,
        'api': 8.0,
        'ndvi': 0.10,
        'ndwi': 0.10,
    }

    for col, noise_scale in numeric_noise.items():
        if col in nearest.columns:
            weighted = (nearest[col].values * weights).sum()
            candidate[col] = weighted + np.random.normal(0, noise_scale)

    if 'landuse' in nearest.columns:
        candidate['landuse'] = np.random.choice(nearest['landuse'].values)
    if 'landuse_type' in nearest.columns:
        candidate['landuse_type'] = np.random.choice(nearest['landuse_type'].values)

    candidate['threatened_people'] = 0.0
    candidate['threatened_households'] = 0.0
    candidate['threatened_property'] = 0.0
    candidate['threat_target'] = '无'

    candidate['lon'] = cand_lon
    candidate['lat'] = cand_lat
    candidate['label'] = 0

    return candidate


# 步骤2 - 空间约束负样本生成
def generate_negative_samples(pos_df, ratio=1.0, min_dist_km=0.8,
                               random_state=42, correct_elevation=True,
                               verbose=True):
    """
    生成负样本（非灾害点）

    参数:
    - pos_df: 正样本DataFrame（灾害点）
    - ratio: 负样本/正样本 比例 (默认1:1)
    - min_dist_km: 距灾害点最小距离(km) (默认0.8km)
    - random_state: 随机种子
    - correct_elevation: 是否校正高程分布 (默认True)

    返回:
    - neg_df: 负样本DataFrame，含label=0
    """
    np.random.seed(random_state)
    n_pos = len(pos_df)
    n_neg = int(n_pos * ratio)

    if verbose:
        print("")
        print("负样本（非灾害点）生成:")
        print(f"  正样本: {n_pos} 条")
        print(f"  生成比例: 1:{ratio} → {n_neg} 条")
        print(f"  距灾害点最小距离: {min_dist_km} km")

    lat_min, lat_max = pos_df['lat'].min(), pos_df['lat'].max()
    lon_min, lon_max = pos_df['lon'].min(), pos_df['lon'].max()

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_min -= lat_range * 0.05
    lat_max += lat_range * 0.05
    lon_min -= lon_range * 0.05
    lon_max += lon_range * 0.05

    if verbose:
        print(f"  采样范围: lat [{lat_min:.3f}, {lat_max:.3f}], lon [{lon_min:.3f}, {lon_max:.3f}]")

    coords_rad = np.radians(pos_df[['lat', 'lon']].values)
    tree = BallTree(coords_rad, metric='haversine')

    min_dist_rad = min_dist_km / 6371.0

    neg_samples = []
    attempts = 0
    max_attempts = n_neg * 30

    while len(neg_samples) < n_neg and attempts < max_attempts:
        attempts += 1

        cand_lat = np.random.uniform(lat_min, lat_max)
        cand_lon = np.random.uniform(lon_min, lon_max)

        dist, _ = tree.query(
            np.radians([[cand_lat, cand_lon]]), k=1
        )

        if dist[0][0] >= min_dist_rad:
            cand = _inherit_features_from_neighbors(pos_df, cand_lat, cand_lon)
            neg_samples.append(cand)

        if verbose and len(neg_samples) > 0 and len(neg_samples) % 500 == 0:
            print(f"  已生成 {len(neg_samples)}/{n_neg} 个负样本 (尝试{attempts}次)")

    if verbose:
        success_rate = len(neg_samples) / attempts * 100
        print(f"  生成完成: {len(neg_samples)}/{n_neg} (成功率 {success_rate:.1f}%)")

    if len(neg_samples) < n_neg:
        if verbose:
            print(f"  [!] 未能达到目标数量 {n_neg}，使用当前 {len(neg_samples)} 个")

    neg_df = pd.DataFrame(neg_samples)

    if correct_elevation and 'elevation' in pos_df.columns and len(neg_df) > 0:
        if verbose:
            pre_mean = neg_df['elevation'].mean()
        neg_df = correct_elevation_distribution(pos_df, neg_df, random_state=random_state)
        if verbose:
            post_mean = neg_df['elevation'].mean()
            pos_mean = pos_df['elevation'].mean()
            print(f"  高程校正: 负样本均值 {pre_mean:.0f}m → {post_mean:.0f}m"
                  f" (正样本均值 {pos_mean:.0f}m)")

    return neg_df


# 步骤3 - 高程分布校正
def correct_elevation_distribution(pos_df, neg_df, random_state=42):
    pos_elev = pos_df['elevation'].values.reshape(-1, 1)
    neg_elev = neg_df['elevation'].values.reshape(-1, 1)

    qt_neg = QuantileTransformer(n_quantiles=min(1000, len(neg_elev)),
                                  output_distribution='uniform',
                                  random_state=random_state)
    neg_ranks = qt_neg.fit_transform(neg_elev).flatten()

    pos_sorted = np.sort(pos_elev.flatten())
    n_pos = len(pos_sorted)
    neg_indices = (neg_ranks * (n_pos - 1)).astype(int)
    neg_indices = np.clip(neg_indices, 0, n_pos - 1)
    neg_df['elevation'] = pos_sorted[neg_indices]

    pos_std = pos_elev.std()
    noise = np.random.RandomState(random_state).normal(0, pos_std * 0.02, size=len(neg_df))
    neg_df['elevation'] = np.clip(neg_df['elevation'] + noise, pos_elev.min(), pos_elev.max())

    return neg_df


# 步骤4 - 负样本质量验证
def validate_negative_samples(pos_df, neg_df, verbose=True):
    if verbose:
        print("")
        print("负样本质量检查:")

    numeric_cols = ['elevation', 'aspect', 'tpi', 'distance_to_river',
                    'rain_3d', 'rain_7d', 'rain_30d', 'api', 'ndvi', 'ndwi']

    warnings = 0
    for col in numeric_cols:
        if col not in pos_df.columns or col not in neg_df.columns:
            continue
        p_mean = pos_df[col].mean()
        n_mean = neg_df[col].mean()
        p_std = pos_df[col].std()
        n_std = neg_df[col].std()
        if p_std > 0 and abs(p_mean - n_mean) > p_std:
            warnings += 1
            if verbose:
                print(f"  [!] {col}: 正样本均值={p_mean:.2f}, 负样本均值={n_mean:.2f} (漂移>{p_std:.2f})")

    for col in ['threatened_people', 'threatened_households', 'threatened_property']:
        if col in neg_df.columns:
            non_zero = (neg_df[col] > 0).sum()
            if non_zero > 0:
                warnings += 1
                if verbose:
                    print(f"  [!] {col}: {non_zero} 个负样本有非零值")

    if warnings == 0:
        if verbose:
            print("  [OK] 所有特征分布合理")
    else:
        if verbose:
            print(f"  [!] {warnings} 个特征存在分布偏差")

    if verbose:
        print("")
        print(f"  正样本空间范围: lon [{pos_df['lon'].min():.3f}, {pos_df['lon'].max():.3f}]"
              f", lat [{pos_df['lat'].min():.3f}, {pos_df['lat'].max():.3f}]")
        print(f"  负样本空间范围: lon [{neg_df['lon'].min():.3f}, {neg_df['lon'].max():.3f}]"
              f", lat [{neg_df['lat'].min():.3f}, {neg_df['lat'].max():.3f}]")

    return warnings


# 步骤5 - RF质量检验
def rf_quality_check(pos_df, neg_df, feature_cols=None, cv=5, verbose=True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    pos = pos_df.copy()
    neg = neg_df.copy()
    pos['label'] = 1
    neg['label'] = 0
    combined = pd.concat([pos, neg], ignore_index=True)

    if feature_cols is None:
        exclude = ['label', 'lon', 'lat', 'threat_target',
                   'triggering_factor', 'hazard_type', 'danger_grade',
                   'landuse_type',
                   'danger_code', 'hazard_code',
                   'has_rain', 'has_earthquake', 'has_human',
                   'threatened_people', 'threatened_households', 'threatened_property',
                   'exposure_index',
                   'ndvi_missing', 'ndwi_missing', 'rain_missing']
        feature_cols = [c for c in combined.columns
                        if c not in exclude
                        and combined[c].dtype in ['float64', 'int64', 'float32', 'int32']
                        and combined[c].nunique() > 1]

    X = combined[feature_cols].fillna(0).values
    y = combined['label'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
        rf.fit(X[train_idx], y[train_idx])
        y_prob = rf.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], y_prob))

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    if mean_auc > 0.97:
        status = 'FAIL'
        diagnosis = (f"正负样本 AUC={mean_auc:.4f} 过高(>{0.97})，"
                     f"负样本与正样本差异过大，模型过度可分。\n"
                     f"  建议: 增大缓冲区 或 增大特征噪声 或 缩小采样范围")
    elif mean_auc >= 0.85:
        status = 'PASS'
        diagnosis = (f"正负样本 AUC={mean_auc:.4f} 在合理范围(0.85-0.97)，负样本质量合格")
    elif mean_auc >= 0.80:
        status = 'WARN'
        diagnosis = (f"正负样本 AUC={mean_auc:.4f} 偏低，负样本接近正样本分布。\n"
                     f"  建议: 减小缓冲区 或 减小特征噪声")
    else:
        status = 'FAIL'
        diagnosis = (f"正负样本 AUC={mean_auc:.4f} 过低(<{0.80})，"
                     f"负样本与正样本几乎不可分，可能包含潜在灾害点。\n"
                     f"  建议: 减小缓冲区 或 重新选择非灾害区")

    if verbose:
        print("")
        print("RF质量检验:")
        print(f"  特征数: {len(feature_cols)}")
        print(f"  交叉验证: {cv}折")
        print(f"  各折AUC: {[f'{a:.4f}' for a in aucs]}")
        print(f"  平均AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  状态: {status}")
        print(f"  诊断: {diagnosis}")
        print("")

    return {'mean_auc': mean_auc, 'std_auc': std_auc, 'aucs': aucs, 'status': status, 'diagnosis': diagnosis, 'feature_cols': feature_cols}


def suggest_adjustment(qc_result, current_min_dist=0.8):
    mean_auc = qc_result['mean_auc']
    print("参数优化建议:")
    print("")
    if mean_auc > 0.97:
        print(f"  当前缓冲区: {current_min_dist}km → 建议增大至 {current_min_dist + 0.2}km")
    elif mean_auc >= 0.85:
        print(f"  当前缓冲区: {current_min_dist}km ✅ 合适，无需调整")
    elif mean_auc >= 0.80:
        print(f"  当前缓冲区: {current_min_dist}km → 建议减小至 {max(0.3, current_min_dist - 0.2)}km")
    else:
        print(f"  当前缓冲区: {current_min_dist}km → 建议减小至 0.3km，重新定义非灾害区")


# 步骤6 - 提取天然负样本
def extract_real_negatives(raw_path=None, verbose=True):
    """
    从原始数据中提取天然负样本。

    来源A: 非降雨诱发的滑坡/不稳定斜坡/泥石流 (与正样本同类型但触发机制不同)
    来源B: 崩塌 + 地面塌陷 (非目标灾害类型)
    """
    if raw_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        raw_path = os.path.join(project_root, 'dataset', 'aba_disaster_distribution.csv')

    if verbose:
        print("")
        print("提取天然负样本:")

    raw = pd.read_csv(raw_path, encoding='utf-8-sig')

    # 来源A: 保留类型中非降雨诱发的
    slope_types = ['滑坡', '不稳定斜坡', '泥石流']
    mask_a = raw['hazard_type'].isin(slope_types) & (~raw['triggering_factor'].str.contains('降雨', na=True))
    source_a = raw[mask_a].copy()

    # 来源B: 非目标类型
    other_types = ['崩塌', '地面塌陷']
    mask_b = raw['hazard_type'].isin(other_types)
    source_b = raw[mask_b].copy()

    real_neg = pd.concat([source_a, source_b], ignore_index=True)

    if verbose:
        print(f"  来源A（非降雨坡体灾害）: {len(source_a)} 条")
        print(f"  来源B（崩塌+地面塌陷）: {len(source_b)} 条")
        print(f"  合计: {len(real_neg)} 条")

    if len(real_neg) == 0:
        return pd.DataFrame()

    # 应用与正样本相同的清洗+插补
    from pre_process.data_cleaning import drop_redundant_columns, drop_sparse_null_rows, filter_outliers, drop_duplicates, add_missing_flags
    real_neg = drop_redundant_columns(real_neg, verbose=False)
    real_neg = drop_sparse_null_rows(real_neg, verbose=False)
    real_neg = filter_outliers(real_neg, verbose=False)
    real_neg = drop_duplicates(real_neg, verbose=False)
    real_neg = add_missing_flags(real_neg)

    from pre_process.imputation import impute_all
    real_neg = impute_all(real_neg, verbose=False)

    real_neg['label'] = 0

    if verbose:
        print(f"  清洗+插补后: {len(real_neg)} 条")

    return real_neg


# 步骤7 - 三源混合负样本生成（核心入口）
def generate_hybrid_negatives(pos_df, ratio=1.0, hybrid_ratio=0.3,
                               min_dist_km=0.8, correct_elevation=True,
                               random_state=42, verbose=True):
    """
    三源混合负样本生成

    参数:
    - pos_df: 正样本DataFrame
    - ratio: 负/正比例
    - hybrid_ratio: 真实数据占比 (默认0.3，即30%真实+70%空间采样)
    """
    np.random.seed(random_state)
    n_pos = len(pos_df)
    n_total = int(n_pos * ratio)
    n_real = int(n_total * hybrid_ratio)
    n_syn = n_total - n_real

    if verbose:
        print("")
        print("三源混合负样本生成:")
        print(f"  正样本: {n_pos} 条 | 目标负样本: {n_total} 条")
        print(f"    来源A+B（真实灾害）: {n_real} 条 ({hybrid_ratio*100:.0f}%)")
        print(f"    来源C（空间采样）:   {n_syn} 条 ({(1-hybrid_ratio)*100:.0f}%)")

    parts = []

    # 来源A+B
    if n_real > 0:
        real_neg = extract_real_negatives(verbose=verbose)
        avail = len(real_neg)

        if avail >= n_real:
            real_neg = real_neg.sample(n=n_real, random_state=random_state)
            if verbose:
                print(f"  → 从{avail}条中抽取{n_real}条")
        elif avail > 0:
            n_real = avail
            n_syn = n_total - n_real
            if verbose:
                print(f"  → 天然负样本不足({avail}<{n_real})，使用全部{avail}条"
                      f"，剩余{n_syn}条由空间采样补齐")
        else:
            n_real = 0
            n_syn = n_total

        if n_real > 0:
            parts.append(real_neg)

    # 来源C
    if n_syn > 0:
        if verbose and n_real > 0:
            print("")
        syn_neg = generate_negative_samples(
            pos_df, ratio=n_syn / n_pos,
            min_dist_km=min_dist_km,
            correct_elevation=correct_elevation,
            random_state=random_state, verbose=verbose
        )
        if len(syn_neg) > 0:
            parts.append(syn_neg)

    if len(parts) == 0:
        return pd.DataFrame(), {'n_real': 0, 'n_synthetic': 0, 'real_ratio': 0.0}

    neg_df = pd.concat(parts, ignore_index=True)

    if verbose:
        print("")
        print(f"  => 混合完成: 共{len(neg_df)}条 (真实{parts[0].shape[0] if len(parts)>0 and n_real>0 else 0}"
              f" + 空间采样{len(neg_df)-n_real})")

    return neg_df, {'n_real': n_real, 'n_synthetic': len(neg_df) - n_real,
                     'real_ratio': n_real / max(len(neg_df), 1)}


if __name__ == "__main__":
    from data_cleaning import clean_data
    from imputation import impute_all
    from pipeline import build_susceptibility_features

    pos_df = clean_data(verbose=False)
    pos_df = impute_all(pos_df, verbose=False)
    pos_df, _ = build_susceptibility_features(pos_df, verbose=False)
    print(f"正样本: {len(pos_df)} 条")

    neg_df, info = generate_hybrid_negatives(pos_df, ratio=1.0, hybrid_ratio=0.3, verbose=True)
    print(f"负样本: {len(neg_df)} 条 (真实{info['n_real']}+空间采样{info['n_synthetic']})")

    validate_negative_samples(pos_df, neg_df, verbose=True)
    qc = rf_quality_check(pos_df, neg_df, verbose=True)
    suggest_adjustment(qc)
