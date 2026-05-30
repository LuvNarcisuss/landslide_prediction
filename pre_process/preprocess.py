import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans


def preprocess_data(verbose=False, use_augmentation=False):
    """改进的数据预处理

    新增特征：
    1. ndwi - 归一化水体指数
    2. api - 前期降水指数
    3. lon - 经度
    4. 地理聚类特征 - KMeans在经纬度上的聚类
    5. 交互特征：降雨×高程、坡度×降雨、NDVI变化率等
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'dataset', 'aba_disaster_distribution.csv')
    data = pd.read_csv(data_path, encoding='utf-8-sig')

    # === 标签构建（不变） ===
    rain_triggered = data['triggering_factor'].str.contains('降雨', na=False)
    is_landslide = (data['hazard_type'] == '滑坡')
    label = (~(rain_triggered & is_landslide)).astype(int).values

    n_samples = len(data)

    # === 特征构建 ===

    # 1. 基础连续特征
    elevation = data['elevation'].values.astype(float)
    aspect = data['aspect'].values.astype(float)
    ndvi = data['ndvi'].values.astype(float)
    ndwi = data['ndwi'].values.astype(float)
    landuse = data['landuse'].values.astype(float)
    distance_to_river = data['distance_to_river'].values.astype(float)
    tpi = data['tpi'].values.astype(float)
    rain_3d = data['rain_3d'].values.astype(float)
    rain_7d = data['rain_7d'].values.astype(float)
    rain_30d = data['rain_30d'].values.astype(float)
    api = data['api'].values.astype(float)
    lat = data['lat'].values.astype(float)
    lon = data['lon'].values.astype(float)

    # 2. 诱发因素分解
    has_rain = data['triggering_factor'].str.contains('降雨', na=False).astype(int).values
    has_earthquake = data['triggering_factor'].str.contains('地震', na=False).astype(int).values
    has_human = data['triggering_factor'].str.contains('人为', na=False).astype(int).values

    # 3. 降雨比率
    eps = 1e-8
    rain_3d_ratio = np.where(rain_30d > eps, rain_3d / np.maximum(rain_30d, eps), 0.0)
    rain_7d_ratio = np.where(rain_30d > eps, rain_7d / np.maximum(rain_30d, eps), 0.0)
    # 短期降雨强度（3天降雨占7天的比例）
    rain_intensity = np.where(rain_7d > eps, rain_3d / np.maximum(rain_7d, eps), 0.0)

    # 4. 地形特征
    valley_risk = np.maximum(0, -tpi) / (distance_to_river + 1)
    # 坡度方向与降雨交互（南向坡更易受降雨影响）
    aspect_rain = np.cos(np.radians(aspect)) * rain_3d
    # 高程降雨交互（高海拔+强降雨）
    elev_rain = elevation * np.log1p(rain_3d)

    # 5. NDVI相关
    ndvi_rain = ndvi * np.log1p(rain_3d) # 4. 地形特征
    valley_risk = np.maximum(0, -tpi) / (distance_to_river + 1)
    # 坡度方向与降雨交互（南向坡更易受降雨影响）
    aspect_rain = np.cos(np.radians(aspect)) * rain_3d
    # 高程降雨交互（高海拔+强降雨）
    elev_rain = elevation * np.log1p(np.maximum(rain_3d, 0))

    # 5. NDVI相关
    ndvi_rain = ndvi * np.log1p(np.maximum(rain_3d, 0))
    # NDWI-NDVI 差异（植被含水量指标）
    ndwi_ndvi_diff = ndwi - ndvi

    # 6. 地形位置综合指数
    tpi_elev = tpi * elevation / 1000.0

    # 7. 距河流远近的风险（对数变换）
    log_dist_river = np.log1p(np.maximum(distance_to_river, 0))

    # 8. 缺失指示器
    ndvi_missing = np.isnan(ndvi).astype(int)
    ndwi_missing = np.isnan(ndwi).astype(int)
    rain_missing = np.isnan(rain_30d).astype(int)
    api_missing = np.isnan(api).astype(int)

    # 9. 地理聚类特征（经纬度区域划分）
    coords = np.column_stack([lat, lon])
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    geo_cluster = kmeans.fit_predict(coords)
    # 到各聚类中心的距离
    geo_dist = kmeans.transform(coords)

    # 组装DataFrame
    X = pd.DataFrame({
        'elevation': elevation, 'aspect': aspect,
        'ndvi': ndvi, 'ndwi': ndwi,
        'landuse': landuse,
        'distance_to_river': distance_to_river,
        'log_dist_river': log_dist_river,
        'tpi': tpi,
        'rain_3d': rain_3d, 'rain_7d': rain_7d, 'rain_30d': rain_30d,
        'api': api,
        'lat': lat, 'lon': lon,
        'has_rain': has_rain, 'has_earthquake': has_earthquake, 'has_human': has_human,
        'rain_3d_ratio': rain_3d_ratio,
        'rain_7d_ratio': rain_7d_ratio,
        'rain_intensity': rain_intensity,
        'valley_risk': valley_risk,
        'aspect_rain': aspect_rain,
        'elev_rain': elev_rain,
        'ndvi_rain': ndvi_rain,
        'ndwi_ndvi_diff': ndwi_ndvi_diff,
        'tpi_elev': tpi_elev,
        'ndvi_missing': ndvi_missing,
        'ndwi_missing': ndwi_missing,
        'rain_missing': rain_missing,
        'api_missing': api_missing,
        'geo_cluster': geo_cluster,
    })

    # 加入地理距离特征
    for k in range(10):
        X[f'geo_dist_{k}'] = geo_dist[:, k]

    features = list(X.columns)

    # === 缺失值填充（先拆分再fit/transform） ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, label, test_size=0.2, random_state=42, stratify=label
    )

    # 综合特征插补（利用多变量相关性）
    impute_cols = ['rain_3d', 'rain_7d', 'rain_30d', 'api',
                   'rain_3d_ratio', 'rain_7d_ratio', 'rain_intensity',
                   'ndvi', 'ndwi', 'ndwi_ndvi_diff',
                   'elevation', 'tpi', 'elev_rain',
                   'ndvi_rain', 'aspect_rain', 'tpi_elev',
                   'log_dist_river', 'valley_risk',
                   'lat', 'lon']

    # ndwi缺失率高(43%)，先中位数填充再IterativeImputer
    for col in ['ndvi', 'ndwi']:
        med_val = data[col].median()
        X_train[col].fillna(med_val, inplace=True)
        X_test[col].fillna(med_val, inplace=True)

    imputer = IterativeImputer(max_iter=50, random_state=42, sample_posterior=False)
    imputer.fit(X_train[impute_cols])
    X_train_imp = imputer.transform(X_train[impute_cols])
    X_test_imp = imputer.transform(X_test[impute_cols])

    for i, col in enumerate(impute_cols):
        X_train.loc[:, col] = X_train_imp[:, i]
        X_test.loc[:, col] = X_test_imp[:, i]

    # 确保没有任何NaN残留
    assert not np.any(np.isnan(X_train.values)), "训练集仍有NaN!"
    assert not np.any(np.isnan(X_test.values)), "测试集仍有NaN!"

    # 重新计算插补后的衍生特征
    eps = 1e-8
    for df_set in [X_train, X_test]:
        # 重新计算降雨比率（防止插补后溢出）
        df_set['rain_3d_ratio'] = np.where(df_set['rain_30d'] > eps,
                                            df_set['rain_3d'] / df_set['rain_30d'], 0.0)
        df_set['rain_7d_ratio'] = np.where(df_set['rain_30d'] > eps,
                                            df_set['rain_7d'] / df_set['rain_30d'], 0.0)
        df_set['rain_intensity'] = np.where(df_set['rain_7d'] > eps,
                                              df_set['rain_3d'] / df_set['rain_7d'], 0.0)
        # 重新计算交互特征
        df_set['elev_rain'] = df_set['elevation'] * np.log1p(np.maximum(df_set['rain_3d'], 0))
        df_set['ndvi_rain'] = df_set['ndvi'] * np.log1p(np.maximum(df_set['rain_3d'], 0))
        df_set['aspect_rain'] = np.cos(np.radians(df_set['aspect'])) * df_set['rain_3d']
        df_set['ndwi_ndvi_diff'] = df_set['ndwi'] - df_set['ndvi']
        df_set['tpi_elev'] = df_set['tpi'] * df_set['elevation'] / 1000.0

    # 标准化
    # 确保无NaN
    for col in X_train.columns:
        if X_train[col].isnull().any():
            X_train[col].fillna(X_train[col].median() if X_train[col].dtype.kind in 'fc' else 0, inplace=True)
        if X_test[col].isnull().any():
            X_test[col].fillna(X_train[col].median() if X_train[col].dtype.kind in 'fc' else 0, inplace=True)

    scaler = StandardScaler()
    binary_cols = ['has_rain', 'has_earthquake', 'has_human',
                   'ndvi_missing', 'ndwi_missing', 'rain_missing', 'api_missing']
    scale_cols = [c for c in features if c not in binary_cols]
    id_cols = binary_cols  # 二值特征不做标准化

    scaler.fit(X_train[scale_cols])
    X_train_scaled = np.concatenate([
        scaler.transform(X_train[scale_cols]),
        X_train[id_cols].values
    ], axis=1)
    X_test_scaled = np.concatenate([
        scaler.transform(X_test[scale_cols]),
        X_test[id_cols].values
    ], axis=1)

    if verbose:
        n_0 = (label == 0).sum()
        n_1 = (label == 1).sum()
        print("数据预处理完成")
        print(f"特征数: {len(features)}")
        print(f"总样本: {len(label)}")
        print(f"  标签 0 (降雨诱发滑坡): {n_0} ({n_0/len(label):.2%})")
        print(f"  标签 1 (其他/非滑坡):  {n_1} ({n_1/len(label):.2%})")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(verbose=True)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
