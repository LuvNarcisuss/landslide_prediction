import os
import numpy as np
from catboost import CatBoostClassifier
from pre_process import preprocess_data


# 当前 20 个特征的名称与顺序（与 preprocess.py 中 features 列表一致）
FEATURE_NAMES = [
    'elevation', 'aspect', 'ndvi', 'distance_to_river', 'tpi',
    'rain_3d', 'rain_7d', 'rain_30d', 'lat', 'rain_3d_ratio',
    'dist_epicenter', 'rain_accum_ratio', 'elevation_tpi',
    'rain_earthquake', 'rain_human', 'river_distance_ratio',
    'ndvi_elevation', 'has_rain', 'has_earthquake', 'has_human',
]
def analyze_feature_importance():
    """训练CatBoost模型，输出特征重要性排名"""
    X_train, X_test, y_train, y_test = preprocess_data()

    model = CatBoostClassifier(random_state=42, verbose=False,
                               allow_writing_files=False, iterations=500)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("特征重要性排序:")
    print(f"{'特征':25s} {'重要性':>8s}  {'累计':>8s}")
    print("-" * 45)
    cumsum = 0
    for idx in sorted_idx:
        cumsum += importances[idx] / 100
        pct = importances[idx]
        marker = " <<<" if cumsum <= 0.95 else " >>>"
        print(f"{FEATURE_NAMES[idx]:25s} {pct:7.2f}%  {cumsum*100:6.2f}%{marker}")

    print(f"\n总特征数: {len(FEATURE_NAMES)}")
    return model.feature_importances_


if __name__ == "__main__":
    analyze_feature_importance()


