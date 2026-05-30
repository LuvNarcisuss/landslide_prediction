# 滑坡预测 (Landslide Prediction)

基于机器学习的滑坡灾害预测二分类项目。利用真实灾害隐患点普查数据，通过多种分类模型进行训练与对比评估。

## 目录

- [项目概述](#项目概述)
- [数据说明](#数据说明)
- [特征工程](#特征工程)
- [数据增强](#数据增强)
- [模型列表](#模型列表)
- [改进集成算法](#改进集成算法)
- [实验结果](#实验结果)
- [快速开始](#快速开始)
- [训练日志](#训练日志)
- [项目结构](#项目结构)

---

## 项目概述

滑坡是常见的地质灾害，其发生与地形地貌、水文条件、植被覆盖及降雨量等因素密切相关。本项目基于四川省汶川县耿达镇的实际灾害隐患点数据，构建并对比多种机器学习分类模型，实现滑坡灾害的二元预测。

**任务类型**：二分类  
**目标变量**：`label`（0 = 降雨诱发滑坡，1 = 其他/非滑坡）  
**评估指标**：准确率、精确率、召回率、F1 值、特异性、AUC-ROC、AUC-PR  
**排序指标**：AUC-PR（更关注概率排序质量）

---

## 数据说明

### 数据集

| 文件 | 样本数 | 用途 |
|------|:------:|:-----|
| `dataset/aba_disaster_distribution.csv` | 7,315 条 | **真实数据**（汶川县耿达镇灾害隐患点普查） |
| `dataset/simulated_landslide_large.csv` | 20,000 条 | 模拟数据 |
| `dataset/simulated_landslide_small.csv` | 2,000 条 | 模拟数据（小规模实验） |

### 标签构建规则

```
triggering_factor 含"降雨" + hazard_type 为"滑坡" → 标签 0（降雨诱发滑坡）
其余全部 → 标签 1（其他/非滑坡）
```

样本分布：标签 0 共 2,144 条 (29.31%)，标签 1 共 5,171 条 (70.69%)

---

## 特征工程

数据预处理流程（`data_process/preprocess.py`），输出 **43 维特征**：

### 基础特征（17 维）

| 特征 | 说明 |
|:-----|:-----|
| `elevation` | 海拔 (m) |
| `aspect` | 坡向 (0-360°) |
| `ndvi` | 归一化植被指数 |
| `ndwi` | 归一化水体指数 |
| `landuse` | 土地利用类型编码 |
| `distance_to_river` | 距河流距离 (m) |
| `tpi` | 地形位置指数 |
| `rain_3d` | 前 3 天降雨量 |
| `rain_7d` | 前 7 天降雨量 |
| `rain_30d` | 前 30 天降雨量 |
| `api` | 前期降水指数 |
| `lat` | 纬度 |
| `lon` | 经度 |
| `has_rain` | 是否含降雨诱发因素（二值） |
| `has_earthquake` | 是否含地震诱发因素（二值） |
| `has_human` | 是否含人为诱发因素（二值） |
| `ndvi_missing` / `ndwi_missing` / `rain_missing` / `api_missing` | 缺失指示器（4 个） |

### 交互特征（6 维）

| 特征 | 公式 | 说明 |
|:-----|:----|:-----|
| `rain_3d_ratio` | rain_3d / rain_30d | 短期降雨强度占比 |
| `rain_7d_ratio` | rain_7d / rain_30d | 中期降雨强度占比 |
| `rain_intensity` | rain_3d / rain_7d | 降雨集中度 |
| `valley_risk` | max(0,-tpi) / (dist_river+1) | 低洼近河风险指数 |
| `aspect_rain` | cos(aspect) × rain_3d | 坡向降雨交互 |
| `elev_rain` | elevation × log1p(rain_3d) | 高程降雨交互 |
| `ndvi_rain` | ndvi × log1p(rain_3d) | 植被降雨交互 |
| `ndwi_ndvi_diff` | ndwi - ndvi | 植被含水量差异 |
| `tpi_elev` | tpi × elevation / 1000 | 地形高程综合指数 |
| `log_dist_river` | log1p(distance_to_river) | 距河距离对数变换 |

### 地理聚类特征（11 维）

- `geo_cluster`：KMeans(n=10) 在 (lat, lon) 上的聚类标签
- `geo_dist_0` ~ `geo_dist_9`：到各聚类中心的距离

### 缺失值处理

- **降雨特征**：`IterativeImputer` 多重链式方程插补（max_iter=20）
- **ndvi/ndwi**：先中位数预填充，再 IterativeImputer 迭代插补
- 先拆分再 fit/transform，防止数据泄漏

### 标准化

- 连续特征做 `StandardScaler` Z-score 标准化
- 二值特征不做标准化

---

## 数据增强

`data_process/data_augmentation.py` 提供训练时数据增强：

| 方法 | 说明 |
|:----|:-----|
| **SMOTE** | 对少数类（标签 0）合成过采样，使训练集趋于平衡 |
| **高斯噪声增强** | 对连续特征加 5% 标准差噪声，生成变体样本 |
| **混合增强** | SMOTE + 噪声组合 |

**增强仅在训练时作用于内存数据**，不修改硬盘上的原始 CSV 文件。

---

## 模型列表

### 基础模型（9 个）

位于 `ML_algorithms/`，训练后自动保存至 `models/*.pkl`：

| 模型 | 文件 | 调参方式 | 类别平衡 |
|:----|:-----|:--------|:--------|
| 逻辑回归 | `logistic_regression.py` | 默认参数 | `class_weight='balanced'` |
| 决策树 | `decision_tree.py` | 默认参数 | `class_weight='balanced'` |
| 支持向量机 | `svm.py` | `probability=True` | `class_weight='balanced'` |
| 随机森林 | `random_forest.py` | 默认参数 | `class_weight='balanced_subsample'` |
| K 近邻 | `knn.py` | 默认参数 | — |
| 梯度提升 | `gboost.py` | 默认参数 | — |
| XGBoost | `xgboost.py` | 默认参数 | `scale_pos_weight=2.41` |
| CatBoost | `catboost.py` | `allow_writing_files=False` | `auto_class_weights='Balanced'` |
| LightGBM | `lightgbm.py` | 默认参数 | `class_weight='balanced'` |

### 改进集成算法（4 个）

位于 `improve_algorithms/`，从 `models/` 加载预训练基模型，无需重新训练：

| 模型 | 文件 | 核心策略 |
|:----|:-----|:--------|
| **Blending** | `blending.py` | 9 基模型概率 + 原始特征 + XGBoost 元学习器 |
| **Ensemble Avg** | `ensemble_avg.py` | 9 模型 AUC 加权 + 概率 + 统计特征 + 校准 |
| **Voting** | `voting.py` | 9 模型 AUC 加权 + 扩展元特征 + Plalibrated |
| **Stacking** | `stacking.py` | 扩展元特征（含交互项）+ Optuna + 校准 |

---

## 改进集成算法

### 架构统一

四个改进算法共享以下架构：

```
models/*.pkl → 加载 9 个预训练基模型
       ↓
    基模型概率预测 (9 维)
       ↓
    扩展元特征:
      - 9 个基模型概率
      - 统计量 (mean, std, max, min)
      - 交互特征 (top-5 概率两两乘积)
      - 原始 41 维特征
       ↓
    StandardScaler 标准化
       ↓
    XGBoost 元学习器 (Optuna 20 轮调参)
       ↓
    CalibratedClassifierCV (sigmoid 校准)
       ↓
    阈值优化 → 最终预测
```

### 各算法特点

| 算法 | 训练策略 | 独有特征 | Optuna 目标 |
|:----|:--------|:---------|:------------|
| **Blending** | 80/20 切分，混合集训练元学习器 | 概率 + 均值 + 标准差 + 原始特征 | AUC-ROC |
| **Ensemble Avg** | 全量训练集，Stacking 式元学习 | 概率 + 均值/标准差/最大/最小 + 原始特征 | AUC-ROC |
| **Voting** | 全量训练集，Stacking 式元学习 | 同 Ensemble Avg | AUC-ROC |
| **Stacking** | 全量训练集，含模型间交互项 | 概率 + 统计量 + top-5 交互乘积 + 原始特征 | AUC-ROC |

---

## 实验结果

### 最新性能对比（按 AUC-PR 排序）

| 排名 | 模型 | 准确率 | 精确率 | 召回率 | F1 值 | 特异性 | AUC-ROC | AUC-PR |
|:---:|:-----|:-----:|:-----:|:-----:|:----:|:------:|:-------:|:------:|
| 1 | LightGBM | 0.7423 | 0.7726 | 0.9004 | 0.8316 | 0.3613 | 0.7343 | 0.8625 |
| 2 | Blending(深度优化) | 0.7314 | 0.7651 | 0.8946 | 0.8248 | 0.3380 | 0.7052 | 0.8375 |
| 3 | CatBoost | 0.7334 | 0.7880 | 0.8520 | 0.8188 | 0.4476 | 0.7349 | 0.8642 |
| 4 | 随机森林 | 0.7225 | 0.7643 | 0.8781 | 0.8173 | 0.3473 | 0.7372 | 0.8691 |
| 5 | Ensemble(Avg深度优化) | 0.7218 | 0.8144 | 0.7853 | 0.7996 | 0.5688 | 0.7393 | 0.8672 |
| 6 | Voting(深度优化) | 0.7218 | 0.8144 | 0.7853 | 0.7996 | 0.5688 | 0.7394 | 0.8673 |
| 7 | 决策树 | 0.6869 | 0.7540 | 0.8269 | 0.7887 | 0.3497 | 0.6332 | 0.7801 |
| 8 | 逻辑回归 | 0.6582 | 0.8055 | 0.6809 | 0.7379 | 0.6037 | 0.6969 | 0.8372 |
| 9 | GBoost | 0.6637 | 0.8219 | 0.6692 | 0.7377 | 0.6503 | 0.7257 | 0.8576 |
| 10 | XGBoost | 0.6569 | 0.8333 | 0.6431 | 0.7260 | 0.6900 | 0.7371 | 0.8660 |
| 11 | Stacking(OOF深度优化) | 0.6500 | 0.8329 | 0.6315 | 0.7184 | 0.6946 | 0.7248 | 0.8547 |
| 12 | 支持向量机 | 0.6405 | 0.8223 | 0.6267 | 0.7113 | 0.6737 | 0.7027 | 0.8445 |
| 13 | K 近邻 | 0.6275 | 0.7907 | 0.6431 | 0.7093 | 0.5897 | 0.6648 | 0.8079 |

> 注：结果来源于单次运行，含随机性（Optuna、SMOTE）。最优阈值为验证集上多目标优化（0.6×F1 + 0.4×特异性）所得。

### 优化成效对比

| 指标 | 优化前 | 优化后最优 | 提升 |
|:----|:-----:|:---------:|:----:|
| **AUC-PR** | 0.8625 (LightGBM) | **0.8691** (随机森林) | ▲0.007 |
| **AUC-ROC** | 0.7343 (LightGBM) | **0.7394** (Voting) | ▲0.005 |
| **AUC-PR Ensemble** | 0.8625 | **0.8673** (Voting深度优化) | ▲0.005 |
| **F1 Blending** | 0.7123 | **0.8248** | ▲0.113 |

---

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, optuna, imbalanced-learn, joblib

### 运行全部模型评估

```bash
python train_models.py
```

依次训练 9 个基础模型 + 4 个改进算法，输出 `model_performance.csv` 和评估曲线。

### 单独运行集成算法

```bash
# 1. 先训练基础模型，生成预训练文件
python ML_algorithms/xgboost.py
python ML_algorithms/lightgbm.py
python ML_algorithms/catboost.py
python ML_algorithms/random_forest.py
python ML_algorithms/gboost.py
python ML_algorithms/svm.py
python ML_algorithms/logistic_regression.py
python ML_algorithms/decision_tree.py
python ML_algorithms/knn.py

# 2. 再运行改进算法（从 models/ 加载预训练模型）
python improve_algorithms/blending.py
python improve_algorithms/ensemble_avg.py
python improve_algorithms/voting.py
python improve_algorithms/stacking.py
```

### 运行特征重要性分析

```bash
python data_process/feature_analysis.py
```

---

## 训练日志

运行 `train_models.py` 后自动生成训练日志：

```
train_logs/
├── train_log_20260524_100252.md    # 按时间戳命名
└── ...
```

日志包含三大部分：

1. **训练基本信息** — 开始/结束时间、总耗时、数据特征
2. **模型性能对比** — 按 AUC-PR 排序的 Markdown 表格
3. **训练过程输出** — 所有控制台输出（含 Optuna 调参日志）

---

## 项目结构

```
landslide_prediction/
├── data_process/                        # 数据预处理包
│   ├── __init__.py                      # 导出 preprocess_data, augment_training_data
│   ├── preprocess.py                    # 数据预处理（43维特征）
│   ├── data_augmentation.py             # SMOTE + 噪声增强
│   └── feature_analysis.py              # 特征重要性分析
├── ML_algorithms/                       # 基础模型（9个）
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── svm.py
│   ├── knn.py
│   ├── random_forest.py
│   ├── gboost.py
│   ├── xgboost.py
│   ├── catboost.py
│   └── lightgbm.py
├── improve_algorithms/                  # 改进集成算法（4个）
│   ├── __init__.py
│   ├── blending.py
│   ├── ensemble_avg.py
│   ├── voting.py
│   └── stacking.py
├── models/                              # 预训练模型 pkl 文件
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── catboost.pkl
│   ├── random_forest.pkl
│   ├── gboost.pkl
│   ├── svm.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── knn.pkl
├── train_models.py                      # 统一训练入口
├── train_logs.py                        # 训练日志生成器
├── performance_evaluation.py            # 统一评估脚本
├── visualisation.py                     # 可视化模块
├── train_logs/                          # 训练日志输出目录
├── evaluation_curve/                    # 评估曲线输出目录
├── dataset/                             # 数据集
├── model_performance.csv                # 模型性能对比表
├── requirements.txt
└── README.md
```

---

## 贡献者

邓双林
