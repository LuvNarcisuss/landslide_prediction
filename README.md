# 阿坝州降雨诱发滑坡易发性评价 — Pipeline V2

基于 **阿坝州地质灾害隐患点普查数据（7315条）**，采用**空间约束负样本生成 + 二阶段易发性-暴露度分离**的机器学习建模框架，实现滑坡易发性（Susceptibility）的二分类评价。

采用**空间约束负样本生成 + 二阶段易发性-暴露度分离**的机器学习建模框架，符合滑坡易发性评价的国际学术规范。

---

## 目录

- [项目架构](#项目架构)
- [核心创新](#核心创新)
- [数据说明](#数据说明)
- [V2 流程](#v2-流程)
  - [1. 数据清洗](#1-数据清洗)
  - [2. 缺失值插补](#2-缺失值插补)
  - [3. 特征工程](#3-特征工程)
  - [4. 负样本生成](#4-负样本生成)
  - [5. 二阶段建模策略](#5-二阶段建模策略)
  - [6. 模型训练与评估](#6-模型训练与评估)
- [模型列表](#模型列表)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [评估指标](#评估指标)
- [注意事项](#注意事项)

---

## 项目架构

```
dataset/aba_disaster_distribution.csv (7315×31)
                         │
    ┌────────────────────▼────────────────────┐
    │         Pipeline V2 预处理              │
    │  (data_cleaning → imputation →         │
    │   pipeline_v2 → negative_sampling)     │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │         train_models_v2.py              │
    │  9个基模型 + 4个集成模型 + Optuna + SHAP │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │         hazard_mapping.py               │
    │  五级危险性分类 + 综合风险(易发性×暴露度)  │
    └─────────────────────────────────────────┘
```

| 核心特性 | 说明 |
|:---------|:-----|
| 标签定义 | 灾害点=1，非灾害点=0（外部生成） |
| 正样本 | 滑坡+不稳定斜坡+泥石流(5622) |
| 负样本 | 三源混合 + GIS空间约束随机生成 |
| 特征数 | 27维（分离暴露特征） |
| 暴露度 | 分离到风险分析阶段 |
| 学术规范 | 强（外部负样本，二阶段分离） |

---

## 核心创新

| 创新点 | 说明 |
|:-------|:-----|
| **易发性·暴露度二阶段分离** | 分类模型仅用地形+降雨+植被特征，暴露特征（人口/财产）不参与分类，用于风险分析 |
| **三源混合负样本生成** | 真实非降雨灾害 + 非目标灾害类型 + 空间约束随机采样，三源混合 + RF质量检验 |
| **多尺度降雨特征** | rain_3d（短期）、rain_7d（中期）、rain_30d（长期）+ 降雨强度比 + API |
| **综合暴露指数** | 加权融合威胁人口、户数、财产的归一化暴露指标 |
| **RF 负样本质量检验** | 随机森林交叉验证正负样本AUC，确保负样本质量在合理范围(0.85-0.97) |

### 研究范式

```
阶段1: 滑坡易发性 (Susceptibility)
  特征: 地形 + 降雨 + 植被 + 土地利用
  模型: RF / XGB / LGB / CAT 等
  输出: P(灾害|地形,降雨,植被)

         ↓

阶段2: 滑坡风险 (Risk)
  输入: 易发性概率 × 暴露度(人口/财产)
  输出: Risk = Susceptibility × Exposure
```

---

## 数据说明

### 原始数据

| 项目 | 内容 |
|:-----|:-----|
| 数据文件 | `dataset/aba_disaster_distribution.csv` |
| 样本量 | **7,315 条** |
| 维度 | **31 列** |
| 数据来源 | Landsat 7/8/5 遥感影像 + 气象站 + 实地普查 |
| 研究区域 | 四川省阿坝藏族羌族自治州 |
| 经度范围 | 100.02°E – 104.40°E |
| 纬度范围 | 30.67°N – 34.26°N |
| 高程范围 | 845 m – 4,514 m |
| 时间跨度 | 2008 – 2020 年 |

### 灾害类型分布

| 类型 | 样本数 | 占比 | V2 处理 |
|:-----|:------:|:----:|:--------|
| **滑坡** | 2,204 | 30.1% | 正样本 |
| **不稳定斜坡** | 1,113 | 15.2% | 正样本 |
| **泥石流** | 2,305 | 31.5% | 正样本 |
| **崩塌** | 1,689 | 23.1% | 天然负样本来源 |
| **地面塌陷** | 4 | 0.1% | 天然负样本来源 |

### 危险等级分布

| 等级 | 样本数 | 占比 |
|:----:|:------:|:----:|
| 小 | 6,311 | 86.3% |
| 中 | 944 | 12.9% |
| 大 | 42 | 0.57% |
| 特大 | 18 | 0.25% |

> 高危险样本（大+特大）仅 60 条(0.8%)，因此不以危险等级作为二分类标签。

---

## V2 流程

### 整体流程

```
dataset/aba_disaster_distribution.csv (7315×31)
                         │
    ┌────────────────────▼────────────────────┐
    │              data_cleaning.py            │
    │  • 灾害类型筛选 → 滑坡+不稳定斜坡+泥石流    │
    │  • 删除10个冗余字段                       │
    │  • 空值行删除 + 异常值IQR过滤 + 去重       │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │              imputation.py               │
    │  • NDVI (43%缺失) → 随机森林回归插补      │
    │  • NDWI (43%缺失) → NDVI线性回归 + RF    │
    │  • 降雨 (19.6%缺失) → MICE多变量插补      │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │       pipeline_v2.py 特征工程            │
    │  • 地形因子：elevation, aspect, tpi...   │
    │  • 降雨因子：rain_3d/7d/30d, api         │
    │  • 植被因子：ndvi, ndwi                  │
    │  • 衍生特征：rain_ratio, valley_risk等   │
    │  • 暴露特征分离（不参与分类）              │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │       negative_sampling.py              │
    │  • 三源混合负样本策略                      │
    │  • 来源A: 非降雨坡体灾害 (真实数据)       │
    │  • 来源B: 崩塌+地面塌陷 (真实数据)         │
    │  • 来源C: BallTree空间随机采样            │
    │  • 高程分布校正 + RF质量检验              │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │          train_models_v2.py              │
    │  • 9个基模型 + 4个集成模型                │
    │  • [可选] Optuna超参优化                  │
    │  • 概率校准 + 阈值优化                    │
    │  • SHAP特征重要性分析                     │
    │  • ROC/PR评估曲线                        │
    │  • 结果排名CSV输出                        │
    └─────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │            hazard_mapping.py             │
    │  • 五级危险性分类 (quantile/equal)        │
    │  • 综合风险 = 易发性 × 暴露度            │
    │  • 空间散点图 + 分布直方图 + 仪表盘       │
    │  • 带地形底图的危险性分布图               │
    └─────────────────────────────────────────┘
```

---

### 1. 数据清洗

**文件**: `pre_process/data_cleaning.py`

#### 1a. 灾害类型筛选

只保留与滑坡同源的坡体灾害：

| 保留 | 原因 |
|:-----|:-----|
| 滑坡 | 核心研究对象 |
| 不稳定斜坡 | 滑坡前期状态 |
| 泥石流 | 与滑坡同降雨驱动机制，源自有滑坡堆积物 |

**排除**: 崩塌(1689条) + 地面塌陷(4条) → 作为 V2 天然负样本来源

#### 1b. 删除字段

```
删除: id, discovery_date, location, data_source,
      development_trend, industry, industry_transferred,
      disaster_type, disaster_scale, scale_grade
原因: 纯元数据 / 缺失 > 80% / 与hazard_type冗余
```

#### 1c. 空值行删除

| 字段 | 缺失率 | 处理方式 |
|:-----|:------:|:---------|
| `triggering_factor` | 0.9% | 行删除 |
| `threat_target` | 1.0% | 行删除 |

#### 1d. 异常值过滤

```
• elevation ∈ [500, 5500] m
• aspect ∈ [0, 360]°
• rain_* ≥ 0
• tpi: 1%~99% IQR
• distance_to_river: 1%~99% IQR
```

#### 1e. 重复值删除 + 缺失标记

- `lon + lat` 双重去重
- 添加缺失指示器 `ndvi_missing`, `ndwi_missing`, `rain_missing`

---

### 2. 缺失值插补

**文件**: `pre_process/imputation.py`

#### 2a. NDVI 插补（43% 缺失）

```python
# 随机森林回归预测 NDVI
特征: elevation + tpi + distance_to_river + lon + lat + landuse
模型: RandomForestRegressor(n_estimators=200, max_depth=10)
验证: 训练R² ≈ 0.6-0.8
```

#### 2b. NDWI 插补（43% 缺失）

```
两步插补:
  Step 1: NDWI ~ NDVI 线性回归 (NDVI-NDWI高度负相关)
  Step 2: 剩余缺失用 RF 预测
```

#### 2c. 降雨数据插补（19.6% 缺失）

```python
# MICE 多重链式方程插补
模型: IterativeImputer(RF, max_iter=30)
联合插补: rain_3d + rain_7d + rain_30d + api + elevation + lon + lat
```

---

### 3. 特征工程

**文件**: `pre_process/pipeline_v2.py`

#### 易发性特征（27个，用于分类）

| 类别 | 特征 | 数量 |
|:-----|:-----|:----:|
| **地形** | `elevation`, `aspect`, `tpi`, `distance_to_river` | 4 |
| **降雨** | `rain_3d`, `rain_7d`, `rain_30d`, `api` | 4 |
| **植被** | `ndvi`, `ndwi` | 2 |
| **坐标** | `lon`, `lat` | 2 |
| **土地利用** | `landuse`（One-hot编码为6个哑变量） | 1→7 |
| **衍生特征** | 见下表 | 8 |

**衍生特征详解**

| 特征 | 公式 | 物理意义 |
|:-----|:-----|:---------|
| `rain_ratio_3d_30d` | rain_3d / rain_30d | 短期降雨集中度 |
| `rain_ratio_7d_30d` | rain_7d / rain_30d | 中期降雨集中度 |
| `rain_intensity` | rain_3d / rain_7d | 降雨强度变化率 |
| `valley_risk` | max(0, -tpi) / (dist_river + 1) | 沟谷汇水风险 |
| `elev_rain` | elevation × log(1 + rain_3d) | 高海拔+强降雨交互 |
| `aspect_rain` | cos(aspect) × rain_3d | 迎风坡降雨增强 |
| `ndvi_elev` | ndvi × elevation / 1000 | 植被覆盖与高程耦合 |
| `log_dist_river` | log1p(distance_to_river) | 距河距对数变换 |

#### 暴露特征（4个，不参与分类，用于风险分析）

| 特征 | 说明 |
|:-----|:-----|
| `threatened_people` | 威胁人口数 |
| `threatened_households` | 威胁户数 |
| `threatened_property` | 威胁财产（万元） |
| `exposure_index` | 综合暴露指数 = 0.4×People + 0.3×HH + 0.3×Property |

> **关键设计**：暴露特征在正样本中有正值，负样本中全为0。若加入分类特征，模型会"作弊"（仅靠暴露度就能区分），无法学到真实的滑坡规律。因此将其分离到风险分析阶段。

---

### 4. 负样本生成

**文件**: `pre_process/negative_sampling.py`

#### 三源混合负样本策略

本模块采用 **三源混合策略** 生成非灾害点负样本：

```
来源A: 同一数据集中非降雨诱发的坡体灾害（同类型但触发机制不同）
来源B: 非目标灾害类型（崩塌 + 地面塌陷）
来源C: 空间随机采样（BallTree 缓冲区过滤 + 特征继承 + 噪声）
```

| 来源 | 方法 | 特点 | 默认比例 |
|:-----|:-----|:-----|:--------|
| A | 非降雨诱发的滑坡/不稳定斜坡/泥石流 | 真实分布，天然负样本 | 混合占比30% |
| B | 崩塌 + 地面塌陷 | 真实分布，非目标灾害 | 混合占比30% |
| C | BallTree空间随机采样 | 覆盖无灾害区域，数量可控 | 混合占比70% |

#### 来源C：空间约束随机采样

```python
1. 确定采样范围: 正样本经纬度范围扩展5%
2. 构建空间索引: BallTree(haversine球面距离)
3. 循环采样:
   a. 在范围内随机生成候选点(lat, lon)
   b. 查询最近灾害点距离
   c. 距离 > 800m → 接受，否则重试
4. 特征继承:
   a. 找到最近3个灾害点
   b. 加权平均(按反距离) + 高斯噪声
   c. 暴露度置0
5. 高程校正: QuantileTransformer 匹配正样本高程分布
6. 标签: 0（非灾害点）
```

#### 参数配置

| 参数 | 默认值 | 说明 |
|:-----|:------:|:-----|
| `ratio` | 1.0 | 负:正样本比例 |
| `min_dist_km` | 0.8 | 距灾害点最小距离(km) |
| `n_neighbors` | 3 | 特征继承的最近邻数 |
| `hybrid_ratio` | 0.3 | 混合负样本中真实数据占比 |
| `correct_elevation` | True | 是否校正高程分布 |

#### RF 质量检验

使用随机森林交叉验证检测正负样本的可分性，确保负样本质量：

| AUC 范围 | 状态 | 含义 |
|:--------:|:----:|:-----|
| > 0.97 | FAIL | 差异过大，需增大缓冲区或特征噪声 |
| 0.85 – 0.97 | PASS | 合理范围，负样本质量合格 |
| 0.80 – 0.85 | WARN | 偏低，负样本接近正样本分布 |
| < 0.80 | FAIL | 几乎不可分，可能包含潜在灾害点 |

---

### 5. 二阶段建模策略

这是本项目的核心方法论设计。

#### 第一阶段：滑坡易发性（Susceptibility）

```python
输入特征: 地形 + 降雨 + 植被 + 土地利用（27维）
标签: 1 = 灾害点, 0 = 非灾害点
模型: 9个分类器 + 4个集成方法
输出: P(灾害 | 地形,降雨,植被)
```

#### 第二阶段：滑坡风险（Risk）

```python
输入: 易发性概率 P_sus
      暴露度指数 E = 0.4·People + 0.3·HH + 0.3·Property
风险: R = P_sus × E
分级: 极低 / 低 / 中 / 高 / 极高
```

这样设计的原因：

| 常见错误做法 | 本项目的正确做法 |
|:-------------|:-----------------|
| 暴露特征直接加入分类模型（模型学会"有人的地方就是灾害"） | 暴露特征分离到风险阶段（模型学习真实滑坡规律） |
| 忽略暴露度，仅预测灾害概率（无法评估实际危害程度） | 二阶段级联：易发性 × 暴露度（同时考虑自然和人文因素） |

---

### 6. 模型训练与评估

**文件**: `train_models_v2.py`

```
加载数据 (V2 Pipeline)
    │
    ▼
┌── 9个基模型 ──────────────────────────────────────────┐
│  logistic_regression  → 逻辑回归                       │
│  decision_tree        → 决策树                         │
│  svm                  → 支持向量机(rbf核)              │
│  knn                  → K近邻(k=7, distance权重)      │
│  random_forest        → 随机森林(200棵树)              │
│  gboost               → 梯度提升(200轮, lr=0.1)       │
│  xgboost              → XGBoost(正负比权重)           │
│  lightgbm             → LightGBM(balanced权重)        │
│  catboost             → CatBoost(Balanced权重)        │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌── 4个集成模型 ─────────────────────────────────────────┐
│  ensemble_avg   → 9模型概率算术平均                     │
│  voting         → VotingClassifier(软投票)             │
│  stacking       → StackingClassifier(5折CV+LR元模型)  │
│  blending       → hold-out验证集+LR元模型              │
└──────────────────────────────────────────────────────┘
    │
    ▼  (可选 --optuna-trials N)
┌── Optuna超参优化 ──────────────────────────────────────┐
│  random_forest_tuned  贝叶斯调参30-50次                │
│  xgboost_tuned        搜索学习率/树深/正则化           │
│  lightgbm_tuned       搜索叶子数/子采样/正则化          │
│  catboost_tuned       搜索深度/学习率/L2正则            │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌── 后处理 ──────────────────────────────────────────────┐
│  CalibratedClassifierCV(sigmoid)  概率校准              │
│  阈值搜索 (0.05-0.95, 步长0.01)   多目标优化            │
│  目标: 0.6×F1 + 0.4×特异性                             │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌── 输出 ────────────────────────────────────────────────┐
│  results_v2/model_performance.csv  13+模型按F1排名      │
│  results_v2/roc_curves.png         ROC曲线              │
│  results_v2/pr_curves.png          PR曲线               │
│  results_v2/shap_summary.png       SHAP特征重要性       │
│  results_v2/shap_bar.png           SHAP柱状图           │
│  results_v2/shap_importance.csv    SHAP值排序           │
│  models_v2/*.pkl                   训练好的模型          │
└──────────────────────────────────────────────────────┘
```

---

## 模型列表

### 基模型（9个）

| 模型 | 参数配置 |
|:-----|:---------|
| 逻辑回归 | `class_weight='balanced', C=1.0, solver='saga'` |
| 决策树 | `max_depth=8, class_weight='balanced'` |
| SVM | `kernel='rbf', probability=True, class_weight='balanced'` |
| 随机森林 | `n_estimators=200, class_weight='balanced_subsample'` |
| KNN | `n_neighbors=7, weights='distance'` |
| GBDT | `n_estimators=200, max_depth=4, lr=0.1` |
| XGBoost | `scale_pos_weight=正负比` |
| CatBoost | `auto_class_weights='Balanced'` |
| LightGBM | `class_weight='balanced'` |

### 集成模型（4个）

| 模型 | 策略 |
|:-----|:------|
| **平均值集成** | 9个模型概率的算术平均 |
| **Voting** | 软投票（概率均值） |
| **Stacking** | 5折交叉验证 + 逻辑回归元学习器 |
| **Blending** | hold-out验证集 + 逻辑回归元学习器 |

### 后处理

```python
• 概率校准: CalibratedClassifierCV(sigmoid, cv=3)
• 阈值优化: 多目标搜索 (0.6×F1 + 0.4×特异性), 步长0.01
```

---

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, optuna, imbalanced-learn, shap, joblib, contextily

### 运行 V2 完整训练

```bash
# 默认 1:1 正负样本比例（V2主流程）
python train_models_v2.py

# 自定义负样本比例（快速测试用 0.5）
python train_models_v2.py --neg-ratio 0.5

# 三源混合负样本中真实数据占比
python train_models_v2.py --hybrid-ratio 0.5

# 缓冲区距离（负样本距灾害点最小距离 km）
python train_models_v2.py --buffer-dist 1.0

# 开启 Optuna 超参优化（对RF/XGB/LGB/CAT各跑30次）
python train_models_v2.py --optuna-trials 30

# 使用 GPU 加速
python train_models_v2.py --use-gpu

# 综合命令：负样本1:1 + Optuna调参 + 默认混合
python train_models_v2.py --neg-ratio 1.0 --optuna-trials 30

# 跳过 RF 质量检验
python train_models_v2.py --no-quality-check

# 跳过 SHAP 分析
python train_models_v2.py --no-shap
```

### 危险性分级与风险制图

先训练模型，然后运行：

```bash
# 自动选择最佳模型
python hazard_mapping.py

# 指定模型
python hazard_mapping.py --model-path models_v2/xgboost.pkl

# 用等距分级法（默认五分位数）
python hazard_mapping.py --method equal
```

### 输出文件

```
results_v2/
├── model_performance.csv              # 模型性能排名（按F1排序）
├── roc_curves.png                     # ROC曲线对比
├── pr_curves.png                      # PR曲线对比
├── shap_summary.png                   # SHAP特征重要性
├── shap_bar.png                       # SHAP柱状图
├── shap_importance.csv                # SHAP值排序
│
├── susceptibility_distribution.png    # 易发性概率分布
├── hazard_classification.png          # 五级危险性空间分布
├── hazard_classification_basemap.png  # 带地形底图的危险性分布图
├── risk_map.png                       # 综合风险(易发性×暴露度)空间分布
├── hazard_analysis_dashboard.png      # 综合分析仪表盘
├── hazard_breakdown.csv               # 各等级统计汇总
└── hazard_prediction_full.csv         # 全量预测数据

models_v2/
├── xgboost.pkl
├── lightgbm.pkl
├── random_forest.pkl
├── catboost.pkl
├── xgboost_tuned.pkl                  # Optuna调优版
├── random_forest_tuned.pkl
├── voting.pkl
├── stacking.pkl
├── blending.pkl
└── ...
```

### 自定义调用

```python
from pre_process.pipeline import preprocess_data

# 获取训练数据
X_train, X_test, y_train, y_test, extra = preprocess_data(
  neg_ratio=1.0, verbose=True
)

# 易发性特征
feature_names = extra['feature_names']  # 27个特征名

# 暴露特征（风险分析用）
exposure_train = extra['exposure_train']  # 训练集暴露度
exposure_test = extra['exposure_test']  # 测试集暴露度

# 用你训练的模型预测易发性
y_prob = model.predict_proba(X_test)[:, 1]

# 计算综合风险
risk = y_prob * exposure_test['exposure_index'].values
```

---

## 项目结构

```
landslide_prediction/
├── dataset/                                # 数据集
│   ├── aba_disaster_distribution.csv       # 原始数据（7315×31，只读）
│   ├── aba_disaster_processed.csv          # 处理后数据
│   ├── simulated_landslide_large.csv       # 模拟数据（20000条）
│   └── simulated_landslide_small.csv       # 模拟数据（2000条）
│
├── pre_process/                            # 预处理模块
│   ├── __init__.py                         # 导出模块
│   ├── pipeline_v2.py                      # V2核心特征工程
│   ├── data_cleaning.py                    # 数据清洗
│   ├── imputation.py                       # 缺失值插补
│   └── negative_sampling.py                # 三源混合负样本生成
│
├── train_models_v2.py                      # V2训练入口
├── hazard_mapping.py                       # 危险性分级与风险分析
├── visualisation.py                        # 可视化模块
│
├── models_v2/                              # 模型输出（自动创建）
├── results_v2/                             # 结果输出（自动创建）
│
├── requirements.txt                        # 依赖
├── README.md
├── CLAUDE.md                               # 项目指令
└── 项目整体实现流程.md                      # 架构说明
```

---

## 评估指标

| 指标 | 公式 | 说明 |
|:-----|:-----|:-----|
| **准确率** | (TP+TN) / (TP+TN+FP+FN) | 总体正确率 |
| **精确率** | TP / (TP+FP) | 预测为正的样本中实际为正的比例 |
| **召回率** | TP / (TP+FN) | 实际为正的样本中被正确识别的比例 |
| **F1值** | 2×P×R / (P+R) | 精确率和召回率的调和平均 |
| **特异性** | TN / (TN+FP) | 负样本被正确识别的比例 |
| **AUC-ROC** | ROC曲线下面积 | 整体排序质量 |
| **AUC-PR** | PR曲线下面积 | 适合不平衡数据 |

> **排序指标**（V2）：所有模型按 **F1值** 排序（兼顾精确率和召回率）。

---

## Optuna 超参优化

`train_models_v2.py` 集成了 Optuna 贝叶斯超参优化，支持对4个主要模型进行自动调参。

### 优化目标

| 模型 | 调优参数 | 搜索空间 |
|:-----|:---------|:---------|
| **随机森林** | n_estimators, max_depth, min_samples_split, max_features | 100-600棵树, 5-25层 |
| **XGBoost** | n_estimators, max_depth, lr, subsample, colsample, gamma, reg | 含正则化参数 |
| **LightGBM** | n_estimators, max_depth, num_leaves, lr, subsample, reg | 含叶子节点数 |
| **CatBoost** | iterations, depth, lr, subsample, l2_leaf_reg | 4-10层深度 |

### 运行方式

```bash
# 每个模型30次调参（约10-15分钟）
python train_models_v2.py --optuna-trials 30

# 每个模型50次（更充分，约20-30分钟）
python train_models_v2.py --optuna-trials 50
```

### 输出

```
Optuna 调参 random_forest (30 次)... 最佳F1=0.9852
Optuna 调参 xgboost (30 次)...      最佳F1=0.9921
Optuna 调参 lightgbm (30 次)...     最佳F1=0.9915
Optuna 调参 catboost (30 次)...     最佳F1=0.9918
```

调优后的模型自动保存为 `models_v2/{name}_tuned.pkl`，并参与最终排名。

---

## 危险性分级与风险分析

`hazard_mapping.py` 在模型训练完成后，对全量数据进行易发性分级和综合风险评估。

### 二阶段风险框架

```
第一阶段: 滑坡易发性 Susceptibility
  输入: 地形 + 降雨 + 植被 + 土地利用
  输出: P(灾害) ∈ [0, 1]
  分级: 极低 / 低 / 中 / 高 / 极高

第二阶段: 综合风险 Risk
  风险 = 易发性 × 暴露度
  暴露度 = 0.4·(People) + 0.3·(HH) + 0.3·(Property)
  分级: 极低 / 低 / 中 / 高 / 极高
```

### 分级方法

| 方法 | 说明 |
|:-----|:-----|
| `quantile`（默认） | 五分位数等样本量分级，每级恰好20% |
| `equal` | 等距分级 [0,0.2,0.4,0.6,0.8,1] |

### 输出图表

| 图表 | 内容 |
|:-----|:------|
| `susceptibility_distribution.png` | 易发性概率分布直方图 + 五级标注 |
| `hazard_classification.png` | 五级危险性空间散点图 |
| `hazard_classification_basemap.png` | 带地形底图的危险性分布图 |
| `risk_map.png` | 综合风险(易发性×暴露度)空间分布 |
| `hazard_analysis_dashboard.png` | 综合分析仪表盘（分布+空间+统计） |
| `hazard_breakdown.csv` | 各等级统计汇总 |
| `hazard_prediction_full.csv` | 全量预测数据下载 |

---

---

## 注意事项

1. **负样本生成**：负样本的合理性直接影响模型质量。建议根据研究区实际情况调整 `min_dist_km` 和 `hybrid_ratio` 参数。运行后查看 RF 质量检验结果，AUC 应在 0.85-0.97 之间。

2. **NDVI插补**：43%的NDVI缺失率较高，若有条件建议从GEE重新提取以替代插补。

3. **暴露度分析**：当前风险分析为简化模型（线性加权），实际应用中可考虑更复杂的暴露度评估方法。

4. **模型选择**：若追求可解释性优先选择随机森林或XGBoost；若追求性能可尝试Stacking集成。

5. **GPU加速**：对于大规模训练，可使用 `--use-gpu` 参数加速 XGBoost/LightGBM/CatBoost 训练。

6. **dataset/ 目录禁止修改**：原始数据文件是只读的，数据增强仅在训练时作用于内存。

---

## 引用说明

若使用本项目的代码或方法，请引用：

```bibtex
@software{landslide_prediction_v2,
  title = {阿坝州降雨诱发滑坡易发性评价 - Pipeline V2},
  author = {Deng Shuanglin},
  year = {2026},
  description = {基于三源混合负样本生成和二阶段易发性-暴露度分离的滑坡易发性机器学习评价}
}
```

---

> 本文档覆盖项目 V2 完整流程。
