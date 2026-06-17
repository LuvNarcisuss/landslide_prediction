# 滑坡预测 (Landslide Prediction)

基于机器学习的滑坡灾害预测二分类项目。利用四川省汶川县耿达镇的真实灾害隐患点普查数据，通过多种分类模型进行训练与对比评估。

## 技术栈

- 语言：Python 3.8+
- 机器学习：scikit-learn、XGBoost、LightGBM、CatBoost
- 超参优化：Optuna
- 数据增强：imbalanced-learn (SMOTE)
- 可视化：matplotlib

## 常用命令

### 环境安装

```bash
pip install -r requirements.txt
```

### 运行全部模型评估

```bash
python train_models.py
```

### 单独训练基础模型

```bash
python ML_algorithms/xgboost.py
python ML_algorithms/lightgbm.py
python ML_algorithms/catboost.py
python ML_algorithms/random_forest.py
python ML_algorithms/gboost.py
python ML_algorithms/svm.py
python ML_algorithms/logistic_regression.py
python ML_algorithms/decision_tree.py
python ML_algorithms/knn.py
```

### 运行改进集成算法（需先训练基础模型）

```bash
python improve_algorithms/blending.py
python improve_algorithms/ensemble_avg.py
python improve_algorithms/voting.py
python improve_algorithms/stacking.py
```

### 特征重要性分析

```bash
python pre_process/feature_analysis.py
```

## 项目结构

```
landslide_prediction/
├── pre_process/                   # 数据预处理包
│   ├── __init__.py                # 导出 preprocess_data, augment_training_data
│   ├── preprocess.py              # 数据预处理（43维特征）
│   ├── data_augmentation.py       # SMOTE + 噪声增强
│   └── feature_analysis.py        # 特征重要性分析
├── ML_algorithms/                 # 基础模型（9个）
│   ├── train_logistic_regression.py
│   ├── train_decision_tree.py
│   ├── train_svm.py
│   ├── train_knn.py
│   ├── train_random_forest.py
│   ├── train_gboost.py
│   ├── train_xgboost.py
│   ├── train_catboost.py
│   └── train_lightgbm.py
├── improve_algorithms/            # 改进集成算法（4个）
│   ├── train_imp_xgboost.py
│   ├── train_imp_catboost.py
│   ├── train_imp_lightgbm.py
│   ├── train_ensemble_avg.py
│   ├── train_blending.py
│   ├── train_voting.py
│   └── train_stacking.py
├── models/                        # 预训练模型 pkl 文件
├── dataset/                       # 数据集
├── train_logs/                    # 训练日志输出目录
├── train_models.py                # 统一训练入口
├── train_logs.py                  # 训练日志生成器
├── visualisation.py               # 可视化模块
├── requirements.txt
└── README.md
```

## 编码规范

### 注释格式

使用 `# 步骤x - 简短说明` 平铺注释，禁止框线装饰。

正确：
```python
# 步骤1 - 加载数据
# 步骤2 - 特征工程
```

禁止：
```python
# ╔══════════════════════╗
# ║  步骤1 — 加载数据   ║
# ╚══════════════════════╝
```

### 图片标题全部用中文

所有 matplotlib 文字标签必须用中文（`set_title`、`set_xlabel`、`set_ylabel`、`text`），英文人名除外。代码中需设置中文字体：

```python
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False
```

### 输出格式

禁止 `=====` 或 `----` 分割线，用 `print("")` 空行分隔不同步骤的输出。

### 其他

- 不导入未使用的库（尤其不要默认导入 seaborn）
- 生成的 `.py` 文件必须可直接运行，无需用户手动补步骤
- 每次回复时都需要提行说“主人您好，我是reasonix，很高兴为您服务！”

## 注意事项

- **依赖更新**：修改 `requirements.txt` 后提醒用户运行 `pip install -r requirements.txt`，不要自行执行 pip install
- **`dataset/` 目录禁止修改**：原始数据文件是只读的，数据增强仅在训练时作用于内存
- **`models/` 目录**：改进集成算法从这里加载预训练基模型，修改前需确认不影响已有模型
- **代码修改流程**：提出方案供用户审查，方案通过后直接批量执行，无需逐个编辑等待批准
