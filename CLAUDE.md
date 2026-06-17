# 滑坡预测 (Landslide Prediction) — Pipeline V2

基于机器学习的滑坡易发性评价二分类项目。利用四川省阿坝州真实灾害隐患点普查数据，采用**空间约束负样本生成 + 二阶段易发性-暴露度分离**框架进行建模评估。

## 技术栈

- 语言：Python 3.8+
- 机器学习：scikit-learn、XGBoost、LightGBM、CatBoost
- 超参优化：Optuna
- 数据预处理：pandas、numpy
- 可视化：matplotlib
- 模型解释：SHAP

## 常用命令

### 环境安装

```bash
pip install -r requirements.txt
```

### 运行 V2 完整训练

```bash
python train_models_v2.py                                    # 默认 1:1 负样本
python train_models_v2.py --neg-ratio 0.5                    # 快速测试
python train_models_v2.py --optuna-trials 30                 # 开启 Optuna 调参
python train_models_v2.py --use-gpu                          # GPU 加速
python train_models_v2.py --buffer-dist 1.0                  # 自定义缓冲区距离
python train_models_v2.py --hybrid-ratio 0.5                 # 混合负样本比例
```

### 危险性分级与风险制图

```bash
python hazard_mapping.py
python hazard_mapping.py --method equal
```

## 项目结构

```
landslide_prediction/
├── pre_process/                   # 数据预处理包
│   ├── __init__.py                # 导出模块
│   ├── pipeline_v2.py             # 特征工程（27维易发性特征）
│   ├── data_cleaning.py           # 数据清洗
│   ├── imputation.py              # 缺失值插补（NDVI RF回归 + 降雨MICE）
│   └── negative_sampling.py       # 三源混合负样本生成
├── models_v2/                     # 预训练模型 pkl 文件
├── results_v2/                    # 训练结果输出目录
├── dataset/                       # 数据集（只读）
├── train_models_v2.py             # V2 训练主入口
├── hazard_mapping.py              # 危险性分级与风险分析
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
# ══════════════════════
# 步骤1 — 加载数据   
# ══════════════════════
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
- 每次回复时都需要提行说"主人您好，我是claude，很高兴为您服务！"

## 注意事项

- **依赖更新**：修改 `requirements.txt` 后提醒用户运行 `pip install -r requirements.txt`，不要自行执行 pip install
- **`dataset/` 目录禁止修改**：原始数据文件是只读的，数据增强仅在训练时作用于内存
- **`models_v2/` 目录**：预训练模型存放处，修改前需确认不影响已有模型
- **代码修改流程**：提出方案供用户审查，方案通过后直接批量执行，无需逐个编辑等待批准
