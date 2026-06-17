"""
危险性分级与风险分析
基于训练好的最佳模型，对全量数据进行滑坡易发性分级与风险制图。

输出:
- susceptibility_distribution.png   — 易发性概率分布直方图
- hazard_classification.png         — 五级危险性空间散点图
- hazard_classification_basemap.png — 带地形底图的危险性分布图
- risk_map.png                      — 风险(易发性×暴露度)分级图
- hazard_breakdown.csv              — 各等级统计汇总

用法:
    python hazard_mapping.py
    python hazard_mapping.py --model-path models_v2/xgboost.pkl
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)

from pre_process.pipeline import preprocess_data
from pre_process.data_cleaning import clean_data
from pre_process.imputation import impute_all


# 步骤1 - 加载最佳模型
def load_best_model(model_path=None):
    """加载最佳模型，自动从排名中选第一个"""
    if model_path and os.path.exists(model_path):
        print(f"加载指定模型: {model_path}")
        return joblib.load(model_path), os.path.basename(model_path).replace('.pkl', '')

    # 从性能排名中选最佳
    perf_path = os.path.join(RESULTS_DIR, 'model_performance.csv')
    if os.path.exists(perf_path):
        perf = pd.read_csv(perf_path)
        best_name = perf.iloc[0]['模型名称']
        model_path = os.path.join(ROOT_DIR, 'models_v2', f'{best_name}.pkl')
        if os.path.exists(model_path):
            print(f"自动选择最佳模型: {best_name}")
            return joblib.load(model_path), best_name

    # 按优先级尝试常见模型
    for name in ['xgboost', 'lightgbm', 'random_forest', 'catboost']:
        mp = os.path.join(ROOT_DIR, 'models_v2', f'{name}.pkl')
        if os.path.exists(mp):
            print(f"使用模型: {name}")
            return joblib.load(mp), name

    print(f"错误: 未找到模型文件 ({model_path})")
    print("请先运行: python train_models_v2.py")
    sys.exit(1)


# 步骤2 - 获取全量数据的预测概率
def get_full_predictions(model, extra, verbose=True):
    """对全量数据（正样本+负样本）进行预测"""
    X_train_df = extra['X_train_df']
    X_test_df = extra['X_test_df']
    exposure_train = extra['exposure_train']
    exposure_test = extra['exposure_test']

    # 合并训练集和测试集
    X_full = pd.concat([X_train_df, X_test_df], ignore_index=True)
    exp_full = pd.concat([exposure_train, exposure_test], ignore_index=True)

    # 预测概率
    try:
        proba = model.predict_proba(X_full.values)[:, 1]
    except Exception:
        try:
            proba = model.predict_proba(X_full)[:, 1]
        except Exception:
            proba = model.predict(X_full).astype(float)

    if verbose:
        print(f"全量样本: {len(proba)} 条")
        print(f"预测概率范围: [{proba.min():.4f}, {proba.max():.4f}]")
        print(f"预测概率均值: {proba.mean():.4f}")

    return proba, X_full, exp_full


# 步骤3 - 五级危险性分类
def classify_hazard(proba, method='quantile', verbose=True):
    """
    将易发性概率分为5级

    分级方法:
    - 'quantile': 五分位数（每级20%样本）
    - 'equal': 等距划分 [0,0.2,0.4,0.6,0.8,1]
    - 'custom': 自定义阈值 [0.1, 0.3, 0.5, 0.7]
    """
    if method == 'quantile':
        bins = [0] + [np.percentile(proba, p) for p in [20, 40, 60, 80]] + [1]
        # 确保边界单调
        bins = sorted(set(bins))
    elif method == 'equal':
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    else:
        bins = [0, 0.1, 0.3, 0.5, 0.7, 1]

    labels = ['极低', '低', '中', '高', '极高']
    hazard_level = pd.cut(proba, bins=bins, labels=labels, include_lowest=True)
    # 兼容 pandas 不同版本：Categorical 或 Series
    hazard_code = (hazard_level.codes if hasattr(hazard_level, 'codes')
                   else hazard_level.cat.codes) + 1  # 1-5

    if verbose:
        print(f"\n五级危险性分类 (方法: {method}):")
        print(f"  阈值: {[round(b, 4) for b in bins]}")
        for l in labels:
            count = (hazard_level == l).sum()
            print(f"  {l}: {count} 条 ({count/len(proba)*100:.1f}%)")

    return hazard_level, hazard_code, bins


# 步骤4 - 综合风险分析
def compute_risk(proba, exposure_df, exposure_cols, verbose=True):
    """
    综合风险 = 易发性概率 × 暴露指数

    暴露指数 = 0.4·(people/max) + 0.3·(hh/max) + 0.3·(property/max)
    """
    if 'exposure_index' in exposure_df.columns:
        exposure = exposure_df['exposure_index'].values
    else:
        # 手动计算
        eps = 1e-8
        e = np.zeros(len(exposure_df))
        for col in ['threatened_people', 'threatened_households', 'threatened_property']:
            if col in exposure_df.columns:
                e += exposure_df[col].values / (exposure_df[col].max() + eps)
        exposure = e / max(e.max(), eps)

    risk = proba * exposure

    if verbose:
        print(f"\n综合风险分析:")
        print(f"  风险值范围: [{risk.min():.6f}, {risk.max():.6f}]")
        print(f"  风险均值: {risk.mean():.6f}")
        print(f"  高风险(>0.1)样本: {(risk > 0.1).sum()} 条 ({(risk > 0.1).sum()/len(risk)*100:.1f}%)")

    return risk, exposure


def classify_risk(risk, verbose=True):
    """将综合风险分为5级"""
    bins = [0, 0.001, 0.01, 0.05, 0.1, 1]
    labels = ['极低', '低', '中', '高', '极高']
    risk_level = pd.cut(risk, bins=bins, labels=labels, include_lowest=True)

    if verbose:
        print(f"\n五级风险等级:")
        for l in labels:
            count = (risk_level == l).sum()
            print(f"  {l}: {count} 条 ({count/len(risk)*100:.1f}%)")

    return risk_level


# 步骤5 - 可视化
def plot_susceptibility_distribution(proba, save_dir, verbose=True):
    """易发性概率分布直方图"""
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(proba, bins=50, alpha=0.7, color='steelblue', edgecolor='white')

    # 标注分级阈值
    thresholds = [0.2, 0.4, 0.6, 0.8]
    labels = ['极低', '低', '中', '高', '极高']
    colors = ['green', 'yellowgreen', 'orange', 'tomato', 'darkred']

    for t in thresholds:
        plt.axvline(t, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # 填充区域颜色
    for i, (lo, hi) in enumerate(zip([0]+thresholds, thresholds+[1])):
        plt.axvspan(lo, hi, alpha=0.05, color=colors[i])

    plt.xlabel('滑坡易发性概率', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.title('滑坡易发性概率分布', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    # 添加标注
    for i, (lo, hi) in enumerate(zip([0]+thresholds, thresholds+[1])):
        mid = (lo + hi) / 2
        plt.text(mid, plt.ylim()[1] * 0.95, labels[i],
                 ha='center', fontsize=9, color=colors[i], fontweight='bold')

    plt.tight_layout()
    path = os.path.join(save_dir, 'susceptibility_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"易发性分布图: {path}")


def plot_hazard_scatter(proba, hazard_level, lon, lat, save_dir, verbose=True):
    """危险性空间分布散点图"""
    colors_map = {'极低': 'green', '低': 'lime', '中': 'orange', '高': 'red', '极高': 'darkred'}
    colors = [colors_map.get(l, 'gray') for l in hazard_level]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(lon, lat, c=colors, s=8, alpha=0.6, edgecolors='none')

    # 图例
    for label, color in colors_map.items():
        plt.scatter([], [], c=color, label=label, s=30)
    plt.legend(title='危险性等级', loc='upper right', fontsize=10)

    plt.xlabel('经度', fontsize=12)
    plt.ylabel('纬度', fontsize=12)
    plt.title('滑坡危险性空间分布', fontsize=14)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    path = os.path.join(save_dir, 'hazard_classification.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"危险性空间分布: {path}")


def plot_risk_map(risk_level, lon, lat, save_dir, verbose=True):
    """综合风险空间分布图"""
    colors_map = {'极低': 'green', '低': 'lime', '中': 'orange', '高': 'red', '极高': 'darkred'}
    colors = [colors_map.get(l, 'gray') for l in risk_level]

    plt.figure(figsize=(12, 9))
    plt.scatter(lon, lat, c=colors, s=8, alpha=0.6, edgecolors='none')

    for label, color in colors_map.items():
        plt.scatter([], [], c=color, label=label, s=30)
    plt.legend(title='风险等级', loc='upper right', fontsize=10)

    plt.xlabel('经度', fontsize=12)
    plt.ylabel('纬度', fontsize=12)
    plt.title('综合滑坡风险分布 (易发性 × 暴露度)', fontsize=14)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    path = os.path.join(save_dir, 'risk_map.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"综合风险分布: {path}")




# 步骤 - 带地形底图的危险性分布图 (NEW)
def plot_hazard_map_basemap(hazard_level, lon, lat, save_dir, title=None, verbose=True):
    """
    带地形底图的滑坡危险性空间分布图。
    使用 contextily 加载 Stamen Terrain 在线地形图作为背景，
    叠加五级危险性散点。

    需要安装: pip install contextily
    如未安装则自动降级为纯散点图。
    """
    colors_map = {'极低': '#228B22', '低': '#7CFC00', '中': '#FFA500',
                  '高': '#FF4500', '极高': '#8B0000'}
    colors = [colors_map.get(l, 'gray') for l in hazard_level]

    fig, ax = plt.subplots(figsize=(14, 11))

    # 尝试加载地形底图
    basemap_loaded = False
    try:
        import contextily as ctx
        # 画范围框
        margin_lon = (lon.max() - lon.min()) * 0.05
        margin_lat = (lat.max() - lat.min()) * 0.05
        ax.set_xlim(lon.min() - margin_lon, lon.max() + margin_lon)
        ax.set_ylim(lat.min() - margin_lat, lat.max() + margin_lat)

        # 添加 Stamen Terrain 地形底图
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Stamen.Terrain,
                        alpha=0.85)
        basemap_loaded = True
        if verbose:
            print("  地形底图: Stamen Terrain (contextily)")
    except ImportError:
        if verbose:
            print("  contextily 未安装，使用纯色背景 (pip install contextily)")
    except Exception as e:
        if verbose:
            print(f"  地形底图加载失败: {e}，使用纯色背景")

    # 叠加散点
    scatter = ax.scatter(lon, lat, c=colors, s=6, alpha=0.7,
                         edgecolors='white', linewidths=0.3, zorder=5)

    # 图例
    for label, color in colors_map.items():
        ax.scatter([], [], c=color, label=label, s=40, edgecolors='black', linewidth=0.5)
    ax.legend(title='危险性等级', loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    # 坐标轴
    ax.set_xlabel('经度 (E)', fontsize=12)
    ax.set_ylabel('纬度 (N)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('阿坝州滑坡危险性分布图', fontsize=14, fontweight='bold')

    # 网格
    ax.grid(alpha=0.2, zorder=0)

    # 添加比例尺和指北针（纯文本标注）
    ax.text(0.02, 0.02, '▲ 北', transform=ax.transAxes,
            fontsize=11, va='bottom', ha='left', fontweight='bold')

    # 坐标轴格式化
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, 'hazard_classification_basemap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"带底图危险性分布: {path}")

    return basemap_loaded


# 步骤 - 带地形底图的综合风险分布图 (NEW)
def plot_risk_map_basemap(risk_level, lon, lat, save_dir, verbose=True):
    """带地形底图的综合风险分布图"""
    colors_map = {'极低': '#228B22', '低': '#7CFC00', '中': '#FFA500',
                  '高': '#FF4500', '极高': '#8B0000'}
    colors = [colors_map.get(l, 'gray') for l in risk_level]

    fig, ax = plt.subplots(figsize=(14, 11))

    try:
        import contextily as ctx
        margin_lon = (lon.max() - lon.min()) * 0.05
        margin_lat = (lat.max() - lat.min()) * 0.05
        ax.set_xlim(lon.min() - margin_lon, lon.max() + margin_lon)
        ax.set_ylim(lat.min() - margin_lat, lat.max() + margin_lat)
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Stamen.Terrain,
                        alpha=0.85)
    except Exception:
        pass

    ax.scatter(lon, lat, c=colors, s=6, alpha=0.7,
               edgecolors='white', linewidths=0.3, zorder=5)

    for label, color in colors_map.items():
        ax.scatter([], [], c=color, label=label, s=40, edgecolors='black', linewidth=0.5)
    ax.legend(title='风险等级', loc='lower right', fontsize=10, framealpha=0.9)

    ax.set_xlabel('经度 (E)', fontsize=12)
    ax.set_ylabel('纬度 (N)', fontsize=12)
    ax.set_title('阿坝州综合滑坡风险分布 (易发性 × 暴露度)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.text(0.02, 0.02, '▲ 北', transform=ax.transAxes, fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(save_dir, 'risk_map_basemap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"带底图风险分布: {path}")



def plot_hazard_features(lon, lat, proba, extra, save_dir, verbose=True):
    """特征与易发性概率的空间关联"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    scatter_data = [
        ('滑坡易发性概率', proba, 'Reds'),
        ('高程 (m)', None, 'viridis'),
        ('坡度 (°)', None, 'terrain'),
    ]

    for idx, (title, _, cmap) in enumerate(scatter_data):
        ax = axes[0, idx]
        sc = ax.scatter(lon, lat, c=proba if idx == 0 else proba,
                        cmap=cmap, s=6, alpha=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        plt.colorbar(sc, ax=ax, shrink=0.8)

    # 概率分箱柱状图
    ax = axes[1, 0]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    labels = ['极低', '低', '中', '高', '极高']
    counts = [((proba >= bins[i]) & (proba < bins[i+1])).sum() for i in range(len(bins)-1)]
    colors = ['green', 'lime', 'orange', 'red', 'darkred']
    bars = ax.bar(labels, counts, color=colors, alpha=0.7)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(c), ha='center', fontsize=9)
    ax.set_title('各等级样本数', fontsize=12)
    ax.set_ylabel('样本数')

    # 暴露度分布
    ax = axes[1, 1]
    exp_data = None
    if 'exposure_train' in extra:
        exp_data = extra['exposure_train']
    elif 'exposure_test' in extra:
        exp_data = extra['exposure_test']
    if exp_data is not None and 'exposure_index' in exp_data.columns:
        exp_vals = np.concatenate([
            extra['exposure_train']['exposure_index'].values,
            extra['exposure_test']['exposure_index'].values
        ]) if 'exposure_train' in extra else extra['exposure_test']['exposure_index'].values
        ax.hist(exp_vals, bins=30, color='purple', alpha=0.6, edgecolor='white')
        ax.set_title('暴露度分布', fontsize=12)
        ax.set_xlabel('综合暴露指数')
        ax.set_ylabel('样本数')

    # 统计信息
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = (
        f"易发性统计\n"
        f"均值: {proba.mean():.4f}\n"
        f"标准差: {proba.std():.4f}\n"
        f"中位数: {np.median(proba):.4f}\n"
        f"高风险(>0.6): {(proba>0.6).sum()}条\n"
        f"极高风险(>0.8): {(proba>0.8).sum()}条"
    )
    ax.text(0.1, 0.5, stats_text, fontsize=13, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('滑坡易发性综合评估', fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'hazard_analysis_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"综合分析仪表盘: {path}")


# 步骤6 - 输出统计汇总
def save_hazard_summary(proba, hazard_level, hazard_code, risk, risk_level, extra, save_dir, verbose=True):
    """保存分级统计汇总表"""
    X_train_df = extra['X_train_df']
    X_test_df = extra['X_test_df']
    X_full = pd.concat([X_train_df, X_test_df], ignore_index=True)
    exp_full = pd.concat([
        extra['exposure_train'], extra['exposure_test']
    ], ignore_index=True)

    summary = pd.DataFrame({
        '易发性概率': proba,
        '危险性等级': hazard_level,
        '危险性编码': hazard_code,
        '综合风险': risk,
        '风险等级': risk_level,
    })

    # 加入暴露特征
    for col in exp_full.columns:
        if col in exp_full.columns:
            summary[col] = exp_full[col].values

    # 按危险性等级汇总
    breakdown = summary.groupby('危险性等级', observed=True).agg(
        样本数=('易发性概率', 'count'),
        平均易发性=('易发性概率', 'mean'),
        易发性标准差=('易发性概率', 'std'),
        平均风险=('综合风险', 'mean'),
    ).round(4)

    break_path = os.path.join(save_dir, 'hazard_breakdown.csv')
    breakdown.to_csv(break_path, encoding='utf-8-sig')
    if verbose:
        print(f"\n危险性等级汇总:")
        print(breakdown.to_string())
        print(f"\n汇总保存: {break_path}")

    # 保存全量数据
    full_path = os.path.join(save_dir, 'hazard_prediction_full.csv')
    summary.to_csv(full_path, index=False, encoding='utf-8-sig')
    if verbose:
        print(f"全量预测保存: {full_path} ({len(summary)} 条)")

    return summary, breakdown


# 主函数
def main(model_path=None, method='quantile'):
    """运行完整危险性分级与风险分析"""
    print("")
    print("滑坡危险性分级与风险分析")
    print("")

    start = datetime.now()

    # 1. 加载模型
    model, model_name = load_best_model(model_path)
    print(f"模型: {model_name}")

    # 2. 获取全量预测数据
    print("")
    print("加载数据...")
    _, _, _, _, extra = preprocess_data(
        neg_ratio=1.0, verbose=False
    )

    print("")
    proba, X_full, exp_full = get_full_predictions(model, extra)

    # 获取经纬度
    lon = X_full['lon'].values if 'lon' in X_full.columns else np.zeros(len(proba))
    lat = X_full['lat'].values if 'lat' in X_full.columns else np.zeros(len(proba))

    # 3. 危险性分级
    hazard_level, hazard_code, bins = classify_hazard(proba, method=method)

    # 4. 综合风险分析
    risk, exposure = compute_risk(proba, exp_full, extra['exposure_cols'])
    risk_level = classify_risk(risk)

    # 5. 可视化
    print("")
    print("生成可视化...")
    plot_susceptibility_distribution(proba, RESULTS_DIR)
    plot_hazard_scatter(proba, hazard_level, lon, lat, RESULTS_DIR)
    plot_risk_map(risk_level, lon, lat, RESULTS_DIR)
    plot_hazard_map_basemap(hazard_level, lon, lat, RESULTS_DIR, verbose=True)
    plot_risk_map_basemap(risk_level, lon, lat, RESULTS_DIR, verbose=True)
    plot_hazard_features(lon, lat, proba, extra, RESULTS_DIR)

    # 6. 输出统计
    summary, breakdown = save_hazard_summary(
        proba, hazard_level, hazard_code, risk, risk_level, extra, RESULTS_DIR
    )

    end = datetime.now()
    print("")
    print("")
    print(f"完成! 耗时: {end - start}")
    print(f"输出目录: {RESULTS_DIR}")
    print("")

    return summary, breakdown


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='滑坡危险性分级与风险分析')
    parser.add_argument('--model-path', type=str, default=None,
                        help='指定模型路径 (默认自动选择最佳模型)')
    parser.add_argument('--method', type=str, default='quantile',
                        choices=['quantile', 'equal'],
                        help='分级方法: quantile(五分位数) / equal(等距)')
    args = parser.parse_args()

    main(model_path=args.model_path, method=args.method)
