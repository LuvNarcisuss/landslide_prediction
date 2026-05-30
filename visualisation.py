import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                              accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import warnings

# 设置中文字体
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def _get_probability(model, X):
    """获取模型预测概率，兼容 predict_proba 和 decision_function"""
    try:
        return model.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            y_scores = model.decision_function(X)
            return (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
        except (AttributeError, NotImplementedError):
            return None


def plot_roc_curves(models_info, X_test, y_test, save_dir=None):
    """
    绘制多个模型的 ROC 曲线对比图

    参数
    ----
    models_info : list of (name, model)
    X_test, y_test : 测试集
    save_dir : 图片保存目录
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'roc_curves.png')

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='随机分类器 (AUC=0.5)')

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_info)))
    valid_models = [(name, model) for name, model in models_info if _get_probability(model, X_test) is not None]

    for (name, model), color in zip(valid_models, colors[:len(valid_models)]):
        y_prob = _get_probability(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=1.5, label=f'{name} (AUC={roc_auc:.3f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    plt.title('ROC 曲线对比', fontsize=14)
    plt.legend(loc='lower right', fontsize=9, framealpha=0.8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"ROC 曲线已保存到: {save_path}")


def plot_pr_curves(models_info, X_test, y_test, save_dir=None):
    """
    绘制多个模型的 Precision-Recall 曲线对比图

    参数
    ----
    models_info : list of (name, model)
    X_test, y_test : 测试集
    save_dir : 图片保存目录
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pr_curves.png')

    plt.figure(figsize=(10, 8))

    # 基线：正样本比例
    pos_ratio = y_test.mean()
    plt.plot([0, 1], [pos_ratio, pos_ratio], 'k--', lw=1, alpha=0.5,
             label=f'随机分类器 (AP={pos_ratio:.3f})')

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_info)))
    valid_models = [(name, model) for name, model in models_info if _get_probability(model, X_test) is not None]

    for (name, model), color in zip(valid_models, colors[:len(valid_models)]):
        y_prob = _get_probability(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, color=color, lw=1.5, label=f'{name} (AP={ap:.3f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('PR 曲线对比', fontsize=14)
    plt.legend(loc='lower left', fontsize=9, framealpha=0.8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"PR 曲线已保存到: {save_path}")


def _prepare_prob_and_thresholds(models_info, X_test, y_test, n_thresholds=200):
    """辅助函数：获取各模型概率并生成公共阈值序列"""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    results = []
    for name, model in models_info:
        y_prob = _get_probability(model, X_test)
        if y_prob is None:
            continue
        results.append((name, y_prob))
    return results, thresholds


def plot_ks_curves(models_info, X_test, y_test, save_dir=None):
    """
    绘制多个模型的 KS 曲线对比图

    KS统计量 = max(TPR - FPR)，衡量模型区分正负样本的能力。
    KS > 0.4 表示区分能力较好。
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ks_curves.png')

    probs, thresholds = _prepare_prob_and_thresholds(models_info, X_test, y_test)
    if not probs:
        return

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(probs)))

    for (name, y_prob), color in zip(probs, colors):
        tpr_list, fpr_list = [], []
        for t in thresholds:
            yp = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
            tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        ks = max(np.array(tpr_list) - np.array(fpr_list))
        ks_idx = np.argmax(np.array(tpr_list) - np.array(fpr_list))

        plt.plot(thresholds, tpr_list, color=color, lw=1.5, linestyle='-')
        plt.plot(thresholds, fpr_list, color=color, lw=1.5, linestyle='--')
        plt.plot(thresholds[ks_idx], tpr_list[ks_idx], 'o', color=color, markersize=6)
        plt.annotate(f'{name} KS={ks:.3f}',
                     xy=(thresholds[ks_idx], tpr_list[ks_idx]),
                     xytext=(thresholds[ks_idx] + 0.05, tpr_list[ks_idx] - 0.05),
                     fontsize=8, color=color)

    plt.xlabel('阈值 (Threshold)', fontsize=12)
    plt.ylabel('比率 (Rate)', fontsize=12)
    plt.title('KS 曲线对比', fontsize=14)
    plt.legend(['TPR', 'FPR'] + [n for n, _ in probs], loc='center right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"KS 曲线已保存到: {save_path}")


def plot_gain_curves(models_info, X_test, y_test, save_dir=None):
    """
    绘制多个模型的累积增益曲线

    展示按预测概率排序后，前 x% 的样本能捕获多少比例的正样本。
    曲线越高，模型在顶部排序越有效。
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'gain_curves.png')

    probs, _ = _prepare_prob_and_thresholds(models_info, X_test, y_test)
    if not probs:
        return

    n_total = len(y_test)
    n_pos = y_test.sum()

    plt.figure(figsize=(10, 8))

    # 随机分类器基线
    plt.plot([0, 100], [0, 100 * n_pos / n_total], 'k--', lw=1, alpha=0.5,
             label=f'随机分类器')

    # 完美分类器
    plt.plot([0, n_pos / n_total * 100, 100], [0, 100, 100], 'k:', lw=1, alpha=0.5,
             label='完美分类器')

    colors = plt.cm.tab10(np.linspace(0, 1, len(probs)))

    for (name, y_prob), color in zip(probs, colors):
        # 按概率降序排序
        sorted_idx = np.argsort(y_prob)[::-1]
        sorted_y = y_test[sorted_idx]
        cum_pos = np.cumsum(sorted_y)
        pct_sample = np.arange(1, n_total + 1) / n_total * 100
        pct_pos = cum_pos / n_pos * 100

        plt.plot(pct_sample, pct_pos, color=color, lw=1.5, label=name)

    plt.xlim([0, 100])
    plt.ylim([0, 105])
    plt.xlabel('样本百分比 (%)', fontsize=12)
    plt.ylabel('正样本捕获率 (%)', fontsize=12)
    plt.title('累积增益曲线对比', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"累积增益曲线已保存到: {save_path}")


def plot_threshold_analysis(models_info, X_test, y_test, save_dir=None,
                            metrics=['F1值', '准确率', '召回率', '精确率', '特异性']):
    """
    绘制阈值-指标灵敏度分析曲线

    展示不同阈值下各指标的变化趋势，帮助选择最优决策阈值。
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'threshold_analysis.png')

    # 任选一个模型（选第一个有效的）
    for name, model in models_info:
        y_prob = _get_probability(model, X_test)
        if y_prob is not None:
            best_model_name = name
            best_y_prob = y_prob
            break
    else:
        return

    thresholds = np.linspace(0.0, 1.0, 200)

    fig, ax = plt.subplots(figsize=(10, 7))

    all_metrics = {
        'F1值': [], '准确率': [], '召回率': [], '精确率': [], '特异性': []
    }

    for t in thresholds:
        yp = (best_y_prob >= t).astype(int)
        all_metrics['准确率'].append(accuracy_score(y_test, yp))
        all_metrics['精确率'].append(precision_score(y_test, yp, zero_division=0))
        all_metrics['召回率'].append(recall_score(y_test, yp, zero_division=0))
        all_metrics['F1值'].append(f1_score(y_test, yp, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
        all_metrics['特异性'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    colors_metric = {'F1值': '#2ca02c', '准确率': '#1f77b4', '召回率': '#ff7f0e',
                     '精确率': '#d62728', '特异性': '#9467bd'}
    for m in metrics:
        ax.plot(thresholds, all_metrics[m], color=colors_metric.get(m, '#333'),
                lw=2, label=m)

    # 标记当前默认阈值0.5
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
    ax.annotate('默认阈值=0.5', xy=(0.5, 0.5), xytext=(0.52, 0.45),
                fontsize=10, color='gray')

    ax.set_xlabel('阈值 (Threshold)', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title(f'阈值-指标灵敏度分析 ({best_model_name})', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"阈值分析曲线已保存到: {save_path}")
