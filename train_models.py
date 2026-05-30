import os
import sys
import io
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from pre_process import *
from ML_algorithms import *
from improve_algorithms import *
from visualisation import *
from train_logs import train_log

warnings.filterwarnings('ignore')
root_dir = os.path.dirname(os.path.abspath(__file__))

model_names = [
    'logistic_regression',
    'decision_tree',
    'svm',
    'random_forest',
    'knn',
    'gboost',
    # 'catboost',
    # 'lightgbm',
    # 'xgboost',

    'imp_catboost',
    'imp_lightgbm',
    'imp_xgboost',
]

model_funcs = [
    # 基础模型
    logistic_regression_model,  # 逻辑回归
    decision_tree_model,        # 决策树
    svm_model,                  # SVM
    random_forest_model,        # 随机森林
    knn_model,                  # KNN
    gboost_model,               # GBDT
    catboost_model,             # CatBoost
    lightgbm_model,             # LightGBM
    xgboost_model,              # XGBoost

    # 改进模型
    imp_catboost_model,         # CatBoost(Optuna贝叶斯调参)
    imp_lightgbm_model,         # LightGBM(Optuna贝叶斯调参)
    imp_xgboost_model,          # XGBoost(Optuna贝叶斯调参)
]

# 阈值搜索与完整评估
def find_best_threshold(model, X_val, y_val, n_thresholds=100, alpha=0.6):
    """
    多目标阈值搜索：score = alpha * F1 + (1-alpha) * specificity
    alpha=0.6: 偏重F1，兼顾特异性；alpha=0.5: 两者等权
    """
    try:
        y_prob = model.predict_proba(X_val)[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            y_scores = model.decision_function(X_val)
            y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
        except (AttributeError, NotImplementedError):
            return 0.5, 0

    best_score, best_th = 0, 0.5
    best_f1_at_th = 0
    for t in np.linspace(0.05, 0.95, n_thresholds):
        yp = (y_prob >= t).astype(int)
        f1_t = f1_score(y_val, yp, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, yp, labels=[0, 1]).ravel()
        spec_t = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        score = alpha * f1_t + (1 - alpha) * spec_t
        if score > best_score:
            best_score, best_th = score, t
            best_f1_at_th = f1_t
    return best_th, best_f1_at_th

# 完整评估
def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            y_scores = model.decision_function(X_test)
            y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
        except:
            y_pred = model.predict(X_test)
            y_prob = y_pred.astype(float)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
    except:
        roc_auc = pr_auc = np.nan

    return {
        '准确率': (tp + tn) / (tp + tn + fp + fn),
        '精确率': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        '召回率': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'F1值': f1_score(y_test, y_pred, zero_division=0),
        '特异性': specificity,
        'AUC-ROC': roc_auc,
        'AUC-PR': pr_auc,
        '最优阈值': threshold,
    }

def main():
    # 训练所有模型并收集指标与模型对象
    print("训练所有模型并评估性能")
    # 开始训练时间
    start_time = datetime.now()
    print(f"\n开始训练时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results_list = []
    models_list = []

    for func in model_funcs:
        metrics, model = func()
        results_list.append(metrics)
        models_list.append(model)

    # 训练集成模型（传入预训练模型）
    v_metrics, v_model = voting_model(model_names)
    s_metrics, s_model = stacking_model(model_names)
    b_metrics, b_model = blending_model(model_names)
    e_metrics, e_model = ensemble_average_model(model_names)

    results_list.extend([v_metrics, s_metrics, b_metrics, e_metrics])
    models_list.extend([v_model, s_model, b_model, e_model])

    # 加载数据，拆分验证集用于阈值搜索
    X_train, X_test, y_train, y_test = preprocess_data()
    _, X_val, _, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 收集训练数据信息
    dataset_info = {}
    try:
        X_t, X_te, y_t, y_te = preprocess_data()
        dataset_info["特征数"] = str(X_t.shape[1])
        dataset_info["总样本"] = str(len(y_t) + len(y_te))
        n0 = int((y_t == 0).sum() + (y_te == 0).sum())
        n1 = int((y_t == 1).sum() + (y_te == 1).sum())
        dataset_info["标签0(降雨诱发滑坡)"] = str(n0)
        dataset_info["标签1(其他/非滑坡)"] = str(n1)
        dataset_info["训练集形状"] = str(X_t.shape)
        dataset_info["测试集形状"] = str(X_te.shape)
    except:
        pass

    print("\n阈值优化结果：")
    calibrated_models = []
    final_results = []
    for metrics, model in zip(results_list, models_list):
        name = metrics['模型名称']

        # Voting 模型自带阈值调优结果，直接使用
        if model is None:
            result = {'模型名称': name}
            result.update({k: metrics.get(k, 0) for k in
                           ['准确率','精确率','召回率','F1值','特异性','AUC-ROC','AUC-PR','最优阈值']})
            result['默认阈值F1'] = metrics.get('F1值', 0)
            result['默认阈值特异性'] = metrics.get('特异性', 0)
            final_results.append(result)
            print(f"\n  {name}:")
            print(f"    最优阈值: {metrics.get('最优阈值', 0.5):.3f}")
            print(f"    F1={metrics.get('F1值', 0):.4f}, 特异性={metrics.get('特异性', 0):.2%}")
            continue

        # 概率校准：用 CalibratedClassifierCV 校准模型概率输出
        try:
            base_cls = model.__class__
            base_params = model.get_params()
            calib_model = CalibratedClassifierCV(base_cls(**base_params), cv=3, method='sigmoid')
            calib_model.fit(X_train, y_train)
            calib_model_for_val = calib_model
        except Exception:
            calib_model = model
            calib_model_for_val = model

        calibrated_models.append((name, calib_model))

        best_th, val_f1 = find_best_threshold(calib_model_for_val, X_val, y_val)
        default_metrics = evaluate_with_threshold(calib_model, X_test, y_test, threshold=0.5)
        opt_metrics = evaluate_with_threshold(calib_model, X_test, y_test, threshold=best_th)

        result = {'模型名称': name}
        result.update(opt_metrics)
        result['默认阈值F1'] = default_metrics['F1值']
        result['默认阈值特异性'] = default_metrics['特异性']
        final_results.append(result)

        print(f"\n  {name}:")
        print(f"    最优阈值: {best_th:.3f} (验证集F1={val_f1:.4f})")
        print(f"    默认阈值 → F1={default_metrics['F1值']:.4f}, 特异性={default_metrics['特异性']:.2%}")
        print(f"    最优阈值 → F1={opt_metrics['F1值']:.4f}, 特异性={opt_metrics['特异性']:.2%}")
        print(f"    AUC-ROC={opt_metrics['AUC-ROC']:.4f}, AUC-PR={opt_metrics['AUC-PR']:.4f}")

    # 训练结束时间
    end_time = datetime.now()
    print(f"\n训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练总时长: {end_time - start_time}")

    # 排序与保存
    results_df = pd.DataFrame(final_results).sort_values('F1值', ascending=False).reset_index(drop=True)
    results_df['排名'] = range(1, len(results_df) + 1)
    cols = ['排名', '模型名称'] + [c for c in results_df.columns if c not in ['排名', '模型名称']]
    results_df = results_df[cols]

    results_file = os.path.join(root_dir, 'results','model_performance.csv')
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')

    print("\n最终排名（按 F1 值排序）")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n最佳模型: {best['模型名称']} (F1={best['F1值']:.4f}, AUC-ROC={best['AUC-ROC']:.4f}, AUC-PR={best['AUC-PR']:.4f}, 阈值={best['最优阈值']:.3f})")

    print(f"\n模型结果已保存到: {results_file}")

    # 生成评估曲线
    plot_roc_curves(calibrated_models, X_test, y_test)
    plot_pr_curves(calibrated_models, X_test, y_test)
    plot_ks_curves(calibrated_models, X_test, y_test)
    plot_gain_curves(calibrated_models, X_test, y_test)
    plot_threshold_analysis(calibrated_models, X_test, y_test)

    # 生成本次训练的日志
    log_name = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_log(log_name, dataset_info, start_time, end_time, model_names, results_df)

if __name__ == "__main__":
    main()
