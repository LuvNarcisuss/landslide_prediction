import os
import sys
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
from pre_process import preprocess_data

# 添加根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def imp_xgboost_model(verbose = False):

    # 加载数据
    X_train, X_test, y_train, y_test = preprocess_data()

    # 创建并训练模型（Optuna 贝叶斯调参）
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        }
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model = XGBClassifier(**params, random_state=42, verbosity=0, scale_pos_weight=2.41)
            model.fit(X_train[train_idx], y_train[train_idx])
            y_pred = model.predict(X_train[val_idx])
            scores.append(f1_score(y_train[val_idx], y_pred))
        return np.mean(scores)

    # 控制 Optuna 日志输出
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=15, timeout=180, show_progress_bar=False)

    best_params = study.best_params
    model = XGBClassifier(**best_params, random_state=42, verbosity=0, scale_pos_weight=2.41)
    model.fit(X_train, y_train)
    # 保存模型
    joblib.dump(model, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'imp_xgboost.pkl'))

    # 预测
    y_pred = model.predict(X_test)

    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 输出结果
    print("\nXGBoost(Optuna贝叶斯调参)模型训练完成")
    print(f"最佳参数: {best_params}")
    print(f"最佳CV-F1: {study.best_value:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    print(f"特异性: {specificity:.4f}")

    return {
        '模型名称': 'imp_XGBoost',
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1值': f1,
        '特异性': specificity,
    }, model

if __name__ == "__main__":
    imp_xgboost_model(verbose =  True)
