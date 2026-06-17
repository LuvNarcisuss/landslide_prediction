"""
模型训练 主脚本
使用 Pipeline（数据清洗 + NDVI插补 + 负样本生成 + 特征工程）的完整建模流程。

训练全部9个基模型 + 4个集成模型 + Optuna超参优化 + SHAP分析。

用法：
    python train_models.py                          # 默认训练
    python train_models.py --optuna-trials 30       # 开启Optuna调参
    python train_models.py --neg-ratio 0.5          # 自定义负样本比例
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from pre_process.pipeline import preprocess_data
from visualisation import plot_roc_curves, plot_pr_curves
from train_logs import generate_train_log

# 创建结果目录
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# 步骤1 - 模型定义
def get_all_estimators(use_gpu=False):
    """返回所有基模型 (name, model) 列表"""
    estimators = []
    _gpu = use_gpu

    estimators.append(('logistic_regression', LogisticRegression(
        random_state=42, class_weight='balanced', C=1.0, max_iter=1000, solver='saga'
    )))

    estimators.append(('decision_tree', DecisionTreeClassifier(
        max_depth=8, random_state=42, class_weight='balanced'
    )))

    estimators.append(('svm', SVC(
        kernel='rbf', probability=True, random_state=42,
        class_weight='balanced', gamma='scale', C=1.0
    )))

    estimators.append(('random_forest', RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced_subsample', n_jobs=-1
    )))

    estimators.append(('knn', KNeighborsClassifier(
        n_neighbors=7, weights='distance', n_jobs=-1
    )))

    estimators.append(('gboost', GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )))

    _cat_params = {'random_state': 42, 'verbose': False, 'allow_writing_files': False,
                 'auto_class_weights': 'Balanced'}
    if _gpu:
        _cat_params['task_type'] = 'GPU'
    estimators.append(('catboost', CatBoostClassifier(**_cat_params)))

    _lgb_params = {'random_state': 42, 'verbose': -1, 'class_weight': 'balanced'}
    if _gpu:
        _lgb_params['device'] = 'gpu'
    estimators.append(('lightgbm', LGBMClassifier(**_lgb_params)))

    _xgb_params = {'random_state': 42, 'verbosity': 0, 'scale_pos_weight': (5322 / 2661)}
    if _gpu:
        _xgb_params['tree_method'] = 'gpu_hist'
        _xgb_params['predictor'] = 'gpu_predictor'
    estimators.append(('xgboost', XGBClassifier(**_xgb_params)))

    return estimators


# 步骤2 - 获取预测概率（兼容各种模型）
def _get_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            s = model.decision_function(X)
            return (s - s.min()) / (s.max() - s.min() + 1e-10)
        except (AttributeError, NotImplementedError):
            return model.predict(X).astype(float)


# 步骤3 - 训练单个模型
def train_model(name, model_cls, X_train, y_train, X_test, y_test, verbose=True):
    """训练并评估单个模型"""
    if verbose:
        print(f"  训练 {name}...", end=" ")

    try:
        model = model_cls.fit(X_train, y_train)
    except Exception as e:
        if verbose:
            print(f"失败: {e}")
        return None, None

    # 预测
    y_prob = _get_proba(model, X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    # 评估指标
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
    except Exception:
        roc_auc = pr_auc = np.nan

    metrics = {
        '模型名称': name,
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred, zero_division=0),
        '召回率': recall_score(y_test, y_pred, zero_division=0),
        'F1值': f1_score(y_test, y_pred, zero_division=0),
        '特异性': specificity,
        'AUC-ROC': roc_auc,
        'AUC-PR': pr_auc,
    }

    if verbose:
        print(f"F1={metrics['F1值']:.4f}, AUC={metrics['AUC-ROC']:.4f}")

    # 保存模型
    model_path = os.path.join(MODELS_DIR, f'{name}.pkl')
    joblib.dump(model, model_path)

    return metrics, model


# 步骤4 - 训练集成模型
def train_ensemble_models(estimators, models_dict, X_train, y_train, X_test, y_test, verbose=True):
    """训练4种集成模型"""
    ensemble_results = []

    # 准备基模型预测概率（用于集成）
    train_probas = np.column_stack([
        _get_proba(models_dict[name], X_train) for name, _ in estimators
        if name in models_dict and models_dict[name] is not None
    ])
    test_probas = np.column_stack([
        _get_proba(models_dict[name], X_test) for name, _ in estimators
        if name in models_dict and models_dict[name] is not None
    ])
    valid_names = [name for name, _ in estimators
                   if name in models_dict and models_dict[name] is not None]

    if verbose:
        print(f"\n集成模型 (基模型: {len(valid_names)} 个):")

    # 4a: 平均值集成
    if verbose:
        print("  训练 ensemble_avg...", end=" ")

    avg_proba = np.mean(test_probas, axis=1)
    avg_pred = (avg_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, avg_pred, labels=[0, 1]).ravel()
    avg_metrics = {
        '模型名称': 'ensemble_avg',
        '准确率': accuracy_score(y_test, avg_pred),
        '精确率': precision_score(y_test, avg_pred, zero_division=0),
        '召回率': recall_score(y_test, avg_pred, zero_division=0),
        'F1值': f1_score(y_test, avg_pred, zero_division=0),
        '特异性': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'AUC-ROC': roc_auc_score(y_test, avg_proba) if len(np.unique(y_test)) > 1 else np.nan,
        'AUC-PR': average_precision_score(y_test, avg_proba),
    }
    if verbose:
        print(f"F1={avg_metrics['F1值']:.4f}")
    ensemble_results.append(avg_metrics)

    # 4b: 投票集成（软投票）
    if verbose:
        print("  训练 voting...", end=" ")

    from sklearn.ensemble import VotingClassifier
    voting = VotingClassifier(
        estimators=[(n, models_dict[n]) for n in valid_names],
        voting='soft'
    )
    try:
        voting.fit(X_train, y_train)
        v_prob = _get_proba(voting, X_test)
        v_pred = (v_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, v_pred, labels=[0, 1]).ravel()
        v_metrics = {
            '模型名称': 'voting',
            '准确率': accuracy_score(y_test, v_pred),
            '精确率': precision_score(y_test, v_pred, zero_division=0),
            '召回率': recall_score(y_test, v_pred, zero_division=0),
            'F1值': f1_score(y_test, v_pred, zero_division=0),
            '特异性': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'AUC-ROC': roc_auc_score(y_test, v_prob) if len(np.unique(y_test)) > 1 else np.nan,
            'AUC-PR': average_precision_score(y_test, v_prob),
        }
        if verbose:
            print(f"F1={v_metrics['F1值']:.4f}")
        ensemble_results.append(v_metrics)
        joblib.dump(voting, os.path.join(MODELS_DIR, 'voting.pkl'))
    except Exception as e:
        if verbose:
            print(f"失败: {e}")

    # 4c: Stacking
    if verbose:
        print("  训练 stacking...", end=" ")

    from sklearn.ensemble import StackingClassifier
    stacking = StackingClassifier(
        estimators=[(n, models_dict[n]) for n in valid_names],
        final_estimator=LogisticRegression(random_state=42, C=1.0, max_iter=1000),
        cv=5, stack_method='predict_proba', n_jobs=-1
    )
    try:
        stacking.fit(X_train, y_train)
        s_prob = _get_proba(stacking, X_test)
        s_pred = (s_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, s_pred, labels=[0, 1]).ravel()
        s_metrics = {
            '模型名称': 'stacking',
            '准确率': accuracy_score(y_test, s_pred),
            '精确率': precision_score(y_test, s_pred, zero_division=0),
            '召回率': recall_score(y_test, s_pred, zero_division=0),
            'F1值': f1_score(y_test, s_pred, zero_division=0),
            '特异性': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'AUC-ROC': roc_auc_score(y_test, s_prob) if len(np.unique(y_test)) > 1 else np.nan,
            'AUC-PR': average_precision_score(y_test, s_prob),
        }
        if verbose:
            print(f"F1={s_metrics['F1值']:.4f}")
        ensemble_results.append(s_metrics)
        joblib.dump(stacking, os.path.join(MODELS_DIR, 'stacking.pkl'))
    except Exception as e:
        if verbose:
            print(f"失败: {e}")

    # 4d: Blending
    if verbose:
        print("  训练 blending...", end=" ")

    try:
        # 用验证集训练元模型
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # 基模型在验证集上的预测
        val_probas = np.column_stack([
            _get_proba(models_dict[n], X_val) for n in valid_names
        ])

        # 元模型：逻辑回归
        meta_model = LogisticRegression(random_state=42, C=1.0, max_iter=1000)
        meta_model.fit(val_probas, y_val)

        # 测试集预测
        blend_proba = meta_model.predict_proba(test_probas)[:, 1]
        blend_pred = (blend_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, blend_pred, labels=[0, 1]).ravel()
        b_metrics = {
            '模型名称': 'blending',
            '准确率': accuracy_score(y_test, blend_pred),
            '精确率': precision_score(y_test, blend_pred, zero_division=0),
            '召回率': recall_score(y_test, blend_pred, zero_division=0),
            'F1值': f1_score(y_test, blend_pred, zero_division=0),
            '特异性': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'AUC-ROC': roc_auc_score(y_test, blend_proba) if len(np.unique(y_test)) > 1 else np.nan,
            'AUC-PR': average_precision_score(y_test, blend_proba),
        }
        if verbose:
            print(f"F1={b_metrics['F1值']:.4f}")
        ensemble_results.append(b_metrics)
        joblib.dump(meta_model, os.path.join(MODELS_DIR, 'blending.pkl'))
    except Exception as e:
        if verbose:
            print(f"失败: {e}")

    return ensemble_results


# 步骤5 - 阈值优化
def find_best_threshold(model, X_val, y_val, n_thresholds=100, alpha=0.6):
    """搜索最优分类阈值"""
    y_prob = _get_proba(model, X_val)

    best_score, best_th = 0, 0.5
    for t in np.linspace(0.05, 0.95, n_thresholds):
        yp = (y_prob >= t).astype(int)
        f1_t = f1_score(y_val, yp, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, yp, labels=[0, 1]).ravel()
        spec_t = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        score = alpha * f1_t + (1 - alpha) * spec_t
        if score > best_score:
            best_score, best_th = score, t
    return best_th


# 步骤6 - Optuna 超参优化
def _objective_random_forest(trial, X, y, cv=3):
    """随机森林 Optuna 目标函数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'class_weight': 'balanced_subsample',
        'random_state': 42, 'n_jobs': -1,
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()


def _objective_xgboost(trial, X, y, cv=3):
    """XGBoost Optuna 目标函数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'random_state': 42, 'verbosity': 0,
    }
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    params['scale_pos_weight'] = n_neg / max(n_pos, 1)
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()


def _objective_lightgbm(trial, X, y, cv=3):
    """LightGBM Optuna 目标函数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'class_weight': 'balanced',
        'random_state': 42, 'verbose': -1,
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()


def _objective_catboost(trial, X, y, cv=3):
    """CatBoost Optuna 目标函数"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500, step=50),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_state': 42, 'verbose': False, 'allow_writing_files': False,
        'auto_class_weights': 'Balanced',
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()


def optimize_with_optuna(X_train, y_train, n_trials=30, cv=3, verbose=True, use_gpu=False):
    """
    用 Optuna 对4个主要模型进行超参优化

    返回:
        tuned_models: dict {name: (best_params, best_model)}
    """
    try:
        import optuna
    except ImportError:
        if verbose:
            print("  Optuna 未安装，跳过调参 (pip install optuna)")
        return {}

    configs = {
        'random_forest': {'objective': _objective_random_forest, 'model_class': RandomForestClassifier},
        'xgboost': {'objective': _objective_xgboost, 'model_class': XGBClassifier},
        'lightgbm': {'objective': _objective_lightgbm, 'model_class': LGBMClassifier},
        'catboost': {'objective': _objective_catboost, 'model_class': CatBoostClassifier},
    }
    tuned_models = {}

    for name, cfg in configs.items():
        if verbose:
            print(f"  Optuna 调参 {name} ({n_trials} 次)...", end=" ")

        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                study_name=f'opt_{name}'
            )
            study.optimize(
                lambda trial: cfg['objective'](trial, X_train, y_train, cv=cv),
                n_trials=n_trials, show_progress_bar=False
            )
            best_params = study.best_params
            best_score = study.best_value

            if name == 'xgboost':
                n_pos = y_train.sum()
                n_neg = len(y_train) - n_pos
                best_params['scale_pos_weight'] = n_neg / max(n_pos, 1)
                best_params['random_state'] = 42; best_params['verbosity'] = 0
                if use_gpu:
                    best_params['tree_method'] = 'gpu_hist'
                    best_params['predictor'] = 'gpu_predictor'
            elif name == 'lightgbm':
                best_params['class_weight'] = 'balanced'
                best_params['random_state'] = 42; best_params['verbose'] = -1
                if use_gpu:
                    best_params['device'] = 'gpu'
            elif name == 'catboost':
                best_params['random_state'] = 42; best_params['verbose'] = False
                best_params['allow_writing_files'] = False
                best_params['auto_class_weights'] = 'Balanced'
                if use_gpu:
                    best_params['task_type'] = 'GPU'
            elif name == 'random_forest':
                best_params['class_weight'] = 'balanced_subsample'
                best_params['random_state'] = 42; best_params['n_jobs'] = -1

            model = cfg['model_class'](**best_params)
            if verbose:
                print(f"最佳F1={best_score:.4f}")

            tuned_models[name] = (best_params, model)

        except Exception as e:
            if verbose:
                print(f"失败: {e}")
            tuned_models[name] = (None, None)

    return tuned_models


def train_tuned_models(tuned_models, X_train, y_train, X_test, y_test, verbose=True):
    """训练Optuna优化的模型并评估"""
    tuned_results = []

    for name, (params, model) in tuned_models.items():
        if model is None:
            continue
        tuned_name = f'{name}_tuned'
        if verbose:
            print(f"  训练 {tuned_name}...", end=" ")
        try:
            model.fit(X_train, y_train)
            y_prob = _get_proba(model, X_test)
            y_pred = (y_prob >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            metrics = {
                '模型名称': tuned_name,
                '准确率': accuracy_score(y_test, y_pred),
                '精确率': precision_score(y_test, y_pred, zero_division=0),
                '召回率': recall_score(y_test, y_pred, zero_division=0),
                'F1值': f1_score(y_test, y_pred, zero_division=0),
                '特异性': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'AUC-ROC': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
                'AUC-PR': average_precision_score(y_test, y_prob),
            }
            if verbose:
                print(f"F1={metrics['F1值']:.4f}, AUC={metrics['AUC-ROC']:.4f}")
            joblib.dump(model, os.path.join(MODELS_DIR, f'{tuned_name}.pkl'))
            tuned_results.append(metrics)
        except Exception as e:
            if verbose:
                print(f"  失败: {e}")

    return tuned_results


# 步骤7 - 阈值优化（调优后评估）
def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """用给定阈值评估模型"""
    y_prob = _get_proba(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred, zero_division=0),
        '召回率': recall_score(y_test, y_pred, zero_division=0),
        'F1值': f1_score(y_test, y_pred, zero_division=0),
        '特异性': specificity,
        'AUC-ROC': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
        'AUC-PR': average_precision_score(y_test, y_prob),
        '最优阈值': threshold,
    }


# 步骤6 - SHAP 分析
def shap_analysis(best_model, best_name, X_train, X_test, feature_names, save_dir,
                  all_models=None):
    """对最佳模型进行SHAP解释"""
    print("")
    print("SHAP 模型解释:")

    try:
        import shap
        # 取200个样本加速
        X_sample = X_test[:200] if len(X_test) > 200 else X_test

        # 集成模型（stacking/voting/blending）降级到基树模型做SHAP
        # TreeExplainer 比 KernelExplainer 快1000倍以上
        ensemble_names = ('stacking', 'voting', 'blending', 'ensemble_avg')
        if best_name in ensemble_names and all_models is not None:
            for tree_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
                if tree_name in all_models and all_models[tree_name] is not None:
                    best_model = all_models[tree_name]
                    best_name = tree_name + '_for_shap'
                    print(f"  集成模型{best_name.split('_for_shap')[0]}降级为{tree_name}做SHAP分析")
                    break

        # 根据模型类型选择SHAP解释器
        if 'xgboost' in best_name or 'lightgbm' in best_name or 'catboost' in best_name:
            explainer = shap.TreeExplainer(best_model)
        elif 'random_forest' in best_name or 'gboost' in best_name:
            explainer = shap.TreeExplainer(best_model)
        else:
            # 对非树模型使用KernelExplainer
            # 大幅降低样本量加速：background=50, samples=50
            print("  使用 KernelExplainer（非树模型），采样加速...")
            X_bg = X_train[:50] if len(X_train) > 50 else X_train
            X_sample = X_test[:50] if len(X_test) > 50 else X_test
            explainer = shap.KernelExplainer(
                best_model.predict_proba, X_bg
            )

        shap_values = explainer.shap_values(X_sample)

        # 处理二分类输出
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        # SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            show=False, max_display=20
        )
        plt.title(f"SHAP 特征重要性 - {best_name}", fontsize=14)
        plt.tight_layout()
        shap_summary_path = os.path.join(save_dir, 'shap_summary.png')
        plt.savefig(shap_summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  SHAP Summary 已保存: {shap_summary_path}")

        # SHAP Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            plot_type='bar',
            show=False, max_display=20
        )
        plt.title(f"SHAP 特征重要性（柱状图） - {best_name}", fontsize=14)
        plt.tight_layout()
        shap_bar_path = os.path.join(save_dir, 'shap_bar.png')
        plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  SHAP Bar 已保存: {shap_bar_path}")

        # 计算平均绝对SHAP值
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            '特征': feature_names,
            'SHAP值': mean_shap
        }).sort_values('SHAP值', ascending=False)
        shap_importance.to_csv(
            os.path.join(save_dir, 'shap_importance.csv'),
            index=False, encoding='utf-8-sig'
        )
        print(f"  SHAP 重要性已保存: shap_importance.csv")
        print(f"  前5重要特征: {shap_importance['特征'].head(5).tolist()}")

    except ImportError:
        print("  SHAP 未安装，跳过 (pip install shap)")
    except Exception as e:
        print(f"  SHAP 分析失败: {e}")


# 步骤7 - 保存评估结果
def save_results(results_df, save_dir):
    """保存评估结果CSV"""
    results_df = results_df.sort_values('F1值', ascending=False).reset_index(drop=True)
    results_df['排名'] = range(1, len(results_df) + 1)
    cols = ['排名', '模型名称'] + [c for c in results_df.columns if c not in ['排名', '模型名称']]
    results_df = results_df[cols]

    results_path = os.path.join(save_dir, 'model_performance.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    return results_df


# 主函数
def main(neg_ratio=1.0, run_shap=True, optuna_trials=0, buffer_dist=0.8,
          quality_check=True, hybrid_ratio=0.3, use_gpu=False):
    """运行完整训练流程"""
    start_time = datetime.now()
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"负样本比例: 1:{neg_ratio}")
    print(f"混合负样本: 真实数据{hybrid_ratio*100:.0f}% + 空间采样{(1-hybrid_ratio)*100:.0f}%")
    print(f"GPU加速: {'开启(XGB/LGB/CAT)' if use_gpu else '关闭'}")
    print(f"缓冲区距离: {buffer_dist}km")
    print(f"质量检验: {'开启' if quality_check else '跳过'}")
    print(f"Optuna调参: {'开启(' + str(optuna_trials) + '次)' if optuna_trials > 0 else '跳过'}")
    print(f"模型保存目录: {MODELS_DIR}")
    print(f"结果保存目录: {RESULTS_DIR}")
    print("")

    # 加载数据
    print("加载数据 (Pipeline)")

    X_train, X_test, y_train, y_test, extra = preprocess_data(
        neg_ratio=neg_ratio, verbose=True,
        min_dist_km=buffer_dist, run_quality_check=quality_check,
        hybrid_ratio=hybrid_ratio
    )
    feature_names = extra['feature_names']

    # 划分验证集用于阈值优化
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"验证集: {X_val.shape[0]} 条")

    # 训练基模型
    print("")
    print("训练基模型")

    estimators = get_all_estimators(use_gpu=use_gpu)
    all_metrics = []
    all_models = {}
    calibrated_models = []

    for name, model_cls in estimators:
        metrics, model = train_model(
            name, model_cls,
            X_train, y_train, X_test, y_test,
            verbose=True
        )
        if metrics is not None:
            all_metrics.append(metrics)
            all_models[name] = model
            calibrated_models.append((name, model))

    # 训练集成模型
    print("")
    print("训练集成模型")

    ensemble_metrics = train_ensemble_models(
        estimators, all_models,
        X_train, y_train, X_test, y_test,
        verbose=True
    )
    all_metrics.extend(ensemble_metrics)

    # Optuna 超参优化（可选）
    if optuna_trials > 0:
        print("")
        print(f"Optuna 超参优化 ({optuna_trials} 次/模型)")

        tuned_models = optimize_with_optuna(
            X_train, y_train, n_trials=optuna_trials, verbose=True,
            use_gpu=use_gpu
        )
        tuned_results = train_tuned_models(
            tuned_models, X_train, y_train, X_test, y_test, verbose=True
        )
        all_metrics.extend(tuned_results)

        # 将调优后的模型加入作图列表
        for name, (_, model) in tuned_models.items():
            if model is not None:
                tuned_name = f'{name}_tuned'
                all_models[tuned_name] = model
                calibrated_models.append((tuned_name, model))

    # 阈值优化与最终评估
    print("")
    print("阈值优化与最终评估")

    final_results = []
    for metrics_entry in all_metrics:
        name = metrics_entry['模型名称']

        # 获取模型
        if name in all_models:
            model = all_models[name]
        else:
            # 集成模型用 metrics 直接的结果
            final_results.append(metrics_entry)
            continue

        try:
            # 概率校准
            try:
                calib = CalibratedClassifierCV(
                    model.__class__(**model.get_params()),
                    cv=3, method='sigmoid'
                )
                calib.fit(X_train, y_train)
                calib_model = calib
            except Exception:
                calib_model = model

            # 阈值搜索
            best_th = find_best_threshold(calib_model, X_val, y_val)
            opt_metrics = evaluate_with_threshold(calib_model, X_test, y_test, threshold=best_th)

            result = {'模型名称': name}
            result.update(opt_metrics)
            final_results.append(result)

            print(f"  {name}: 阈值={best_th:.3f}, "
                  f"F1={opt_metrics['F1值']:.4f}, "
                  f"AUC-ROC={opt_metrics['AUC-ROC']:.4f}, "
                  f"召回率={opt_metrics['召回率']:.4f}")
        except Exception as e:
            print(f"  {name}: 阈值优化失败 ({e})")
            final_results.append(metrics_entry)

    # 结果排名
    print("")
    print("最终排名（按 F1 值排序）")

    results_df = pd.DataFrame(final_results)
    results_df = save_results(results_df, RESULTS_DIR)
    print(results_df.to_string(index=False))

    best_row = results_df.iloc[0]
    print("")
    print(f"最佳模型: {best_row['模型名称']}")
    print(f"  F1={best_row['F1值']:.4f}")
    print(f"  AUC-ROC={best_row['AUC-ROC']:.4f}")
    print(f"  AUC-PR={best_row['AUC-PR']:.4f}")
    print(f"  召回率={best_row['召回率']:.4f}")
    print(f"  特异性={best_row['特异性']:.4f}")

    # 可视化
    print("")
    print("生成评估图表")

    # 构建 (name, model) 列表用于画图
    plot_models = []
    for name, model in all_models.items():
        plot_models.append((name, model))
    # 添加集成模型（用包装类）
    # 对于平均集成，包装为可调用对象
    class EnsembleWrapper:
        def __init__(self, proba_func):
            self._proba_func = proba_func
        def predict_proba(self, X):
            p = self._proba_func(X)
            return np.column_stack([1-p, p])
        def predict(self, X):
            return (self._proba_func(X) >= 0.5).astype(int)

    # 添加集成模型到绘图列表
    # 需要重建预测概率函数
    try:
        plot_roc_curves(plot_models, X_test, y_test, save_dir=RESULTS_DIR)
        plot_pr_curves(plot_models, X_test, y_test, save_dir=RESULTS_DIR)
        print(f"  ROC/PR 曲线已保存到 {RESULTS_DIR}")
    except Exception as e:
        print(f"  绘图失败: {e}")

    # SHAP 分析
    if run_shap:
        best_model_name = best_row['模型名称']
        # 获取最佳模型
        best_model = None
        if best_model_name in all_models:
            best_model = all_models[best_model_name]
        else:
            # 尝试加载pkl
            try:
                best_model = joblib.load(os.path.join(MODELS_DIR, f'{best_model_name}.pkl'))
            except Exception:
                pass

        if best_model is not None:
            shap_analysis(best_model, best_model_name, X_train, X_test, feature_names, RESULTS_DIR, all_models=all_models)

    # SHAP Top5 特征提取
    shap_top5 = None
    shap_path = os.path.join(RESULTS_DIR, 'shap_importance.csv')
    if os.path.exists(shap_path):
        try:
            import pandas as _pd
            _sf = _pd.read_csv(shap_path)
            shap_top5 = _sf['特征'].head(5).tolist()
        except Exception:
            pass

    # 生成训练日志
    end_time = datetime.now()
    log_config = {
        'neg_ratio': neg_ratio,
        'optuna_trials': optuna_trials,
        'use_gpu': use_gpu,
        'buffer_dist': buffer_dist,
        'hybrid_ratio': hybrid_ratio,
        'quality_check': quality_check,
        'run_shap': run_shap,
        'n_total': len(y_train) + len(y_test) if 'y_train' in dir() else None,
        'n_pos': int(y_train.sum()) + int(y_test.sum()) if 'y_train' in dir() else None,
        'n_neg': int((1 - y_train).sum()) + int((1 - y_test).sum()) if 'y_train' in dir() else None,
        'n_features': len(feature_names) if 'feature_names' in dir() else None,
        'train_shape': str(X_train.shape),
        'test_shape': str(X_test.shape),
    }
    log_name = f"train_log_{end_time.strftime('%Y%m%d_%H%M%S')}"
    shap_info = None
    if run_shap and shap_top5:
        shap_info = {
            'best_model': best_row['模型名称'],
            'top5': shap_top5,
            'importance_file': 'results/shap_importance.csv',
        }

    generate_train_log(
        log_name=log_name,
        config=log_config,
        start_time=start_time,
        end_time=end_time,
        results_df=results_df,
        all_metrics=final_results,
        shap_info=shap_info,
        calibrated_models=[name for name, _ in calibrated_models],
    )

    # 训练时长
    duration = end_time - start_time
    print("")
    print(f"训练完成!")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print(f"结果文件: {os.path.join(RESULTS_DIR, 'model_performance.csv')}")
    print(f"模型目录: {MODELS_DIR}")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='滑坡易发性评价 训练脚本')
    parser.add_argument('--neg-ratio', type=float, default=1.0,
                        help='负样本比例 (默认1.0，即1:1)')
    parser.add_argument('--no-shap', action='store_true',
                        help='跳过SHAP分析')
    parser.add_argument('--optuna-trials', type=int, default=0,
                        help='Optuna调参次数 (默认0=不调参)')
    parser.add_argument('--buffer-dist', type=float, default=0.8,
                        help='负样本距灾害点缓冲区距离km (默认0.8)')
    parser.add_argument('--no-quality-check', action='store_true',
                        help='跳过RF质量检验')
    parser.add_argument('--hybrid-ratio', type=float, default=0.3,
                        help='混合负样本中真实数据占比 (默认0.3，即30%%)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='使用GPU加速 (XGBoost/LightGBM/CatBoost)')
    args = parser.parse_args()

    main(neg_ratio=args.neg_ratio, run_shap=not args.no_shap,
         optuna_trials=args.optuna_trials,
         buffer_dist=args.buffer_dist,
         quality_check=not args.no_quality_check,
         hybrid_ratio=args.hybrid_ratio,
         use_gpu=args.use_gpu)
