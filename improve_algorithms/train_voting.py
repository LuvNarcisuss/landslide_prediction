import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pre_process import *
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def _safe_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except:
        try:
            s = model.decision_function(X)
            return (s - s.min()) / (s.max() - s.min() + 1e-10)
        except:
            return model.predict(X).astype(float)


def _get_all_estimators():
    estimators = []
    estimators.append(('logistic_regression', LogisticRegression(random_state=42, class_weight='balanced', C=1.0, max_iter=1000, solver='saga')))
    estimators.append(('decision_tree', DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced')))
    estimators.append(('svm', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced', gamma='scale', C=1.0)))
    estimators.append(('random_forest', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample', n_jobs=-1)))
    estimators.append(('knn', KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)))
    estimators.append(('gboost', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)))
    try:
        estimators.append(('catboost', CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False, auto_class_weights='Balanced')))
    except:
        pass
    try:
        estimators.append(('lightgbm', LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')))
    except:
        pass
    try:
        estimators.append(('xgboost', XGBClassifier(random_state=42, verbosity=0, scale_pos_weight=2.41)))
    except:
        pass

    return estimators


def _get_auc_weights(estimators, X_train, y_train):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = np.zeros(len(estimators))
    for i, (_, model) in enumerate(estimators):
        fold_aucs = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = clone(model)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            fold_aucs.append(roc_auc_score(y_train[va_idx], _safe_proba(m, X_train[va_idx])))
        aucs[i] = np.mean(fold_aucs)
    temp = 1.5
    exp_s = np.exp((aucs - aucs.max()) / temp)
    return exp_s / exp_s.sum()


def voting_model(model_names):

    X_train, X_test, y_train, y_test = preprocess_data()
    print("\n训练Voting集成")
    try:
        X_train, y_train = augment_training_data(X_train, y_train, method='smote', verbose=False)
    except:
        pass

    # 从本地 models/ 目录加载预训练好的基模型
    import joblib as _jl
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    pkl_paths = [os.path.join(model_dir, f'{n}.pkl') for n in model_names]
    available = [(n, p) for n, p in zip(model_names, pkl_paths) if os.path.exists(p)]

    if available:
        trained = [(n, _jl.load(p)) for n, p in available]
        print(f"  从 models/ 加载 {len(trained)} 个预训练基模型: {[n for n,_ in trained]}")
        # 直接用加载的模型计算 AUC 权重
        weights = np.array([roc_auc_score(y_train, _safe_proba(m, X_train)) for _, m in trained])
        temp = 1.5
        exp_s = np.exp((weights - weights.max()) / temp)
        weights = exp_s / exp_s.sum()
    else:
        estimators = _get_all_estimators()
        weights = _get_auc_weights(estimators, X_train, y_train)
        trained = [(n, clone(m).fit(X_train, y_train)) for n, m in estimators]

    train_prob = np.zeros(len(X_train))
    test_prob = np.zeros(len(X_test))
    for i, (_, m) in enumerate(trained):
        train_prob += _safe_proba(m, X_train) * weights[i]
        test_prob += _safe_proba(m, X_test) * weights[i]

    best_score, best_th = 0, 0.5
    for t in np.linspace(0.05, 0.95, 180):
        yp = (train_prob >= t).astype(int)
        f1_t = f1_score(y_train, yp, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_train, yp, labels=[0, 1]).ravel()
        spec_t = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        score = 0.6 * f1_t + 0.4 * spec_t
        if score > best_score:
            best_score, best_th = score, t

    y_pred = (test_prob >= best_th).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc = roc_auc_score(y_test, test_prob)
    pr_auc = average_precision_score(y_test, test_prob)

    names = [n for n, _ in trained]
    print(f"  模型:{len(trained)}个 阈值={best_th:.3f} F1={f1:.4f} AUC={roc_auc:.4f} PR={pr_auc:.4f}")

    class _VoteWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, trained=None, w=None, threshold=0.5):
            self.trained = trained; self.w = w; self.threshold = threshold
        def predict_proba(self, X):
            p = sum(_safe_proba(m, X) * self.w[i] for i, (_, m) in enumerate(self.trained))
            return np.column_stack([1-p, p])
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    return {
        '模型名称': 'Voting',
        '准确率': accuracy, '精确率': precision, '召回率': recall,
        'F1值': f1, '特异性': specificity,
        'AUC-ROC': roc_auc, 'AUC-PR': pr_auc, '最优阈值': best_th,
    }, _VoteWrapper(trained, weights, best_th)


if __name__ == "__main__":
    model_names = [
        'logistic_regression',
        'decision_tree',
        'svm',
        'random_forest',
        'knn',
        'gboost',
        'catboost',
        'lightgbm',
        'xgboost',

        'imp_catboost',
        'imp_lightgbm',
        'imp_xgboost',
    ]
    voting_model(model_names)
