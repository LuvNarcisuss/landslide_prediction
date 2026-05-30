import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from pre_process import *
from ML_algorithms import *
from improve_algorithms import *
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
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


# ... existing code ...

def stacking_model(model_names):

    X_train, X_test, y_train, y_test = preprocess_data()
    print("\n训练Stacking集成")
    try:
        X_train, y_train = augment_training_data(X_train, y_train, method='smote', verbose=False)
        print(f"  数据增强: {len(X_train)} samples")
    except:
        pass

    # 从本地 models/ 目录加载预训练好的基模型
    import joblib as _jl
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    pkl_paths = [os.path.join(model_dir, f'{n}.pkl') for n in model_names]
    available = [(n, p) for n, p in zip(model_names, pkl_paths) if os.path.exists(p)]

    if available:
        estimators = [(n, _jl.load(p)) for n, p in available]
        print(f"  从 models/ 加载 {len(estimators)} 个预训练基模型: {[n for n,_ in estimators]}")
    else:
        # 回退：创建新模型
        estimators = _get_all_estimators()
        print(f"  创建 {len(estimators)} 个新基模型")

    n_models = len(estimators)
    n_features = X_train.shape[1]

    # 元特征生成（从本地加载的模型直接预测，不做折叠重训练）
    meta_train = np.zeros((len(X_train), n_models))
    meta_test = np.zeros((len(X_test), n_models))

    for i, (name, m) in enumerate(estimators):
        meta_train[:, i] = _safe_proba(m, X_train)
        meta_test[:, i] = _safe_proba(m, X_test)

    trained = list(estimators)

    meta_test = np.column_stack([_safe_proba(m, X_test) for _, m in trained])

    # 拼接原始特征
    meta_train_full = np.hstack([meta_train, X_train])
    meta_test_full = np.hstack([meta_test, X_test])

    scaler = StandardScaler()
    meta_train_scaled = scaler.fit_transform(meta_train_full)
    meta_test_scaled = scaler.transform(meta_test_full)

    meta = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, verbosity=0, scale_pos_weight=2.41)
    meta.fit(meta_train_scaled, y_train)

    train_prob = meta.predict_proba(meta_train_scaled)[:, 1]
    test_prob = meta.predict_proba(meta_test_scaled)[:, 1]

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

    print(f"  基模型:{n_models}个 元特征:{meta_train_scaled.shape[1]}维(概率{n_models}+原始{n_features})")
    print(f"  模型:{len(estimators)}个 阈值={best_th:.3f} F1={f1:.4f} AUC={roc_auc:.4f} PR={pr_auc:.4f}")

    class _StackWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, trained=None, scaler_obj=None, meta_model=None, threshold=0.5):
            self.trained = trained; self.scaler_obj = scaler_obj
            self.meta_model = meta_model; self.threshold = threshold
        def predict_proba(self, X):
            meta = np.column_stack([_safe_proba(m, X) for _, m in self.trained])
            meta_full = np.hstack([meta, X])
            return self.meta_model.predict_proba(self.scaler_obj.transform(meta_full))
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    return {
        '模型名称': 'Stacking',
        '准确率': accuracy, '精确率': precision, '召回率': recall,
        'F1值': f1, '特异性': specificity,
        'AUC-ROC': roc_auc, 'AUC-PR': pr_auc, '最优阈值': best_th,
    }, _StackWrapper(trained, scaler, meta, best_th)


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
    stacking_model(model_names)
