import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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


def _train_estimators(estimators, X_train, y_train):
    return [(n, clone(m).fit(X_train, y_train)) for n, m in estimators]


def blending_model(model_names):

    X_train, X_test, y_train, y_test = preprocess_data()
    print("\n训练Blending集成")
    try:
        X_train, y_train = augment_training_data(X_train, y_train, method='smote', verbose=False)
    except:
        pass

    X_sub, X_blend, y_sub, y_blend = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # 从本地 models/ 目录加载预训练好的基模型
    import joblib as _jl
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    pkl_paths = [os.path.join(model_dir, f'{n}.pkl') for n in model_names]
    available = [(n, p) for n, p in zip(model_names, pkl_paths) if os.path.exists(p)]

    if available:
        trained = [(n, _jl.load(p)) for n, p in available]
        print(f"  从 models/ 加载 {len(trained)} 个预训练基模型: {[n for n,_ in trained]}")
    else:
        estimators = _get_all_estimators()
        trained = _train_estimators(estimators, X_sub, y_sub)

    blend_meta = np.column_stack([m.predict_proba(X_blend)[:, 1] for _, m in trained])
    test_meta = np.column_stack([m.predict_proba(X_test)[:, 1] for _, m in trained])
    blend_full = np.hstack([blend_meta, X_blend])
    test_full = np.hstack([test_meta, X_test])

    scaler = StandardScaler()
    blend_scaled = scaler.fit_transform(blend_full)
    test_scaled = scaler.transform(test_full)

    meta = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, verbosity=0, scale_pos_weight=2.41)
    meta.fit(blend_scaled, y_blend)

    test_prob = meta.predict_proba(test_scaled)[:, 1]
    train_prob = meta.predict_proba(blend_scaled)[:, 1]

    best_score, best_th = 0, 0.5
    for t in np.linspace(0.05, 0.95, 180):
        yp = (train_prob >= t).astype(int)
        f1_t = f1_score(y_blend, yp, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_blend, yp, labels=[0, 1]).ravel()
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

    print(f"  模型:{len(trained)}个 阈值={best_th:.3f} F1={f1:.4f} AUC={roc_auc:.4f} PR={pr_auc:.4f}")

    class _BlendWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, trained=None, scaler_obj=None, meta_model=None, threshold=0.5):
            self.trained = trained; self.scaler_obj = scaler_obj
            self.meta_model = meta_model; self.threshold = threshold
        def predict_proba(self, X):
            meta = np.column_stack([m.predict_proba(X)[:, 1] for _, m in self.trained])
            meta_full = np.hstack([meta, X])
            return self.meta_model.predict_proba(self.scaler_obj.transform(meta_full))
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    return {
        '模型名称': 'Blending',
        '准确率': accuracy, '精确率': precision, '召回率': recall,
        'F1值': f1, '特异性': specificity,
        'AUC-ROC': roc_auc, 'AUC-PR': pr_auc, '最优阈值': best_th,
    }, _BlendWrapper(trained, scaler, meta, best_th)


if __name__ == "__main__":
    model_names= [
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
    blending_model(model_names)
