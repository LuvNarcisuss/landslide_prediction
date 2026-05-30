import os
import sys
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pre_process import preprocess_data

# 添加根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def catboost_model():

    # 加载数据
    X_train, X_test, y_train, y_test = preprocess_data()

    # 创建并训练模型
    model = CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False)
    model.fit(X_train, y_train)
    # 保存模型
    joblib.dump(model, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'catboost.pkl'))

    # 预测
    y_pred = model.predict(X_test)

    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    # 输出结果
    print("\nCatBoost模型训练完成")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print(f"特异性: {specificity:.4f}")

    return {
        '模型名称': 'CatBoost',
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1值': f1,
        '特异性': specificity,
    }, model

if __name__ == "__main__":
    catboost_model()
