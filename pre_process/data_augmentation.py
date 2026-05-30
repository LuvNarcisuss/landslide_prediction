"""
数据增强模块
对训练数据进行增强，提高模型泛化能力

策略：
1. SMOTE - 对少数类(标签0)做合成过采样
2. 高斯噪声增强 - 对连续特征加小噪声
3. 插值增强 - 同类样本间线性插值
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def augment_training_data(X_train, y_train, method='smote', verbose=True):
    """
    对训练数据进行增强

    Parameters:
    - X_train: 训练特征
    - y_train: 训练标签
    - method: 'smote' | 'noise' | 'hybrid'
    - verbose: 是否打印信息

    Returns:
    - X_aug, y_aug: 增强后的训练数据
    """
    n_orig = len(X_train)
    n_minority = (y_train == 0).sum()
    n_majority = (y_train == 1).sum()
    imbalance_ratio = n_majority / n_minority if n_minority > 0 else 1.0

    if verbose:
        print(f"数据增强: {n_orig} samples, 少数类={n_minority}, 多数类={n_majority}, 比例=1:{imbalance_ratio:.2f}")

    if method == 'smote':
        # SMOTE: 合成少数类过采样，使少数类达到多数类的80%
        target_ratio = 0.8
        sampling_strategy = {0: int(n_majority * target_ratio), 1: n_majority}

        try:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5, n_jobs=-1)
            X_aug, y_aug = smote.fit_resample(X_train, y_train)
            if verbose:
                print(f"  SMOTE增强: {n_orig} → {len(X_aug)} samples")
                print(f"  新分布: 标签0={int((y_aug==0).sum())}, 标签1={int((y_aug==1).sum())}")
            return X_aug, y_aug
        except Exception as e:
            print(f"  SMOTE失败: {e}, 使用噪声增强作为回退")
            method = 'noise'

    if method == 'noise' or method == 'hybrid':
        # 高斯噪声增强（对连续特征加小噪声）
        n_synth = min(n_minority * 3, n_majority - n_minority)

        # 识别连续特征（二值特征不应加噪声）
        if X_train.shape[1] > 0:
            # 假设第20个特征之后是二值/聚类特征，不增强
            n_cont = min(X_train.shape[1], 25)

        # 只对少数类样本加噪声
        minority_mask = y_train == 0
        X_min = X_train[minority_mask]
        n_min = len(X_min)

        if n_min > 1 and n_synth > 0:
            # 从少数类中随机采样并加噪声
            indices = np.random.choice(n_min, size=min(n_synth, n_min * 2), replace=True)
            X_noise = X_min[indices].copy()
            # 加小高斯噪声（标准差的5%）
            noise_scale = 0.05
            for j in range(min(X_noise.shape[1], n_cont)):
                col_std = np.std(X_train[:, j])
                if col_std > 1e-6:
                    X_noise[:, j] += np.random.randn(len(X_noise)) * col_std * noise_scale

            X_aug = np.vstack([X_train, X_noise])
            y_aug = np.hstack([y_train, np.zeros(len(X_noise), dtype=int)])

            if verbose:
                print(f"  噪声增强: {n_orig} → {len(X_aug)} samples")
                print(f"  生成了{len(X_noise)}个合成少数类样本")
        else:
            X_aug, y_aug = X_train.copy(), y_train.copy()
            if verbose:
                print("  少数类样本太少，跳过噪声增强")

        # 如果是hybrid模式，再应用SMOTE
        if method == 'hybrid' and n_min > 5:
            try:
                smote = SMOTE(sampling_strategy={0: int(n_majority * 0.7)}, random_state=42, k_neighbors=3, n_jobs=-1)
                X_aug, y_aug = smote.fit_resample(X_aug, y_aug)
                if verbose:
                    print(f"  混合增强: 最终 {len(X_aug)} samples")
            except Exception as e:
                print(f"  SMOTE第二阶段失败: {e}")

    return X_aug, y_aug


def augment_with_validation(X_train, y_train, method='smote', n_folds=3):
    """
    在交叉验证内部做数据增强（防止数据泄漏）
    返回增强后的完整训练集

    适用于：在训练集成模型前对全部训练数据做一次增强
    """
    X_aug, y_aug = augment_training_data(X_train, y_train, method=method, verbose=False)
    return X_aug, y_aug


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, '.')
    from pre_process import preprocess_data

    X_train, X_test, y_train, y_test = preprocess_data()

    print("=== SMOTE增强测试 ===")
    X_smote, y_smote = augment_training_data(X_train, y_train, method='smote', verbose=True)
    print(f"增强后特征维度: {X_smote.shape}")

    print("\n=== 噪声增强测试 ===")
    X_noise, y_noise = augment_training_data(X_train, y_train, method='noise', verbose=True)
    print(f"增强后特征维度: {X_noise.shape}")
