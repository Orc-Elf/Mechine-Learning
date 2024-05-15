from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def print_dataset_shapes(X_train, X_test):
    print("训练集大小: {}".format(X_train.shape))
    print("测试集大小: {}".format(X_test.shape))


# 读取iris数据集
wine = datasets.load_wine()
X = wine.data
y = wine.target

if __name__ == '__main__':
    # 100次留出法
    holdout_iterations = 100
    for i in range(holdout_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        print_dataset_shapes(X_train, X_test)

    # k折交叉验证法
    k_folds = 10  # k 折交叉验证中的折数
    cross_val_rounds = 3  # p 次 k 折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    for i in range(cross_val_rounds):
        print(f"第 {i + 1} 次 k 折交叉验证：")
        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            print(f"折 {fold + 1}:")
            print_dataset_shapes(X_train, X_test)

        print()  # 每次交叉验证后打印一个空行
