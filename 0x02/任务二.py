import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import roc_curve, auc, confusion_matrix
import altair as alt

# 1. 加载并准备 Iris 数据集
iris = load_iris()
X, y = iris.data, iris.target

# 随机打乱数据集
rng = np.random.RandomState(42)  # 设置随机种子，确保结果可复现
shuffle_index = rng.permutation(len(X))
X, y = X[shuffle_index], y[shuffle_index]

# 模拟模型预测概率（0到1之间的随机数）
y_proba = rng.rand(len(y), 3)  # 生成多分类概率


# 2. 定义计算性能指标的函数
def calculate_performance_metrics(y_true, y_proba):
    """计算各种性能指标"""
    y_pred_class = y_proba.argmax(axis=1)  # 获取概率最大的类别作为预测类别

    # 1. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_class)

    # 2. 从混淆矩阵计算指标
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # 3. 计算各类别的指标
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)

    # 4. 计算总体的指标
    accuracy = np.sum(TP) / cm.sum()
    error_rate = 1 - accuracy
    mse = np.mean((y_true - y_proba.max(axis=1)) ** 2)  # 使用最大概率计算 MSE

    # 5. ROC 曲线和 AUC 值
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return {
        'mse': mse,
        'error_rate': error_rate,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
    }


# 3. 计算性能指标
results = calculate_performance_metrics(y, y_proba)

# 打印性能指标
print(f"均方误差 (MSE): {results['mse']:.4f}")
print(f"错误率: {results['error_rate']:.4f}")
print(f"精度: {results['accuracy']:.4f}")
for i in range(3):
    print(f"类别 {i} - "
          f"查准率: {results['precision'][i]:.4f} | "
          f"查全率: {results['recall'][i]:.4f} | "
          f"F1 指数: {results['f1_score'][i]:.4f} | "
          f"AUC 值: {results['roc_auc'][i]:.4f}")

print("\n混淆矩阵:")
print(results['confusion_matrix'])

# 4. 绘制 ROC 曲线
roc_df = pd.DataFrame({
    '假正例率': np.concatenate([results['fpr'][i] for i in range(3)]),
    '真正例率': np.concatenate([results['tpr'][i] for i in range(3)]),
    '类别': np.repeat(['类别 0', '类别 1', '类别 2'], [len(results['fpr'][i]) for i in range(3)])
})

roc_chart = alt.Chart(roc_df).mark_line(point=True).encode(
    x='假正例率',
    y='真正例率',
    color='类别:N',
    tooltip=['假正例率', '真正例率', '类别']
).properties(
    title='Iris 各类别的 ROC 曲线'
).interactive()

diagonal_line = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(strokeDash=[5, 5], color='gray').encode(
    x='x',
    y='y'
)

combined_chart = roc_chart + diagonal_line
combined_chart.show()
