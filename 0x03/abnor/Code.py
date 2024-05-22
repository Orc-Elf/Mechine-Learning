import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

def load_data():
    """加载数据"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.feature_names, iris.target_names

def split_data(X, y):
    """划分训练集、验证集和测试集"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, criterion='entropy', preprune=False, max_depth=None, ccp_alpha=0.0):
    """建立决策树 (支持预剪枝和后剪枝)"""
    clf = DecisionTreeClassifier(criterion=criterion, splitter='best', random_state=42, max_depth=max_depth, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """评估模型"""
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def visualize_tree(clf, feature_names, target_names, filename):
    """可视化决策树"""
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=target_names, filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename)

def main():
    X, y, feature_names, target_names = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # C4.5
    clf_c45 = train_model(X_train, y_train)
    clf_c45_preprune = train_model(X_train, y_train, preprune=True, max_depth=3)
    clf_c45_postprune = train_model(X_train, y_train, ccp_alpha=0.015)

    # CART
    clf_cart = train_model(X_train, y_train, criterion='gini')
    clf_cart_preprune = train_model(X_train, y_train, criterion='gini', preprune=True, max_depth=3)
    clf_cart_postprune = train_model(X_train, y_train, criterion='gini', ccp_alpha=0.012)

    # 评估模型 (在验证集上选择最佳模型)
    models = [clf_c45, clf_c45_preprune, clf_c45_postprune, clf_cart, clf_cart_preprune, clf_cart_postprune]
    model_names = ['C4.5', 'C4.5 (预剪枝)', 'C4.5 (后剪枝)', 'CART', 'CART (预剪枝)', 'CART (后剪枝)']
    best_accuracy = 0
    best_model = None
    for model, name in zip(models, model_names):
        accuracy = evaluate_model(model, X_val, y_val)
        print(f"{name} 验证集准确率: {accuracy}")  # 在名称中添加剪枝标识
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    print(f"\n最佳模型: {model_names[models.index(best_model)]}, 验证集准确率: {best_accuracy}")

    # 在测试集上评估最佳模型
    test_accuracy = evaluate_model(best_model, X_test, y_test)
    print(f"最佳模型在测试集上的准确率: {test_accuracy}")
    visualize_tree(best_model, feature_names, target_names, f"iris_{model_names[models.index(best_model)]}_best")

if __name__ == "__main__":
    main()