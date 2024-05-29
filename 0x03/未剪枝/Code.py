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
    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, criterion='entropy'):
    """建立决策树"""
    clf = DecisionTreeClassifier(criterion=criterion, splitter='best', random_state=42)
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
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    clf_c45 = train_model(X_train, y_train, criterion='entropy')
    clf_cart = train_model(X_train, y_train, criterion='gini')
    
    accuracy_c45 = evaluate_model(clf_c45, X_test, y_test)
    accuracy_cart = evaluate_model(clf_cart, X_test, y_test)
    
    print("C4.5 准确率:", accuracy_c45)
    print("CART 准确率:", accuracy_cart)
    
    visualize_tree(clf_c45, feature_names, target_names, "iris_c45")
    visualize_tree(clf_cart, feature_names, target_names, "iris_cart")

if __name__ == "__main__":
    main()