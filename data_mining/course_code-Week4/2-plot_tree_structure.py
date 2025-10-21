# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:42:36 2024

@author: Neal LONG
"""
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load data
iris = load_iris()

# Determine feature matrix X and taget array Y
X = iris.data[:100,:2]
Y = iris.target[:100]

# Create and train decision tree on all 
clf = DecisionTreeClassifier()
clf.fit(X, Y)

# Plot tree structure
plt.figure(1, figsize=(6, 6))
plot_tree(clf, filled=True)
plt.show()