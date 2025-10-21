# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:42:36 2023

@author: Neal LONG
"""
import matplotlib.pyplot as plt
from plot_model_label import plot_mesh_labels
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


    
    
# Load data
iris = load_iris()

# We only take the first two corresponding features
X = iris.data[:, :2]
Y = iris.target

# Create an instance of DecisionTreeClassifier with all default parameters
tree = DecisionTreeClassifier()

#and fit the data.
tree.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each label
# and put the boundary with results of dense points' labels
x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5
y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5
h = 0.01  # step size in the mesh
plt.figure(1, figsize=(12, 12))

plot_mesh_labels(plt, tree, x_min ,x_max, y_min, y_max, 
                 iris.feature_names[0], iris.feature_names[1], h)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, facecolor='k', cmap=plt.cm.RdYlBu)


# # Identify and plot errors
# Y_pred = tree.predict(X)
# X_err = X[Y != Y_pred]
# print("There are {} errors/mismatches".format(sum(Y != Y_pred)))
# plt.scatter(X_err[:, 0], X_err[:, 1], marker='x', c='r', edgecolors='r')


# Complete the plot and display it
plt.title("Decision surface of a decision tree ")
plt.axis("tight")
plt.show()