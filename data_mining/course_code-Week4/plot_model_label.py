# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 11:19:39 2025

@author: Neal
"""

import numpy as np
import matplotlib.pyplot as plt
from simple_model import SimpleRuleClassifier



def plot_mesh_labels(plt, model, x_min, x_max, y_min, y_max, x_label, y_label, step = 0.01):
    """
    Use the labels predicted by the model to  color the rectangular area 
    defined by the coordinates [x_min, x_max] and [y_min, y_max].
    """
   
    # Create a mesh grid for the area [x_min, x_max]x[y_min, y_max] with step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict the lables for the mesh grid
    mesh_labels = model.predict(mesh_points)
    mesh_labels_array = mesh_labels.reshape(xx.shape)

    # Plot the mesh grid colored by the predicted labels
    plt.pcolormesh(xx, yy, mesh_labels_array, cmap=plt.cm.RdYlBu,shading='auto')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(x_label)
    plt.ylabel(y_label)

if __name__ == "__main__":
    model = SimpleRuleClassifier()
    plt.figure(1, figsize=(6, 6))

    plot_mesh_labels(plt, model, 0 ,11, 0, 11, "X", "Y", 0.01)
    