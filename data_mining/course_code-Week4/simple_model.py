# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 11:47:29 2025

@author: Neal
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class SimpleRuleClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple scikit-learn compatible classifier that predicts based on a rule:
    - If both features X < 2 and Y < 2, predict 0.
    - Otherwise, predict 1.
    
    The input X should be a 2D array of shape (n_samples, 2), where each row is [X, Y].
    """
    
    def fit(self, X, y=None):
        """
        Fit the model (no actual fitting is needed for this rule-based classifier).
        
        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            The input features.
        y : array-like, shape (n_samples,), default=None
            The target values (ignored in this model).
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # No parameters to learn; just return self
        return self
    
    def predict(self, X):
        """
        Predict the class labels for samples in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            The input features.
        
        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted class labels (0 or 1).
        """
        X = np.asarray(X)
        if X.shape[1] != 2:
            raise ValueError("X must have exactly 2 features (X, Y).")
        
        # Apply the rule: 0 if both < 2, else 1
        predictions = np.where((X[:, 0] < 2) & (X[:, 1] < 2), 0, 1)
        return predictions.astype(int)