# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:30:33 2025

@author: Neal LONG
Ref:
    1.plt.semilogx: https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.semilogx.html

Please perform 10-fold cross-validation to identify the optimal gamma for the SVC model from the provided gamma_candidates.

Note:
    0. Set random_state=0 for all models and cross-validation procedures.
    1. Use plt.semilogx to visualize the training and cross-validation score curve, as the range of gamma in "gamma_candidates" spans different scales.
    2. The plot title should reflect the number of folds (k) used in cross-validation and your student ID (stu_id) formatted as: f"Validation Curve based on {k}-fold CV by {stu_id}".
    3. Utilize accuracy as the scoring metric.
    4. Implement 10-fold stratified cross-validation.
    5. Refer to "fitting_graph.ipynb" from Week 7 for guidance.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.model_selection import StratifiedKFold

X, y = load_digits(return_X_y=True)
gamma_candidates = np.logspace(-6, -1, 5)

stu_id = "YOUR_STUDENT_ID"
k = 10

cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

train_scores, val_scores = validation_curve(
    SVC(random_state=0),
    X, y,
    param_name="gamma",
    param_range=gamma_candidates,
    cv=cv,
    scoring="accuracy"
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.semilogx(gamma_candidates, train_mean, marker='o', label='Training score', linewidth=2)
plt.semilogx(gamma_candidates, val_mean, marker='o', label='Cross-validation score', linewidth=2)
plt.fill_between(gamma_candidates, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(gamma_candidates, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.title(f"Validation Curve based on {k}-fold CV by {stu_id}")
plt.legend(loc='best')
plt.grid(True)
plt.savefig('./Q1_validation_curve.png', dpi=150, bbox_inches='tight')

print(f"Gamma candidates: {gamma_candidates}")
print(f"Training scores (mean): {train_mean}")
print(f"Cross-validation scores (mean): {val_mean}")
print(f"Best gamma index: {np.argmax(val_mean)}")
print(f"Best gamma value: {gamma_candidates[np.argmax(val_mean)]}")