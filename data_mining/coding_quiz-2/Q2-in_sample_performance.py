# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:17:40 2025

@author: Neal

Based on the given training data (X, y_true), evaluate and compare the best 
in-sample performance(accuracy) of 4 types of models with settings specified as below:
    1. DecisionTreeClassifier with 
        the best `max_depth` from cadidates provided in `max_depth_candidates`
    2. LogisticRegression with 
        `max_iter`=100000, 
        and the best `C` from cadidates provided in `C_candidates`
    3. LinearSVC with 
        'max_iter`=100000, 
        `dual`='auto',
        and the best `C` from cadidates provided in `C_candidates`
    4. SVC with 
        `max_iter`=100000, 
        and the best `C` from cadidates provided in `C_candidates`

Note：
    1. Always set random_state = 0 for all the models
    2. Record the performance of each parameter for each type of model and then
        sort to get the best model parameter(s) and corresponding performance for each model
    3. Compare the performance of 4 types of models
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd

#%% Provided dataset and settings

max_depth_candidates = [1,5,10,15,20]
C_candidates = [0.01,0.1,1,10,100]

data = pd.read_csv(r'./data/creditcard_train.csv')[:50000]
y_true = data.pop('label')
X = data


#%% Try with DecisionTreeClassifier with defaul settings
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y_true)
y_pred = clf.predict(X)
print("The in-sample accuracy of the built decision tree is", accuracy_score(y_true, y_pred))

#%% 1. DecisionTreeClassifier - Find best max_depth
print("\n" + "="*60)
print("1. DecisionTreeClassifier - Tuning max_depth")
print("="*60)

dt_results = []
for max_depth in max_depth_candidates:
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X, y_true)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    dt_results.append({'max_depth': max_depth, 'accuracy': accuracy})
    print(f"max_depth={max_depth}: accuracy={accuracy:.6f}")

# Sort by accuracy to find best
dt_results_df = pd.DataFrame(dt_results).sort_values('accuracy', ascending=False)
best_dt = dt_results_df.iloc[0]
print(f"\nBest DecisionTree: max_depth={best_dt['max_depth']}, accuracy={best_dt['accuracy']:.6f}")

#%% 2. LogisticRegression - Find best C
print("\n" + "="*60)
print("2. LogisticRegression - Tuning C")
print("="*60)

lr_results = []
for C in C_candidates:
    clf = LogisticRegression(C=C, max_iter=100000, random_state=0)
    clf.fit(X, y_true)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    lr_results.append({'C': C, 'accuracy': accuracy})
    print(f"C={C}: accuracy={accuracy:.6f}")

# Sort by accuracy to find best
lr_results_df = pd.DataFrame(lr_results).sort_values('accuracy', ascending=False)
best_lr = lr_results_df.iloc[0]
print(f"\nBest LogisticRegression: C={best_lr['C']}, accuracy={best_lr['accuracy']:.6f}")

#%% 3. LinearSVC - Find best C
print("\n" + "="*60)
print("3. LinearSVC - Tuning C")
print("="*60)

lsvc_results = []
for C in C_candidates:
    clf = LinearSVC(C=C, max_iter=100000, dual='auto', random_state=0)
    clf.fit(X, y_true)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    lsvc_results.append({'C': C, 'accuracy': accuracy})
    print(f"C={C}: accuracy={accuracy:.6f}")

# Sort by accuracy to find best
lsvc_results_df = pd.DataFrame(lsvc_results).sort_values('accuracy', ascending=False)
best_lsvc = lsvc_results_df.iloc[0]
print(f"\nBest LinearSVC: C={best_lsvc['C']}, accuracy={best_lsvc['accuracy']:.6f}")

#%% 4. SVC - Find best C
print("\n" + "="*60)
print("4. SVC - Tuning C")
print("="*60)

svc_results = []
for C in C_candidates:
    clf = SVC(C=C, max_iter=100000, random_state=0)
    clf.fit(X, y_true)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    svc_results.append({'C': C, 'accuracy': accuracy})
    print(f"C={C}: accuracy={accuracy:.6f}")

# Sort by accuracy to find best
svc_results_df = pd.DataFrame(svc_results).sort_values('accuracy', ascending=False)
best_svc = svc_results_df.iloc[0]
print(f"\nBest SVC: C={best_svc['C']}, accuracy={best_svc['accuracy']:.6f}")

#%% Final Comparison
print("\n" + "="*60)
print("FINAL COMPARISON - Best Model from Each Type")
print("="*60)

comparison = pd.DataFrame([
    {'Model': 'DecisionTreeClassifier', 
     'Best_Parameter': f"max_depth={best_dt['max_depth']}", 
     'Accuracy': best_dt['accuracy']},
    {'Model': 'LogisticRegression', 
     'Best_Parameter': f"C={best_lr['C']}", 
     'Accuracy': best_lr['accuracy']},
    {'Model': 'LinearSVC', 
     'Best_Parameter': f"C={best_lsvc['C']}", 
     'Accuracy': best_lsvc['accuracy']},
    {'Model': 'SVC', 
     'Best_Parameter': f"C={best_svc['C']}", 
     'Accuracy': best_svc['accuracy']}
]).sort_values('Accuracy', ascending=False)

print(comparison.to_string(index=False))
print("\n" + "="*60)
print(f"🏆 WINNER: {comparison.iloc[0]['Model']} with {comparison.iloc[0]['Best_Parameter']}")
print(f"   Best Accuracy: {comparison.iloc[0]['Accuracy']:.6f}")
print("="*60)
