# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:17:40 2025

@author: Neal

Based on the given train-test split data, (X_train, y_train) & (X_test, y_test), 
evaluate and compare the best out-sample performance(accuracy) of 4 types of models
with settings specified as below:
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
from sklearn.model_selection import train_test_split

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

# By a single train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true, 
                                                test_size=0.4, random_state= 0, stratify=y_true)

#%% Try with DecisionTreeClassifier with defaul settings
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
print("The out-sample accuracy of the built decision tree is", accuracy_score(y_test, y_test_pred))

#%% #++insert your code below++
#%% 1. DecisionTreeClassifier - Find best max_depth (Out-of-Sample)
print("\n" + "="*60)
print("1. DecisionTreeClassifier - Tuning max_depth (Out-of-Sample)")
print("="*60)

dt_results = []
for max_depth in max_depth_candidates:
    # Train on training set
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X_train, y_train)
    
    # Evaluate on test set (out-of-sample)
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Optional: also record training accuracy for comparison
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    dt_results.append({
        'max_depth': max_depth, 
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })
    print(f"max_depth={max_depth}: train_acc={train_accuracy:.6f}, test_acc={test_accuracy:.6f}")

# Sort by test accuracy to find best
dt_results_df = pd.DataFrame(dt_results).sort_values('test_accuracy', ascending=False)
best_dt = dt_results_df.iloc[0]
print(f"\n✓ Best DecisionTree: max_depth={best_dt['max_depth']}")
print(f"  Train Accuracy: {best_dt['train_accuracy']:.6f}")
print(f"  Test Accuracy:  {best_dt['test_accuracy']:.6f}")

#%% 2. LogisticRegression - Find best C (Out-of-Sample)
print("\n" + "="*60)
print("2. LogisticRegression - Tuning C (Out-of-Sample)")
print("="*60)

lr_results = []
for C in C_candidates:
    # Train on training set
    clf = LogisticRegression(C=C, max_iter=100000, random_state=0)
    clf.fit(X_train, y_train)
    
    # Evaluate on test set (out-of-sample)
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Optional: also record training accuracy
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    lr_results.append({
        'C': C, 
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })
    print(f"C={C}: train_acc={train_accuracy:.6f}, test_acc={test_accuracy:.6f}")

# Sort by test accuracy to find best
lr_results_df = pd.DataFrame(lr_results).sort_values('test_accuracy', ascending=False)
best_lr = lr_results_df.iloc[0]
print(f"\n✓ Best LogisticRegression: C={best_lr['C']}")
print(f"  Train Accuracy: {best_lr['train_accuracy']:.6f}")
print(f"  Test Accuracy:  {best_lr['test_accuracy']:.6f}")

#%% 3. LinearSVC - Find best C (Out-of-Sample)
print("\n" + "="*60)
print("3. LinearSVC - Tuning C (Out-of-Sample)")
print("="*60)

lsvc_results = []
for C in C_candidates:
    # Train on training set
    clf = LinearSVC(C=C, max_iter=100000, dual='auto', random_state=0)
    clf.fit(X_train, y_train)
    
    # Evaluate on test set (out-of-sample)
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Optional: also record training accuracy
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    lsvc_results.append({
        'C': C, 
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })
    print(f"C={C}: train_acc={train_accuracy:.6f}, test_acc={test_accuracy:.6f}")

# Sort by test accuracy to find best
lsvc_results_df = pd.DataFrame(lsvc_results).sort_values('test_accuracy', ascending=False)
best_lsvc = lsvc_results_df.iloc[0]
print(f"\n✓ Best LinearSVC: C={best_lsvc['C']}")
print(f"  Train Accuracy: {best_lsvc['train_accuracy']:.6f}")
print(f"  Test Accuracy:  {best_lsvc['test_accuracy']:.6f}")

#%% 4. SVC - Find best C (Out-of-Sample)
print("\n" + "="*60)
print("4. SVC - Tuning C (Out-of-Sample)")
print("="*60)
print("⚠️  Warning: SVC training may take a while...")

svc_results = []
for C in C_candidates:
    print(f"Training SVC with C={C}...", end=" ")
    # Train on training set
    clf = SVC(C=C, max_iter=100000, random_state=0)
    clf.fit(X_train, y_train)
    
    # Evaluate on test set (out-of-sample)
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Optional: also record training accuracy
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    svc_results.append({
        'C': C, 
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })
    print(f"Done! train_acc={train_accuracy:.6f}, test_acc={test_accuracy:.6f}")

# Sort by test accuracy to find best
svc_results_df = pd.DataFrame(svc_results).sort_values('test_accuracy', ascending=False)
best_svc = svc_results_df.iloc[0]
print(f"\n✓ Best SVC: C={best_svc['C']}")
print(f"  Train Accuracy: {best_svc['train_accuracy']:.6f}")
print(f"  Test Accuracy:  {best_svc['test_accuracy']:.6f}")

#%% Final Comparison (Out-of-Sample Performance)
print("\n" + "="*60)
print("FINAL COMPARISON - Best Out-of-Sample Performance")
print("="*60)

comparison = pd.DataFrame([
    {
        'Model': 'DecisionTreeClassifier', 
        'Best_Parameter': f"max_depth={best_dt['max_depth']}", 
        'Train_Accuracy': best_dt['train_accuracy'],
        'Test_Accuracy': best_dt['test_accuracy'],
        'Overfitting_Gap': best_dt['train_accuracy'] - best_dt['test_accuracy']
    },
    {
        'Model': 'LogisticRegression', 
        'Best_Parameter': f"C={best_lr['C']}", 
        'Train_Accuracy': best_lr['train_accuracy'],
        'Test_Accuracy': best_lr['test_accuracy'],
        'Overfitting_Gap': best_lr['train_accuracy'] - best_lr['test_accuracy']
    },
    {
        'Model': 'LinearSVC', 
        'Best_Parameter': f"C={best_lsvc['C']}", 
        'Train_Accuracy': best_lsvc['train_accuracy'],
        'Test_Accuracy': best_lsvc['test_accuracy'],
        'Overfitting_Gap': best_lsvc['train_accuracy'] - best_lsvc['test_accuracy']
    },
    {
        'Model': 'SVC', 
        'Best_Parameter': f"C={best_svc['C']}", 
        'Train_Accuracy': best_svc['train_accuracy'],
        'Test_Accuracy': best_svc['test_accuracy'],
        'Overfitting_Gap': best_svc['train_accuracy'] - best_svc['test_accuracy']
    }
]).sort_values('Test_Accuracy', ascending=False)

print(comparison.to_string(index=False))

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {comparison.iloc[0]['Model']}")
print(f"   Parameter: {comparison.iloc[0]['Best_Parameter']}")
print(f"   Test Accuracy: {comparison.iloc[0]['Test_Accuracy']:.6f}")
print(f"   Overfitting Gap: {comparison.iloc[0]['Overfitting_Gap']:.6f}")
print("="*60)

#%% Optional: Save results to CSV
dt_results_df.to_csv('dt_results_outsample.csv', index=False)
lr_results_df.to_csv('lr_results_outsample.csv', index=False)
lsvc_results_df.to_csv('lsvc_results_outsample.csv', index=False)
svc_results_df.to_csv('svc_results_outsample.csv', index=False)
comparison.to_csv('model_comparison_outsample.csv', index=False)
