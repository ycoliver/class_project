# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:17:40 2025

@author: Neal

Based on the given training data (X, y_true), use the defined `skf` to 
perform a 5-fold corss-validation to evaluate and compare the best out-sample 
performance(accuracy) of 4 types of models with settings specified as below:
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
    4. Consider using cross_val_score to perform corss-validation with 
        the defined `skf`
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd

#%% Provided dataset and settings
best_model_parameters = []
max_depth_candidates = [1,5,10,15,20]
C_candidates = [0.01,0.1,1,10,100]

data = pd.read_csv(r'./data/creditcard_train.csv')[:50000]
y_true = data.pop('label')
X = data
skf = StratifiedKFold(n_splits=5)


#%% #++insert your code below++
#%% 1. DecisionTreeClassifier - 5-Fold Cross-Validation
print("="*70)
print("1. DecisionTreeClassifier - Tuning max_depth (5-Fold CV)")
print("="*70)

dt_results = []
for max_depth in max_depth_candidates:
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(clf, X, y_true, cv=skf, scoring='accuracy')
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    dt_results.append({
        'max_depth': max_depth,
        'cv_mean_accuracy': mean_score,
        'cv_std_accuracy': std_score,
        'cv_scores': cv_scores
    })
    
    print(f"max_depth={max_depth:2d}: mean={mean_score:.6f} (+/- {std_score:.6f})")
    print(f"  Individual fold scores: {cv_scores}")

# Sort by mean CV accuracy
dt_results_df = pd.DataFrame(dt_results).sort_values('cv_mean_accuracy', ascending=False)
best_dt = dt_results_df.iloc[0]

print(f"\n✓ Best DecisionTree:")
print(f"  max_depth = {best_dt['max_depth']}")
print(f"  CV Mean Accuracy = {best_dt['cv_mean_accuracy']:.6f} (+/- {best_dt['cv_std_accuracy']:.6f})")

best_model_parameters.append({
    'Model': 'DecisionTreeClassifier',
    'Best_Parameter': f"max_depth={best_dt['max_depth']}",
    'CV_Mean_Accuracy': best_dt['cv_mean_accuracy'],
    'CV_Std_Accuracy': best_dt['cv_std_accuracy']
})

#%% 2. LogisticRegression - 5-Fold Cross-Validation
print("\n" + "="*70)
print("2. LogisticRegression - Tuning C (5-Fold CV)")
print("="*70)

lr_results = []
for C in C_candidates:
    clf = LogisticRegression(C=C, max_iter=100000, random_state=0)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(clf, X, y_true, cv=skf, scoring='accuracy')
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    lr_results.append({
        'C': C,
        'cv_mean_accuracy': mean_score,
        'cv_std_accuracy': std_score,
        'cv_scores': cv_scores
    })
    
    print(f"C={C:6.2f}: mean={mean_score:.6f} (+/- {std_score:.6f})")
    print(f"  Individual fold scores: {cv_scores}")

# Sort by mean CV accuracy
lr_results_df = pd.DataFrame(lr_results).sort_values('cv_mean_accuracy', ascending=False)
best_lr = lr_results_df.iloc[0]

print(f"\n✓ Best LogisticRegression:")
print(f"  C = {best_lr['C']}")
print(f"  CV Mean Accuracy = {best_lr['cv_mean_accuracy']:.6f} (+/- {best_lr['cv_std_accuracy']:.6f})")

best_model_parameters.append({
    'Model': 'LogisticRegression',
    'Best_Parameter': f"C={best_lr['C']}",
    'CV_Mean_Accuracy': best_lr['cv_mean_accuracy'],
    'CV_Std_Accuracy': best_lr['cv_std_accuracy']
})

#%% 3. LinearSVC - 5-Fold Cross-Validation
print("\n" + "="*70)
print("3. LinearSVC - Tuning C (5-Fold CV)")
print("="*70)

lsvc_results = []
for C in C_candidates:
    clf = LinearSVC(C=C, max_iter=100000, dual='auto', random_state=0)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(clf, X, y_true, cv=skf, scoring='accuracy')
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    lsvc_results.append({
        'C': C,
        'cv_mean_accuracy': mean_score,
        'cv_std_accuracy': std_score,
        'cv_scores': cv_scores
    })
    
    print(f"C={C:6.2f}: mean={mean_score:.6f} (+/- {std_score:.6f})")
    print(f"  Individual fold scores: {cv_scores}")

# Sort by mean CV accuracy
lsvc_results_df = pd.DataFrame(lsvc_results).sort_values('cv_mean_accuracy', ascending=False)
best_lsvc = lsvc_results_df.iloc[0]

print(f"\n✓ Best LinearSVC:")
print(f"  C = {best_lsvc['C']}")
print(f"  CV Mean Accuracy = {best_lsvc['cv_mean_accuracy']:.6f} (+/- {best_lsvc['cv_std_accuracy']:.6f})")

best_model_parameters.append({
    'Model': 'LinearSVC',
    'Best_Parameter': f"C={best_lsvc['C']}",
    'CV_Mean_Accuracy': best_lsvc['cv_mean_accuracy'],
    'CV_Std_Accuracy': best_lsvc['cv_std_accuracy']
})

#%% 4. SVC - 5-Fold Cross-Validation
print("\n" + "="*70)
print("4. SVC - Tuning C (5-Fold CV)")
print("="*70)
print("⚠️  Warning: SVC with cross-validation may take a while...")

svc_results = []
for C in C_candidates:
    print(f"\nTraining SVC with C={C}...")
    clf = SVC(C=C, max_iter=100000, random_state=0)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(clf, X, y_true, cv=skf, scoring='accuracy')
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    svc_results.append({
        'C': C,
        'cv_mean_accuracy': mean_score,
        'cv_std_accuracy': std_score,
        'cv_scores': cv_scores
    })
    
    print(f"C={C:6.2f}: mean={mean_score:.6f} (+/- {std_score:.6f})")
    print(f"  Individual fold scores: {cv_scores}")

# Sort by mean CV accuracy
svc_results_df = pd.DataFrame(svc_results).sort_values('cv_mean_accuracy', ascending=False)
best_svc = svc_results_df.iloc[0]

print(f"\n✓ Best SVC:")
print(f"  C = {best_svc['C']}")
print(f"  CV Mean Accuracy = {best_svc['cv_mean_accuracy']:.6f} (+/- {best_svc['cv_std_accuracy']:.6f})")

best_model_parameters.append({
    'Model': 'SVC',
    'Best_Parameter': f"C={best_svc['C']}",
    'CV_Mean_Accuracy': best_svc['cv_mean_accuracy'],
    'CV_Std_Accuracy': best_svc['cv_std_accuracy']
})

#%% Final Comparison - Cross-Validation Results
print("\n" + "="*70)
print("FINAL COMPARISON - Best 5-Fold Cross-Validation Performance")
print("="*70)

comparison_df = pd.DataFrame(best_model_parameters).sort_values('CV_Mean_Accuracy', ascending=False)

# Format the output
comparison_df['Performance'] = comparison_df.apply(
    lambda row: f"{row['CV_Mean_Accuracy']:.6f} (+/- {row['CV_Std_Accuracy']:.6f})", 
    axis=1
)

print(comparison_df[['Model', 'Best_Parameter', 'Performance']].to_string(index=False))

print("\n" + "="*70)
print(f"🏆 BEST MODEL (based on Cross-Validation):")
print(f"   Model: {comparison_df.iloc[0]['Model']}")
print(f"   Best Parameter: {comparison_df.iloc[0]['Best_Parameter']}")
print(f"   CV Mean Accuracy: {comparison_df.iloc[0]['CV_Mean_Accuracy']:.6f}")
print(f"   CV Std Accuracy:  {comparison_df.iloc[0]['CV_Std_Accuracy']:.6f}")
print("="*70)

#%% Statistical Significance Analysis
print("\n📊 Model Stability Analysis (based on CV std):")
print("-" * 70)
for _, row in comparison_df.iterrows():
    std = row['CV_Std_Accuracy']
    if std < 0.001:
        status = "✓ Excellent Stability"
    elif std < 0.005:
        status = "✓ Good Stability"
    elif std < 0.01:
        status = "⚠ Moderate Stability"
    else:
        status = "⚠ High Variance"
    print(f"{row['Model']:25s}: std={std:.6f} - {status}")


import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('5-Fold Cross-Validation Results', fontsize=16, fontweight='bold')

# Plot 1: DecisionTree
ax1 = axes[0, 0]
ax1.errorbar(dt_results_df['max_depth'], 
                dt_results_df['cv_mean_accuracy'],
                yerr=dt_results_df['cv_std_accuracy'],
                marker='o', capsize=5, capthick=2)
ax1.set_xlabel('max_depth')
ax1.set_ylabel('CV Mean Accuracy')
ax1.set_title('DecisionTreeClassifier')
ax1.grid(True, alpha=0.3)

# Plot 2: LogisticRegression
ax2 = axes[0, 1]
ax2.errorbar(range(len(C_candidates)), 
                lr_results_df['cv_mean_accuracy'],
                yerr=lr_results_df['cv_std_accuracy'],
                marker='s', capsize=5, capthick=2)
ax2.set_xticks(range(len(C_candidates)))
ax2.set_xticklabels([str(c) for c in lr_results_df['C']])
ax2.set_xlabel('C')
ax2.set_ylabel('CV Mean Accuracy')
ax2.set_title('LogisticRegression')
ax2.grid(True, alpha=0.3)

# Plot 3: LinearSVC
ax3 = axes[1, 0]
ax3.errorbar(range(len(C_candidates)), 
                lsvc_results_df['cv_mean_accuracy'],
                yerr=lsvc_results_df['cv_std_accuracy'],
                marker='^', capsize=5, capthick=2)
ax3.set_xticks(range(len(C_candidates)))
ax3.set_xticklabels([str(c) for c in lsvc_results_df['C']])
ax3.set_xlabel('C')
ax3.set_ylabel('CV Mean Accuracy')
ax3.set_title('LinearSVC')
ax3.grid(True, alpha=0.3)

# Plot 4: SVC
ax4 = axes[1, 1]
ax4.errorbar(range(len(C_candidates)), 
                svc_results_df['cv_mean_accuracy'],
                yerr=svc_results_df['cv_std_accuracy'],
                marker='d', capsize=5, capthick=2)
ax4.set_xticks(range(len(C_candidates)))
ax4.set_xticklabels([str(c) for c in svc_results_df['C']])
ax4.set_xlabel('C')
ax4.set_ylabel('CV Mean Accuracy')
ax4.set_title('SVC')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cv_results_comparison.png', dpi=300, bbox_inches='tight')
print("📈 Visualization saved as 'cv_results_comparison.png'")
plt.show()