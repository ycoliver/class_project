# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:36:37 2025

@author: Neal
"""

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset with correlated features
X, y = make_classification(n_samples=5000, n_features=10, n_informative=5, 
                           n_redundant=3,
                           weights=[0.9, 0.1], random_state=42)

# Convert to DataFrame
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)

# fix random seed
random_seed = 12

# Manually add high correlation
np.random.seed(random_seed) 
df['feature_8'] = df['feature_0'] * 0.95 + np.random.normal(0, 0.1, len(df))
df['feature_9'] = df['feature_1'] * 0.85 + np.random.normal(0, 0.1, len(df))

# %%  Step 1: CORRECTED CODE - Split FIRST, then transform
# ++Rewritten code to fix data leaking issue++

# FIRST: Split the data BEFORE any transformation
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=random_seed)

# Method 1: Remove correlated features (threshold=0.8)
# ONLY use training data to determine which features to drop
def remove_highly_correlated(df_train, df_test, threshold):
    """Remove highly correlated features based on TRAINING data only"""
    corr_matrix = df_train.corr().abs()  # Calculate correlation ONLY on training data
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Features to drop: {to_drop}")
    # Apply the same dropping to both train and test
    return df_train.drop(columns=to_drop), df_test.drop(columns=to_drop)

# Apply removal (fit on train, transform both)
X_train_red, X_test_red = remove_highly_correlated(X_train, X_test, threshold=0.8)
print(f"Original features: {X_train.shape[1]}, Reduced features: {X_train_red.shape[1]}")

# Method 2: Apply PCA (retain components explaining 95% variance)
# FIT scaler and PCA ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
X_test_scaled = scaler.transform(X_test)        # Transform test data using training parameters

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)  # Fit on training data
X_test_pca = pca.transform(X_test_scaled)         # Transform test data using training parameters
print(f"PCA reduced to {X_train_pca.shape[1]} components (explaining {sum(pca.explained_variance_ratio_):.2%} variance)")


# %%  Step 2: Train and evaluate simple model (GaussianNB) with holdout evaluation
model = GaussianNB()

# Original (with correlations)
model.fit(X_train, y_train)
y_pred_orig = model.predict(X_test)
f1_orig = f1_score(y_test, y_pred_orig)

# After removal
model.fit(X_train_red, y_train)
y_pred_red = model.predict(X_test_red)
f1_red = f1_score(y_test, y_pred_red)

# After PCA
model.fit(X_train_pca, y_train)
y_pred_pca = model.predict(X_test_pca)
f1_pca = f1_score(y_test, y_pred_pca)

print("\n" + "="*70)
print("Model Performance (F1-Score):")
print("="*70)
print(f"Original (with correlations): {f1_orig:.4f}")
print(f"After Removing Correlated Features: {f1_red:.4f}")
print(f"After PCA Reduction: {f1_pca:.4f}")
print("="*70)

# %%  ANSWER QUESTIONS
print("\n" + "="*70)
print("ANSWERS TO QUESTIONS (random_seed = {})".format(random_seed))
print("="*70)

# Question 1
print(f"\nQuestion 1: F1 score after removing correlated features (corrected code)")
print(f"Answer: {f1_red:.4f}")

# Question 2
print(f"\nQuestion 2: Should we remove correlated features?")
f1_diff_remove = f1_red - f1_orig
print(f"F1 difference (Removed - Original): {f1_diff_remove:.4f}")
if f1_red > f1_orig:
    print("Answer: YES - F1 score improved after removing correlated features")
elif f1_red < f1_orig:
    print("Answer: NO - F1 score decreased after removing correlated features")
else:
    print("Answer: NEUTRAL - F1 score unchanged after removing correlated features")

# Question 3
print(f"\nQuestion 3: Should we proceed with PCA?")
f1_diff_pca = f1_pca - f1_orig
print(f"F1 difference (PCA - Original): {f1_diff_pca:.4f}")
if f1_pca > f1_orig:
    print("Answer: YES - F1 score improved after PCA")
elif f1_pca < f1_orig:
    print("Answer: NO - F1 score decreased after PCA")
else:
    print("Answer: NEUTRAL - F1 score unchanged after PCA")

print("="*70)

# %%  Question 4: Test with different random seeds
print("\n" + "="*70)
print("Question 4: Testing with different random_seed values")
print("="*70)

test_seeds = [9, 10, 11, 12]
results_summary = []

for seed in test_seeds:
    print(f"\n--- Testing with random_seed = {seed} ---")
    
    # Regenerate data with new seed for the added features
    np.random.seed(seed)
    df_temp = df.copy()
    df_temp['feature_8'] = df_temp['feature_0'] * 0.95 + np.random.normal(0, 0.1, len(df_temp))
    df_temp['feature_9'] = df_temp['feature_1'] * 0.85 + np.random.normal(0, 0.1, len(df_temp))
    
    # Split data
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
        df_temp, y, test_size=0.2, stratify=y, random_state=seed)
    
    # Method 1: Remove correlated features
    corr_matrix = X_train_temp.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    X_train_red_temp = X_train_temp.drop(columns=to_drop)
    X_test_red_temp = X_test_temp.drop(columns=to_drop)
    
    # Method 2: PCA
    scaler_temp = StandardScaler()
    X_train_scaled_temp = scaler_temp.fit_transform(X_train_temp)
    X_test_scaled_temp = scaler_temp.transform(X_test_temp)
    pca_temp = PCA(n_components=0.95)
    X_train_pca_temp = pca_temp.fit_transform(X_train_scaled_temp)
    X_test_pca_temp = pca_temp.transform(X_test_scaled_temp)
    
    # Evaluate
    model_temp = GaussianNB()
    
    model_temp.fit(X_train_temp, y_train_temp)
    f1_orig_temp = f1_score(y_test_temp, model_temp.predict(X_test_temp))
    
    model_temp.fit(X_train_red_temp, y_train_temp)
    f1_red_temp = f1_score(y_test_temp, model_temp.predict(X_test_red_temp))
    
    model_temp.fit(X_train_pca_temp, y_train_temp)
    f1_pca_temp = f1_score(y_test_temp, model_temp.predict(X_test_pca_temp))
    
    print(f"Original F1: {f1_orig_temp:.4f}")
    print(f"After Removal F1: {f1_red_temp:.4f} (diff: {f1_red_temp - f1_orig_temp:+.4f})")
    print(f"After PCA F1: {f1_pca_temp:.4f} (diff: {f1_pca_temp - f1_orig_temp:+.4f})")
    
    # Determine recommendations
    remove_rec = "YES" if f1_red_temp > f1_orig_temp else ("NO" if f1_red_temp < f1_orig_temp else "NEUTRAL")
    pca_rec = "YES" if f1_pca_temp > f1_orig_temp else ("NO" if f1_pca_temp < f1_orig_temp else "NEUTRAL")
    
    results_summary.append({
        'seed': seed,
        'f1_orig': f1_orig_temp,
        'f1_red': f1_red_temp,
        'f1_pca': f1_pca_temp,
        'remove_rec': remove_rec,
        'pca_rec': pca_rec
    })

# Summary table
print("\n" + "="*70)
print("SUMMARY OF RESULTS ACROSS DIFFERENT RANDOM SEEDS")
print("="*70)
print(f"{'Seed':<8} {'Original':<12} {'Removed':<12} {'PCA':<12} {'Remove?':<10} {'PCA?':<10}")
print("-"*70)
for result in results_summary:
    print(f"{result['seed']:<8} {result['f1_orig']:<12.4f} {result['f1_red']:<12.4f} "
          f"{result['f1_pca']:<12.4f} {result['remove_rec']:<10} {result['pca_rec']:<10}")

print("\n" + "="*70)
print("Question 4 Answer: Are conclusions consistent across different random_seeds?")
print("="*70)

# Check consistency
remove_recommendations = [r['remove_rec'] for r in results_summary]
pca_recommendations = [r['pca_rec'] for r in results_summary]

if len(set(remove_recommendations)) == 1:
    print(f"✓ Removing correlated features: CONSISTENT - Always {remove_recommendations[0]}")
else:
    print(f"✗ Removing correlated features: INCONSISTENT - Recommendations vary: {set(remove_recommendations)}")

if len(set(pca_recommendations)) == 1:
    print(f"✓ PCA: CONSISTENT - Always {pca_recommendations[0]}")
else:
    print(f"✗ PCA: INCONSISTENT - Recommendations vary: {set(pca_recommendations)}")

print("\nConclusion: The recommendations are", end=" ")
if len(set(remove_recommendations)) == 1 and len(set(pca_recommendations)) == 1:
    print("CONSISTENT across all tested random seeds.")
else:
    print("NOT CONSISTENT across all tested random seeds.")
print("="*70)