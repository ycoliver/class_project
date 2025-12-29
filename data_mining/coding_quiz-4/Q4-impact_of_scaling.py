# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:16:24 2025

@author: Neal LONG
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline


data_dir = r'./data/bank_marketing_train.csv'

print("\n")
print("#"*30)
print('Load training data:')
df_train_raw=pd.read_csv(data_dir)
print("Raw traininig data",df_train_raw.shape)
print("\n")
print("#"*30)
print('Data clean and feature engineering:')


df_train_raw = df_train_raw.replace(to_replace={"unknown":np.nan}).infer_objects(copy=False)
df_train_raw =df_train_raw.dropna()

# %% Define the df_train_with_pdays and df_train_without_pdays
y_true = df_train_raw.pop('y')

df_train_with_pdays = df_train_raw

df_train_without_pdays = df_train_raw.drop('pdays',axis=1)

print("Shape of df_train_with_pdays is {} and df_train_without_pdays is {}".format(
    df_train_with_pdays.shape,df_train_without_pdays.shape))

#  Define the 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# %%
#++insert your code below++

# Define k values to test
k_values = [100, 200, 300, 400, 500, 600]

# Store results
results = {}

print("\n" + "="*70)
print("STARTING KNN MODEL EVALUATION WITH DIFFERENT SETTINGS")
print("="*70)

# ==================== Setting 1: With pdays + StandardScaler ====================
print("\n" + "="*70)
print("Setting 1: With pdays + StandardScaler")
print("="*70)

preprocessor_1 = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(drop='first', sparse_output=False), make_column_selector(dtype_include=object))
)

pipeline_1 = Pipeline([
    ('preprocessor', preprocessor_1),
    ('classifier', KNeighborsClassifier())
])

param_grid_1 = {'classifier__n_neighbors': k_values}

grid_search_1 = GridSearchCV(
    pipeline_1, param_grid_1, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
)

print("Running GridSearchCV...")
grid_search_1.fit(df_train_with_pdays, y_true)

results['With pdays + StandardScaler'] = {
    'best_k': grid_search_1.best_params_['classifier__n_neighbors'],
    'best_score': grid_search_1.best_score_
}

print(f"Best K: {results['With pdays + StandardScaler']['best_k']}")
print(f"Best Average AUC-ROC: {results['With pdays + StandardScaler']['best_score']:.4f}")


# ==================== Setting 2: With pdays + Normalizer ====================
print("\n" + "="*70)
print("Setting 2: With pdays + Normalizer")
print("="*70)

preprocessor_2 = make_column_transformer(
    (Normalizer(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(drop='first', sparse_output=False), make_column_selector(dtype_include=object))
)

pipeline_2 = Pipeline([
    ('preprocessor', preprocessor_2),
    ('classifier', KNeighborsClassifier())
])

param_grid_2 = {'classifier__n_neighbors': k_values}

grid_search_2 = GridSearchCV(
    pipeline_2, param_grid_2, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
)

print("Running GridSearchCV...")
grid_search_2.fit(df_train_with_pdays, y_true)

results['With pdays + Normalizer'] = {
    'best_k': grid_search_2.best_params_['classifier__n_neighbors'],
    'best_score': grid_search_2.best_score_
}

print(f"Best K: {results['With pdays + Normalizer']['best_k']}")
print(f"Best Average AUC-ROC: {results['With pdays + Normalizer']['best_score']:.4f}")


# ==================== Setting 3: Without pdays + StandardScaler ====================
print("\n" + "="*70)
print("Setting 3: Without pdays + StandardScaler")
print("="*70)

preprocessor_3 = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(drop='first', sparse_output=False), make_column_selector(dtype_include=object))
)

pipeline_3 = Pipeline([
    ('preprocessor', preprocessor_3),
    ('classifier', KNeighborsClassifier())
])

param_grid_3 = {'classifier__n_neighbors': k_values}

grid_search_3 = GridSearchCV(
    pipeline_3, param_grid_3, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
)

print("Running GridSearchCV...")
grid_search_3.fit(df_train_without_pdays, y_true)

results['Without pdays + StandardScaler'] = {
    'best_k': grid_search_3.best_params_['classifier__n_neighbors'],
    'best_score': grid_search_3.best_score_
}

print(f"Best K: {results['Without pdays + StandardScaler']['best_k']}")
print(f"Best Average AUC-ROC: {results['Without pdays + StandardScaler']['best_score']:.4f}")


# ==================== Setting 4: Without pdays + Normalizer ====================
print("\n" + "="*70)
print("Setting 4: Without pdays + Normalizer")
print("="*70)

preprocessor_4 = make_column_transformer(
    (Normalizer(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(drop='first', sparse_output=False), make_column_selector(dtype_include=object))
)

pipeline_4 = Pipeline([
    ('preprocessor', preprocessor_4),
    ('classifier', KNeighborsClassifier())
])

param_grid_4 = {'classifier__n_neighbors': k_values}

grid_search_4 = GridSearchCV(
    pipeline_4, param_grid_4, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
)

print("Running GridSearchCV...")
grid_search_4.fit(df_train_without_pdays, y_true)

results['Without pdays + Normalizer'] = {
    'best_k': grid_search_4.best_params_['classifier__n_neighbors'],
    'best_score': grid_search_4.best_score_
}

print(f"Best K: {results['Without pdays + Normalizer']['best_k']}")
print(f"Best Average AUC-ROC: {results['Without pdays + Normalizer']['best_score']:.4f}")


# ==================== Summary and Analysis ====================
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)
print(f"{'Setting':<40} {'Best K':<10} {'AUC-ROC':<10}")
print("-"*70)
for setting, result in results.items():
    print(f"{setting:<40} {result['best_k']:<10} {result['best_score']:.4f}")

# Find best setting
best_setting = max(results.items(), key=lambda x: x[1]['best_score'])

print("\n" + "="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

# Question 1: Best setting
print(f"\nQuestion 1: What is the best setting that yields the highest average AUC-ROC score?")
print(f"Answer: {best_setting[0]}")
print(f"  Best K: {best_setting[1]['best_k']}")
print(f"  Best AUC-ROC: {best_setting[1]['best_score']:.4f}")

# Question 2: Score for "Without pdays + Normalizer"
print(f"\nQuestion 2: Average AUC-ROC score for 'Without pdays + Normalizer'?")
score_q2 = results['Without pdays + Normalizer']['best_score']
print(f"Answer: {score_q2:.2f}")

# Question 3: Which scaler is better?
print(f"\nQuestion 3: Which rescaling method produces better results?")

# Compare with pdays
with_pdays_standard = results['With pdays + StandardScaler']['best_score']
with_pdays_normalizer = results['With pdays + Normalizer']['best_score']

# Compare without pdays
without_pdays_standard = results['Without pdays + StandardScaler']['best_score']
without_pdays_normalizer = results['Without pdays + Normalizer']['best_score']

print(f"  With pdays:")
print(f"    StandardScaler: {with_pdays_standard:.4f}")
print(f"    Normalizer:     {with_pdays_normalizer:.4f}")
print(f"    Winner: {'StandardScaler' if with_pdays_standard > with_pdays_normalizer else 'Normalizer'}")

print(f"  Without pdays:")
print(f"    StandardScaler: {without_pdays_standard:.4f}")
print(f"    Normalizer:     {without_pdays_normalizer:.4f}")
print(f"    Winner: {'StandardScaler' if without_pdays_standard > without_pdays_normalizer else 'Normalizer'}")

# Determine which is better overall
if with_pdays_standard > with_pdays_normalizer and without_pdays_standard > without_pdays_normalizer:
    scaler_answer = "StandardScaler"
    explanation = "StandardScaler performs better in BOTH settings"
elif with_pdays_normalizer > with_pdays_standard and without_pdays_normalizer > without_pdays_standard:
    scaler_answer = "Normalizer"
    explanation = "Normalizer performs better in BOTH settings"
else:
    scaler_answer = "It depends"
    explanation = "Different scalers perform better in different settings"

print(f"\nAnswer: {scaler_answer}")
print(f"Explanation: {explanation}")

# Question 4: Is pdays essential?
print(f"\nQuestion 4: Is including the feature 'pdays' essential for better performance?")

# Compare StandardScaler settings
standard_with = results['With pdays + StandardScaler']['best_score']
standard_without = results['Without pdays + StandardScaler']['best_score']

# Compare Normalizer settings
normalizer_with = results['With pdays + Normalizer']['best_score']
normalizer_without = results['Without pdays + Normalizer']['best_score']

print(f"  StandardScaler:")
print(f"    With pdays:    {standard_with:.4f}")
print(f"    Without pdays: {standard_without:.4f}")
print(f"    Better: {'With pdays' if standard_with > standard_without else 'Without pdays'}")

print(f"  Normalizer:")
print(f"    With pdays:    {normalizer_with:.4f}")
print(f"    Without pdays: {normalizer_without:.4f}")
print(f"    Better: {'With pdays' if normalizer_with > normalizer_without else 'Without pdays'}")

# Determine if pdays is essential
if standard_with > standard_without and normalizer_with > normalizer_without:
    pdays_answer = "Yes"
    explanation = "Including pdays improves performance in BOTH scaler settings"
elif standard_without > standard_with and normalizer_without > normalizer_with:
    pdays_answer = "No"
    explanation = "Excluding pdays improves performance in BOTH scaler settings"
else:
    pdays_answer = "It depends"
    explanation = "pdays helps in one setting but hurts in another"

print(f"\nAnswer: {pdays_answer}")
print(f"Explanation: {explanation}")

print("\n" + "="*70)
print("FINAL ANSWERS FOR ANSWER BOOK")
print("="*70)
print(f"Q1: {best_setting[0]}")
print(f"Q2: {score_q2:.2f}")
print(f"Q3: {scaler_answer}")
print(f"Q4: {pdays_answer}")
print("="*70)