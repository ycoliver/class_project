# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:51:34 2024
@author: Neal
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#load the dataset
df = pd.read_csv('./data/diabetes_data.csv')
X = df.drop(columns = ['diabetes'])
y = df['diabetes']

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
stratify=y, random_state=1)

#++insert your code below ++

# 1.1 Decision Tree Classifier with GridSearchCV
dt = DecisionTreeClassifier(random_state=0)
dt_params = {'max_depth': [1, 5, 15, 20, 25]}
dt_grid = GridSearchCV(dt, dt_params, scoring='accuracy', cv=5)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_

# 1.2 Random Forest Classifier with GridSearchCV
rf = RandomForestClassifier(random_state=0)
rf_params = {'n_estimators': [50, 100, 200]}
rf_grid = GridSearchCV(rf, rf_params, scoring='accuracy', cv=5)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

# 1.3 Gaussian Naive Bayes with default parameters
nb = GaussianNB()
nb.fit(X_train, y_train)

# 2.1 Soft Voting Classifier
soft_voting_clf = VotingClassifier(
    estimators=[('dt', dt_best), ('rf', rf_best), ('nb', nb)],
    voting='soft'
)
soft_voting_clf.fit(X_train, y_train)

# 2.2 Hard Voting Classifier
hard_voting_clf = VotingClassifier(
    estimators=[('dt', dt_best), ('rf', rf_best), ('nb', nb)],
    voting='hard'
)
hard_voting_clf.fit(X_train, y_train)

# 3. Evaluate all classifiers on hold-out test data
dt_acc = accuracy_score(y_test, dt_best.predict(X_test))
rf_acc = accuracy_score(y_test, rf_best.predict(X_test))
nb_acc = accuracy_score(y_test, nb.predict(X_test))
soft_acc = accuracy_score(y_test, soft_voting_clf.predict(X_test))
hard_acc = accuracy_score(y_test, hard_voting_clf.predict(X_test))# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:51:34 2024
@author: Neal
1. Build, optimize and train 3 base classifiers based on the training data
   (X_train, y_train) as below:
    1.1 dt classifier: the best DecisionTreeClassifier with random_state = 0, 
              and with best value of 'max_depth' in [1 ,5, 15, 20, 25], which is 
              selected by GridSearchCV with accuracy score and 5-fold CV
    1.2 rf classifier: the best RandomForestClassifier with random_state = 0, 
              and with best value of 'n_estimators' in [50, 100, 200], which is 
              selected by GridSearchCV with accuracy score and 5-fold CV
    1.3 nb classifier:  GaussianNB with default parameter settings
2. Build 2 ensemble learning models with above 3 base classifiers,  and train
   them on the training data (X_train, y_train) as below :
    2.1 soft voting classifier: VotingClassifier with 'soft' voting
    2.2 hard voting classifier: VotingClassifier with 'hard' voting
3. Evaluate and compare the accuracy score of above 3 base classifiers and 
   2 ensemble/voting classifiers on the hold-out test data, (X_test, y_test),
   and answer questions accordingly
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#load the dataset
df = pd.read_csv('./data/diabetes_data.csv')
X = df.drop(columns = ['diabetes'])
y = df['diabetes']
#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
stratify=y, random_state=1)

#++insert your code below ++ to build/optimize different models on 
# training data (X_train, y_train) as required, and then evaluate their 
# performance (accuracy score) on the hold-out test data, (X_test, y_test)

# ============================================================================
# 1.1 Decision Tree Classifier with GridSearchCV
# ============================================================================
dt = DecisionTreeClassifier(random_state=0)
dt_params = {'max_depth': [1, 5, 15, 20, 25]}
dt_grid = GridSearchCV(dt, dt_params, scoring='accuracy', cv=5)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_
print("DT best max_depth:", dt_grid.best_params_['max_depth'])

# ============================================================================
# 1.2 Random Forest Classifier with GridSearchCV
# ============================================================================
rf = RandomForestClassifier(random_state=0)
rf_params = {'n_estimators': [50, 100, 200]}
rf_grid = GridSearchCV(rf, rf_params, scoring='accuracy', cv=5)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
print("RF best n_estimators:", rf_grid.best_params_['n_estimators'])

# ============================================================================
# 1.3 Gaussian Naive Bayes Classifier
# ============================================================================
nb = GaussianNB()
nb.fit(X_train, y_train)

# ============================================================================
# 2.1 Soft Voting Classifier
# ============================================================================
soft_voting = VotingClassifier(
    estimators=[('dt', dt_best), ('rf', rf_best), ('nb', nb)],
    voting='soft'
)
soft_voting.fit(X_train, y_train)

# ============================================================================
# 2.2 Hard Voting Classifier
# ============================================================================
hard_voting = VotingClassifier(
    estimators=[('dt', dt_best), ('rf', rf_best), ('nb', nb)],
    voting='hard'
)
hard_voting.fit(X_train, y_train)

# ============================================================================
# 3. Evaluate all classifiers on test data
# ============================================================================
dt_acc = accuracy_score(y_test, dt_best.predict(X_test))
rf_acc = accuracy_score(y_test, rf_best.predict(X_test))
nb_acc = accuracy_score(y_test, nb.predict(X_test))
soft_acc = accuracy_score(y_test, soft_voting.predict(X_test))
hard_acc = accuracy_score(y_test, hard_voting.predict(X_test))

print("\n" + "="*60)
print("Accuracy Scores on Hold-out Test Data:")
print("="*60)
print(f"dt classifier accuracy: {round(dt_acc, 4)}")
print(f"rf classifier accuracy: {round(rf_acc, 4)}")
print(f"nb classifier accuracy: {round(nb_acc, 4)}")
print(f"soft voting classifier accuracy: {round(soft_acc, 4)}")
print(f"hard voting classifier accuracy: {round(hard_acc, 4)}")

# ============================================================================
# Answer the questions
# ============================================================================
print("\n" + "="*60)
print("Answers to Questions:")
print("="*60)

# Q2-1: Which base classifier demonstrates the highest performance?
base_classifiers = {'dt classifier': dt_acc, 'rf classifier': rf_acc, 'nb classifier': nb_acc}
best_base = max(base_classifiers, key=base_classifiers.get)
print(f"\nQ2-1: Which base classifier demonstrates the highest performance on the hold-out test data?")
print(f"Answer: {best_base}")

# Q2-2: Which ensemble classifier yields the best performance?
ensemble_classifiers = {'soft voting classifier': soft_acc, 'hard voting classifier': hard_acc}
best_ensemble = max(ensemble_classifiers, key=ensemble_classifiers.get)
print(f"\nQ2-2: Which ensemble classifier yields the best performance on the hold-out test data?")
print(f"Answer: {best_ensemble}")

# Q2-3: Can the best ensemble classifier outperform the best base classifier?
best_base_acc = max(base_classifiers.values())
best_ensemble_acc = max(ensemble_classifiers.values())
outperform = "Yes" if best_ensemble_acc > best_base_acc else "No"
print(f"\nQ2-3: Can the best ensemble classifier outperform the best base classifiers?")
print(f"Answer: {outperform}")
print(f"(Best base accuracy: {round(best_base_acc, 4)}, Best ensemble accuracy: {round(best_ensemble_acc, 4)})")
