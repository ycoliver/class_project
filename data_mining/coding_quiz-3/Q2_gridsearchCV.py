from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

gamma_candidates = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
C_candidates = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,1000]

param_grid = {
    'C': C_candidates,
    'gamma': gamma_candidates
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

grid_search = GridSearchCV(
    SVC(random_state=0),
    param_grid=param_grid,
    cv=cv,
    scoring='recall_micro'
)

grid_search.fit(X, y)

print("Best C:", grid_search.best_params_['C'])
print("Best gamma:", grid_search.best_params_['gamma'])
print("Best cross-validation score:", grid_search.best_score_)