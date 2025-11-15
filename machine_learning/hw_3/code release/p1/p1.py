import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

df_train = pd.read_csv('./training_data.csv')
x_train = df_train['x'].values
y_train = df_train['y'].values

df_test = pd.read_csv('./test_data.csv')
x_test = df_test['x_test'].values
y_test = df_test['y_test'].values

def create_polynomial_features(x, degree=8):
    X = np.zeros((len(x), degree + 1))
    for i in range(degree + 1):
        X[:, i] = x ** i
    return X

X_train = create_polynomial_features(x_train, 8)
y_train_vec = y_train.reshape(-1, 1)

X_test = create_polynomial_features(x_test, 8)
y_test_vec = y_test.reshape(-1, 1)

theta_hat = np.linalg.lstsq(X_train, y_train_vec, rcond=None)[0]

xx = np.arange(-1.5, 1.55, 0.05)
yy_true = xx**2
X_plot = create_polynomial_features(xx, 8)
y_pred = X_plot @ theta_hat

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x_train, y_train, label='data', 
           facecolors='none', edgecolors='#2E8CC9', linewidths=1.5)
ax.plot(xx, yy_true, linewidth=2, color='#D95319', label='target function')
ax.plot(xx, y_pred, linewidth=2, color='green', label='fitted curve')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-0.5, 2.5])
ax.set_xticks([-1, 0, 1])
legend_font_properties = {'family': 'Times New Roman', 'size': 20}
ax.legend(prop=legend_font_properties, edgecolor='black')
fig.set_facecolor('white')
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(20)
ax.tick_params(direction='in', width=1.5, length=6)
ax.set_xlabel('x', fontsize=25, fontname='Times New Roman')
ax.set_ylabel('y', fontsize=25, fontname='Times New Roman')
plt.tight_layout()
fig.savefig('./a2_fitted_curve.pdf', dpi=300, bbox_inches='tight')
plt.close()

test_error_a3 = np.linalg.norm(X_test @ theta_hat - y_test_vec, 2)
print(f"Test error (a3): {test_error_a3}")

lambda_candidates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 0.5, 0.8, 1, 2, 5, 10, 15, 20, 50, 100]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

validation_errors = []

for lam in lambda_candidates:
    fold_errors = []
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train_vec[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train_vec[val_idx]
        
        theta_reg = np.linalg.solve(X_fold_train.T @ X_fold_train + lam * np.eye(9), 
                                     X_fold_train.T @ y_fold_train)
        
        val_error = np.linalg.norm(X_fold_val @ theta_reg - y_fold_val, 2)
        fold_errors.append(val_error)
    
    validation_errors.append(np.mean(fold_errors))

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(lambda_candidates, validation_errors, linewidth=2, marker='o', color='#2E8CC9')
ax.set_xscale('log')
ax.set_xlabel('λ', fontsize=25, fontname='Times New Roman')
ax.set_ylabel('Validation Error', fontsize=25, fontname='Times New Roman')
fig.set_facecolor('white')
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(20)
ax.tick_params(direction='in', width=1.5, length=6)
plt.tight_layout()
fig.savefig('./b1_validation_error.pdf', dpi=300, bbox_inches='tight')
plt.close()

lambda_values = [0.01, 0.1, 0.8, 5]

for lam in lambda_values:
    theta_reg = np.linalg.solve(X_train.T @ X_train + lam * np.eye(9), 
                                X_train.T @ y_train_vec)
    
    y_pred_reg = X_plot @ theta_reg
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_train, y_train, label='data', 
               facecolors='none', edgecolors='#2E8CC9', linewidths=1.5)
    ax.plot(xx, yy_true, linewidth=2, color='#D95319', label='target function')
    ax.plot(xx, y_pred_reg, linewidth=2, color='green', label=f'fitted curve (λ={lam})')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-0.5, 2.5])
    ax.set_xticks([-1, 0, 1])
    legend_font_properties = {'family': 'Times New Roman', 'size': 20}
    ax.legend(prop=legend_font_properties, edgecolor='black')
    fig.set_facecolor('white')
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(20)
    ax.tick_params(direction='in', width=1.5, length=6)
    ax.set_xlabel('x', fontsize=25, fontname='Times New Roman')
    ax.set_ylabel('y', fontsize=25, fontname='Times New Roman')
    plt.tight_layout()
    fig.savefig(f'./b2_lambda_{lam}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    test_error = np.linalg.norm(X_test @ theta_reg - y_test_vec, 2)
    print(f"Test error for lambda={lam}: {test_error}")

### Final Output
'''
Test error (a3): 4.56073889670133
Test error for lambda=0.01: 0.6633900485856986
Test error for lambda=0.1: 0.6341378045467686
Test error for lambda=0.8: 0.6492886142936496
Test error for lambda=5: 0.8206567071942981
'''