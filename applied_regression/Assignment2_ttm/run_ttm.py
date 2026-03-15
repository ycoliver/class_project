import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# --- Data Loading ---
url = "https://www.statlearning.com/s/Auto.csv"
auto = pd.read_csv(url).replace('?', np.nan).dropna()
auto['horsepower'] = pd.to_numeric(auto['horsepower'])

print(f"Dataset Loaded. Shape: {auto.shape}")

# --- QUESTION 8: SIMPLE LINEAR REGRESSION ---
X_simple = sm.add_constant(auto[['horsepower']])
y = auto['mpg']
model_simple = sm.OLS(y, X_simple).fit()

# Fig 1: SLR Fit
plt.figure(figsize=(9, 5))
sns.regplot(x='horsepower', y='mpg', data=auto, 
            scatter_kws={'alpha':0.4, 'edgecolor':'w'}, 
            line_kws={'color':'#e74c3c', 'lw':2.5})
plt.title('Figure 1: MPG vs Horsepower Regression Fit', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('Analysis_Fig1_SLR_Fit.png', dpi=300)

# Fig 2: SLR Diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fitted = model_simple.fittedvalues
resids = model_simple.resid

axes[0, 0].scatter(fitted, resids, alpha=0.5)
axes[0, 0].axhline(0, color='r', linestyle='--')
axes[0, 0].set(title='Residuals vs Fitted', xlabel='Fitted', ylabel='Residuals')

stats.probplot(resids, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

standardized_resids = np.sqrt(np.abs(resids / resids.std()))
axes[1, 0].scatter(fitted, standardized_resids, alpha=0.5)
axes[1, 0].set(title='Scale-Location', xlabel='Fitted', ylabel='√|Std Resids|')

leverage = model_simple.get_influence().hat_matrix_diag
axes[1, 1].scatter(leverage, resids, alpha=0.5)
axes[1, 1].axhline(0, color='r', linestyle='--')
axes[1, 1].set(title='Residuals vs Leverage', xlabel='Leverage', ylabel='Residuals')

plt.subplots_adjust(hspace=0.3, wspace=0.25)
plt.savefig('Analysis_Fig2_SLR_Diag.png', dpi=300)

# --- QUESTION 9: MULTIPLE LINEAR REGRESSION ---
# 修正 KeyError: 确保只选取数值列，且排除 mpg
numeric_cols = auto.select_dtypes(include=[np.number]).columns.tolist()

# Fig 3: Pairplot (Excluding name automatically as it is non-numeric)
g = sns.pairplot(auto[numeric_cols], diag_kind='kde', 
                 plot_kws={'alpha':0.4, 's':20, 'edgecolor':'none'})
g.fig.suptitle('Figure 3: Predictor Scatterplot Matrix', y=1.02, fontsize=14)
plt.savefig('Analysis_Fig3_Pairplot.png', dpi=300)

# Fig 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = auto[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, square=True, cbar_kws={"shrink": .8})
plt.title('Figure 4: Correlation Heatmap', fontsize=13)
plt.savefig('Analysis_Fig4_Heatmap.png', dpi=300)

# MLR Fit
X_multi = sm.add_constant(auto[numeric_cols].drop('mpg', axis=1))
model_multi = sm.OLS(auto['mpg'], X_multi).fit()
print("\nMultiple Linear Regression Summary:")
print(model_multi.summary())

# Fig 5: MLR Diagnostics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(model_multi.fittedvalues, model_multi.resid, alpha=0.5)
axes[0].axhline(0, color='r', linestyle='--')
axes[0].set(title='MLR Residuals vs Fitted', xlabel='Fitted', ylabel='Residuals')

infl = model_multi.get_influence()
axes[1].scatter(infl.hat_matrix_diag, model_multi.resid, alpha=0.5)
axes[1].set(title='MLR Leverage Plot', xlabel='Leverage', ylabel='Residuals')

plt.tight_layout()
plt.savefig('Analysis_Fig5_MLR_Diag.png', dpi=300)

# --- Statistical Testing ---
print("\n" + "="*30 + " Interaction Effects " + "="*30)
for v1, v2 in [('displacement', 'weight'), ('horsepower', 'weight')]:
    X_int = auto[[v1, v2]].copy()
    X_int['Interaction'] = auto[v1] * auto[v2]
    X_int = sm.add_constant(X_int)
    p_val = sm.OLS(auto['mpg'], X_int).fit().pvalues['Interaction']
    print(f"{v1} * {v2} p-value: {p_val:.4e}")

print("\n" + "="*30 + " Transformation Analysis " + "="*30)
log_w_X = sm.add_constant(auto[numeric_cols].drop('mpg', axis=1).assign(weight=np.log(auto['weight'])))
log_w_r2 = sm.OLS(auto['mpg'], log_w_X).fit().rsquared
print(f"Base R2: {model_multi.rsquared:.4f}")
print(f"Log(Weight) R2: {log_w_r2:.4f} (Diff: {log_w_r2 - model_multi.rsquared:.4f})")

print("\nDone. Files generated: Analysis_Fig1...Fig5")