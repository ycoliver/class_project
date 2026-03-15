import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("MDS5202 Assignment 2 - Python Analysis")
print("="*70)

# Load Auto dataset
# You can download from: https://www.statlearning.com/s/Auto.csv
url = "https://www.statlearning.com/s/Auto.csv"
auto = pd.read_csv(url)

# Remove missing values
auto = auto.replace('?', np.nan)
auto = auto.dropna()

# Convert horsepower to numeric (it may be string in some versions)
auto['horsepower'] = pd.to_numeric(auto['horsepower'])

print("\nDataset loaded successfully!")
print(f"Dataset shape: {auto.shape}")
print("\nFirst few rows:")
print(auto.head())

print("\n" + "="*70)
print("QUESTION 8: SIMPLE LINEAR REGRESSION")
print("="*70)

# Question 8(a): Simple Linear Regression
print("\n(a) Simple Linear Regression: mpg ~ horsepower")
print("-" * 50)

X_simple = auto[['horsepower']]
y = auto['mpg']

# Add constant for intercept
X_simple_const = sm.add_constant(X_simple)

# Fit the model
model_simple = sm.OLS(y, X_simple_const).fit()

print(model_simple.summary())

print("\n" + "="*50)
print("INTERPRETATIONS:")
print("="*50)

# i. Is there a relationship?
p_value_hp = model_simple.pvalues['horsepower']
print(f"\ni. Is there a relationship between horsepower and mpg?")
print(f"   p-value = {p_value_hp:.4e}")
if p_value_hp < 0.05:
    print(f"   YES - The p-value < 0.05 indicates a statistically significant relationship.")
else:
    print(f"   NO - The p-value >= 0.05 indicates no significant relationship.")

# ii. How strong is the relationship?
r_squared = model_simple.rsquared
print(f"\nii. How strong is the relationship?")
print(f"    R-squared = {r_squared:.4f}")
print(f"    Horsepower explains {r_squared*100:.2f}% of the variance in mpg.")
if r_squared > 0.6:
    print(f"    This is a STRONG relationship.")
elif r_squared > 0.3:
    print(f"    This is a MODERATE relationship.")
else:
    print(f"    This is a WEAK relationship.")

# iii. Positive or negative?
coef_hp = model_simple.params['horsepower']
print(f"\niii. Is the relationship positive or negative?")
print(f"     Coefficient = {coef_hp:.4f}")
if coef_hp > 0:
    print(f"     POSITIVE - As horsepower increases, mpg increases.")
else:
    print(f"     NEGATIVE - As horsepower increases, mpg decreases.")

# iv. Prediction for horsepower = 98
print(f"\niv. Prediction for horsepower = 98:")
new_data = pd.DataFrame({'const': [1], 'horsepower': [98]})
prediction = model_simple.predict(new_data)
pred_interval = model_simple.get_prediction(new_data).summary_frame(alpha=0.05)

print(f"    Predicted mpg: {prediction.values[0]:.2f}")
print(f"    95% Confidence Interval: [{pred_interval['mean_ci_lower'].values[0]:.2f}, {pred_interval['mean_ci_upper'].values[0]:.2f}]")
print(f"    95% Prediction Interval: [{pred_interval['obs_ci_lower'].values[0]:.2f}, {pred_interval['obs_ci_upper'].values[0]:.2f}]")

# Question 8(b): Plot
print("\n(b) Creating scatter plot with regression line...")
plt.figure(figsize=(10, 6))
plt.scatter(auto['horsepower'], auto['mpg'], alpha=0.5, edgecolors='k')
plt.plot(auto['horsepower'], model_simple.predict(X_simple_const), 
         color='red', linewidth=2, label='Regression Line')
plt.xlabel('Horsepower', fontsize=12)
plt.ylabel('MPG', fontsize=12)
plt.title('Simple Linear Regression: MPG vs Horsepower', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q8b_regression_plot.png', dpi=300, bbox_inches='tight')
print("    Saved as 'q8b_regression_plot.png'")

# Question 8(c): Diagnostic plots
print("\n(c) Creating diagnostic plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
fitted_values = model_simple.fittedvalues
residuals = model_simple.resid
axes[0, 0].scatter(fitted_values, residuals, alpha=0.5, edgecolors='k')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Scale-Location plot
standardized_residuals = np.sqrt(np.abs(residuals / residuals.std()))
axes[1, 0].scatter(fitted_values, standardized_residuals, alpha=0.5, edgecolors='k')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')
axes[1, 0].set_title('Scale-Location Plot')

# Residuals vs Leverage
influence = model_simple.get_influence()
leverage = influence.hat_matrix_diag
axes[1, 1].scatter(leverage, residuals, alpha=0.5, edgecolors='k')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.savefig('q8c_diagnostic_plots.png', dpi=300, bbox_inches='tight')
print("    Saved as 'q8c_diagnostic_plots.png'")

print("\nDiagnostic Analysis:")
print("- Residuals vs Fitted: Check for non-linear patterns")
print("- Q-Q Plot: Check for normality of residuals")
print("- Scale-Location: Check for homoscedasticity")
print("- Residuals vs Leverage: Identify influential points")

print("\n" + "="*70)
print("QUESTION 9: MULTIPLE LINEAR REGRESSION")
print("="*70)

# Question 9(a): Scatterplot matrix
print("\n(a) Creating scatterplot matrix...")
numeric_cols = auto.select_dtypes(include=[np.number]).columns.tolist()
if 'name' in numeric_cols:
    numeric_cols.remove('name')

fig = plt.figure(figsize=(14, 14))
pd.plotting.scatter_matrix(auto[numeric_cols], alpha=0.5, figsize=(14, 14), 
                          diagonal='kde', edgecolors='k')
plt.suptitle('Scatterplot Matrix of Auto Dataset', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('q9a_scatterplot_matrix.png', dpi=300, bbox_inches='tight')
print("    Saved as 'q9a_scatterplot_matrix.png'")

# Question 9(b): Correlation matrix
print("\n(b) Correlation Matrix:")
print("-" * 50)
corr_matrix = auto[numeric_cols].corr()
print(corr_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q9b_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n    Saved as 'q9b_correlation_heatmap.png'")

# Question 9(c): Multiple linear regression
print("\n(c) Multiple Linear Regression: mpg ~ all variables (except name)")
print("-" * 50)

# Prepare features
feature_cols = [col for col in numeric_cols if col != 'mpg']
X_multi = auto[feature_cols]
y = auto['mpg']

# Add constant
X_multi_const = sm.add_constant(X_multi)

# Fit model
model_multi = sm.OLS(y, X_multi_const).fit()
print(model_multi.summary())

print("\nKey Findings:")
print(f"R-squared: {model_multi.rsquared:.4f}")
print(f"Adjusted R-squared: {model_multi.rsquared_adj:.4f}")
print("\nStatistically significant predictors (p < 0.05):")
for var, pval in model_multi.pvalues.items():
    if pval < 0.05 and var != 'const':
        print(f"  - {var}: p-value = {pval:.4f}")

# Question 9(d): Diagnostic plots for multiple regression
print("\n(d) Creating diagnostic plots for multiple regression...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

fitted_multi = model_multi.fittedvalues
resid_multi = model_multi.resid

# Residuals vs Fitted
axes[0, 0].scatter(fitted_multi, resid_multi, alpha=0.5, edgecolors='k')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(resid_multi, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Scale-Location
std_resid_multi = np.sqrt(np.abs(resid_multi / resid_multi.std()))
axes[1, 0].scatter(fitted_multi, std_resid_multi, alpha=0.5, edgecolors='k')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')
axes[1, 0].set_title('Scale-Location Plot')

# Residuals vs Leverage
influence_multi = model_multi.get_influence()
leverage_multi = influence_multi.hat_matrix_diag
axes[1, 1].scatter(leverage_multi, resid_multi, alpha=0.5, edgecolors='k')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Leverage')

# Identify high leverage points
high_leverage_idx = np.where(leverage_multi > 3 * leverage_multi.mean())[0]
print(f"\nHigh leverage points (leverage > 3 * mean): {high_leverage_idx.tolist()}")

plt.tight_layout()
plt.savefig('q9d_diagnostic_plots_multi.png', dpi=300, bbox_inches='tight')
print("    Saved as 'q9d_diagnostic_plots_multi.png'")

# Question 9(e): Interaction effects
print("\n(e) Testing Interaction Effects:")
print("-" * 50)

# Test a few key interactions
interactions_to_test = [
    ('displacement', 'weight'),
    ('horsepower', 'weight'),
    ('acceleration', 'year')
]

interaction_results = []
for var1, var2 in interactions_to_test:
    X_interact = auto[[var1, var2]].copy()
    X_interact[f'{var1}:{var2}'] = auto[var1] * auto[var2]
    X_interact = sm.add_constant(X_interact)
    
    model_interact = sm.OLS(auto['mpg'], X_interact).fit()
    interaction_coef = model_interact.params[f'{var1}:{var2}']
    interaction_pval = model_interact.pvalues[f'{var1}:{var2}']
    
    print(f"\n{var1} × {var2}:")
    print(f"  Coefficient: {interaction_coef:.6f}")
    print(f"  p-value: {interaction_pval:.4f}")
    
    if interaction_pval < 0.05:
        print(f"  *** SIGNIFICANT at α = 0.05")
        interaction_results.append((var1, var2, interaction_pval))

print("\nSignificant interactions found:")
for var1, var2, pval in interaction_results:
    print(f"  - {var1} × {var2} (p = {pval:.4f})")

# Question 9(f): Variable transformations
print("\n(f) Testing Variable Transformations:")
print("-" * 50)

transformations = {
    'log(horsepower)': np.log(auto['horsepower']),
    'sqrt(displacement)': np.sqrt(auto['displacement']),
    'weight^2': auto['weight'] ** 2,
    'log(weight)': np.log(auto['weight'])
}

print("\nOriginal model R-squared:", model_multi.rsquared.round(4))
print("\nTesting transformations:")

best_r2 = model_multi.rsquared
best_transform = "None"

for transform_name, transform_data in transformations.items():
    # Create new feature set with transformation
    X_transform = auto[feature_cols].copy()
    
    # Replace the original variable with transformed version
    var_name = transform_name.split('(')[1].split(')')[0] if '(' in transform_name else transform_name.split('^')[0]
    
    if var_name in X_transform.columns:
        X_transform[var_name] = transform_data
    
    X_transform_const = sm.add_constant(X_transform)
    model_transform = sm.OLS(auto['mpg'], X_transform_const).fit()
    
    print(f"\n{transform_name}:")
    print(f"  R-squared: {model_transform.rsquared:.4f}")
    print(f"  Improvement: {(model_transform.rsquared - model_multi.rsquared):.4f}")
    
    if model_transform.rsquared > best_r2:
        best_r2 = model_transform.rsquared
        best_transform = transform_name

print(f"\n{'='*50}")
print(f"Best transformation: {best_transform}")
print(f"R-squared: {best_r2:.4f}")
print(f"{'='*50}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nAll plots have been saved as PNG files.")
print("Results are ready to be included in the LaTeX report.")
print("\nGenerated files:")
print("  - q8b_regression_plot.png")
print("  - q8c_diagnostic_plots.png")
print("  - q9a_scatterplot_matrix.png")
print("  - q9b_correlation_heatmap.png")
print("  - q9d_diagnostic_plots_multi.png")