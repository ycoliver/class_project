import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

print("=== 开始执行 Task-1 ===")

TRAIN_FILE = './bank_marketing_train.csv'
TEST_FILE = './bank_marketing_test.csv'
EXAMPLE_FILE = './bank_marketing_test_scores(example).csv'
OUTPUT_FILE = './bank_marketing_test_scores.csv'
ROC_FILE = './roc_curve.png'

for file in [TRAIN_FILE, TEST_FILE]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"错误: 未找到文件 '{file}'，请确保它与脚本在同一目录下。")

print(f"正在加载数据...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\n=== 数据清洗与预处理 ===")

numeric_features = [
    'age', 'campaign', 'pdays', 'previous', 
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
    'euribor3m', 'nr.employed', 
    'feature_1', 'feature_2'
]

categorical_features = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 
    'contact', 'month', 'day_of_week', 'poutcome', 
    'feature_3', 'feature_4', 'feature_5'
]

def clean_data(df, num_cols):
    df_clean = df.copy()
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_clean

print("正在清洗数值列中的异常值和无穷大值...")
train_df = clean_data(train_df, numeric_features)
test_df = clean_data(test_df, numeric_features)

X = train_df.drop('y', axis=1)
y_raw = train_df['y']
X_test = test_df.copy()

le = LabelEncoder()
y = le.fit_transform(y_raw)
if 'yes' in le.classes_:
    pos_index = list(le.classes_).index('yes')
    if pos_index != 1:
        y = 1 - y
print(f"目标变量分布: Yes(1)={sum(y)}, No(0)={len(y)-sum(y)}")

missing_cols = set(numeric_features + categorical_features) - set(X.columns)
if missing_cols:
    raise ValueError(f"缺少列: {missing_cols}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

print("\n=== 执行 5-Fold Cross-Validation 并绘制 ROC 曲线 ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fig, ax = plt.subplots(figsize=(8, 6))

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    clf.fit(X_train_fold, y_train_fold)
    y_pred_proba_fold = clf.predict_proba(X_val_fold)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_val_fold, y_pred_proba_fold)
    auc_score = roc_auc_score(y_val_fold, y_pred_proba_fold)
    aucs.append(auc_score)
    
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    
    ax.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold+1} (AUC = {auc_score:.4f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, 'b-', linewidth=2,
        label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.2,
                label='± 1 Std. Dev.')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - 5-Fold Cross Validation', fontsize=14)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ROC_FILE, dpi=150)
plt.close()

print(f"-"*30)
print(f"Mean AUC Score: {mean_auc:.5f} ± {std_auc:.5f}")
print(f"ROC曲线已保存至: {os.path.abspath(ROC_FILE)}")
print(f"-"*30)

print("\n=== 生成测试集预测结果 ===")
clf.fit(X, y)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

has_header = False
try:
    if os.path.exists(EXAMPLE_FILE):
        with open(EXAMPLE_FILE, 'r') as f:
            first_line = f.readline().strip()
            try:
                float(first_line)
                has_header = False
            except ValueError:
                has_header = True
except:
    pass

submission = pd.DataFrame(y_pred_proba)
if has_header:
    submission.columns = ["ranking_score"]
    submission.to_csv(OUTPUT_FILE, index=False)
else:
    submission.to_csv(OUTPUT_FILE, index=False, header=False)

print(f"成功! 结果已保存至: {os.path.abspath(OUTPUT_FILE)}")