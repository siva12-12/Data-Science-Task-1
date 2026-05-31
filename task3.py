# ============================================================
#   Task 4: Sales Prediction using Python
#   CodeAlpha Data Science Internship
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────
df = pd.read_csv(r'D:\Disk\intern\Advertising.csv', index_col=0)
print(f'Shape: {df.shape}')
print(df.head())

# ── 2. CHECK DATA ─────────────────────────────────────────
print('\nMissing Values:\n', df.isnull().sum())
print('\nBasic Stats:\n', df.describe())

# ── 3. VISUALIZATIONS ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sales Distribution
sns.histplot(df['Sales'], bins=20, kde=True, color='steelblue', ax=axes[0][0])
axes[0][0].set_title('Sales Distribution')
axes[0][0].set_xlabel('Sales (units)')

# Plot 2: Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[0][1])
axes[0][1].set_title('Correlation Heatmap')

# Plot 3: Ad Spend vs Sales (all 3 channels)
for col, color in zip(['TV', 'Radio', 'Newspaper'], ['tomato', 'green', 'purple']):
    axes[1][0].scatter(df[col], df['Sales'], alpha=0.5, label=col, color=color)
axes[1][0].set_title('Ad Spend vs Sales')
axes[1][0].set_xlabel('Ad Spend (thousands)')
axes[1][0].set_ylabel('Sales')
axes[1][0].legend()

# Plot 4: Boxplot of ad channels
df[['TV', 'Radio', 'Newspaper']].plot(kind='box', ax=axes[1][1], patch_artist=True)
axes[1][1].set_title('Ad Budget Distribution by Channel')
axes[1][1].set_ylabel('Budget (thousands)')

plt.suptitle('Sales Prediction - EDA', fontsize=16)
plt.tight_layout()
plt.show()

# ── 4. SPLIT DATA ─────────────────────────────────────────
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# ── 5. TRAIN MODELS ───────────────────────────────────────
models = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree'     : DecisionTreeRegressor(random_state=42),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
print('\n=== Model Comparison ===')
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model': model,
        'R2'   : r2_score(y_test, y_pred),
        'MAE'  : mean_absolute_error(y_test, y_pred),
        'RMSE' : np.sqrt(mean_squared_error(y_test, y_pred))
    }
    print(f'{name:22s} → R²: {results[name]["R2"]:.4f} | MAE: {results[name]["MAE"]:.2f} | RMSE: {results[name]["RMSE"]:.2f}')

# ── 6. BEST MODEL EVALUATION ──────────────────────────────
best_name  = max(results, key=lambda x: results[x]['R2'])
best_model = results[best_name]['model']
y_pred_best = best_model.predict(X_test)

print(f'\n🏆 Best Model : {best_name}')
print(f'   R² Score   : {results[best_name]["R2"]:.4f}')
print(f'   MAE        : {results[best_name]["MAE"]:.2f}')
print(f'   RMSE       : {results[best_name]["RMSE"]:.2f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.6, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Predicted Sales')
axes[0].set_title(f'Actual vs Predicted — {best_name}')

# Feature Importance (Random Forest)
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=['TV', 'Radio', 'Newspaper'])
importances.sort_values().plot(kind='barh', ax=axes[1], color=['purple', 'green', 'tomato'])
axes[1].set_title('Feature Importance — Random Forest')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.show()

# ── 7. PREDICT CUSTOM AD SPEND ────────────────────────────
# [TV_budget, Radio_budget, Newspaper_budget] in thousands
new_ad = np.array([[230.1, 37.8, 69.2]])
predicted_sales = best_model.predict(new_ad)
print(f'\nPredicted Sales: {predicted_sales[0]:.2f} units')