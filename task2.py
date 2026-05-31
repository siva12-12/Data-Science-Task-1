# ============================================================
#   Task 3: Car Price Prediction with Machine Learning
#   CodeAlpha Data Science Internship
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────
df = pd.read_csv('car_data.csv')
print(f'Shape: {df.shape}')
print(df.head())

# ── 2. CHECK DATA ─────────────────────────────────────────
print(df.info())
print('\nMissing Values:\n', df.isnull().sum())

# ── 3. FEATURE ENGINEERING ────────────────────────────────
df['Car_Age'] = 2025 - df['Year']
df.drop(columns=['Car_Name', 'Year'], inplace=True)

le = LabelEncoder()
df['Fuel_Type']    = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

# ── 4. VISUALIZATIONS ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Selling Price Distribution
sns.histplot(df['Selling_Price'], bins=30, kde=True, color='steelblue', ax=axes[0][0])
axes[0][0].set_title('Distribution of Selling Price')
axes[0][0].set_xlabel('Selling Price (Lakhs)')

# Plot 2: Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[0][1])
axes[0][1].set_title('Feature Correlation Heatmap')

# Plot 3: Present Price vs Selling Price
axes[1][0].scatter(df['Present_Price'], df['Selling_Price'], alpha=0.5, color='coral')
axes[1][0].set_title('Present Price vs Selling Price')
axes[1][0].set_xlabel('Present Price (Lakhs)')
axes[1][0].set_ylabel('Selling Price (Lakhs)')

# Plot 4: Car Age vs Selling Price
axes[1][1].scatter(df['Car_Age'], df['Selling_Price'], alpha=0.5, color='green')
axes[1][1].set_title('Car Age vs Selling Price')
axes[1][1].set_xlabel('Car Age (Years)')
axes[1][1].set_ylabel('Selling Price (Lakhs)')

plt.suptitle('Car Price Prediction - EDA', fontsize=16)
plt.tight_layout()
plt.show()

# ── 5. SPLIT DATA ─────────────────────────────────────────
X = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# ── 6. TRAIN MODELS ───────────────────────────────────────
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

# ── 7. BEST MODEL EVALUATION ──────────────────────────────
best_name  = max(results, key=lambda x: results[x]['R2'])
best_model = results[best_name]['model']
y_pred_best = best_model.predict(X_test)

print(f'\n🏆 Best Model : {best_name}')
print(f'   R² Score   : {results[best_name]["R2"]:.4f}')
print(f'   MAE        : {results[best_name]["MAE"]:.2f} Lakhs')
print(f'   RMSE       : {results[best_name]["RMSE"]:.2f} Lakhs')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.6, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price (Lakhs)')
axes[0].set_ylabel('Predicted Price (Lakhs)')
axes[0].set_title(f'Actual vs Predicted — {best_name}')

# Feature Importance
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importances.plot(kind='barh', ax=axes[1], color='teal')
axes[1].set_title('Feature Importance — Random Forest')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.show()

# ── 8. PREDICT CUSTOM CAR ─────────────────────────────────
# [Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner, Car_Age]
# Fuel_Type: CNG=0, Diesel=1, Petrol=2
# Selling_type: Dealer=0, Individual=1
# Transmission: Automatic=0, Manual=1
new_car = np.array([[5.59, 27000, 2, 0, 1, 0, 11]])
predicted_price = best_model.predict(new_car)
print(f'\nPredicted Selling Price: ₹ {predicted_price[0]:.2f} Lakhs')