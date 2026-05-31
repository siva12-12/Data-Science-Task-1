import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')
print('✅ Libraries imported!')

df = pd.read_csv('Iris.csv')  # CSV file same folder la irukanum

# Id column useful illa, drop panniduvom
df.drop(columns=['Id'], inplace=True)

print(f'Shape: {df.shape}')
df.head(10)

print(df.info())
print('\n', df.describe())
print('\nSpecies Count:\n', df['Species'].value_counts())
print('\nMissing Values:\n', df.isnull().sum())

# Plot 1: Species Count
plt.figure(figsize=(6,4))
sns.countplot(x='Species', data=df, palette='Set2')
plt.title('Count of Each Iris Species')
plt.show()

# Plot 2: Pairplot
sns.pairplot(df, hue='Species', palette='Set1')
plt.show()

# Plot 3: Boxplots
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, feature in enumerate(features):
    ax = axes[i//2][i%2]
    sns.boxplot(x='Species', y=feature, data=df, palette='Set2', ax=ax)
    ax.set_title(feature)
plt.tight_layout()
plt.show()

# Plot 4: Heatmap
plt.figure(figsize=(7,5))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]

# Species column text -> number convert pannanum
le = LabelEncoder()
y = le.fit_transform(df['Species'])
# Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f'Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM'                : SVC(kernel='rbf', random_state=42),
    'KNN'                : KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f'{name:25s} → {acc*100:.2f}%')

best_name  = max(results, key=results.get)
best_model = models[best_name]
y_pred_best = best_model.predict(X_test_scaled)

print(f'🏆 Best Model: {best_name} ({results[best_name]*100:.2f}%)\n')
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap='Blues')
plt.title(f'Confusion Matrix — {best_name}')
plt.show()

# [SepalLength, SepalWidth, PetalLength, PetalWidth]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
new_scaled = scaler.transform(new_flower)

prediction = best_model.predict(new_scaled)
print(f'Predicted Species: 🌸 {le.inverse_transform(prediction)[0]}')