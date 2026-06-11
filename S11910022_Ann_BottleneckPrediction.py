# 1. Import tools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

plt.style.use("seaborn-v0_8")


# 1. Load Data

df = pd.read_csv("semiconductor_cycle_time_simulation.csv")

print("/n Data Info:")
print(df.info())
print("/n First 5 Data:")
print(df.head())
print("/n Last 5 Data:")
print(df.tail())

print(f"Total entries: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Bottleneck %: {df['bottleneck_flag'].mean()*100:.2f}%")
print(f"Cycle Time avg: {df['cycle_time'].mean():.2f} hours")


# 2. Prepare Features

# Features
X = df[['wip_level', 'utilization', 'downtime_min', 'rework_flag', 'queue_time']]
y_reg = df['cycle_time']  # For regression
y_clf = df['bottleneck_flag']  # For classification

# Split data
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.columns.tolist()}")


# 3. Regression Models - Cycle Time Prediction

# Models
reg_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

reg_results = {}

for name, model in reg_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train_reg)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test_reg, predictions)
    r2 = r2_score(y_test_reg, predictions)
    
    reg_results[name] = {'model': model, 'mae': mae, 'r2': r2}
    
    print(f"  MAE: {mae:.2f} hours")
    print(f"  R2: {r2:.3f}")

# Regression Results Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE comparison
axes[0].bar(reg_results.keys(), [reg_results[m]['mae'] for m in reg_results], color=['#264653', '#e76f51', '#2a9d8f'])
axes[0].set_title('Regression MAE Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('MAE (hours)')
axes[0].tick_params(axis='x', rotation=45)

# R2 comparison
axes[1].bar(reg_results.keys(), [reg_results[m]['r2'] for m in reg_results], color=['#264653', '#e76f51', '#2a9d8f'])
axes[1].set_title('Regression R2 Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('R2 Score')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Regression Results', fontsize=25, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("regression_results.png", dpi=300, bbox_inches='tight')
plt.show()

# Best regression model
best_reg = max(reg_results, key=lambda x: reg_results[x]['r2'])
print(f"\nBest Regression Model: {best_reg} (R2 = {reg_results[best_reg]['r2']:.3f})")


# 4. Classification Models - Bottleneck Warning

# Models
clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

clf_results = {}

for name, model in clf_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_clf, y_train_clf)
    predictions = model.predict(X_test_clf)
    
    accuracy = accuracy_score(y_test_clf, predictions)
    
    clf_results[name] = {'model': model, 'accuracy': accuracy}
    
    print(f"  Accuracy: {accuracy:.3f}")

# Classification Results Plot
plt.figure(figsize=(10, 5))
plt.bar(clf_results.keys(), [clf_results[m]['accuracy'] for m in clf_results], 
        color=['#264653', '#e76f51', '#2a9d8f'])
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)
plt.tick_params(axis='x', rotation=45)

plt.yticks([0.90, 0.95, 1.00], ['90%', '95%', '100%'])

plt.suptitle('Classification Results', fontsize=25, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("classification_results.png", dpi=300, bbox_inches='tight')
plt.show()

# Best classification model
best_clf = max(clf_results, key=lambda x: clf_results[x]['accuracy'])
print(f"\nBest Classification Model: {best_clf} (Accuracy = {clf_results[best_clf]['accuracy']:.3f})")


# 5. Feature Importance - Random Forest

# Random Forest Regression Feature Importance
rf_reg = reg_results['Random Forest']['model']
rf_clf = clf_results['Random Forest']['model']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression importance
axes[0].barh(X.columns, rf_reg.feature_importances_, color='#2a9d8f')
axes[0].set_title('Cycle Time - Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].set_xlabel('Importance')

# Classification importance
axes[1].barh(X.columns, rf_clf.feature_importances_, color='#e76f51')
axes[1].set_title('Bottleneck - Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].set_xlabel('Importance')

plt.suptitle('Feature Importance', fontsize=25, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nFeature Importance (Random Forest Regression):")
for feat, imp in zip(X.columns, rf_reg.feature_importances_):
    print(f"  {feat}: {imp:.3f}")


# 6. Confusion Matrix - Best Classification Model

best_clf_model = clf_results[best_clf]['model']
predictions = best_clf_model.predict(X_test_clf)

cm = confusion_matrix(y_test_clf, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 14})
plt.title(f'Confusion Matrix - {best_clf}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Normal', 'Bottleneck'])
plt.yticks([0.5, 1.5], ['Normal', 'Bottleneck'])

plt.suptitle('Confusion Matrix', fontsize=25, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()


# 7. Summary

print("\n Regression (Cycle Time Prediction):")
for name, results in reg_results.items():
    print(f"  {name}: MAE = {results['mae']:.2f} hrs, R2 = {results['r2']:.3f}")
print(f"  → Best: {best_reg}")

print("\n Classification (Bottleneck Warning):")
for name, results in clf_results.items():
    print(f"  {name}: Accuracy = {results['accuracy']:.3f}")
print(f"  → Best: {best_clf}")
