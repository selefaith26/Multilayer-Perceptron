"""
Apple Quality Detector — Multilayer Perceptron
MLP Assignment - Grand Canyon University
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_score, recall_score, f1_score,
                              accuracy_score)
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('outputs', exist_ok=True)


# LOAD AND PREPROCESS DATA

print("\n  APPLE QUALITY DETECTOR — Multilayer Perceptron")
print("  ----------\n")

print("  [1] Loading and Preprocessing Data")

df = pd.read_csv('apple_quality.csv')
df = df.dropna(subset=['Quality'])
df = df[df['Quality'].isin(['good', 'bad'])]

print(f"  Dataset shape    : {df.shape}")
print(f"  Good apples      : {(df['Quality'] == 'good').sum()}")
print(f"  Bad apples       : {(df['Quality'] == 'bad').sum()}")
print(f"  Missing values   : {df.isnull().sum().sum()}")


# SUBSET DATA

print("\n  [2] Subsetting Data")

feature_cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness',
                'Juiciness', 'Ripeness', 'Acidity']

X = df[feature_cols].values.astype(float)
y = np.where(df['Quality'].values == 'good', 1, 0)

print(f"  Features used    : {feature_cols}")
print(f"  Target classes   : good=1, bad=0")


# SPLIT DATA

print("\n  [3] Splitting Data into Training and Testing Sets")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler      = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")


# BUILD MLP MODEL

print("\n  [4] Building the MLP Model")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
    activation='relu',                  # ReLU activation
    solver='adam',                      # Adam optimizer
    max_iter=500,
    random_state=42,
    early_stopping=True,                # stop when validation loss stops improving
    validation_fraction=0.1,
    n_iter_no_change=15
)

mlp.fit(X_train_std, y_train)

print(f"  Architecture     : Input(7) → 128 → 64 → 32 → Output(2)")
print(f"  Activation       : ReLU")
print(f"  Optimizer        : Adam")
print(f"  Epochs trained   : {mlp.n_iter_}")
print(f"  Final loss       : {mlp.loss_:.4f}")


# RUN MODEL — MAKE PREDICTIONS

print("\n  [5] Making Predictions")

y_train_pred = mlp.predict(X_train_std)
y_test_pred  = mlp.predict(X_test_std)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)

print(f"  Training accuracy : {train_acc*100:.1f}%")
print(f"  Test accuracy     : {test_acc*100:.1f}%")
print(f"  Overfitting gap   : {(train_acc - test_acc)*100:.1f}%")


# CLASSIFICATION RESULTS

print("\n  [6] Classification Results")

prec  = precision_score(y_test, y_test_pred, average='weighted')
rec   = recall_score(y_test,    y_test_pred, average='weighted')
f1    = f1_score(y_test,        y_test_pred, average='weighted')

print(f"  Precision  : {prec:.3f}")
print(f"  Recall     : {rec:.3f}")
print(f"  F-Measure  : {f1:.3f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_test_pred,
      target_names=['bad', 'good']))


# CONFUSION MATRIX

print("\n  [7] Confusion Matrix")

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"  True Negative  (bad  → bad)  : {tn}")
print(f"  False Positive (bad  → good) : {fp}")
print(f"  False Negative (good → bad)  : {fn}")
print(f"  True Positive  (good → good) : {tp}")


# PLOT 1 — LEARNING CURVE

plt.figure(figsize=(8, 4))
plt.plot(mlp.loss_curve_, color='#FF85A1', linewidth=2, label='Training Loss')
if mlp.validation_scores_ is not None:
    val_loss = [1 - s for s in mlp.validation_scores_]
    plt.plot(val_loss, color='#A29BFE', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plot1_learning_curve.png')
plt.close()
print("\n  Plot saved: plot1_learning_curve.png")


# PLOT 2 — CONFUSION MATRIX

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='RdPu')
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Bad', 'Good'])
ax.set_yticklabels(['Bad', 'Good'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — MLP Apple Detector')
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14,
                color='white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.savefig('outputs/plot2_confusion_matrix.png')
plt.close()
print("  Plot saved: plot2_confusion_matrix.png")


# PLOT 3 — METRICS BAR CHART

metrics = ['Accuracy', 'Precision', 'Recall', 'F-Measure']
values  = [test_acc, prec, rec, f1]
colors  = ['#FF85A1', '#A29BFE', '#81ECEC', '#FFEAA7']

plt.figure(figsize=(7, 4))
bars = plt.bar(metrics, values, color=colors, edgecolor='white')
plt.ylabel('Score')
plt.title('MLP Performance Metrics')
plt.ylim(0, 1.15)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f'{val:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/plot3_metrics.png')
plt.close()
print("  Plot saved: plot3_metrics.png")


# PLOT 4 — SCORE DISTRIBUTION

y_scores = mlp.predict_proba(X_test_std)[:, 1]

plt.figure(figsize=(8, 4))
plt.hist(y_scores[y_test == 1], bins=30, alpha=0.6,
         color='#A29BFE', label='Good apples')
plt.hist(y_scores[y_test == 0], bins=30, alpha=0.6,
         color='#FF85A1', label='Bad apples')
plt.axvline(x=0.5, color='black', linestyle='--',
            linewidth=2, label='Decision boundary')
plt.xlabel('Predicted Probability of Good')
plt.ylabel('Count')
plt.title('MLP Score Distribution by Apple Quality')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plot4_score_distribution.png')
plt.close()
print("  Plot saved: plot4_score_distribution.png")


# FINAL SUMMARY

print(f"\n  ----------")
print(f"  FINAL SUMMARY")
print(f"  ----------")
print(f"  Architecture     : 7 → 128 → 64 → 32 → 2")
print(f"  Training accuracy: {train_acc*100:.1f}%")
print(f"  Test accuracy    : {test_acc*100:.1f}%")
print(f"  Precision        : {prec:.3f}")
print(f"  Recall           : {rec:.3f}")
print(f"  F-Measure        : {f1:.3f}")
print(f"  ----------\n")
print("  Done. Plots saved to outputs/ folder.\n")
