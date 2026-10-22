"""
Gesture Recognition Using Data Analysis
==========================================
Seminar: Hand Gesture Recognition - A Data-Driven Approach using Machine Learning

WHAT THIS PROGRAM DOES
A complete, self-contained data-analysis pipeline for hand gesture
recognition:
  1. Build a labeled dataset of hand-gesture FEATURES (the kind you'd get
     from MediaPipe's 21 hand landmarks - distances/angles between
     fingertips and the wrist, not raw pixels)
  2. Exploratory Data Analysis (EDA) - structure, stats, class balance,
     feature correlation
  3. Visualization - class distribution, correlation heatmap, PCA plot
  4. Train/test split + model training (Random Forest)
  5. Evaluation - accuracy, classification report, confusion matrix,
     feature importance

WHY SYNTHETIC FEATURES, NOT RAW IMAGES
Real MediaPipe landmark data needs a webcam session to collect. To let
you RUN this end-to-end right now (for testing/rehearsal) it generates
a dataset with the same shape and statistical character as real landmark
data would have - each gesture class is a distinct cluster in feature
space, with realistic noise. Swap in `real_data.csv` (see bottom of file)
once you've collected real samples with MediaPipe, and every line of
analysis below runs unchanged.

HOW TO RUN
    pip install numpy pandas matplotlib seaborn scikit-learn
    python gesture_data_analysis.py
Plots are saved to ./outputs/ and also shown on screen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

sns.set_style("whitegrid")
# This creates 'outputs' inside your current directory instead of the system root
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)

GESTURES = ["Open Palm", "Fist", "Thumbs Up", "Peace", "OK Sign", "Pointing"]

FEATURES = [
    "thumb_index_dist",    # distance between thumb tip and index tip
    "index_extension",     # index fingertip distance from wrist
    "middle_extension",    # middle fingertip distance from wrist
    "ring_extension",      # ring fingertip distance from wrist
    "pinky_extension",     # pinky fingertip distance from wrist
    "palm_width",          # distance between index_mcp and pinky_mcp
    "thumb_angle",         # angle of thumb relative to palm
    "fingers_up_count",    # how many fingers are extended
]

# ---------------------------------------------------------------------
# 1. BUILD DATASET (each gesture = a distinct region in feature space,
#    matching the real geometric signature of that hand pose)
# ---------------------------------------------------------------------
def make_gesture_dataset(n_per_class=180):
    # (mean feature vector, std dev) per gesture - hand-designed to reflect
    # real geometry, e.g. "Fist" has short extensions & 0 fingers up.
    profiles = {
        "Open Palm":  ([0.18, 0.85, 0.90, 0.85, 0.75, 0.55, 60, 5], 0.05),
        "Fist":       ([0.05, 0.25, 0.28, 0.25, 0.22, 0.35, 15, 0], 0.03),
        "Thumbs Up":  ([0.30, 0.28, 0.30, 0.27, 0.24, 0.36, 80, 1], 0.04),
        "Peace":      ([0.22, 0.85, 0.88, 0.30, 0.25, 0.45, 30, 2], 0.05),
        "OK Sign":    ([0.03, 0.30, 0.85, 0.82, 0.74, 0.50, 25, 3], 0.04),
        "Pointing":   ([0.20, 0.88, 0.30, 0.26, 0.23, 0.38, 20, 1], 0.04),
    }
    rows = []
    for gesture, (mean, std) in profiles.items():
        samples = RNG.normal(loc=mean, scale=std, size=(n_per_class, len(FEATURES)))
        samples[:, -1] = np.clip(np.round(samples[:, -1]), 0, 5)  # fingers_up_count stays a whole number 0-5
        samples[:, :-1] = np.clip(samples[:, :-1], 0, None)       # distances/angles can't go negative
        for row in samples:
            rows.append(list(row) + [gesture])
    df = pd.DataFrame(rows, columns=FEATURES + ["gesture"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


df = make_gesture_dataset()

# ---------------------------------------------------------------------
# 2. EDA - STRUCTURE & SUMMARY STATISTICS
# ---------------------------------------------------------------------
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]} samples, {df.shape[1] - 1} features, "
      f"{df['gesture'].nunique()} classes\n")
print("First 5 rows:")
print(df.head(), "\n")
print("Summary statistics:")
print(df.describe().round(2), "\n")
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# ---------------------------------------------------------------------
# 3. EDA - CLASS DISTRIBUTION
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
order = df["gesture"].value_counts().index
sns.countplot(data=df, y="gesture", order=order, color="#1C7293")
plt.title("Samples per Gesture Class")
plt.xlabel("Count")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=150)
plt.close()
print("Saved 01_class_distribution.png")

# ---------------------------------------------------------------------
# 4. EDA - FEATURE CORRELATION HEATMAP
#    Which features move together? e.g. index/middle/ring extension
#    are likely correlated since an open hand extends all of them.
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6.5))
corr = df[FEATURES].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved 02_correlation_heatmap.png")

# ---------------------------------------------------------------------
# 5. EDA - PCA PROJECTION (do gestures separate in 2D?)
# ---------------------------------------------------------------------
X = df[FEATURES].values
y = df["gesture"].values

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6.5))
for gesture in GESTURES:
    mask = y == gesture
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=gesture, s=25, alpha=0.7)
plt.title("PCA Projection of Gesture Features (2D)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_pca_projection.png", dpi=150)
plt.close()
print("Saved 03_pca_projection.png")

# ---------------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------------------
# 7. MODEL TRAINING
# ---------------------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=GESTURES)
plt.figure(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=GESTURES, yticklabels=GESTURES)
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_confusion_matrix.png", dpi=150)
plt.close()
print("Saved 04_confusion_matrix.png")

# ---------------------------------------------------------------------
# 9. FEATURE IMPORTANCE (which features drove the model's decisions?)
# ---------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#065A82")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_feature_importance.png", dpi=150)
plt.close()
print("Saved 05_feature_importance.png")

print("\nAll analysis outputs saved in ./outputs/")

# ---------------------------------------------------------------------
# USING REAL DATA INSTEAD
# ---------------------------------------------------------------------
# Once you've collected real landmark data (e.g. by logging MediaPipe's
# 21 landmarks per frame from live_gesture_demo.py into a CSV with the
# same FEATURES columns + a "gesture" label column), just replace:
#
#     df = make_gesture_dataset()
# with:
#     df = pd.read_csv("real_data.csv")
#
# Every EDA/training/evaluation step below runs unchanged on real data.
