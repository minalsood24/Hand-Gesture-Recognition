
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv("gesture_data.csv")

print("Dataset shape:", df.shape)
print("\nGesture distribution:")
print(df.iloc[:, -1].value_counts())

# 2. Separate features and labels
X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
y = df.iloc[:, -1]

# Remove rows containing missing values
valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# 3. Check the dataset
if y.nunique() < 2:
    raise ValueError("At least two gesture classes are required.")

if y.value_counts().min() < 2:
    raise ValueError(
        "Each gesture needs at least two samples."
    )

print("\nFeatures:", X.shape[1])
print("Total usable samples:", len(X))

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save the trained model
joblib.dump(model, "gesture_model.pkl")

print("\nModel saved as gesture_model.pkl")