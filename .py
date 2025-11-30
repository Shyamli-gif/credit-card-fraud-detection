# credit_card_fraud.py

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE

# 2. Load dataset (download from Kaggle: creditcard.csv)
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print(df['Class'].value_counts())  # 0 = legit, 1 = fraud

# 3. Split features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# 4. Train-test split (stratified to preserve fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 5. Handle imbalance with SMOTE (oversampling minority class)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Resampled training set shape:", X_train_res.shape, y_train_res.shape)

# 6. Train model (Random Forest)
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)

# 7. Predictions
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

# 8. Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

plt.figure(figsize=(6,4))
plt.plot(recall, precision, marker='.')
plt.title(f"Precision-Recall Curve (AP={ap_score:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()
