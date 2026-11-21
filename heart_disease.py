"""
Task 1 - CLASSIFICATION: Heart Disease Prediction (UCI Heart Disease)
Sections follow the task sheet: A dataset | B preprocessing | C split |
D prediction on new values | E predicted vs actual | F metrics | G visualizations.
Run cell by cell in PyCharm (# %% markers) or run the whole file.
"""
# %% Imports & config
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
                             roc_curve, classification_report)
from imblearn.over_sampling import SMOTE

from data_loader import load_heart

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
APPLY_SMOTE = True      # harmless on balanced data; required by the task's "handle imbalance"
K_FEATURES = 12         # number of features kept by feature selection

# %% A. Dataset
df = load_heart()
print(df.shape)
print(df.head())
df.info()

# %% B1. Explore: missing values, duplicates, class balance
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nTarget distribution:\n", df["target"].value_counts())

plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df)
plt.title("Target distribution (0 = healthy, 1 = heart disease)")
plt.savefig(OUT / "heart_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.savefig(OUT / "heart_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% B2. Remove duplicates + detect/remove outliers (IQR, 3x to avoid dropping too many rows)
df = df.drop_duplicates()

num_cols = ["trestbps", "chol", "thalach", "oldpeak"]
plt.figure(figsize=(10, 4))
df[num_cols].boxplot()
plt.title("Boxplots before outlier removal")
plt.savefig(OUT / "heart_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

mask = pd.Series(True, index=df.index)
for c in num_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask &= df[c].between(q1 - 3 * iqr, q3 + 3 * iqr)
print(f"Outlier rows removed: {(~mask).sum()}")
df = df[mask]

# %% C. Train/Test split (80/20, stratified). Everything below is fitted on TRAIN only.
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# %% B3. Missing values (median imputation, fitted on train)
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

# %% B4. Encoding categorical variables (one-hot)
CAT = ["cp", "restecg", "slope", "thal"]


def encode(d: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(d, columns=CAT, drop_first=True, dtype=int)


X_train = encode(X_train)
X_test = encode(X_test).reindex(columns=X_train.columns, fill_value=0)

# %% B5. Feature scaling
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)

# %% B6. Feature selection (ANOVA F-test)
selector = SelectKBest(f_classif, k=min(K_FEATURES, X_train_s.shape[1]))
selector.fit(X_train_s, y_train)
selected = X_train_s.columns[selector.get_support()].tolist()
print("Selected features:", selected)
X_train_f = X_train_s[selected]
X_test_f = X_test_s[selected]

# %% B7. Handle imbalance (SMOTE on train only)
print("Train class counts before:", y_train.value_counts().to_dict())
if APPLY_SMOTE:
    X_train_f, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_f, y_train)
    print("Train class counts after :", pd.Series(y_train).value_counts().to_dict())

# %% Train several models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

# %% F. Evaluation metrics
rows, fitted = [], {}
for name, m in models.items():
    m.fit(X_train_f, y_train)
    pred = m.predict(X_test_f)
    proba = m.predict_proba(X_test_f)[:, 1]
    fitted[name] = (m, pred, proba)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
results = pd.DataFrame(rows).set_index("Model").round(3)
print(results)
results.to_csv(OUT / "heart_results.csv")

best_name = results["F1"].idxmax()
best_model, best_pred, best_proba = fitted[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_pred, target_names=["Healthy", "Disease"]))
cv = cross_val_score(best_model, X_train_f, y_train, cv=5, scoring="f1")
print(f"5-fold CV F1 on train: {cv.mean():.3f} +/- {cv.std():.3f}")

# %% E + G. Visualizations: confusion matrix, ROC curves, predicted vs actual, feature importance
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, best_pred),
                       display_labels=["Healthy", "Disease"]).plot(ax=ax, cmap="Blues")
ax.set_title(f"Confusion matrix - {best_name}")
plt.savefig(OUT / "heart_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 5))
for name, (_, _, proba) in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend()
plt.savefig(OUT / "heart_roc.png", dpi=150, bbox_inches="tight")
plt.show()

comp = pd.DataFrame({"Actual": y_test.value_counts().sort_index(),
                     "Predicted": pd.Series(best_pred).value_counts().sort_index()})
comp.index = ["Healthy", "Disease"]
comp.plot(kind="bar", rot=0, figsize=(6, 4), title="Actual vs Predicted (test set)")
plt.savefig(OUT / "heart_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

rf = fitted["Random Forest"][0]
imp = pd.Series(rf.feature_importances_, index=selected).sort_values()
imp.plot(kind="barh", figsize=(7, 5), title="Feature importance (Random Forest)")
plt.savefig(OUT / "heart_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% D. Predict new patients
new_patients = pd.DataFrame([
    # high-risk profile
    {"age": 62, "sex": 1, "cp": 4, "trestbps": 150, "chol": 270, "fbs": 1, "restecg": 2,
     "thalach": 110, "exang": 1, "oldpeak": 3.0, "slope": 2, "ca": 2, "thal": 7},
    # low-risk profile
    {"age": 35, "sex": 0, "cp": 2, "trestbps": 118, "chol": 190, "fbs": 0, "restecg": 0,
     "thalach": 172, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3},
])[X.columns]


def preprocess_new(raw: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(imputer.transform(raw), columns=raw.columns)
    d = encode(d).reindex(columns=X_train.columns, fill_value=0)
    d = pd.DataFrame(scaler.transform(d), columns=X_train.columns)
    return d[selected]


new_X = preprocess_new(new_patients)
pred_new = best_model.predict(new_X)
prob_new = best_model.predict_proba(new_X)[:, 1]
for i, (p, pr) in enumerate(zip(pred_new, prob_new), 1):
    print(f"Patient {i}: {'Heart disease' if p == 1 else 'Healthy'} (probability = {pr:.2f})")
