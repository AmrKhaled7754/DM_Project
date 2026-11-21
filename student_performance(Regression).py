"""
Task 2 - REGRESSION: Student Performance Prediction (UCI Student Performance)
Target: G3 (final grade, 0-20).
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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import load_student

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
# G1 and G2 (earlier grades) are extremely correlated with G3.
# Run once with False and once with True, then compare the results in your report.
USE_G1_G2 = False
K_FEATURES = 15

# %% A. Dataset
df = load_student()
print(df.shape)
print(df.head())
df.info()

# %% B1. Explore
print("\nMissing values:", int(df.isnull().sum().sum()))
print("Duplicates:", df.duplicated().sum())
print(df["G3"].describe())

plt.figure(figsize=(6, 4))
sns.histplot(df["G3"], bins=20, kde=True)
plt.title("Distribution of final grade (G3)")
plt.savefig(OUT / "student_g3_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 9))
sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", annot=False)
plt.title("Correlation heatmap (numeric columns)")
plt.savefig(OUT / "student_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% B2. Duplicates + missing values
df = df.drop_duplicates()
# Missing values: numeric -> median, categorical -> mode (this dataset normally has none)
for c in df.columns:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median() if df[c].dtype != "object" else df[c].mode()[0])

if not USE_G1_G2:
    df = df.drop(columns=["G1", "G2"])

# %% C. Train/Test split (80/20)
X_raw = df.drop(columns="G3")
y = df["G3"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE)

# %% B3. Outliers (IQR capping, bounds learned from train)
CAP_COLS = ["absences"]
bounds = {}
for c in CAP_COLS:
    q1, q3 = X_train_raw[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    bounds[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

plt.figure(figsize=(5, 4))
sns.boxplot(y=X_train_raw["absences"])
plt.title("Absences (before capping)")
plt.savefig(OUT / "student_absences_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()


def cap(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c, (lo, hi) in bounds.items():
        d[c] = d[c].clip(lo, hi)
    return d


# %% B4. Encoding (one-hot for text columns) + B5. Scaling + B6. Feature selection
CAT = X_raw.select_dtypes("object").columns.tolist()


def encode(d: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(d, columns=CAT, drop_first=True, dtype=int)


X_train = encode(cap(X_train_raw))
X_test = encode(cap(X_test_raw)).reindex(columns=X_train.columns, fill_value=0)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)

selector = SelectKBest(f_regression, k=min(K_FEATURES, X_train_s.shape[1]))
selector.fit(X_train_s, y_train)
selected = X_train_s.columns[selector.get_support()].tolist()
print("Selected features:", selected)
X_train_f, X_test_f = X_train_s[selected], X_test_s[selected]

# %% Train models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

# %% F. Evaluation metrics
rows, fitted = [], {}
for name, m in models.items():
    m.fit(X_train_f, y_train)
    pred = m.predict(X_test_f)
    fitted[name] = (m, pred)
    rows.append({
        "Model": name,
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    })
results = pd.DataFrame(rows).set_index("Model").round(3)
print(results)
results.to_csv(OUT / f"student_results_{'with' if USE_G1_G2 else 'without'}_G1G2.csv")

best_name = results["R2"].idxmax()
best_model, best_pred = fitted[best_name]
print(f"\nBest model: {best_name}")
cv = cross_val_score(best_model, X_train_f, y_train, cv=5, scoring="r2")
print(f"5-fold CV R2 on train: {cv.mean():.3f} +/- {cv.std():.3f}")

# %% E + G. Visualizations
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.6)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title(f"Predicted vs Actual - {best_name}")
plt.legend()
plt.savefig(OUT / "student_pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(y_test - best_pred, bins=20, kde=True)
plt.title("Residuals (actual - predicted)")
plt.savefig(OUT / "student_residuals.png", dpi=150, bbox_inches="tight")
plt.show()

rf = fitted["Random Forest"][0]
pd.Series(rf.feature_importances_, index=selected).sort_values().plot(
    kind="barh", figsize=(7, 6), title="Feature importance (Random Forest)")
plt.savefig(OUT / "student_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

results["RMSE"].plot(kind="bar", rot=15, title="RMSE by model (lower is better)", figsize=(7, 4))
plt.savefig(OUT / "student_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% D. Predict new students
def predict_new(raw: pd.DataFrame) -> np.ndarray:
    d = encode(cap(raw)).reindex(columns=X_train.columns, fill_value=0)
    d = pd.DataFrame(scaler.transform(d), columns=X_train.columns)
    return best_model.predict(d[selected])


template = X_raw.iloc[[0]].copy()
good, weak = template.copy(), template.copy()
good["studytime"], good["failures"], good["absences"], good["higher"] = 4, 0, 2, "yes"
weak["studytime"], weak["failures"], weak["absences"], weak["higher"] = 1, 3, 20, "no"
if USE_G1_G2:
    good["G1"], good["G2"] = 15, 16
    weak["G1"], weak["G2"] = 6, 5
new_students = pd.concat([good, weak], ignore_index=True)
for i, p in enumerate(predict_new(new_students), 1):
    print(f"Student {i}: predicted final grade G3 = {p:.1f} / 20")
