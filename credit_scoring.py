"""
CodeAlpha ML Internship - Task 1: Credit Scoring Model
Dataset : German Credit (UCI / OpenML "credit-g"), 1000 applicants
Models  : Logistic Regression, Decision Tree, Random Forest
Target  : 1 = bad credit (high risk / default), 0 = good credit
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay)

RS = 42

# 1. Load data ---------------------------------------------------------------
def load_german_credit():
    """Real German Credit data from OpenML. Falls back to synthetic demo data
    only if there is no internet, so the pipeline can still be tested."""
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml("credit-g", version=1, as_frame=True)
        X = d.data.copy()
        y = (d.target == "bad").astype(int)
        print("Loaded REAL German Credit dataset from OpenML")
        return X, y, True
    except Exception as e:
        print("!! Could not download dataset (", type(e).__name__, ")")
        print("!! Using SYNTHETIC demo data. Run with internet for real results.")
        rng = np.random.default_rng(RS)
        n = 1000
        X = pd.DataFrame({
            "duration": rng.integers(4, 72, n),
            "credit_amount": rng.gamma(2.0, 1500, n).round(),
            "age": rng.integers(19, 75, n),
            "installment_commitment": rng.integers(1, 5, n),
            "existing_credits": rng.integers(1, 4, n),
            "num_dependents": rng.integers(1, 3, n),
            "checking_status": rng.choice(["<0", "0<=X<200", ">=200", "no checking"], n),
            "savings_status": rng.choice(["<100", "100<=X<500", ">=500", "unknown"], n),
            "employment": rng.choice(["unemployed", "<1", "1<=X<4", ">=4"], n),
            "housing": rng.choice(["own", "rent", "free"], n),
            "purpose": rng.choice(["car", "radio/tv", "education", "business"], n),
        })
        score = (0.03 * X.duration + 0.0002 * X.credit_amount - 0.02 * X.age
                 + (X.checking_status == "<0") * 0.9
                 + (X.savings_status == "<100") * 0.5
                 + (X.employment == "unemployed") * 0.7)
        p = 1 / (1 + np.exp(-(score - score.mean()) * 1.3))
        y = pd.Series((rng.random(n) < p).astype(int))
        return X, y, False

X, y, is_real = load_german_credit()
print("Shape:", X.shape, "| Bad-credit rate: %.1f%%" % (100 * y.mean()))

# 2. Feature engineering -------------------------------------------------------
X["credit_per_month"] = X["credit_amount"] / X["duration"]
X["credit_to_age"] = X["credit_amount"] / X["age"]
X["is_long_term"] = (X["duration"] > 24).astype(int)
X["is_young"] = (X["age"] < 25).astype(int)
X["log_credit_amount"] = np.log1p(X["credit_amount"])

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# 3. Split ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS)

# 4. Models (class_weight handles imbalance) -----------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                            class_weight="balanced", random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                            class_weight="balanced", random_state=RS),
}
pipes = {n: Pipeline([("prep", prep), ("clf", m)]) for n, m in models.items()}

# 5. Train + evaluate ----------------------------------------------------------
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
rows = []
for name, pipe in pipes.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "CV ROC-AUC": cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean(),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
results = pd.DataFrame(rows).set_index("Model").round(4)
print("\n=== Model Comparison (positive class = bad credit) ===")
print(results)
results.to_csv("results.csv")

best_name = results["ROC-AUC"].idxmax()
best = pipes[best_name]
print(f"\nBest model (by ROC-AUC): {best_name}")
print(classification_report(y_test, best.predict(X_test),
                            target_names=["Good credit", "Bad credit"]))

# 6. Plots ---------------------------------------------------------------------
cm = confusion_matrix(y_test, best.predict(X_test))
plt.figure(figsize=(4.5, 3.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in pipes.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("ROC Curves - Credit Scoring")
plt.tight_layout(); plt.savefig("roc_curves.png", dpi=150); plt.close()

rf = pipes["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(12)
plt.figure(figsize=(7, 5))
imp.plot(kind="barh")
plt.title("Top 12 Features - Random Forest")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

print("\nSaved: results.csv, confusion_matrix.png, roc_curves.png, feature_importance.png")
if not is_real:
    print("NOTE: these numbers are from SYNTHETIC data - rerun with internet before submitting!")
