"""Train, compare and evaluate loan-default models.

Run:  python -m src.train
- 5-fold stratified cross-validation of 4 models on the training set
- best model (by ROC-AUC) evaluated once on a held-out 20% test set
- decision threshold tuned for the best F1 on the training folds
- saves models/model.joblib, reports/metrics.json and charts in reports/
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, classification_report,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "loans.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TARGET = "default"
CATEGORICAL = ["purpose", "home_ownership", "city_tier"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: ratios that lenders actually look at."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["income"]
    df["emi_to_income"] = df["existing_emi"] * 12 / df["income"]
    df["has_prior_default"] = (df["prior_defaults"] > 0).astype(int)
    return df


def build_preprocessor(numeric: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, numeric), ("cat", cat, CATEGORICAL)])


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced",
                                                n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=4,
                                                            class_weight="balanced", random_state=42),
    }


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    df = add_features(pd.read_csv(DATA))
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "f1": "f1"}
    comparison = {}
    print(f"Training rows: {len(X_train):,} | test rows: {len(X_test):,} | default rate: {y.mean():.1%}\n")
    print(f"{'Model':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'Recall':>9}{'F1':>8}")
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor(numeric)), ("model", model)])
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        comparison[name] = {k: round(float(np.mean(s[f"test_{k}"])), 3) for k in scoring}
        r = comparison[name]
        print(f"{name:<22}{r['roc_auc']:>9.3f}{r['pr_auc']:>9.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}")

    best_name = max(comparison, key=lambda n: comparison[n]["roc_auc"])
    best = Pipeline([("prep", build_preprocessor(numeric)), ("model", candidate_models()[best_name])])

    # tune the decision threshold on out-of-fold predictions (no test-set leakage)
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    threshold = float(thr[np.argmax(f1s)])

    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    test = {
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "pr_auc": round(average_precision_score(y_test, proba), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "threshold": round(threshold, 3),
    }
    print(f"\nBest model: {best_name}\nHeld-out test set: {test}\n")
    print(classification_report(y_test, pred, target_names=["Repaid", "Default"]))

    # charts
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax[0], name=best_name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_title("ROC curve (test set)")
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Repaid", "Default"], ax=ax[1],
                                            cmap="Blues", colorbar=False)
    ax[1].set_title(f"Confusion matrix (threshold {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(REPORTS / "roc_confusion.png", dpi=120)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.tail(10).plot.barh(ax=ax, color="#2E5C8A")
    ax.set_title("Top 10 features (drop in ROC-AUC when shuffled)")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=120)

    joblib.dump({"pipeline": best, "threshold": threshold, "model_name": best_name}, MODELS / "model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps({
        "cv_comparison": comparison, "best_model": best_name, "test": test,
        "top_features": importance.sort_values(ascending=False).head(5).round(4).to_dict(),
    }, indent=2))
    print(f"Saved model to {MODELS / 'model.joblib'} and reports to {REPORTS}/")


if __name__ == "__main__":
    main()
