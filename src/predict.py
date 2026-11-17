"""Score a single applicant with the saved model.

Run:  python -m src.predict --age 29 --income 480000 --credit_score 610 --loan_amount 400000 \
        --loan_term_months 36 --existing_emi 9000 --prior_defaults 1 --employment_years 2 \
        --purpose Personal --home_ownership Rent --city_tier "Tier 2"
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.train import add_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


def score(applicant: dict) -> dict:
    bundle = joblib.load(MODEL_PATH)
    row = dict(applicant)
    term_years = row["loan_term_months"] / 12
    row["debt_to_income"] = (row["existing_emi"] * 12 + row["loan_amount"] / term_years) / row["income"]
    X = add_features(pd.DataFrame([row]))
    p = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    return {"default_probability": round(p, 3),
            "decision": "High risk - review" if p >= bundle["threshold"] else "Low risk - approve",
            "threshold": round(bundle["threshold"], 3)}


def main():
    ap = argparse.ArgumentParser()
    for f in ["age", "income", "credit_score", "loan_amount", "existing_emi", "employment_years"]:
        ap.add_argument(f"--{f}", type=float, required=True)
    ap.add_argument("--loan_term_months", type=int, default=36)
    ap.add_argument("--prior_defaults", type=int, default=0)
    ap.add_argument("--purpose", default="Personal", choices=["Home", "Car", "Education", "Personal", "Business"])
    ap.add_argument("--home_ownership", default="Rent", choices=["Own", "Rent", "Mortgage"])
    ap.add_argument("--city_tier", default="Tier 1", choices=["Tier 1", "Tier 2", "Tier 3"])
    print(score(vars(ap.parse_args())))


if __name__ == "__main__":
    main()
