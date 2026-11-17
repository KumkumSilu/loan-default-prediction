"""Generate a realistic synthetic loan dataset (safe to publish).

Run:  python -m src.generate_data
Writes data/loans.csv with 10,000 applicants and a ~20% default rate.
Default risk is driven by credit score, debt-to-income, prior defaults,
employment length and loan size, plus random noise and some missing values,
so the models have a realistic (not perfect) problem to learn.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "loans.csv"


def generate(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    employment_years = np.clip(rng.gamma(2.0, 3.0, n), 0, age - 18).round(1)
    income = np.round(rng.lognormal(13.1, 0.5, n), -3)            # INR per year, median ~5 lakh
    credit_score = np.clip(rng.normal(690, 70, n), 300, 900).round()
    loan_amount = np.round(income * rng.uniform(0.1, 1.5, n), -3)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n, p=[0.1, 0.2, 0.3, 0.2, 0.2])
    existing_emi = np.round(income / 12 * rng.beta(2, 6, n), -2)
    debt_to_income = np.round((existing_emi * 12 + loan_amount / (loan_term_months / 12)) / income, 3)
    prior_defaults = rng.choice([0, 1, 2, 3], n, p=[0.8, 0.13, 0.05, 0.02])
    purpose = rng.choice(["Home", "Car", "Education", "Personal", "Business"], n, p=[0.25, 0.2, 0.15, 0.3, 0.1])
    home_ownership = rng.choice(["Own", "Rent", "Mortgage"], n, p=[0.3, 0.45, 0.25])
    city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[0.4, 0.35, 0.25])

    # hidden risk score -> probability of default
    z = (
        -2.4
        - 0.018 * (credit_score - 690)
        + 2.2 * (debt_to_income - 0.45)
        + 0.75 * prior_defaults
        - 0.06 * employment_years
        + 0.25 * (loan_amount / income)
        + np.where(purpose == "Personal", 0.35, 0) + np.where(purpose == "Business", 0.45, 0)
        + np.where(home_ownership == "Rent", 0.2, 0)
        - 0.012 * (age - 40)
        + rng.normal(0, 0.6, n)
    )
    default = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income, "employment_years": employment_years,
        "credit_score": credit_score, "loan_amount": loan_amount,
        "loan_term_months": loan_term_months, "existing_emi": existing_emi,
        "debt_to_income": debt_to_income, "prior_defaults": prior_defaults,
        "purpose": purpose, "home_ownership": home_ownership, "city_tier": city_tier,
        "default": default,
    })
    # realistic gaps in the data
    for col, frac in [("employment_years", 0.04), ("credit_score", 0.03), ("existing_emi", 0.02)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT} (default rate {df['default'].mean():.1%})")
