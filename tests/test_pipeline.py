from src.generate_data import generate
from src.train import add_features


def test_dataset_shape_and_default_rate():
    df = generate(n=3000, seed=1)
    assert len(df) == 3000
    assert 0.1 < df["default"].mean() < 0.35          # realistic class imbalance
    assert df["credit_score"].isna().any()            # has missing values to handle


def test_risk_signal_direction():
    df = generate(n=5000, seed=2)
    low, high = df[df.credit_score < 620], df[df.credit_score > 760]
    assert low["default"].mean() > high["default"].mean()
    assert df[df.prior_defaults > 0]["default"].mean() > df[df.prior_defaults == 0]["default"].mean()


def test_feature_engineering():
    df = add_features(generate(n=100, seed=3))
    assert {"loan_to_income", "emi_to_income", "has_prior_default"} <= set(df.columns)
