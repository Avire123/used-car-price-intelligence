import os
import joblib
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = "data/processed_car_listings.csv"
PROD_MODEL_PATH = "models/car_price_model.pkl"
CANDIDATE_MODEL_PATH = "models/candidate_car_price_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


@pytest.fixture(scope="module")
def evaluation_data():
    """Load processed dataset and prepare test split."""
    assert os.path.exists(PROCESSED_DATA_PATH), f"Data missing at {PROCESSED_DATA_PATH}"
    df = pd.read_csv(PROCESSED_DATA_PATH)

    features = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "location",
        "vehicle_age",
        "mileage",
        "mileage_per_year",
        "engine_size",
        "luxury_brand",
        "automatic_flag",
    ]
    target = "price"

    X = df[features]
    y = df[target]

    assert os.path.exists(PREPROCESSOR_PATH), f"Preprocessor missing at {PREPROCESSOR_PATH}"
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_prep = preprocessor.transform(X)

    # Fixed seed for reproducible test splits
    _, X_test, _, y_test = train_test_split(
        X_prep, y, test_size=0.2, random_state=42
    )

    return X_test, y_test


def test_candidate_performance_and_regression(evaluation_data):
    X_test, y_test = evaluation_data

    # 1. Ensure candidate model exists
    assert os.path.exists(CANDIDATE_MODEL_PATH), (
        f"Candidate model artifact not found at {CANDIDATE_MODEL_PATH}"
    )
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)

    # 2. Compute Candidate Metrics
    candidate_preds = candidate_model.predict(X_test)
    candidate_mae = mean_absolute_error(y_test, candidate_preds)
    candidate_r2 = r2_score(y_test, candidate_preds)

    # 3. Baseline Quality Check
    MIN_R2_THRESHOLD = 0.60
    assert candidate_r2 >= MIN_R2_THRESHOLD, (
        f"Candidate R² score ({candidate_r2:.4f}) is below minimum acceptable threshold ({MIN_R2_THRESHOLD})."
    )

    # 4. Compare with Production Model (if exists)
    if os.path.exists(PROD_MODEL_PATH):
        prod_model = joblib.load(PROD_MODEL_PATH)
        prod_preds = prod_model.predict(X_test)
        prod_mae = mean_absolute_error(y_test, prod_preds)
        prod_r2 = r2_score(y_test, prod_preds)

        # Allow at most 2% MAE regression tolerance
        MAX_ALLOWED_MAE_REGRESSION = 1.02
        assert candidate_mae <= (prod_mae * MAX_ALLOWED_MAE_REGRESSION), (
            f"Model Regression Detected! "
            f"Candidate MAE (KSh {candidate_mae:,.2f}) degraded compared to Prod MAE (KSh {prod_mae:,.2f})."
        )

        assert candidate_r2 >= (prod_r2 - 0.02), (
            f"R² Score Regressed! Candidate R² ({candidate_r2:.4f}) vs Prod R² ({prod_r2:.4f})."
        )