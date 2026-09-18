import os
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Suppress Optuna verbose trial output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_preprocessor():
    categorical_cols = ["make", "model", "condition", "fuel_type", "location"]
    numerical_cols = [
        "vehicle_age",
        "mileage",
        "mileage_per_year",
        "engine_size",
        "luxury_brand",
        "automatic_flag",
    ]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )


def train_candidate_model(
    data_path="data/processed_car_listings.csv",
    candidate_model_path="models/candidate_car_price_model.pkl",
    preprocessor_path="models/preprocessor.pkl",
    n_trials=5,
    cv_folds=3,
):
    # 1. Ensure directories exist
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Run features.py first!"
        )

    # 2. Load Data
    df = pd.read_csv(data_path)

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

    # 3. Fit Preprocessor
    preprocessor = build_preprocessor()
    X_prep = preprocessor.fit_transform(X)

    # 4. Optuna Objective Function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        model = XGBRegressor(**params)
        kf = KFold(n_splits=min(cv_folds, len(df)), shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_prep, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1
        )

        return -np.mean(scores)

    print(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Best CV MAE: KSh {study.best_value:,.2f}")
    print("Best Parameters:", best_params)

    # 5. Train Candidate Model on Full Data
    final_model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    final_model.fit(X_prep, y)

    # Evaluate on full dataset
    preds = final_model.predict(X_prep)
    print(f"Full Dataset R2 Score: {r2_score(y, preds):.4f}")
    print(f"Full Dataset MAE: KSh {mean_absolute_error(y, preds):,.2f}")

    # 6. Save Candidate Model & Preprocessor Artifacts
    joblib.dump(final_model, candidate_model_path)
    joblib.dump(preprocessor, preprocessor_path)

    # If no production model exists yet (first run), duplicate candidate as production model
    prod_model_path = "models/car_price_model.pkl"
    if not os.path.exists(prod_model_path):
        joblib.dump(final_model, prod_model_path)
        print(f"Initial production model initialized at '{prod_model_path}'.")

    print(f"Candidate model artifact successfully saved to '{candidate_model_path}'.")


if __name__ == "__main__":
    train_candidate_model()