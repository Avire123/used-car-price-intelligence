from pathlib import Path

import pandas as pd

from data.train import train_candidate_model


def test_train_candidate_model_runs_fast(tmp_path):
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed_car_listings.csv"
    if not data_path.exists():
        raise FileNotFoundError("processed dataset is missing; run features.py first")

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_candidate_model(
        data_path=str(data_path),
        candidate_model_path=str(model_dir / "candidate_car_price_model.pkl"),
        preprocessor_path=str(model_dir / "preprocessor.pkl"),
        n_trials=1,
        cv_folds=2,
    )

    assert (model_dir / "candidate_car_price_model.pkl").exists()
    assert (model_dir / "preprocessor.pkl").exists()
