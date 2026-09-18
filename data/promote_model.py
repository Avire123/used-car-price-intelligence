import os
import shutil


def promote_candidate_to_production(
    candidate_path="models/candidate_car_price_model.pkl",
    prod_path="models/car_price_model.pkl",
):
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(
            f"Candidate model not found at '{candidate_path}'. Cannot promote."
        )

    # Overwrite production model with candidate model
    shutil.copyfile(candidate_path, prod_path)
    print(f"Successfully promoted '{candidate_path}' -> '{prod_path}'.")


if __name__ == "__main__":
    promote_candidate_to_production()