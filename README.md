# Used Car Price Intelligence

Used Car Price Intelligence is a machine learning project that estimates fair market values for used vehicles and highlights bargain deals. It combines feature engineering, model training, and a Streamlit web app for interactive valuation.

The project is designed for the used-car market and uses vehicle attributes such as make, model, year, mileage, engine size, transmission, fuel type, condition, and location to predict price. It also classifies a listing as a bargain, fair price, or overpriced listing based on the estimated market value.

## Features

- Data ingestion and cleaning for vehicle listing data
- Feature engineering for vehicle age, mileage per year, engine size, luxury branding, and automatic transmission flags
- XGBoost regression model for price prediction
- Hyperparameter tuning with Optuna
- Streamlit interface for price estimation and bargain detection
- Regression and data quality tests

## Project Structure

- `features.py` – feature engineering pipeline for processed vehicle data
- `data/raw_car_listings.csv` – raw listing dataset
- `data/processed_car_listings.csv` – engineered feature data used for training
- `data/train.py` – model training pipeline and artifact generation
- `data/app.py` – Streamlit app for used-car valuation
- `models/` – serialized trained model and preprocessing artifacts
- `tests/` – project validation tests

## Tech Stack

- Python
- pandas
- scikit-learn
- XGBoost
- Optuna
- Streamlit
- pytest

## Getting Started

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Generate processed listing data
5. Train the model
6. Launch the app

### Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Prepare the data

```bash
python features.py
```

### Train the model

```bash
python data/train.py
```

### Run the Streamlit app

```bash
python -m streamlit run data/app.py --server.headless true --server.port 8501
```

## Example Use Case

The app helps buyers and sellers estimate a fair market price for a vehicle and quickly decide whether a listing is a good deal. Users can adjust vehicle attributes and see the predicted value alongside the asking price.

## Notes

This repository includes a fallback synthetic dataset generation path for demo or placeholder listing environments. For production use, you can replace the synthetic data source with a live car marketplace scraper or a real listing dataset.

## License

This project is provided for educational and portfolio use.
