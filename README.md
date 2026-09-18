# Used Car Price Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/ML-XGBoost-FFB000?style=for-the-badge&logo=xgboost" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit" />
</p>

A machine learning project for estimating fair used-car prices and identifying bargain deals in online listings. The system combines data processing, feature engineering, price prediction, and an interactive Streamlit dashboard for buyer-friendly market valuation.

## Overview

Used Car Price Intelligence analyzes vehicle attributes such as make, model, year, mileage, engine size, transmission, fuel type, condition, and location to predict a fair market value. It also compares the asking price against the model estimate and flags deals as:

- Bargain
- Fair price
- Overpriced

This project is ideal for used-car marketplaces, price comparison tools, and personal vehicle-value analysis.

## Key Features

- Data cleaning and preparation for vehicle listings
- Feature engineering for age, mileage-per-year, engine size, luxury-brand flag, and automatic transmission flag
- XGBoost regression model for price prediction
- Optuna-based hyperparameter tuning
- Streamlit web app for quick interactive valuation
- Regression-style testing and validation workflow

## Project Structure

- `features.py` — feature engineering pipeline for processed listing data
- `data/raw_car_listings.csv` — raw vehicle listing dataset
- `data/processed_car_listings.csv` — engineered data used to train the model
- `data/train.py` — model training and artifact creation pipeline
- `data/app.py` — Streamlit app for market valuation and bargain detection
- `models/` — saved trained model and preprocessing objects
- `tests/` — validation tests for scraper and training flow

## Tech Stack

- Python
- pandas
- scikit-learn
- XGBoost
- Optuna
- Streamlit
- pytest

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/Avire123/used-car-price-intelligence.git
cd used-car-price-intelligence
```

### 2) Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Prepare the dataset

```bash
python features.py
```

### 5) Train the model

```bash
python data/train.py
```

### 6) Run the app

```bash
python -m streamlit run data/app.py --server.headless true --server.port 8501
```

Then open the local browser at:

```text
http://localhost:8501
```

## Example Use Case

A buyer can enter a vehicle's make, model, mileage, age, condition, fuel type, and asking price to see:

- the estimated fair market price,
- the gap between asking and predicted value,
- whether the listing is a bargain or overpriced.

## Notes

This repository includes a synthetic fallback dataset for demo or placeholder-data environments. For production deployment, it can be extended with a real listing source or marketplace scraper.

## License

This project is intended for educational, demonstration, and portfolio use.

## Repository Status

- Model training pipeline included
- Streamlit UI included
- GitHub repository published and ready for collaboration
