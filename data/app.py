import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Used Car Price Intelligence",
    page_icon="🚗",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


model, preprocessor = load_artifacts()

# Header
st.title("🚗 Used Car Price Intelligence & Bargain Finder")
st.markdown(
    "Predict fair market values for used vehicle listings and discover high-value deals using Machine Learning."
)

if model is None or preprocessor is None:
    st.error(
        "Model or Preprocessor artifacts not found! Please run `python train.py` first."
    )
    st.stop()

# 2. Input Form Layout
st.sidebar.header("🔧 Vehicle Specifications")

makes = [
    "Toyota",
    "Subaru",
    "Mazda",
    "Nissan",
    "Honda",
    "Mercedes-Benz",
    "BMW",
    "Audi",
    "Volkswagen",
    "Lexus",
]
make = st.sidebar.selectbox("Make", sorted(makes))

models_dict = {
    "Toyota": ["Premio", "Fielder", "Prado", "Vitz", "RAV4", "Harrier", "Land Cruiser"],
    "Subaru": ["Outback", "Forester", "Impreza", "XV", "Legacy"],
    "Mazda": ["CX-5", "Demio", "Axela", "Atenza"],
    "Nissan": ["X-Trail", "Note", "Dualis", "Juke", "Serena"],
    "Honda": ["Fit", "CR-V", "Vezel", "Civic"],
    "Mercedes-Benz": ["C200", "E250", "GLE", "C180"],
    "BMW": ["320i", "X5", "520i", "X3"],
    "Audi": ["A4", "Q5", "A6", "Q3"],
    "Volkswagen": ["Golf", "Tiguan", "Passat"],
    "Lexus": ["RX200t", "NX200t", "LX570"],
}

available_models = models_dict.get(make, ["Other"])
selected_model = st.sidebar.selectbox("Model", sorted(available_models))

year = st.sidebar.slider("Year of Manufacture", 2005, 2026, 2018)
mileage = st.sidebar.number_input(
    "Mileage (KM)", min_value=1000, max_value=300000, value=75000, step=5000
)
engine_cc = st.sidebar.number_input(
    "Engine Capacity (cc)", min_value=600, max_value=6000, value=1800, step=100
)

transmission = st.sidebar.radio("Transmission", ["Automatic", "Manual"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
condition = st.sidebar.selectbox(
    "Condition", ["Foreign Used", "Locally Used", "Brand New"]
)
location = st.sidebar.selectbox(
    "Location", ["Nairobi", "Mombasa", "Nakuru", "Eldoret", "Kisumu"]
)

listing_price = st.sidebar.number_input(
    "Seller's Asking Price (KSh)",
    min_value=100000,
    max_value=30000000,
    value=2200000,
    step=50000,
)

# 3. Feature Engineering Transformation
current_year = 2026
vehicle_age = max(current_year - year, 1)
mileage_per_year = mileage / vehicle_age
engine_size = engine_cc / 1000.0

luxury_brands = {
    "Mercedes-Benz",
    "BMW",
    "Audi",
    "Lexus",
    "Porsche",
    "Range Rover",
    "Jaguar",
}
luxury_brand = 1 if make in luxury_brands else 0
automatic_flag = 1 if transmission == "Automatic" else 0

input_df = pd.DataFrame(
    [
        {
            "make": make,
            "model": selected_model,
            "condition": condition,
            "fuel_type": fuel_type,
            "location": location,
            "vehicle_age": vehicle_age,
            "mileage": mileage,
            "mileage_per_year": mileage_per_year,
            "engine_size": engine_size,
            "luxury_brand": luxury_brand,
            "automatic_flag": automatic_flag,
        }
    ]
)

# 4. Inference & Market Valuation Display
st.subheader("📊 Fair Market Valuation")

col1, col2, col3 = st.columns(3)

# Transform features and predict
input_prep = preprocessor.transform(input_df)
predicted_price = float(model.predict(input_prep)[0])

price_diff = listing_price - predicted_price
pct_diff = (price_diff / predicted_price) * 100

with col1:
    st.metric(
        label="Predicted Market Value",
        value=f"KSh {predicted_price:,.0f}",
    )

with col2:
    st.metric(
        label="Seller Asking Price",
        value=f"KSh {listing_price:,.0f}",
    )

with col3:
    st.metric(
        label="Price Variance",
        value=f"KSh {abs(price_diff):,.0f}",
        delta=f"{pct_diff:+.1f}% vs Estimated Value",
        delta_color="inverse",
    )

# 5. Bargain Classification Card
st.markdown("---")
st.subheader("💡 Deal Evaluation")

if pct_diff <= -8.0:
    st.success(
        f"🎯 **BARGAIN DEAL DETECTED!**\n\nThis listing is priced **{abs(pct_diff):.1f}% below** the estimated market value (Save ~KSh {abs(price_diff):,.0f})."
    )
elif pct_diff >= 8.0:
    st.error(
        f"⚠️ **OVERPRICED LISTING!**\n\nThis vehicle is listed **{pct_diff:.1f}% above** the predicted market price. Consider negotiating at least KSh {abs(price_diff):,.0f} off."
    )
else:
    st.info(
        f"⚖️ **FAIR MARKET PRICE**\n\nThe asking price aligns within normal market variance range (±8%)."
    )

# 6. Feature Summary Table
with st.expander("🔍 View Processed Feature Inputs"):
    st.dataframe(input_df)