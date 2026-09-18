import pandas as pd


def process_features(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    df = df.copy()

    # 1. Vehicle Age
    df["vehicle_age"] = current_year - df["year"]
    df["vehicle_age"] = df["vehicle_age"].apply(lambda x: max(x, 1))

    # 2. Mileage Per Year
    df["mileage_per_year"] = df["mileage"] / df["vehicle_age"]

    # 3. Engine Size Numerical Extraction (e.g. '2000 cc' -> 2.0)
    df["engine_size"] = (
        df["engine"].str.extract(r"(\d+)").astype(float) / 1000.0
    )

    # 4. Luxury Brand Indicator Flag
    luxury_brands = {
        "Mercedes-Benz",
        "BMW",
        "Audi",
        "Lexus",
        "Porsche",
        "Range Rover",
        "Jaguar",
    }
    df["luxury_brand"] = df["make"].isin(luxury_brands).astype(int)

    # 5. Automatic Transmission Flag
    df["automatic_flag"] = (
        df["transmission"].str.contains("Auto", case=False, na=False).astype(int)
    )

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw_car_listings.csv")
    df_clean = process_features(df)
    df_clean.to_csv("data/processed_car_listings.csv", index=False)
    print("Features engineered and saved to 'data/processed_car_listings.csv'.")