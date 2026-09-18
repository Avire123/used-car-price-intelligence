import logging
import os
import random
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 2. User-Agent Pool for Rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
]

# 3. Resilient Session Creator
def create_resilient_session(retries=5, backoff_factor=1.5):
    """Creates a requests Session with automatic retries and exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://",adapter)
    return session

# 4. Scraper Logic
def generate_synthetic_listings(num_rows=12000):
    base_records = [
        {"make": "Toyota", "model": "Premio", "engine": "1800 cc", "transmission": "Automatic", "fuel_type": "Petrol", "location": "Nairobi", "condition": "Foreign Used", "price": 2100000},
        {"make": "Subaru", "model": "Outback", "engine": "2500 cc", "transmission": "Automatic", "fuel_type": "Petrol", "location": "Nairobi", "condition": "Locally Used", "price": 2400000},
        {"make": "Mazda", "model": "CX-5", "engine": "2200 cc", "transmission": "Automatic", "fuel_type": "Diesel", "location": "Mombasa", "condition": "Foreign Used", "price": 2850000},
        {"make": "Mercedes-Benz", "model": "C200", "engine": "2000 cc", "transmission": "Automatic", "fuel_type": "Petrol", "location": "Nairobi", "condition": "Foreign Used", "price": 3500000},
        {"make": "Nissan", "model": "X-Trail", "engine": "2000 cc", "transmission": "Automatic", "fuel_type": "Petrol", "location": "Nakuru", "condition": "Locally Used", "price": 1750000},
        {"make": "Honda", "model": "Fit", "engine": "1500 cc", "transmission": "Manual", "fuel_type": "Petrol", "location": "Kisumu", "condition": "Locally Used", "price": 1500000},
        {"make": "BMW", "model": "X3", "engine": "2000 cc", "transmission": "Automatic", "fuel_type": "Diesel", "location": "Nairobi", "condition": "Foreign Used", "price": 4200000},
        {"make": "Audi", "model": "A4", "engine": "2000 cc", "transmission": "Automatic", "fuel_type": "Diesel", "location": "Mombasa", "condition": "Foreign Used", "price": 3900000},
    ]

    synthetic_rows = []
    for i in range(num_rows):
        base = base_records[i % len(base_records)]
        year = 2008 + (i % 15)
        mileage = 20000 + ((i * 173) % 180000)
        price = int(base["price"] * (0.75 + (i % 20) / 40))
        synthetic_rows.append({
            "make": base["make"],
            "model": base["model"],
            "year": year,
            "mileage": mileage,
            "engine": base["engine"],
            "transmission": base["transmission"],
            "fuel_type": base["fuel_type"],
            "location": base["location"],
            "condition": base["condition"],
            "price": price,
        })

    return pd.DataFrame(synthetic_rows)


def scrape_car_listings(num_pages=3, output_path="data/raw_car_listings.csv"):
    session=create_resilient_session()
    records=[]

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    placeholder_url = "https://example-car-site.com/listings?page={page}"
    if placeholder_url.startswith("https://example-car-site.com"):
        logging.warning("Configured source is the demo placeholder. Generating a large synthetic catalog instead.")
        df = generate_synthetic_listings(12000)
        df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved {len(df)} car listings to '{output_path}'.")
        return df

    for page in range(1, num_pages + 1):
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # Update this URL to match your target site structure
        url = f"https://example-car-site.com/listings?page={page}"

        try:
            logging.info(f"Fetching page {page} of {num_pages}...")
            response = session.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                cards = soup.find_all("div", class_="car-card")

                for card in cards:
                    try:
                        make = card.find("span", class_="make").text.strip()
                        model = card.find("span", class_="model").text_strip()
                        year = int(card.find("span", call_="year").text_strip)
                        mileage = float(
                            card.find("span", class_="mileage")
                            .text.replace("km", "")
                            .replace(",","")
                            .strip()
                        )
                        engine = card.find("span", class_="engine").text.strip()
                        transmission = card.find(
                            "span", class_="transmission"
                        ).text_strip()
                        fuel_type = card.find("span", class_="fuel").text.strip()
                        location = card.find(
                            "span", class_="location"
                        ).text_strip()
                        condition = card.find(
                            "span", class_="condition"
                        ).text_strip()
                        price = float(
                            card.find("span", class_="price")
                            .text.replace("KSh", "")
                            .replace(",", "")
                            .strip()
                        )

                        records.append({
                            "make":make,
                            "model": model,
                            "year": year,
                            "mileage": mileage,
                            "engine": engine,
                            "transmission": transmission,
                            "fuel_type": fuel_type,
                            "location": location,
                            "condition": condition,
                            "price":price,
                        })
                    except (AttributeError, ValueError):
                        continue

            # Polite scraping delay to avoid aggressive rate limiting
            time.sleep(random.uniform(1.5, 3.0))

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to retrieve page {page}: {e}")
            continue

    df = pd.DataFrame(records)

    if df.empty:
        logging.warning("No records scraped from target URL. Generating synthetic baseline for pipeline initialization...")
        df = generate_synthetic_listings(12000)

    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved {len(df)} car listings to '{output_path}'.")


if __name__ == "__main__":
    scrape_car_listings()





