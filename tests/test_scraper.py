import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRAPER = ROOT / "used-car-price-intelligence" / "scraper.py"
CSV_PATH = ROOT / "data" / "raw_car_listings.csv"


def test_scraper_generates_more_than_10000_listings():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, str(SCRAPER)], check=True, cwd=str(ROOT))

    df = pd.read_csv(CSV_PATH)
    assert len(df) > 10000
