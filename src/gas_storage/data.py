"""Henry Hub spot price data (FRED series DHHNGSP, USD/MMBtu)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_fred(series: str, cache: Path) -> pd.Series:
    """Download a FRED daily series (cached to CSV) and return it without missing days."""
    if not cache.exists():
        # curl rather than urllib: some macOS Python builds ship without CA certificates
        raw = subprocess.run(["curl", "-sL", "--max-time", "60", FRED_URL.format(series=series)],
                             check=True, capture_output=True, text=True).stdout
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(raw)
    df = pd.read_csv(cache, parse_dates=["observation_date"], na_values=["."])
    s = df.set_index("observation_date")[series].astype(float).dropna()
    s.index.name = "date"
    return s


def henry_hub(data_dir: Path) -> pd.Series:
    return fetch_fred("DHHNGSP", data_dir / "henry_hub_daily.csv").rename("henry_hub")


def year_fraction(dates: pd.DatetimeIndex) -> pd.Index:
    """Calendar time in years since 2000-01-01, so weekends and holidays get their true length."""
    return (dates - pd.Timestamp("2000-01-01")).days / 365.25
