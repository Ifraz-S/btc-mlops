"""Step 1 of the pipeline: download daily BTC-USD prices and save them as CSV.

Primary source: Yahoo Finance (yfinance library).
Backup source:  CoinGecko public API (last 365 days), used if Yahoo fails.

Run it directly:  python src/etl_fetch.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("etl_fetch")

COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def fetch_from_yfinance(ticker: str = config.TICKER, start: str = "2018-01-01") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")
    # Newer yfinance versions return two header rows (e.g. ('Close', 'BTC-USD')).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df[COLUMNS]


def fetch_from_coingecko(days: int = 365) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    resp = requests.get(
        url, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=30
    )
    resp.raise_for_status()
    payload = resp.json()
    prices = pd.DataFrame(payload["prices"], columns=["ts", "Close"])
    volumes = pd.DataFrame(payload["total_volumes"], columns=["ts", "Volume"])
    df = prices.merge(volumes, on="ts")
    df["Date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    # CoinGecko's free endpoint only gives one price per day, so we use it
    # for Open/High/Low too. Our features only rely on Close and Volume.
    df["Open"] = df["High"] = df["Low"] = df["Close"]
    return df[COLUMNS].drop_duplicates("Date", keep="last")


def fetch_btc_data(out_path: Path = config.RAW_DATA_PATH) -> Path:
    try:
        df = fetch_from_yfinance()
        source = "yfinance"
    except Exception as exc:  # network problems, rate limits, etc.
        log.warning("yfinance failed (%s). Falling back to CoinGecko.", exc)
        df = fetch_from_coingecko()
        source = "coingecko"

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(
        "Saved %d rows from %s (%s to %s) -> %s",
        len(df),
        source,
        df["Date"].min().date(),
        df["Date"].max().date(),
        out_path,
    )
    return out_path


if __name__ == "__main__":
    fetch_btc_data()
