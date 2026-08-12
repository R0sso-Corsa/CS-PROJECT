"""
Data ingestion layer.

Reuses the yfinance data acquisition pattern from the existing scripts.
Produces standardised OHLCV DataFrames.
"""

from __future__ import annotations

import sys
from pathlib import Path
SCRIPT_PATH = Path(__file__).resolve()
# Walk up from this file's location until we find the directory that contains 'src/'
PROJECT_ROOT = SCRIPT_PATH
while PROJECT_ROOT.parent != PROJECT_ROOT:
    if (PROJECT_ROOT / "src").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import yfinance as yf

from src.data.features import FeatureEngineer, FEATURE_COLS

logger = logging.getLogger(__name__)


class DataIngestor:
    """Download and cache market data for one or more tickers."""

    def __init__(self, cache_dir: str | Path = "./data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fe = FeatureEngineer()

    def fetch(
        self,
        ticker: str,
        start: str = "2017-01-01",
        end: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        end = end or datetime.now().strftime("%Y-%m-%d")
        cache_path = self.cache_dir / f"{ticker.replace('-', '_').replace('=', '_')}_{end}.parquet"
        if cache_path.exists() and not force_refresh:
            raw = pd.read_parquet(cache_path)
            logger.info("Loaded from cache: %s", cache_path)
        else:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True)
            if raw.empty:
                raise RuntimeError(f"No data returned for {ticker}")
            try:
                raw.to_parquet(cache_path)
            except ImportError:
                pass
            logger.info("Downloaded %s rows for %s", len(raw), ticker)
        df = self.fe.transform(raw)
        return df

    def fetch_multi(self, tickers: list[str], **kwargs) -> dict[str, pd.DataFrame]:
        return {t: self.fetch(t, **kwargs) for t in tickers}

    @staticmethod
    def get_latest_price(ticker: str) -> float:
        df = yf.download(ticker, period="1d", auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"No data for {ticker}")
        return float(df["Close"].iloc[-1])
