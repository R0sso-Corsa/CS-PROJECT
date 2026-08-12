FEATURE_COLS = [
    "Close", "Volume", "SMA_14", "RSI_14", "MACD",
    "Signal_Line", "Upper_BB", "Lower_BB",
]


"""
Feature engineering pipeline.

Matches exactly the 8-column feature set produced by the original
``pytorch_train_cpp.py`` ``add_features()`` function so model checkpoints
-trained before this refactor load without modification.
"""


import logging
from typing import Optional

import pandas as pd

FEATURE_COLS = [
    "Close", "Volume", "SMA_14", "RSI_14", "MACD",
    "Signal_Line", "Upper_BB", "Lower_BB",
]

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Stateful feature pipeline — wraps the logic from pytorch_train_cpp.py."""

    def __init__(self, ticker: str = "BTC-USD"):
        self.ticker = ticker

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        out["SMA_14"] = out["Close"].rolling(window=14).mean()
        delta = out["Close"].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        ag = gain.ewm(com=13, adjust=False).mean()
        al = loss.ewm(com=13, adjust=False).mean()
        with pd.option_context("mode.chained_assignment", None):
            out["RSI_14"] = 100 - (100 / (1 + ag / al))
        out["MACD"] = (
            out["Close"].ewm(span=12, adjust=False).mean()
            - out["Close"].ewm(span=26, adjust=False).mean()
        )
        out["Signal_Line"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["20_SMA"] = out["Close"].rolling(window=20).mean()
        out["Std_Dev"] = out["Close"].rolling(window=20).std()
        out["Upper_BB"] = out["20_SMA"] + (out["Std_Dev"] * 2)
        out["Lower_BB"] = out["20_SMA"] - (out["Std_Dev"] * 2)
        out.ffill(inplace=True)
        out.bfill(inplace=True)
        logger.info("Feature transform complete: %s", out.shape)
        return out

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self.transform(df)
        return prepared[FEATURE_COLS]

    @staticmethod
    def infill_na(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Fill remaining NAs in a subset of columns."""
        sub = df[cols].ffill().bfill()
        sub.fillna(0, inplace=True)
        df[cols] = sub
        return df
