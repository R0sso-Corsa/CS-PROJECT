"""Trend-following strategy: signal = sign(MACD_slope + RSI_momentum)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .recursive import Signal

logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    """MACD + RSI momentum signal."""

    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def generate(self, df: pd.DataFrame) -> Signal:
        close = df["Close"].values
        if len(close) < self.lookback + 30:
            return Signal(ticker=str(df.index[0]) if len(df) else "",
                          side=0, confidence=0.0, meta={"reason": "insufficient data"})

        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd  = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        delta  = pd.Series(close).diff()
        gain   = delta.clip(lower=0)
        loss   = (-delta).clip(upper=0)
        avg_gain = gain.ewm(span=self.lookback, adjust=False).mean().values
        avg_loss = loss.ewm(span=self.lookback, adjust=False).mean().values
        rs  = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain),
                        where=avg_loss != 0)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        macd_slope  = macd[-1] - macd[-2]
        rsi_val     = rsi[-1]
        score       = macd_slope + max(0, rsi_val - 50) / 50.0 - max(0, 50 - rsi_val) / 50.0
        side   = 1 if score > 0.15 else -1 if score < -0.15 else 0
        conf   = min(abs(score) / 2.0, 1.0)
        return Signal(ticker="", side=side, confidence=conf,
                      meta={"macd_slope": float(macd_slope), "rsi": float(rsi_val)})
