"""Mean-reversion strategy: signal = sign(Bollinger z-score)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .recursive import Signal

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """Bollinger Band z-score signal."""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def generate(self, df: pd.DataFrame) -> Signal:
        close = df["Close"].values
        if len(close) < self.window + 5:
            return Signal(ticker="", side=0, confidence=0.0,
                          meta={"reason": "insufficient data"})

        sma  = pd.Series(close).rolling(self.window).mean().values
        std  = pd.Series(close).rolling(self.window).std(ddof=0).values
        z    = (close[-1] - sma[-1]) / (std[-1] + 1e-9)
        side = -1 if z > self.num_std else 1 if z < -self.num_std else 0
        conf = min(abs(z) / (self.num_std * 2), 1.0)
        return Signal(ticker="", side=side, confidence=conf,
                      meta={"zscore": round(float(z), 4), "sma": round(float(sma[-1]), 4)})
