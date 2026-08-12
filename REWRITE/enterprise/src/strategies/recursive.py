"""
Recursive trading strategy engine.

Architecture
------------
A ``RecursiveStrategy`` wraps one or more sub-strategies. The sub-strategies
produce a raw signal (+1/-1/0), the recursive layer re-evaluates those
signals against prior outcomes and updates confidence weights.  The risk
module then applies position sizing before orders are generated.

Strategies
----------
1. ``TrendFollowingStrategy`` — signal = sign(MACD_slope + RSI_momentum)
2. ``MeanReversionStrategy`` — signal = sign(Bollinger_zscore)
3. ``RecursiveStrategy`` — weighted ensemble with dynamic confidence from PnL

Both are derived from ``BaseStrategy``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    ticker: str
    side: int  # +1 long, -1 short, 0 flat
    confidence: float
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Signal({self.ticker}, side={self.side}, conf={self.confidence:.2f})"


class BaseStrategy(ABC):
    """Abstract trading strategy."""

    def __init__(self, name: str):
        self.name = name
        self.history: list[Signal] = []

    @abstractmethod
    def generate(self, df: pd.DataFrame, forecast: pd.DataFrame | None = None) -> Signal:
        """Produce a signal given recent data and optional forecast."""
        pass

    def update_history(self, signal: Signal, realised_pnl: float) -> None:
        self.history.append(signal)
        logger.debug("%s history updated: PnL=%.4f", self.name, realised_pnl)


class TrendFollowingStrategy(BaseStrategy):
    """MACD + RSI momentum signal."""

    def __init__(self, rsi_threshold: int = 50):
        super().__init__("TrendFollowing")
        self.rsi_threshold = rsi_threshold

    def generate(self, df: pd.DataFrame, forecast=None) -> Signal:
        row = df.iloc[-1]
        sig = 0
        conf = 0.5
        if row["MACD"] > row["Signal_Line"]:
            sig += 1
            conf += 0.1
        else:
            sig -= 1
            conf += 0.1
        if row["RSI_14"] > self.rsi_threshold:
            sig += 1
        elif row["RSI_14"] < (100 - self.rsi_threshold):
            sig -= 1
        sig = int(np.clip(sig, -1, 1))
        conf = min(conf, 1.0)
        return Signal(ticker=df.attrs.get("ticker", ""), side=sig, confidence=conf,
                      meta={"macd": row["MACD"], "rsi": row["RSI_14"]})


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band z-score signal."""

    def __init__(self, z_entry: float = 1.5, z_exit: float = 0.5):
        super().__init__("MeanReversion")
        self.z_entry = z_entry
        self.z_exit = z_exit

    def generate(self, df: pd.DataFrame, forecast=None) -> Signal:
        close = df["Close"].iloc[-1]
        sma = df["20_SMA"].iloc[-1]
        std = df["Std_Dev"].iloc[-1]
        if std == 0 or np.isnan(std):
            return Signal(ticker=df.attrs.get("ticker", ""), side=0, confidence=0.0)
        z = (close - sma) / std
        sig = 0
        if z < -self.z_entry:
            sig = 1  # expect reversion upward
        elif z > self.z_entry:
            sig = -1
        conf = min(abs(z) / 2.0, 1.0)
        return Signal(ticker=df.attrs.get("ticker", ""), side=sig, confidence=round(conf, 4),
                      meta={"z_score": round(float(z), 4)})


class RecursiveStrategy(BaseStrategy):
    """Weighted ensemble that updates weights from past PnL performance.

    Parameters
    ----------
    strategies: list of BaseStrategy subclasses (not instances).
    learning_rate: how fast weights adapt after each outcome.
    """

    def __init__(self, strategies: list[type[BaseStrategy]], learning_rate: float = 0.1):
        super().__init__("RecursiveEnsemble")
        self.sub_strategies = [s() for s in strategies]
        self.weights = np.ones(len(self.sub_strategies)) / len(self.sub_strategies)
        self.lr = learning_rate
        self._last_positions: dict[str, int] = {}
        self._entry_prices: dict[str, float] = {}

    def generate(self, df: pd.DataFrame, forecast: pd.DataFrame | None = None) -> Signal:
        ticker = df.attrs.get("ticker", "")
        raw = np.array([s.generate(df, forecast).side for s in self.sub_strategies], dtype=float)
        raw_sig = np.dot(self.weights, raw)
        side = int(np.sign(raw_sig)) if abs(raw_sig) > 0.3 else 0
        conf = float(np.clip(abs(raw_sig), 0.0, 1.0))
        return Signal(ticker=ticker, side=side, confidence=round(conf, 4),
                      meta={s.name: round(float(w), 4)
                            for s, w in zip(self.sub_strategies, self.weights)})

    def update_weights(self, ticker: str, pnl: float) -> None:
        """Reinforce strategies that made money, penalise losing ones."""
        ticker = ticker or "unknown"
        outs = []
        for i, s in enumerate(self.sub_strategies):
            hist_sigs = [h.side for h in s.history if h.ticker == ticker]
            outs.append(hist_sigs[-1] if hist_sigs else 0)
        outs = np.array(outs, dtype=float)
        reward = outs * pnl
        self.weights = self.weights + self.lr * reward
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        logger.info("Updated recursive weights: %s", {s.name: round(w, 4) for s, w in zip(self.sub_strategies, self.weights)})
