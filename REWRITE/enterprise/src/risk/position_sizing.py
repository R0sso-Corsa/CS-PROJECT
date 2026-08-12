"""
Position sizing engine.

Implements multiple sizing models:
- Fixed fractional (risk_parity)
- Kelly Criterion with fractional Kelly (cap at cfg max)
- Volatility targeting (ATR-based)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionSizer:
    """Multi-method position sizer."""

    capital: float = 100_000.0
    max_position_pct: float = 0.20
    method: Literal["risk_parity", "kelly", "volatility"] = "risk_parity"
    kelly_cap: float = 0.25

    def size(self, signal_confidence: float, atr: float,
             current_price: float, stop_loss: float) -> dict:
        """Return sizing decision dict."""
        if self.method == "risk_parity":
            risk_amount = self.capital * 0.01  # 1% risk per trade
            dist = abs(current_price - stop_loss)
            if dist == 0:
                units = 0.0
            else:
                units = risk_amount / dist
            max_units = (self.capital * self.max_position_pct) / current_price
            units = min(units, max_units)
            return {"units": round(units, 6), "method": "risk_parity",
                    "risk_amount": risk_amount}

        elif self.method == "kelly":
            kelly = KellyCriterion.calculate(
                win_prob=signal_confidence,
                win_loss_ratio=2.0,  # default 2:1 reward
            )[0]
            kelly = min(kelly, self.kelly_cap)
            alloc = self.capital * kelly * self.max_position_pct
            units = alloc / current_price
            return {"units": round(units, 6), "method": "kelly", "kelly_fraction": kelly}

        elif self.method == "volatility":
            vol_target = 0.15  # 15% annual target
            daily_vol = atr / current_price if current_price else 0
            if daily_vol == 0:
                return {"units": 0.0, "method": "volatility", "daily_vol": 0.0}
            alloc = (vol_target / daily_vol) * self.capital * 0.1
            units = alloc / current_price
            return {"units": round(units, 6), "method": "volatility",
                    "daily_vol": round(daily_vol, 6)}

        raise ValueError(f"Unknown sizing method: {self.method}")


class KellyCriterion:
    @staticmethod
    def calculate(win_prob: float, win_loss_ratio: float,
                  fraction: float = 0.5) -> tuple[float, float]:
        """Calculate Kelly fraction.

        Returns (kelly_fraction, expected_growth_rate).
        """
        if win_prob <= 0 or win_prob >= 1 or win_loss_ratio <= 0:
            return 0.0, 0.0
        k = win_prob - ((1 - win_prob) / win_loss_ratio)
        k = k * fraction  # fractional Kelly
        return max(0.0, k), win_prob * (1 + k) + (1 - win_prob) * (1 - k) - 1
