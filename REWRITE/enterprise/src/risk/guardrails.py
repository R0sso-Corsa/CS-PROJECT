"""
Risk guardrails: circuit breakers, max drawdown, VaR, stop-loss enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskGuardrails:
    """Enforce risk limits before orders are placed."""

    max_drawdown_pct: float = 0.15
    var_confidence: float = 0.95
    commission_rate: float = 0.001
    slippage_pct: float = 0.0005
    max_open_positions: int = 3
    initial_capital: float = 100_000.0

    # ── state ────────────────────────────────────────────────────────────────
    _peak: float = 0.0
    _open_positions: list[dict] | None = None
    _trades: list[dict] | None = None

    def __post_init__(self):
        self._peak = self.initial_capital
        self._open_positions = []
        self._trades = []

    def check_equity(self, current_equity: float) -> bool:
        """Return True if trading should be halted (drawdown breach)."""
        self._peak = max(self._peak, current_equity)
        dd = (self._peak - current_equity) / self._peak
        if dd >= self.max_drawdown_pct:
            logger.warning("Max drawdown breached: %.2f%% (limit %.2f%%)",
                           dd * 100, self.max_drawdown_pct * 100)
            return False
        return True

    def can_open(self, ticker: str, side: int) -> bool:
        if side == 0:
            return False
        if any(p["ticker"] == ticker for p in self._open_positions):
            logger.info("Already holding %s", ticker)
            return False
        if len(self._open_positions) >= self.max_open_positions:
            logger.info("Max open positions reached (%d).", self.max_open_positions)
            return False
        return True

    def open_position(self, ticker: str, side: int,
                      entry_price: float, units: float) -> dict:
        pos = {
            "ticker": ticker, "side": side,
            "entry": entry_price, "units": units,
            "stop_loss": 0.0, "take_profit": 0.0,
        }
        self._open_positions.append(pos)
        return pos

    def close_position(self, ticker: str, exit_price: float, reason: str = "") -> dict | None:
        for i, p in enumerate(self._open_positions):
            if p["ticker"] == ticker:
                pnl = p["side"] * p["units"] * (exit_price - p["entry"])
                pnl -= self.commission_rate * p["units"] * (p["entry"] + exit_price)
                pnl -= self.slippage_pct * p["units"] * exit_price
                trade = {**p, "exit": exit_price, "pnl": pnl, "reason": reason}
                self._trades.append(trade)
                self._open_positions.pop(i)
                logger.info("Closed %s: PnL=%.4f  reason=%s", ticker, pnl, reason)
                return trade
        return None

    def compute_var(self, returns: pd.Series | np.ndarray, horizon: int = 1) -> float:
        """Historical VaR at ``var_confidence`` for given horizon (in days)."""
        r = np.asarray(returns, dtype=np.float)
        if len(r) == 0:
            return 0.0
        alpha = 1.0 - self.var_confidence
        var = np.percentile(r, alpha * 100)
        return float(var * np.sqrt(horizon))

    def get_stop_loss(self, atr: float, entry: float, side: int,
                      mult: float = 2.0) -> float:
        """ATR-based stop-loss price."""
        if side == 1:
            return entry - mult * atr
        elif side == -1:
            return entry + mult * atr
        return entry

    def open_positions(self) -> list[dict]:
        return list(self._open_positions)

    def trade_summary(self) -> pd.DataFrame | None:
        if not self._trades:
            return None
        return pd.DataFrame(self._trades)


@dataclass
class RiskReport:
    """Summary statistics from RiskGuardrails._trades."""

    guardrail: "RiskGuardrails"

    @property
    def trades_df(self) -> pd.DataFrame | None:
        return self.guardrail.trade_summary()

    @property
    def total_pnl(self) -> float:
        df = self.trades_df
        return float(df["pnl"].sum()) if df is not None else 0.0

    @property
    def win_rate(self) -> float:
        df = self.trades_df
        if df is None or len(df) == 0:
            return 0.0
        return float((df["pnl"] > 0).mean())

    @property
    def profit_factor(self) -> float:
        df = self.trades_df
        if df is None or len(df) == 0:
            return 0.0
        gains = df.loc[df["pnl"] > 0, "pnl"].sum()
        losses = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
        return float(gains / losses) if losses > 0 else float("inf")
