"""
Event-driven backtesting engine.

Runs the recursive strategy against historical data, tracks equity, enforces
risk guardrails, and produces a BacktestResult with full metrics.
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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data.ingestion import DataIngestor
from src.data.features import FeatureEngineer, FEATURE_COLS
from src.models import LSTMForecaster, ModelConfig
from src.strategies.recursive import RecursiveStrategy
from src.risk.guardrails import RiskGuardrails

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Full backtest report."""
    ticker: str
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Backtest Results — {self.ticker}", "-" * 40]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class BacktestEngine:
    """Event-driven backtest over historical OHLCV data."""

    def __init__(
        self,
        ticker: str,
        capital: float = 100_000.0,
        strategy: RecursiveStrategy | None = None,
        guardrails: RiskGuardrails | None = None,
        model_cfg: ModelConfig | None = None,
    ):
        self.ticker = ticker
        self.capital = capital
        self.strategy = strategy or RecursiveStrategy(
            strategies=[], learning_rate=0.1
        )
        self.guardrails = guardrails or RiskGuardrails(initial_capital=capital)
        self.model = LSTMForecaster(model_cfg) if model_cfg else None
        self.ingestor = DataIngestor()

    def _get_price_at(self, df: pd.DataFrame, idx) -> float:
        return float(df.loc[idx, "Close"]) if idx in df.index else 0.0

    def _get_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        if len(df) < window + 1:
            return 0.0
        sub = df.tail(window + 1)
        high = sub["High"].values
        low = sub["Low"].values
        close = sub["Close"].values
        tr = np.maximum(high[1:] - low[1:], np.maximum(
            abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        return float(np.mean(tr)) if len(tr) > 0 else 0.0

    def run(self, start: str = "2024-01-01", end: str | None = None) -> BacktestResult:
        end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
        logger.info("Backtesting %s from %s to %s", self.ticker, start, end)
        df = self.ingestor.fetch(self.ticker)
        mask = (df.index >= start) & (df.index <= end)
        data = df.loc[mask].copy()
        if data.empty:
            raise RuntimeError("No data in backtest window")

        equity = self.capital
        peak = equity
        equity_curve = []
        trades = []
        step = max(self.model.cfg.prediction_days if self.model else 30, 5)
        for i in range(step, len(data), step):
            window = data.iloc[max(0, i - 200):i]
            if len(window) < 30:
                continue
            forecast_df = None
            if self.model:
                try:
                    fp, _, lo, hi = self.model._run_monte_carlo_rollout(
                        self.model.model, self.model.scaler,
                        self.model.scaler.transform(window[FEATURE_COLS].values),
                        future_day=step,
                    )
                    forecast_df = pd.DataFrame({
                        "Predicted_Price": fp, "CI95_lower": lo, "CI95_upper": hi,
                    })
                except Exception:
                    pass
            sig = self.strategy.generate(window, forecast_df)
            entry = float(window["Close"].iloc[-1])
            atr = self._get_atr(window)

            if sig.side != 0 and self.guardrails.can_open(self.ticker, sig.side):
                sizer = type("obj", (object,), {
                    "capital": equity, "max_position_pct": 0.20,
                })()
                sz = self.guardrails.open_position(self.ticker, sig.side, entry, 0.0)
                sl = self.guardrails.get_stop_loss(atr, entry, sig.side)
                future = data.iloc[i:min(i + step, len(data))]
                exit_price = None
                reason = "timeout"
                for _, row in future.iterrows():
                    if sig.side == 1 and row["Low"] <= sl:
                        exit_price = sl
                        reason = "stop_loss"
                        break
                    elif sig.side == -1 and row["High"] >= sl:
                        exit_price = sl
                        reason = "stop_loss"
                        break
                    elif sig.side == 1 and sig.confidence > 0.7:
                        tp = entry + 3.5 * atr
                        if row["High"] >= tp:
                            exit_price = tp
                            reason = "take_profit"
                            break
                    if reason == "timeout":
                        exit_price = float(row["Close"])
                        reason = "expiry"
                if exit_price is not None:
                    trade = self.guardrails.close_position(self.ticker, exit_price, reason)
                    if trade:
                        equity += trade["pnl"]
                        trades.append(trade)
                        self.strategy.update_history(sig, trade["pnl"])
            peak = max(peak, equity)
            equity_curve.append({"date": data.index[i], "equity": equity})

        ec = pd.DataFrame(equity_curve).set_index("date")["equity"]
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["ticker", "side", "entry", "exit", "pnl", "reason"])
        metrics = self._compute_metrics(ec, trades_df)
        return BacktestResult(ticker=self.ticker, trades=trades_df,
                              equity_curve=ec, metrics=metrics)

    @staticmethod
    def _compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
        if equity.empty:
            return {}
        rets = equity.pct_change().dropna()
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
        cummax = equity.cummax()
        mdd = float(((cummax - equity) / cummax).max()) if not equity.empty else 0.0
        wr = float((trades["pnl"] > 0).mean()) if len(trades) else 0.0
        pf = float(
            trades.loc[trades["pnl"] > 0, "pnl"].sum() /
            abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
        ) if len(trades) and (trades["pnl"] < 0).any() else float("inf")
        return {
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(mdd * 100, 2),
            "win_rate_pct": round(wr * 100, 2),
            "profit_factor": round(pf, 4) if pf != float("inf") else "inf",
            "total_trades": len(trades),
            "final_equity": round(float(equity.iloc[-1]), 2),
        }
