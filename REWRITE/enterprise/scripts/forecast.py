"""CLI: run forecast for a ticker."""
from __future__ import annotations
import argparse
from pathlib import Path

from src.models.models_package import LSTMForecaster, ModelConfig, LSTMModel
from src.data.ingestion import DataIngestor
from src.data.features import FEATURE_COLS

import torch
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()
    cfg = ModelConfig(ticker=args.ticker, future_days=args.horizon)
    fc = LSTMForecaster(cfg)
    ing = DataIngestor()
    raw = ing.fetch(args.ticker)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(ing.fe.transform(raw)[FEATURE_COLS].values)
    ckpt = Path(f"models/checkpoints/{args.ticker.replace('-','_').replace('=','_')}.pt")
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}")
        return
    model = LSTMModel(input_size=8, hidden_size=cfg.hidden_size,
                      num_layers=cfg.num_layers, dropout=cfg.initial_dropout)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    fu, _, lo, hi = fc._run_monte_carlo_rollout(model, scaler,
        scaler.transform(ing.fe.transform(raw)[FEATURE_COLS].values[-cfg.prediction_days*3:]), args.horizon)
    dates = pd.date_range(start=raw.index[-1] + pd.Timedelta(days=1), periods=args.horizon, freq="B")
    for d, p, l, h in zip(dates, fu, lo, hi):
        print(f"{d.date()}: ${p:,.2f}  [${l:,.2f} – ${h:,.2f}]")


if __name__ == "__main__":
    main()
