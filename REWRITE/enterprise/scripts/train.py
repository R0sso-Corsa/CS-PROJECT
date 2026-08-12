"""CLI: train a model for one or more tickers."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.utils.config import load_config
from src.models.models_package import LSTMForecaster, ModelConfig
from src.data.ingestion import DataIngestor


def main():
    parser = argparse.ArgumentParser(description="Train LSTM forecaster")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()
    cfg = ModelConfig(ticker=args.ticker, epochs=args.epochs, future_days=args.horizon)
    fc = LSTMForecaster(cfg)
    res = fc.train(args.ticker)
    print(f"Trained {args.ticker}  RMSE={res['rmse']:.2f}  in {res['train_seconds']}s")
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
