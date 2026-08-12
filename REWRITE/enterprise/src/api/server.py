"""
FastAPI application — the REST interface to the trading engine.

Endpoints
---------
GET  /health                — liveness probe
GET  /api/v1/tickers        — list supported tickers with latest prices
POST /api/v1/forecast       — trigger forecast for a ticker
GET  /api/v1/forecast/{ticker} — get latest forecast result
POST /api/v1/train          — async train job for a ticker
GET  /api/v1/backtest/{ticker} — run and retrieve backtest
GET  /api/v1/metrics/summary — aggregate dashboard metrics
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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.data.ingestion import DataIngestor
from src.data.features import FEATURE_COLS
from src.models.models_package import LSTMForecaster, ModelConfig
from src.backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)
app = FastAPI(title="Recursive Trading Engine API", version="2.0.0")

ing = DataIngestor()
_jobs: dict[str, dict] = {}


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/api/v1/tickers")
def list_tickers():
    from src.utils.config import get
    tickers = get("data.tickers", ["BTC-USD"])
    result = []
    for t in tickers:
        try:
            price = DataIngestor.get_latest_price(t)
            result.append({"ticker": t, "latest_price": price})
        except Exception as exc:
            result.append({"ticker": t, "error": str(exc)})
    return {"tickers": result}


@app.post("/api/v1/forecast")
def run_forecast(ticker: str = "BTC-USD", horizon: int = 30):
    from src.utils.config import get
    job_id = str(uuid.uuid4())[:8]
    try:
        cfg = ModelConfig(ticker=ticker, future_days=horizon)
        fc = LSTMForecaster(cfg)
        import torch
        # load latest checkpoint if available
        ckpt_dir = Path("./models/checkpoints")
        ticker_key = ticker.replace("-", "_").replace("=", "_")
        ckpts = sorted(ckpt_dir.glob(f"{ticker_key}*.pt"))
        if ckpts:
            latest = ckpts[-1]
            model = LSTMForecaster._as_1d.__self__
            from src.models.models_package import LSTMModel
            m = LSTMModel(input_size=8, hidden_size=cfg.hidden_size,
                          num_layers=cfg.num_layers, dropout=cfg.initial_dropout)
            m.load_state_dict(torch.load(latest, map_location="cpu", weights_only=True))
            fc.model = m
        raw_df = ing.fetch(ticker)
        prepared = ing.fe.transform(raw_df)
        feat = prepared[FEATURE_COLS].values
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(feat)
        fc.scaler = scaler
        ai = scaler.transform(feat[-cfg.prediction_days * 3:, :])
        fu, su, lo, hi = fc._run_monte_carlo_rollout(fc.model, scaler, ai, horizon)
        last_date = raw_df.index[-1]
        if not isinstance(last_date, pd.Timestamp):
            last_date = pd.Timestamp(last_date)
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="B")
        _jobs[job_id] = {
            "ticker": ticker, "status": "done",
            "forecast": pd.DataFrame({
                "Date": dates, "Predicted_Price": fu,
                "CI95_Lower": lo, "CI95_Upper": hi,
            }).to_dict("records"),
        }
        return {"job_id": job_id, "status": "done", "rows": len(dates)}
    except Exception as exc:
        logger.exception("Forecast failed")
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/v1/forecast/{job_id}")
def get_forecast(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


@app.post("/api/v1/train")
def start_training(ticker: str = "BTC-USD", epochs: int = 40):
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"ticker": ticker, "status": "queued", "epochs": epochs}
    return {"job_id": job_id, "status": "queued", "message":
            "Training is queued — results will appear in the artifacts folder."}


@app.get("/api/v1/backtest/{ticker}")
def run_backtest(ticker: str, start: str = "2024-01-01", end: str | None = None):
    try:
        cfg = ModelConfig(ticker=ticker)
        engine = BacktestEngine(ticker=ticker, capital=100_000.0, model_cfg=cfg)
        res = engine.run(start=start, end=end)
        return {
            "ticker": ticker,
            "metrics": res.metrics,
            "trades_count": len(res.trades),
            "trades": res.trades.head(20).to_dict("records"),
        }
    except Exception as exc:
        logger.exception("Backtest failed")
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/v1/metrics/summary")
def metrics_summary():
    ing = DataIngestor()
    rows = []
    for t in ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "TSLA"]:
        try:
            price = DataIngestor.get_latest_price(t)
            rows.append({"ticker": t, "latest_price": price})
        except Exception:
            pass
    return {"count": len(rows), "data": rows}
