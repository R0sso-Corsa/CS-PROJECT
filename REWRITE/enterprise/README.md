# Recursive Trading Engine v2.0

Enterprise-grade recursive trading algorithm platform built on the existing
`pytorch_train_cpp.py` BiLSTM pipeline.

## Architecture

```
enterprise/
├── src/
│   ├── models/          # LSTM (BiLSTM dropout), AttentionLSTM, Transformer
│   │   └── __init__.py  # LSTMForecaster, ModelConfig, ModelRegistry
│   ├── data/            # DataIngestor (yfinance), FeatureEngineer (8 features)
│   ├── strategies/      # TrendFollowing, MeanReversion, Recursive ensemble
│   ├── risk/            # PositionSizer, KellyCriterion, RiskGuardrails
│   ├── backtest/        # Event-driven backtest engine
│   ├── dashboard/       # Streamlit (5 pages)
│   └── api/             # FastAPI REST server
├── config/config.yaml  # Single source of truth for all settings
├── data/               # Raw/processed/features storage
├── models/             # Checkpoints + model registry
├── tests/              # Pytest suite (unit + integration)
├── scripts/            # train.py / forecast.py CLI entry points
├── docker/             # Dockerfile + docker-compose.yml
└── pyproject.toml      # Build system + dependencies
```

## Installation

```bash
cd C:/Users/paron/Desktop/Dev/CS_PROJECT/REWRITE
pip install -r enterprise/requirements.txt
```

## Train a model

```bash
python enterprise/scripts/train.py --ticker BTC-USD --epochs 40
```

Outputs land in `enterprise/artifacts/` (models, predictions, forecasts).

## Launch dashboard

```bash
streamlit run enterprise/src/dashboard/app.py
```

## Start API

```bash
uvicorn enterprise.src.api.server:app --host 0.0.0.0 --port 8000
```

## Run tests

```bash
cd enterprise
pytest tests/ -v
```

## Key features

- **BiLSTM model** (matches existing `pytorch_train_cpp.py` checkpoints)
- **AttentionLSTM** — Bahdanau-style attention variant
- **TransformerForecaster** — MHA sequence-to-one model
- **Monte Carlo dropout** — 100 stochastic forward passes, 95% CI bands
- **8-feature technical set** — Close, Volume, SMA, RSI, MACD, Signal, Bollinger Bands
- **3 trading strategies** — Trend Following, Mean Reversion, Recursive Ensemble
- **Recursive ensemble** — dynamically re-weights sub-strategies from PnL history
- **Risk management** — ATR stops, Kelly sizing, drawdown circuit breakers, VaR
- **Event-driven backtester** — Sharpe, Sortino, Calmar, max drawdown, profit factor
- **Streamlit dashboard** — Overview, Forecast, Backtest, Model Manager pages
- **FastAPI backend** — `/forecast`, `/train`, `/backtest`, `/metrics` endpoints
- **Docker** — one-command deploy for both dashboard + API
