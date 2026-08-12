from __future__ import annotations
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_os.environ.setdefault("MIOpen_DISABLE", "1")
_os.environ.setdefault("PYTORCH_DISABLE_DYNAMO", "1")
_os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch as _torch
_torch.set_default_device("cpu")
try:
    _torch._dynamo.config.disable = True
except Exception:
    pass
if "CUDA_VISIBLE_DEVICES" in _os.environ:
    del _os.environ["CUDA_VISIBLE_DEVICES"]
if "MIOpen_DISABLE" in _os.environ:
    del _os.environ["MIOpen_DISABLE"]
if "PYTORCH_DISABLE_DYNAMO" in _os.environ:
    del _os.environ["PYTORCH_DISABLE_DYNAMO"]
del _torch, _os
"""Recursive Trading Engine — single-page dashboard."""


import sys
from pathlib import Path

# Bootstrap: ensure the `enterprise/` root is on sys.path
ROOT = Path(__file__).resolve().parents[2]

# Headless rendering fallback before any matplotlib/plotting import
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
_plt.ion()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Backend contracts (flat imports — no rte.* namespace magic)
from src.data.ingestion  import DataIngestor
from src.data.features   import FEATURE_COLS
from src.models import LSTMForecaster, ModelConfig, LSTMModel
from src.backtest.engine import BacktestEngine

st.set_page_config(page_title="Recursive Trading Engine", layout="wide")

with st.sidebar:
    st.header("⚙️ Controls")
    ticker = st.selectbox("Ticker", ["BTC-USD","ETH-USD","SOL-USD","AAPL","TSLA","GC=F"])
    st.caption("PyTorch + ROCm  •  Version 2.0")

page = st.radio("Page", ["Overview","Forecast","Backtest","Models"], horizontal=True)

# === Overview ===
if page == "Overview":
    st.header("📈 Overview")
    ing = DataIngestor()
    try:
        df = ing.fetch(ticker)
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"${df['Close'].iloc[-1]:,.2f}")
    c2.metric("24h High",  f"${df['High'].iloc[-1]:,.2f}")
    c3.metric("24h Low",   f"${df['Low'].iloc[-1]:,.2f}")
    c4.metric("Vol (last)", f"{df['Volume'].iloc[-1]:,.0f}")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index[-120:], open=df['Open'][-120:], high=df['High'][-120:],
        low=df['Low'][-120:], close=df['Close'][-120:], name=ticker,
    )])
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"{ticker} — 120d")
    st.plotly_chart(fig, width="stretch")

# === Forecast ===
elif page == "Forecast":
    st.header("🔮 Monte Carlo Forecast")
    horizon = st.slider("Horizon", 5, 60, 30)
    mc_runs = st.slider("MC dropout runs", 20, 500, 100)
    if st.button("Run Forecast", type="primary"):
        try:
            cfg   = ModelConfig(ticker=ticker, future_days=horizon, num_monte_carlo_runs=mc_runs)
            fc    = LSTMForecaster(cfg)
            ing   = DataIngestor()
            raw   = ing.fetch(ticker)
            prep  = ing.fe.transform(raw)
            feat  = prep[FEATURE_COLS].values
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler().fit(feat)
            ai = scaler.transform(feat)[-cfg.prediction_days * 3:]
            ckpt_dir = ROOT / "models" / "checkpoints"
            tk = ticker.replace("-", "_").replace("=", "_")
            ckpts = sorted(ckpt_dir.glob(f"{tk}*.pt")) if ckpt_dir.exists() else []
            if not ckpts:
                st.error("No checkpoint for this ticker. Use the Models tab to train one.")
            else:
                ckpt = ckpts[-1]
                model = LSTMModel(input_size=8, hidden_size=cfg.hidden_size,
                                  num_layers=cfg.num_layers, dropout=cfg.initial_dropout)
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                model.eval()
                model.to("cpu")
                fu, su, lo, hi = fc._run_monte_carlo_rollout(model, scaler, ai, horizon)
                dates = pd.date_range(start=raw.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="B")
                f_df = pd.DataFrame({"Date": dates, "Forecast": fu, "P5": lo, "P95": hi})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=raw.index[-60:], y=raw["Close"], mode="lines", name="Actual (60d)"))
                fig.add_trace(go.Scatter(x=f_df["Date"], y=f_df["Forecast"], mode="lines+markers", name="Forecast"))
                fig.add_trace(go.Scatter(x=list(f_df["Date"]) + list(f_df["Date"][::-1]),
                                         y=list(f_df["P95"]) + list(f_df["P5"][::-1]),
                                         fill="toself", name="95% CI"))
                fig.update_layout(height=600, title=f"{ticker} {horizon}-day forecast")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(f_df, width="stretch")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

# === Backtest ===
elif page == "Backtest":
    st.header("🔁 Backtest")
    start = st.date_input("Start", value=pd.Timestamp("2024-01-01").date())
    end   = st.date_input("End",   value=pd.Timestamp.today().date())
    if st.button("Run Backtest", type="primary"):
        try:
            cfg    = ModelConfig(ticker=ticker)
            engine = BacktestEngine(ticker=ticker, capital=100_000.0, model_cfg=cfg)
            res    = engine.run(start=str(start), end=str(end))
            cols   = st.columns(len(res.metrics))
            for (k, v), col in zip(res.metrics.items(), cols):
                col.metric(k, v)
            with st.expander("Metrics"):
                st.dataframe(res.metrics)
            st.subheader("Equity curve")
            st.line_chart(res.equity_curve)
            with st.expander("Trade log"):
                trades = res.trades if isinstance(res.trades, pd.DataFrame) else pd.DataFrame(res.trades)
                st.dataframe(trades)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

# === Models ===
else:
    st.header("🧠 Model Manager")
    epochs = st.slider("Epochs", 5, 200, 40)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train New Model", type="primary"):
            try:
                cfg = ModelConfig(ticker=ticker, epochs=epochs)
                fc  = LSTMForecaster(cfg)
                res = fc.train(ticker)
                st.success(f"Done — RMSE={res['rmse']:.2f}  in {res.get('train_seconds','?')}s")
                st.text(Path(res.get('model_path','?')).name)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
    with c2:
        if st.button("List Checkpoints"):
            st.rerun()
    ckpt_dir = ROOT / "models" / "checkpoints"
    tk = ticker.replace("-", "_").replace("=", "_")
    ckpts = sorted(ckpt_dir.glob(f"{tk}*.pt"), reverse=True) if ckpt_dir.exists() else []
    if ckpts:
        for c in ckpts:
            st.text(f"{c.name}  ({c.stat().st_size / 1_048_576:.1f} MB)")
    else:
        st.info("No checkpoints yet.")
