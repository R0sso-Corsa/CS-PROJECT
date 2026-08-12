"""
Streamlit dashboard for the Recursive Trading Engine.

Usage:
    streamlit run src/dashboard/app.py
"""

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
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.ingestion import DataIngestor
from src.data.features import FEATURE_COLS
from src.models.models_package import LSTMForecaster, ModelConfig, LSTMModel
from src.backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Recursive Trading Engine", layout="wide", page_icon="\U0001F4C8")

with st.sidebar:
    st.header("\u2699\ufe0f Controls")
    ticker = st.selectbox("Ticker", ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "TSLA", "GC=F"])
    model_type = st.selectbox("Model variant (future MR)", ["Standard LSTM"])
    st.markdown("---")
    st.markdown("**Navigation**")
    st.page_link("pages/0_Overview.py", label="Overview", icon="\U0001F4CA")
    st.page_link("pages/1_Forecast.py", label="Forecast", icon="\U0001F52E")
    st.page_link("pages/2_Backtest.py", label="Backtest", icon="\U0001F501")
    st.page_link("pages/3_Models.py", label="Models", icon="\U0001F9E0")

page = st.navigation([
    st.Page("overview", title="Overview", icon="\U0001F4CA"),
    st.Page("forecast", title="Forecast", icon="\U0001F52E"),
    st.Page("backtest", title="Backtest", icon="\U0001F501"),
    st.Page("models", title="Models", icon="\U0001F9E0"),
])


def _metric_card(label, value, delta=None):
    st.metric(label, value, delta)

def _log_tail(n=30):
    log_path = Path(__file__).resolve().parent.parent.parent / "logs" / "engine.log"
    if log_path.exists():
        return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    return "(no log file yet)"


def overview():
    st.title("\U0001F4CA Recursive Trading Engine — Overview")
    ing = DataIngestor()
    try:
        prepared = ing.fetch(ticker)
        row = ing.fe.transform(prepared).iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close", f"${row['Close']:,.2f}")
        c2.metric("RSI(14)", f"{row['RSI_14']:.1f}")
        c3.metric("MACD", f"{row['MACD']:.2f}")
        bb = row["Upper_BB"] - row["Lower_BB"]
        c4.metric("Bollinger Width", f"{bb:,.2f}")
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        return

    sub = ing.fetch(ticker).tail(180)
    ckpt_dir = Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints"
    ticker_key = ticker.replace("-", "_").replace("=", "_")
    ckpts = sorted(ckpt_dir.glob(f"{ticker_key}*.pt"))
    st.caption(f"Checkpoints: {len(ckpts)}  |  Latest: {ckpts[-1].name if ckpts else 'none'}")

    fig = go.Figure(data=[go.Candlestick(
        x=sub.index, open=sub["Open"], high=sub["High"],
        low=sub["Low"], close=sub["Close"], name=ticker,
    )])
    fig.update_layout(title=f"{ticker} — last 180d", height=520, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Recent log"):
        st.code(_log_tail(), language="text")
