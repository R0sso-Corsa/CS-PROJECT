"""
Recursive Trading Engine 2.0

Enterprise platform for LSTM/Transformer-based trading with recursive signal
ensemble, risk management, backtesting, Streamlit dashboard, and FastAPI.

Quick start
-----------
    # install (from repo root)
    pip install -r enterprise/requirements.txt

    # train BTC-USD
    python enterprise/scripts/train.py --ticker BTC-USD --epochs 40

    # dashboard
    streamlit run enterprise/src/dashboard/app.py

    # API
    uvicorn enterprise.src.api.server:app --reload
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "CS Project"
