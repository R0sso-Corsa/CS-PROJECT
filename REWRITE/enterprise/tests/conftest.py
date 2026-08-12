"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def toy_ohlcv():
    """60-day synthetic OHLCV series."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(60) * 0.5)
    high = close + np.random.rand(60) * 0.3
    low = close - np.random.rand(60) * 0.3
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": 1_000_000 + np.random.rand(60)*500_000},
                        index=idx)
