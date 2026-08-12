"""Strategy tests."""
import numpy as np
import pandas as pd
import pytest
from src.strategies.recursive import TrendFollowingStrategy, MeanReversionStrategy, RecursiveStrategy


@pytest.fixture
def trending_df():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    c = np.linspace(50, 80, 30)
    df = pd.DataFrame({
        "Close": c, "SMA_14": c, "20_SMA": c, "Std_Dev": 1.0,
        "RSI_14": np.full(30, 60.0),
        "MACD": np.full(30, 1.0), "Signal_Line": np.full(30, 0.0),
        "Upper_BB": c + 2, "Lower_BB": c - 2,
    }, index=idx)
    df.attrs["ticker"] = "TEST"
    return df


def test_trend_bullish(trending_df):
    sig = TrendFollowingStrategy().generate(trending_df)
    assert sig.side == 1
    assert sig.confidence > 0


def test_mean_reversion_overbought():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    c = np.linspace(120, 100, 30)  # strongly downward
    df = pd.DataFrame({
        "Close": c, "SMA_14": c, "20_SMA": c, "Std_Dev": 1.0,
        "RSI_14": np.full(30, 70.0),
        "MACD": np.full(30, -1.0), "Signal_Line": np.full(30, 0.0),
        "Upper_BB": c + 2, "Lower_BB": c - 2,
    }, index=idx)
    df.attrs["ticker"] = "TEST"
    sig = MeanReversionStrategy().generate(df)
    assert sig.side in (-1, 0)


def test_recursive_weights_sum_to_one(trending_df):
    rs = RecursiveStrategy(strategies=[TrendFollowingStrategy, MeanReversionStrategy])
    sig = rs.generate(trending_df)
    assert abs(sum(rs.weights) - 1.0) < 1e-3
