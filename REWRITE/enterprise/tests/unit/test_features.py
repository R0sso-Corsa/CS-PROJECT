"""Feature engineering tests."""
import numpy as np
import pandas as pd
import pytest
from src.data.features import FeatureEngineer, FEATURE_COLS


def test_feature_columns(toy_ohlcv: pd.DataFrame):
    fe = FeatureEngineer()
    out = fe.transform(toy_ohlcv)
    for col in FEATURE_COLS:
        assert col in out.columns
    assert not out[FEATURE_COLS].isna().any().any()


def test_feature_output_shape(toy_ohlcv: pd.DataFrame):
    fe = FeatureEngineer()
    out = fe.transform(toy_ohlcv)
    assert out.shape[0] == 60


def test_sma_14(toy_ohlcv: pd.DataFrame):
    fe = FeatureEngineer()
    out = fe.transform(toy_ohlcv)
    # SMA should equal mean of first 14 closes
    expected = pd.Series([np.nan]*13 + [toy_ohlcv["Close"].iloc[:14].mean()]*47, index=toy_ohlcv.index)
    pd.testing.assert_series_equal(out["SMA_14"], expected, check_names=False)


def test_rsi_bounded():
    fe = FeatureEngineer()
    rsi_col = "RSI_14"
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    df = pd.DataFrame({"Close": np.linspace(100, 110, 60), "Volume": 1e6}, index=idx)
    out = fe.transform(df)
    assert out[rsi_col].between(0, 100).all()
