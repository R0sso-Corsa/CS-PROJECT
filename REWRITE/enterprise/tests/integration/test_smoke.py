"""Smoke tests that hit the network (yfinance)."""
import pytest
from src.data.ingestion import DataIngestor
from src.data.features import FeatureEngineer, FEATURE_COLS


def test_ingestor_fetch():
    ing = DataIngestor(cache_dir="/tmp/test_cache_rte")
    df = ing.fetch("BTC-USD", start="2024-06-01", end="2024-06-03")
    assert not df.empty
    assert "Close" in df.columns


def test_feature_engineer_pipeline():
    ing = DataIngestor(cache_dir="/tmp/test_cache_rte2")
    df = ing.fetch("AAPL", start="2024-06-01", end="2024-06-05")
    mat = ing.fe.get_feature_matrix(df)
    assert mat.shape[1] == len(FEATURE_COLS)
    assert not mat.isna().any().any()
