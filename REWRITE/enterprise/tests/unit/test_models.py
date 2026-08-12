"""Model package tests (no GPU required)."""
import numpy as np
import torch
import pytest
from src.models.models_package import ModelConfig, LSTMModel, AttentionLSTM, TransformerForecaster, LSTMForecaster


@pytest.fixture
def dummy_input():
    return torch.randn(4, 30, 8)


def test_lstm_forward(dummy_input):
    m = LSTMModel(input_size=8, hidden_size=64, num_layers=2, dropout=0.3)
    m.eval()
    out = m(dummy_input)
    assert out.shape == (4, 1)


def test_attention_lstm_forward(dummy_input):
    m = AttentionLSTM(input_size=8, hidden_size=64, num_layers=2, dropout=0.3)
    m.eval()
    out = m(dummy_input)
    assert out.shape == (4, 1)


def test_transformer_forward(dummy_input):
    m = TransformerForecaster(input_size=8, d_model=64, nhead=4, num_layers=2)
    m.eval()
    out = m(dummy_input)
    assert out.shape == (4, 1)


def test_modelconfig_defaults():
    cfg = ModelConfig()
    assert cfg.ticker == "BTC-USD"
    assert cfg.epochs == 40
    assert cfg.hidden_size == 500


def test_dynamic_dropout():
    p = LSTMForecaster.get_dynamic_dropout(0, 40, 0.6, 0.1)
    assert abs(p - 0.6) < 0.01
    p = LSTMForecaster.get_dynamic_dropout(40, 40, 0.6, 0.1)
    assert abs(p - 0.1) < 0.01
