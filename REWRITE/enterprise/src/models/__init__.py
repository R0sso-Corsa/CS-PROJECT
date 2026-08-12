"""PyTorch model zoo: LSTM + Transformer with MC dropout forecasting."""

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
import datetime as dt
import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

try:
    import scipy.stats as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    ticker: str = "BTC-USD"
    prediction_days: int = 30
    epochs: int = 40
    batch_size: int = 16
    hidden_size: int = 500
    num_layers: int = 4
    initial_dropout: float = 0.6
    final_dropout: float = 0.1
    optimizer_name: str = "Ranger"
    weight_decay: float = 0.05
    future_days: int = 30
    num_monte_carlo_runs: int = 100
    learning_rate: float = 0.001
    use_compile: bool = True


class SequenceDataset(Dataset):
    """NumPy sliding-window adapter for PyTorch DataLoader."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMModel(nn.Module):
    """Bidirectional LSTM matching pytorch_train_cpp.py architecture."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 500,
        num_layers: int = 4,
        dropout: float = 0.6,
    ):
        super().__init__()
        layers = []
        in_size = input_size
        # Force CPU init to bypass ROCm/HIP device issues if torch.cuda probes ran earlier
        torch_device = torch.device("cpu")
        for _ in range(num_layers):
            layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size // 2,
                    bidirectional=True,
                    batch_first=True,
                )
            )
            layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().float()
        out = x
        for i in range(0, len(self.layers), 2):
            out, _ = self.layers[i](out)
            out = self.layers[i + 1](out)
        return self.fc(out[:, -1, :])


class Attention(nn.Module):
    """Bahdanau-style attention over LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = torch.matmul(energy, self.v)
        attn_weights = torch.softmax(scores, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
        return context, attn_weights


class AttentionLSTM(nn.Module):
    """BiLSTM + Bahdanau attention for sequence-to-one prediction."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        context, _ = self.attention(hidden_cat, lstm_out)
        return self.fc(self.dropout(context))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Lightweight transformer for sequence-to-one prediction."""

    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.proj(x))
        z = self.encoder(z)
        return self.fc(self.dropout(z[:, -1]))


class LSTMForecaster:
    """Object-oriented wrapper around pytorch_train_cpp.py logic.

    Provides ``.train(ticker)``, ``.forecast(ticker, horizon)``, and
    ``.load(path)`` / ``.save(path)`` so the training script can be
    imported and driven by the strategies / API modules.
    """

    def __init__(self, cfg: "ModelConfig | None" = None):
        self.cfg = cfg or ModelConfig()
        self.device = torch.device("cpu")
        self.model: nn.Module | None = None
        self.scaler: MinMaxScaler | None = None
        self.feature_cols = [
            "Close", "Volume", "SMA_14", "RSI_14", "MACD",
            "Signal_Line", "Upper_BB", "Lower_BB",
        ]
        self._dirs: dict[str, Path] | None = None

    def _resolve_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _as_1d(a) -> np.ndarray:
        x = np.asarray(a)
        if x.ndim > 1:
            x = np.squeeze(x)
        return x.reshape(-1)

    def _get_dirs(self) -> dict[str, Path]:
        if self._dirs:
            return self._dirs
        from src.utils.config import get
        root = Path(get("artifacts.output_dir", "./artifacts"))
        self._dirs = {
            "root": root,
            "models": root / "models",
            "predictions": root / "predictions",
            "forecasts": root / "forecasts",
        }
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return self._dirs

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        out["SMA_14"] = out["Close"].rolling(window=14).mean()
        delta = out["Close"].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        ag = gain.ewm(com=13, adjust=False).mean()
        al = loss.ewm(com=13, adjust=False).mean()
        out["RSI_14"] = 100 - (100 / (1 + ag / al))
        out["MACD"] = (
            out["Close"].ewm(span=12, adjust=False).mean()
            - out["Close"].ewm(span=26, adjust=False).mean()
        )
        out["Signal_Line"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["20_SMA"] = out["Close"].rolling(window=20).mean()
        out["Std_Dev"] = out["Close"].rolling(window=20).std()
        out["Upper_BB"] = out["20_SMA"] + (out["Std_Dev"] * 2)
        out["Lower_BB"] = out["20_SMA"] - (out["Std_Dev"] * 2)
        out.ffill(inplace=True)
        out.bfill(inplace=True)
        return out

    @staticmethod
    def _build_sequences(values, prediction_days: int):
        values = np.ascontiguousarray(values, dtype=np.float32)
        t, f = values.shape
        if t <= prediction_days:
            return np.empty((0, prediction_days, f), np.float32), np.empty((0,), np.float32)
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            win = sliding_window_view(values, prediction_days, axis=0)[:-1]
            if win.shape[1] == f and win.shape[2] == prediction_days:
                win = np.swapaxes(win, 1, 2)
            elif win.shape[1] != prediction_days or win.shape[2] != f:
                raise RuntimeError(f"Sliding window shape {win.shape}")
            return np.ascontiguousarray(win), values[prediction_days:, 0].copy()
        except Exception:
            xs = [values[i - prediction_days:i, :] for i in range(prediction_days, len(values))]
            return (
                np.asarray(xs, dtype=np.float32),
                np.asarray([values[i, 0] for i in range(prediction_days, len(values))], dtype=np.float32),
            )

    @staticmethod
    def _build_test_windows(ai_inputs, prediction_days: int):
        ai = np.ascontiguousarray(ai_inputs, dtype=np.float32)
        t, f = ai.shape
        if t <= prediction_days:
            return np.empty((0, prediction_days, f), np.float32)
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            win = sliding_window_view(ai, prediction_days, axis=0)[:-1]
            if win.shape[1] == f and win.shape[2] == prediction_days:
                win = np.swapaxes(win, 1, 2)
            elif win.shape[1] != prediction_days or win.shape[2] != f:
                raise RuntimeError(f"Sliding window shape {win.shape}")
            return np.ascontiguousarray(win)
        except Exception:
            xs = [ai[i - prediction_days:i, :] for i in range(prediction_days, len(ai))]
            return np.asarray(xs, dtype=np.float32)

    @staticmethod
    def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
        return max(final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs))

    @staticmethod
    def set_dropout(model, new_p: float):
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = new_p

    def _run_monte_carlo_rollout(self, model, scaler, ai_inputs, future_day):
        pred_days = self.cfg.prediction_days
        mc_runs = self.cfg.num_monte_carlo_runs
        real = ai_inputs[-pred_days:, :].copy()
        lv = {
            "vol": float(ai_inputs[-1, 1]),
            "sma": float(ai_inputs[-1, 2]),
            "rsi": float(ai_inputs[-1, 3]),
            "macd": float(ai_inputs[-1, 4]),
            "signal": float(ai_inputs[-1, 5]),
            "upper": float(ai_inputs[-1, 6]),
            "lower": float(ai_inputs[-1, 7]),
        }
        future_preds, future_std = [], []
        model.train()
        for _ in range(future_day):
            seq = real[-pred_days:].reshape(1, pred_days, 8)
            t_in = torch.from_numpy(seq).float().to(self.device, non_blocking=torch.cuda.is_available())
            with torch.no_grad():
                mc = [model(t_in).squeeze() for _ in range(mc_runs)]
            samples = torch.stack(mc).detach().cpu().numpy()
            future_preds.append(float(samples[0]))
            future_std.append(float(np.std(samples)))
            real = np.vstack((real, [
                future_preds[-1], lv["vol"], lv["sma"], lv["rsi"],
                lv["macd"], lv["signal"], lv["upper"], lv["lower"],
            ]))
        model.eval()
        sp = np.asarray(future_preds, np.float64).reshape(-1, 1)
        fp = scaler.inverse_transform(np.concatenate((sp, np.zeros((sp.shape[0], 7))), axis=1))[:, 0]
        fp = self._as_1d(fp)
        sm = float(scaler.data_min_[0]); sx = float(scaler.data_max_[0]); scale = sx - sm
        sn = np.asarray(future_std, np.float64)
        su = sn * scale
        z = float(st.norm.ppf(0.975)) if st is not None else 1.96
        return fp, su, fp - z * su, fp + z * su

    def train(self, ticker: str = "BTC-USD", output_dir: str = "") -> dict:
        t0 = time.time()
        import yfinance as _yf
        raw = _yf.download(ticker, start=dt.datetime(2017, 1, 1), end=dt.datetime.now(), auto_adjust=True)
        if raw.empty:
            raise RuntimeError(f"No data for {ticker}")
        data = self.add_features(raw)
        feat = data[self.feature_cols].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(feat)
        x, y = self._build_sequences(scaled[:-self.cfg.prediction_days], self.cfg.prediction_days)
        if len(x) == 0:
            raise RuntimeError("Insufficient data")

        ds = SequenceDataset(x, y)
        nw = 2 if torch.cuda.is_available() and os.name != "nt" else 0
        dl = DataLoader(ds, batch_size=min(self.cfg.batch_size, len(ds)),
                        shuffle=True, pin_memory=torch.cuda.is_available(),
                        num_workers=nw, persistent_workers=bool(nw))

        model = LSTMModel(input_size=8, hidden_size=self.cfg.hidden_size,
                          num_layers=self.cfg.num_layers, dropout=self.cfg.initial_dropout).to(self.device)
        if torch.cuda.is_available() and self.cfg.use_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass
        criterion = nn.MSELoss()
        try:
            import torch_optimizer as topt
            opt_cls = getattr(topt, self.cfg.optimizer_name)
            optimizer = opt_cls(model.parameters(), weight_decay=self.cfg.weight_decay)
        except Exception:
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay=self.cfg.weight_decay)

        for epoch in range(self.cfg.epochs):
            model.train()
            p = self.get_dynamic_dropout(epoch, self.cfg.epochs,
                                         self.cfg.initial_dropout, self.cfg.final_dropout)
            self.set_dropout(model, p)
            loss_acc = torch.zeros((), device=self.device)
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=torch.cuda.is_available())
                yb = yb.to(self.device, non_blocking=torch.cuda.is_available()).unsqueeze(1)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                loss_acc += loss.detach() * xb.size(0)
            ep_loss = (loss_acc / len(ds)).item()
            logger.info("Epoch %d/%d  loss=%.6f  dropout=%.3f",
                        epoch + 1, self.cfg.epochs, ep_loss, p)

        train_secs = round(time.time() - t0, 2)

        # Evaluation
        t0eval = dt.datetime(2025, 6, 1)
        t1eval = dt.datetime.now()
        t_raw = _yf.download(ticker, start=t0eval, end=t1eval, auto_adjust=True)
        test = self.add_features(t_raw)
        ac = test["Close"]
        if isinstance(ac, pd.DataFrame):
            ac = ac.iloc[:, 0]
        actual = self._as_1d(ac.values)
        combined = pd.concat((data[self.feature_cols], test[self.feature_cols]), axis=0)
        ai = scaler.transform(combined[len(combined) - len(test) - self.cfg.prediction_days:].values)
        x_test = self._build_test_windows(ai, self.cfg.prediction_days)
        model.train()
        with torch.inference_mode():
            xt = torch.from_numpy(x_test).float().to(self.device)
            preds = model(xt).cpu().numpy()
        inv = scaler.inverse_transform(
            np.concatenate((self._as_1d(preds).reshape(-1, 1), np.zeros((len(preds), 7))), axis=1)
        )[:, 0]
        pp = self._as_1d(inv)
        rmse = math.sqrt(mean_squared_error(actual[:len(pp)], pp))
        mae = mean_absolute_error(actual[:len(pp)], pp)

        dirs = self._get_dirs()
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = dirs["models"] / f"{ticker}_model_{stamp}.pt"
        torch.save(model.state_dict(), model_path)
        pred_df = pd.DataFrame({"Date": test.index[:len(pp)], "Predicted": pp,
                                "Actual": actual[:len(pp)]})
        pred_csv = dirs["predictions"] / f"{ticker}_predictions_{stamp}.csv"
        pred_df.to_csv(pred_csv, index=False)

        last_hist = data.index[-1]
        if not isinstance(last_hist, pd.Timestamp):
            last_hist = pd.Timestamp(last_hist)
        fp, su, lo, hi = self._run_monte_carlo_rollout(model, scaler, ai, self.cfg.future_days)
        fd = pd.date_range(start=last_hist + pd.Timedelta(days=1), periods=self.cfg.future_days, freq="D")
        f_df = pd.DataFrame({"Date": fd, "Predicted_Price": fp,
                             "Std_unscaled_approx": su, "CI95_lower": lo, "CI95_upper": hi})
        fc_csv = dirs["forecasts"] / f"{ticker}_future_{self.cfg.future_days}d_{stamp}.csv"
        f_df.to_csv(fc_csv, index=False)

        self.model = model
        self.scaler = scaler
        return {
            "ticker": ticker, "rmse": rmse, "mae": mae, "train_seconds": train_secs,
            "model_path": str(model_path), "predictions_csv": str(pred_csv),
            "forecast_csv": str(fc_csv), "forecast_shape": f_df.shape,
        }
