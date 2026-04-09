import argparse
import datetime as dt
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
import yfinance as yf

try:
    import scipy.stats as st
except ImportError:
    st = None


def _as_1d(a):
    """numpy 1d array; yfinance + model outputs can be (n,) or (n, 1)."""
    x = np.asarray(a)
    if x.ndim > 1:
        x = np.squeeze(x)
    return x.reshape(-1)


@dataclass
class Config:
    ticker: str = "BTC-USD"
    prediction_days: int = 30
    epochs: int = 40
    batch_size: int = 1500
    hidden_size: int = 500
    num_layers: int = 4
    initial_dropout: float = 0.6
    final_dropout: float = 0.1
    optimizer_name: str = "Ranger"
    weight_decay: float = 0.05
    output_dir: str = "."
    future_day: int = 30
    num_monte_carlo_runs: int = 100


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=500, num_layers=4, dropout=0.6):
        super().__init__()
        layers = []
        in_size = input_size
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

    def forward(self, x):
        out = x
        for i in range(0, len(self.layers), 2):
            out, _ = self.layers[i](out)
            out = self.layers[i + 1](out)
        out = out[:, -1, :]
        return self.fc(out)


def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
    return max(
        final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs)
    )


def set_dropout(model, new_p):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = new_p


def build_sequences(values, prediction_days):
    """Sliding windows without a Python loop (NumPy C stride / view)."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    t, f = values.shape
    if t <= prediction_days:
        return np.empty((0, prediction_days, f), np.float32), np.empty((0,), np.float32)
    # as_strided view: same semantics as values[i-p:i] predicting values[i,0]
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        win = sliding_window_view(values, prediction_days, axis=0)[
            :-1
        ]  # drop window that would target row T
        # NumPy can return (N, seq, feat) or (N, feat, seq) depending on version; LSTM needs (N, seq, feat)
        if win.shape[1] == f and win.shape[2] == prediction_days:
            win = np.swapaxes(win, 1, 2)
        elif win.shape[1] != prediction_days or win.shape[2] != f:
            raise RuntimeError(
                f"Sliding window shape {win.shape}; expected (N, {prediction_days}, {f}) "
                f"or (N, {f}, {prediction_days})."
            )
        y = values[prediction_days:, 0].copy()
        x = np.ascontiguousarray(win)
    except Exception:
        x, y = [], []
        for i in range(prediction_days, len(values)):
            x.append(values[i - prediction_days : i, :])
            y.append(values[i, 0])
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
    return x, y


def build_sliding_test_windows(ai_inputs, prediction_days):
    """Vectorized test windows from scaled matrix (same as Python loop over i)."""
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
            raise RuntimeError(
                f"Test sliding window shape {win.shape}; expected (N, {prediction_days}, {f})."
            )
        return np.ascontiguousarray(win)
    except Exception:
        xs = []
        for i in range(prediction_days, len(ai)):
            xs.append(ai[i - prediction_days : i, :])
        return np.asarray(xs, dtype=np.float32)


def add_features(df):
    out = df.copy()
    # yfinance often returns columns like (Close, Ticker); without this, .values is wide → wrong LSTM shape
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out["SMA_14"] = out["Close"].rolling(window=14).mean()
    delta = out["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI_14"] = 100 - (100 / (1 + rs))
    exp1 = out["Close"].ewm(span=12, adjust=False).mean()
    exp2 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = exp1 - exp2
    out["Signal_Line"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["20_SMA"] = out["Close"].rolling(window=20).mean()
    out["Std_Dev"] = out["Close"].rolling(window=20).std()
    out["Upper_BB"] = out["20_SMA"] + (out["Std_Dev"] * 2)
    out["Lower_BB"] = out["20_SMA"] - (out["Std_Dev"] * 2)
    out.ffill(inplace=True)
    out.bfill(inplace=True)
    return out


def run_monte_carlo_rollout(
    model,
    scaler,
    ai_inputs,
    prediction_days,
    future_day,
    num_monte_carlo_runs,
    device,
    use_gpu,
):
    """Rolling next-step forecast; other features held at last-known scaled values (matches pytorch_plotted)."""
    real_data = ai_inputs[-prediction_days:, :].copy()
    last_actual_scaled_volume = float(ai_inputs[-1, 1])
    last_actual_scaled_sma = float(ai_inputs[-1, 2])
    last_actual_scaled_rsi = float(ai_inputs[-1, 3])
    last_actual_scaled_macd = float(ai_inputs[-1, 4])
    last_actual_scaled_signal_line = float(ai_inputs[-1, 5])
    last_actual_scaled_upper_bb = float(ai_inputs[-1, 6])
    last_actual_scaled_lower_bb = float(ai_inputs[-1, 7])

    future_predictions = []
    future_predictions_std = []
    model.train()
    day_pbar = tqdm(
        range(future_day),
        desc="Future rollout",
        unit="day",
        colour="green",
        ascii=False,
        leave=True,
    )
    for _ in day_pbar:
        input_seq = real_data[-prediction_days:].reshape(1, prediction_days, 8)
        t_in = torch.from_numpy(input_seq).float().to(device, non_blocking=use_gpu)
        monte_carlo = []
        with torch.no_grad():
            for _ in range(num_monte_carlo_runs):
                monte_carlo.append(model(t_in).squeeze())
        monte_carlo = torch.stack(monte_carlo).detach().cpu().numpy()
        next_pred = float(monte_carlo[0])
        future_predictions_std.append(float(np.std(monte_carlo)))
        future_predictions.append(next_pred)
        new_row = np.array(
            [
                next_pred,
                last_actual_scaled_volume,
                last_actual_scaled_sma,
                last_actual_scaled_rsi,
                last_actual_scaled_macd,
                last_actual_scaled_signal_line,
                last_actual_scaled_upper_bb,
                last_actual_scaled_lower_bb,
            ],
            dtype=np.float64,
        )
        real_data = np.vstack((real_data, new_row))

    model.eval()
    future_scaled = np.asarray(future_predictions, dtype=np.float64).reshape(-1, 1)
    dummy = np.zeros((future_scaled.shape[0], 7))
    future_prices = scaler.inverse_transform(
        np.concatenate((future_scaled, dummy), axis=1)
    )[:, 0]
    future_prices = _as_1d(future_prices)

    close_min = float(scaler.data_min_[0])
    close_max = float(scaler.data_max_[0])
    scaling_factor_close = close_max - close_min
    std_ns = np.asarray(future_predictions_std, dtype=np.float64)
    std_unscaled = std_ns * scaling_factor_close
    z = float(st.norm.ppf(0.975)) if st is not None else 1.96
    lower = future_prices - z * std_unscaled
    upper = future_prices + z * std_unscaled
    return future_prices, std_unscaled, lower, upper


def main():
    parser = argparse.ArgumentParser(description="Fast training-only script (cpp variant).")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prediction-days", type=int, default=30)
    parser.add_argument(
        "--future-days",
        type=int,
        default=30,
        help="Rolling forecast horizon after the last historical bar (same method as pytorch_plotted).",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=100,
        help="Monte Carlo dropout forward passes per forecast day.",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (default: on for CUDA when not ROCm).",
    )
    args = parser.parse_args()

    cfg = Config(
        ticker=args.ticker,
        epochs=args.epochs,
        batch_size=args.batch_size,
        prediction_days=args.prediction_days,
        output_dir=args.output_dir,
        future_day=args.future_days,
        num_monte_carlo_runs=args.mc_runs,
    )

    device = torch.device(
        "cuda" if (args.device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    use_gpu = device.type == "cuda"
    hip = torch.version.hip is not None
    use_compile = use_gpu and not hip and (not args.no_compile)

    torch.backends.cudnn.enabled = torch.version.hip is None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"Ticker: {cfg.ticker}")
    print(f"Device: {device}")

    start = dt.datetime(2017, 1, 1)
    end = dt.datetime.now()
    raw = yf.download(cfg.ticker, start=start, end=end, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"No data found for ticker {cfg.ticker}")

    data = add_features(raw)
    feature_cols = [
        "Close",
        "Volume",
        "SMA_14",
        "RSI_14",
        "MACD",
        "Signal_Line",
        "Upper_BB",
        "Lower_BB",
    ]
    scaler = MinMaxScaler(feature_range=(0, 1))
    feat = data[feature_cols].values
    if feat.ndim != 2 or feat.shape[1] != len(feature_cols):
        raise RuntimeError(
            f"Expected scaled feature matrix (_, {len(feature_cols)}), got {feat.shape}. "
            "If using yfinance, ensure MultiIndex columns were flattened (see add_features)."
        )
    scaled = scaler.fit_transform(feat)
    x, y = build_sequences(scaled[:-cfg.prediction_days], cfg.prediction_days)
    if len(x) == 0:
        raise RuntimeError("Insufficient data to create training sequences.")

    dataset = SequenceDataset(x, y)
    nw = 2 if (use_gpu and os.name != "nt") else 0
    _dl_kw = dict(
        dataset=dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=nw,
        persistent_workers=bool(nw),
    )
    if nw > 0:
        _dl_kw["prefetch_factor"] = 2
    dataloader = DataLoader(**_dl_kw)

    model = LSTMModel(
        input_size=8,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.initial_dropout,
    ).to(device)
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (reduce-overhead).")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    criterion = nn.MSELoss()

    import torch_optimizer as optim

    optimizer = getattr(optim, cfg.optimizer_name)(
        model.parameters(), weight_decay=cfg.weight_decay
    )

    t0 = time.time()
    epoch_pbar = trange(
        cfg.epochs,
        desc="Epochs",
        unit="epoch",
        colour="blue",
        ascii=False,
    )
    for epoch in epoch_pbar:
        model.train()
        new_p = get_dynamic_dropout(
            epoch, cfg.epochs, cfg.initial_dropout, cfg.final_dropout
        )
        set_dropout(model, new_p)
        epoch_loss_acc = torch.zeros((), device=device)

        batch_pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
            unit="batch",
            colour="blue",
            ascii=False,
            mininterval=0.5,
        )
        for batch_idx, (xb, yb) in enumerate(batch_pbar, start=1):
            xb = xb.to(device, non_blocking=use_gpu)
            yb = yb.to(device, non_blocking=use_gpu).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss_acc += loss.detach() * xb.size(0)

            if (batch_idx % 25 == 0) or (batch_idx == len(dataloader)):
                batch_pbar.set_postfix(
                    batch_loss=f"{loss.item():.6f}",
                    dropout=f"{new_p:.3f}",
                )

        epoch_loss = (epoch_loss_acc / len(dataset)).item()
        epoch_pbar.set_postfix(loss=f"{epoch_loss:.6f}", dropout=f"{new_p:.3f}")

    train_seconds = time.time() - t0
    print(f"Training time: {train_seconds:.2f}s")

    # Quick evaluation on recent segment.
    test_start = dt.datetime(2025, 6, 1)
    test_raw = yf.download(cfg.ticker, test_start, end, auto_adjust=True)
    test = add_features(test_raw)
    close_col = test["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    actual = _as_1d(close_col.values)

    total = pd.concat((data[feature_cols], test[feature_cols]), axis=0)
    ai_inputs = scaler.transform(
        total[len(total) - len(test) - cfg.prediction_days :].values
    )
    x_test = build_sliding_test_windows(ai_inputs, cfg.prediction_days)
    if len(x_test) == 0:
        raise RuntimeError("Insufficient test windows.")

    model.train()  # preserve dropout behavior from source script
    with torch.inference_mode():
        xt = torch.from_numpy(x_test).float().to(device, non_blocking=use_gpu)
        preds = model(xt).cpu().numpy()

    dummy = np.zeros((preds.shape[0], 7))
    pred_prices = scaler.inverse_transform(
        np.concatenate((_as_1d(preds).reshape(-1, 1), dummy), axis=1)
    )[:, 0]
    pred_prices = _as_1d(pred_prices)

    # One prediction per row of test (same indexing as pytorch_plotted sliding window).
    n_pred = len(pred_prices)
    actual = actual[:n_pred]
    dates = np.asarray(pd.DatetimeIndex(test.index[:n_pred]))

    mse = mean_squared_error(actual, pred_prices)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, pred_prices)
    print(f"RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(cfg.output_dir, f"{cfg.ticker}_model_cpp_{stamp}.pt")
    torch.save(model.state_dict(), model_path)
    pred_df = pd.DataFrame(
        {"Date": dates, "Predicted": pred_prices, "Actual": actual}
    )
    pred_path = os.path.join(cfg.output_dir, f"{cfg.ticker}_predictions_cpp_{stamp}.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved model: {model_path}")
    print(f"Saved predictions: {pred_path}")

    print(f"\n{'=' * 60}")
    print(
        f"Next {cfg.future_day}-day forecast "
        f"(Monte Carlo dropout, {cfg.num_monte_carlo_runs} runs/day)"
    )
    print(f"{'=' * 60}")

    last_hist = data.index[-1]
    if not isinstance(last_hist, pd.Timestamp):
        last_hist = pd.Timestamp(last_hist)

    future_prices, std_u, lower, upper = run_monte_carlo_rollout(
        model,
        scaler,
        ai_inputs,
        cfg.prediction_days,
        cfg.future_day,
        cfg.num_monte_carlo_runs,
        device,
        use_gpu,
    )
    future_dates = pd.date_range(
        start=last_hist + pd.Timedelta(days=1),
        periods=cfg.future_day,
        freq="D",
    )
    future_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted_Price": future_prices,
            "Std_unscaled_approx": std_u,
            "CI95_lower": lower,
            "CI95_upper": upper,
        }
    )
    future_path = os.path.join(
        cfg.output_dir,
        f"{cfg.ticker}_future_{cfg.future_day}d_cpp_{stamp}.csv",
    )
    future_df.to_csv(future_path, index=False)
    print(f"Saved future forecast: {future_path}")

    close_series = data["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    current_price = float(_as_1d(close_series.values)[-1])
    final_fc = float(future_prices[-1])
    pct = (final_fc - current_price) / current_price * 100.0
    print(
        f"Last close: ${current_price:.2f}  "
        f"Forecast day {cfg.future_day}: ${final_fc:.2f}  ({pct:+.2f}%)"
    )


if __name__ == "__main__":
    main()
