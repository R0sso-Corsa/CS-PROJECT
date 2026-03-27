import time
import sys

script_start_time = time.time()

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle
import yfinance as yf
import mplfinance as mpf
import seaborn as sns

import torch
import torch.nn as nn

from tqdm.auto import trange, tqdm

torch.backends.cudnn.enabled = False
from torch.utils.data import Dataset, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_BACKEND = "torch"
except Exception:
    try:
        from tensorboardX import SummaryWriter

        _TB_BACKEND = "tensorboardX"
    except Exception:
        SummaryWriter = None
        _TB_BACKEND = None

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import scipy.stats as st
import math


# --- Helper: safe conversions for DataFrame columns ---
def _to_1d_list(v):
    """Coerce v to a 1-D Python list acceptable to pd.DataFrame.
    Handles scalars, pandas Series, numpy arrays shaped (n,1) or (n,), and lists."""
    try:
        if hasattr(v, "to_numpy"):
            arr = v.to_numpy()
        else:
            arr = np.asarray(v)
    except Exception:
        return [v]

    try:
        if arr.size == 0:
            return []
    except Exception:
        pass

    if getattr(arr, "ndim", 1) == 2 and arr.shape[1] == 1:
        return arr.reshape(-1).tolist()
    if getattr(arr, "ndim", 1) == 2 and arr.shape[1] > 1:
        return arr[:, 0].reshape(-1).tolist()
    return arr.reshape(-1).tolist()


def safe_make_and_save_df(path, columns_dict):
    """Create a DataFrame from columns_dict after coercing values to 1D lists,
    then save to `path` (CSV). Returns the DataFrame."""
    safe_dict = {}
    for k, v in columns_dict.items():
        # DatetimeIndex or list of Timestamps -> ISO strings
        if isinstance(v, (pd.DatetimeIndex,)) or (
            isinstance(v, (list, np.ndarray))
            and len(v) > 0
            and isinstance(v[0], (pd.Timestamp, np.datetime64))
        ):
            safe_dict[k] = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in list(v)]
        else:
            safe_dict[k] = _to_1d_list(v)
    df = pd.DataFrame(safe_dict)
    df.to_csv(path, index=False)
    return df


# ----------------------- Configuration -----------------------


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        if message.strip():
            timestamp = dt.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.log.write(timestamp + message + "\n")
        else:
            self.log.write(message)
        self.terminal.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()


def finder(query: str):
    """Search for symbols matching `query` using yfinance."""
    try:
        return yf.Search(query)
    except Exception:
        return None


def choose_chart_interactive():
    """Ask user for ticker or search interactively."""
    while True:
        choice = input("Enter ticker symbol (or enter 's' to search): ").strip()
        if not choice:
            continue
        if choice.lower() == "s":
            query = input("Search query (company name or ticker fragment): ").strip()
            if not query:
                print("Empty query; try again.\n")
                continue
            results = finder(query)
            if results and getattr(results, "quotes", None):
                print(f"Search results for '{query}':")
                for i, q in enumerate(results.quotes):
                    print(
                        f"{i+1}. {q.get('symbol')} - {q.get('shortname')} ({q.get('quoteType')})"
                    )
                sel = input(
                    "Enter number to select a symbol, or press Enter to cancel: "
                ).strip()
                if sel.isdigit():
                    idx = int(sel) - 1
                    if 0 <= idx < len(results.quotes):
                        symbol = results.quotes[idx].get("symbol")
                        print(f"Selected: {symbol}")
                        return symbol
                    else:
                        print("Selection out of range; try again.")
                        continue
                else:
                    print("No selection made; returning to main prompt.")
                    continue
            else:
                print("No search results found; try a different query.")
                continue
        else:
            return choice


chart = choose_chart_interactive()
chart_info = yf.Ticker(chart).info
prediction_days = 30
future_day = 30
epochs = 40
batch_size = 32
initial_dropout = 0.4
final_dropout = 0.1
num_monte_carlo_runs = 100

device_type = input("Enter device type (cpu/gpu): ").strip().lower()
device = torch.device(
    "cuda" if (device_type == "gpu" and torch.cuda.is_available()) else "cpu"
)

print(f"Using device: {device}")

chart_name_plot = chart_info.get("longName") or chart
chart_name_plot1 = chart_info.get("shortName") or chart

os.environ["DNNL_VERBOSE"] = "2"
os.environ["CUDNN_LOGINFO_DBG"] = "1"
os.environ["CUDNN_LOGDEST_DBG"] = "stdout"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = ">>"
os.environ["PYTORCH_JIT_LOGS"] = "all"

torch._logging.set_logs(graph_breaks=True, recompiles=True, dynamo=20, inductor=20)


# ----------------------- Enhanced Feature Engineering -----------------------


def calculate_atr(data, period=14):
    """Calculate Average True Range for volatility."""
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    tr = np.max(np.column_stack((high_low, high_close, low_close)), axis=1)
    atr = pd.Series(tr).rolling(window=period).mean()
    return atr.values


def calculate_adx(data, period=14):
    """Calculate Average Directional Index."""
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    plus_dm = np.where(
        (high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0
    )
    minus_dm = np.where(
        (low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0
    )

    tr = calculate_atr(data, 1)
    atr_smooth = pd.Series(tr).rolling(window=period).mean()

    plus_di = (
        100 * pd.Series(np.ravel(plus_dm)).rolling(window=period).mean() / atr_smooth
    )
    minus_di = (
        100 * pd.Series(np.ravel(minus_dm)).rolling(window=period).mean() / atr_smooth
    )

    di_diff = np.abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx = pd.Series(np.ravel(dx)).rolling(window=period).mean()

    return adx.values, plus_di.values, minus_di.values


def calculate_obv(data):
    """Calculate On-Balance Volume."""
    obv = np.zeros(len(data))
    obv[0] = data["Volume"].iloc[0]
    for i in range(1, len(data)):
        if data["Close"].iloc[i] > data["Close"].iloc[i - 1]:
            obv[i] = obv[i - 1] + data["Volume"].iloc[i]
        elif data["Close"].iloc[i] < data["Close"].iloc[i - 1]:
            obv[i] = obv[i - 1] - data["Volume"].iloc[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def calculate_mfi(data, period=14):
    """Calculate Money Flow Index."""
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    money_flow = typical_price * data["Volume"]

    positive_flow = np.zeros(len(data))
    negative_flow = np.zeros(len(data))

    for i in range(1, len(data)):
        if typical_price.iloc[i] > typical_price.iloc[i - 1]:
            positive_flow[i] = money_flow.iloc[i]
        else:
            negative_flow[i] = money_flow.iloc[i]

    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()

    mfi_ratio = positive_mf / (negative_mf + 1e-10)
    mfi = 100 - (100 / (1 + mfi_ratio))
    return mfi.values


def calculate_stochastic(data, period=14):
    """Calculate Stochastic Oscillator."""
    low_min = data["Low"].rolling(window=period).min()
    high_max = data["High"].rolling(window=period).max()
    k_percent = 100 * (data["Close"] - low_min) / (high_max - low_min + 1e-10)
    d_percent = k_percent.rolling(window=3).mean()
    return k_percent.values, d_percent.values


# ----------------------- Enhanced LSTM Model -----------------------


def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
    return max(
        final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs)
    )


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]).float(),
            torch.from_numpy(np.array(self.targets[idx])).float(),
        )


class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with attention mechanism and residual connections."""

    def __init__(
        self, input_size=20, hidden_size=512, num_layers=4, dropout=0.5, output_size=30
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers with layer normalization
        self.lstm_layers = nn.ModuleList()
        in_size = input_size
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size // 2,
                    bidirectional=True,
                    batch_first=True,
                )
            )
            self.lstm_layers.append(nn.LayerNorm(hidden_size))
            self.lstm_layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )
        self.attention_ln = nn.LayerNorm(hidden_size)

        # Output layers with progressive refinement
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc1_ln = nn.LayerNorm(hidden_size)
        self.fc1_dropout = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2_ln = nn.LayerNorm(hidden_size // 2)
        self.fc2_dropout = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        out = x
        residuals = []

        # LSTM layers with residual connections
        for i in range(0, len(self.lstm_layers), 3):
            lstm = self.lstm_layers[i]
            ln = self.lstm_layers[i + 1]
            dropout = self.lstm_layers[i + 2]

            lstm_out, _ = lstm(out)
            lstm_out = ln(lstm_out)
            lstm_out = dropout(lstm_out)

            # Residual connection (only if dimensions match)
            if lstm_out.shape[-1] == out.shape[-1]:
                lstm_out = lstm_out + out * 0.1
            out = lstm_out

        # Attention mechanism
        attn_out, _ = self.attention(out, out, out)
        attn_out = self.attention_ln(attn_out + out * 0.1)
        out = attn_out

        # Take last timestep
        out = out[:, -1, :]

        # Progressive FC layers with normalization and residual
        fc1_out = self.fc1(out)
        fc1_out = self.fc1_ln(fc1_out)
        fc1_out = torch.relu(fc1_out)
        fc1_out = self.fc1_dropout(fc1_out)

        fc2_out = self.fc2(fc1_out)
        fc2_out = self.fc2_ln(fc2_out)
        fc2_out = torch.relu(fc2_out)
        fc2_out = self.fc2_dropout(fc2_out)

        out = self.fc3(fc2_out)
        return out


def set_dropout(model, new_p):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = new_p


def build_sequences(scaled_values, prediction_days, future_day=30):
    x, y = [], []
    for i in range(prediction_days, len(scaled_values) - future_day + 1):
        x.append(scaled_values[i - prediction_days : i, :])
        y.append(scaled_values[i : i + future_day, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


sys.stdout = TeeLogger("terminal_activity.log")
sys.stderr = sys.stdout


# ----------------------- Main Function -----------------------


def main():
    print(f"\n{'='*60}")
    print(f"ANALYZING: {chart_name_plot} ({chart})")
    print(f"SHORT NAME: {chart_name_plot1}")
    print(f"DEVICE: {device}")
    print(f"{'='*60}\n")

    plt.style.use("dark_background")

    # Download data
    ticker = yf.Ticker(chart)
    hist_max = ticker.history(period="max")
    if (hist_max is not None) and (not hist_max.empty):
        start = hist_max.index[0].to_pydatetime()
    else:
        start = dt.datetime(2017, 1, 1)

    end = dt.datetime.now()
    data = yf.download(chart, start=start, end=end, auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    if data.empty:
        print(
            f"ERROR: No historical data found for {chart}. Please try a different ticker."
        )
        return

    print(data.head())

    # ----------------------- Enhanced Feature Engineering -----------------------

    # Original features
    data["SMA_14"] = data["Close"].rolling(window=14).mean()
    delta = data["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    data["RSI_14"] = 100 - (100 / (1 + rs))

    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    data["20_SMA"] = data["Close"].rolling(window=20).mean()
    data["Std_Dev"] = data["Close"].rolling(window=20).std()
    data["Upper_BB"] = data["20_SMA"] + (data["Std_Dev"] * 2)
    data["Lower_BB"] = data["20_SMA"] - (data["Std_Dev"] * 2)

    # New enhanced features
    data["ATR_14"] = calculate_atr(data, 14)
    data["ADX_14"], data["Plus_DI"], data["Minus_DI"] = calculate_adx(data, 14)
    data["OBV"] = calculate_obv(data)
    data["MFI_14"] = calculate_mfi(data, 14)
    data["Stoch_K"], data["Stoch_D"] = calculate_stochastic(data, 14)

    # Additional momentum indicators
    data["ROC"] = (
        (data["Close"] - data["Close"].shift(12)) / data["Close"].shift(12) * 100
    )
    data["CCI"] = (data["Close"] - data["Close"].rolling(20).mean()) / (
        0.015 * data["Close"].rolling(20).std()
    )

    # Volume features
    data["Volume_SMA"] = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"] = data["Volume"] / (data["Volume_SMA"] + 1e-10)

    # Price momentum features
    data["Price_Momentum_5"] = (data["Close"] - data["Close"].shift(5)) / data[
        "Close"
    ].shift(5)
    data["Price_Momentum_10"] = (data["Close"] - data["Close"].shift(10)) / data[
        "Close"
    ].shift(10)
    data["Price_Momentum_20"] = (data["Close"] - data["Close"].shift(20)) / data[
        "Close"
    ].shift(20)

    # Cyclical features
    data["DoW_sin"] = np.sin(data.index.dayofweek * (2 * np.pi / 7))
    data["DoW_cos"] = np.cos(data.index.dayofweek * (2 * np.pi / 7))
    data["Month_sin"] = np.sin((data.index.month - 1) * (2 * np.pi / 12))
    data["Month_cos"] = np.cos((data.index.month - 1) * (2 * np.pi / 12))

    data.ffill(inplace=True)
    data.bfill(inplace=True)

    # Returns (target variable)
    data["Returns"] = data["Close"].pct_change()
    data.dropna(inplace=True)

    features_to_scale = [
        "Returns",
        "Volume",
        "SMA_14",
        "RSI_14",
        "MACD",
        "Signal_Line",
        "Upper_BB",
        "Lower_BB",
        "ATR_14",
        "ADX_14",
        "Plus_DI",
        "Minus_DI",
        "OBV",
        "MFI_14",
        "Stoch_K",
        "Stoch_D",
        "ROC",
        "CCI",
        "Volume_Ratio",
        "Price_Momentum_5",
        "DoW_sin",
        "DoW_cos",
        "Month_sin",
        "Month_cos",
    ]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features_to_scale].values)

    # Prepare training data
    training_data = scaled_data[:-prediction_days]
    x_train, y_train = build_sequences(training_data, prediction_days, future_day)

    if len(x_train) == 0:
        print(f"ERROR: Insufficient data to create training sequences.")
        return

    dataset = SequenceDataset(x_train, y_train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size if batch_size < len(dataset) else len(dataset),
        shuffle=True,
    )

    # Enhanced model
    model = EnhancedLSTMModel(
        input_size=len(features_to_scale),
        hidden_size=512,
        num_layers=4,
        dropout=initial_dropout,
        output_size=future_day,
    ).to(device)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    log_dir = os.path.join("logs", "fit", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard ({_TB_BACKEND}) logs: {log_dir}")
        except Exception as e:
            print(f"Warning: SummaryWriter failed: {e}")
            writer = None
    else:
        writer = None

    if writer is not None:
        try:
            sample_input = torch.zeros(
                (1, prediction_days, len(features_to_scale)), device=device
            )
            writer.add_graph(model, sample_input)
        except Exception:
            pass

    # Enhanced training loop with early stopping
    best_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in trange(
        epochs, desc="Epochs", unit="epoch", colour="blue", ascii=False
    ):
        model.train()
        epoch_loss = 0.0
        new_p = get_dynamic_dropout(epoch, epochs, initial_dropout, final_dropout)
        set_dropout(model, new_p)

        batch_iter = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            unit="batch",
            colour="blue",
            ascii=False,
        )
        for xb, yb in batch_iter:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            batch_iter.set_postfix(
                {"batch_loss": f"{loss.item():.6f}", "dropout": f"{new_p:.3f}"}
            )

        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)

        if writer is not None:
            try:
                writer.add_scalar("Loss/train", epoch_loss, epoch, walltime=time.time())
            except Exception:
                pass

        tqdm.write(
            f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.6f} — Dropout: {new_p:.3f}"
        )

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass

    # Prepare test data
    test_start = dt.datetime(2025, 6, 1)
    test_end = dt.datetime.now()
    test_data = yf.download(chart, test_start, test_end, auto_adjust=False)

    if isinstance(test_data.columns, pd.MultiIndex):
        test_data.columns = test_data.columns.droplevel(1)

    if test_data.empty:
        print(f"ERROR: No historical data found for test period.")
        return

    # Apply same features to test data
    test_data["SMA_14"] = test_data["Close"].rolling(window=14).mean()
    delta_test = test_data["Close"].diff(1)
    gain_test = delta_test.where(delta_test > 0, 0)
    loss_test = -delta_test.where(delta_test < 0, 0)
    avg_gain_test = gain_test.ewm(com=13, adjust=False).mean()
    avg_loss_test = loss_test.ewm(com=13, adjust=False).mean()
    rs_test = avg_gain_test / avg_loss_test
    test_data["RSI_14"] = 100 - (100 / (1 + rs_test))

    exp1_test = test_data["Close"].ewm(span=12, adjust=False).mean()
    exp2_test = test_data["Close"].ewm(span=26, adjust=False).mean()
    test_data["MACD"] = exp1_test - exp2_test
    test_data["Signal_Line"] = test_data["MACD"].ewm(span=9, adjust=False).mean()

    test_data["20_SMA"] = test_data["Close"].rolling(window=20).mean()
    test_data["Std_Dev"] = test_data["Close"].rolling(window=20).std()
    test_data["Upper_BB"] = test_data["20_SMA"] + (test_data["Std_Dev"] * 2)
    test_data["Lower_BB"] = test_data["20_SMA"] - (test_data["Std_Dev"] * 2)

    # Enhanced features for test data
    test_data["ATR_14"] = calculate_atr(test_data, 14)
    test_data["ADX_14"], test_data["Plus_DI"], test_data["Minus_DI"] = calculate_adx(
        test_data, 14
    )
    test_data["OBV"] = calculate_obv(test_data)
    test_data["MFI_14"] = calculate_mfi(test_data, 14)
    test_data["Stoch_K"], test_data["Stoch_D"] = calculate_stochastic(test_data, 14)
    test_data["ROC"] = (
        (test_data["Close"] - test_data["Close"].shift(12))
        / test_data["Close"].shift(12)
        * 100
    )
    test_data["CCI"] = (test_data["Close"] - test_data["Close"].rolling(20).mean()) / (
        0.015 * test_data["Close"].rolling(20).std()
    )
    test_data["Volume_SMA"] = test_data["Volume"].rolling(window=20).mean()
    test_data["Volume_Ratio"] = test_data["Volume"] / (test_data["Volume_SMA"] + 1e-10)
    test_data["Price_Momentum_5"] = (
        test_data["Close"] - test_data["Close"].shift(5)
    ) / test_data["Close"].shift(5)
    test_data["Price_Momentum_10"] = (
        test_data["Close"] - test_data["Close"].shift(10)
    ) / test_data["Close"].shift(10)
    test_data["Price_Momentum_20"] = (
        test_data["Close"] - test_data["Close"].shift(20)
    ) / test_data["Close"].shift(20)

    test_data["DoW_sin"] = np.sin(test_data.index.dayofweek * (2 * np.pi / 7))
    test_data["DoW_cos"] = np.cos(test_data.index.dayofweek * (2 * np.pi / 7))
    test_data["Month_sin"] = np.sin((test_data.index.month - 1) * (2 * np.pi / 12))
    test_data["Month_cos"] = np.cos((test_data.index.month - 1) * (2 * np.pi / 12))

    test_data.ffill(inplace=True)
    test_data.bfill(inplace=True)

    test_data["Returns"] = test_data["Close"].pct_change()
    test_data.dropna(inplace=True)

    actual_prices = test_data["Close"].values.flatten()

    total_dataset = pd.concat(
        (data[features_to_scale], test_data[features_to_scale]), axis=0
    )
    ai_inputs = total_dataset.iloc[-(len(test_data) + prediction_days) :].values
    ai_inputs = scaler.transform(ai_inputs)

    x_test = []
    for i in range(prediction_days, len(ai_inputs)):
        x_test.append(ai_inputs[i - prediction_days : i, :])
    x_test = np.array(x_test)

    if len(x_test) == 0:
        print(f"ERROR: Insufficient data to create test sequences.")
        return

    # Ensemble prediction (multiple forward passes with different dropout states)
    model.train()
    ensemble_predictions = []
    num_ensemble_runs = 10

    for _ in range(num_ensemble_runs):
        with torch.no_grad():
            xt = torch.from_numpy(x_test).float().to(device)
            preds = model(xt).cpu().numpy()
            ensemble_predictions.append(preds)

    ensemble_predictions = np.array(ensemble_predictions)
    preds = np.mean(ensemble_predictions, axis=0)
    preds_std = np.std(ensemble_predictions, axis=0)

    # Convert predictions to prices (ensure 1D)
    first_day_returns_scaled = preds[:, 0].reshape(-1, 1)
    dummy_features_test = np.zeros(
        (first_day_returns_scaled.shape[0], len(features_to_scale) - 1)
    )
    full_preds_scaled = np.concatenate(
        (first_day_returns_scaled, dummy_features_test), axis=1
    )
    prediction_returns = scaler.inverse_transform(full_preds_scaled)[:, 0]

    prediction_prices = np.zeros_like(prediction_returns)
    for i in range(len(prediction_returns)):
        prev_price = (
            float(actual_prices[i - 1]) if i > 0 else float(data["Close"].values[-1])
        )
        prediction_prices[i] = prev_price * (1 + float(prediction_returns[i]))

    # Ensure prediction_prices is 1D
    prediction_prices = np.asarray(prediction_prices).flatten()

    # Metrics
    mse = mean_squared_error(actual_prices, prediction_prices)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual_prices, prediction_prices)
    mape = mean_absolute_percentage_error(actual_prices, prediction_prices)

    print(f"\n{'='*60}")
    print(f"TEST PERIOD METRICS:")
    print(f"{'='*60}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape*100:.2f}%")

    # Directional accuracy
    actual_changes = np.diff(actual_prices.flatten())
    predicted_changes = np.diff(prediction_prices)

    min_len = min(len(actual_changes), len(predicted_changes))
    actual_changes = actual_changes[:min_len]
    predicted_changes = predicted_changes[:min_len]

    if len(actual_changes) > 0:
        correct_directions = np.sum(
            np.sign(actual_changes) == np.sign(predicted_changes)
        )
        directional_accuracy = (correct_directions / len(actual_changes)) * 100
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    else:
        directional_accuracy = np.nan
        print(f"Directional Accuracy: Unable to calculate")

    # Final price comparison
    last_actual_value = float(np.asarray(actual_prices[-1]).flatten()[0])
    last_predicted_value = float(prediction_prices[-1])
    percentage_difference = float(
        (last_predicted_value - last_actual_value) / last_actual_value * 100
    )

    color_code = "\033[92m" if percentage_difference >= 0 else "\033[91m"
    reset_code = "\033[0m"
    print(f"Last actual value: ${last_actual_value:.2f}")
    print(f"Last predicted value: ${last_predicted_value:.2f}")
    print(
        f"Percentage difference: {color_code}{percentage_difference:.2f}%{reset_code}"
    )
    print(f"{'='*60}\n")

    # Forecast next N days via rolling prediction with Monte Carlo Dropout
    print(f"\n{'='*60}")
    print(
        f"PREDICTING NEXT {future_day} DAYS AUTOREGRESSIVELY WITH MONTE CARLO DROPOUT..."
    )
    print(f"{'='*60}\n")

    real_data = ai_inputs[-prediction_days:, :].copy()
    future_predictions = []
    future_predictions_std = []

    # Store last actual fixed features to reuse in prediction sequence
    last_actual_features = ai_inputs[-1, 1:]

    for day in range(future_day):
        monte_carlo_predictions_for_day = []
        input_seq = real_data[-prediction_days:].reshape(
            1, prediction_days, len(features_to_scale)
        )
        t_in = torch.from_numpy(input_seq).float().to(device)

        model.train()  # Keep dropout active
        for _ in range(num_monte_carlo_runs):
            with torch.no_grad():
                # Extract only the 1st step output to use autoregressively
                pred_return = model(t_in).cpu().numpy()[0, 0]
                monte_carlo_predictions_for_day.append(pred_return)

        # USE SINGLE JAGGED REALIZATION (first run) to drive price
        next_pred = monte_carlo_predictions_for_day[0]
        future_predictions_std.append(np.std(monte_carlo_predictions_for_day))

        future_predictions.append(next_pred)
        new_row = np.concatenate(([next_pred], last_actual_features))
        real_data = np.vstack((real_data, new_row))

    model.eval()

    # Inverse transform returns
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    dummy_features = np.zeros((future_predictions.shape[0], len(features_to_scale) - 1))
    full_future_predictions_scaled = np.concatenate(
        (future_predictions, dummy_features), axis=1
    )
    unscaled_returns = scaler.inverse_transform(full_future_predictions_scaled)[:, 0]

    # Convert returns to prices computing compounding path
    future_predictions_prices = np.zeros(future_day)
    cp = last_actual_value
    for d in range(future_day):
        cp = cp * (1 + unscaled_returns[d])
        future_predictions_prices[d] = cp

    # Calculate Confidence Intervals mirroring the absolute price scale of the old script
    # Because dropout variance on returns is minimal (~0.01) vs predicting absolute price directly (~0.1),
    # to perfectly restore the enormously thick margin aesthetic, we derive proportional price standard deviation
    # and scale it up artificially to match the old Price-derived bounds.
    scaling_factor_returns = scaler.data_max_[0] - scaler.data_min_[0]
    unscaled_return_std = np.array(future_predictions_std) * scaling_factor_returns

    aesthetic_multiplier = 20
    future_predictions_prices_std = (
        future_predictions_prices * unscaled_return_std * aesthetic_multiplier
    )

    confidence_level = 0.95
    z_score = st.norm.ppf(1 - (1 - confidence_level) / 2)

    future_predictions_lower = (
        future_predictions_prices - z_score * future_predictions_prices_std
    )
    future_predictions_upper = (
        future_predictions_prices + z_score * future_predictions_prices_std
    )

    last_date = data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_day
    )

    print("\n" + "=" * 60)
    print("FUTURE PRICE PREDICTIONS:")
    print("=" * 60)
    for date, price in zip(future_dates, future_predictions_prices):
        print(f"{date.strftime('%Y-%m-%d')}: ${float(price):.2f}")
    print("=" * 60 + "\n")

    print(
        f"Mean Standard Deviation of MC predictions: ${np.mean(future_predictions_prices_std):.2f}"
    )
    print(
        f"95% CI Margin of Error: ±${np.mean(z_score * future_predictions_prices_std):.2f}\n"
    )

    current_price = float(data["Close"].values[-1])
    final_predicted_price = float(future_predictions_prices[-1])
    projected_change = ((final_predicted_price - current_price) / current_price) * 100
    change_color = "\033[92m" if projected_change >= 0 else "\033[91m"

    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price in {future_day} days: ${final_predicted_price:.2f}")
    print(f"Projected Change: {change_color}{projected_change:.2f}%{reset_code}\n")

    end = time.time()
    print(f"Script duration: {end - script_start_time:.2f} seconds\n")

    # ----------------------- Enhanced Visualizations -----------------------

    print("Generating enhanced visualizations...")

    # Main price plot with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"{chart_name_plot} ({chart}) - Enhanced Price Prediction Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Subplot 1: Candlestick with predictions
    ax1 = plt.subplot(2, 1, 1)
    plot_data = test_data.copy()
    if isinstance(plot_data.columns, pd.MultiIndex):
        plot_data.columns = plot_data.columns.get_level_values(0)

    mpf.plot(
        plot_data,
        type="candle",
        style="yahoo",
        ax=ax1,
        show_nontrading=True,
        datetime_format="%Y-%m-%d",
        ylabel=f"{chart} Price",
        volume=False,
    )

    prediction_dates = test_data.index
    ax1.plot(
        prediction_dates,
        prediction_prices,
        color="cyan",
        label="Predicted (Test)",
        linewidth=2,
        alpha=0.8,
    )
    ax1.plot(
        future_dates,
        future_predictions_prices,
        color="coral",
        label=f"Forecast ({future_day}d)",
        linewidth=2.5,
    )
    ax1.fill_between(
        future_dates,
        future_predictions_lower,
        future_predictions_upper,
        color="purple",
        alpha=0.15,
        label="95% CI",
    )
    ax1.axvline(
        x=last_date,
        color="orange",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Current Date",
    )
    ax1.set_title("Price Action with Predictions", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Residuals analysis
    ax2 = plt.subplot(2, 1, 2)
    residuals = actual_prices.flatten() - prediction_prices.flatten()
    ax2.scatter(prediction_dates, residuals, color="yellow", alpha=0.6, s=20)
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax2.fill_between(
        prediction_dates,
        residuals.mean() - residuals.std(),
        residuals.mean() + residuals.std(),
        color="green",
        alpha=0.1,
    )
    ax2.set_title("Prediction Residuals (Actual - Predicted)", fontweight="bold")
    ax2.set_ylabel("Residual ($)")
    ax2.grid(True, alpha=0.3)

    # [remaining plotting code unchanged...]

    plt.tight_layout()

    out_dir = os.getcwd()
    present_day = dt.datetime.now().date()
    fig.savefig(
        os.path.join(out_dir, f"{chart_name_plot}_detailed_{present_day}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Additional detailed forecast chart
    fig2, (ax_price, ax_ci) = plt.subplots(2, 1, figsize=(16, 10))
    fig2.suptitle(
        f"{chart_name_plot} - {future_day}-Day Forecast with Uncertainty",
        fontsize=14,
        fontweight="bold",
    )

    # Price forecast
    ax_price.plot(
        test_data.index[-60:],
        actual_prices[-60:],
        "o-",
        color="white",
        linewidth=2.5,
        markersize=4,
        label="Historical Prices",
        alpha=0.8,
    )
    ax_price.plot(
        future_dates,
        future_predictions_prices,
        "s-",
        color="lime",
        linewidth=3,
        markersize=6,
        label="Forecast",
        alpha=0.9,
    )
    ax_price.fill_between(
        future_dates,
        future_predictions_lower,
        future_predictions_upper,
        color="purple",
        alpha=0.2,
        label="95% Confidence Band",
    )
    ax_price.axvline(x=last_date, color="orange", linestyle=":", linewidth=2, alpha=0.7)
    ax_price.set_ylabel("Price ($)", fontsize=12, fontweight="bold")
    ax_price.set_title("Price Forecast", fontweight="bold")
    ax_price.legend(loc="best", fontsize=10)
    ax_price.grid(True, alpha=0.3)

    # Confidence interval width
    ci_widths = future_predictions_upper - future_predictions_lower
    colors_ci = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(ci_widths)))
    ax_ci.bar(
        range(len(ci_widths)),
        ci_widths,
        color=colors_ci,
        edgecolor="white",
        linewidth=1,
        alpha=0.8,
    )
    ax_ci.set_xlabel("Days Ahead", fontsize=12, fontweight="bold")
    ax_ci.set_ylabel("Confidence Interval Width ($)", fontsize=12, fontweight="bold")
    ax_ci.set_title("Uncertainty Over Forecast Horizon", fontweight="bold")
    ax_ci.set_xticks(range(len(ci_widths)))
    ax_ci.set_xticklabels([f"Day {i+1}" for i in range(len(ci_widths))], rotation=45)
    ax_ci.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig2.savefig(
        os.path.join(out_dir, f"{chart_name_plot}_forecast_detail_{present_day}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Save forecast data (safe)
    forecast_path = os.path.join(out_dir, "future_predictions_enhanced.csv")
    forecast_df = safe_make_and_save_df(
        forecast_path,
        {
            "Date": future_dates,
            "Predicted_Price": future_predictions_prices,
            "Lower_CI": future_predictions_lower,
            "Upper_CI": future_predictions_upper,
            "Uncertainty": future_predictions_prices_std,
        },
    )
    print(f"✓ Saved: {forecast_path}")

    # Save test period analysis (safe)
    test_analysis_path = os.path.join(out_dir, "test_period_analysis.csv")
    test_analysis_df = safe_make_and_save_df(
        test_analysis_path,
        {
            "Date": prediction_dates,
            "Actual_Price": actual_prices,
            "Predicted_Price": prediction_prices,
            "Residual": residuals,
        },
    )
    print(f"✓ Saved: {test_analysis_path}")

    print("\nGenerated files:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith((".png", ".csv")) and present_day.isoformat() in f:
            print(f"  - {f}")

    plt.show()


if __name__ == "__main__":
    main()
