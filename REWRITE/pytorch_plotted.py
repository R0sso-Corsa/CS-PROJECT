import time
import sys
import triton

script_start_time = time.time()
import torch_optimizer as optim  # EXTENDED LIST OF TORCH OPTIMISERS [IMPORTANT == GO THROUGH EFFICIENT OPTIONS (100+ to use)]
import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import yfinance as yf
import mplfinance as mpf
import seaborn as sns  # Added seaborn for aesthetics

import torch
import torch.nn as nn

# progress bars
from tqdm.auto import trange, tqdm

# Keep cuDNN enabled on CUDA for fast LSTM kernels.
# Disable only on ROCm where some users hit MIOpen runtime issues.
torch.backends.cudnn.enabled = torch.version.hip is None
from torch.utils.data import Dataset, DataLoader

# TensorBoard writer: prefer torch's bundled SummaryWriter, fall back to tensorboardX, else disable
try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_BACKEND = "torch"
except Exception:
    try:
        from tensorboardX import SummaryWriter  # type: ignore

        _TB_BACKEND = "tensorboardX"
    except Exception:
        SummaryWriter = None
        _TB_BACKEND = None

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)  # Added mean_absolute_error
import scipy.stats as st  # Added for confidence interval calculation
import math  # Added for sqrt in RMSE calculation


# ----------------------- Configuration -----------------------

import sys
import datetime


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        # Adds a timestamp only to the start of new lines in the log file
        if message.strip():
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.log.write(timestamp + message + "\n")
        else:
            self.log.write(message)

        self.terminal.write(message)
        self.log.flush()  # Writes to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()


def finder(query: str):
    """Search for symbols matching `query` using yfinance and return the Search response."""
    try:
        return yf.Search(query)
    except Exception:
        return None


def choose_chart_interactive():
    """Ask the user to enter a ticker or search for one interactively.

    Return a chosen ticker symbol (string).
    """
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
                        f"{i + 1}. {q.get('symbol')} - {q.get('shortname')} ({q.get('quoteType')})"
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


prediction_days = 30
future_day = 30
epochs = 40
batch_size = 16
initial_dropout = 0.6
final_dropout = 0.1
train_time = 2
num_monte_carlo_runs = 100  # Number of forward passes for Monte Carlo Dropout
optimizer_name = "Ranger"  # Name of optimizer from torch_optimizer (e.g. "Ranger", "Adafactor", "RAdam")
show_batch_progress = True  # False is faster; per-batch tqdm adds Python overhead.
use_amp = False  # Often slower for LSTM on ROCm/AMD; enable manually to test.


# ----------------------- End Configuration -----------------------


def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
    return max(
        final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs)
    )


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        # Materialize tensors once to avoid per-sample numpy->tensor conversion overhead.
        self.sequences = torch.from_numpy(sequences).float()
        self.targets = torch.from_numpy(np.asarray(targets)).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=500, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = input_size
        for i in range(num_layers):
            # MODIFIED: Changed to bidirectional LSTM
            self.layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size // 2,
                    bidirectional=True,
                    batch_first=True,
                )
            )
            self.layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size  # Bidirectional LSTM output size is 2 * hidden_size

        # final linear to produce single value
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out = x
        for i in range(0, len(self.layers), 2):
            lstm = self.layers[i]
            dropout = self.layers[i + 1]
            out, _ = lstm(
                out
            )  # out shape: (batch, seq_len, hidden_size * 2 for bidirectional)
            out = dropout(out)

        # take last timestep
        out = out[
            :, -1, :
        ]  # For bidirectional, output is concatenated of forward and backward at last timestep
        out = self.fc(out)
        return out


def set_dropout(model, new_p):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = new_p


def build_sequences(scaled_values, prediction_days):
    x, y = [], []
    for i in range(prediction_days, len(scaled_values)):
        x.append(scaled_values[i - prediction_days : i, :])
        y.append(scaled_values[i, 0])  # y_train still predicts only 'Close'
    x = np.array(x)
    y = np.array(y)
    # x = np.reshape(x, (x.shape[0], x.shape[1], 1)) # No longer need to reshape to 1 feature
    return x, y


sys.stdout = TeeLogger("terminal_activity.log")
sys.stderr = sys.stdout  # thankfully captures the Traceback/Errors


def main():
    # ask user for chart symbol (interactive search available)
    chart = choose_chart_interactive()
    chart_info = yf.Ticker(chart).info
    chart_name_plot = chart_info.get("longName") or chart
    chart_name_plot_short = chart_info.get("shortName") or chart

    device_type = input("Enter device type (cpu/gpu): ").strip().lower()
    device = torch.device(
        "cuda" if (device_type == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
    )
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["GPU_MAX_HEAP_SIZE"] = "100"
    os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    # Keep deep debug logs off by default; they add major runtime overhead.
    os.environ["DNNL_VERBOSE"] = "0"
    os.environ["CUDNN_LOGINFO_DBG"] = "0"
    os.environ["PYTORCH_JIT_LOG_LEVEL"] = ""
    os.environ["PYTORCH_JIT_LOGS"] = ""
    os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"  # RDNA4-specific scheduling

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(1.00, device=0)

    if os.environ.get("TORCH_VERBOSE_LOGS", "0") == "1":
        torch._logging.set_logs(
            graph_breaks=True,
            recompiles=True,
            dynamo=20,
            inductor=20,
        )

    print(f"\n{'=' * 60}")
    print(f"ANALYZING: {chart_name_plot} ({chart})")
    print(f"SHORT NAME: {chart_name_plot_short}")
    print(f"DEVICE: {device}")
    print(f"{'=' * 60}\n")

    # Apply dark theme for improved aesthetics
    plt.style.use("dark_background")

    # figure for plotting
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"{chart_name_plot} - Detailed Prediction Analysis",
        fontsize=16,
        fontweight="bold",
    )
    graph = fig.add_subplot(2, 1, 1)

    # determine earliest date available
    ticker = yf.Ticker(chart)
    hist_max = ticker.history(period="max")
    if (hist_max is not None) and (not hist_max.empty):
        start = hist_max.index[0].to_pydatetime()
    else:
        start = dt.datetime(2017, 1, 1)

    end = dt.datetime.now()
    data = yf.download(chart, start=start, end=end, auto_adjust=True)

    # Handle case where yfinance returns insufficient data
    if data.empty:
        print(
            f"ERROR: No historical data found for {chart}. Please try a different ticker."
        )
        return

    print(data.head())

    # 1. Calculate the 14-day Simple Moving Average (SMA)
    data["SMA_14"] = data["Close"].rolling(window=14).mean()

    # 2. Calculate the 14-day Relative Strength Index (RSI)
    # Calculate daily price changes
    delta = data["Close"].diff(1)

    # Separate positive (gain) and negative (loss) changes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Compute exponential moving averages (EMA) for gains and losses
    # Using com=13 for a 14-day EMA (span = com + 1)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()

    # Calculate Relative Strength (RS) and then RSI
    rs = avg_gain / avg_loss
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # 3. Calculate MACD
    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # 4. Calculate Bollinger Bands
    data["20_SMA"] = data["Close"].rolling(window=20).mean()
    data["Std_Dev"] = data["Close"].rolling(window=20).std()
    data["Upper_BB"] = data["20_SMA"] + (data["Std_Dev"] * 2)
    data["Lower_BB"] = data["20_SMA"] - (data["Std_Dev"] * 2)

    # 5. Fill any NaN values that result from SMA and RSI calculations
    # Fill forward then backward to handle NaNs at the beginning and end
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    # MODIFIED: scaled_data now includes 'Close', 'Volume', 'SMA_14', 'RSI_14', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB'
    scaled_data = scaler.fit_transform(
        data[
            [
                "Close",
                "Volume",
                "SMA_14",
                "RSI_14",
                "MACD",
                "Signal_Line",
                "Upper_BB",
                "Lower_BB",
            ]
        ].values
    )

    # prepare training data (exclude last prediction_days to avoid incomplete windows)
    training_data = scaled_data[:-prediction_days]
    x_train, y_train = build_sequences(training_data, prediction_days)

    # Check if x_train is empty
    if len(x_train) == 0:
        print(
            f"ERROR: Insufficient data to create training sequences. Only {len(training_data)} data points available after feature engineering, but {prediction_days} prediction days are required."
        )
        print(
            "Please choose a ticker with more historical data or reduce 'prediction_days'."
        )
        return

    dataset = SequenceDataset(x_train, y_train)
    use_gpu = device.type == "cuda"
    # Interactive input is now inside main(), so worker spawn is safe on Windows.
    use_worker_processes = use_gpu
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size if batch_size < len(dataset) else len(dataset),
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=(2 if use_worker_processes else 0),
        persistent_workers=use_worker_processes,
    )

    model = LSTMModel(
        input_size=8, hidden_size=500, num_layers=4, dropout=initial_dropout
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), weight_decay=0.05)
    amp_enabled = bool(use_amp and (device.type == "cuda") and (torch.version.hip is None))
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # tensorboard writer (create only if available)
    log_dir = os.path.join("logs", "fit", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard ({_TB_BACKEND}) logs: {log_dir}")
        except Exception as e:
            print(f"Warning: SummaryWriter failed to initialize: {e}")
            writer = None
    else:
        writer = None
        print("TensorBoard not available; continuing without it.")

    # try to log the model graph (best-effort)
    if writer is not None:
        try:
            sample_input = torch.zeros((1, prediction_days, 8), device=device)
            writer.add_graph(model, sample_input)
        except Exception:
            # some models / environments don't support add_graph; ignore
            pass

    # training loop with dynamic dropout
    # training with progress bars (trange for epochs, tqdm for batch progress)
    for epoch in trange(
        epochs, desc="Epochs", unit="epoch", colour="blue", ascii=False
    ):
        model.train()
        epoch_loss_accum = torch.zeros((), device=device)
        new_p = get_dynamic_dropout(epoch, epochs, initial_dropout, final_dropout)
        set_dropout(model, new_p)

        if show_batch_progress:
            batch_iter = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
                unit="batch",
                colour="blue",
                ascii=False,
                mininterval=0.5,
            )
        else:
            batch_iter = dataloader

        for batch_idx, (xb, yb) in enumerate(batch_iter, start=1):
            xb = xb.to(device, non_blocking=use_gpu)
            yb = yb.to(device, non_blocking=use_gpu).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(xb)
                loss = criterion(outputs, yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss_accum += loss.detach() * xb.size(0)

            # Avoid per-batch .item() sync; update UI periodically.
            if show_batch_progress and (
                (batch_idx % 25 == 0) or (batch_idx == len(dataloader))
            ):
                batch_iter.set_postfix({"dropout": f"{new_p:.3f}"})

        epoch_loss = (epoch_loss_accum / len(dataset)).item()
        if writer is not None:
            try:
                writer.add_scalar("Loss/train", epoch_loss, epoch, walltime=time.time())
            except Exception:
                pass

        # update epoch-level progress description
        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} — Loss: {epoch_loss:.6f} — Dropout: {new_p:.3f}"
        )

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass

    # Prepare test data
    use_earliest_test = False
    if use_earliest_test:
        test_start = start
    else:
        test_start = dt.datetime(2025, 6, 1)
    test_end = dt.datetime.now()
    test_data = yf.download(chart, test_start, test_end, auto_adjust=True)

    # Handle case where yfinance returns insufficient data for test period
    if test_data.empty:
        print(
            f"ERROR: No historical data found for {chart} in the test period. Please try a different ticker or adjust the test_start date."
        )
        return

    # Calculate SMA and RSI for test_data as well
    test_data["SMA_14"] = test_data["Close"].rolling(window=14).mean()
    delta_test = test_data["Close"].diff(1)
    gain_test = delta_test.where(delta_test > 0, 0)
    loss_test = -delta_test.where(delta_test < 0, 0)
    avg_gain_test = gain_test.ewm(com=13, adjust=False).mean()
    avg_loss_test = loss_test.ewm(com=13, adjust=False).mean()
    rs_test = avg_gain_test / avg_loss_test
    test_data["RSI_14"] = 100 - (100 / (1 + rs_test))

    # Calculate MACD for test_data
    exp1_test = test_data["Close"].ewm(span=12, adjust=False).mean()
    exp2_test = test_data["Close"].ewm(span=26, adjust=False).mean()
    test_data["MACD"] = exp1_test - exp2_test
    test_data["Signal_Line"] = test_data["MACD"].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands for test_data
    test_data["20_SMA"] = test_data["Close"].rolling(window=20).mean()
    test_data["Std_Dev"] = test_data["Close"].rolling(window=20).std()
    test_data["Upper_BB"] = test_data["20_SMA"] + (test_data["Std_Dev"] * 2)
    test_data["Lower_BB"] = test_data["20_SMA"] - (test_data["Std_Dev"] * 2)

    test_data.ffill(inplace=True)
    test_data.bfill(inplace=True)

    actual_prices = test_data["Close"].values

    # MODIFIED: total_dataset now includes 'Close', 'Volume', 'SMA_14', 'RSI_14', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB'
    total_dataset = pd.concat(
        (
            data[
                [
                    "Close",
                    "Volume",
                    "SMA_14",
                    "RSI_14",
                    "MACD",
                    "Signal_Line",
                    "Upper_BB",
                    "Lower_BB",
                ]
            ],
            test_data[
                [
                    "Close",
                    "Volume",
                    "SMA_14",
                    "RSI_14",
                    "MACD",
                    "Signal_Line",
                    "Upper_BB",
                    "Lower_BB",
                ]
            ],
        ),
        axis=0,
    )
    ai_inputs = total_dataset[
        len(total_dataset) - len(test_data) - prediction_days :
    ].values
    ai_inputs = scaler.transform(ai_inputs)

    x_test = []
    for i in range(prediction_days, len(ai_inputs)):
        x_test.append(ai_inputs[i - prediction_days : i, :])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 8))

    # Check if x_test is empty
    if len(x_test) == 0:
        print(
            f"ERROR: Insufficient data to create test sequences. Only {len(ai_inputs)} data points available for testing, but {prediction_days} prediction days are required."
        )
        print(
            "Please ensure sufficient historical data is available for the test period or reduce 'prediction_days'."
        )
        return

    # predict
    # Use model.train() instead of eval() to keep dropout active for jagged (stochastic) predictions
    model.train()
    with torch.inference_mode():
        xt = torch.from_numpy(x_test).float().to(device, non_blocking=use_gpu)
        preds = model(xt).cpu().numpy()

    # Inverse transform predictions. Need to create a dummy array for the additional features (Volume, SMA_14, RSI_14).
    dummy_features_test = np.zeros_like(
        preds, shape=(preds.shape[0], 7)
    )  # 3 dummy features for Volume, SMA, RSI
    full_preds_scaled = np.concatenate((preds, dummy_features_test), axis=1)
    prediction_prices = scaler.inverse_transform(full_preds_scaled)[:, 0].reshape(-1, 1)

    # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = mean_squared_error(actual_prices, prediction_prices)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.2f}")

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual_prices, prediction_prices)
    print(f"Mean Absolute Error (MAE) on test data: {mae:.2f}")

    # Calculate Directional Accuracy
    actual_changes = np.diff(
        actual_prices.flatten()
    )  # Flatten actual_prices before diff

    predicted_prices_flat = prediction_prices.flatten()
    predicted_changes = np.diff(predicted_prices_flat)

    # Ensure both arrays are of the same length after diff
    min_len = min(len(actual_changes), len(predicted_changes))
    actual_changes = actual_changes[:min_len]
    predicted_changes = predicted_changes[:min_len]

    directional_accuracy = np.nan  # Default to NaN if calculation is skipped

    if len(actual_changes) > 0 and len(predicted_changes) > 0:
        # Check if arrays are indeed 1D before comparison
        if np.sign(actual_changes).ndim == 1 and np.sign(predicted_changes).ndim == 1:
            correct_directions = np.sum(
                np.sign(actual_changes) == np.sign(predicted_changes)
            )
            directional_accuracy = (correct_directions / len(actual_changes)) * 100
        else:
            print(
                "WARNING: np.sign resulted in non-1D arrays for directional accuracy calculation."
            )
            print(
                f"DEBUG: np.sign(actual_changes).ndim: {np.sign(actual_changes).ndim}"
            )
            print(
                f"DEBUG: np.sign(predicted_changes).ndim: {np.sign(predicted_changes).ndim}"
            )
    else:
        print(
            "WARNING: Insufficient data to calculate directional accuracy (actual_changes or predicted_changes is empty)."
        )

    print(f"Directional Accuracy on test data: {directional_accuracy:.2f}%")

    prediction_dates = test_data.index
    prediction_dates_offset = prediction_dates  # + pd.Timedelta(days=-1) # Removed offset to align with actual dates

    last_actual_value = float(np.asarray(actual_prices[-1]).flatten()[0])
    last_predicted_value = float(np.asarray(prediction_prices[-1]).flatten()[0])
    percentage_difference = float(
        (last_predicted_value - last_actual_value) / last_actual_value * 100
    )

    color_code = "\033[92m" if percentage_difference >= 0 else "\033[91m"
    reset_code = "\033[0m"
    print(f"Last actual value: {last_actual_value:.2f}")
    print(f"Last predicted value: {last_predicted_value:.2f}")
    print(
        "Percentage difference:",
        f"{color_code}{percentage_difference:.2f}%{reset_code}",
    )

    # Forecast next N days via rolling prediction with Monte Carlo Dropout
    print(f"\n{'=' * 60}")
    print(f"PREDICTING NEXT {future_day} DAYS WITH MONTE CARLO DROPOUT...")
    print(f"{'=' * 60}\n")

    # Get the last prediction_days of scaled data including all features
    real_data = ai_inputs[-prediction_days:, :].copy()
    future_predictions = []
    future_predictions_std = []  # Initialized future_predictions_std

    # Store the last actual scaled values for Volume, SMA_14, RSI_14 for use in future predictions
    # Assuming 'Close', 'Volume', 'SMA_14', 'RSI_14' order
    last_actual_scaled_volume = ai_inputs[-1, 1]
    last_actual_scaled_sma = ai_inputs[-1, 2]
    last_actual_scaled_rsi = ai_inputs[-1, 3]
    last_actual_scaled_macd = ai_inputs[-1, 4]
    last_actual_scaled_signal_line = ai_inputs[-1, 5]
    last_actual_scaled_upper_bb = ai_inputs[-1, 6]
    last_actual_scaled_lower_bb = ai_inputs[-1, 7]

    for day in range(future_day):
        monte_carlo_predictions_for_day = (
            []
        )  # Initialized monte_carlo_predictions_for_day
        input_seq = real_data[-prediction_days:].reshape(1, prediction_days, 8)
        t_in = torch.from_numpy(input_seq).float().to(device, non_blocking=use_gpu)

        # Enable dropout during inference for Monte Carlo
        model.train()  # Set model to training mode to enable dropout
        with torch.no_grad():  # Still no_grad for predictions, but dropout is active
            for _ in range(num_monte_carlo_runs):
                monte_carlo_predictions_for_day.append(model(t_in).squeeze())

        monte_carlo_predictions_for_day = (
            torch.stack(monte_carlo_predictions_for_day).detach().cpu().numpy()
        )

        # Set model back to evaluation mode after Monte Carlo runs
        model.eval()

        # Calculate mean and standard deviation of Monte Carlo predictions for STATS
        # next_pred = np.mean(monte_carlo_predictions_for_day) # Calculate mean (Smooth)

        # USE SINGLE REALIZATION FOR JAGGED TRAJECTORY
        # We take the first run as the "path" we follow
        next_pred = float(monte_carlo_predictions_for_day[0])  # single realization path

        future_predictions_std.append(
            np.std(monte_carlo_predictions_for_day)
        )  # Calculate std dev

        future_predictions.append(next_pred)
        # Append the predicted 'Close' price and the last known 'Volume', 'SMA_14', 'RSI_14'
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
            ]
        )
        real_data = np.vstack((real_data, new_row))
        # if (day + 1) % 10 == 0:
        print(f"Predicted day {day + 1}/{future_day}")

    model.eval()

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    # Inverse transform. Need to create dummy arrays for the additional features (Volume, SMA_14, RSI_14).
    dummy_features = np.zeros_like(
        future_predictions, shape=(future_predictions.shape[0], 7)
    )  # 3 dummy features
    full_future_predictions_scaled = np.concatenate(
        (future_predictions, dummy_features), axis=1
    )
    future_predictions_prices = scaler.inverse_transform(
        full_future_predictions_scaled
    )[:, 0].reshape(-1, 1)

    # Calculate confidence intervals
    # Get the scaling factor for the 'Close' price
    # This is (max_close - min_close) from the training data
    # Assuming 'Close' is the first feature (index 0) in the scaler
    close_min = scaler.data_min_[0]
    close_max = scaler.data_max_[0]
    scaling_factor_close = close_max - close_min

    # Unscale the standard deviations
    future_predictions_prices_std_unscaled = (
        np.array(future_predictions_std) * scaling_factor_close
    )

    confidence_level = 0.95
    # Calculate the Z-score for the desired confidence level (two-tailed)
    z_score = st.norm.ppf(1 - (1 - confidence_level) / 2)

    future_predictions_lower = (
        future_predictions_prices.flatten()
        - z_score * future_predictions_prices_std_unscaled
    )
    future_predictions_upper = (
        future_predictions_prices.flatten()
        + z_score * future_predictions_prices_std_unscaled
    )

    last_date = data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_day
    )

    print("\n" + "=" * 60)
    print("FUTURE PRICE PREDICTIONS:")
    print("=" * 60)
    for date, price in zip(future_dates, future_predictions_prices):
        print(f"{date.strftime('%Y-%m-%d')}: ${float(price[0]):.2f}")
    print("=" * 60 + "\n")

    # Print confidence interval info
    print(
        f"Mean Standard Deviation of Monte Carlo predictions (scaled): {np.mean(future_predictions_std):.4f}"
    )
    print(
        f"Approximate Average Margin of Error for 95% Confidence Interval (unscaled): {np.mean(z_score * future_predictions_prices_std_unscaled):.2f}\n"
    )

    # --------------------------------------------------------------current_price = float(data["Close"].values[-1])
    current_price = float(data["Close"].values[-1].item())
    final_predicted_price = float(future_predictions_prices[-1][0])
    projected_change = ((final_predicted_price - current_price) / current_price) * 100
    change_color = "\033[92m" if projected_change >= 0 else "\033[91m"
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price in {future_day} days: ${final_predicted_price:.2f}")
    print(f"Projected Change: {change_color}{projected_change:.2f}%{reset_code}\n")

    end = time.time()
    print(
        "Script concluded for a duration of {:.2f} seconds".format(
            end - script_start_time
        )
    )

    # Plotting
    # Prepare data for mplfinance
    # Flatten columns if MultiIndex (common in newer yfinance versions)
    plot_data = test_data.copy()
    if isinstance(plot_data.columns, pd.MultiIndex):
        plot_data.columns = plot_data.columns.get_level_values(0)

    # MPLFinance plotting
    # We plot on the existing 'graph' method. The graph object passed to mpf.plot means this will draw on top of it. mpf can draw it's own figures with `figscale` and `style` parameters. If you don't use a ax you should set it to true so the graph is displayed.k
    # Note: style='yahoo' gives standard green/red candles.
    # show_nontrading=True ensures the x-axis remains linear date-based, matching our other line plots.
    mpf.plot(
        plot_data,
        type="candle",
        style="yahoo",
        ax=graph,
        show_nontrading=True,
        datetime_format="%Y-%m-%d",
        ylabel=f"{chart_name_plot} Price",
    )

    # graph.plot(prediction_dates, actual_prices, color='white', label='Actual Prices', linewidth=2) # Replaced by candlesticks
    graph.plot(
        prediction_dates_offset,
        prediction_prices,
        color="cyan",
        label="Predicted Prices (Test Period)",
        linewidth=2,
    )  # Changed color to cyan for visibility in dark mode
    graph.plot(
        future_dates,
        future_predictions_prices,
        color="coral",
        label=f"Future Forecast ({future_day} days)",
        linewidth=2.5,
    )

    # Plot confidence intervals
    graph.fill_between(
        future_dates,
        future_predictions_lower,
        future_predictions_upper,
        color="purple",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    graph.axvline(
        x=last_date,
        color="orange",
        linestyle=":",
        linewidth=2,
        label="Current Date",
        alpha=0.7,
    )
    graph.set_title(
        f"{chart_name_plot} Price Prediction with {future_day}-Day Forecast (with Monte Carlo Dropout Confidence)"
    )
    graph.set_xlabel("Date")
    # graph.set_ylabel(f'{chart} Price') # Handled by mpf
    graph.legend(loc="upper left")
    graph.grid(True, alpha=0.3)
    # fig.autofmt_xdate() # mpf handles this

    # Layer 2: Residuals analysis underneath
    ax_res = plt.subplot(2, 1, 2)
    residuals = actual_prices.flatten() - prediction_prices.flatten()
    ax_res.scatter(prediction_dates, residuals, color="yellow", alpha=0.6, s=20)
    ax_res.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax_res.fill_between(
        prediction_dates,
        residuals.mean() - residuals.std(),
        residuals.mean() + residuals.std(),
        color="green",
        alpha=0.1,
    )
    ax_res.set_title("Prediction Residuals (Actual - Predicted)", fontweight="bold")
    ax_res.set_ylabel("Residual ($)")
    ax_res.set_xlabel("Date")
    ax_res.grid(True, alpha=0.3)

    # Interactive hover annotation showing nearest actual/predicted/future values
    annotation = graph.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="#333333", ec="white", alpha=0.9),
        fontsize=9,
    )
    annotation.set_visible(False)

    # Precompute numeric date arrays and flattened price arrays for fast nearest-point lookup
    actual_dnums = mdates.date2num(prediction_dates)
    pred_dnums = mdates.date2num(prediction_dates_offset)
    future_dnums = mdates.date2num(future_dates)

    # For hover, we still valid actual_vals from the data
    actual_vals = np.array(actual_prices).flatten()
    pred_vals = prediction_prices.flatten()
    future_vals = future_predictions_prices.flatten()

    def motion_hover(event):
        if event.inaxes != graph:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        x = event.xdata
        if x is None:
            return

        # find nearest points in each series
        idx_actual = np.argmin(np.abs(actual_dnums - x)) if len(actual_dnums) else None
        dist_actual = (
            abs(actual_dnums[idx_actual] - x) if idx_actual is not None else np.inf
        )

        idx_pred = np.argmin(np.abs(pred_dnums - x)) if len(pred_dnums) else None
        dist_pred = abs(pred_dnums[idx_pred] - x) if idx_pred is not None else np.inf

        idx_future = np.argmin(np.abs(future_dnums - x)) if len(future_dnums) else None
        dist_future = (
            abs(future_dnums[idx_future] - x) if idx_future is not None else np.inf
        )

        # choose nearest among the three series
        nearest = "none"
        if (
            idx_actual is not None
            and dist_actual <= dist_pred
            and dist_actual <= dist_future
        ):
            nearest = "actual"
        elif (
            idx_pred is not None
            and dist_pred <= dist_actual
            and dist_pred <= dist_future
        ):
            nearest = "pred"
        elif idx_future is not None:
            nearest = "future"
        else:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return

        if nearest == "actual" and idx_actual is not None:
            dnum = actual_dnums[idx_actual]
            date = mdates.num2date(dnum)
            actual = actual_vals[idx_actual]
            # find predicted nearest to this actual date (may be offset)
            pred_idx = np.argmin(np.abs(pred_dnums - dnum)) if len(pred_dnums) else None
            predicted = pred_vals[pred_idx] if pred_idx is not None else float("nan")
        elif nearest == "pred" and idx_pred is not None:
            dnum = pred_dnums[idx_pred]
            date = mdates.num2date(dnum)
            predicted = pred_vals[idx_pred]
            act_idx = (
                np.argmin(np.abs(actual_dnums - dnum))
                if len(actual_dnums)
                else float("nan")
            )
            actual = (
                actual_vals[act_idx]
                if not np.isnan(act_idx) and act_idx is not None
                else float("nan")
            )
        elif nearest == "future" and idx_future is not None:
            dnum = future_dnums[idx_future]
            date = mdates.num2date(dnum)
            predicted = future_vals[idx_future]
            actual = float("nan")
        else:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return

        actual_text = f"${actual:.2f}" if (not np.isnan(actual)) else "N/A"
        pred_text = f"${predicted:.2f}"
        text = f"{date.strftime('%Y-%m-%d')}\nActual: {actual_text}\nPredicted: {pred_text}"

        # position annotation near cursor
        annotation.xy = (event.xdata, event.ydata)
        annotation.set_text(text)
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", motion_hover)

    # save outputs
    out_dir = os.getcwd()
    present_day = dt.datetime.now().date()
    fig.savefig(
        os.path.join(
            out_dir, f"{optimizer_name}_{chart_name_plot_short}_{present_day}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted_Price for next day": future_predictions_prices.flatten(),
        }
    )
    forecast_df.to_csv(
        os.path.join(out_dir, "future_predictions_pytorch.csv"), index=False
    )

    # ----------------------- SECONDARY WINDOW: FORECAST DETAIL -----------------------
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
        future_predictions_prices.flatten(),
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
    ax_ci.set_xticklabels([f"Day {i + 1}" for i in range(len(ci_widths))], rotation=45)
    ax_ci.grid(True, alpha=0.3, axis="y")

    fig2.tight_layout()
    fig2.savefig(
        os.path.join(
            out_dir,
            f"{optimizer_name}_{chart_name_plot_short}_forecast_detail_{present_day}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    print("\nSaved:", [f for f in os.listdir(out_dir) if f.endswith((".png", ".csv"))])

    plt.show()


# ==============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           DATA DICTIONARY                                 ║
# ║                        pytorch_plotted.py v2.0                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ==============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CONFIGURATION PARAMETERS                                                    │
# ├──────────────────────────┬─────────────┬────────────────────────────────────┤
# │ Variable                 │ Type        │ Description                        │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ chart                    │ str         │ Ticker symbol entered by user      │
# │                          │             │ (e.g. "AAPL", "TSLA", "BTC-USD")   │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ chart_info               │ dict        │ yf.Ticker(chart).info metadata     │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ chart_name_plot          │ str         │ Full company name for plot titles  │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ chart_name_plot_short    │ str         │ Short name for output filenames    │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ prediction_days          │ int = 30    │ LSTM lookback window length        │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ future_day               │ int = 30    │ Days to forecast ahead             │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ epochs                   │ int = 40    │ Training epochs                    │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ batch_size               │ int = 32    │ Batch size for DataLoader          │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ initial_dropout          │ float = 0.6 │ Starting dropout rate (epoch 0)    │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ final_dropout            │ float = 0.1 │ Final dropout rate (epoch N)       │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ train_time               │ int = 2     │ Reserved (unused)                  │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ num_monte_carlo_runs     │ int = 100   │ MC dropout passes per forecast day │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ device_type              │ str         │ "cpu" or "gpu" (user input)        │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ device                   │ torch.device│ Resolved compute device            │
# ├──────────────────────────┼─────────────┼────────────────────────────────────┤
# │ confidence_level         │ float = 0.95│ CI confidence (default 95%)        │
# └──────────────────────────┴─────────────┴────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MODEL COMPONENTS                                                            │
# ├──────────────────────────┼──────────────────────────────────────────────────┤
# │ LSTMModel                │ Bidirectional LSTM stack                         │
# │                          │ • num_layers × (LSTM + Dropout) pairs            │
# │                          │ • hidden_size=500, bidirectional=True            │
# │                          │ • Final Linear(hidden_size → 1)                  │
# ├──────────────────────────┼──────────────────────────────────────────────────┤
# │ SequenceDataset          │ PyTorch Dataset wrapper for (x_train, y_train)   │
# ├──────────────────────────┼──────────────────────────────────────────────────┤
# │ model                    │ LSTMModel instance on `device`                   │
# ├──────────────────────────┼──────────────────────────────────────────────────┤
# │ criterion                │ nn.MSELoss (Mean Squared Error)                  │
# ├──────────────────────────┼──────────────────────────────────────────────────┤
# │ optimizer                │ torch_optimizer.Lamb (weight_decay=0.05)         │
# └──────────────────────────┴──────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ RAW DATA                                                                    │
# ├──────────────────────────┬──────────────────────────────────────────────────┤
# │ data                     │ pd.DataFrame - Full OHLCV history                │
# │ test_data                │ pd.DataFrame - Evaluation window (2025-06-01→)   │
# │ total_dataset            │ pd.DataFrame - Concatenation of data+test        │
# │ actual_prices            │ np.ndarray - Ground truth Close prices           │
# └──────────────────────────┴──────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ FEATURE ENGINEERING                                                         │
# ├──────────────────────────┬──────────────────────────────────────────────────┤
# │ SMA_14                   │ 14-day Simple Moving Average                     │
# │ RSI_14                   │ 14-day Relative Strength Index (0-100)           │
# │ MACD                     │ 12-day EMA − 26-day EMA                          │
# │ Signal_Line              │ 9-day EMA of MACD                                │
# │ Upper_BB                 │ Upper Bollinger Band (20-SMA + 2σ)               │
# │ Lower_BB                 │ Lower Bollinger Band (20-SMA − 2σ)               │
# └──────────────────────────┴──────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ SCALING & SEQUENCES                                                         │
# ├──────────────────────────┬──────────────────────────────────────────────────┤
# │ scaler                   │ MinMaxScaler(feature_range=(0,1))                │
# │ scaled_data              │ np.ndarray (N, 8) - All scaled features          │
# │ training_data            │ np.ndarray - scaled_data[:-prediction_days]      │
# │ ai_inputs                │ np.ndarray - Test context for x_test             │
# │ x_train                  │ np.ndarray (N, prediction_days, 8)               │
# │ y_train                  │ np.ndarray (N,) - Next-day Close targets         │
# │ x_test                   │ np.ndarray (len(test_data), prediction_days, 8)  │
# │ dataset                  │ SequenceDataset instance                         │
# │ dataloader               │ DataLoader (shuffle=True)                        │
# └──────────────────────────┴──────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ TRAINING METRICS                                                            │
# ├──────────────────────────┬──────────────────────────────────────────────────┤
# │ epoch_loss               │ MSE loss per epoch                               │
# │ new_p                    │ Dynamic dropout rate for current epoch           │
# │ writer                   │ TensorBoard SummaryWriter (or None)              │
# │ log_dir                  │ "logs/fit/YYYYMMDD-HHMMSS" path                  │
# │ _TB_BACKEND              │ "torch" | "tensorboardX" | None                  │
# └──────────────────────────┴──────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ EVALUATION METRICS                                                          │
# ├──────────────────────────┬──────────────────────────────────────────────────┤
# │ preds                    │ Raw LSTM output (scaled [0,1])                   │
# │ prediction_prices        │ Inverse-transformed predictions (raw $)          │
# │ mse                      │ Mean Squared Error                               │
# │ rmse                     │ Root MSE - primary accuracy metric               │
# │ mae                      │ Mean Absolute Error                              │
# │ directional_accuracy     │ % of correct price direction predictions         │
# └──────────────────────────┴──────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MONTE CARLO FORECAST                                                        │
# ├──────────────────────────┬──────────────────────────────────────────────────┤
# │ real_data                │ Rolling buffer (grows each forecast day)         │
# │ monte_carlo_predictions  │ List of num_monte_carlo_runs outputs             │
# │ future_predictions       │ Scaled Close predictions (future_day,)           │
# │ future_predictions_std   │ Std dev across MC runs per day                   │
# │ future_predictions_prices│ Raw $ predictions (inverse transformed)          │
# │ close_min / close_max    │ Training data min/max Close prices               │
# │ scaling_factor_close     │ close_max - close_min                            │
# │ z_score                  │ ~1.96 for 95% CI                                 │
# │ future_predictions_lower │ Lower CI bound per day                           │
# │ future_predictions_upper │ Upper CI bound per day                           │
# │ future_dates             │ DatetimeIndex for forecast dates                 │
# └──────────────────────────┴──────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PLOTTING OBJECTS                                                            │
# ├──────────────────────────┬──────────────────────────────────────────────────┤
# │ fig                      │ Primary Figure (20×14)                           │
# │ fig2                     │ Secondary Figure (16×10)                         │
# │ graph                    │ Axes - Candlestick + predictions overlay         │
# │ ax_res                   │ Axes - Residuals scatter plot                    │
# │ ax_price                 │ Axes - Forecast line with CI band                │
# │ ax_ci                    │ Axes - CI width bar chart                        │
# │ annotation               │ Hover tooltip (matplotlib Annotation)            │
# │ actual_dnums             │ Date numbers for hover lookup                    │
# │ pred_dnums               │ Date numbers for hover lookup                    │
# │ future_dnums             │ Date numbers for hover lookup                    │
# │ forecast_df              │ pd.DataFrame → future_predictions_pytorch.csv    │
# └──────────────────────────┴──────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ UTILITY FUNCTIONS                                                           │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ TeeLogger                │ Dual-output logger (terminal + file)             │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ get_dynamic_dropout()    │ Linear annealing schedule                        │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ set_dropout()            │ In-place dropout rate setter                     │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ build_sequences()        │ Sliding window sequence builder                  │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ choose_chart_interactive() │ CLI ticker input with search                   │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ finder()                 │ yf.Search wrapper                                │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌───────────────────────────────────────────────────────────────────────────────────┐
# │ TORCH_OPTIMIZER REFERENCE (Strengths, Weaknesses & Speed)                         │
# ├──────────────┬──────────┬────────────────────────────┼────────────────────────────┤
# │ Optimizer    │ Speed    │ Strengths                  │ Weaknesses                 │
# ├──────────────┼──────────┼────────────────────────────┼────────────────────────────┤
# │ AccSGD       │ Fast     │ Accelerated, good for CNN  │ Can be unstable            │ # - FAILED
# │ AdaBelief    │ Fast     │ Fast conv, good generaliz. │ Epsilon sensitive          │ # - SUCCESS
# │ AdaBound     │ Med      │ Adam to SGD transition     │ Hyperparam sensitive       │ # - FAILED
# │ Adafactor    │ Med      │ Low memory footprint       │ Slower to converge         │ # - SUCCESS
# │ Adahessian   │ Slow     │ Superb 2nd order conv.     │ Compute & memory heavy     │ # - PASSED
# │ AdaMod       │ Med      │ Protects from sudden jumps │ Extra hyperparameters      │ # - SUCCESS
# │ AdamP        │ Fast     │ Prevents weight norm grow  │ Context-dependent          │
# │ AggMo        │ Med      │ Dampens oscillations       │ Higher memory usage        │ # - FAILED
# │ Apollo       │ Med      │ Robust to ill-conditioned  │ Less practically tested    │
# │ DiffGrad     │ Fast     │ Friction-based LR change   │ Slower initial steps       │
# │ Lamb         │ V.Fast   │ Great for large batch size │ Diverges on small batch    │ # - SUCCESS
# │ LARS         │ V.Fast   │ Enables huge batch sizes   │ Needs careful warmup       │ # - FAILED
# │ MADGRAD      │ Fast     │ SGD generaliz + Adam speed │ Needs extra float buffer   │
# │ NovoGrad     │ Med      │ Good for large NLP models  │ Tricky for vision tasks    │
# │ PID          │ Med      │ Helps dampen oscillations  │ Needs P,I,D tuning         │
# │ QHAdam       │ Fast     │ Flexible (Quasi-Hyper)     │ Hard to tune (v1, v2)      │
# │ QHM          │ Med      │ Bridges SGD and SGDM       │ Needs careful tuning       │
# │ RAdam        │ Fast     │ Automated warmup, robust   │ Slightly slow initially    │
# │ Ranger       │ Fast     │ RAdam + LookAhead (SOTA)   │ Uses more compute/mem      │ # - SUCCESS
# │ RangerQH     │ Fast     │ QHAdam + LookAhead         │ Complex hyperparams        │
# │ RangerVA     │ Fast     │ Ranger + Variance Adaption │ Complex implementation     │
# │ SGDP         │ Fast     │ Controls weight growth     │ Needs weight decay         │
# │ SGDW         │ Fast     │ Decouples weight decay     │ Still SGD-like speed       │
# │ Shampoo      │ V.Slow   │ Extraordinary step conv.   │ Extreme matrix compute     │ # - FAILED
# │ SWATS        │ Fast     │ Adam to SGD automated      │ Switch heuristic varies    │
# │ Yogi         │ Fast     │ Controls LR increase       │ Slow initial phase         │
# └──────────────┴──────────┴────────────────────────────┴────────────────────────────┘
# took me like an hour to format </3
# ==============================================================================

if __name__ == "__main__":
    main()
