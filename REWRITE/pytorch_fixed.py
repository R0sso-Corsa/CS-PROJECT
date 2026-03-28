import time
import sys

script_start_time = time.time()

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as st
import math


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
    try:
        return yf.Search(query)
    except Exception:
        return None


def choose_chart_interactive():
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


chart = choose_chart_interactive()
chart_info = yf.Ticker(chart).info
prediction_days = 30
future_day = 30
epochs = 150
batch_size = 32
initial_dropout = 0.3
final_dropout = 0.1
val_split = 0.15
num_monte_carlo_runs = 100

device_type = input("Enter device type (cpu/gpu): ").strip().lower()
device = torch.device(
    "cuda" if (device_type == "gpu" and torch.cuda.is_available()) else "cpu"
)
print(f"Using device: {device}")

chart_name_plot = chart_info.get("longName") or chart
chart_name_plot_short = chart_info.get("shortName") or chart

sys.stdout = TeeLogger("terminal_activity.log")
sys.stderr = sys.stdout


def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.3, final_rate=0.1):
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


class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = input_size
        for i in range(num_layers):
            self.layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size // 2,
                    bidirectional=True,
                    batch_first=True,
                )
            )
            self.layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = x
        for i in range(0, len(self.layers), 2):
            lstm = self.layers[i]
            dropout = self.layers[i + 1]
            out, _ = lstm(out)
            out = dropout(out)
        out = out[:, -1, :]
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
        y.append(scaled_values[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


def compute_technical_indicators(df):
    df["SMA_14"] = df["Close"].rolling(window=14).mean()
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["20_SMA"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["Upper_BB"] = df["20_SMA"] + (df["Std_Dev"] * 2)
    df["Lower_BB"] = df["20_SMA"] - (df["Std_Dev"] * 2)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


FEATURES = [
    "Close",
    "Volume",
    "SMA_14",
    "RSI_14",
    "MACD",
    "Signal_Line",
    "Upper_BB",
    "Lower_BB",
]


def main():
    print(f"\n{'=' * 60}")
    print(f"ANALYZING: {chart_name_plot} ({chart})")
    print(f"SHORT NAME: {chart_name_plot_short}")
    print(f"DEVICE: {device}")
    print(f"{'=' * 60}\n")

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"{chart_name_plot} - Detailed Prediction Analysis",
        fontsize=16,
        fontweight="bold",
    )
    graph = fig.add_subplot(2, 1, 1)

    ticker = yf.Ticker(chart)
    hist_max = ticker.history(period="max")
    if (hist_max is not None) and (not hist_max.empty):
        start = hist_max.index[0].to_pydatetime()
    else:
        start = dt.datetime(2017, 1, 1)

    end = dt.datetime.now()
    data = yf.download(chart, start=start, end=end, auto_adjust=True)

    if data.empty:
        print(
            f"ERROR: No historical data found for {chart}. Please try a different ticker."
        )
        return

    print(data.head())
    data = compute_technical_indicators(data)

    train_end_idx = len(data) - prediction_days
    if train_end_idx < prediction_days * 2:
        print(
            f"ERROR: Insufficient data. Need at least {prediction_days * 2} rows for training."
        )
        return

    train_data = data.iloc[:train_end_idx].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data[FEATURES].values)

    scaled_train = scaler.transform(train_data[FEATURES].values)
    x_train_full, y_train_full = build_sequences(scaled_train, prediction_days)

    val_size = int(len(x_train_full) * val_split)
    train_size = len(x_train_full) - val_size

    x_train = x_train_full[:train_size]
    y_train = y_train_full[:train_size]
    x_val = x_train_full[train_size:]
    y_val = y_train_full[train_size:]

    print(f"\nData split: {train_size} train samples, {val_size} validation samples")

    train_dataset = SequenceDataset(x_train, y_train)
    val_dataset = SequenceDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_size=8, hidden_size=256, num_layers=2, dropout=initial_dropout
    ).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

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

    if writer is not None:
        try:
            sample_input = torch.zeros((1, prediction_days, 8), device=device)
            writer.add_graph(model, sample_input)
        except Exception:
            pass

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 30

    for epoch in trange(
        epochs, desc="Epochs", unit="epoch", colour="blue", ascii=False
    ):
        model.train()
        epoch_loss = 0.0
        new_p = get_dynamic_dropout(epoch, epochs, initial_dropout, final_dropout)
        set_dropout(model, new_p)

        batch_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            unit="batch",
            colour="blue",
            ascii=False,
        )
        for xb, yb in batch_iter:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
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

        epoch_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        if writer is not None:
            try:
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            except Exception:
                pass

        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} — Train Loss: {epoch_loss:.6f} — Val Loss: {val_loss:.6f} — LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                model.load_state_dict(torch.load("best_model.pt"))
                break

    if writer is not None:
        writer.close()

    use_earliest_test = False
    if use_earliest_test:
        test_start = start
    else:
        test_start = dt.datetime(2025, 6, 1)
    test_end = dt.datetime.now()
    test_data = yf.download(chart, test_start, test_end, auto_adjust=True)

    if test_data.empty:
        print(f"ERROR: No historical data found for {chart} in the test period.")
        return

    test_data = compute_technical_indicators(test_data)
    actual_prices = test_data["Close"].values

    total_dataset = pd.concat([data[FEATURES], test_data[FEATURES]], axis=0)

    ai_inputs = total_dataset[
        len(total_dataset) - len(test_data) - prediction_days :
    ].values
    ai_inputs = scaler.transform(ai_inputs)

    x_test = []
    for i in range(prediction_days, len(ai_inputs)):
        x_test.append(ai_inputs[i - prediction_days : i, :])
    x_test = np.array(x_test)

    if len(x_test) == 0:
        print(f"ERROR: Insufficient data to create test sequences.")
        return

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x_test).float().to(device)
        preds = model(xt).cpu().numpy()

    dummy_features_test = np.zeros_like(preds, shape=(preds.shape[0], 7))
    full_preds_scaled = np.concatenate((preds, dummy_features_test), axis=1)
    prediction_prices = scaler.inverse_transform(full_preds_scaled)[:, 0].reshape(-1, 1)

    mse = mean_squared_error(actual_prices, prediction_prices)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual_prices, prediction_prices)
    print(f"\nRoot Mean Squared Error (RMSE) on test data: {rmse:.2f}")
    print(f"Mean Absolute Error (MAE) on test data: {mae:.2f}")

    actual_changes = np.diff(actual_prices.flatten())
    predicted_prices_flat = prediction_prices.flatten()
    predicted_changes = np.diff(predicted_prices_flat)
    min_len = min(len(actual_changes), len(predicted_changes))
    actual_changes = actual_changes[:min_len]
    predicted_changes = predicted_changes[:min_len]

    if len(actual_changes) > 0:
        correct_directions = np.sum(
            np.sign(actual_changes) == np.sign(predicted_changes)
        )
        directional_accuracy = (correct_directions / len(actual_changes)) * 100
    else:
        directional_accuracy = 0.0

    print(f"Directional Accuracy on test data: {directional_accuracy:.2f}%")

    prediction_dates = test_data.index

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
        f"Percentage difference: {color_code}{percentage_difference:.2f}%{reset_code}"
    )

    print(f"\n{'=' * 60}")
    print(f"PREDICTING NEXT {future_day} DAYS WITH MONTE CARLO DROPOUT...")
    print(f"{'=' * 60}\n")

    real_data = ai_inputs[-prediction_days:, :].copy()
    future_predictions = []
    future_predictions_std = []

    last_actual_scaled_volume = ai_inputs[-1, 1]
    last_actual_scaled_sma = ai_inputs[-1, 2]
    last_actual_scaled_rsi = ai_inputs[-1, 3]
    last_actual_scaled_macd = ai_inputs[-1, 4]
    last_actual_scaled_signal_line = ai_inputs[-1, 5]
    last_actual_scaled_upper_bb = ai_inputs[-1, 6]
    last_actual_scaled_lower_bb = ai_inputs[-1, 7]

    for day in range(future_day):
        monte_carlo_predictions_for_day = []
        input_seq = real_data[-prediction_days:].reshape(1, prediction_days, 8)
        t_in = torch.from_numpy(input_seq).float().to(device)

        model.train()
        for _ in range(num_monte_carlo_runs):
            with torch.no_grad():
                monte_carlo_predictions_for_day.append(model(t_in).cpu().numpy()[0, 0])

        model.eval()

        next_pred = np.mean(monte_carlo_predictions_for_day)

        future_predictions_std.append(np.std(monte_carlo_predictions_for_day))
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
            ]
        )
        real_data = np.vstack((real_data, new_row))
        if (day + 1) % 10 == 0:
            print(f"Predicted day {day + 1}/{future_day}")

    model.eval()

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    dummy_features = np.zeros_like(
        future_predictions, shape=(future_predictions.shape[0], 7)
    )
    full_future_predictions_scaled = np.concatenate(
        (future_predictions, dummy_features), axis=1
    )
    future_predictions_prices = scaler.inverse_transform(
        full_future_predictions_scaled
    )[:, 0].reshape(-1, 1)

    close_min = scaler.data_min_[0]
    close_max = scaler.data_max_[0]
    scaling_factor_close = close_max - close_min

    future_predictions_prices_std_unscaled = (
        np.array(future_predictions_std) * scaling_factor_close
    )

    confidence_level = 0.95
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

    print(
        f"Mean Standard Deviation of Monte Carlo predictions (scaled): {np.mean(future_predictions_std):.4f}"
    )
    print(
        f"Approximate Average Margin of Error for 95% Confidence Interval (unscaled): {np.mean(z_score * future_predictions_prices_std_unscaled):.2f}\n"
    )

    current_price = float(data["Close"].values[-1].item())
    final_predicted_price = float(future_predictions_prices[-1][0])
    projected_change = ((final_predicted_price - current_price) / current_price) * 100
    change_color = "\033[92m" if projected_change >= 0 else "\033[91m"
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price in {future_day} days: ${final_predicted_price:.2f}")
    print(f"Projected Change: {change_color}{projected_change:.2f}%{reset_code}\n")

    end = time.time()
    print(f"Script concluded for a duration of {end - script_start_time:.2f} seconds")

    plot_data = test_data.copy()
    if isinstance(plot_data.columns, pd.MultiIndex):
        plot_data.columns = plot_data.columns.get_level_values(0)

    mpf.plot(
        plot_data,
        type="candle",
        style="yahoo",
        ax=graph,
        show_nontrading=True,
        datetime_format="%Y-%m-%d",
        ylabel=f"{chart_name_plot} Price",
    )

    graph.plot(
        prediction_dates,
        prediction_prices,
        color="cyan",
        label="Predicted Prices (Test Period)",
        linewidth=2,
    )
    graph.plot(
        future_dates,
        future_predictions_prices,
        color="coral",
        label=f"Future Forecast ({future_day} days)",
        linewidth=2.5,
    )
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
    graph.legend(loc="upper left")
    graph.grid(True, alpha=0.3)

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

    annotation = graph.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="#333333", ec="white", alpha=0.9),
        fontsize=9,
    )
    annotation.set_visible(False)

    actual_dnums = mdates.date2num(prediction_dates)
    actual_vals = np.array(actual_prices).flatten()
    pred_vals = prediction_prices.flatten()
    future_dnums = mdates.date2num(future_dates)
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
        idx_actual = np.argmin(np.abs(actual_dnums - x)) if len(actual_dnums) else None
        dist_actual = (
            abs(actual_dnums[idx_actual] - x) if idx_actual is not None else np.inf
        )
        idx_future = np.argmin(np.abs(future_dnums - x)) if len(future_dnums) else None
        dist_future = (
            abs(future_dnums[idx_future] - x) if idx_future is not None else np.inf
        )

        nearest = "none"
        if idx_actual is not None and dist_actual <= dist_future:
            nearest = "actual"
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
            pred_idx = (
                np.argmin(np.abs(future_dnums - dnum)) if len(future_dnums) else None
            )
            predicted = future_vals[pred_idx] if pred_idx is not None else float("nan")
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
        annotation.xy = (event.xdata, event.ydata)
        annotation.set_text(text)
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", motion_hover)

    out_dir = os.getcwd()
    present_day = dt.datetime.now().date()
    fig.savefig(
        os.path.join(out_dir, f"{chart_name_plot_short}_{present_day}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    forecast_df = pd.DataFrame(
        {"Date": future_dates, "Predicted_Price": future_predictions_prices.flatten()}
    )
    forecast_df.to_csv(
        os.path.join(out_dir, "future_predictions_pytorch.csv"), index=False
    )

    fig2, (ax_price, ax_ci) = plt.subplots(2, 1, figsize=(16, 10))
    fig2.suptitle(
        f"{chart_name_plot} - {future_day}-Day Forecast with Uncertainty",
        fontsize=14,
        fontweight="bold",
    )

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
            out_dir, f"{chart_name_plot_short}_forecast_detail_{present_day}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    print("\nSaved:", [f for f in os.listdir(out_dir) if f.endswith((".png", ".csv"))])

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("\n" + "=" * 60)
        print("ERROR ENCOUNTERED:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        input("\nPress Enter to exit...")
