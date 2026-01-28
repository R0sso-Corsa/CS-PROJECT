import time
script_start_time = time.time()

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import yfinance as yf

import torch
import torch.nn as nn
# progress bars
from tqdm.auto import trange, tqdm
# Disable MIOpen (cudnn) to avoid RuntimeError: miopenStatusUnknownError with LSTM on ROCm
torch.backends.cudnn.enabled = False
from torch.utils.data import Dataset, DataLoader
# TensorBoard writer: prefer torch's bundled SummaryWriter, fall back to tensorboardX, else disable
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_BACKEND = 'torch'
except Exception:
    try:
        from tensorboardX import SummaryWriter  # type: ignore
        _TB_BACKEND = 'tensorboardX'
    except Exception:
        SummaryWriter = None
        _TB_BACKEND = None
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error # Added for RMSE calculation
import scipy.stats as st # Added for confidence interval calculation
import math # Added for sqrt in RMSE calculation


# ----------------------- Configuration -----------------------
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
        if choice.lower() == 's':
            query = input("Search query (company name or ticker fragment): ").strip()
            if not query:
                print("Empty query; try again.")
                continue
            results = finder(query)
            if results and getattr(results, 'quotes', None):
                print(f"Search results for '{query}':")
                for i, q in enumerate(results.quotes):
                    print(f"{i+1}. {q.get('symbol')} - {q.get('shortname')} ({q.get('quoteType')})")
                sel = input("Enter number to select a symbol, or press Enter to cancel: ").strip()
                if sel.isdigit():
                    idx = int(sel) - 1
                    if 0 <= idx < len(results.quotes):
                        symbol = results.quotes[idx].get('symbol')
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


# ask user for chart symbol (interactive search available)
chart = choose_chart_interactive()
prediction_days = 60
future_day = 30
epochs = 100
batch_size = 1000
initial_dropout = 0.5
final_dropout = 0.1
train_time = 2

device_type = input("Enter device type (cpu/cuda): ").strip().lower()

device = torch.device('cuda' if (device_type == 'cuda' and torch.cuda.is_available()) else 'cpu')
print(f"Using device: {device}")

# ----------------------- End Configuration -----------------------

def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
    return max(final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs))


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]).float(), torch.from_numpy(np.array(self.targets[idx])).float()


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=500, num_layers=4, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = input_size
        for i in range(num_layers):
            # return_sequences equivalent: we set batch_first=True and take full output for intermediate layers
            self.layers.append(nn.LSTM(input_size=in_size, hidden_size=hidden_size, batch_first=True))
            self.layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size

        # final linear to produce single value
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out = x
        for i in range(0, len(self.layers), 2):
            lstm = self.layers[i]
            dropout = self.layers[i+1]
            out, _ = lstm(out)  # out shape: (batch, seq_len, hidden_size)
            out = dropout(out)

        # take last timestep
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
        x.append(scaled_values[i-prediction_days:i, 0])
        y.append(scaled_values[i, 0])
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y


def main():
    # figure for plotting
    fig = plt.figure(figsize=(18, 9))
    graph = fig.add_subplot(1, 1, 1)

    # determine earliest date available
    ticker = yf.Ticker(chart)
    hist_max = ticker.history(period='max')
    if (hist_max is not None) and (not hist_max.empty):
        start = hist_max.index[0].to_pydatetime()
    else:
        start = dt.datetime(2017, 1, 1)

    end = dt.datetime.now()
    data = yf.download(chart, start=start, end=end)
    print(data.head())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # prepare training data (exclude last prediction_days to avoid incomplete windows)
    training_data = scaled_data[:-prediction_days]
    x_train, y_train = build_sequences(training_data, prediction_days)

    dataset = SequenceDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size if batch_size < len(dataset) else len(dataset), shuffle=True)

    model = LSTMModel(input_size=1, hidden_size=500, num_layers=4, dropout=initial_dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # tensorboard writer (create only if available)
    log_dir = os.path.join('logs', 'fit', dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard ({_TB_BACKEND}) logs: {log_dir}")
        except Exception as e:
            print(f"Warning: SummaryWriter failed to initialize: {e}")
            writer = None
    else:
        writer = None
        print('TensorBoard not available; continuing without it.')

    # try to log the model graph (best-effort)
    if writer is not None:
        try:
            sample_input = torch.zeros((1, prediction_days, 1), device=device)
            writer.add_graph(model, sample_input)
        except Exception:
            # some models / environments don't support add_graph; ignore
            pass

    # training loop with dynamic dropout
    # training with progress bars (trange for epochs, tqdm for batch progress)
    for epoch in trange(epochs, desc='Epochs', unit='epoch'):
        model.train()
        epoch_loss = 0.0
        new_p = get_dynamic_dropout(epoch, epochs, initial_dropout, final_dropout)
        set_dropout(model, new_p)

        batch_iter = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False, unit='batch')
        for xb, yb in batch_iter:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

            # update batch progress with current batch loss
            batch_iter.set_postfix({'batch_loss': f'{loss.item():.6f}', 'dropout': f'{new_p:.3f}'})

        epoch_loss /= len(dataset)
        if writer is not None:
            try:
                writer.add_scalar('Loss/train', epoch_loss, epoch, walltime=time.time())
            except Exception:
                pass

        # update epoch-level progress description
        tqdm.write(f"Epoch {epoch+1}/{epochs} \u2014 Loss: {epoch_loss:.6f} \u2014 Dropout: {new_p:.3f}")

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
        test_start = dt.datetime(2024, 1, 1)
    test_end = dt.datetime.now()
    test_data = yf.download(chart, test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    ai_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    ai_inputs = ai_inputs.reshape(-1, 1)
    ai_inputs = scaler.transform(ai_inputs)

    x_test = []
    for i in range(prediction_days, len(ai_inputs)):
        x_test.append(ai_inputs[i-prediction_days:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # predict
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x_test).float().to(device)
        preds = model(xt).cpu().numpy()

    prediction_prices = scaler.inverse_transform(preds)

    # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = mean_squared_error(actual_prices, prediction_prices)
    rmse = math.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.2f}")

    prediction_dates = test_data.index
    prediction_dates_offset = prediction_dates + pd.Timedelta(days=-1)

    last_actual_value = float(np.asarray(actual_prices[-1]).flatten()[0])
    last_predicted_value = float(np.asarray(prediction_prices[-1]).flatten()[0])
    percentage_difference = float((last_predicted_value - last_actual_value) / last_actual_value * 100)

    color_code = "\033[92m" if percentage_difference >= 0 else "\033[91m"
    reset_code = "\033[0m"
    print(f"Last actual value: {last_actual_value:.2f}")
    print(f"Last predicted value: {last_predicted_value:.2f}")
    print("Percentage difference:", f"{color_code}{percentage_difference:.2f}%{reset_code}")

    # Forecast next N days via rolling prediction with Monte Carlo Dropout
    print(f"\n{'='*60}")
    print(f"PREDICTING NEXT {future_day} DAYS WITH MONTE CARLO DROPOUT...")
    print(f"{'='*60}\n")

    num_monte_carlo_runs = 10 # Number of forward passes for Monte Carlo Dropout

    real_data = ai_inputs[-prediction_days:, 0].copy()
    future_predictions = []
    future_predictions_std = [] # Initialized future_predictions_std

    for day in range(future_day):
        monte_carlo_predictions_for_day = [] # Initialized monte_carlo_predictions_for_day
        input_seq = real_data[-prediction_days:].reshape(1, prediction_days, 1)
        t_in = torch.from_numpy(input_seq).float().to(device)

        # Enable dropout during inference for Monte Carlo
        model.train() # Set model to training mode to enable dropout
        for _ in range(num_monte_carlo_runs):
            with torch.no_grad(): # Still no_grad for predictions, but dropout is active
                monte_carlo_predictions_for_day.append(model(t_in).cpu().numpy()[0, 0])

        # Set model back to evaluation mode after Monte Carlo runs
        model.eval()

        # Calculate mean and standard deviation of Monte Carlo predictions
        next_pred = np.mean(monte_carlo_predictions_for_day) # Calculate mean
        future_predictions_std.append(np.std(monte_carlo_predictions_for_day)) # Calculate std dev

        future_predictions.append(next_pred)
        real_data = np.append(real_data, next_pred)
        if (day + 1) % 10 == 0:
            print(f"Predicted day {day+1}/{future_day}")

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_prices = scaler.inverse_transform(future_predictions)

    # Calculate confidence intervals
    # Get the scaling factor for the 'Close' price
    # This is (max_close - min_close) from the training data
    # Assuming 'Close' is the only feature (index 0) in the scaler
    close_min = scaler.data_min_[0]
    close_max = scaler.data_max_[0]
    scaling_factor_close = close_max - close_min

    # Unscale the standard deviations
    future_predictions_prices_std_unscaled = np.array(future_predictions_std) * scaling_factor_close

    confidence_level = 0.95
    # Calculate the Z-score for the desired confidence level (two-tailed)
    z_score = st.norm.ppf(1 - (1 - confidence_level) / 2)

    future_predictions_lower = future_predictions_prices.flatten() - z_score * future_predictions_prices_std_unscaled
    future_predictions_upper = future_predictions_prices.flatten() + z_score * future_predictions_prices_std_unscaled


    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_day)

    print('\n' + '='*60)
    print('FUTURE PRICE PREDICTIONS:')
    print('='*60)
    for date, price in zip(future_dates, future_predictions_prices):
        print(f"{date.strftime('%Y-%m-%d')}: ${float(price[0]):.2f}")
    print('='*60 + '\n')

    # Print confidence interval info
    print(f"Mean Standard Deviation of Monte Carlo predictions (scaled): {np.mean(future_predictions_std):.4f}")
    print(f"Approximate Average Margin of Error for 95% Confidence Interval (unscaled): {np.mean(z_score * future_predictions_prices_std_unscaled):.2f}\n")


    current_price = float(data['Close'].values[-1])
    final_predicted_price = float(future_predictions_prices[-1][0])
    projected_change = ((final_predicted_price - current_price) / current_price) * 100
    change_color = "\033[92m" if projected_change >= 0 else "\033[91m"
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price in {future_day} days: ${final_predicted_price:.2f}")
    print(f"Projected Change: {change_color}{projected_change:.2f}%{reset_code}\n")

    end = time.time()
    print("Script concluded for a duration of {:.2f} seconds".format(end - script_start_time))

    # Plotting
    graph.plot(prediction_dates, actual_prices, color='black', label='Actual Prices', linewidth=2)
    graph.plot(prediction_dates_offset, prediction_prices, color='green', label='Predicted Prices (Test Period)', linewidth=2)
    graph.plot(future_dates, future_predictions_prices, color='red', label=f'Future Forecast ({future_day} days)', linewidth=2.5, linestyle='--', marker='o', markersize=4)

    # Plot confidence intervals
    graph.fill_between(future_dates, future_predictions_lower, future_predictions_upper, color='purple', alpha=0.2, label='95% Confidence Interval')

    graph.axvline(x=last_date, color='orange', linestyle=':', linewidth=2, label='Current Date', alpha=0.7)
    graph.set_title(f'{chart} Price Prediction with {future_day}-Day Forecast (with Monte Carlo Dropout Confidence)')
    graph.set_xlabel('Date')
    graph.set_ylabel(f'{chart} Price')
    graph.legend(loc='upper left')
    graph.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    # Interactive hover annotation showing nearest actual/predicted/future values
    annotation = graph.annotate(
        '', xy=(0, 0), xytext=(15, 15), textcoords='offset points',
        bbox=dict(boxstyle='round', fc='w', alpha=0.9), fontsize=9
    )
    annotation.set_visible(False)

    # Precompute numeric date arrays and flattened price arrays for fast nearest-point lookup
    actual_dnums = mdates.date2num(prediction_dates)
    pred_dnums = mdates.date2num(prediction_dates_offset)
    future_dnums = mdates.date2num(future_dates)

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
        dist_actual = abs(actual_dnums[idx_actual] - x) if idx_actual is not None else np.inf

        idx_pred = np.argmin(np.abs(pred_dnums - x)) if len(pred_dnums) else None
        dist_pred = abs(pred_dnums[idx_pred] - x) if idx_pred is not None else np.inf

        idx_future = np.argmin(np.abs(future_dnums - x)) if len(future_dnums) else None
        dist_future = abs(future_dnums[idx_future] - x) if idx_future is not None else np.inf

        # choose nearest among the three series
        nearest = 'none'
        if dist_actual <= dist_pred and dist_actual <= dist_future:
            nearest = 'actual'
        elif dist_pred <= dist_actual and dist_pred <= dist_future:
            nearest = 'pred'
        else:
            nearest = 'future'

        if nearest == 'actual' and idx_actual is not None:
            dnum = actual_dnums[idx_actual]
            date = mdates.num2date(dnum)
            actual = actual_vals[idx_actual]
            # find predicted nearest to this actual date (may be offset)
            pred_idx = np.argmin(np.abs(pred_dnums - dnum)) if len(pred_dnums) else None
            predicted = pred_vals[pred_idx] if pred_idx is not None else float('nan')
        elif nearest == 'pred' and idx_pred is not None:
            dnum = pred_dnums[idx_pred]
            date = mdates.num2date(dnum)
            predicted = pred_vals[idx_pred]
            act_idx = np.argmin(np.abs(actual_dnums - dnum)) if len(actual_dnums) else None
            actual = actual_vals[act_idx] if act_idx is not None else float('nan')
        elif nearest == 'future' and idx_future is not None:
            dnum = future_dnums[idx_future]
            date = mdates.num2date(dnum)
            predicted = future_vals[idx_future]
            actual = float('nan')
        else:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return

        actual_text = f'\u00a3{actual:.2f}' if (not np.isnan(actual)) else 'N/A'
        pred_text = f'\u00a3{predicted:.2f}'
        text = f'{date.strftime("%Y-%m-%d")}\nActual: {actual_text}\nPredicted: {pred_text}'

        # position annotation near cursor
        annotation.xy = (event.xdata, event.ydata)
        annotation.set_text(text)
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", motion_hover)

    # save outputs
    out_dir = os.getcwd()
    present_day = dt.datetime.now().date()
    fig.savefig(os.path.join(out_dir, f"{present_day}.png"), dpi=300, bbox_inches='tight')
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_predictions_prices.flatten()})
    forecast_df.to_csv(os.path.join(out_dir, 'future_predictions_pytorch.csv'), index=False)
    print('\nSaved:', [f for f in os.listdir(out_dir) if f.endswith(('.png', '.csv'))])



    plt.show()


if __name__ == '__main__':
    main()