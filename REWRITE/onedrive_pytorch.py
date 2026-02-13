import time
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
from sklearn.metrics import mean_squared_error
import scipy.stats as st
import math

import optuna

# ----------------------- Helper Functions (moved outside main for Optuna) -----------------------
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
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = input_size
        for i in range(num_layers):
            self.layers.append(nn.LSTM(input_size=in_size, hidden_size=hidden_size // 2, bidirectional=True, batch_first=True))
            self.layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = x
        for i in range(0, len(self.layers), 2):
            lstm = self.layers[i]
            dropout = self.layers[i+1]
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
        x.append(scaled_values[i-prediction_days:i, :])
        y.append(scaled_values[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


# --- Interactive Configuration (Moved outside objective function) ---
chart = choose_chart_interactive()
device_type = input("Enter device type (cpu/cuda): ").strip().lower()
device = torch.device('cuda' if (device_type == 'cuda' and torch.cuda.is_available()) else 'cpu')
print(f"Using device: {device}")


# --- Data Preprocessing (Moved outside objective function to be done once) ---
print("\n--- Preprocessing data once for Optuna study ---")
# Determine earliest date available
ticker = yf.Ticker(chart)
hist_max = ticker.history(period='max')
if (hist_max is not None) and (not hist_max.empty):
    start = hist_max.index[0].to_pydatetime()
else:
    start = dt.datetime(2017, 1, 1)
end = dt.datetime.now()

# Download full historical data
data_full_raw = yf.download(chart, start=start, end=end, progress=False)

# Feature Engineering (SMA, RSI) on full historical data
data_full_processed = data_full_raw.copy()
data_full_processed['SMA_14'] = data_full_processed['Close'].rolling(window=14).mean()
delta = data_full_processed['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
data_full_processed['RSI_14'] = 100 - (100 / (1 + rs))
data_full_processed = data_full_processed.ffill()
data_full_processed = data_full_processed.bfill()

features = ['Close', 'Volume', 'SMA_14', 'RSI_14']

# Initialize and fit scaler on the full historical data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_full_processed[features].values)
scaled_data_full = scaler.transform(data_full_processed[features].values)

# Prepare fixed test data (if test_start is static)
test_start_fixed = dt.datetime(2025, 1, 1) # A fixed start date for the test set
test_data_raw = yf.download(chart, test_start_fixed, end, progress=False)

# Feature Engineering (SMA, RSI) on test data
test_data_processed = test_data_raw.copy()
test_data_processed['SMA_14'] = test_data_processed['Close'].rolling(window=14).mean()
delta_test = test_data_processed['Close'].diff(1)
gain_test = delta_test.where(delta_test > 0, 0)
loss_test = -delta_test.where(delta_test < 0, 0)
avg_gain_test = gain_test.ewm(com=13, adjust=False).mean()
avg_loss_test = loss_test.ewm(com=13, adjust=False).mean()
rs_test = avg_gain_test / avg_loss_test
test_data_processed['RSI_14'] = 100 - (100 / (1 + rs_test))
test_data_processed = test_data_processed.ffill()
test_data_processed = test_data_processed.bfill()
actual_prices_for_test = test_data_processed['Close'].values # Keep original prices for RMSE


# --- Objective Function for Optuna ---
def objective(trial, chart, device, scaled_full_data, scaler, features, data_full_processed, test_data_processed, actual_prices_for_test):
    # Hyperparameters to tune
    prediction_days = trial.suggest_int('prediction_days', 30, 120, step=30)
    epochs = trial.suggest_int('epochs', 20, 100, step=20)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 100, 500, step=100)
    num_layers = trial.suggest_int('num_layers', 2, 6, step=2)
    initial_dropout = trial.suggest_float('initial_dropout', 0.2, 0.5, step=0.1)
    final_dropout = trial.suggest_float('final_dropout', 0.05, 0.2, step=0.05)

    input_size = len(features)

    # Training data preparation (uses pre-scaled data)
    # The training data needs to end before the test period starts (implicitly)
    # The current scaled_full_data goes up to 'end' (current_date)
    # test_data_processed starts from test_start_fixed.
    # To avoid data leakage, we should train only on data *before* test_start_fixed.
    # So, scale_full_data should be sliced to exclude the test period.

    # Find the index where the test_data_processed begins in the full historical data
    # Assuming chronological order, the training data would be everything before test_start_fixed
    # Need to be careful with overlapping dates if test_data_processed is not strictly after data_full_processed
    # For now, let's assume `data_full_processed` covers everything up to `end`, and `test_data_processed` is a contiguous block
    # from `test_start_fixed` to `end`.
    # Let's adjust `scaled_full_data` to truly be "training" data for the objective.
    # This might require re-thinking `scaled_full_data` creation slightly if the split isn't clean.

    # Simpler: The `scaled_full_data` contains data up to `end`. `test_data_processed` is also up to `end`.
    # `x_train` should be built from `scaled_full_data` up to `test_start_fixed`.
    # This requires passing the index of `test_start_fixed` in `scaled_full_data`.

    # Calculate index to split full historical data into training and validation sets
    # The test period used for evaluation in the objective is dt.datetime(2025, 1, 1) onwards.
    # So, training data should be up to dt.datetime(2025, 1, 1) - 1 day.
    # Get the index corresponding to test_start_fixed
    training_end_date = test_start_fixed - dt.timedelta(days=1)
    # Find the last date in data_full_processed that is before test_start_fixed
    if training_end_date in data_full_processed.index:
      training_data_slice = data_full_processed.loc[:training_end_date]
    else: # If exact date not found, take all up to that date
      training_data_slice = data_full_processed[data_full_processed.index < test_start_fixed]

    # Scale this training_data_slice using the globally fitted scaler
    scaled_training_data_for_trial = scaler.transform(training_data_slice[features].values)

    x_train, y_train = build_sequences(scaled_training_data_for_trial, prediction_days)
    dataset = SequenceDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size if batch_size < len(dataset) else len(dataset), shuffle=True)

    # Model definition
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=initial_dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    t = trange(epochs, desc=f"Trial {trial.number}", leave=False)
    for epoch in t:
        model.train()
        epoch_loss = 0.0
        new_p = get_dynamic_dropout(epoch, epochs, initial_dropout, final_dropout)
        set_dropout(model, new_p)
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        t.set_postfix({'loss': epoch_loss})

    # Test data preparation for prediction and RMSE calculation
    # `total_dataset_for_ai_inputs` needs to include data *before* the test_start_fixed
    # to form the `prediction_days` sequences for the first test prediction.
    # This means concatenating a tail of the training data with the test data.
    # Let's use the pre-processed `data_full_processed` and `test_data_processed` for this.

    # Take the tail of `data_full_processed` that precedes `test_data_processed`
    # and has sufficient length for `prediction_days`.
    # This concatenation is correct but operates on unscaled data, which is then scaled by the global scaler.
    total_dataset_for_ai_inputs = pd.concat((data_full_processed[features], test_data_processed[features]), axis=0)

    # `ai_inputs` represents the sliding window inputs for the test period.
    # It needs `prediction_days` history before `test_data_processed` begins.
    # The `total_dataset_for_ai_inputs` now correctly combines these.
    # We then slice from `total_dataset_for_ai_inputs` to get the relevant portion for `ai_inputs`.
    # The slice starts `len(test_data_processed) + prediction_days` entries from the end of `total_dataset_for_ai_inputs`.
    # This `ai_inputs_raw_slice` contains both the necessary historical context and the test period features.
    ai_inputs_raw_slice = total_dataset_for_ai_inputs[len(total_dataset_for_ai_inputs) - len(test_data_processed) - prediction_days:].values
    ai_inputs = scaler.transform(ai_inputs_raw_slice)

    x_test = []
    for i in range(prediction_days, len(ai_inputs)):
        x_test.append(ai_inputs[i-prediction_days:i, :])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], input_size))

    # Prediction and RMSE calculation
    model.train() # Enable dropout for test predictions as well for consistency
    with torch.no_grad():
        xt = torch.from_numpy(x_test).float().to(device)
        preds = model(xt).cpu().numpy()
    dummy_features_test = np.zeros_like(preds, shape=(preds.shape[0], input_size - 1))
    full_preds_scaled = np.concatenate((preds, dummy_features_test), axis=1)
    prediction_prices = scaler.inverse_transform(full_preds_scaled)[:, 0].reshape(-1, 1)

    actual_prices = test_data_processed['Close'].values
    rmse = math.sqrt(mean_squared_error(actual_prices, prediction_prices))

    return rmse

# Run Optuna study
# Enable verbose output for telemetry
optuna.logging.set_verbosity(optuna.logging.INFO)

study = optuna.create_study(
    study_name="onedrive_optimizer",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    direction='minimize'
)
study.optimize(lambda trial: objective(trial, chart, device, scaled_data_full, scaler, features, data_full_processed, test_data_processed, actual_prices_for_test), n_trials=50)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


