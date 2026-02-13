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
# Disable MIOpen/cuDNN for LSTMs to prevent crash on some ROCm setups
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
    # Search for ticker symbol using yfinance; return None if failed
    try:
        return yf.Search(query)
    except Exception:
        return None

def choose_chart_interactive():
    # Interactively prompt user for ticker symbol; supports search
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
    # Linearly decay dropout rate over epochs
    return max(final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs))


class SequenceDataset(Dataset):
    # PyTorch Dataset for sequence data
    def __init__(self, sequences, targets):
        # Initialize dataset with sequences and targets
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        # Return total number of sequences
        return len(self.sequences)

    def __getitem__(self, idx):
        # Retrieve sequence and target at index
        return torch.from_numpy(self.sequences[idx]).float(), torch.from_numpy(np.array(self.targets[idx])).float()


class LSTMModel(nn.Module):
    # LSTM model with dynamic dropout and final linear layer
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        # Initialize LSTM layers and fully connected output layer
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = input_size
        for i in range(num_layers):
            self.layers.append(nn.LSTM(input_size=in_size, hidden_size=hidden_size // 2, bidirectional=True, batch_first=True))
            self.layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Forward pass through LSTM layers and final linear layer
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
    # Update dropout probability for all Dropout layers
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = new_p


def build_sequences(scaled_values, prediction_days):
    # Create sliding window sequences
    x, y = [], []
    for i in range(prediction_days, len(scaled_values)):
        x.append(scaled_values[i-prediction_days:i, :])
        y.append(scaled_values[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


# --- 1. System Configuration ---
chart = choose_chart_interactive()
device_type = input("Enter device type (cpu/cuda): ").strip().lower()
device = torch.device('cuda' if (device_type == 'cuda' and torch.cuda.is_available()) else 'cpu')
print(f"Using device: {device}")


# --- 2. Data Acquisition & Preprocessing ---
print("\n--- Preprocessing data once for Optuna study ---")

# Determine start date based on ticker history
ticker = yf.Ticker(chart)
hist_max = ticker.history(period='max')
if (hist_max is not None) and (not hist_max.empty):
    start = hist_max.index[0].to_pydatetime()
else:
    start = dt.datetime(2017, 1, 1)
end = dt.datetime.now()

# Download full historical data
data_full_raw = yf.download(chart, start=start, end=end, progress=False)

# Calculate technical indicators (SMA, RSI)
data_full_processed = data_full_raw.copy()
data_full_processed['SMA_14'] = data_full_processed['Close'].rolling(window=14).mean()

delta = data_full_processed['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()

rs = avg_gain / avg_loss
data_full_processed['RSI_14'] = 100 - (100 / (1 + rs))

# Fill missing values
data_full_processed = data_full_processed.ffill().bfill()

features = ['Close', 'Volume', 'SMA_14', 'RSI_14']

# Global scaler fitting (Note: scaling on full data before split carries some risk of leakage)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_full_processed[features].values)
scaled_data_full = scaler.transform(data_full_processed[features].values)

# Prepare specific test set (fixed window for study comparison)
test_start_fixed = dt.datetime(2025, 1, 1)
test_data_raw = yf.download(chart, test_start_fixed, end, progress=False)

# Re-calculate indicators for test set independently
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


# --- 3. Optimization Objective ---
def objective(trial, chart, device, scaled_full_data, scaler, features, data_full_processed, test_data_processed, actual_prices_for_test):
    # Optuna objective function; trains model and returns RMSE for a given trial
    # Hyperparameters
    prediction_days = trial.suggest_int('prediction_days', 30, 120, step=30)
    epochs = trial.suggest_int('epochs', 20, 100, step=20)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 100, 500, step=100)
    num_layers = trial.suggest_int('num_layers', 2, 6, step=2)
    initial_dropout = trial.suggest_float('initial_dropout', 0.2, 0.5, step=0.1)
    final_dropout = trial.suggest_float('final_dropout', 0.05, 0.2, step=0.05)
    
    input_size = len(features)

    # --- Training Data Preparation ---
    # Train on data strictly before test period (2025-01-01) to avoid leakage
    training_end_date = test_start_fixed - dt.timedelta(days=1)
    
    # Slice the full dataset up to the training end date
    if training_end_date in data_full_processed.index:
        training_data_slice = data_full_processed.loc[:training_end_date]
    else:
        training_data_slice = data_full_processed[data_full_processed.index < test_start_fixed]

    # Scale using the global scaler
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

    # --- Test Data Preparation ---
    # Prep inputs for test period by prepending train data tail (for history window)
    total_dataset_for_ai_inputs = pd.concat((data_full_processed[features], test_data_processed[features]), axis=0)
    
    # Slice to get exactly [test_start - prediction_days, end]
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

study.optimize(lambda trial: objective(trial, chart, device, scaled_data_full, scaler, features, data_full_processed, test_data_processed, actual_prices_for_test), n_trials=20)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")

trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")

for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


