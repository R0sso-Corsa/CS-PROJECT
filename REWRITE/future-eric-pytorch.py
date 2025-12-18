import time
script_start_time = time.time()

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas
import yfinance as yf
import datetime as dt
import os

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

using_backend = "cpu"
device = torch.device("cpu")
try:
    import torch_directml as dml
    device = dml.device()   # torch_directml device object
    using_backend = "directml"
except Exception:
    # fallback to CUDA/ROCm if available (Linux/ROCm or NVIDIA CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        using_backend = "cuda/rocm"
print(f"Using backend: {using_backend}))")

# create a reusable figure/axis early so other code can add to it later
fig = plt.figure(figsize=(18, 9))
graph = fig.add_subplot(1, 1, 1)

# create quickly editable variables to change graph parameters. NOTE: dont change "end" parameter if you want to include current day records.
crypto_currency = 'BTC'
against_currency = 'GBP'
start = dt.datetime(2017, 1, 1)
end = dt.datetime.now()

data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)

# quick sanity check of downloaded dataframe (see if data downloaded correctly)
print(data.head())

# scale the 'Close' price into the range [0, 1] for stable LSTM training
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# how many past days to use as input for each prediction (sliding window length)
prediction_days = 120
future_day = 30

# prepare training dataset: remove the final `prediction_days` so we can make complete input windows of length `prediction_days` that have a following target
training_data = scaledData[:-prediction_days]
x_train, y_train = [], []

for x in range(prediction_days, len(training_data)):
    x_train.append(scaledData[x-prediction_days:x, 0])
    y_train.append(scaledData[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# dynamic dropout helper
def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
    """Calculate dropout rate that decreases linearly with epochs"""
    return max(final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs))

# PyTorch model: stacked LSTMs with Dropout between layers
class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, dropout=0.5):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.drop2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.drop3 = nn.Dropout(dropout)

        self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.drop4 = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)

        out, _ = self.lstm2(out)
        out = self.drop2(out)

        out, _ = self.lstm3(out)
        out = self.drop3(out)

        out, _ = self.lstm4(out)
        # take last timestep's output for prediction
        out = out[:, -1, :]
        out = self.drop4(out)

        out = self.fc(out)
        return out

def set_dropout_rate(model, p):
    """Set dropout probability for all nn.Dropout modules in the model"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(p)

# Training hyperparameters
epochs = 10
initial_dropout = 0.5
final_dropout = 0.1
batch_size = 32
learning_rate = 1e-3
train_time = 1  # retained from original; used to control number of extra LSTM additions earlier (not needed here)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare DataLoader
tensor_x = torch.from_numpy(x_train).float()
tensor_y = torch.from_numpy(y_train).float().unsqueeze(1)
dataset = TensorDataset(tensor_x, tensor_y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate model, loss, optimizer
model = PriceLSTM(input_size=1, hidden_size=100, dropout=initial_dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with dynamic dropout applied at epoch start
for epoch in range(epochs):
    model.train()
    new_p = get_dynamic_dropout(epoch, epochs, initial_dropout, final_dropout)
    set_dropout_rate(model, new_p)
    epoch_losses = []

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    print(f"Epoch {epoch+1}/{epochs} - Dropout: {new_p:.3f} - Loss: {avg_loss:.6f}")

# testing ai NN

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(f'{crypto_currency}-{against_currency}', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pandas.concat((data['Close'], test_data['Close']), axis=0)

ai_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
ai_inputs = ai_inputs.reshape(-1, 1)
ai_inputs = scaler.transform(ai_inputs)

x_test = []
for x in range(prediction_days, len(ai_inputs)):
    x_test.append(ai_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict on test set
model.eval()
with torch.no_grad():
    xt = torch.from_numpy(x_test).float().to(device)
    preds = model(xt).cpu().numpy()

prediction_prices = scaler.inverse_transform(preds)

prediction_dates = test_data.index
prediction_dates_offset = prediction_dates + pandas.Timedelta(days=1)

last_actual_value = actual_prices[-1]
last_predicted_value = prediction_prices[-1][0]

last_actual_value = float(np.asarray(last_actual_value).flatten()[0])
last_predicted_value = float(np.asarray(last_predicted_value).flatten()[0])

percentage_difference = float((last_predicted_value - last_actual_value) / last_actual_value * 100)

if percentage_difference >= 0:
    color_code = "\033[92m"
else:
    color_code = "\033[91m"
reset_code = "\033[0m"

print(f"Last actual value: {last_actual_value:.2f}")
print(f"Last predicted value: {last_predicted_value:.2f}")
print("Percentage difference:", f"{color_code}{percentage_difference:.2f}%{reset_code}")

# ===== PREDICT NEXT future_day DAYS =====
print(f"\n{'='*60}")
print(f"PREDICTING NEXT {future_day} DAYS...")
print(f"{'='*60}\n")

real_data = ai_inputs[-prediction_days:, 0].copy()
future_predictions = []

model.eval()
with torch.no_grad():
    for day in range(future_day):
        input_sequence = real_data[-prediction_days:].reshape(1, prediction_days, 1)
        xt = torch.from_numpy(input_sequence).float().to(device)
        next_pred_scaled = model(xt).cpu().numpy()  # shape (1,1)
        future_predictions.append(next_pred_scaled[0, 0])
        real_data = np.append(real_data, next_pred_scaled[0, 0])

        if (day + 1) % 10 == 0:
            print(f"Predicted day {day + 1}/{future_day}")

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions_prices = scaler.inverse_transform(future_predictions)

last_date = data.index[-1]
future_dates = pandas.date_range(start=last_date + pandas.Timedelta(days=1), periods=future_day)

print(f"\n{'='*60}")
print(f"FUTURE PRICE PREDICTIONS:")
print(f"{'='*60}")
for date, price in zip(future_dates, future_predictions_prices):
    print(f"{date.strftime('%Y-%m-%d')}: £{float(price[0]):.2f}")
print(f"{'='*60}\n")

current_price = float(data['Close'].values[-1])
final_predicted_price = float(future_predictions_prices[-1][0])
projected_change = ((final_predicted_price - current_price) / current_price) * 100

if projected_change >= 0:
    change_color = "\033[92m"
else:
    change_color = "\033[91m"

print(f"Current Price: £{current_price:.2f}")
print(f"Predicted Price in {future_day} days: £{final_predicted_price:.2f}")
print(f"Projected Change: {change_color}{projected_change:.2f}%{reset_code}\n")

end = time.time()
timer = end - script_start_time
print("Script concluded for a duration of {:.2f} seconds".format(timer))

# ===== ADD ALL DATA TO EXISTING GRAPH =====

graph.plot(prediction_dates, actual_prices, color='black', label='Actual Prices', linewidth=2)
graph.plot(prediction_dates_offset, prediction_prices, color='green', label='Predicted Prices (Test Period)', linewidth=2)

graph.plot(future_dates, future_predictions_prices, color='red', label=f'Future Forecast ({future_day} days)',
          linewidth=2.5, linestyle='--', marker='o', markersize=4)

for x1, y1, x2, y2 in zip(prediction_dates, actual_prices,
                          prediction_dates_offset, prediction_prices):
    graph.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

graph.axvline(x=last_date, color='orange', linestyle=':', linewidth=2, label='Current Date', alpha=0.7)

graph.set_title(f'{crypto_currency} Price Prediction with {future_day}-Day Forecast')
graph.set_xlabel('Date')
graph.set_ylabel(f'{crypto_currency} Price ({against_currency})')
graph.legend(loc='upper left')
graph.grid(True, alpha=0.3)
fig.autofmt_xdate()

out_dir = os.getcwd()
fig.savefig(os.path.join(out_dir, "plot_highres.png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "plot_highres.svg"), bbox_inches="tight")

forecast_df = pandas.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions_prices.flatten()
})
forecast_df.to_csv(os.path.join(out_dir, "future_predictions.csv"), index=False)

print("\nSaved:", [f for f in os.listdir(out_dir) if f.endswith(('.png', '.svg', '.csv'))])

plt.show()

print(f"\n{'='*60}")
print("REMINDER: This version uses PyTorch for model definition and training.")
print(f"{'='*60}\n")

# small diagnostic to print detected bytes around the error
from pathlib import Path
b = Path("future-eric-pytorch.py").read_bytes()
print(b[:200])

