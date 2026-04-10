import time
script_start_time = time.time()

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas
import yfinance as yf
import datetime as dt
import os

# PyTorch imports instead of TensorFlow
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler

# Device configuration - works with both CUDA (NVIDIA/ROCm) and CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"{'='*60}\n")

# Create figure early
fig = plt.figure(figsize=(18, 9))
graph = fig.add_subplot(1, 1, 1)

# Parameters
chart = '^GSPC'  # S&P 500
start = dt.datetime(2017, 1, 1)
end = dt.datetime.now()

# Download data
data = yf.download(f'{chart}', start=start, end=end)
print(data.head())

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Parameters
prediction_days = 120
future_day = 30

# Prepare training data
training_data = scaledData[:-prediction_days]
x_train, y_train = [], []

for x in range(prediction_days, len(training_data)):
    x_train.append(scaledData[x-prediction_days:x, 0])
    y_train.append(scaledData[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Convert to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)

# Training parameters
epochs = 3
initial_dropout = 0.5
final_dropout = 0.1
batch_size = 1000
train_time = 3  # number of middle LSTM layers

# Define dynamic dropout function
def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
    """Calculate dropout rate that decreases linearly with epochs"""
    return max(final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs))

# Define LSTM Model
class DynamicLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=500, num_middle_layers=3, initial_dropout=0.5):
        super(DynamicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_middle_layers = num_middle_layers
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(initial_dropout)
        
        # Middle LSTM layers
        self.middle_lstms = nn.ModuleList()
        self.middle_dropouts = nn.ModuleList()
        for i in range(num_middle_layers):
            self.middle_lstms.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))
            self.middle_dropouts.append(nn.Dropout(initial_dropout))
        
        # Additional LSTM layer
        self.lstm_extra = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_extra = nn.Dropout(initial_dropout)
        
        # Final LSTM layer (no return_sequences)
        self.lstm_final = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_final = nn.Dropout(initial_dropout)
        
        # Dense output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # First LSTM
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Middle LSTM layers
        for i in range(self.num_middle_layers):
            out, _ = self.middle_lstms[i](out)
            out = self.middle_dropouts[i](out)
        
        # Extra LSTM
        out, _ = self.lstm_extra(out)
        out = self.dropout_extra(out)
        
        # Final LSTM
        out, _ = self.lstm_final(out)
        out = self.dropout_final(out)
        
        # Take last time step and pass through dense layer
        out = self.fc(out[:, -1, :])
        return out
    
    def update_dropout_rate(self, new_rate):
        """Update dropout rate for all dropout layers"""
        self.dropout1.p = new_rate
        for dropout in self.middle_dropouts:
            dropout.p = new_rate
        self.dropout_extra.p = new_rate
        self.dropout_final.p = new_rate

# Create model and move to device
model = DynamicLSTM(input_size=1, hidden_size=500, num_middle_layers=train_time, 
                    initial_dropout=initial_dropout).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TensorBoard setup
log_dir = os.path.join("logs", "fit", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

print(f"\n{'='*60}")
print(f"TensorBoard logs will be saved to: {log_dir}")
print(f"To view TensorBoard, run this command in your terminal:")
print(f"tensorboard --logdir=logs/fit")
print(f"Then open http://localhost:6006 in your browser")
print(f"{'='*60}\n")

# Create DataLoader for batching
dataset = TensorDataset(x_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Starting training...")
print(f"{'='*60}")

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    # Calculate and update dropout rate
    dropout_rate = get_dynamic_dropout(epoch, epochs, initial_dropout, final_dropout)
    model.update_dropout_rate(dropout_rate)
    print(f"\nEpoch {epoch+1}/{epochs}: Dropout rate set to {dropout_rate:.3f}")
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # Calculate average loss
    avg_loss = epoch_loss / num_batches
    
    # Log to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Dropout_rate', dropout_rate, epoch)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

print(f"{'='*60}")
print("Training complete!\n")

# Close TensorBoard writer
writer.close()

# Testing the model
print("Testing model on historical data...")

test_start = dt.datetime(2010, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(f'{chart}', test_start, test_end)
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

# Convert to tensor and move to device
x_test_tensor = torch.FloatTensor(x_test).to(device)

# Make predictions
model.eval()
with torch.no_grad():
    prediction_prices_tensor = model(x_test_tensor)

# Move back to CPU and convert to numpy
prediction_prices = prediction_prices_tensor.cpu().numpy()
prediction_prices = scaler.inverse_transform(prediction_prices)

prediction_dates = test_data.index
prediction_dates_offset = prediction_dates + pandas.Timedelta(days=-1)

last_actual_value = float(actual_prices[-1])
last_predicted_value = float(prediction_prices[-1][0])
percentage_difference = float((last_predicted_value - last_actual_value) / last_actual_value * 100)

# Color coding
if percentage_difference >= 0:
    color_code = "\033[92m"  # Green
else:
    color_code = "\033[91m"  # Red
reset_code = "\033[0m"

print(f"Last actual value: ${last_actual_value:.2f}")
print(f"Last predicted value: ${last_predicted_value:.2f}")
print("Percentage difference:", f"{color_code}{percentage_difference:.2f}%{reset_code}")

# Predict future days
print(f"\n{'='*60}")
print(f"PREDICTING NEXT {future_day} DAYS...")
print(f"{'='*60}\n")

real_data = ai_inputs[-prediction_days:, 0].copy()
future_predictions = []

model.eval()
with torch.no_grad():
    for day in range(future_day):
        input_sequence = torch.FloatTensor(real_data[-prediction_days:]).reshape(1, prediction_days, 1).to(device)
        next_pred_scaled = model(input_sequence)
        next_pred_scaled_cpu = next_pred_scaled.cpu().numpy()[0, 0]
        
        future_predictions.append(next_pred_scaled_cpu)
        real_data = np.append(real_data, next_pred_scaled_cpu)
        
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
    print(f"{date.strftime('%Y-%m-%d')}: ${float(price[0]):.2f}")
print(f"{'='*60}\n")

current_price = float(data['Close'].values[-1])
final_predicted_price = float(future_predictions_prices[-1][0])
projected_change = ((final_predicted_price - current_price) / current_price) * 100

if projected_change >= 0:
    change_color = "\033[92m"
else:
    change_color = "\033[91m"

print(f"Current Price: ${current_price:.2f}")
print(f"Predicted Price in {future_day} days: ${final_predicted_price:.2f}")
print(f"Projected Change: {change_color}{projected_change:.2f}%{reset_code}\n")

end = time.time()
timer = end - script_start_time
print("Script concluded for a duration of {:.2f} seconds".format(timer))

# Plotting
graph.plot(prediction_dates, actual_prices, color='black', label='Actual Prices', linewidth=2)
graph.plot(prediction_dates_offset, prediction_prices, color='green', label='Predicted Prices (Test Period)', linewidth=2)
graph.plot(future_dates, future_predictions_prices, color='red', label=f'Future Forecast ({future_day} days)',
          linewidth=2.5, linestyle='--', marker='o', markersize=4)

for x1, y1, x2, y2 in zip(prediction_dates, actual_prices,
                          prediction_dates_offset, prediction_prices):
    graph.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

graph.axvline(x=last_date, color='orange', linestyle=':', linewidth=2, label='Current Date', alpha=0.7)

graph.set_title(f'{chart} Price Prediction with {future_day}-Day Forecast (PyTorch)')
graph.set_xlabel('Date')
graph.set_ylabel(f'{chart} Price (USD)')
graph.legend(loc='upper left')
graph.grid(True, alpha=0.3)
fig.autofmt_xdate()

# Save outputs
out_dir = os.getcwd()
fig.savefig(os.path.join(out_dir, f"{chart.replace('^', '')}_pytorch.png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "plot_highres_pytorch.svg"), bbox_inches="tight")

forecast_df = pandas.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions_prices.flatten()
})
forecast_df.to_csv(os.path.join(out_dir, "future_predictions_pytorch.csv"), index=False)

print("\nSaved:", [f for f in os.listdir(out_dir) if f.endswith(('.png', '.svg', '.csv'))])

plt.show()

print(f"\n{'='*60}")
print(f"REMINDER: To view TensorBoard visualization, run:")
print(f"tensorboard --logdir=logs/fit")
print(f"{'='*60}\n")

# Save the model
model_path = os.path.join(out_dir, "sp500_lstm_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler,
    'prediction_days': prediction_days
}, model_path)
print(f"Model saved to: {model_path}")
print("\n✅ SCRIPT COMPLETE - GPU ACCELERATION ENABLED" if torch.cuda.is_available() else "\n✅ SCRIPT COMPLETE - CPU MODE")