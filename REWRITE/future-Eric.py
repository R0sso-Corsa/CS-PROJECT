import time
script_start_time = time.time()

import pickle                  # save/reload Python objects (plots/data)
import numpy as np           # numerical arrays and operations
import matplotlib.pyplot as plt # plotting
import pandas                   # dataframes and manipulation
import yfinance as yf           # download market data from Yahoo Finance [Yahoo Finance API is unofficial and may break {ACTUAL YAHOO FINANCE API IS DISCONTINUED AND UNAVAILABLE}]
import datetime as dt

from sklearn.preprocessing import MinMaxScaler           # scale values to 0-1 for NN
from tensorflow.keras.models import Sequential           # import TensorFlow Keras API
from tensorflow.keras.layers import Dense, LSTM, Dropout # import TensorFlow Keras API
from tensorflow.keras.callbacks import TensorBoard, Callback                # TensorBoard callback for visualization
from sklearn.ensemble import GradientBoostingRegressor
import os                                                # filesystem operations (saving files, cwd)

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

# scale the 'Close' price into the range [0, 1] for stable LSTM training (feature_range shrinks the input values to a small range to avoid processing large numbers that can destabilize training)
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# how many past days to use as input for each prediction (sliding window length)
prediction_days = 365 # 60 default
future_day = 365     # NUMBER OF DAYS TO PREDICT INTO THE FUTURE (used for the forward recursive forecast)

# Training horizon: predict 1 day ahead for model (so test predictions are one-day ahead)
train_horizon = 1

# prepare training dataset: build windows from the full scaled series
x_train, y_train = [], []

# build input (x) and target (y) sliding windows for training
# ensure target index (x + train_horizon - 1) exists
for x in range(prediction_days, len(scaledData) - train_horizon + 1):
    x_train.append(scaledData[x-prediction_days:x, 0])
    y_train.append(scaledData[x + train_horizon - 1, 0])

# convert to NumPy arrays and reshape to (samples, timesteps, features)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Add this function near the top of your file, after imports:
def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
    """Calculate dropout rate that decreases linearly with epochs"""
    return max(final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs))

# Then modify your model definition section. Replace the current LSTM/Dropout pattern with:
epochs = 2  # increase training epochs for better learning
initial_dropout = 0.5  # Start with lower dropout to avoid oversmoothing
final_dropout = 0.05   # End with small dropout
batchSize = 64
train_time = 1 # do not use decimals
class DynamicDropoutCallback(Callback):
    def __init__(self, total_epochs, initial_rate=0.5, final_rate=0.1):
        super().__init__()
        self.total_epochs = total_epochs
        self.initial_rate = initial_rate
        self.final_rate = final_rate

    def on_epoch_begin(self, epoch, logs=None):
        new_rate = get_dynamic_dropout(epoch, self.total_epochs,
                                     self.initial_rate, self.final_rate)
        # Update dropout rates in all layers
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                layer.rate = new_rate
        print(f"\nEpoch {epoch+1}: Dropout rate set to {new_rate:.3f}")

# Create model with initial dropout rates
ai = Sequential()
ai.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
ai.add(Dropout(initial_dropout))

for i in range(train_time):
    ai.add(LSTM(units=100, return_sequences=True))
    ai.add(Dropout(initial_dropout))

ai.add(LSTM(units=100, return_sequences=True))
ai.add(Dropout(initial_dropout))

ai.add(LSTM(units=100))
ai.add(Dropout(initial_dropout))

ai.add(Dense(units=1))

# Add the dynamic dropout callback to your training
dynamic_dropout = DynamicDropoutCallback(epochs, initial_dropout, final_dropout)

# TENSORBOARD SETUP
# Create a unique log directory with timestamp to avoid overwriting previous runs
log_dir = os.path.join("logs", "fit", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,           # Record weight histograms every epoch
    write_graph=True,            # Visualize the model graph
    write_images=True,          # Don't write model weights as images (can be large)
    update_freq='epoch',         # Update metrics after each epoch
    profile_batch=0,             # Disable profiling to save resources
    embeddings_freq=0            # Disable embedding visualization\
)

print(f"\n{'='*60}")
print(f"TensorBoard logs will be saved to: {log_dir}")
print(f"To view TensorBoard, run this command in your terminal:")
print(f"tensorboard --logdir=logs/fit")
print(f"Then open http://localhost:6006 in your browser")
print(f"{'='*60}\n")

ai.compile(optimizer='adam', loss='mean_absolute_error')

# Modify your model.fit call to include the new callback:
ai.fit(x_train, y_train,
       epochs=epochs,
       batch_size=batchSize,
       callbacks=[tensorboard_callback, dynamic_dropout],
       verbose=1)

# Train a residual model (gradient boosting) on flattened windows to recover sharper local moves
try:
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    pred_train_scaled = ai.predict(x_train)
    residuals = y_train - pred_train_scaled.flatten()
    gbr_res = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    gbr_res.fit(x_train_flat, residuals)
    print("Trained residual GradientBoostingRegressor to sharpen forecasts.")
except Exception as e:
    print("Residual model training failed:", e)



# testing ai NN

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

# download test period OHLC data and extract actual closing prices
test_data = yf.download(f'{crypto_currency}-{against_currency}', test_start, test_end)
actual_prices = test_data['Close'].values

# combine historical and test close series so we can build ai inputs that include the last `prediction_days` values before the test period
train_series = data['Close'][data.index < test_start]
total_dataset = pandas.concat((train_series, test_data['Close']), axis=0)

# slice the last (len(test_data) + prediction_days) values to get the inputs needed for creating sliding windows that cover the test period
ai_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

# ensure shape is (N,1) for the scaler
ai_inputs = ai_inputs.reshape(-1, 1)
ai_inputs = scaler.transform(ai_inputs) # Use transform instead of fit_transform

x_test = []

# build test sequences the same way as training (sliding windows of length prediction_days)
x_test = []
for x in range(prediction_days, len(ai_inputs)):
    x_test.append(ai_inputs[x-prediction_days:x, 0])

# convert to numpy array and reshape to (samples, timesteps, features) for LSTM
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# run the ai to get normalized predictions and apply residual corrections if available
prediction_prices_scaled = ai.predict(x_test)
try:
    # apply residual corrections learned by GradientBoosting (if trained)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    corrections = gbr_res.predict(x_test_flat).reshape(-1, 1)
    prediction_prices_scaled = prediction_prices_scaled + corrections
except Exception:
    pass
prediction_prices = scaler.inverse_transform(prediction_prices_scaled)

# capture test period dates to align plot x-axis
prediction_dates = test_data.index

# create an offset of +1 day for predicted points so plotted predictions appear one day ahead
prediction_dates_offset = prediction_dates + pandas.Timedelta(days=1)

# extract last actual and predicted scalar values for reporting
last_actual_value = actual_prices[-1]
last_predicted_value = prediction_prices[-1][0]

# normalize possible array wrappers to plain Python floats for formatting
last_actual_value = float(np.asarray(last_actual_value).flatten()[0])
last_predicted_value = float(np.asarray(last_predicted_value).flatten()[0])

# compute percentage difference (predicted relative to actual)
percentage_difference = float((last_predicted_value - last_actual_value) / last_actual_value * 100)

# create output colours for the terminal report (green=better, red=worse) NOTE: DONT FORGET THE "m" AT THE END OF THE CODES
if percentage_difference >= 0:
    color_code = "\033[92m"  # Green
else:
    color_code = "\033[91m"  # Red
# colour code reset variable
reset_code = "\033[0m"

# output value report to terminal
print(f"Last actual value: {last_actual_value:.2f}")
print(f"Last predicted value: {last_predicted_value:.2f}")

# print percentage difference with colour coding and colour reset
print("Percentage difference:",f"{color_code}{percentage_difference:.2f}%{reset_code}")

# Don't show plot yet - we'll add future predictions first

# ===== PREDICT NEXT 30 DAYS =====
print(f"\n{'='*60}")
print(f"PREDICTING NEXT {future_day} DAYS...")
print(f"{'='*60}\n")

# Get the last prediction_days of scaled data to start the rolling forecast
real_data = ai_inputs[-prediction_days:, 0].copy()

# Store future predictions
future_predictions = []

# Rolling forecast: predict one day at a time, apply residual correction each step
for day in range(future_day):
    input_sequence = real_data[-prediction_days:].reshape(1, prediction_days, 1)

    # LSTM one-step prediction (scaled)
    next_pred_scaled = ai.predict(input_sequence, verbose=0)[0, 0]

    # residual correction from gradient booster (flattened input)
    try:
        input_flat = input_sequence.reshape(1, -1)
        res_corr = float(gbr_res.predict(input_flat)[0])
    except Exception:
        res_corr = 0.0

    corrected = next_pred_scaled + res_corr
    future_predictions.append(corrected)

    # Append corrected prediction to inputs for next step (keeps aggressiveness)
    real_data = np.append(real_data, corrected)

    if (day + 1) % 10 == 0:
        print(f"Predicted day {day + 1}/{future_day}")

# Convert predictions back to original price scale
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions_prices = scaler.inverse_transform(future_predictions)

# Create future dates starting from the day after the last known date
last_date = data.index[-1]
future_dates = pandas.date_range(start=last_date + pandas.Timedelta(days=1), periods=future_day)

# Print future predictions
print(f"\n{'='*60}")
print(f"FUTURE PRICE PREDICTIONS:")
print(f"{'='*60}")
for date, price in zip(future_dates, future_predictions_prices):
    print(f"{date.strftime('%Y-%m-%d')}: £{float(price[0]):.2f}")
print(f"{'='*60}\n")

# Calculate projected change
current_price = float(data['Close'].values[-1])
final_predicted_price = float(future_predictions_prices[-1][0])
projected_change = ((final_predicted_price - current_price) / current_price) * 100

if projected_change >= 0:
    change_color = "\033[92m"  # Green
else:
    change_color = "\033[91m"  # Red

print(f"Current Price: £{current_price:.2f}")
print(f"Predicted Price in {future_day} days: £{final_predicted_price:.2f}")
print(f"Projected Change: {change_color}{projected_change:.2f}%{reset_code}\n")

end = time.time()
timer = end - script_start_time
print("Script concluded for a duration of {:.2f} seconds".format(timer))

# ===== ADD ALL DATA TO EXISTING GRAPH =====

# Plot actual prices on the existing graph
graph.plot(prediction_dates, actual_prices, color='black', label='Actual Prices', linewidth=2)

# Plot predicted prices (test period) on the existing graph
graph.plot(prediction_dates_offset, prediction_prices, color='green', label='Predicted Prices (Test Period)', linewidth=2)

# Plot future predictions on the existing graph with a different color (red)
graph.plot(future_dates, future_predictions_prices, color='red', label=f'Future Forecast ({future_day} days)',
          linewidth=2.5, linestyle='--', marker='o', markersize=4)

# Plot connection lines between actual and predicted (optional, can be commented out if too cluttered)
for x1, y1, x2, y2 in zip(prediction_dates, actual_prices,
                          prediction_dates_offset, prediction_prices):
    graph.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

# Add vertical line at current date
graph.axvline(x=last_date, color='orange', linestyle=':', linewidth=2, label='Current Date', alpha=0.7)

# set axis labels/title for the graph
graph.set_title(f'{crypto_currency} Price Prediction with {future_day}-Day Forecast')
graph.set_xlabel('Date')
graph.set_ylabel(f'{crypto_currency} Price ({against_currency})')
graph.legend(loc='upper left')
graph.grid(True, alpha=0.3)
fig.autofmt_xdate()

def annotate_point(event):
   if event.inaxes:
      x_cursor, y_cursor = event.xdata, event.ydata
      index = np.argmin(np.abs(x - x_cursor))  # Find nearest index to the cursor position
      graph.annotate(
         f'({x[index]:.2f}, {y[index]:.2f})', xy=(x[index], y[index]),
         xytext=(x[index] + 1, y[index] + 0.5), arrowprops=dict(facecolor='black', arrowstyle='->'),
         fontsize=8, color='black')
   fig.canvas.draw_idle()

#IMAGE EXPORT - high resolution PNG and SVG (vector)
# save high-resolution raster (PNG) and vector (SVG) files into the current working directory
out_dir = os.getcwd()
fig.savefig(os.path.join(out_dir, "plot_highres.png"), dpi=600, bbox_inches="tight")  # high DPI raster
fig.savefig(os.path.join(out_dir, "plot_highres.svg"), bbox_inches="tight")           # vector (infinite resolution)

# Export future predictions to CSV
forecast_df = pandas.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions_prices.flatten()
})
forecast_df.to_csv(os.path.join(out_dir, "future_predictions.csv"), index=False)

# report saved files
print("\nSaved:", [f for f in os.listdir(out_dir) if f.endswith(('.png', '.svg', '.csv'))])

# Show the plot
plt.show()

print(f"\n{'='*60}")
print(f"REMINDER: To view TensorBoard visualization, run:")
print(f"tensorboard --logdir=logs/fit")
print(f"{'='*60}\n")



# python "C:\Users\Paron\OneDrive\Desktop\testing\ai_predict.py"


