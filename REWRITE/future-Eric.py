import time
script_start_time = time.time()
print("Execution time recording...")


import pickle                  # save/reload Python objects (plots/data)
import numpy as np           # numerical arrays and operations
import matplotlib.pyplot as plt # plotting
import matplotlib.dates as mdates
import pandas                   # dataframes and manipulation
import yfinance as yf           # download market data from Yahoo Finance [Yahoo Finance API is unofficial and may break {ACTUAL YAHOO FINANCE API IS DISCONTINUED AND UNAVAILABLE}]
import datetime as dt

from sklearn.preprocessing import MinMaxScaler           # scale values to 0-1 for NN
from tensorflow.keras.models import Sequential           # import TensorFlow Keras API
from tensorflow.keras.layers import Dense, LSTM, Dropout # import TensorFlow Keras API
from tensorflow.keras.callbacks import TensorBoard, Callback, EarlyStopping                # TensorBoard callback for visualization
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
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

# define test split BEFORE fitting scalers to avoid leakage
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
train_series = data['Close'][data.index < test_start]

# Work in log-returns to force the model to learn dynamics instead of levels
# compute train log-returns and fit scaler on returns only (no leakage)
train_log = np.log(train_series)
train_returns = train_log.diff().dropna().values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_returns = scaler.fit_transform(train_returns)

# multi-step horizon (days) for direct forecasting
HORIZON = future_day

# how many past days to use as input for each prediction (sliding window length)
prediction_days = 365 # 60 default
future_day = 365     # NUMBER OF DAYS TO PREDICT INTO THE FUTURE (used for the forward recursive forecast)

# Training horizon: predict 1 day ahead for model (so test predictions are one-day ahead)
train_horizon = 1

# prepare training dataset: build windows from the scaled returns
x_train, y_train = [], []

# build input (x) and target (y) sliding windows for training using returns
# ensure target index exists in the returns array
for i in range(prediction_days, len(scaled_returns) - HORIZON + 1):
    x_train.append(scaled_returns[i-prediction_days:i, 0])
    # target is the next HORIZON returns starting at i
    y_train.append(scaled_returns[i:i+HORIZON, 0])

# convert to NumPy arrays and reshape to (samples, timesteps, features)
x_train, y_train = np.array(x_train), np.array(y_train)  # y_train shape -> (n_samples, HORIZON)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create a time-ordered validation split to avoid leakage (walk-forward style)
val_size = max(1, int(0.1 * x_train.shape[0]))  # 10% for validation, at least 1 sample
x_val = x_train[-val_size:]
y_val = y_train[-val_size:]
# keep the earlier portion for training
x_train = x_train[:-val_size]
y_train = y_train[:-val_size]

# Early stopping callback to prevent overfitting (restores best weights)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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

ai.add(Dense(units=HORIZON))

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
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback, dynamic_dropout, early_stop],
    verbose=1)

# Train a residual model (gradient boosting) on flattened windows to recover sharper local moves
try:
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    pred_train_scaled = ai.predict(x_train)  # shape (n_samples, HORIZON)
    residuals = y_train - pred_train_scaled  # shape (n_samples, HORIZON)
    gbr_base = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    gbr_res = MultiOutputRegressor(gbr_base)
    gbr_res.fit(x_train_flat, residuals)
    print("Trained multi-output residual GradientBoostingRegressor to sharpen forecasts.")
except Exception as e:
    print("Residual model training failed:", e)



# testing ai NN

# download test period OHLC data and extract actual closing prices
test_data = yf.download(f'{crypto_currency}-{against_currency}', test_start, test_end)
actual_prices = test_data['Close'].values

# combine historical and test close series so we can build ai inputs that include the last `prediction_days` values before the test period
train_series = data['Close'][data.index < test_start]
total_dataset = pandas.concat((train_series, test_data['Close']), axis=0)

# compute log-returns on the combined series and take the last (len(test_data)+prediction_days) returns
total_log = np.log(total_dataset)
total_returns = total_log.diff().dropna()
ai_inputs_returns = total_returns[-(len(test_data) + prediction_days):].values.reshape(-1, 1)

# scale returns (scaler already fit on train returns)
ai_inputs = scaler.transform(ai_inputs_returns)

x_test = []

# build test sequences the same way as training (sliding windows of length prediction_days)
x_test = []
for x in range(prediction_days, len(ai_inputs)):
    x_test.append(ai_inputs[x-prediction_days:x, 0])

# convert to numpy array and reshape to (samples, timesteps, features) for LSTM
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# run the ai to get normalized multi-step predictions and apply residual corrections if available
pred_all_scaled = ai.predict(x_test)  # shape (n_test_samples, HORIZON)
try:
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    corrections = gbr_res.predict(x_test_flat)  # shape (n_test_samples, HORIZON)
    pred_all_scaled = pred_all_scaled + corrections
except Exception:
    pass

# For test-period plotting/evaluation use the 1-day ahead predictions (first column)
pred_first_scaled = pred_all_scaled[:, 0].reshape(-1, 1)
pred_first_returns = scaler.inverse_transform(pred_first_scaled).flatten()
# approximate predicted prices for test: previous actual price * exp(predicted_return)
prediction_prices = (actual_prices * np.exp(pred_first_returns)).reshape(-1, 1)

# capture test period dates to align plot x-axis (keep actual test dates)
prediction_dates = test_data.index

# create one-day offset for predicted points so plotted predictions appear one day ahead
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


# Direct multi-step forecast from the last available input using the trained model
last_input = ai_inputs[-prediction_days:].reshape(1, prediction_days, 1)
last_pred_scaled = ai.predict(last_input)[0]  # shape (HORIZON,)
try:
    last_pred_scaled = last_pred_scaled + gbr_res.predict(last_input.reshape(1, -1))[0]
except Exception:
    pass

# inverse-transform returns and reconstruct price trajectory
pred_returns = scaler.inverse_transform(last_pred_scaled.reshape(-1, 1)).flatten()
last_price = float(data['Close'].values[-1])
future_prices = []
price = last_price
for r in pred_returns:
    price = price * np.exp(r)
    future_prices.append(price)
future_predictions_prices = np.array(future_prices).reshape(-1, 1)

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

def on_move(event):
    if event.inaxes:
        # Remove previous crosshair lines (no setter on Axes.lines)
        for line in list(graph.lines):
            try:
                if (line.get_linestyle() == '--') and (line.get_color() in ('red', 'blue')):
                    line.remove()
            except Exception:
                # ignore any unexpected line objects
                pass

        # Add new crosshair lines
        graph.axvline(event.xdata, color='red', linestyle='--')  # Vertical line
        graph.axhline(event.ydata, color='blue', linestyle='--')  # Horizontal line
        plt.draw()

fig.canvas.mpl_connect('motion_notify_event', on_move)

# Precompute numeric date arrays and flattened price arrays for fast nearest-point lookup
actual_dnums = mdates.date2num(prediction_dates)
pred_dnums = mdates.date2num(prediction_dates_offset)
future_dnums = mdates.date2num(future_dates)

actual_vals = np.array(actual_prices).flatten()
pred_vals = prediction_prices.flatten()
future_vals = future_predictions_prices.flatten()

# persistent annotation that follows the mouse
annotation = graph.annotate(
    '', xy=(0, 0), xytext=(15, 15), textcoords='offset points',
    bbox=dict(boxstyle='round', fc='w', alpha=0.9), fontsize=9
)
annotation.set_visible(False)


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

    actual_text = f'£{actual:.2f}' if (not np.isnan(actual)) else 'N/A'
    pred_text = f'£{predicted:.2f}'
    text = f'{date.strftime("%Y-%m-%d")}\nActual: {actual_text}\nPredicted: {pred_text}'

    # position annotation near cursor
    annotation.xy = (event.xdata, event.ydata)
    annotation.set_text(text)
    annotation.set_visible(True)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", motion_hover)

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


