import pickle                   # save/reload Python objects (plots/data)
import numpy as np              # numerical arrays and operations
import matplotlib.pyplot as plt # plotting
import pandas                   # dataframes and manipulation
import yfinance as yf           # download market data from Yahoo Finance [Yahoo Finance API is unofficial and may break {ACTUAL YAHOO FINANCE API IS DISCONTINUED AND UNAVAILABLE}]
import datetime as dt

from sklearn.preprocessing import MinMaxScaler          # scale values to 0-1 for NN
from keras.layers import Dense, LSTM, Dropout           # import neural network layers
from keras.models import Sequential                     # import NN model
import os                                               # filesystem operations (saving files, cwd)

# create a reusable figure/axis early so other code can add to it later
fig = plt.figure(figsize=(18, 9))          
ax = fig.add_subplot(1, 1, 1)

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
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# how many past days to use as input for each prediction (sliding window length)
prediction_days = 60
future_day = 30     # unused in current code, kept for possible future use

# prepare training dataset: remove the final `prediction_days` so we can make complete input windows of length `prediction_days` that have a following target
training_data = scaled_data[:-prediction_days]
x_train, y_train = [], []

# build input (x) and target (y) sliding windows for training
for x in range(prediction_days, len(training_data)):
    # take the window of `prediction_days` ending at index x-1 and the target at x
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

# convert to NumPy arrays and reshape to (samples, timesteps, features)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# establish NN model

# Initialize a Sequential neural network (layers will be stacked in order)
model = Sequential()

# First LSTM layer:
# - 100 units (neurons) in the LSTM cell
# - return_sequences=True, outputs a sequence of hidden states (needed when stacking multiple LSTM layers, so the next LSTM gets a sequence input)
# - input_shape = (timesteps, features) = (x_train.shape[1], 1), here, timesteps = number of days in each input window (prediction_days),
#   features = 1 (just the closing price per day)
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# Dropout layer:
# - randomly "drops" 20% of neurons during training
# - helps prevent overfitting by not letting the network rely too heavily on specific paths
model.add(Dropout(0.2))

# Second LSTM layer:
# - again 100 LSTM units
# - return_sequences=True because another LSTM follows
model.add(LSTM(units=100, return_sequences=True))

# Dropout again (20%)
model.add(Dropout(0.2))

# Third (final) LSTM layer:
# - 100 LSTM units
# - return_sequences=False by default (so only the final output of the sequence is returned)
#   this makes sense here, because after the last LSTM you only need one output vector
model.add(LSTM(units=100))

# Dropout again (20%)
model.add(Dropout(0.2))

# Dense (fully connected) output layer:
# - units=1 because we want to predict a single numeric value (the next closing price)
model.add(Dense(units=1))

# RUN-DOWN: Lines 63 to 89 set up a 3-layer neural network. First, the AI generates its first initial prediction based on the first 60 days of data. It then drops 20% of the neurons to avoid creating a bias in the model. 
# The second layer takes the output of the first layer and refines it further, again dropping 20% of neurons. The third layer does the same, but only outputs a single value (the predicted price) instead of a sequence. 
# Finally, another 20% of neurons are dropped before the final Dense layer produces the output.
# The whole reason we drop 20% of the neurons to avoid overfitting is that if we don't, the model might just memorize the training data instead of learning to generalize from it.


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=10000000)
# {CONTROL NUMBER OF PASSES} (HIGHER BATCH SIZE = FASTER TRAINING, LOWER = SLOWER TRAINING [BUT MORE ACCURATE])

# testing model NN

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

# download test period OHLC data and extract actual closing prices
test_data = yf.download(f'{crypto_currency}-{against_currency}', test_start, test_end)
actual_prices = test_data['Close'].values

# combine historical and test close series so we can build model inputs that include the last `prediction_days` values before the test period
total_dataset = pandas.concat((data['Close'], test_data['Close']), axis=0)

# slice the last (len(test_data) + prediction_days) values to get the inputs needed for creating sliding windows that cover the test period
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

# ensure shape is (N,1) for the scaler
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs) # NOTE: use scaler.transform here to reuse the scaler fitted on training data. Current code calls fit_transform which refits the scaler on combined data.

x_test = []

# build test sequences the same way as training (sliding windows of length prediction_days)
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

# convert to numpy array and reshape to (samples, timesteps, features) for LSTM
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# run the model to get normalized predictions and inverse transform to original scale NOTE: PREDICTION STARTS HERE
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

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

# CREATE PLOT
plt.plot(prediction_dates, actual_prices, color='black', label='actual prices')
plt.plot(prediction_dates_offset, prediction_prices, color='green', label='predicted prices (offset +1 day)')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Date')
plt.ylabel(f'{crypto_currency} price')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
for x1, y1, x2, y2 in zip(prediction_dates, actual_prices,
                          prediction_dates_offset, prediction_prices):
    plt.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
plt.show()


# PREDICT NEXT

real_data = model_inputs[-prediction_days:, 0]  # Get the last 60 elements
real_data = np.array(real_data)
real_data = np.reshape(real_data, (1, prediction_days, 1))

# predict the next single-step value and invert scaling
prediction = model.predict(real_data)
predicton = scaler.inverse_transform(prediction)

# add annotated lines to the previously created ax (main figure created at top of file)
ax.plot(prediction_dates, actual_prices, color='black', label='actual prices', linewidth=1.2, antialiased=True)
ax.plot(prediction_dates_offset, prediction_prices, color='green', label='predicted (+1d)', linewidth=1.2, antialiased=True)

# set axis labels/title for the ax-based figure
ax.set_title(f'{crypto_currency} price prediction')
ax.set_xlabel('Date')
ax.set_ylabel(f'{crypto_currency} price')
ax.legend(loc='upper left')
fig.autofmt_xdate()


#IMAGE EXPORT - high resolution PNG and SVG (vector)
# plot again on ax to ensure the exported figure contains the desired lines and styling
ax.plot(prediction_dates, actual_prices, color='black', label='actual prices', linewidth=1.2, antialiased=True)
ax.plot(prediction_dates_offset, prediction_prices, color='green', label='predicted (+1d)', linewidth=1.2, antialiased=True)

# finalize titles/labels/legend for export
ax.set_title(f'{crypto_currency} price prediction')
ax.set_xlabel('Date')
ax.set_ylabel(f'{crypto_currency} price')
ax.legend(loc='upper left')
fig.autofmt_xdate()

# save high-resolution raster (PNG) and vector (SVG) files into the current working directory
out_dir = os.getcwd()
fig.savefig(os.path.join(out_dir, "plot_highres.png"), dpi=600, bbox_inches="tight")  # high DPI raster
fig.savefig(os.path.join(out_dir, "plot_highres.svg"), bbox_inches="tight")           # vector (infinite resolution)

# report saved files and avoid attempting to open a GUI when running headless (e.g. in Docker)
print("Saved:", os.listdir(out_dir))
if os.environ.get("DISPLAY"):
    plt.show()
else:
    print("No DISPLAY — open plot_highres.png or plot_highres.svg on the host.")

 # not updated (1 version behind)

 # C0mpt0n1991