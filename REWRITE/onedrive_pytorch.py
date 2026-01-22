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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler


# ----------------------- Configuration -----------------------
chart = '^GSPC'
prediction_days = 60
future_day = 30
epochs = 20
batch_size = 1000
initial_dropout = 0.5
final_dropout = 0.1
train_time = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def build_sequences(scaled_values, prediction_days):
    x, y = [], []
    for i in range(prediction_days, len(scaled_values)):
        x.append(scaled_values[i-prediction_days:i, 0])
        y.append(scaled_values[i, 0])
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y


def set_dropout(model, new_p):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = new_p


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

    # tensorboard writer
    log_dir = os.path.join('logs', 'fit', dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # training loop with dynamic dropout
    for epoch in range(epochs):
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
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.6f} — Dropout: {new_p:.3f}")

    writer.close()

    # Prepare test data
    use_earliest_test = False
    if use_earliest_test:
        test_start = start
    else:
        test_start = dt.datetime(2010, 1, 1)
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

    # Forecast next N days via rolling prediction
    real_data = ai_inputs[-prediction_days:, 0].copy()
    future_predictions = []
    for day in range(future_day):
        input_seq = real_data[-prediction_days:].reshape(1, prediction_days, 1)
        with torch.no_grad():
            t_in = torch.from_numpy(input_seq).float().to(device)
            next_pred = model(t_in).cpu().numpy()[0, 0]
        future_predictions.append(next_pred)
        real_data = np.append(real_data, next_pred)
        if (day + 1) % 10 == 0:
            print(f"Predicted day {day+1}/{future_day}")

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_prices = scaler.inverse_transform(future_predictions)

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_day)

    print('\n' + '='*60)
    print('FUTURE PRICE PREDICTIONS:')
    print('='*60)
    for date, price in zip(future_dates, future_predictions_prices):
        print(f"{date.strftime('%Y-%m-%d')}: ${float(price[0]):.2f}")
    print('='*60 + '\n')

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
    for x1, y1, x2, y2 in zip(prediction_dates, actual_prices, prediction_dates_offset, prediction_prices):
        graph.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    graph.axvline(x=last_date, color='orange', linestyle=':', linewidth=2, label='Current Date', alpha=0.7)
    graph.set_title(f'{chart} Price Prediction with {future_day}-Day Forecast')
    graph.set_xlabel('Date')
    graph.legend(loc='upper left')
    graph.grid(True, alpha=0.3)
    fig.autofmt_xdate()

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
