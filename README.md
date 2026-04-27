# Stock Price Prediction using PyTorch

This repository contains a robust machine learning pipeline for forecasting stock prices using a deep Bidirectional Long Short-Term Memory (LSTM) network built with PyTorch.

The primary script for prediction and visualization is `REWRITE/pytorch_plotted.py`.

The project also now includes a PHP/XAMPP web forecast portal in `web/`. The website lets a user log in, search for tickers, queue a new forecast, view saved forecast graphs, and hand long-running training work to the Windows machine that runs the PyTorch scripts.

## Core Features
- **Automated Data Ingestion**: Seamlessly fetches historical market data using `yfinance`.
- **Advanced Technical Indicators**: Real-time feature engineering including 14-day Simple Moving Average (SMA), 14-day Relative Strength Index (RSI), MACD, and 20-day Bollinger Bands.
- **Deep Bidirectional LSTM**: Uses a multi-layered Bidirectional LSTM neural network architecture to capture complex temporal sequence dependencies.
- **Uncertainty Estimation**: Implements Monte Carlo Dropout during inference to generate stochastic price trajectories and calculate accurate 95% confidence intervals.
- **Interactive Prompts**: Features an interactive search to easily look up and select stock tickers from the console.
- **Rich Visualizations**: Generates professional, dark-themed candlestick charts utilizing `mplfinance` and `matplotlib`, overlaid with historical actuals, test-period predicted prices, and future confidence bounds.
- **Extensive Logging**: Custom `TeeLogger` automatically saves all terminal activities to `terminal_activity.log`, and the pipeline supports full `TensorBoard` integration.
- **Web Forecast Queue**: A PHP site stores forecast requests in MySQL, runs one remote training job at a time, imports completed graph assets, and keeps per-ticker job and graph history.
- **Remote Training Bridge**: `web/tools/run_remote_prediction_job.py` runs the separated PyTorch training pipeline, creates summary/detail/residual plots headlessly, and returns a JSON manifest for the PHP worker to import.

## Requirements

Ensure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash
pip install torch numpy pandas matplotlib mplfinance yfinance scikit-learn scipy seaborn tqdm tensorboardX
```

*Note: If you are running on an AMD GPU setup (ROCm), ensure you install the PyTorch version compiled specifically for ROCm.*

## Recent Bug Fixes and Improvements applied to `pytorch_plotted.py`

* **MIOpen (ROCm) LSTM Error**: Added `torch.backends.cudnn.enabled = False` as a workaround to prevent the `RuntimeError: miopenStatusUnknownError` when training Bidirectional LSTMs on AMD GPUs.
* **Disappearing Progress Bars**: Switched to `tqdm.auto` for robust progress bars and explicitly set specific formats (`colour="blue"`, `ascii=False`) to ensure batch training progress is always visible.
* **Pandas `fillna` Incompatibility**: Resolved a `TypeError` regarding the deprecated `method` keyword argument in `fillna()` by updating the syntax to use the more modern `.ffill(inplace=True)` and `.bfill(inplace=True)` methods.
* **Statistically Accurate Confidence Intervals**: Replaced simplified standard deviation scaling with `scipy.stats` functionality to derive mathematically precise Z-scores for the 95% confidence bounds in Monte Carlo dropout predictions.
* **Optuna Database Tracking Failures**: Made structural stability fixes resolving sqlite-related database locking failures when recording hyperparameter tuning trials with Optuna.
* **Unicode Parse Errors**: Cleaned up file parsing and encoding configurations across scripts to resolve 'unexpected unicode' execution errors.
* **Optimizer Investigation**: Confirmed that `torch_optimizer` version 0.3.0 does not include Lion, then compared the available optimizers. Lamb was selected for the LSTM workflow because it is fast, layer-adaptive, and suitable for deep sequence models.
* **Interactive Hang Diagnosis**: Confirmed that some apparent hangs were caused by scripts waiting for interactive input, especially device selection, rather than failing at the optimizer line itself.
* **Website Queue Import**: Added remote job settings, manifest storage, status messages, residual plot support, and a queue debug page for inspecting worker configuration and generated SSH/PowerShell commands.
* **Successful Web Job Evidence**: A SOL-USD web queue run completed as job 28, generating model, prediction CSV, forecast CSV, summary plot, detail plot, residuals plot, and `manifest.json` under `REWRITE/separated/outputs/web_jobs/sol-usd/job_28/`.

## Usage

To run the primary forecasting script from the repository root:

```bash
python REWRITE/pytorch_plotted.py
```

### Execution Flow:
1. **Ticker Selection**: The script will interactively ask you for a stock ticker fragment. You can search or directly input recognizable symbols (e.g. `AAPL`, `TSLA`, `SPY`).
2. **Device Selection**: You will be prompted to select a compute device (`cpu` or `cuda`).
3. **Training & Inference**: The model will construct technical indicators as features, scale the data, train the LSTM over specified epochs, and perform a stochastic backtest using Monte Carlo Dropout to evaluate test error (RMSE/MAE).
4. **Output**: Finally, it predicts future prices and renders the comprehensive forecast graph. 

## Web Portal Usage

The website lives in `web/` and is designed for a Linux XAMPP deployment that can SSH into this Windows project machine.

1. Import `web/sql/schema.sql` into the MySQL/MariaDB database.
2. Apply the extra migration files in `web/sql/` when updating an existing database.
3. Configure database, SSH, remote Python, and remote repo settings in `web/includes/config.php`.
4. Point Apache/XAMPP at the `web/` directory.
5. Use `web/queue_debug.php` after logging in to check queue settings, worker status, known hosts, SSH key paths, and generated remote commands.

When a forecast is requested from the website, the PHP worker starts one queued job, connects to the Windows machine over SSH, runs the Python helper through PowerShell-safe quoting, imports the manifest and PNG assets, and marks the job as completed or failed.
