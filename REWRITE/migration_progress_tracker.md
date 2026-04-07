# AI Stock Predictor: Migration & Progress Tracker
**From TensorFlow to PyTorch Plotted**

This document serves as a comprehensive log detailing the evolution of our stock prediction model, mapping the journey from the original TensorFlow implementation to the highly refined, feature-rich PyTorch version (`pytorch_plotted.py`). It highlights major architectural shifts, error resolutions, and the required code snippets for each milestone.

---

## 1. The Original TensorFlow Implementation
Initially, the project was based on TensorFlow/Keras (via scripts like `ai_predict.py` and `tensor.py`). While it achieved base functionality, it presented significant runtime limitations on AMD GPU architectures and proved difficult to iterate on for complex Monte Carlo forecasting and real-time visualization integrations.

**Drivers for Migration:** 
* Insufficient native ROCm support leading to crashes on AMD hardware.
* Difficulties implementing jagged, multi-step predictions.
* The need for dynamic, on-the-fly hyperparameter optimization.

---

## 2. Initial PyTorch Migration & Hardware Compatibility
The initial port to PyTorch (`pytorch_plotted.py`) translated the vanilla LSTM layers but was quickly met with backend engine errors on Windows with AMD GPUs.

### 🐛 Error Fixed: PyTorch Ranger Warning & MIOpen Crash
* **What Happened:** A `RuntimeError: miopenStatusUnknownError` would pop up when the LSTM initialized, breaking the entire script.
* **Why It Was Needed:** PyTorch attempts to route LSTM operations through pure CUDA / cuDNN backends that are unreliable on AMD ROCm drivers for Windows. 
* **The Fix:**
```python
# Disable MIOpen (cudnn) to avoid RuntimeError with LSTM on ROCm
torch.backends.cudnn.enabled = False
```

---

## 3. Bidirectional Architecture Upgrades
To enhance prediction accuracy, the base LSTM was upgraded to a **Bidirectional LSTM**, allowing the network to process historical sequence data in both directions (past-to-present and present-to-past).

### Code Snippet: Bidirectional Layers
```python
self.layers.append(
    nn.LSTM(
        input_size=in_size,
        hidden_size=hidden_size // 2,
        bidirectional=True,
        batch_first=True,
    )
)
```
* **Explanation:** Halving `hidden_size` inside the layer ensures that the concatenated outputs (forward + backward) retain the expected matrix dimensionality. This implementation extracts deeper context from recent swing highs and lows.

---

## 4. Expanding Feature Engineering
Relying solely on "Close" prices was insufficient. The dataset was expanded to calculate indicators directly on `yfinance` data before scaling. 

### 🐛 Error Fixed: ADX Calculation Errors & Feature Gaps
* **What Happened:** Faulty ADX tracking code provided NaN values and messed up scaling constraints.
* **The Fix:** We stripped the faulty logic and implemented stable vectorised Pandas methods for **MACD**, **Bollinger Bands**, **SMA**, and **RSI**.

### Code Snippet: MACD & Bollinger Bands
```python
# Calculate MACD
exp1 = data["Close"].ewm(span=12, adjust=False).mean()
exp2 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = exp1 - exp2
data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

# Calculate Bollinger Bands
data["20_SMA"] = data["Close"].rolling(window=20).mean()
data["Std_Dev"] = data["Close"].rolling(window=20).std()
data["Upper_BB"] = data["20_SMA"] + (data["Std_Dev"] * 2)
data["Lower_BB"] = data["20_SMA"] - (data["Std_Dev"] * 2)

data.ffill(inplace=True)
data.bfill(inplace=True)
```
* **Explanation:** These are fully vectorised and prevent iterative row calculations that drastically slowed down preprocessing. Backfilling controls the NaNs caused by the rolling window.

---

## 5. Terminal UX & Logger Integrations
During the transition to PyTorch loop structures, we encountered minor syntax/identation snags and UI regressions while formatting the console output.

### 🐛 Error Fixed: Disappearing Progress Bars & Unexpected Indentation
* **What Happened:** Standard `print()` statements from optimization callbacks were rewriting the console cursor position, erasing the blue `tqdm` progress bars. Furthermore, missing dictionary wrappers threw unexpected indentation format errors.
* **The Fix:**
```python
# Replaced standard print() overlapping our UI with tqdm.write()
tqdm.write(
    f"Epoch {epoch+1}/{epochs} \u2014 Loss: {epoch_loss:.6f} \u2014 Dropout: {new_p:.3f}"
)
```
We also included the new `TeeLogger` wrapper class which accurately flushed text to `terminal_activity.log` without breaking stdout line carriages for `tqdm`.

---

## 6. Monte Carlo Dropout & Jagged Realism
One of the most profound upgrades was fixing the "averaging-out" problem. Future predictions originally resulted in perfectly smooth, impossible-looking curves. 

### 🐛 Error Fixed: Unrealistic Smooth Price Paths
* **What Happened:** Generating 50 deterministic forward passes and taking the mean smoothed away everyday stock volatility. 
* **The Fix:** We kept dropout active during inference (`model.train()` instead of `model.eval()`) and isolated a single run for the trajectory, using the distribution exclusively to map the standard deviation (confidence interval).

### Code Snippet: Jagged Trajectory Fix
```python
# model.eval() <-- REMOVED: Replaced with model.train() to enforce dropout
model.train() 

# USE SINGLE REALIZATION FOR JAGGED TRAJECTORY
# By extracting index [0], we treat it as a realistic random walk
next_pred = monte_carlo_predictions_for_day[0] 

# However, we still capture the variance of all runs to plot thickness
future_predictions_std.append(
    np.std(monte_carlo_predictions_for_day)
)
```
* **Why It Was Needed:** This explicitly returns a noisy simulation curve while safely calculating the upper/lower bounds of the 95% confidence interval visually represented by the `fill_between` shade.

---

## 7. Data Dictionary Formulation
Finally, to handle Optimizer tracking complexities, we formalized a `Data Dictionary` immediately inside the script, detailing what each variable controls—cementing the code as human-readable and production-ready.

```python
# ==============================================================================
# IMPORTANT VARIABLES EXPLANATION
# ==============================================================================
# prediction_days: The lookback window used as sequence input.
# num_monte_carlo_runs: Number of stochastic forward passes to estimate uncertainty.
# future_predictions_prices: The model's forecasted stock prices for the next `future_day`.
# ...
```

---
*End of Tracker*
