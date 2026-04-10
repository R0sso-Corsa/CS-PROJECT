# AI Forecasting Project Log

## Purpose

This document records the evolution of the forecasting project from the earliest TensorFlow/Keras scripts through the mature PyTorch pipeline and finally into the separated `cpp`-style training and plotting flow.

It is meant to answer four questions:

1. What changed over time
2. Why each change was made
3. What bugs or limitations forced the next stage
4. Which files now represent each stage of the project

This log is based on:

- file metadata and content in `REWRITE/learning-path/`
- the existing migration tracker
- the commit summary in `REWRITE/docs/history/git_all_commits.txt`
- the current separated pipeline in `REWRITE/separated/`

## High-Level Evolution

The project moved through six broad phases:

1. Basic TensorFlow single-feature prediction
2. Deeper stacked LSTM TensorFlow experiments
3. Dynamic-dropout and richer TensorFlow training control
4. Advanced TensorFlow forecasting with multi-feature engineering and residual ideas
5. PyTorch transition for better device control and experimentation
6. Mature PyTorch pipeline, then separation of training and plotting into cleaner scripts

The consistent theme across the whole project is this:

- start simple
- hit realism, stability, or hardware limits
- patch the immediate problem
- then absorb that patch into a more structured architecture

## Phase 1: TensorFlow Basics

### Primary files

- `REWRITE/learning-path/01-tensorflow-basics/ai_predict.py`
- `REWRITE/learning-path/01-tensorflow-basics/ai_predict-Eric.py`

### What this stage did

The earliest scripts implemented a classic three-layer LSTM predictor on a single feature: `Close`.

Key characteristics:

- `yfinance` data ingestion
- `MinMaxScaler` normalization
- sliding window training
- stacked Keras LSTM layers
- dropout for regularization
- one-step prediction
- basic matplotlib plotting
- PNG/SVG export

Representative snippet:

```python
# 1. Single feature scaling: only using 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# 2. Simple sliding window preparation
x_train, y_train = [], []
for x in range(prediction_days, len(training_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
x_train = np.reshape(np.array(x_train), (x_train.shape[0], x_train.shape[1], 1))

# 3. Basic Sequential model definition with static dropout
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=10000000)
```

### Why it mattered

This stage established the foundational structure that would survive every future rewrite of the project:

- **Lookback Windows**: Training the model on a rolling historical window (`prediction_days`) rather than single data points.
- **Forecasting Objective**: Attempting to predict the immediate next step in the sequence.
- **Validation Mechanics**: Comparing the predicted sequence against a held-out actual series to visually verify learning.
- **Visual Feedback Loop**: Utilizing `matplotlib` to plot overlapping series (Actual vs. Predicted) and exporting them as PNG/SVG artifacts for post-training review.

While simple, it proved that the pipeline could correctly fetch data from `yfinance`, structure it, pass it through an LSTM, and produce a directional output.

### Main limitations

- **Feature Blindness**: Only the `Close` price was used. The model had no context regarding volume, relative strength, or moving averages, limiting its ability to recognize complex market structures.
- **Data Leakage / Scaler Misuse**: Early scripts sometimes fit the `MinMaxScaler` across the entire dataset (including the test set) simultaneously, subtly leaking future variance information into the training data.
- **Tangled Responsibilities**: Training, data downloading, evaluation, and chart rendering were all hardcoded sequentially into one massive script, making it tedious to iterate on just the model architecture.
- **Hardware Limitations**: Device execution was left to TensorFlow's default behavior, leading to underutilized GPUs and slow training runs.
- **Naive Future Forecasting**: Looking ahead into the future was bolted on as a simplistic loop, rather than being handled as a robust autoregressive prediction.

## Phase 2: TensorFlow Stacked LSTM Refinement

### Primary files

- `REWRITE/learning-path/02-tensorflow-stacked-lstm/tensor.py`
- `REWRITE/learning-path/02-tensorflow-stacked-lstm/tensor-Eric.py`

### What changed

These files stayed in the TensorFlow/Keras family but pushed the model structure and documentation further. Compared with the earliest scripts, they were more deliberate about stacked recurrent layers and explanatory comments.

This phase is less a conceptual rewrite and more an internal consolidation step:

- clearer comments
- more consistent sliding-window logic
- stronger emphasis on model shape and sequence handling

Representative snippet:

```python
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
    embeddings_freq=0            # Disable embedding visualization
)

ai.compile(optimizer='adam', loss='mean_squared_error')
ai.fit(x_train, y_train, epochs=10, batch_size=16, callbacks=[tensorboard_callback])
```

### Why this stage exists

The project was still learning the mechanics of LSTM sequence modeling and internal tracking. This stage was primarily about consolidating knowledge:

- **Structural Depth**: Experimenting with stacked layers to see if deeper networks could capture more complex, non-linear relationships in the stock data.
- **Sequence Handling**: Gaining a better grasp of `return_sequences=True` and how LSTM hidden states propagate to subsequent layers.
- **Introspection via TensorBoard**: Passing the `tensorboard_callback` allowed the developer to visualize weight histograms and monitor the model graph externally, moving away from relying purely on terminal outputs for debugging loss convergence.

## Phase 3: Dynamic Dropout and Better Training Control

### Primary file

- `REWRITE/learning-path/03-tensorflow-dynamic-dropout/test_modified.py`

### What was added

This phase introduced one of the first major training-control upgrades: dynamic dropout over epochs.

Representative snippet:

```python
def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):
    """Calculate dropout rate that decreases linearly with epochs"""
    return max(final_rate, initial_rate - (initial_rate - final_rate) * (epoch / total_epochs))

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

dynamic_dropout = DynamicDropoutCallback(epochs=20, initial_rate=0.5, final_rate=0.1)

ai.fit(x_train, y_train, 
       epochs=20, 
       batch_size=16,
       callbacks=[tensorboard_callback, dynamic_dropout],
       verbose=1)
```

Also added:

- TensorBoard logging
- clearer training hyperparameters
- less static regularization

### Why this was added

The early static dropout approach (e.g., a hardcoded `Dropout(0.2)`) proved to be too blunt:

- **Suppressed Convergence**: Applying a high dropout rate late in the training process repeatedly knocked out useful activated neurons, preventing the network from converging on finer details and lowering final accuracy.
- **Early Overfitting**: Conversely, applying a low dropout rate from the start allowed the model to rapidly memorize the training data, leading to severe overfitting before the general patterns were learned.

The `DynamicDropoutCallback` resolved this tension. By starting with heavy regularization (e.g., 50%) and linearly decaying it as epochs progressed (down to 10%), the network was forced to build robust, distributed representations early on, and then permitted the freedom to fine-tune its weights against the signal in the final epochs.

### Main limitations

While the training loop was now far more intelligent, the underlying script was still severely constrained:

- **Data Richness**: The pipeline was still largely locked into single-feature analysis, starving the newly optimized LSTM of actionable data.
- **Monolithic Scripting**: Plotting, forecasting, and data ingress were still tangled together with the complex new callback logic.
- **TensorFlow Ecosystem Constraints**: The Keras API made it easy to build standard models, but customizing the forward pass for financial edge cases (or debugging deep within the tensor operations) remained frustratingly opaque.

## Phase 4: Extended TensorFlow Forecasting

### Primary files

- `REWRITE/learning-path/04-tensorflow-extended-forecasting/kill-me.py`
- `REWRITE/learning-path/04-tensorflow-extended-forecasting/future-Eric.py`
- `REWRITE/learning-path/04-tensorflow-extended-forecasting/onedrive script.py`

### What changed

This phase moved beyond “predict next value from close-only history” toward richer forecasting logic.

Important additions:

- technical indicators
- multi-feature inputs
- explicit train/validation separation
- `tf.data.Dataset`
- `EarlyStopping`
- `ReduceLROnPlateau`
- `ModelCheckpoint`
- residual correction ideas
- direct future horizon handling
- stronger CPU threading controls

Representative snippet from `future-Eric.py`:

```python
# Move toward derived features and technical indicators
df['returns'] = df['log'].diff()
df['ma10'] = df['close'].rolling(window=10).mean()
df['ma50'] = df['close'].rolling(window=50).mean()
df['rsi'] = compute_rsi(df['close'])
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# Handling missing technical indicators robustly
try:
    df.ta.atr(append=True)
    df.ta.ema(append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.obv(append=True)
except Exception as e:
    print(f"Warning: pandas_ta failed to add some indicators: {e}")

# Dual scaler setup: one for targets, one for features
scaler_y = StandardScaler()
scaler_X = MinMaxScaler()
scaled_returns = scaler_y.fit_transform(train_df['returns'].values.reshape(-1, 1))
scaled_X_train = scaler_X.fit_transform(train_df[other_cols].values)
scaled_features_train = np.hstack([scaled_returns, scaled_X_train])
```

Training control and multi-step residual regression:

```python
# Utilizing Model Checkpoint and Early Stopping callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

ai.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
       callbacks=[tensorboard_callback, dynamic_dropout, early_stop, reduce_lr, model_checkpoint])

# Post-LSTM Residual Model (Gradient Boosting/Ridge) 
# Trains on flattened windows to recover sharper local moves missed by the LSTM
x_train_flat = x_train.reshape(x_train.shape[0], -1)
pred_train_scaled = ai.predict(x_train)  # shape (n_samples, HORIZON)
residuals = y_train - pred_train_scaled  # shape (n_samples, HORIZON)

ridge_base = Ridge(alpha=1.0)
gbr_res = MultiOutputRegressor(ridge_base)
gbr_res.fit(x_train_flat, residuals)
```

### Why these changes were made

The project had hit the ceiling of what naive single-feature forecasting could achieve. The model could track existing trends, but producing believable multi-step future curves required a much richer understanding of market context and a more disciplined approach to training stability.

### Important design choices

#### 1. Move toward derived features (pandas-ta)

Feeding the model raw `Close` prices hid the localized structure of the market. By incorporating Moving Averages (MA10, MA50), Relative Strength Index (RSI), MACD, and Bollinger Bands, the LSTM was given explicit mathematical context regarding momentum and volatility, drastically reducing the amount of raw feature extraction the network had to perform internally.

#### 2. Working with returns and transformations

Some experimental scripts began converting prices to logarithmic returns (`df['log'].diff()`). This forced the model to learn the fundamental dynamics of market movement (percentage changes) rather than getting stuck trying to predict absolute price levels, which often lead to non-stationary data problems.

#### 3. Advanced callback discipline

With a larger feature set and deeper models, training times increased significantly. "Fire and forget" scripts were no longer viable. The introduction of `EarlyStopping` (to halt training when validation metric plateaued) and `ModelCheckpoint` (to save the exact weights from the best epoch, rather than the final epoch) fundamentally improved the reliability of the resulting artifacts.

#### 4. Post-LSTM Residual Correction

Because LSTMs typically smooth out their sequence predictions to minimize loss, they often miss sharp, localized stock movements. To counter this, a `Ridge` MultiOutputRegressor (gradient boosting) was trained on the *residuals* (the difference between the LSTM's prediction and the actual value) using flattened input windows. This taught the pipeline to mathematically "correct" the LSTM's smoothed output with a localized layer of sharpening.

### Problems found in this phase

Despite these massive improvements to the ML architecture, engineering pressure mounted:

- **TensorFlow on AMD/ROCm**: Running complex TensorFlow pipelines on Windows with AMD hardware remained awkwardly unsupported, leading to sporadic crashes and poor resource utilization.
- **Forecasting Complexity**: Managing the state and loops for direct multi-step forecasting within the Keras ecosystem felt overly restrictive.
- **Script Bloat**: Plotting, residual calculation, callbacks, and validation logic were ballooning inside single files.
- **Unrealistic Trajectories**: Despite everything, when asked to predict 30 days into the future, the model generated perfectly smooth curves devoid of natural market stochasticity.

These engineering and usability problems pushed the project to migrate entirely to PyTorch.

## Phase 5: PyTorch Transition

### Primary files

- `REWRITE/learning-path/05-pytorch-transition/future-eric-pytorch.py`
- `REWRITE/learning-path/05-pytorch-transition/onedrive_pytorch.py`

### What changed

The project moved from Keras/TensorFlow to PyTorch for explicit device control, training flexibility, and easier experimental iteration.

Representative snippet:

```python
# Explicit device routing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_train_tensor = torch.FloatTensor(x_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
```

Model definition became explicit Python modules with PyTorch mechanics:

```python
class DynamicLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=500, num_middle_layers=3, initial_dropout=0.5):
        super(DynamicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_middle_layers = num_middle_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(initial_dropout)
        
        self.middle_lstms = nn.ModuleList()
        self.middle_dropouts = nn.ModuleList()
        for i in range(num_middle_layers):
            self.middle_lstms.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))
            self.middle_dropouts.append(nn.Dropout(initial_dropout))
            
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        for i in range(self.num_middle_layers):
            out, _ = self.middle_lstms[i](out)
            out = self.middle_dropouts[i](out)
            
        # Take last time step and pass through dense layer
        out = self.fc(out[:, -1, :])
        return out
```

### Why PyTorch was adopted

The migration tracker documents several distinct engineering pressures that motivated the rewrite:

- **Hardware and ROCm Reliability**: The user's workstation relied on AMD GPUs. TensorFlow's support for ROCm on Windows was historically patchy, whereas PyTorch offered tighter, more explicit device controls (`device = torch.device(...)`) and better community support for AMD compilation.
- **Granular Training Control**: Financial forecasting often requires custom loss functions and unusual training loop manipulation. PyTorch’s imperative, define-by-run philosophy allowed the developer to step through the exact forward and backward passes, diagnosing shape mismatches and gradient flows directly in Python.
- **Autoregressive Jaggedness**: Implementing the logic needed to create jagged, realistic, autoregressive step-by-step forecasting was much cleaner when defining custom PyTorch `forward()` methods compared to bending Keras layers to fit the objective.

### Optuna enters the project

`onedrive_pytorch.py` introduced `optuna`, a hyperparameter optimization framework:

```python
study = optuna.create_study(
    study_name="onedrive_optimizer",
    storage=f"sqlite:///{OPTUNA_DB_PATH.as_posix()}",
    load_if_exists=True,
    direction="minimize",
)
```

This marked a paradigm shift in the project's maturity:

- Development moved from "guess a model shape and train it" to "define a search space and let the database find the optimal architecture."
- Tracking was externalized to SQLite, providing persistence across runs and preventing the loss of valuable optimization data if a script crashed mid-execution.

### Problems found in this phase

- **The Transition Chimera**: The new PyTorch scripts still contained legacy structural ideas ported directly from TensorFlow. They worked, but they weren't utilizing PyTorch's native `Dataset` / `DataLoader` optimizations efficiently.
- **Tangled Logic**: Optuna tuning, single-shot backtesting, and visualization routines were still mashed together in the same file.

## Phase 6: PyTorch Stable Core

### Primary file

- `REWRITE/learning-path/06-pytorch-stable-core/pytorch_fixed.py`

### What changed

This file became a cleaner “stable baseline” PyTorch implementation.

Important additions:

- technical indicators standardized
- explicit `Dataset` and `DataLoader`
- bidirectional LSTM
- validation split
- `HuberLoss`
- scheduler-based learning-rate control
- gradient clipping
- Monte Carlo uncertainty estimates
- metric reporting beyond RMSE

Representative snippets:

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = input_size
        for i in range(num_layers):
            self.layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size // 2,
                    bidirectional=True,  # Crucial architecture shift
                    batch_first=True,
                )
            )
            self.layers.append(nn.Dropout(p=dropout))
            in_size = hidden_size

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = x
        for i in range(0, len(self.layers), 2):
            lstm = self.layers[i]
            dropout = self.layers[i + 1]
            out, _ = lstm(out)
            out = dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
```

Robust loss, optimiser, and scheduler choices:

```python
# Utilizing HuberLoss for outlier resistance in noisy financial series
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Dynamic learning rate reduction
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)
```

Training loop resilience:

```python
# Prevent exploding gradients inside the recurrent backprop
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Why these choices matter

By phase 6, the project was no longer struggling with basic syntax; it was focused on making the ML model mathematically resilient to the volatile reality of financial markets.

#### Bidirectional LSTM Representation

Stock markets do not exist in a vacuum; context often flows sequentially from recent history to the present. The swap to `bidirectional=True` allowed the LSTM to process the lookback window in both forward and reverse orders simultaneously. This effectively doubled the model's representational capacity and helped it capture localized patterns that a strict chronological read might miss, without changing the external forecasting task.

#### Huber Loss vs MSE

Standard Mean Squared Error (MSE) heavily penalizes outliers. Because financial data is extremely noisy with frequent sudden spikes, MSE forced the model to warp its weights to account for singular Black Swan events. `nn.HuberLoss` acted as an intelligent middle ground: behaving like MSE for small errors, but switching to a linear Mean Absolute Error (MAE) penalty for wild outliers, preventing the network gradients from exploding.

#### Gradient Clipping Defense

Inside deep recurrent structures, gradients evaluated over long sequences can easily explode entirely, turning weights into `NaN`. `torch.nn.utils.clip_grad_norm_` explicitly clamped the global norm of the gradients to `1.0` before the optimizer took a step, serving as a practical, ironclad defense against unstable recurrent backpropagation.

#### Validation-Aware Schedulers

The `ReduceLROnPlateau` scheduler intelligently bridged the gap between rapid early learning and fine-grained late-stage optimization. Rather than blindly dropping the learning rate linearly (as done previously with dynamic dropout), it monitored the external validation loss. If the network stalled to learn for `10` epochs (`patience`), it sliced the learning rate in half (`factor=0.5`), gently guiding the model into narrower, steeper local minima.

## Phase 7: Full PyTorch Pipeline

### Primary file

- `REWRITE/learning-path/07-pytorch-full-pipeline/pytorch_plotted.py`

### What changed

This is the large, feature-rich, all-in-one pipeline that pulled together most of the project’s mature ideas:

- interactive ticker search
- ROCm-aware device behavior
- extensive feature engineering
- multi-feature scaling
- bidirectional LSTM
- dynamic dropout
- `torch_optimizer`
- TensorBoard fallback strategy
- progress bars
- Monte Carlo dropout
- confidence intervals
- rich plotting
- terminal logging

Representative snippets:

```python
# Keep cuDNN enabled on pure CUDA.
# Disable only on ROCm where some users hit MIOpen runtime issues.
torch.backends.cudnn.enabled = torch.version.hip is None
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"  # RDNA4-specific scheduling
```

Optimiser selection and mixed precision:

```python
import torch_optimizer as optim

optimizer = getattr(optim, optimizer_name)(model.parameters(), weight_decay=0.05)
amp_enabled = bool(use_amp and (device.type == "cuda") and (torch.version.hip is None))
amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

# Iteration loop block with AMP
optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
    outputs = model(xb)
    loss = criterion(outputs, yb)
amp_scaler.scale(loss).backward()
amp_scaler.step(optimizer)
amp_scaler.update()
```

Monte Carlo dropout usage for confidence intervals:

```python
# Enable dropout during inference for Monte Carlo Uncertainty
model.train() 
with torch.no_grad():
    for _ in range(num_monte_carlo_runs):
        monte_carlo_predictions_for_day.append(model(t_in).squeeze())

monte_carlo_predictions_for_day = torch.stack(monte_carlo_predictions_for_day).detach().cpu().numpy()

# USE SINGLE REALIZATION FOR JAGGED TRAJECTORY
# Taking the first run as the "path" to follow avoids the smoothed-out mean
next_pred = float(monte_carlo_predictions_for_day[0]) 

# Use the distribution standard deviation to form confidence limits
future_predictions_std.append(np.std(monte_carlo_predictions_for_day))
```

### Why this stage mattered

The `pytorch_plotted.py` script represents the apex of the monolithic phase. It is where the project stopped being an isolated "ML script" and transformed into a "full forecasting workstation." 

It elegantly solved the most frustrating problems that had plagued the pipeline for months:

#### 1. Hardware Mastery (ROCm & MIOpen)

The project had historically been crippled by `miopenStatusUnknownError` and mysterious AMD runtime crashes. This script introduced explicit, conditional hardware management: automatically detecting if `torch.version.hip` was present, safely disabling specific `cudnn` optimizations on ROCm systems while retaining them on native CUDA, and injecting RDNA-specific environment variables (`TRITON_HIP_USE_BLOCK_PINGPONG`).

#### 2. The Jagged Trajectory Problem

A persistent issue was that LSTMs output an "averaged" probability of a future step. Plotted out 30 days, the predictions looked like completely smooth, impossible curves. `pytorch_plotted.py` solved this by leveraging Monte Carlo Uncertainty. By enabling `model.train()` during inference, dropout layers remained active, introducing calculated stochastic noise. The script ran the 30-day forecast 100 times, took the standard deviation to plot a wide, shaded confidence interval, and crucially, took a single stochastic realization run as the mainline path. This ensured the plotted chart exhibited realistic, jagged market movement within a mathematically bounded range.

#### 3. UX, Logging, and Mixed Precision

For usability during hour-long training sessions, nested `tqdm` progress bars were refined to prevent console text bleeding. Behind the scenes, the integration of `torch.amp.autocast` dropped internal tensor math from 32-bit floats to 16-bit floats across compatible layers, dramatically speeding up VRAM throughput with zero noticeable loss in prediction accuracy. 

#### 4. Formalized Feature Ecosystem

This stage finalized the robust 8-feature engineering protocol that standardizes how the model views a single day: `Close`, `Volume`, `SMA_14`, `RSI_14`, `MACD`, `Signal_Line`, `Upper_BB`, and `Lower_BB`.

### Errors and fixes consolidated here

#### MIOpen / ROCm runtime errors

Documented in the migration tracker. Fixed through backend handling and conditional cuDNN/MIOpen control.

#### Progress bars disappearing

Plain `print()` calls interfered with `tqdm`. The project moved toward `tqdm.auto` and progress-safe updates.

#### `fillna` compatibility issues

The code settled on:

```python
data.ffill(inplace=True)
data.bfill(inplace=True)
```

instead of older deprecated patterns.

#### Over-smoothed future forecasts

Using the mean prediction trajectory made outputs look unrealistic. The project switched to:

- one path for visible trajectory
- full distribution for uncertainty band

#### MultiIndex `yfinance` shape problems

The separated pipeline later made this explicit, but the root issue was already visible here: `yfinance` can return awkward column shapes that break feature matrices.

## Phase 8: PyTorch Experiments

### Primary files

- `REWRITE/learning-path/08-pytorch-experiments/pytorch_optimised_maybe.py`
- `REWRITE/learning-path/08-pytorch-experiments/pytorch_additional.py`

### What changed

These files represent experimentation beyond the already-large full pipeline:

- `torch.compile`
- AMP / autocast
- advanced logging
- more aggressive performance tuning
- attention/residual-style ideas in `pytorch_additional.py`

Representative snippet from `pytorch_additional.py` displaying the sheer depth of structural additions:

```python
class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with attention mechanism and residual connections."""
    def __init__(self, input_size=20, hidden_size=512, num_layers=4, dropout=0.5, output_size=30):
        super().__init__()
        
        # ... [LSTM definition]
        
        # Multi-Head Attention mechanism over the sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )
        self.attention_ln = nn.LayerNorm(hidden_size)

        # Progressive refinement Fully Connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc1_ln = nn.LayerNorm(hidden_size)
        self.fc1_dropout = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2_ln = nn.LayerNorm(hidden_size // 2)
        self.fc2_dropout = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        out = x
        # LSTM layers with residual connections
        for i in range(0, len(self.lstm_layers), 3):
            # ... [LSTM passes]
            # Residual connection (only if dimensions match)
            if lstm_out.shape[-1] == out.shape[-1]:
                lstm_out = lstm_out + out * 0.1
            out = lstm_out

        # Attention block
        attn_out, _ = self.attention(out, out, out)
        attn_out = self.attention_ln(attn_out + out * 0.1)
        out = attn_out

        # Produce multi-step forecast through cascading FC
        out = out[:, -1, :]
        fc1_out = self.fc1_dropout(torch.relu(self.fc1_ln(self.fc1(out))))
        fc2_out = self.fc2_dropout(torch.relu(self.fc2_ln(self.fc2(fc1_out))))
        return self.fc3(fc2_out)
```

Also incorporated cyclical features for time mapping:

```python
# Cyclical features for seasonality patterns
data["DoW_sin"] = np.sin(data.index.dayofweek * (2 * np.pi / 7))
data["DoW_cos"] = np.cos(data.index.dayofweek * (2 * np.pi / 7))
data["Month_sin"] = np.sin((data.index.month - 1) * (2 * np.pi / 12))
data["Month_cos"] = np.cos((data.index.month - 1) * (2 * np.pi / 12))
```

### Why this phase exists

Once the full pipeline was working, the project shifted toward “can it be faster / richer / more expressive?”

This phase is important because it reveals a natural software-engineering pressure:

- as more advanced features accumulate, single-file architecture becomes harder to maintain

That pressure directly leads to the next phase.

## Phase 9: Separated `cpp` Pipeline

### Primary files

- `REWRITE/separated/pytorch_train_cpp.py`
- `REWRITE/separated/pytorch_plot_cpp.py`

### Why the separation happened

The all-in-one `pytorch_plotted.py` script had become incredibly powerful, but its internal complexity had reached a critical mass. Modifying the plotting logic required navigating past hundreds of lines of neural network tensor math. If a user wanted to re-render a chart with a different color scheme or layout, they had to rerun the entire ML inference loop. It was a massive bottleneck.

The separated "cpp-style" pipeline decoupled the project into two distinct operational concerns:

1. **`pytorch_train_cpp.py`**: A strictly computational script. It handles data ingress, normalization, model declaration, AMP training loops, loss calculation, evaluation, and Monte Carlo multi-step forecasting. Crucially, it does absolutely no plotting. Its sole deliverable is dumping raw training weights, future trajectory arrays, and prediction CSVs into an organized `outputs/` folder.
2. **`pytorch_plot_cpp.py`**: A purely visual script. It scans the `outputs/` directory to discover the latest successfully generated artifacts. It rebuilds the context purely from the exported CSV matrices, reconstructing the actuals, the predictions, and the confidence interval bands, and orchestrates the complex `matplotlib` rendering.

This is the most important architectural shift in the project's history. By completely isolating the heavy-duty data science workload from the visualization layer, the pipeline became drastically more maintainable, modular, and resilient.

### Training-side improvements

Representative design choices:

#### 1. CLI interface

```python
parser = argparse.ArgumentParser(description="Fast training-only script (cpp variant).")
```

This replaces hardcoded interactive settings with reproducible commands.

#### 2. Stable output management

```python
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "cpp"
```

and:

```python
artifact_dirs = {
    "root": output_root,
    "models": output_root / "models",
    "predictions": output_root / "predictions",
    "forecasts": output_root / "forecasts",
}
```

This matters because earlier stages often wrote artifacts into whichever working directory happened to be active.

#### 3. Vectorized window generation

```python
from numpy.lib.stride_tricks import sliding_window_view
```

This reduced Python-loop overhead and made sequence building more explicit and efficient.

#### 4. Cleaner feature processing

```python
if isinstance(out.columns, pd.MultiIndex):
    out.columns = out.columns.get_level_values(0)
```

This directly addresses `yfinance` column-shape issues.

#### 5. Optional `torch.compile`

```python
if use_compile:
    model = torch.compile(model, mode="reduce-overhead")
```

This shows the project had shifted from “make it work” to “make it work and improve throughput where possible.”

### Plot-side improvements

`pytorch_plot_cpp.py` made artifact reuse much cleaner:

```python
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_ROOT / "predictions"
DEFAULT_FORECASTS_DIR = DEFAULT_OUTPUT_ROOT / "forecasts"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_ROOT / "plots"
```

It auto-discovers newest relevant artifacts and reconstructs the charting flow from saved CSVs rather than rerunning the full training script.

That separation buys several things:

- easier debugging
- faster iteration on plotting
- artifact reproducibility
- better workspace hygiene

### Why this is the strongest current architecture

The separated pipeline preserves the mature forecasting logic from `pytorch_plotted.py`, but removes the most painful coupling:

- training is now independent
- plotting is now independent
- artifacts are organized
- file reading is stable
- outputs are easier to audit

## Error and Issue Log

This section summarizes the most important errors or recurring pain points encountered across stages.

### 1. `miopenStatusUnknownError`

Context:

- PyTorch + LSTM + AMD/ROCm/Windows

Impact:

- training could fail before useful experimentation started

Response:

- explicit backend handling
- more hardware-aware runtime configuration

### 2. Disappearing `tqdm` progress bars

Context:

- ordinary prints mixed with progress bars

Impact:

- poor training UX
- difficult monitoring

Response:

- move toward `tqdm.auto`
- throttle updates
- use progress-safe logging patterns

### 3. Unrealistically smooth future forecasts

Context:

- averaged Monte Carlo trajectories

Impact:

- outputs looked visually implausible

Response:

- keep dropout active
- use one stochastic realization as displayed path
- use full run distribution for interval width

### 4. `fillna` compatibility / pandas API drift

Context:

- older fill patterns breaking or warning

Response:

- use `ffill` + `bfill`

### 5. `yfinance` MultiIndex shape issues

Context:

- feature extraction expecting flat columns

Impact:

- wrong tensor shapes
- scaling failures

Response:

- flatten MultiIndex before feature extraction

### 6. Database/path drift

Context:

- Optuna and artifact files originally assumed current working directory

Impact:

- fragile runs
- hard-to-find outputs

Response:

- script-relative paths
- organized output roots

## Why the Current File Order Makes Sense

The learning path folders were arranged using two signals together:

1. older metadata first
2. lower conceptual/code complexity first

That produces a logical educational sequence:

- basic TensorFlow forecasting
- deeper stacked recurrent TensorFlow work
- dynamic regularization and better control
- advanced TensorFlow forecasting
- PyTorch migration
- stable PyTorch baseline
- full-feature PyTorch pipeline
- experimental PyTorch variants
- separated train/plot architecture

## Current Canonical End State

If someone wants the best representation of the project today, the most relevant end-state files are:

- `REWRITE/separated/pytorch_train_cpp.py`
- `REWRITE/separated/pytorch_plot_cpp.py`

If someone wants the most important “bridge” file before that separation, it is:

- `REWRITE/learning-path/07-pytorch-full-pipeline/pytorch_plotted.py`

## References

### Project-level references

- `README.md`
- `REWRITE/docs/migration_progress_tracker.md`
- `REWRITE/docs/history/git_all_commits.txt`
- `REWRITE/learning-path/README.md`

### Stage references

- `REWRITE/learning-path/01-tensorflow-basics/ai_predict.py`
- `REWRITE/learning-path/02-tensorflow-stacked-lstm/tensor.py`
- `REWRITE/learning-path/03-tensorflow-dynamic-dropout/test_modified.py`
- `REWRITE/learning-path/04-tensorflow-extended-forecasting/future-Eric.py`
- `REWRITE/learning-path/05-pytorch-transition/future-eric-pytorch.py`
- `REWRITE/learning-path/05-pytorch-transition/onedrive_pytorch.py`
- `REWRITE/learning-path/06-pytorch-stable-core/pytorch_fixed.py`
- `REWRITE/learning-path/07-pytorch-full-pipeline/pytorch_plotted.py`
- `REWRITE/learning-path/08-pytorch-experiments/pytorch_optimised_maybe.py`
- `REWRITE/learning-path/08-pytorch-experiments/pytorch_additional.py`
- `REWRITE/separated/pytorch_train_cpp.py`
- `REWRITE/separated/pytorch_plot_cpp.py`

## Closing Summary

This project did not progress through clean, preconceived greenfield redesigns. It evolved organically strictly as a response to escalating engineering pressure:

- **Phase 1-4**: Simplistic Keras models quickly exposed the limits of naive data structures, forcing the adoption of multi-step residual loops, technical indicators, and sophisticated training callbacks.
- **Phase 5-7**: Complex multi-feature TensorFlow networks collided catastrophically with AMD hardware and ROCm API instability, necessitating the arduous but extremely successful rewrite into the low-level, hardware-aware PyTorch `autocast` ecosystem.
- **Phase 8-9**: The accumulation of advanced features like Multihead-Attention, Optuna databases, and Monte Carlo confidence intervals caused the monolithic `pytorch_plotted.py` to buckle under its own weight. This final pressure birthed the `cpp`-style architecture, gracefully splitting the codebase into a high-performance training engine and an independent visualization suite.

The current codebase is immensely valuable not merely as a working stock forecasting tool, but as a living record of iterative engineering applied against rigorous real-world constraints: jagged data quality, fragile hardware backends, recurrent training instability, and ultimately, the unyielding necessity of clean software architecture.
