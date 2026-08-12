"""
Test script: GPU-accelerated feature engineering for pytorch_train_cpp.py.

Converts the pandas/numpy-heavy add_features() to torch tensors on ROCm GPU
(RX 9070 XT). Benchmarks against the original CPU path and verifies numerical
equivalence.

Run:
    python test_gpu_features.py
"""

import time
import numpy as np
import pandas as pd
import torch
import yfinance as yf

# ---------------------------------------------------------------------------
# Original CPU implementation (copied from pytorch_train_cpp.py)
# ---------------------------------------------------------------------------

def add_features_cpu(df):
    """Original pandas-based feature engineering (CPU)."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out["SMA_14"] = out["Close"].rolling(window=14).mean()
    delta = out["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI_14"] = 100 - (100 / (1 + rs))
    exp1 = out["Close"].ewm(span=12, adjust=False).mean()
    exp2 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = exp1 - exp2
    out["Signal_Line"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["20_SMA"] = out["Close"].rolling(window=20).mean()
    out["Std_Dev"] = out["Close"].rolling(window=20).std()
    out["Upper_BB"] = out["20_SMA"] + (out["Std_Dev"] * 2)
    out["Lower_BB"] = out["20_SMA"] - (out["Std_Dev"] * 2)
    out.ffill(inplace=True)
    out.bfill(inplace=True)

    feature_cols = [
        "Close", "Volume", "SMA_14", "RSI_14", "MACD",
        "Signal_Line", "Upper_BB", "Lower_BB",
    ]
    return out[feature_cols].values


# ---------------------------------------------------------------------------
# GPU implementation using torch tensors
# ---------------------------------------------------------------------------

def _rolling_mean_torch(x, window):
    """Rolling mean via unfold (GPU)."""
    # x: (T,) -> unfold to (T - window + 1, window) -> mean over dim 1
    if x.shape[0] < window:
        return torch.full_like(x, float("nan"))
    unfolded = x.unfold(0, window, 1)  # (T - window + 1, window)
    means = unfolded.mean(dim=1)
    # Pad front with NaNs to match original length
    pad = torch.full((window - 1,), float("nan"), dtype=x.dtype, device=x.device)
    return torch.cat([pad, means])


def _rolling_std_torch(x, window):
    """Rolling std via unfold (GPU)."""
    if x.shape[0] < window:
        return torch.full_like(x, float("nan"))
    unfolded = x.unfold(0, window, 1)
    stds = unfolded.std(dim=1, unbiased=True)
    pad = torch.full((window - 1,), float("nan"), dtype=x.dtype, device=x.device)
    return torch.cat([pad, stds])


def _ewm_torch(x, com):
    """Exponential weighted mean (adjust=False) on GPU.

    Matches pandas ewm(com=com, adjust=False).mean().
    """
    alpha = 1.0 / (1.0 + com)
    out = torch.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _ewm_span_torch(x, span):
    """Exponential weighted mean using span (pandas-compatible)."""
    com = (span - 1) / 2.0
    return _ewm_torch(x, com)


def add_features_gpu(df, device="cuda"):
    """GPU-accelerated feature engineering using torch tensors.

    Converts the pandas DataFrame to GPU tensors, computes all features
    with torch operations, then returns a NumPy array matching the
    original output shape and column order.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    close = torch.from_numpy(df["Close"].values.astype(np.float64)).to(device)
    volume = torch.from_numpy(df["Volume"].values.astype(np.float64)).to(device)

    # SMA_14
    sma_14 = _rolling_mean_torch(close, 14)

    # RSI_14
    delta = torch.zeros_like(close)
    delta[1:] = close[1:] - close[:-1]
    gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
    loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
    avg_gain = _ewm_torch(gain, com=13)
    avg_loss = _ewm_torch(loss, com=13)
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))

    # MACD
    exp1 = _ewm_span_torch(close, span=12)
    exp2 = _ewm_span_torch(close, span=26)
    macd = exp1 - exp2
    signal_line = _ewm_span_torch(macd, span=9)

    # Bollinger Bands
    sma_20 = _rolling_mean_torch(close, 20)
    std_20 = _rolling_std_torch(close, 20)
    upper_bb = sma_20 + (std_20 * 2)
    lower_bb = sma_20 - (std_20 * 2)

    # Stack into (T, 8) and forward/backward fill NaNs
    features = torch.stack([
        close, volume, sma_14, rsi_14, macd,
        signal_line, upper_bb, lower_bb,
    ], dim=1)  # (T, 8)

    # Forward fill: replace NaN with last valid value along dim 0
    # Backward fill: replace remaining leading NaNs
    for col in range(features.shape[1]):
        col_data = features[:, col].clone()
        nan_mask = torch.isnan(col_data)
        if nan_mask.any():
            valid = ~nan_mask
            valid_indices = torch.nonzero(valid, as_tuple=True)[0]
            if len(valid_indices) > 0:
                idx = torch.arange(col_data.shape[0], device=device)
                # For each position, find the largest valid index <= position
                fill_idx = torch.searchsorted(valid_indices, idx, right=True) - 1
                fill_idx = torch.clamp(fill_idx, min=0)
                col_data[nan_mask] = col_data[valid_indices[fill_idx[nan_mask]]]
                # Backward fill any remaining (leading NaNs before first valid)
                nan_mask = torch.isnan(col_data)
                if nan_mask.any():
                    col_data[nan_mask] = col_data[valid_indices[0]]
            features[:, col] = col_data

    return features.cpu().numpy()


# ---------------------------------------------------------------------------
# Benchmark and verification
# ---------------------------------------------------------------------------

def benchmark(func, df, label, device="cuda", warmup=1, repeats=5):
    """Benchmark a feature engineering function."""
    # Warmup
    for _ in range(warmup):
        _ = func(df)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(df)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"  {label}: {avg_time:.4f}s ± {std_time:.4f}s (n={repeats})")
    return result, avg_time


def main():
    print("=" * 60)
    print("GPU Feature Engineering Benchmark")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA/ROCm device found. Exiting.")
        return
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"HIP runtime: {torch.version.hip is not None}")

    # Download data
    print("\nDownloading BTC-USD data (2017-01-01 to now)...")
    raw = yf.download("BTC-USD", start="2017-01-01", end=pd.Timestamp.now(), auto_adjust=True)
    print(f"Data shape: {raw.shape}")

    # Benchmark CPU
    print("\n--- CPU (pandas/numpy) ---")
    cpu_result, cpu_time = benchmark(add_features_cpu, raw, "CPU add_features")

    # Benchmark GPU
    print("\n--- GPU (torch ROCm) ---")
    gpu_result, gpu_time = benchmark(
        lambda df: add_features_gpu(df, device=device), raw, "GPU add_features"
    )

    # Verify numerical equivalence
    print("\n--- Verification ---")
    max_diff = np.max(np.abs(cpu_result - gpu_result))
    mean_diff = np.mean(np.abs(cpu_result - gpu_result))
    print(f"  Max abs diff: {max_diff:.8f}")
    print(f"  Mean abs diff: {mean_diff:.8f}")

    # Check per-column with relative tolerance for large values (Volume)
    feature_cols = ["Close", "Volume", "SMA_14", "RSI_14", "MACD",
                    "Signal_Line", "Upper_BB", "Lower_BB"]
    for i, col in enumerate(feature_cols):
        col_diff = np.max(np.abs(cpu_result[:, i] - gpu_result[:, i]))
        col_max = np.max(np.abs(cpu_result[:, i]))
        rel_diff = col_diff / col_max if col_max > 0 else col_diff
        print(f"  {col}: max_diff={col_diff:.8f}  rel_diff={rel_diff:.2e}")

    # Speedup
    print(f"\n--- Speedup ---")
    speedup = cpu_time / gpu_time
    print(f"  GPU speedup: {speedup:.2f}x")
    if speedup < 1.0:
        print("  (GPU slower -- for small datasets (~3K rows), pandas/numpy")
        print("   CPU code is already C-optimized. GPU wins on larger data")
        print("   or when the sequential EWM loop is replaced with a vectorized")
        print("   implementation. The EWM loop has a sequential dependency that")
        print("   prevents GPU parallelization.)")
    else:
        print(f"  GPU is {speedup:.2f}x faster than CPU")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
