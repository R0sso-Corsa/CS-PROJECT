import argparse
import datetime as dt
import glob
import os
import re

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf


def _default_predictions_csv():
    """Newest *_predictions_cpp_*.csv under cwd and this script's directory."""
    roots = {os.getcwd(), os.path.dirname(os.path.abspath(__file__))}
    candidates = []
    for root in roots:
        pattern = os.path.join(root, "*_predictions_cpp_*.csv")
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _infer_ticker_from_predictions_path(path):
    """Parse ticker from ``{TICKER}_predictions_cpp_{stamp}.csv``."""
    base = os.path.basename(path)
    if "_predictions_cpp_" in base:
        return base.split("_predictions_cpp_")[0]
    return None


def _default_future_csv(predictions_path, ticker):
    """Newest matching ``{ticker}_future_*d_cpp_*.csv`` beside predictions (or cwd)."""
    roots = {
        os.path.dirname(os.path.abspath(predictions_path)) or ".",
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
    }
    candidates = []
    prefix = re.escape(ticker) if ticker else r"[^_]+"
    pat = re.compile(rf"^{prefix}_future_\d+d_cpp_.*\.csv$")
    for root in roots:
        for p in glob.glob(os.path.join(root, "*_future_*d_cpp_*.csv")):
            if pat.match(os.path.basename(p)):
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _as_1d_close(s):
    """Close series or DataFrame column -> 1d numpy float."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    a = np.asarray(s.values, dtype=float)
    return a.reshape(-1)


def _info_names(ticker_symbol):
    try:
        info = yf.Ticker(ticker_symbol).info or {}
        long_name = info.get("longName") or ticker_symbol
        short_name = info.get("shortName") or ticker_symbol
        return long_name, short_name
    except Exception:
        return ticker_symbol, ticker_symbol


def main():
    parser = argparse.ArgumentParser(
        description="Dual-window charts from pytorch_train_cpp CSV outputs (mirrors pytorch_plotted)."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to *_predictions_cpp_*.csv (default: newest in cwd or REWRITE/)",
    )
    parser.add_argument(
        "--future",
        default=None,
        help="Path to *_future_*d_cpp_*.csv (default: newest matching ticker beside predictions)",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Yahoo symbol (default: inferred from predictions filename)",
    )
    parser.add_argument(
        "--test-start",
        default="2025-06-01",
        help="Start of test OHLC window (match pytorch_train_cpp evaluation).",
    )
    parser.add_argument(
        "--optimizer-tag",
        default="CPP",
        help="Label used in saved PNG filenames.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save PNGs to the predictions CSV directory",
    )
    args = parser.parse_args()

    csv_path = args.csv or _default_predictions_csv()
    if not csv_path or not os.path.isfile(csv_path):
        print(
            "No CSV specified and no *_predictions_cpp_*.csv found.\n"
            "Run pytorch_train_cpp.py first or pass --csv."
        )
        raise SystemExit(1)

    if args.csv is None:
        print(f"Using predictions CSV: {csv_path}")

    ticker = args.ticker or _infer_ticker_from_predictions_path(csv_path)
    if not ticker:
        print("Could not infer --ticker from filename; pass --ticker explicitly.")
        raise SystemExit(1)

    future_path = args.future or _default_future_csv(csv_path, ticker)
    if future_path and os.path.isfile(future_path):
        print(f"Using future CSV: {future_path}")
    else:
        future_path = None
        print(
            "No future forecast CSV found; primary chart omits future/CI overlay; "
            "second window is skipped (run pytorch_train_cpp.py to generate *_future_*d_cpp_*.csv)."
        )

    pred_df = pd.read_csv(csv_path)
    required = {"Date", "Predicted", "Actual"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

    pred_df["Date"] = pd.to_datetime(pred_df["Date"])
    pred_df = pred_df.sort_values("Date").reset_index(drop=True)

    chart_name_plot, chart_name_plot_short = _info_names(ticker)

    test_start = pd.Timestamp(args.test_start)
    test_end = dt.datetime.now()
    test_raw = yf.download(ticker, start=test_start, end=test_end, auto_adjust=True)
    if test_raw.empty:
        raise RuntimeError(f"No Yahoo data for {ticker} in test window {test_start} .. {test_end}")

    test_data = test_raw.copy()
    test_data["SMA_14"] = test_data["Close"].rolling(window=14).mean()
    delta_test = test_data["Close"].diff(1)
    gain_test = delta_test.where(delta_test > 0, 0)
    loss_test = -delta_test.where(delta_test < 0, 0)
    avg_gain_test = gain_test.ewm(com=13, adjust=False).mean()
    avg_loss_test = loss_test.ewm(com=13, adjust=False).mean()
    rs_test = avg_gain_test / avg_loss_test
    test_data["RSI_14"] = 100 - (100 / (1 + rs_test))
    exp1_test = test_data["Close"].ewm(span=12, adjust=False).mean()
    exp2_test = test_data["Close"].ewm(span=26, adjust=False).mean()
    test_data["MACD"] = exp1_test - exp2_test
    test_data["Signal_Line"] = test_data["MACD"].ewm(span=9, adjust=False).mean()
    test_data["20_SMA"] = test_data["Close"].rolling(window=20).mean()
    test_data["Std_Dev"] = test_data["Close"].rolling(window=20).std()
    test_data["Upper_BB"] = test_data["20_SMA"] + (test_data["Std_Dev"] * 2)
    test_data["Lower_BB"] = test_data["20_SMA"] - (test_data["Std_Dev"] * 2)
    test_data.ffill(inplace=True)
    test_data.bfill(inplace=True)

    # Match pytorch_train_cpp: predictions row count equals test window length from start
    n = min(len(test_data), len(pred_df))
    if len(test_data) != len(pred_df):
        print(
            f"Note: trimming test ({len(test_data)}) and predictions ({len(pred_df)}) to {n} rows."
        )
    test_data = test_data.iloc[:n].copy()
    pred_df = pred_df.iloc[:n].copy()

    prediction_dates = pd.DatetimeIndex(pred_df["Date"])
    prediction_dates_offset = prediction_dates
    prediction_prices = np.asarray(pred_df["Predicted"].values, dtype=float).reshape(
        -1, 1
    )
    actual_prices = _as_1d_close(test_data["Close"]).reshape(-1, 1)

    future_df = None
    future_dates = pd.DatetimeIndex([])
    future_predictions_prices = np.array([]).reshape(0, 1)
    future_predictions_lower = np.array([])
    future_predictions_upper = np.array([])
    future_day = 0

    if future_path:
        future_df = pd.read_csv(future_path)
        for col in ("Date", "Predicted_Price"):
            if col not in future_df.columns:
                raise ValueError(f"Future CSV missing column {col!r}: {future_path}")
        future_df["Date"] = pd.to_datetime(future_df["Date"])
        future_df = future_df.sort_values("Date")
        future_dates = pd.DatetimeIndex(future_df["Date"])
        future_predictions_prices = np.asarray(
            future_df["Predicted_Price"].values, dtype=float
        ).reshape(-1, 1)
        future_day = len(future_df)
        if "CI95_lower" in future_df.columns and "CI95_upper" in future_df.columns:
            future_predictions_lower = np.asarray(
                future_df["CI95_lower"].values, dtype=float
            )
            future_predictions_upper = np.asarray(
                future_df["CI95_upper"].values, dtype=float
            )
        else:
            future_p = future_predictions_prices.flatten()
            future_predictions_lower = future_p
            future_predictions_upper = future_p

    if future_day > 0:
        last_date = pd.Timestamp(future_dates[0]) - pd.Timedelta(days=1)
    else:
        last_date = pd.Timestamp(prediction_dates[-1])

    plt.style.use("dark_background")

    # ----------------------- Figure 1 (matches pytorch_plotted) -----------------------
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"{chart_name_plot} - Detailed Prediction Analysis (cpp)",
        fontsize=16,
        fontweight="bold",
    )
    graph = fig.add_subplot(2, 1, 1)

    plot_data = test_data.copy()
    if isinstance(plot_data.columns, pd.MultiIndex):
        plot_data.columns = plot_data.columns.get_level_values(0)

    mpf.plot(
        plot_data,
        type="candle",
        style="yahoo",
        ax=graph,
        show_nontrading=True,
        datetime_format="%Y-%m-%d",
        ylabel=f"{chart_name_plot} Price",
    )

    graph.plot(
        prediction_dates_offset,
        prediction_prices,
        color="cyan",
        label="Predicted Prices (Test Period)",
        linewidth=2,
    )

    if future_day > 0:
        graph.plot(
            future_dates,
            future_predictions_prices,
            color="coral",
            label=f"Future Forecast ({future_day} days)",
            linewidth=2.5,
        )
        graph.fill_between(
            future_dates,
            future_predictions_lower,
            future_predictions_upper,
            color="purple",
            alpha=0.2,
            label="95% Confidence Interval",
        )

    graph.axvline(
        x=last_date,
        color="orange",
        linestyle=":",
        linewidth=2,
        label="Current Date",
        alpha=0.7,
    )
    graph.set_title(
        f"{chart_name_plot} Price Prediction with {future_day or '—'}-Day Forecast "
        f"(Monte Carlo Dropout Confidence)"
    )
    graph.set_xlabel("Date")
    graph.legend(loc="upper left")
    graph.grid(True, alpha=0.3)

    ax_res = plt.subplot(2, 1, 2)
    residuals = actual_prices.flatten() - prediction_prices.flatten()
    ax_res.scatter(prediction_dates, residuals, color="yellow", alpha=0.6, s=20)
    ax_res.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax_res.fill_between(
        prediction_dates,
        residuals.mean() - residuals.std(),
        residuals.mean() + residuals.std(),
        color="green",
        alpha=0.1,
    )
    ax_res.set_title("Prediction Residuals (Actual - Predicted)", fontweight="bold")
    ax_res.set_ylabel("Residual ($)")
    ax_res.set_xlabel("Date")
    ax_res.grid(True, alpha=0.3)

    annotation = graph.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="#333333", ec="white", alpha=0.9),
        fontsize=9,
    )
    annotation.set_visible(False)

    actual_dnums = mdates.date2num(prediction_dates)
    pred_dnums = mdates.date2num(prediction_dates_offset)
    future_dnums = mdates.date2num(future_dates) if future_day > 0 else np.array([])

    actual_vals = actual_prices.flatten()
    pred_vals = prediction_prices.flatten()
    future_vals = (
        future_predictions_prices.flatten() if future_day > 0 else np.array([])
    )

    def motion_hover(event):
        if event.inaxes != graph:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        x = event.xdata
        if x is None:
            return

        idx_actual = np.argmin(np.abs(actual_dnums - x)) if len(actual_dnums) else None
        dist_actual = (
            abs(actual_dnums[idx_actual] - x) if idx_actual is not None else np.inf
        )

        idx_pred = np.argmin(np.abs(pred_dnums - x)) if len(pred_dnums) else None
        dist_pred = abs(pred_dnums[idx_pred] - x) if idx_pred is not None else np.inf

        idx_future = (
            np.argmin(np.abs(future_dnums - x)) if len(future_dnums) else None
        )
        dist_future = (
            abs(future_dnums[idx_future] - x) if idx_future is not None else np.inf
        )

        nearest = "none"
        if (
            idx_actual is not None
            and dist_actual <= dist_pred
            and dist_actual <= dist_future
        ):
            nearest = "actual"
        elif (
            idx_pred is not None
            and dist_pred <= dist_actual
            and dist_pred <= dist_future
        ):
            nearest = "pred"
        elif idx_future is not None:
            nearest = "future"
        else:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return

        if nearest == "actual" and idx_actual is not None:
            dnum = actual_dnums[idx_actual]
            date = mdates.num2date(dnum)
            actual = actual_vals[idx_actual]
            pred_idx = np.argmin(np.abs(pred_dnums - dnum)) if len(pred_dnums) else None
            predicted = pred_vals[pred_idx] if pred_idx is not None else float("nan")
        elif nearest == "pred" and idx_pred is not None:
            dnum = pred_dnums[idx_pred]
            date = mdates.num2date(dnum)
            predicted = pred_vals[idx_pred]
            act_idx = (
                np.argmin(np.abs(actual_dnums - dnum))
                if len(actual_dnums)
                else None
            )
            actual = (
                actual_vals[act_idx]
                if act_idx is not None
                else float("nan")
            )
        elif nearest == "future" and idx_future is not None:
            dnum = future_dnums[idx_future]
            date = mdates.num2date(dnum)
            predicted = future_vals[idx_future]
            actual = float("nan")
        else:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return

        actual_text = f"${actual:.2f}" if (not np.isnan(actual)) else "N/A"
        pred_text = f"${predicted:.2f}"
        text = f"{date.strftime('%Y-%m-%d')}\nActual: {actual_text}\nPredicted: {pred_text}"

        annotation.xy = (event.xdata, event.ydata)
        annotation.set_text(text)
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", motion_hover)

    out_dir = os.path.dirname(os.path.abspath(csv_path)) or os.getcwd()
    present_day = dt.datetime.now().date()
    tag = args.optimizer_tag

    saved_paths = []
    if args.save:
        p1 = os.path.join(out_dir, f"{tag}_{chart_name_plot_short}_{present_day}.png")
        fig.savefig(p1, dpi=300, bbox_inches="tight")
        saved_paths.append(p1)

    # ----------------------- Figure 2: forecast detail -----------------------
    if future_day > 0:
        fig2, (ax_price, ax_ci) = plt.subplots(2, 1, figsize=(16, 10))
        fig2.suptitle(
            f"{chart_name_plot} - {future_day}-Day Forecast with Uncertainty (cpp)",
            fontsize=14,
            fontweight="bold",
        )

        ax_price.plot(
            test_data.index[-60:],
            actual_prices[-60:].flatten(),
            "o-",
            color="white",
            linewidth=2.5,
            markersize=4,
            label="Historical Prices",
            alpha=0.8,
        )
        ax_price.plot(
            future_dates,
            future_predictions_prices.flatten(),
            "s-",
            color="lime",
            linewidth=3,
            markersize=6,
            label="Forecast",
            alpha=0.9,
        )
        ax_price.fill_between(
            future_dates,
            future_predictions_lower,
            future_predictions_upper,
            color="purple",
            alpha=0.2,
            label="95% Confidence Band",
        )
        ax_price.axvline(
            x=last_date, color="orange", linestyle=":", linewidth=2, alpha=0.7
        )
        ax_price.set_ylabel("Price ($)", fontsize=12, fontweight="bold")
        ax_price.set_title("Price Forecast", fontweight="bold")
        ax_price.legend(loc="best", fontsize=10)
        ax_price.grid(True, alpha=0.3)

        ci_widths = future_predictions_upper - future_predictions_lower
        colors_ci = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(ci_widths)))
        ax_ci.bar(
            range(len(ci_widths)),
            ci_widths,
            color=colors_ci,
            edgecolor="white",
            linewidth=1,
            alpha=0.8,
        )
        ax_ci.set_xlabel("Days Ahead", fontsize=12, fontweight="bold")
        ax_ci.set_ylabel(
            "Confidence Interval Width ($)", fontsize=12, fontweight="bold"
        )
        ax_ci.set_title("Uncertainty Over Forecast Horizon", fontweight="bold")
        ax_ci.set_xticks(range(len(ci_widths)))
        ax_ci.set_xticklabels(
            [f"Day {i + 1}" for i in range(len(ci_widths))], rotation=45
        )
        ax_ci.grid(True, alpha=0.3, axis="y")

        fig2.tight_layout()
        if args.save:
            p2 = os.path.join(
                out_dir,
                f"{tag}_{chart_name_plot_short}_forecast_detail_{present_day}.png",
            )
            fig2.savefig(p2, dpi=300, bbox_inches="tight")
            saved_paths.append(p2)

    if args.save and saved_paths:
        print("Saved:", saved_paths)

    plt.show()


if __name__ == "__main__":
    main()
