from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def ticker_slug(ticker: str) -> str:
    cleaned = []
    for char in ticker.lower():
        cleaned.append(char if char.isalnum() else "-")
    slug = "".join(cleaned).strip("-")
    return slug or "ticker"


def newest_file(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} in {path}")
    return matches[-1]


def run_training(args: argparse.Namespace, job_dir: Path) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    train_script = repo_root / "REWRITE" / "separated" / "pytorch_train_cpp.py"
    log_path = job_dir / "training.log"

    command = [
        sys.executable,
        str(train_script),
        "--ticker",
        args.ticker,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--prediction-days",
        str(args.prediction_days),
        "--future-days",
        str(args.future_days),
        "--mc-runs",
        str(args.mc_runs),
        "--output-dir",
        str(job_dir),
    ]

    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)

    return {
        "model": newest_file(job_dir / "models", f"{args.ticker}_model_cpp_*.pt"),
        "predictions_csv": newest_file(job_dir / "predictions", f"{args.ticker}_predictions_cpp_*.csv"),
        "forecast_csv": newest_file(job_dir / "forecasts", f"{args.ticker}_future_*d_cpp_*.csv"),
        "training_log": log_path,
    }


def render_summary_plot(predictions_csv: Path, forecast_csv: Path, out_path: Path, ticker: str) -> dict[str, float]:
    pred_df = pd.read_csv(predictions_csv)
    pred_df["Date"] = pd.to_datetime(pred_df["Date"])

    future_df = pd.read_csv(forecast_csv)
    future_df["Date"] = pd.to_datetime(future_df["Date"])

    actual = pred_df["Actual"]
    predicted = pred_df["Predicted"]
    residuals = actual - predicted
    rmse = float((residuals.pow(2).mean()) ** 0.5)
    mae = float(residuals.abs().mean())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pred_df["Date"], actual, color="#0f172a", linewidth=2.1, label="Actual")
    ax.plot(pred_df["Date"], predicted, color="#0ea5e9", linewidth=2.0, label="Predicted")
    ax.plot(future_df["Date"], future_df["Predicted_Price"], color="#16a34a", linewidth=2.2, label="Forecast")
    ax.fill_between(
        future_df["Date"],
        future_df["CI95_lower"],
        future_df["CI95_upper"],
        color="#86efac",
        alpha=0.25,
        label="95% confidence",
    )
    ax.set_title(f"{ticker} forecast overview")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {"rmse": rmse, "mae": mae, "future_days": float(len(future_df))}


def render_detail_plot(forecast_csv: Path, out_path: Path, ticker: str) -> None:
    future_df = pd.read_csv(forecast_csv)
    future_df["Date"] = pd.to_datetime(future_df["Date"])
    x = range(1, len(future_df) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(future_df["Date"], future_df["Predicted_Price"], color="#0891b2", linewidth=2.2)
    axes[0].fill_between(
        future_df["Date"],
        future_df["CI95_lower"],
        future_df["CI95_upper"],
        color="#7dd3fc",
        alpha=0.3,
    )
    axes[0].set_title(f"{ticker} forward forecast")
    axes[0].set_ylabel("Predicted price")
    axes[0].grid(alpha=0.25)

    ci_width = future_df["CI95_upper"] - future_df["CI95_lower"]
    axes[1].bar(x, ci_width, color="#f97316", alpha=0.85)
    axes[1].set_title("Confidence interval width")
    axes[1].set_xlabel("Forecast day")
    axes[1].set_ylabel("Width")
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one remote web forecast job and emit a JSON manifest.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--prediction-days", default=30, type=int)
    parser.add_argument("--future-days", default=30, type=int)
    parser.add_argument("--mc-runs", default=100, type=int)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    job_dir = output_root / ticker_slug(args.ticker) / f"job_{args.job_id}"
    for child in ["models", "predictions", "forecasts", "plots"]:
        (job_dir / child).mkdir(parents=True, exist_ok=True)

    artifacts = run_training(args, job_dir)

    summary_plot = job_dir / "plots" / "summary.png"
    detail_plot = job_dir / "plots" / "detail.png"
    stats = render_summary_plot(artifacts["predictions_csv"], artifacts["forecast_csv"], summary_plot, args.ticker)
    render_detail_plot(artifacts["forecast_csv"], detail_plot, args.ticker)

    manifest = {
        "job_id": args.job_id,
        "ticker": args.ticker,
        "job_dir": str(job_dir),
        "summary": f"Remote training for {args.ticker} completed successfully. Graph assets were generated and are ready for import.",
        "stats": stats,
        "files": {
            "model": str(artifacts["model"]),
            "predictions_csv": str(artifacts["predictions_csv"]),
            "forecast_csv": str(artifacts["forecast_csv"]),
            "summary_plot": str(summary_plot),
            "detail_plot": str(detail_plot),
            "training_log": str(artifacts["training_log"]),
        },
    }

    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, separators=(",", ":")))


if __name__ == "__main__":
    main()

