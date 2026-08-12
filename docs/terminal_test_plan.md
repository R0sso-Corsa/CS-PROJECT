# Terminal Test Plan — Market Forecasting System

Overview

This terminal test plan describes step-by-step checks that map directly to your stated success criteria: the system must download and process market data for a chosen ticker without manual spreadsheet work; produce sensible test-period predictions that can be plotted against actual values; produce plausible future forecasts with uncertainty via Monte Carlo dropout; be runnable by a non-expert through the terminal; and maintain clear evidence of runs and artifacts.

How to run (quick)

Use the project's Python environment. Example commands you can run from the repository root:

`c:/Users/paron/Desktop/Dev/CS_PROJECT/.venv_amd/Scripts/python.exe REWRITE/separated/pytorch_train_cpp.py --ticker AAPL --epochs 2 --output-root artifacts/test_run`

`c:/Users/paron/Desktop/Dev/CS_PROJECT/.venv_amd/Scripts/python.exe REWRITE/separated/pytorch_plot_cpp.py --predictions artifacts/test_run/predictions.csv --future artifacts/test_run/future.csv --out artifacts/test_run/plot.png`

Tests (paragraph form)

T1 — Data download: Verify the CLI can fetch historical market data for a chosen ticker and save a raw CSV. Run the training CLI with the `--only-download` flag and an `--output-root` path. The expected outcome is a raw OHLCV CSV in the artifacts folder containing at least ~100 rows and no columns that are entirely NaN. Pass if the CSV exists, contains a sensible number of rows, and fields like `Date`, `Open`, `Close`, `Volume` are populated.

T2 — Preprocessing: Confirm raw data can be converted into prepared sequences and feature files without manual spreadsheet edits. Use the `--dry-prep` or equivalent preprocessing flag to produce prepared CSVs in the output folder. Expect prepared sequence files and feature columns (including engineered indicators). Pass if prepared files are created and final feature columns contain no NaNs.

T3 — Short training smoke: Run a short training session (1–3 epochs, small batch) to confirm the model code path executes and writes a model file. Example: run the training CLI with `--epochs 2 --batch 16`. The expected result is a saved model file (e.g., `model.pt`) and a printed training log. Pass if the model file is written and the run completes without exceptions.

T4 — Prediction on test period: After a training run, invoke the prediction/inference path to write `predictions.csv` (in-sample/test) and `future.csv` (forecast). Confirm that prediction rows align with actual date indexes. Pass if the predictions file exists and date-index alignment is correct when compared with the actuals CSV.

T5 — Plot generation: Run the plotting script using the produced CSVs to generate a PNG that overlays actual vs predicted traces. Pass if the PNG file exists and visual inspection shows predicted trace overlapping the test-period actuals appropriately (or automated metric such as RMSE within a reasonable range when desired).

T6 — Monte Carlo dropout (uncertainty): Exercise MC dropout inference by performing N stochastic forward passes (e.g., N=50) and writing multiple forecast traces and summary statistics. Expected outputs include a multi-run forecast CSV and a computed mean and standard deviation per forecast timestep. Pass if multiple traces are present and the forecast band (stddev) is non-zero and visually plausible.

T7 — Usability: CLI discoverability and simplicity. Run `--help` for the main scripts and execute a recommended one-line example changing only the `--ticker`. Pass if help text is clear and an end-user can reproduce the example by only swapping the ticker symbol.

T8 — End-to-end single-command demo: Provide or run a wrapper that performs download→train→predict→plot in one invocation and produces the full set of artifacts in one directory. Pass if the wrapper completes and the artifacts (model, CSVs, PNG) are present.

T9 — Robustness to missing data: Introduce simulated missing days or NaNs into the raw CSV, run preprocessing and training, and ensure the system either imputes or fails cleanly with a descriptive error. Pass if no unhandled exceptions occur and logs explain the imputation or failure mode.

T10 — Bad ticker handling: Run the CLI with an invalid ticker symbol and verify the program exits with a non-zero code and a human-readable error message, producing no partial/broken artifacts. Pass if the error message is informative and no artifacts are produced.

T11 — Artifact stability and evidence: Inspect the run artifacts for metadata (timestamp, git SHA or version, parameters) and logs. Pass if each run directory contains a `metadata.json` or `run.log` recording the command, time, seed, and code version/hash.

T12 — Reproducibility (seed): Run the same short training twice using a fixed `--seed` and confirm deterministic or near-deterministic behavior for a small run. Pass if key outputs (loss curve for a short run, or deterministic small-run predictions) match within a tiny tolerance, with any GPU nondeterminism documented.

Acceptance guidance

For visual tests, a human inspection is acceptable; optionally add an automated metric (e.g., holdout RMSE threshold) to convert visual checks into pass/fail criteria. For MC dropout tests, ensure the forecast mean follows recent trend and the confidence band has meaningful width (not zero).

Usability note

Add a brief `examples/README.md` containing the one-line example commands and minimal environment setup (activate venv, Python path). This helps non-expert users run the pipeline by copying one command.

Next steps

I can export this paragraph-style plan to Word (`.docx`) and update or add a small wrapper script that runs an end-to-end demo. Would you like me to generate the `.docx` now?
