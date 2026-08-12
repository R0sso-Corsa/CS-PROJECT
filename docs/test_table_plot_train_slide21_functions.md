| Test No | Test description | Test Data | Expected outcome |
| --- | --- | --- | --- |
| 1 | Train smoke run (functions: main, run_monte_carlo_rollout) | Small synthetic dataset (10 samples), epochs=1 | Process exits 0; checkpoint created; training loss decreases at least once |
| 2 | Plot smoke run (functions: main, _default_predictions_csv, _default_future_csv) | Sample prediction_results CSV (small array) | PNG generated with series plotted, axes and legend present |
| 3 | Checkpoint save & resume (functions: torch.save, torch.load via main workflow) | Save checkpoint after N steps; resume using checkpoint path | Resumed run continues from saved step and produces matching loss curve segment |
| 4 | Metrics logging (functions: print/logging hooks, training loop) | Train run with logging enabled (tensorboard/log file) | Training/validation loss and metrics recorded and accessible |
| 5 | Plot content verification (functions: graph.plot, mpf.plot, _info_names) | Known result vectors with labeled series | Output plot contains same number of series and correct labels |
| 6 | Plot file integrity (function: fig.savefig) | Generate PNG/JPEG for a sample graph | Image file size > 0 and file opens without error |
| 7 | Plot snapshot (CI) (functions: fig.savefig, file hashing utility) | Approved baseline image + newly generated image | Image hash matches baseline or deliberate update approved |
| 8 | Training performance (functions: training loop in main, torch.compile path) | Medium dataset (n~10k), measure epoch time | Epoch time within expected threshold; GPU/CPU util logged |
| 9 | Plot performance (functions: fig.savefig, rendering loops) | Batch generate 100 plots | Average render time acceptable; no memory spikes |
| 10 | Edge case: empty input (functions: add_features, build_sequences) | Empty dataset or missing result file | Script fails gracefully with clear error and non-zero exit; no unhandled exceptions |
| 11 | Corrupt checkpoint handling (functions: torch.load, model state check) | Provide malformed checkpoint file | Script detects corruption, reports error, and does not overwrite a valid checkpoint |
| 12 | CI unit tests for helpers (functions: add_features, build_sequences, _as_1d, _as_1d_close) | Unit tests for plotting helpers and data preprocessing | All unit tests pass; coverage for helper functions > 80% |
| 13 | Plot colorblind palette check (functions: graph.plot, plt.style) | Generate plots using colorblind-friendly palette | Series remain distinguishable; legend accessible |
| 14 | Backward checkpoint compatibility (functions: torch.load, state_dict mapping) | Load checkpoint from older model version | Model loads or gives clear migration error with guidance |
| 15 | Distributed/local multi-GPU smoke (functions: torch.compile / torch.distributed hooks if present) | Run small distributed training on 2 processes | Processes sync, no deadlocks, loss decreases |
| 16 | No-GPU fallback (functions: device selection logic in main) | Run with CUDA disabled on machine without GPU | Script runs on CPU without crashing and logs fallback |
| 17 | Multiple output formats (functions: fig.savefig, save paths) | Export plots as PNG, SVG and PDF | All requested files created and openable |
| 18 | Logging format compliance (functions: logging/json serializer) | Produce JSON-formatted logs for metrics | Logs parseable by monitoring tooling |
| 19 | Path sanitization/security (functions: resolve_output_root, ensure_artifact_dirs) | Attempt path traversal in output path param | Script rejects unsafe paths and logs error |
| 20 | Temp file cleanup (functions: temporary file handling, ensure_artifact_dirs) | Run repeated short jobs | No leftover temp files after runs |
