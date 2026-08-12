| Test No | Test description | Test Data | Expected outcome |
| --- | --- | --- | --- |
| 1 | Train smoke run | Small synthetic dataset (10 samples), epochs=1 | Process exits 0; checkpoint created; training loss decreases at least once |
| 2 | Plot smoke run | Sample prediction_results JSON (small array) | PNG generated with series plotted, axes and legend present |
| 3 | Checkpoint save & resume | Save checkpoint after N steps; resume using checkpoint path | Resumed run continues from saved step and produces matching loss curve segment |
| 4 | Metrics logging | Train run with logging enabled (tensorboard/log file) | Training/validation loss and metrics recorded and accessible |
| 5 | Plot content verification | Known result vectors with labeled series | Output plot contains same number of series and correct labels |
| 6 | Plot file integrity | Generate PNG/JPEG for a sample graph | Image file size > 0 and file opens without error |
| 7 | Plot snapshot (CI) | Approved baseline image + newly generated image | Image hash matches baseline or deliberate update approved |
| 8 | Training performance | Medium dataset (n~10k), measure epoch time | Epoch time within expected threshold; GPU/CPU util logged |
| 9 | Plot performance | Batch generate 100 plots | Average render time acceptable; no memory spikes |
| 10 | Edge case: empty input | Empty dataset or missing result file | Script fails gracefully with clear error and non-zero exit; no unhandled exceptions |
| 11 | Corrupt checkpoint handling | Provide malformed checkpoint file | Script detects corruption, reports error, and does not overwrite a valid checkpoint |
| 12 | CI unit tests for helpers | Unit tests for plotting helpers and data preprocessing | All unit tests pass; coverage for helper functions > 80% |
| 13 | Plot colorblind palette check | Generate plots using colorblind-friendly palette | Series remain distinguishable; legend accessible |
| 14 | Backward checkpoint compatibility | Load checkpoint from older model version | Model loads or gives clear migration error with guidance |
| 15 | Distributed/local multi-GPU smoke | Run small distributed training on 2 processes | Processes sync, no deadlocks, loss decreases |
| 16 | No-GPU fallback | Run with CUDA disabled on machine without GPU | Script runs on CPU without crashing and logs fallback |
| 17 | Multiple output formats | Export plots as PNG, SVG and PDF | All requested files created and openable |
| 18 | Logging format compliance | Produce JSON-formatted logs for metrics | Logs parseable by monitoring tooling |
| 19 | Path sanitization/security | Attempt path traversal in output path param | Script rejects unsafe paths and logs error |
| 20 | Temp file cleanup | Run repeated short jobs | No leftover temp files after runs |
