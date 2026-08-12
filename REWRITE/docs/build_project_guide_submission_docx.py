from __future__ import annotations

import datetime as dt
from pathlib import Path

from build_project_report_docx import (
    DocxBuilder,
    add_comparison,
    add_timeline,
    extract_block,
    write_docx,
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def collect_snippets(repo_root: Path) -> dict[str, str]:
    rewrite_root = repo_root / "REWRITE"
    docs_root = rewrite_root / "docs"
    learning_root = rewrite_root / "learning-path"

    p1 = learning_root / "01-tensorflow-basics" / "ai_predict.py"
    p3 = learning_root / "03-tensorflow-dynamic-dropout" / "test_modified.py"
    p4 = learning_root / "04-tensorflow-extended-forecasting" / "future-Eric.py"
    p5 = learning_root / "05-pytorch-transition" / "future-eric-pytorch.py"
    p6 = learning_root / "06-pytorch-stable-core" / "pytorch_fixed.py"
    p7 = learning_root / "07-pytorch-full-pipeline" / "pytorch_plotted.py"
    p8 = learning_root / "08-pytorch-experiments" / "pytorch_additional.py"
    p9t = rewrite_root / "separated" / "pytorch_train_cpp.py"
    p9p = rewrite_root / "separated" / "pytorch_plot_cpp.py"

    web_login = repo_root / "web" / "login.php"
    web_js = repo_root / "web" / "scripts.js"
    gui_main = repo_root / "gui_main.py"
    users_sql = repo_root / "data" / "root" / "sql" / "USERS.sql"

    return {
        "tf_close_only": extract_block(
            p1,
            "scaler = MinMaxScaler(feature_range=(0, 1))",
            "model.fit(x_train, y_train, epochs=10, batch_size=10000000)",
        ),
        "dynamic_dropout": extract_block(
            p3,
            "def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):",
            "dynamic_dropout = DynamicDropoutCallback(epochs, initial_dropout, final_dropout)",
        ),
        "tf_features": extract_block(
            p4,
            "df['returns'] = df['log'].diff()",
            "scaled_features_train = np.hstack([scaled_returns, scaled_X_train])",
        ),
        "pytorch_transition_model": extract_block(
            p5,
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "writer = SummaryWriter(log_dir)",
        ),
        "pytorch_stable_model": extract_block(
            p6,
            "class LSTMModel(nn.Module):",
            "        return out",
        ),
        "pytorch_stable_training": extract_block(
            p6,
            "criterion = nn.HuberLoss()",
            "            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
        ),
        "pytorch_backend": extract_block(
            p7,
            "torch.backends.cudnn.enabled = torch.version.hip is None",
            'os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"  # RDNA4-specific scheduling',
        ),
        "pytorch_mc": extract_block(
            p7,
            "# Use model.train() instead of eval() to keep dropout active for jagged (stochastic) predictions",
            "future_predictions_std.append(",
        ),
        "attention_model": extract_block(
            p8,
            "class EnhancedLSTMModel(nn.Module):",
            "        return out",
        ),
        "cpp_output_paths": extract_block(
            p9t,
            "SCRIPT_DIR = Path(__file__).resolve().parent",
            "    return artifact_dirs",
        ),
        "cpp_config_model": extract_block(
            p9t,
            "@dataclass",
            "        return self.fc(out)",
        ),
        "cpp_sequence_builders": extract_block(
            p9t,
            "def build_sequences(values, prediction_days):",
            "        return np.asarray(xs, dtype=np.float32)",
        ),
        "cpp_add_features": extract_block(
            p9t,
            "def add_features(df):",
            "    return out",
        ),
        "cpp_rollout": extract_block(
            p9t,
            "def run_monte_carlo_rollout(",
            "    return future_prices, std_unscaled, lower, upper",
        ),
        "cpp_compile": extract_block(
            p9t,
            "    if use_compile:",
            '            print(f"torch.compile skipped: {e}")',
        ),
        "cpp_main_cli_data": extract_block(
            p9t,
            "def main():",
            '        raise RuntimeError("Insufficient data to create training sequences.")',
        ),
        "cpp_training_loop": extract_block(
            p9t,
            "    t0 = time.time()",
            '    print(f"Training time: {train_seconds:.2f}s")',
        ),
        "cpp_eval_save": extract_block(
            p9t,
            "    # Quick evaluation on recent segment.",
            '    print(f"Saved predictions: {pred_path}")',
        ),
        "cpp_forecast_save": extract_block(
            p9t,
            '    print(f"\\n{\'=\' * 60}")',
            '        f"Forecast day {cfg.future_day}: ${final_fc:.2f}  ({pct:+.2f}%)"',
        ),
        "plot_discovery": extract_block(
            p9p,
            'DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_ROOT / "predictions"',
            "    return max(candidates, key=os.path.getmtime)",
        ),
        "plot_cli_validation": extract_block(
            p9p,
            "def main():",
            '    pred_df = pred_df.sort_values("Date").reset_index(drop=True)',
        ),
        "plot_context_future": extract_block(
            p9p,
            "    # Re-download OHLC data only for plotting context.",
            "            future_predictions_upper = future_p",
        ),
        "plot_figure_residuals": extract_block(
            p9p,
            '    plt.style.use("dark_background")',
            '    fig.canvas.mpl_connect("motion_notify_event", motion_hover)',
        ),
        "plot_forecast_window": extract_block(
            p9p,
            "    DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)",
            "    plt.show()",
        ),
        "web_login_form": extract_block(
            web_login,
            '<form method="post">',
            "</form>",
        ),
        "web_validate_login": extract_block(
            web_js,
            "function sanitizeFormData(form) {",
            "const searchInput = document.querySelector",
        ),
        "gui_create_widgets": extract_block(
            gui_main,
            "    def create_widgets(self):",
            '        self.status_label.pack(pady=(15, 0))',
        ),
        "gui_build_command": extract_block(
            gui_main,
            "    def build_command(self):",
            "        return cmd",
        ),
        "users_sql": read_text(users_sql).strip(),
        "project_log_intro": read_text(docs_root / "PROJECT_LOG_FULL.md").split("## Phase 1")[0].strip(),
    }


def add_manual_contents(doc: DocxBuilder) -> None:
    doc.add_heading("Contents", level=1)
    entries = [
        "1. Executive Summary",
        "2. Project Identification and Problem Definition",
        "3. Computational Methods",
        "4. Stakeholders",
        "5. Evaluation of Similar Systems",
        "6. APIs, Libraries, Data Sources and Research Justification",
        "7. Essential Features",
        "8. Project Limitations and Scope Boundaries",
        "9. Requirements",
        "10. Success Criteria",
        "11. Design Decomposition and System Structure",
        "12. Algorithms, IPO Reasoning and Pipelining",
        "13. User Interface Design and Wireframe Discussion",
        "14. Key Variables, Data Structures and Validation",
        "15. Implementation of the Main Training and Plotting Scripts",
        "16. Iterative Test Plan",
        "17. Post-Development Test Plan",
        "18. Evaluation and Final Reflection",
        "19. Bibliography",
    ]
    for entry in entries:
        doc.add_paragraph(entry)


def add_bibliography(doc: DocxBuilder) -> None:
    refs = [
        "AMD (2026) ROCm documentation. Available at: https://rocm.docs.amd.com/ (Accessed: 13 April 2026).",
        "CoinGecko (2026) CoinGecko. Available at: https://www.coingecko.com/ (Accessed: 13 April 2026).",
        "CoinStats (2026) CoinStats. Available at: https://coinstats.app/ (Accessed: 13 April 2026).",
        "GitHub (2026) REST API documentation. Available at: https://docs.github.com/en/rest (Accessed: 13 April 2026).",
        "Matplotlib Development Team (2026) Matplotlib documentation. Available at: https://matplotlib.org/stable/ (Accessed: 13 April 2026).",
        "NumPy Developers (2026) NumPy documentation. Available at: https://numpy.org/doc/stable/ (Accessed: 13 April 2026).",
        "Optuna Contributors (2026) Optuna documentation. Available at: https://optuna.readthedocs.io/en/stable/ (Accessed: 13 April 2026).",
        "pandas development team (2026) pandas documentation. Available at: https://pandas.pydata.org/docs/ (Accessed: 13 April 2026).",
        "Python Software Foundation (2026a) tkinter - Python interface to Tcl/Tk. Available at: https://docs.python.org/3/library/tkinter.html (Accessed: 13 April 2026).",
        "Python Software Foundation (2026b) sqlite3 - DB-API 2.0 interface for SQLite databases. Available at: https://docs.python.org/3/library/sqlite3.html (Accessed: 13 April 2026).",
        "PyTorch Contributors (2026) PyTorch documentation. Available at: https://pytorch.org/docs/stable/ (Accessed: 13 April 2026).",
        "TensorFlow Authors (2026) TensorFlow API documentation. Available at: https://www.tensorflow.org/api_docs (Accessed: 13 April 2026).",
        "TradingView (2026) TradingView. Available at: https://www.tradingview.com/ (Accessed: 13 April 2026).",
        "Yahoo Finance (2026) Yahoo Finance. Available at: https://finance.yahoo.com/ (Accessed: 13 April 2026).",
        "yfinance (2026) yfinance on PyPI. Available at: https://pypi.org/project/yfinance/ (Accessed: 13 April 2026).",
        "mplfinance (2026) mplfinance on PyPI. Available at: https://pypi.org/project/mplfinance/ (Accessed: 13 April 2026).",
    ]
    for ref in refs:
        doc.add_paragraph(ref)


def build_report() -> DocxBuilder:
    docs_dir = Path(__file__).resolve().parent
    repo_root = docs_dir.parent.parent
    snippets = collect_snippets(repo_root)
    today = dt.datetime.now().strftime("%d %B %Y")

    doc = DocxBuilder()
    doc.add_title("Comprehensive Project Report")
    doc.add_subtitle("Guide-aligned Word submission built from the project guide, the earlier draft document, and the full project directory")
    doc.add_subtitle(f"Prepared on {today}")
    doc.add_subtitle("Focus: complete documentation of the project's problem definition, design process, implementation journey, code evolution, testing, evaluation and bibliography")

    add_manual_contents(doc)

    doc.add_page_break()
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraphs(
        """
        This report follows the structure of the project guide while correcting the weak points of the earlier draft document. The earlier draft preserved important ideas, especially the intention to build a stock and cryptocurrency tracking website with stored accounts, favourites and AI-supported forecasts, but it also mixed planned features with completed ones. The purpose of this document is therefore not to repeat the draft at greater length. It is to produce a correct, repository-based account of what the project actually became, how it evolved, and which parts were exploratory, prototyped, implemented or superseded.

        The project began as a broader software engineering idea than a simple machine-learning script. The repository still contains a draw.io top-down diagram for a website and prediction AI, a prototype HTML login form, JavaScript validation logic, a user SQL schema and several exploratory files connected to account or storage ideas. Those artifacts matter because they show that the original problem definition involved user interaction, data persistence and analytical output together rather than only model training in isolation.

        Over time, the forecasting strand became the most technically mature part of the work. The repository records a clear progression from TensorFlow/Keras experiments, through dynamic dropout and richer feature engineering, into a major PyTorch rewrite and finally into a separated training-and-plotting architecture designed for maintainability. This progression is one of the strongest aspects of the project because it shows genuine iteration: each stage solved one problem and revealed another.

        The most accurate final description of the project is therefore a financial forecasting application whose original concept included website and database features, but whose strongest completed deliverable became a machine-learning pipeline with feature engineering, uncertainty estimation, charting, a GUI launcher and organised artifacts. This report documents both strands honestly. The web/database work is treated as authentic project context and prototype evidence, while the PyTorch forecasting pipeline is treated as the strongest completed end state.
        """
    )

    doc.add_page_break()
    doc.add_heading("Project Identification and Problem Definition", level=1)
    doc.add_paragraphs(
        """
        The problem addressed by the project is the difficulty users face when trying to turn raw market data into useful insight. Prices, charts and headlines are widely available, but understanding whether a move reflects noise, trend, momentum or volatility is much harder. The project therefore aimed to build a program that could retrieve financial data, organise it into clearer analytical structures and support the user with both visual presentation and predictive modelling.

        In its earliest form, the intended solution was a market-tracking website with account functionality. The earlier draft described login systems, UUID assignment, stored favourites and the ability to revisit information through a user-oriented interface. The repository supports that claim through a website prototype, SQL schema and top-down design diagram. This means the original project identification genuinely included both interface and analysis concerns.

        As the work developed, the forecasting pipeline became the most complete and technically rigorous strand. The repository now centres on scripts that download historical financial data, create technical indicators, build sequence windows, train recurrent neural networks, evaluate test-period predictions and perform future forecasting with uncertainty intervals. The final state is therefore better described as an AI-assisted forecasting application than a finished deployed account-based website.

        A computer program is a suitable solution because the task is inherently computational. Historical OHLCV data is numerical, sequential and repetitive. Technical indicators are produced through formulae such as rolling averages, exponential smoothing and ratio-based momentum calculations. Neural-network training requires tensor operations, repeated backpropagation and statistical evaluation that would be impractical to carry out manually. The user benefit comes from faster analysis, richer structure and more interpretable chart output than raw data alone would provide.
        """
    )
    doc.add_heading("Prototype Evidence from the Original Concept", level=2)
    doc.add_heading("Login Form Prototype", level=3)
    doc.add_code_block(snippets["web_login_form"])
    doc.add_heading("Prototype Validation Logic", level=3)
    doc.add_code_block(snippets["web_validate_login"])
    doc.add_heading("Planned User Table", level=3)
    doc.add_code_block(snippets["users_sql"])
    doc.add_paragraphs(
        """
        These excerpts show that the original concept genuinely included user-facing authentication and persistent records, even if those ideas were not developed into a secure production-ready web platform. They are important as project-history evidence because they justify discussing a website/database strand in analysis and design, while also making it possible to evaluate honestly why the final strongest deliverable shifted toward forecasting rather than hosted account management.
        """
    )

    doc.add_page_break()
    doc.add_heading("Computational Methods", level=1)
    doc.add_paragraphs(
        """
        The project is strongly suited to computational methods because it depends on abstraction, decomposition, logical sequencing, iteration and large structured datasets. The task is not merely to display a price. It is to transform continuous market history into repeatable program steps that software can execute reliably.

        Thinking abstractly appears in the way the project reduces messy market behaviour into usable variables and derived indicators. Instead of treating financial movement as an unstructured narrative, the code represents it through columns such as Open, High, Low, Close, Volume and indicators such as SMA, RSI, MACD and Bollinger Bands. This abstraction removes unnecessary detail while keeping the information the system can process meaningfully.

        Thinking ahead is visible in the preparation of sequence windows, scaler behaviour and artifact handling. The project identifies in advance that every prediction depends on a lookback period, a consistent feature matrix and the ability to invert scaling later for plotting. The separated `cpp` pipeline also shows forward planning by creating dedicated directories for models, predictions and future forecasts before expensive computation begins.

        Thinking procedurally and through decomposition is one of the clearest strengths of the repository. The larger problem is repeatedly broken into smaller tasks: fetch data, create features, handle missing values, scale the matrix, construct sequences, train the model, evaluate predictions, forecast future values, save artifacts and render plots. Later refactors improve this even further by separating training from plotting. This decomposition makes the solution manageable and testable.

        Thinking logically is equally essential because the entire pipeline depends on ordered relationships between steps. Feature engineering must happen before scaling. Scaling must happen before sequence generation. Training must happen before inference. Artifact generation must happen before plot-only scripts can rediscover results. If the order is broken, outputs become invalid or the program fails. This is exactly why the project fits computational methods so well: it is a structured problem with clear inputs, processes and outputs.
        """
    )

    doc.add_page_break()
    doc.add_heading("Stakeholders", level=1)
    doc.add_paragraphs(
        """
        The primary stakeholder is the end user who wants to inspect market instruments more effectively. In the earliest concept this user was imagined as someone interacting with a website, storing favourites and revisiting information through an account. In the final implemented form the same user benefits more directly from a forecasting application that can download data, generate indicators and present probabilistic charts.

        A second major stakeholder is the developer, who is also effectively the maintainer, tester and system integrator. The repository makes it clear that maintainability became increasingly important as the project grew. Decisions such as separating plotting from training, improving logging and standardising output folders directly serve this stakeholder by making the project easier to continue understanding and extending.

        A third stakeholder is the teacher or assessor. The project guide itself demonstrates that the solution must be explainable and justifiable, not merely runnable. For this stakeholder, the value of the system depends on evidence of decomposition, algorithm design, appropriate libraries, testing and iterative improvement. This is why a detailed report and dated implementation evidence matter as much as the final model output.

        A fourth stakeholder is the tester or peer reviewer. Someone other than the developer may need to operate the GUI, inspect charts, interpret logs or comment on usability. The existence of `gui_main.py` is especially relevant here because it lowers the barrier to using the project without editing internal source code directly.
        """
    )

    doc.add_page_break()
    doc.add_heading("Evaluation of Similar Systems", level=1)
    doc.add_paragraphs(
        """
        Similar systems are important because they show what users expect from financial-analysis software. CoinGecko is a useful comparison point because it presents large amounts of market information in one place and emphasises quick interpretation across multiple assets. The relevant lesson for this project is that a useful market tool should show more than a single isolated price.

        CoinStats is another useful benchmark because it sits closer to portfolio and tracking workflows, which relates to the original account-based concept in the earlier draft. Even though the final implemented project does not become a full production portfolio platform, CoinStats helps justify the early emphasis on returning users, favourites and interface-driven interaction.

        TradingView is especially important as a benchmark for chart quality. It demonstrates that users judge analytical software partly by how clearly the visuals communicate movement and overlays. This supports the project's decision to improve candlestick charting, prediction overlays and confidence intervals rather than leaving the output as raw text.

        Yahoo Finance is both a reference point and a data-origin benchmark. Because the project retrieves data through `yfinance`, it effectively builds analytical behaviour on top of a source users may already recognise. This increases trust and makes the solution more understandable. Across these comparisons, the shared lesson is that good financial software combines clear visuals, structured data, repeated usability and honest analytical support rather than pretending to offer certainty.
        """
    )

    doc.add_page_break()
    doc.add_heading("APIs, Libraries, Data Sources and Research Justification", level=1)
    doc.add_paragraphs(
        """
        The project depends on several important libraries, each chosen because it solves a distinct part of the problem. `yfinance` is central because it provides the historical market data required for every later stage. Without an external data-access layer, the project would have no practical source of OHLCV histories for modelling and charting.

        pandas and NumPy form the core data-processing layer. pandas is responsible for time-indexed tables, rolling calculations, date-aware handling and most indicator construction. NumPy becomes especially important in later stages when sequence windows and scaled feature matrices need to be shaped efficiently for model input. In the separated pipeline it also supports more efficient window generation through `sliding_window_view`.

        TensorFlow was used first because it offered a straightforward path into stacked LSTM experimentation and TensorBoard logging. PyTorch was adopted later because it provided more explicit control over device behaviour, model internals and unusual training or inference logic. The migration was not just stylistic. It was justified by the project's need for clearer hardware-aware control and deeper experimentation.

        Matplotlib and mplfinance were chosen because chart output is essential in a financial project. A forecasting system that cannot communicate results clearly through charts is much less useful to both end users and assessors. Tkinter became relevant later because a graphical launcher improves usability for people who may not want to work entirely from the terminal. sqlite3 and Optuna appear in the repository because the project also explored persistence and more formal experiment tracking.
        """
    )
    doc.add_heading("Code Exhibit: Mature Feature Engineering in the Separated Pipeline", level=2)
    doc.add_code_block(snippets["cpp_add_features"])
    doc.add_paragraphs(
        """
        This function demonstrates why the project needed robust data libraries. It uses pandas for rolling calculations, exponential moving averages, fill operations and column management, and it also shows that the developer learned from earlier `yfinance` shape problems by flattening `MultiIndex` columns before model preparation begins.
        """
    )

    doc.add_page_break()
    doc.add_heading("Essential Features", level=1)
    doc.add_paragraphs(
        """
        The first essential feature is market-data retrieval for a chosen ticker. Without a reliable method for acquiring historical prices and volumes, the project cannot function as a forecasting tool at all. The second essential feature is preprocessing and feature engineering, because the model relies on structured windows and indicators rather than raw, unprepared data.

        The third essential feature is the forecasting model itself. Whether the run involves training from scratch or using existing weights, the system must be able to execute a recurrent neural model and produce predictions. The fourth essential feature is readable output. Test-period predictions, future forecasts and chart overlays are fundamental to the job of the program, not cosmetic extras.

        The fifth essential feature is interaction. Earlier concepts expressed this through login and web navigation. Later implementations expressed it through interactive ticker search, command-line arguments and the Tkinter GUI. A user must be able to choose a symbol and run the analytical process with reasonable control.

        The sixth essential feature is run traceability. Logging and artifact output became increasingly important as the project matured. In the final architecture, prediction and forecast files are written to stable directories so the results of a run can be reviewed later. That makes reproducibility an essential feature rather than a convenience.
        """
    )

    doc.add_page_break()
    doc.add_heading("Project Limitations and Scope Boundaries", level=1)
    doc.add_paragraphs(
        """
        The project does not eliminate the inherent uncertainty of market forecasting. No model in the repository guarantees future prices, and the move toward confidence intervals and stochastic rollouts is itself evidence that the developer recognised this limitation. The software should therefore be understood as a decision-support tool rather than a guarantee engine.

        A second limitation is that the full-stack website vision remained incomplete. The repository contains genuine web and database prototypes, but it does not contain a finished secure hosted platform with production-grade authentication and deployment. This means the strongest completed outcome is the forecasting application rather than the original full website concept.

        A third limitation is dependency on external data behaviour. Because the system relies on Yahoo Finance data through `yfinance`, changes in upstream availability or returned column structures can affect local processing. Later refactors reduce this fragility but do not remove the dependency entirely.

        A fourth limitation is environment sensitivity. Hardware and backend behaviour, especially around ROCm and recurrent layers, influenced several engineering decisions. The software runs, but not every machine behaves identically, which is why explicit device handling and backend workarounds became important.
        """
    )

    doc.add_page_break()
    doc.add_heading("Requirements", level=1)
    doc.add_paragraphs(
        """
        The project requires a usable historical dataset for the target symbol, a Python environment with the relevant numerical and ML libraries, and enough processing power to handle training, plotting and repeated experimentation. In practical terms this means Python plus libraries such as `yfinance`, `pandas`, `numpy`, `matplotlib`, `mplfinance`, `scikit-learn` and `torch`, with TensorFlow required for earlier stages and Optuna useful for tuning work.

        The project also requires reasonable hardware. A modern CPU and sufficient memory are necessary even when running without a GPU. GPU support is helpful but not mandatory, and the repository shows clearly that if GPU acceleration is used, the exact environment matters. This is why hardware should be treated as part of the requirements discussion rather than as an incidental detail.

        Another requirement is time. Forecasting runs with multiple epochs, Monte Carlo sampling and chart rendering are not instantaneous. The system therefore requires not just software packages but a workflow capable of supporting long-running jobs. This helps justify the attention paid to logging, progress bars and reusable artifacts.

        Finally, the late-stage architecture demonstrates that organised output storage became a practical requirement in itself. Once the project reached the stage of repeated experiments and separate plotting, stable output roots for models, predictions and forecasts were necessary for the software to remain usable.
        """
    )

    doc.add_page_break()
    doc.add_heading("Success Criteria", level=1)
    doc.add_paragraphs(
        """
        The system should be judged successful if it can download and process market data for a chosen symbol without requiring manual spreadsheet preparation. It should also produce sensible test-period predictions that can be plotted against actual values. Success here does not mean perfect forecasting. It means outputs that are coherent, non-degenerate and analytically meaningful.

        Another success criterion is that future forecasting should produce a plausible projected path together with uncertainty information. The project's later Monte Carlo dropout work is especially relevant here because it moves the system away from misleadingly smooth deterministic curves and toward more believable probabilistic output.

        The project should also be usable. A non-expert user ought to be able to choose a ticker and run the system through the terminal or GUI without having to edit internal code. Finally, the architecture should be maintainable and well evidenced. Clear decomposition, dated implementation progress, stable artifacts and justified bug fixes are all part of what makes this a successful computing project rather than simply a one-off script.
        """
    )

    doc.add_page_break()
    doc.add_heading("Design Decomposition and System Structure", level=1)
    doc.add_paragraphs(
        """
        The project guide asks for top-down decomposition, and the repository's draw.io structure provides strong evidence for the original conceptual breakdown of the system. At the highest level, the solution was separated into a website, a prediction AI, an interface and a data holder. This is a sensible early decomposition because it distinguishes user interaction, analytical logic and stored information rather than treating the whole system as one undifferentiated block.

        The interface branch then expands into a website main page, coin information pages, a log-in screen, a developer menu and a database editor. This shows that the original design expected multiple forms of interaction rather than a single output screen. It also explains why the repository contains prototype web and credential files even though the final strongest implementation is forecasting-centred.

        The prediction branch divides into datasets, a chosen coin and analytical outputs such as historical values and a predicted curve. That branch maps very naturally onto the mature implemented forecasting workflow. The final PyTorch pipeline still follows the same underlying logic: obtain a dataset, select an instrument, transform the history and produce predicted outputs.

        From a final-state perspective, the cleanest decomposition is slightly different. It would now be described as data ingestion, feature engineering, sequence generation, model training, evaluation, future rollout, artifact management, plotting and GUI interaction. The final repository effectively converges toward this more software-centric decomposition by the time the separated `cpp` pipeline and Tkinter launcher exist.
        """
    )

    doc.add_page_break()
    doc.add_heading("Algorithms, IPO Reasoning and Pipelining", level=1)
    doc.add_paragraphs(
        """
        In input-process-output terms, the modern pipeline is clear. Inputs include a ticker symbol, device choice, hyperparameters and externally downloaded market data. Processes include feature engineering, missing-value handling, scaling, sequence construction, training, evaluation, stochastic future rollout and plotting. Outputs include charts, model weights, prediction CSVs, forecast CSVs and terminal logs.

        One of the most important algorithms is sequence generation. The system does not predict from isolated rows. Instead, it takes a rolling lookback window of prior timesteps and uses that structured sequence to predict the next close value. This is the key algorithmic reason an LSTM is appropriate at all. Later versions improve the same logic by replacing slower Python loops with vectorised window generation.

        Another major algorithmic stage is feature engineering. The system transforms basic OHLCV data into richer descriptors such as moving averages, RSI, MACD, signal lines and Bollinger Bands. This is algorithmically important because it gives the model a better representation of trend, momentum and volatility than raw close prices alone.

        The future forecasting algorithm is especially strong because it combines autoregressive rollout with Monte Carlo dropout. Instead of producing a single clean curve and presenting it as certainty, the system generates multiple stochastic next-step outcomes, uses one realisation as the visible path and uses the wider distribution to form confidence intervals. This makes the output more honest and more informative.
        """
    )
    doc.add_heading("Code Exhibit: Future Rollout Algorithm", level=2)
    doc.add_code_block(snippets["cpp_rollout"])
    doc.add_paragraphs(
        """
        This function shows how the system moves from the last known sequence window through repeated next-step prediction into unscaled future prices and confidence bounds. It also reveals a practical modelling assumption: while the close prediction evolves day by day, the other engineered features are approximated from the last known scaled state. That is a sensible coursework trade-off because it keeps the rollout tractable while still producing meaningful forward scenarios.
        """
    )

    doc.add_page_break()
    doc.add_heading("User Interface Design and Wireframe Discussion", level=1)
    doc.add_paragraphs(
        """
        Interface work appears in two major forms across the repository: the early browser prototype and the later Tkinter GUI. Together they show that usability was an important concern throughout the project, even though the final strongest architecture became forecasting-centred rather than web-centred.

        The early browser prototype demonstrates the original idea of a login-based system with gated interaction. It is important as a wireframe-level artifact because it proves the developer considered how a user would move into the system, not just how the system would compute predictions. The existence of a user table and credential logic reinforces that this was intended as a persistent interaction model rather than a single script run.

        The Tkinter GUI is more closely aligned with the completed analytical workflow. It lets the user enter a ticker, choose a device, set key model parameters and read a live output log. The layout separates controls from output, which is a sensible usability decision because it mirrors the natural distinction between configuring an experiment and observing its progress.

        The GUI also serves an important accessibility function. A user who is less comfortable editing source files can still operate the project through visible fields and buttons. This is one of the clearest examples of the repository responding to stakeholder needs rather than remaining purely developer-centric.
        """
    )
    doc.add_heading("Code Exhibit: Tkinter GUI Layout", level=2)
    doc.add_heading("Widget Construction", level=3)
    doc.add_code_block(snippets["gui_create_widgets"])
    doc.add_heading("Command Construction", level=3)
    doc.add_code_block(snippets["gui_build_command"])
    doc.add_paragraphs(
        """
        These excerpts demonstrate that the GUI is not decorative. It actively captures user parameters and transforms them into a runnable forecasting command. In other words, it is a real interface layer sitting on top of the analytical engine, exactly the kind of usability feature the project guide expects to be justified.
        """
    )

    doc.add_page_break()
    doc.add_heading("Key Variables, Data Structures and Validation", level=1)
    doc.add_paragraphs(
        """
        Several key variables control the behaviour of the system. `prediction_days` determines the lookback window length. `future_day` determines the forecast horizon. `epochs`, `batch_size`, `initial_dropout`, `final_dropout` and `num_monte_carlo_runs` all change the training dynamics, runtime cost and uncertainty estimation behaviour. These values are central to the system and not merely incidental constants.

        The main data structures are pandas DataFrames, NumPy arrays and PyTorch tensors. DataFrames handle labelled market data and rolling indicator calculations. NumPy arrays handle shaped feature matrices and sequence windows. Tensors handle model input and gradient-based learning. This progression is appropriate because each structure is used for the form of computation it supports best.

        The final architecture also depends on file-based data structures, especially CSV artifacts for predictions and future forecasts. These become extremely important once training and plotting are separated because they allow results to be reused without rerunning the entire model. The SQL `USERS` table belongs more to the original prototype branch, but it still demonstrates that persistent structured data was part of the project's design thinking from an early stage.

        Validation appears at several levels. User input is validated through ticker selection and search. Preprocessing is validated by fill operations, column checks and shape checks. Sequence construction is validated by dimensional expectations. Training is validated through loss tracking, validation-aware scheduling and general stability checks. This layered validation is one of the reasons the later project is significantly more robust than the early prototypes.
        """
    )

    doc.add_page_break()
    doc.add_heading("Implementation of the Main Training and Plotting Scripts", level=1)
    doc.add_paragraphs(
        """
        This implementation section focuses on the final main code rather than a timeline-style account. The two most important scripts are `pytorch_train_cpp.py` and `pytorch_plot_cpp.py`. The training script is responsible for collecting historical market data, engineering features, building the LSTM model, training it and saving reusable output files. The plotting script is responsible for finding those saved files and turning them into readable charts.

        Separating the scripts was an important design decision. Earlier versions of the project combined training and plotting in one large workflow, which made small visual changes expensive because the model could need to be trained again. The final design avoids that problem by making training produce stable artifacts, then allowing plotting to run as a separate stage. This makes the project easier to test, easier to explain and easier to improve.

        The decision came from observing the practical problems in the code. Training is slow and hardware-dependent, while plotting is quick and mostly presentation-focused. Keeping them together meant that two very different jobs were sharing the same file and runtime. Splitting them gave each script a clearer purpose and made the whole project more maintainable.
        """
    )

    doc.add_heading("Training Script: Output Structure and Artifact Decisions", level=2)
    doc.add_paragraphs(
        """
        The training script saves outputs into predictable folders. This is not just an organisational choice; it is what allows the plotting script and the website to find the correct result later. The model does not only print values to the terminal. It creates files that can be reused by other parts of the system.

        This decision fixes an important development issue: output sprawl. When a project creates charts, prediction CSV files, forecast CSV files and model results, it becomes difficult to know which file belongs to which run. The final script addresses this by standardising where outputs go and by using clear names for prediction and future-forecast files.
        """
    )
    add_comparison(
        doc,
        "Code Exhibit: Training Artifacts and Plot-Side Discovery",
        "Training Script Output Paths",
        snippets["cpp_output_paths"],
        "Plotting Script Discovery Logic",
        snippets["plot_discovery"],
        """
        These snippets show the link between the two main scripts. The training code writes results in a known structure, and the plotting code searches that structure instead of relying on manual file selection. This makes the final project more reliable because the user does not have to remember file names after each run.
        """,
    )

    doc.add_heading("Training Script: Feature Engineering", level=2)
    doc.add_paragraphs(
        """
        The training script does not use only the closing price. It creates additional indicators so that the model has more information about the market's behaviour. This was a justified decision because financial movement depends on more than one value. Trend, momentum and volatility all matter when trying to model price movement.

        The final feature-engineering approach was chosen because the earlier close-price-only approach was too limited. A close-only model can learn broad movement, but it has less context for sudden changes or repeated patterns. Adding indicators gives the model a richer input while still keeping the data understandable enough to explain at A-level standard.
        """
    )
    doc.add_heading("Code Exhibit: Feature Engineering in the Training Script", level=3)
    doc.add_code_block(snippets["cpp_add_features"])

    doc.add_paragraphs(
        """
        A key error class in this part of development was missing or unusable data after rolling calculations. Technical indicators often create empty values at the start of a dataset because they need a number of previous days before they can be calculated. The fix was to make preprocessing clean these values before the model received the data. This prevents the training stage from failing later with confusing tensor or scaling errors.
        """
    )

    doc.add_heading("Training Script: Model Training and Stability", level=2)
    doc.add_paragraphs(
        """
        The model uses recurrent neural-network logic because the data is sequential. Stock and cryptocurrency prices are ordered by time, so the model needs to learn from windows of previous values rather than treating every row as unrelated. The LSTM design was chosen because it is suitable for sequence data and can preserve information across a lookback window.

        The training script also includes stability decisions. Financial data can contain sharp changes, so a fragile model can produce unstable training loss or unrealistic predictions. The use of stronger training controls, validation checks and controlled output generation makes the script more dependable than a basic demonstration model.
        """
    )
    doc.add_heading("Code Exhibit: Training-Time Optimisation Support", level=3)
    doc.add_code_block(snippets["cpp_compile"])
    doc.add_paragraphs(
        """
        Optional compilation was added as a performance decision rather than as a requirement. This is important because the project had to work on real hardware where GPU, CPU and library support could vary. The script therefore tries to improve speed where possible, but it is not designed to collapse if that optimisation is unavailable.
        """
    )

    doc.add_heading("Training Script: Future Forecasting and Uncertainty", level=2)
    doc.add_paragraphs(
        """
        The future-forecasting section is one of the most important parts of the implementation. A single future line can look misleadingly confident, especially in a market where prices can change quickly. The script therefore uses repeated forecasting runs to support uncertainty estimates. This makes the output more honest because it shows that the model is not claiming perfect certainty.

        This design decision came from a visible output problem: forecasts could appear too smooth and too definite. The fix was not simply to make the chart look more dramatic. The better solution was to use repeated model passes so the final output could include a range of possible outcomes. That makes the graph more useful to a user because it communicates risk as well as direction.
        """
    )
    doc.add_heading("Code Exhibit: Monte Carlo Forecast Rollout", level=3)
    doc.add_code_block(snippets["cpp_rollout"])

    doc.add_heading("Plotting Script: Reusing Training Outputs", level=2)
    doc.add_paragraphs(
        """
        The plotting script was created because graph generation should not depend on retraining. Once the training script has produced prediction and future-forecast files, the plotting script can load them and create the visual output. This makes the system quicker to use and easier to debug.

        The main decision in the plotting script is artifact discovery. Instead of asking the user to manually type every file path, the script searches for the latest suitable output. This reduces user error and makes the script work better as part of the larger website workflow.
        """
    )
    doc.add_heading("Code Exhibit: Plotting Script File Discovery", level=3)
    doc.add_code_block(snippets["plot_discovery"])

    doc.add_heading("Plotting Script: Validation and Error Handling", level=2)
    doc.add_paragraphs(
        """
        The plotting stage also needed validation. A chart can only be useful if the input CSV files contain the expected columns and if the dates and predictions line up correctly. During development, errors such as missing files, incorrect output paths and mismatched expectations between scripts were found. The fix was to make the pipeline use consistent file names and to make the plotting script deliberately search the output folders created by training.

        This is why the training and plotting scripts should be understood together. The training script creates a contract by saving specific artifacts. The plotting script follows that contract by searching for and reading those artifacts. The project became more reliable once both sides of that contract were made explicit.
        """
    )

    doc.add_heading("Detailed Code Walkthrough: Training Script", level=2)
    doc.add_paragraphs(
        """
        The following excerpts document the training script in the order it actually runs. This keeps the implementation section focused on the final main code while still giving enough technical evidence to explain the design decisions. The code begins by centralising configuration, then defines the dataset adapter and LSTM model. Centralising the settings was useful because the same script can be driven from the command line, the website queue and direct testing without changing constants in several places.
        """
    )
    doc.add_heading("Code Exhibit: Configuration, Dataset Adapter and LSTM Model", level=3)
    doc.add_code_block(snippets["cpp_config_model"])

    doc.add_paragraphs(
        """
        Sequence construction is another important part of the implementation. The model cannot train on isolated rows because the task is time-series forecasting. Each sample must contain a lookback window and a target value. The script uses NumPy's sliding-window approach where possible because it is faster than building every window manually in Python, but it still includes a fallback route so the code remains portable.
        """
    )
    doc.add_heading("Code Exhibit: Sequence Construction and Test Windows", level=3)
    doc.add_code_block(snippets["cpp_sequence_builders"])

    doc.add_paragraphs(
        """
        The command-line and data-loading section shows how the script becomes reusable. The user can change ticker, device, epochs, batch size, lookback length, forecast horizon and Monte Carlo runs without editing the source. This was a deliberate improvement because the same training script needed to support quick tests and longer final runs.
        """
    )
    doc.add_heading("Code Exhibit: CLI Arguments, Device Selection and Data Preparation", level=3)
    doc.add_code_block(snippets["cpp_main_cli_data"])

    doc.add_paragraphs(
        """
        The training loop applies dynamic dropout and reports progress across epochs and batches. This is a practical choice because it gives the developer feedback during long runs. It also avoids a hidden training process where the user cannot tell whether the model is progressing, stuck or failing.
        """
    )
    doc.add_heading("Code Exhibit: Training Loop", level=3)
    doc.add_code_block(snippets["cpp_training_loop"])

    doc.add_paragraphs(
        """
        After training, the script evaluates recent predictions and saves the model and prediction CSV. This is the point where the training script hands information to the rest of the project. Saving predictions as CSV was chosen because it is easy for Python, PHP and spreadsheet tools to read, which makes the output useful beyond a single script run.
        """
    )
    doc.add_heading("Code Exhibit: Evaluation and Prediction Export", level=3)
    doc.add_code_block(snippets["cpp_eval_save"])

    doc.add_paragraphs(
        """
        The forecast export completes the training side of the implementation. It writes future dates, predicted prices and confidence interval values. This was necessary because the plotting script needs more than just a line of predictions; it also needs uncertainty data if the final chart is going to show how confident or uncertain the forecast is.
        """
    )
    doc.add_heading("Code Exhibit: Forecast Export", level=3)
    doc.add_code_block(snippets["cpp_forecast_save"])

    doc.add_heading("Detailed Code Walkthrough: Plotting Script", level=2)
    doc.add_paragraphs(
        """
        The plotting script starts with argument parsing and validation. This mirrors the training script's design: the user can pass paths manually, but the script can also find the newest suitable output automatically. This supports both manual debugging and website-driven operation.
        """
    )
    doc.add_heading("Code Exhibit: Plot CLI and Prediction CSV Validation", level=3)
    doc.add_code_block(snippets["plot_cli_validation"])

    doc.add_paragraphs(
        """
        The plotting script then downloads fresh market context and rebuilds indicators for the chart. Importantly, this does not retrain the model or change the saved predictions. It only gives the graph enough historical context to be understandable. This separation is one of the main benefits of the final implementation.
        """
    )
    doc.add_heading("Code Exhibit: Plot Context and Future CSV Loading", level=3)
    doc.add_code_block(snippets["plot_context_future"])

    doc.add_paragraphs(
        """
        The main figure combines candlesticks, model predictions, future forecasts, confidence intervals and residuals. The residual plot was added because a graph should not only show the forecast line; it should also reveal where the model was wrong. This makes the output more honest and more useful for evaluation.
        """
    )
    doc.add_heading("Code Exhibit: Main Plot, Confidence Band and Residuals", level=3)
    doc.add_code_block(snippets["plot_figure_residuals"])

    doc.add_paragraphs(
        """
        The second plotting window focuses on the future forecast. It separates price direction from the width of the confidence interval. This is useful because uncertainty can grow or shrink across the forecast horizon, and that information would be harder to see if everything were compressed into a single line chart.
        """
    )
    doc.add_heading("Code Exhibit: Forecast Detail Window and Saved Images", level=3)
    doc.add_code_block(snippets["plot_forecast_window"])

    doc.add_heading("Main Implementation Errors and Fixes", level=2)
    doc.add_paragraphs(
        """
        Several errors shaped the final implementation. One common problem was missing helper files or mismatched paths when the website tried to call Python scripts. The fix was to standardise where scripts live and to make debug pages show the exact helper path being used. This made failures easier to trace because the project could show whether the file existed, whether Python could execute it and which command was being run.

        Another problem was Python environment mismatch. The terminal could sometimes run a command successfully while the website failed, because the web server was using a different Python interpreter or a different set of installed packages. The fix was to point the website configuration at the correct virtual environment and to expose the configured Python path in the debug output.

        A further issue came from native package compatibility. When `yfinance` loaded through the XAMPP environment, NumPy could fail because XAMPP's bundled `libstdc++` was older than the version required by the installed NumPy wheel. The practical fix was to avoid relying on the wrong runtime library path and to use the Python environment that could import the finance packages correctly.

        Queue execution introduced a separate set of errors. The web server user did not always have permission to write queue logs or lock files, which meant the worker could crash before doing real work. The fix was to correct ownership and permissions for the storage folders used by the queue. This mattered because the model-training workflow depends on background jobs rather than only manual terminal commands.

        Remote training also exposed SSH configuration issues. The project encountered missing host keys, strict host-key checking failures, password prompts, wrong key paths and private-key permission errors. These were fixed by using a dedicated key for the web queue, placing it where the web-server user could read it, setting safe permissions and creating a known-hosts file for the Windows training machine. This turned remote execution from an interactive terminal-only process into something the website could run non-interactively.

        These errors are included because they explain why the final implementation looks the way it does. The project did not simply add code until it worked once. It had to make the code runnable from the website, from the queue worker and across two machines. That is why path handling, artifact naming, permission checks and debug pages became important implementation details.
        """
    )

    doc.add_page_break()
    doc.add_heading("Iterative Test Plan", level=1)
    doc.add_paragraphs(
        """
        The iterative test plan focuses on checkpoints used while the software was still under development. This is important because many failures in a project like this occur before the user ever sees a final chart. Data retrieval, indicator creation, sequence-window shape, training stability and artifact writing all need to be checked in stages.

        A first category of iterative tests concerns ingestion and preprocessing. Example tests include verifying that a known ticker returns non-empty data, confirming that expected columns exist, checking that indicators such as RSI and MACD are created successfully and ensuring that rolling-window missing values are resolved before scaling. These tests are justified because every later stage depends on clean structured inputs.

        A second category concerns sequence construction. Here the developer checks that window arrays have the expected dimensions and that target values align correctly with the intended next-step prediction. These tests are essential because off-by-one errors in sequence models are common and can silently damage results.

        A third category concerns model behaviour. Short training runs can be used to check that loss remains finite, that dropout schedules update properly, that clipping prevents instability and that learning-rate reductions respond sensibly when progress slows. In a forecasting project, stable training behaviour is itself a major development milestone.

        A final iterative category concerns output and integration. The developer checks that test-period predictions align with actual values, that future forecasts contain the requested number of days, that confidence intervals are shaped correctly and, in the separated pipeline, that prediction artifacts are written where the plotting script expects them. These tests are justified because they cover exactly the areas that changed repeatedly during development.
        """
    )

    doc.add_page_break()
    doc.add_heading("Post-Development Test Plan", level=1)
    doc.add_paragraphs(
        """
        Post-development testing checks the finished system against the success criteria rather than only validating isolated modules. An end-to-end test should begin with user input and finish with stored or displayed outputs, proving that the complete pipeline works as a joined-up solution.

        A second post-development test should focus on GUI usability. A user ought to be able to enter parameters, run the process and inspect logs without editing source code. This matters because operability is part of the system's effectiveness, not an optional extra.

        A third test should validate the separated pipeline's biggest architectural claim: artifact reuse. After a training run, the plotting script should rediscover the latest predictions and forecasts and render the output without requiring retraining. If this works, it confirms that the final modular architecture delivers one of its main benefits.

        Additional post-development tests should cover resilience to common data quirks and the clarity of the final charts. The expected outcome is not just that the software runs, but that it produces understandable, believable and reusable analytical outputs.
        """
    )

    doc.add_page_break()
    doc.add_heading("Evaluation and Final Reflection", level=1)
    doc.add_paragraphs(
        """
        The project is successful because it demonstrates real iterative software development under real technical constraints. Its greatest strength is not that it followed a perfect straight line. Its greatest strength is that it responded to evidence. The repository shows ideas being tested, limitations being encountered and architectures being improved as the developer learned more about the problem and the environment.

        The strongest completed aspect of the project is the forecasting pipeline, especially in its later PyTorch and separated-pipeline forms. Feature engineering, bidirectional recurrent modelling, Monte Carlo dropout, artifact management and independent plotting together create a technically credible and maintainable analytical application. The move from overly smooth future curves to stochastic paths with confidence intervals is particularly strong because it improves both statistical honesty and user interpretation.

        The area that remained less complete is the original hosted website vision. That should be acknowledged clearly. The repository contains meaningful prototype evidence for the web/database strand, but not a finished secure deployed platform. This does not weaken the project if it is documented honestly. Instead, it shows that the developer prioritised the strongest achievable deliverable and produced a much more mature analytical tool as a result.

        In conclusion, the project meets the guide's intent well when described accurately. It identifies a real problem, applies appropriate computational methods, researches similar systems and libraries, explains design choices, documents iterative implementation with real code evidence and ends with credible testing and evaluation. The final lesson of the repository is larger than forecasting alone: it shows how a project becomes better by learning from its own earlier versions.
        """
    )

    doc.add_page_break()
    doc.add_heading("Bibliography", level=1)
    doc.add_paragraphs(
        """
        The bibliography below uses Harvard-style references with publicly valid links. Internal project files, the project guide and the earlier draft document were also used to build this report, but the bibliography focuses on externally accessible supporting material.
        """
    )
    add_bibliography(doc)

    return doc


def main() -> None:
    docs_dir = Path(__file__).resolve().parent
    output_path = docs_dir / "CS_PROJECT_GUIDE_FULL_REPORT.docx"
    builder = build_report()
    final_path = output_path
    try:
        write_docx(output_path, builder)
    except PermissionError:
        fallback = docs_dir / f"CS_PROJECT_GUIDE_FULL_REPORT_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        write_docx(fallback, builder)
        final_path = fallback
    print(f"Wrote: {final_path}")
    print(f"Body words: {builder.body_word_count}")
    print(f"Code lines: {builder.code_line_count}")
    print(f"Estimated pages: {builder.estimate_pages()}")


if __name__ == "__main__":
    main()
