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

    web_html = repo_root / "web" / "new.html"
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
        "plot_discovery": extract_block(
            p9p,
            'DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_ROOT / "predictions"',
            "    return max(candidates, key=os.path.getmtime)",
        ),
        "web_login_form": extract_block(
            web_html,
            '<form id="loginForm" onsubmit="event.preventDefault(); validateLogin();">',
            "</form>",
        ),
        "web_validate_login": extract_block(
            web_js,
            "function validateLogin() {",
            "function toggleSlowIframe() {",
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
        "15. Project Diary and Iterative Implementation",
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
    doc.add_title("CS Project Sample Document Continuation")
    doc.add_subtitle("Extended version that preserves the earlier sample document direction and finishes it to exceed the minimum page requirement")
    doc.add_subtitle(f"Prepared on {today}")
    doc.add_subtitle("Built from the sample draft, the project guide and the full repository so the previous information remains while the unfinished parts are completed")

    add_manual_contents(doc)

    doc.add_page_break()
    doc.add_heading("Retained Sample Document Material", level=1)
    doc.add_paragraphs(
        """
        This version has been written as a continuation of the sample document rather than as a full replacement. The earlier draft already contained important information about the project's intended direction, especially the idea of a crypto and stock tracking website with login functionality, stored user data, favourite charts and AI-based forecasting. Instead of removing that material, this report keeps the same broad direction and finishes it with clearer explanation, later implementation evidence and the missing detail needed to satisfy the page requirement.

        The sample draft presented the project as a crypto tracking website that would read and write to a database, store user credentials, assign UUIDs and eventually provide AI prediction capability. That framing is preserved here because it genuinely matches the early concept shown in the repository's web prototype files, SQL schema and top-down design work. What has changed in this continuation version is that the unfinished or overly brief areas are expanded using the stronger evidence from the actual project files, especially the mature PyTorch forecasting pipeline.
        """
    )
    doc.add_heading("Project Identification", level=2)
    doc.add_paragraphs(
        """
        The original sample document described the system as a crypto tracking website with a front end and back end, database-backed user credentials, UUID assignment and the ability to favourite price graphs for later use. That project identification is still valid as the starting concept. The aim was to build a system that would help entrepreneurs, shareholders and market-following users inspect price movement more easily while also benefiting from forecast support.

        The continuation of the project showed that the forecasting strand became the most technically complete part of the work. As a result, the final report keeps the original website-and-user framing but explains that the strongest completed implementation is the analytics and forecasting engine that powers the core market-insight side of the solution. This lets the earlier information remain in place while making the final report more accurate.
        """
    )
    doc.add_heading("Stakeholders", level=2)
    doc.add_paragraphs(
        """
        The sample document already identified key stakeholders such as the teacher, potential client-style users and direct testers. That information remains useful because it shows who the system was intended to serve. The most important stakeholder groups remain end users who want market insight, the developer maintaining the solution, and the teacher or assessor who must judge whether the project demonstrates appropriate analysis, design, implementation and testing.

        In expanded form, the stakeholder discussion also includes future maintainers and peer testers. This matters because the final codebase grew significantly in complexity, and maintainability became an important concern in its own right. The separated training-and-plotting pipeline serves not only the end user but also anyone who needs to rerun, inspect or extend the project later.
        """
    )
    doc.add_heading("Justification", level=2)
    doc.add_paragraphs(
        """
        The original justification in the sample document focused on the existence of similar online crypto trackers and the fact that market users need fast access to useful information. That justification remains strong. Financial data is abundant, but raw data is not the same as insight. A purpose-built software system can reduce complexity by gathering data, displaying it more clearly and supporting analysis with technical indicators and predictive modelling.

        A computer program is especially suitable because the project relies on repeated numerical processing, technical-indicator calculations, sequence preparation, statistical learning and chart rendering. These are all tasks that software performs efficiently and consistently. The justification therefore becomes even stronger once the later implemented forecasting pipeline is taken into account, because the repository clearly proves that the software can do more than simply display a static chart.
        """
    )
    doc.add_heading("Example Evaluation", level=2)
    doc.add_paragraphs(
        """
        The sample draft used CoinStats and CoinGecko as comparison points, and that information should stay because it helps explain what similar systems already do well. CoinStats shows the value of a clean, portfolio-aware market interface with easy access to current movements and graphical summaries. CoinGecko shows the importance of broad market visibility, ranked listings and multi-timescale changes that help the user judge context rather than one isolated number.

        Expanding that earlier evaluation leads to a clearer design lesson: users expect financial tools to combine clarity, speed and visual usefulness. They do not only want a number. They want a chart, movement context, comparative data and a workflow that feels repeatable. Those lessons carry directly into the final forecasting application, especially in the later GUI and plotting work.
        """
    )
    doc.add_heading("AI Prediction Websites", level=2)
    doc.add_paragraphs(
        """
        The sample draft also correctly observed that many major market sites focus more on presentation than on direct prediction, which means community or specialist tools often provide the clearest proof of concept for AI forecasting. That observation remains valuable. It justifies why the project needed to look beyond ordinary trackers and toward machine-learning frameworks and forecasting approaches.

        In the final repository, this concern is reflected not just in research notes but in implementation. The project moved through TensorFlow and then PyTorch to build exactly the kind of predictive component that mainstream trackers often leave out or keep behind premium products. In that sense, the original AI-prediction discussion from the sample document is preserved and strengthened by actual code evidence.
        """
    )

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

        A third stakeholder is the teacher or assessor. The project guide itself demonstrates that the solution must be explainable and justifiable, not merely runnable. For this stakeholder, the value of the system depends on evidence of decomposition, algorithm design, appropriate libraries, testing and iterative improvement. This is why a detailed report and dated implementation diary matter as much as the final model output.

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
    doc.add_heading("APIs", level=2)
    doc.add_heading("API Summary", level=3)
    doc.add_paragraphs(
        """
        The project depends on several important libraries, each chosen because it solves a distinct part of the problem. `yfinance` is central because it provides the historical market data required for every later stage. Without an external data-access layer, the project would have no practical source of OHLCV histories for modelling and charting.

        pandas and NumPy form the core data-processing layer. pandas is responsible for time-indexed tables, rolling calculations, date-aware handling and most indicator construction. NumPy becomes especially important in later stages when sequence windows and scaled feature matrices need to be shaped efficiently for model input. In the separated pipeline it also supports more efficient window generation through `sliding_window_view`.

        TensorFlow was used first because it offered a straightforward path into stacked LSTM experimentation and TensorBoard logging. PyTorch was adopted later because it provided more explicit control over device behaviour, model internals and unusual training or inference logic. The migration was not just stylistic. It was justified by the project's need for clearer hardware-aware control and deeper experimentation.

        Matplotlib and mplfinance were chosen because chart output is essential in a financial project. A forecasting system that cannot communicate results clearly through charts is much less useful to both end users and assessors. Tkinter became relevant later because a graphical launcher improves usability for people who may not want to work entirely from the terminal. sqlite3 and Optuna appear in the repository because the project also explored persistence and more formal experiment tracking.
        """
    )
    doc.add_heading("yFinance", level=3)
    doc.add_paragraphs(
        """
        This subsection intentionally preserves the sample document's separate emphasis on yFinance. The earlier draft was right to treat it as a major project dependency because it became the data foundation for both the TensorFlow and PyTorch forecasting stages. Its ability to return Yahoo Finance data directly into pandas-compatible structures made it ideal for rapid experimentation and later pipeline stability.
        """
    )
    doc.add_heading("TensorFlow Documentation", level=3)
    doc.add_paragraphs(
        """
        The sample draft treated TensorFlow as a major researched library, and that remains appropriate because it genuinely powered the early stages of the implementation. The continuation version therefore keeps TensorFlow as a distinct researched technology, while also explaining that it was later replaced in the strongest implementation path by PyTorch due to practical engineering and compatibility pressures.
        """
    )
    doc.add_heading("GitHub REST API", level=3)
    doc.add_paragraphs(
        """
        The earlier document mentioned the GitHub REST API as a possible way to automate access to code or supporting resources. That idea remains part of the research context even though it did not become the defining implementation pillar. It still demonstrates that the project considered broader software integration possibilities rather than viewing the task only as isolated model training.
        """
    )
    doc.add_heading("Library Evaluation", level=2)
    doc.add_heading("SQLite3", level=3)
    doc.add_paragraphs(
        """
        SQLite3 remains relevant because the original concept included stored user data and because later experiment-management work also touched database-backed storage. The sample document was therefore correct to include it as part of the library evaluation. In the continuation version, SQLite3 is understood both as a prototype web/database dependency and as part of the broader tooling story around structured persistence.
        """
    )
    doc.add_heading("Pandas", level=3)
    doc.add_paragraphs(
        """
        Pandas is one of the most justified libraries in the entire project. The sample draft emphasised its value for dataset handling, and the implemented repository fully confirms that view. It is central to historical data processing, rolling indicators, fill operations and many intermediate transformations required before tensors are built.
        """
    )
    doc.add_heading("Numpy", level=3)
    doc.add_paragraphs(
        """
        NumPy deserves separate mention because the project repeatedly relies on fast numerical array manipulation. In the later separated pipeline, NumPy also supports more advanced and efficient sequence-window generation, showing that it was not just a background dependency but an important performance enabler.
        """
    )
    doc.add_heading("Matplotlib", level=3)
    doc.add_paragraphs(
        """
        Matplotlib remains one of the key visualisation libraries in the report because the project's outputs must be interpretable. The sample document was correct to identify graphing as essential. The continuation version simply extends that by showing how later charts became richer, more polished and more tightly integrated with prediction and uncertainty output.
        """
    )
    doc.add_heading("Pickle", level=3)
    doc.add_paragraphs(
        """
        The sample document included Pickle as part of its library discussion because serialization is useful when preserving processed objects, figures or model-related state. Even though later stages favour more explicit artifact files such as CSV outputs, the earlier inclusion of Pickle remains valid as part of the research and experimentation context.
        """
    )
    doc.add_heading("TensorFlow", level=3)
    doc.add_paragraphs(
        """
        Keeping TensorFlow as a separate library-evaluation subsection preserves the sample document accurately. TensorFlow was not merely researched; it was genuinely used. This is why the continuation document retains it as a significant part of the narrative rather than pretending the project was PyTorch-only from the beginning.
        """
    )
    doc.add_heading("PyTorch (Replaced TensorFlow Due to Compatibility and Control Issues)", level=3)
    doc.add_paragraphs(
        """
        The sample draft already acknowledged that PyTorch replaced TensorFlow due to unforeseen compatibility issues. The repository strongly supports that summary. The continuation version keeps that statement but expands it with fuller explanation of device control, hardware behaviour, experimentation flexibility and long-term maintainability.
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
    doc.add_heading("Possible Issues", level=2)
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
    doc.add_heading("Design", level=1)
    doc.add_heading("Design Decomposition and System Structure", level=2)
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
    doc.add_heading("UI Wireframe", level=1)
    doc.add_heading("User Interface Design and Wireframe Discussion", level=2)
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
    doc.add_heading("Project Diary and Iterative Implementation", level=1)
    doc.add_paragraphs(
        """
        The project diary is the strongest part of the report because it demonstrates how the solution evolved through evidence rather than guesswork. The repository preserves early learning-path files, later refactors and enough dated history to build a realistic timeline. This makes it possible to show not only what the current code does, but also why each stage existed and what technical pressure produced the next rewrite.
        """
    )

    doc.add_heading("Phase A: Original Concept and Prototype Direction", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        The website and credential prototypes are consistent with the earlier draft document and with repository files that appear by December 2025. The draw.io system decomposition, prototype HTML/JavaScript files and SQL schema show that the earliest project direction was active by the end of 2025.
        """,
    )
    doc.add_paragraphs(
        """
        The earliest stage was broad in ambition. The developer was not only trying to predict prices. They were imagining a joined-up platform in which users could log in, revisit information and benefit from AI-assisted charts. This explains the presence of website prototypes and database design even though the final strongest deliverable is the forecasting pipeline.

        This stage is valuable because it identified the software problem in the right broad terms: user interaction, data persistence, market data presentation and predictive support. It is also valuable because it reveals limitations early. Front-end validation and hardcoded credentials are acceptable for exploration, but not for final secure deployment. The project learned from this by narrowing its strongest implementation effort toward the analytical engine.
        """
    )

    doc.add_heading("Phase B: TensorFlow Basics", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        `ai_predict.py` and related TensorFlow-basics files carry file timestamps from 12 December 2025, placing the earliest concrete ML implementation in mid-December 2025.
        """,
    )
    doc.add_paragraphs(
        """
        The first machine-learning implementation used TensorFlow/Keras and aimed to prove that an LSTM could operate on financial history at all. This close-only baseline was deliberately simple: obtain price history, scale it, build sequence windows, train a stacked recurrent model and compare actual against predicted values on a chart.

        Although simple, this phase established several ideas that survived the entire project: lookback windows, external data retrieval with `yfinance`, visual comparison between prediction and reality, and the basic idea that market forecasting could be automated through a repeated computational pipeline. These concepts became the backbone of every later rewrite.

        The phase also exposed immediate weaknesses. Using only close price limited the model's view of market structure. The script was monolithic, with downloading, training, evaluation and plotting all tangled together. These weaknesses did not invalidate the phase; they created the next set of design questions.
        """
    )
    doc.add_heading("Code Exhibit: Close-Only TensorFlow Baseline", level=3)
    doc.add_code_block(snippets["tf_close_only"])

    doc.add_heading("Phase C: Dynamic Dropout and Better Training Control", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        The stacked-LSTM and dynamic-dropout learning-path files share December 2025 creation times, indicating that the latter part of the TensorFlow learning stage focused on improving training behaviour rather than merely proving that a model could run.
        """,
    )
    doc.add_paragraphs(
        """
        Once the baseline worked, the project became more interested in how training was controlled. Static dropout regularised the network, but it did so with the same intensity throughout the training process even though early and late epochs benefit from different levels of regularisation. The dynamic-dropout stage addressed that weakness directly.

        This stage is important because it shows the developer beginning to reason about optimisation quality rather than only raw functionality. The system was no longer just being asked to produce a curve. It was being asked to learn in a more disciplined and observable way. TensorBoard logging and callback-driven dropout updates are evidence of that shift.
        """
    )
    add_comparison(
        doc,
        "Code Exhibit: Static Regularisation Becomes Dynamic",
        "Earlier Static Baseline",
        snippets["tf_close_only"],
        "Dynamic Dropout Callback",
        snippets["dynamic_dropout"],
        """
        The significance of this comparison is conceptual as much as technical. In the earlier code, dropout is just part of the architecture definition. In the later code, dropout becomes something the training loop actively manages over time. That is a meaningful increase in sophistication and a clear example of iterative implementation.
        """,
    )

    doc.add_heading("Phase D: Extended TensorFlow Forecasting", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        `future-Eric.py` has a last write date of 4 January 2026, placing the extended TensorFlow phase at the turn of the year as the project moved from proof of concept into richer modelling.
        """,
    )
    doc.add_paragraphs(
        """
        This phase marks the moment where the project stopped being a simple close-price predictor and started becoming a richer forecasting system. Additional derived features, stronger preprocessing logic and better training control all appear here. The developer was beginning to understand that realistic market modelling requires more context than a single line of historical closes.

        It is also the phase where the cost of complexity began to rise. TensorFlow could still execute the ideas, but the project was increasingly pushing toward custom behaviour and more explicit control. That is why this phase should be read not only as a technical improvement but as preparation for the later migration to PyTorch.
        """
    )
    add_comparison(
        doc,
        "Code Exhibit: Feature Expansion in TensorFlow",
        "Close-Only Baseline",
        snippets["tf_close_only"],
        "Multi-Feature TensorFlow Preparation",
        snippets["tf_features"],
        """
        This comparison captures one of the most important project-wide shifts. The model is no longer starved of context. Instead, the system deliberately constructs trend, momentum and timing information before training begins. That single change greatly improves the credibility of the overall analytical approach.
        """,
    )

    doc.add_heading("Phase E: PyTorch Transition", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        `future-eric-pytorch.py` has a last write date of 26 January 2026, and formal git milestones around PyTorch and Optuna appear from 22 February 2026 onward. This suggests the transition began in late January and became central by late February 2026.
        """,
    )
    doc.add_paragraphs(
        """
        The migration to PyTorch was a genuine turning point. The developer needed more explicit control over devices, clearer forward-pass logic and a framework that made unusual training and inference behaviour easier to express. The move was therefore justified by practical engineering pressure, not just by preference.

        PyTorch also aligned better with the way the project had begun to develop. Once experimentation became deeper and hardware issues more visible, an explicit `nn.Module`-based approach made the code easier to reason about. Even where early PyTorch files still carried ideas from the TensorFlow era, the rewrite created room for later stability improvements, hardware workarounds and richer modelling.
        """
    )
    doc.add_heading("Code Exhibit: Initial PyTorch Rewrite", level=3)
    doc.add_code_block(snippets["pytorch_transition_model"])

    doc.add_heading("Phase F: PyTorch Stable Core", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        `pytorch_fixed.py` was created on 27 March 2026 and refined further into early April 2026, representing the point where the PyTorch branch became a stable baseline rather than only a migration experiment.
        """,
    )
    doc.add_paragraphs(
        """
        The stable-core phase focused on robustness. Bidirectional layers, Huber loss, gradient clipping and validation-aware learning-rate scheduling all show that the project was becoming more concerned with dependable training behaviour. This is the phase where the repository starts to feel less like a direct port and more like a deliberate PyTorch-native implementation.

        That matters because successful forecasting software is not only about architectural ambition. It is also about avoiding fragility. Recurrent models can become unstable, and financial data contains outliers and sharp moves. The stable-core stage addressed those realities directly.
        """
    )
    add_comparison(
        doc,
        "Code Exhibit: Stable Core Modelling and Training",
        "Bidirectional Stable Model",
        snippets["pytorch_stable_model"],
        "Robust Training Controls",
        snippets["pytorch_stable_training"],
        """
        Together, these excerpts show the codebase entering a mature optimisation mindset. Architecture and training control are designed together, and the system begins defending itself against instability rather than merely hoping it does not occur.
        """,
    )

    doc.add_heading("Phase G: Full PyTorch Pipeline and Rich Forecasting", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        Major git milestones appear on 16 March 2026, 18 March 2026, 24 March 2026, 27 March 2026, 29 March 2026, 7 April 2026 and 9 April 2026. This is the densest development period in the repository and corresponds to the growth of `pytorch_plotted.py` into a full-feature pipeline.
        """,
    )
    doc.add_paragraphs(
        """
        `pytorch_plotted.py` is the high point of the monolithic era. It combines market-data retrieval, technical indicators, interactive ticker selection, backend management, logging, model training and chart rendering in one very capable file. This script matters because it concentrates a huge amount of project knowledge in one place.

        It also contains some of the most important engineering lessons in the repository. Hardware control becomes explicit rather than assumed. Backend settings are adjusted for real development-machine constraints. Mixed runtime choices and logging concerns become first-class parts of the code rather than background details.

        The most memorable improvement in this phase is the solution to the smooth-forecast problem. Earlier future trajectories looked too clean to be believable. By using Monte Carlo dropout during inference, keeping dropout active and separating the visible path from the confidence-band logic, the project produced outputs that were more honest and more visually plausible.

        At the same time, this phase created the pressure that led to the next architectural shift. The script became so capable that it also became expensive to maintain. Training, plotting and runtime configuration all lived together. Success was creating complexity faster than the monolithic structure could comfortably absorb.
        """
    )
    add_comparison(
        doc,
        "Code Exhibit: Hardware-Aware Full Pipeline",
        "Backend and Runtime Controls",
        snippets["pytorch_backend"],
        "Stochastic Monte Carlo Inference",
        snippets["pytorch_mc"],
        """
        These excerpts show why the file was both impressive and difficult to evolve. It contained deep knowledge about hardware behaviour and custom forecasting logic in the same operational space. That concentration of capability is one reason the file was powerful, and also one reason separation of concerns became the next sensible step.
        """,
    )

    doc.add_heading("Phase H: Experimental Branches", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        `pytorch_additional.py` carries a creation date of 24 March 2026 and evolved through the late-March and early-April period as a parallel experimentation branch.
        """,
    )
    doc.add_paragraphs(
        """
        The experimental PyTorch branch shows that the project had reached a level of confidence where it could test more ambitious ideas such as attention mechanisms, broader feature sets and deeper architectural variation. This is important evidence of technical curiosity and iterative ambition.

        These experiments also sharpened the case for modular architecture. The more ambitious the models became, the more costly it was to keep everything inside a single dense operational script. The experimental files therefore add value twice: they record serious exploration, and they also explain why the final architecture needed cleaner boundaries.
        """
    )
    doc.add_heading("Code Exhibit: Attention-Oriented Experimental Model", level=3)
    doc.add_code_block(snippets["attention_model"])

    doc.add_heading("Phase I: Separated Training and Plotting Pipeline", level=2)
    add_timeline(
        doc,
        "Dates",
        """
        `pytorch_train_cpp.py` and `pytorch_plot_cpp.py` were created on 9 April 2026, and commit `f9efa69` on 10 April 2026 reorganised the project and standardised the new output structure.
        """,
    )
    doc.add_paragraphs(
        """
        The separated `cpp` pipeline is the most important architectural improvement in the repository. The training script became responsible for data ingestion, feature engineering, model work and artifact creation. The plotting script became responsible for discovering those artifacts and rendering charts. This separation improved maintainability, reproducibility and iteration speed at the same time.

        The new architecture also distilled lessons learned in earlier phases. It flattened `MultiIndex` columns, organised outputs, introduced more efficient window generation and added optional compilation support. These decisions are not isolated refinements. They are the accumulated response to earlier pain points around fragility, runtime cost and output sprawl.
        """
    )
    add_comparison(
        doc,
        "Code Exhibit: Separated Pipeline Infrastructure",
        "Stable Artifact Paths",
        snippets["cpp_output_paths"],
        "Plot-Side Artifact Discovery",
        snippets["plot_discovery"],
        """
        These excerpts capture the essence of the final architecture. Results are stored intentionally and rediscovered intentionally. The system no longer assumes that heavy training must be rerun just to change visual output. That is a major gain in usability and engineering quality.
        """,
    )
    doc.add_heading("Code Exhibit: Optional Throughput Optimisation", level=3)
    doc.add_code_block(snippets["cpp_compile"])
    doc.add_paragraphs(
        """
        The compile hook is a small but telling sign of maturity. The project now treats performance as an engineering concern while still degrading gracefully when that optimisation path is not suitable. That is an appropriate end-state attitude for a codebase that has already had to negotiate real hardware-specific constraints.
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
    output_path = docs_dir / "CS_PROJECT_SAMPLE_CONTINUED_REPORT.docx"
    builder = build_report()
    final_path = output_path
    try:
        write_docx(output_path, builder)
    except PermissionError:
        fallback = docs_dir / f"CS_PROJECT_SAMPLE_CONTINUED_REPORT_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        write_docx(fallback, builder)
        final_path = fallback
    print(f"Wrote: {final_path}")
    print(f"Body words: {builder.body_word_count}")
    print(f"Code lines: {builder.code_line_count}")
    print(f"Estimated pages: {builder.estimate_pages()}")


if __name__ == "__main__":
    main()
