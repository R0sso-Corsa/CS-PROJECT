from __future__ import annotations

import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W = f"{{{W_NS}}}"

ET.register_namespace("w", W_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("xml", XML_NS)


DOC_PATH = Path(r"C:\Users\paron\Desktop\Dev\CS_PROJECT\REWRITE\docs\CS_project_Copy_preserved_extended.docx")
HISTORY_DIR = DOC_PATH.parent / "history"


HEADING_REPLACEMENT = "Python Automation and Supporting Scripts"


BODY_OVERRIDES: dict[int, str] = {
    10: (
        "My project is a crypto-tracking platform with a developed front end and back end that reads from "
        "and writes to a database so I can store user credentials, assign each account a UUID "
        "(Universally Unique Identifier), and let users favourite crypto graphs so they can return to "
        "them later. I also planned for the platform to include AI-based market prediction and account "
        "profiles if time allowed, because I wanted the finished system to combine practical tracking "
        "features with analytical support rather than acting as a simple price viewer."
    ),
    11: (
        "I planned to build the project across several web pages using HTML, CSS, JavaScript, PHP and "
        "Python, with SQL databases handling stored credentials and other persistent information. I use "
        "Git for version control because the project changes across several languages, files and stages, "
        "and I need a reliable record of what I changed, when I changed it and why I changed it. That "
        "matters to me because I regularly test risky ideas, such as new model logic, interface changes "
        "or database adjustments, and version control gives me a safe checkpoint to return to if an "
        "experiment fails. It also lets me compare older and newer versions of the same feature, track "
        "how the coursework evolved over time, and show clear evidence of iterative development rather "
        "than pretending the final version appeared all at once."
    ),
    12: (
        "For hosting user data in a way that can be accessed over a network, I planned around a Linux "
        "environment and a router or modem that could help direct traffic to the website. The website "
        "was designed to include a log-in system in which user credentials would be stored in a database "
        "inside my Linux Mint environment and then accessed through SQL queries in an external Python "
        "script. I also designed a debug menu that would only be reachable through a specific UUID that "
        "I would assign to myself. User account UUIDs would be generated with Python's `uuid` library "
        "using UUID V4 strings, which rely on random generation so that previously generated IDs are not "
        "reused. Those UUIDs would then be stored in an SQL database where I could monitor and edit them "
        "when necessary as part of development and testing."
    ),
    14: (
        "I intended the website to be useful for entrepreneurs and shareholders who want to monitor stock "
        "market activity while also receiving AI-assisted predictions that could support more informed "
        "trading decisions."
    ),
    21: (
        "To judge whether my idea was realistic, I researched examples that already existed online. When "
        "I searched for the phrase \"Crypto Tracker\", I found many websites dedicated to showing crypto "
        "prices, graphs and recent market movement for users who need quick access to that information. "
        "Some of them also offered predicted values, although those prediction features were usually kept "
        "behind paid subscriptions because the owners had invested in training their own AI systems."
    ),
    23: (
        "I found several examples of similar projects that already offered user-friendly layouts and clear "
        "metric descriptions, which helped me judge what my own interface and feature set should aim for."
    ),
    24: (
        "The first example I evaluated was CoinStats (https://coinstats.app). It uses a dark interface "
        "with rounded icons and soft edges, which gives it a sleek and modern look. On the homepage I "
        "could immediately see options for connecting portfolios, along with a section showing the top "
        "five cryptocurrencies and their price graphs. It also presents percentage change information, "
        "and when I selected a coin I was shown a more detailed summary including market cap, price and "
        "different viewing modes such as Price, TradingView, MarketCap and AvailableSupply."
    ),
    28: (
        "Another example I looked at was CoinGecko. Its homepage presents a very large amount of market "
        "data, including headline statistics such as total market capitalisation across all currencies. "
        "Its coin listings include the current price, one-hour change, twenty-four-hour change, seven-day "
        "change, daily volume and a graph showing how the price moved over the previous week. I also "
        "noticed a range of filters, including highlights and category views, which helped me understand "
        "how a market-tracking platform can support both quick browsing and deeper analysis."
    ),
    31: (
        "Although those sites are strong examples of crypto tracking, I noticed that prediction features "
        "are usually not their main focus. Because of that, I also looked at community-driven projects "
        "and smaller AI-focused platforms. These projects were often more cluttered, but they still gave "
        "me proof that price prediction interfaces were possible. One larger and more official-looking "
        "example I researched was Crypticorn, which presents crypto prices alongside possible future "
        "prices and a most-likely projection."
    ),
    33: (
        "The Crypticorn dashboard also shows metrics beyond raw prices and predicted fluctuations, such "
        "as a market-trends meter linked to fear-and-greed style behaviour. I found that useful because "
        "it demonstrates how prediction can be combined with broader market context instead of being shown "
        "as an isolated number. The system appears to consider several factors, including technical "
        "indicators, macroeconomic news, on-chain data, search activity and chart patterns. That gave me "
        "a stronger idea of the types of features that a more advanced prediction tool can consider."
    ),
    41: (
        "The websites I researched rely heavily on APIs, or Application Programming Interfaces, which are "
        "sets of rules that allow software components to communicate with one another. In simple terms, "
        "they let one piece of software request data or services from another. An API layer was clearly "
        "important for my own work as well, especially because the forecasting side of my project depends "
        "on reliable access to market data. One of the tools I expected to use most was `yfinance`, since "
        "it can gather data from Yahoo Finance after the official Yahoo Finance API was discontinued. "
        "That made my data-access layer much simpler to build than if I had needed to construct the whole "
        "request process manually."
    ),
    43: (
        "`yfinance` is a popular open-source Python library that gives me free access to financial market "
        "data from Yahoo Finance, including cryptocurrency prices. It can download historical and near "
        "real-time OHLCV data for assets such as BTC-USD and ETH-USD without requiring API keys or paid "
        "subscriptions. I value it because it returns data as pandas DataFrames, supports intervals from "
        "one minute to monthly data, and integrates naturally with libraries such as scikit-learn, "
        "TensorFlow and PyTorch. In my project it acts as the foundational data layer for training, "
        "indicator creation, historical analysis and later forecasting runs."
    ),
    45: (
        "TensorFlow is an open-source machine-learning and artificial-intelligence library originally "
        "developed by Google's Brain team and released publicly in 2015. I researched it heavily because "
        "it offered an accessible way to build deep-learning models in Python while still relying on a "
        "high-performance backend in C++. Its Keras integration makes neural-network construction easier "
        "to understand, and the wider platform includes tools such as TensorBoard, LiteRT and TensorFlow "
        "Extended, which is why it is often treated as an end-to-end environment for the full machine-"
        "learning lifecycle."
    ),
    46: (
        "Even with TensorFlow's strengths, I understood that cryptocurrency prediction would still be "
        "difficult because markets are volatile and influenced by news, sentiment, macroeconomic events "
        "and trader behaviour. I also knew that hardware behaviour, randomness and GPU-specific execution "
        "could affect training consistency. That did not make TensorFlow useless; it simply meant that I "
        "had to treat predictions as probabilistic and experimental rather than as guaranteed answers. "
        "What still made TensorFlow valuable to me was its flexibility for testing stacked LSTMs, hybrid "
        "architectures, attention-style ideas and other modelling approaches early in the project."
    ),
    47: HEADING_REPLACEMENT,
    48: (
        "I also researched Python-based automation because my project benefits from being able to launch "
        "supporting scripts, move data through repeatable stages and reduce manual setup between runs. "
        "This became especially important once the project grew beyond a single training script and began "
        "to include preprocessing, plotting, artifact generation and launcher logic."
    ),
    49: (
        "Using Python automation tools such as `subprocess` allowed me to think about how different parts "
        "of the system could be orchestrated cleanly. Rather than expecting every task to happen in one "
        "large file, I could separate responsibilities and still run them in a controlled order. That "
        "approach supports maintainability because each step can be tested, replaced or debugged without "
        "rewriting the entire application."
    ),
    50: (
        "This automation-focused approach also improves efficiency and traceability. By running supporting "
        "scripts in a structured way, I can keep outputs consistent, reduce repeated manual mistakes and "
        "make the workflow easier to explain in coursework terms. It fits the wider goal of my project, "
        "which is not just to produce a prediction once, but to build a repeatable analytical pipeline."
    ),
    53: (
        "To make my project run properly in Python, I need several libraries for both compatibility and "
        "functionality. Each one supports a different part of the workflow, from storing data and handling "
        "tables to training models and displaying predictions."
    ),
    55: (
        "SQLite3 is a C library that gives me a lightweight disk-based database without needing a separate "
        "server process. I considered it useful because it is fast, responsive and widely deployed across "
        "websites, operating systems, mobile devices and embedded systems. Its history also supports its "
        "reputation for reliability. In my project, SQLite3 made sense both for the early account-based "
        "website concept and for any structured data I needed to preserve during development."
    ),
    56: "To open a connection to a database in Python, I would use the `sqlite3.connect` method with the appropriate database name.",
    58: "To run SQL statements after connecting, I would create a database cursor and use that cursor as the object that executes queries.",
    60: "Once the cursor exists, I can interact with the database by calling the `execute` method with the SQL statement I want to run.",
    62: "After making a change to the database, I would call `commit()` so the update is written to disk rather than staying only in memory.",
    64: "When I have finished all the required interactions, I can close the connection cleanly with `close()`.",
    66: "That final step matters because it helps ensure all changes have been written safely before Python ends the connection.",
    69: (
        "Pandas is a Python library that I rely on for working with datasets. It can inspect, manipulate "
        "and display structured data, which is why it is widely treated as a core tool in data science."
    ),
    70: (
        "I need pandas because it works alongside many other analytical and forecasting libraries, and "
        "because it helps me clean data and produce more reliable inputs for the prediction pipeline."
    ),
    71: (
        "Pandas was designed mainly for tabular data, which fits my project well because much of the "
        "data I process behaves like spreadsheets or CSV-style tables. That makes it ideal for reading, "
        "writing and transforming the financial datasets used by my algorithm."
    ),
    74: (
        "NumPy is a fundamental package for scientific computing and numerical data handling. I use it "
        "because it gives me efficient array operations, indexing and vectorisation, all of which matter "
        "once financial data has to be reshaped into model-ready inputs. Since much of the library is "
        "implemented in C and C++, it also supports the speed and efficiency I need for repeated "
        "forecasting experiments."
    ),
    76: (
        "Matplotlib is my main Python graphing tool for visualising statistics in different formats. I "
        "use it to present LSTM results through line graphs and other plots so the model output can be "
        "interpreted visually rather than only through raw numbers."
    ),
    79: (
        "Pickle is a serialization and deserialization library that I considered for preserving data, "
        "figures and model-related state between runs. In practice, that means it can help me save a "
        "trained model, store a Matplotlib figure for later extraction or keep prediction results "
        "available without rebuilding them from scratch. It is useful in controlled internal workflows, "
        "although I also recognise that it is unsafe for untrusted data because unpickling can execute "
        "malicious Python callables."
    ),
    83: (
        "TensorFlow is a machine-learning library developed by Google and closely tied to the Keras "
        "framework. I explored it because it let me build neural networks with a relatively approachable "
        "high-level interface while still benefiting from strong hardware acceleration and automatic "
        "differentiation underneath. That made it a sensible starting point for my early LSTM work."
    ),
    87: (
        "PyTorch is a deep-learning library built largely in C++ and widely treated as a serious rival to "
        "TensorFlow. I ended up valuing it because it gave me clearer control over device behaviour, a "
        "more direct Pythonic coding style and stronger practical compatibility with the hardware path I "
        "was testing. I also found its separation into `torch`, `torch.nn` and `torch.optim` useful, "
        "because that made the flow between tensor handling, network construction and optimisation easier "
        "for me to reason about."
    ),
    90: (
        "`torch_optimizer` is a drop-in extension for PyTorch's native optimiser module, and I researched "
        "it because it expands the set of training strategies I can test without rewriting the rest of my "
        "model code. It includes a large collection of research-oriented optimisers such as DiffGrad, "
        "AdaBelief, AdaBound and Ranger, each of which approaches convergence and training stability in a "
        "different way. What mattered most to me was that these optimisers could be swapped into an "
        "otherwise standard PyTorch workflow with very little structural change."
    ),
    94: (
        "One issue I identified was the server laptop that I originally expected to host the data and run "
        "the prediction process. The Linux Mint version on that machine did not properly support the "
        "Python environment I needed for TensorFlow, which created compilation problems and prevented the "
        "algorithm from running to completion. That forced me to think about fallback plans instead of "
        "assuming the original environment would work perfectly."
    ),
    95: (
        "One possible fallback would have been to run the prediction algorithm on a client's device, but "
        "I considered that both unreliable and ethically poor, so I treated it only as a last resort. A "
        "much more realistic option was to generate the predictions on my home device and then transfer "
        "the resulting data to the Linux server for hosting."
    ),
    96: (
        "My home computer is much more capable of running AI workloads, especially on the GPU. I planned "
        "to use my AMD 9070 XT with 16GB of VRAM rather than relying only on the CPU, an AMD Ryzen 5 "
        "9600X with 6 cores and 12 threads. I also looked at WSL2 because my preferred AI stack did not "
        "fully support the exact GPU path I wanted under native Windows. That introduced extra complexity, "
        "but it gave me a practical route toward making the stronger hardware available to the project."
    ),
    97: (
        "I understood that this approach also introduced risk. Because the setup was not fully native, I "
        "could run into compatibility issues, missing packages or long periods of troubleshooting. Even "
        "so, I saw it as a worthwhile trade-off if it meant avoiding a larger operating-system change "
        "while still getting the model running on the hardware I already owned."
    ),
    98: (
        "My plan was to try the Hypervisor route first through a Windows batch installer, then move to a "
        "fuller Windows installation upgrade if software compatibility made the first option too unstable "
        "or too slow."
    ),
    99: (
        "If major complications arose, I was prepared to consider unofficial tools for switching Windows "
        "versions, although I preferred to avoid them because of the risks and controversy attached to "
        "that route."
    ),
    116: "https://pytorch-optimizers.readthedocs.io/en/latest/",
    123: "Smith, E. (2025). Setting up Your API Key. [online] CoinGecko API Documentation. Available at: https://docs.coingecko.com/reference/setting-up-your-api-key [Accessed 5 Jun. 2025].",
    124: "sqlite3 - DB-API 2.0 interface for SQLite databases (no date) Python documentation. Available at: https://docs.python.org/3/library/sqlite3.html (Accessed: June 13, 2025).",
    125: "Requests (no date) PyPI. Available at: https://pypi.org/project/requests/ (Accessed: June 13, 2025).",
    126: "Quickstart - requests 2.32.4 documentation (no date) Readthedocs.io. Available at: https://requests.readthedocs.io/en/latest/user/quickstart/ (Accessed: June 13, 2025).",
    127: "Python Software Foundation. (no date) subprocess - Subprocess management. [online] Available at: https://docs.python.org/3/library/subprocess.html.",
    128: "AMD (2025). Install TensorFlow for ROCm. [online] Available at: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html [Accessed 27 Jun. 2025].",
    146: (
        "I added the following pages to extend the existing sample document without removing or altering "
        "the material, photos, headings or layout that were already present. I designed this addendum to "
        "finish the document against the project guide by adding deeper explanation, implementation "
        "detail, code evidence, testing discussion and evaluation while still leaving the original content "
        "intact."
    ),
    147: (
        "Because my original document already contained project identification, example research, library "
        "discussion and images, I preserved those sections as part of the base file. The additional "
        "sections below act as an expansion rather than a replacement, which allows me to keep the "
        "original structure and media while comfortably exceeding the minimum page requirement."
    ),
    150: (
        "My project is strongly suited to computational methods because I rely on abstraction, "
        "decomposition, logical sequencing, iteration and large structured datasets throughout the build. "
        "The task is not just to display a price; it is to transform continuous market history into "
        "repeatable program steps that software can execute reliably."
    ),
    151: (
        "I use abstract thinking when I reduce messy market behaviour into usable variables and engineered "
        "indicators. Instead of treating financial movement as an unstructured narrative, I represent it "
        "through columns such as Open, High, Low, Close and Volume, together with indicators such as SMA, "
        "RSI, MACD and Bollinger Bands. That abstraction removes unnecessary detail while preserving the "
        "information my software can process meaningfully."
    ),
    152: (
        "I also show forward planning in the way I prepare sequence windows, scaling behaviour and artifact "
        "handling. Every prediction depends on a lookback period, a consistent feature matrix and the "
        "ability to invert scaling later when I plot results. The separated `cpp` pipeline shows the same "
        "kind of planning by creating dedicated directories for models, predictions and future forecasts "
        "before expensive computation begins."
    ),
    153: (
        "Procedural thinking and decomposition are among the clearest strengths of my repository. I keep "
        "breaking the larger problem into smaller tasks: fetch data, create features, handle missing "
        "values, scale the matrix, construct sequences, train the model, evaluate predictions, forecast "
        "future values, save artifacts and render plots. Later refactors improve this even further by "
        "separating training from plotting, which makes the solution easier to manage and test."
    ),
    154: (
        "Logical thinking is equally essential because my whole pipeline depends on the correct order of "
        "steps. Feature engineering has to happen before scaling, scaling has to happen before sequence "
        "generation, training has to happen before inference, and artifact generation has to happen "
        "before plot-only scripts can rediscover results. If I break that order, the outputs become "
        "invalid or the program fails."
    ),
    157: (
        "The primary stakeholder in my project is the end user who wants to inspect market instruments "
        "more effectively. In my earliest concept, I imagined that user interacting with a website, "
        "storing favourites and revisiting information through an account. In the final form I completed, "
        "that same user benefits more directly from a forecasting application that can download data, "
        "generate indicators and present probabilistic charts."
    ),
    158: (
        "A second major stakeholder is me as the developer, because I am also the maintainer, tester and "
        "system integrator. My repository makes it clear that maintainability became more important as the "
        "project grew. Decisions such as separating plotting from training, improving logging and "
        "standardising output folders all make the project easier for me to understand, extend and debug."
    ),
    159: (
        "A third stakeholder is my teacher or assessor. The project guide shows that I need the solution "
        "to be explainable and justifiable, not merely runnable. Because of that, evidence of "
        "decomposition, algorithm design, appropriate library choice, testing and iterative improvement is "
        "just as important as the final model output."
    ),
    160: (
        "A fourth stakeholder is the tester or peer reviewer. Someone other than me may need to operate "
        "the GUI, inspect charts, interpret logs or comment on usability. That is why `gui_main.py` is so "
        "relevant: it lowers the barrier to using the project without requiring direct edits to the source "
        "code."
    ),
    171: (
        "My project depends on several important libraries, each chosen because it solves a distinct part "
        "of the problem. `yfinance` is central because it gives me the historical market data required for "
        "every later stage. Without an external data-access layer, I would not have a practical source of "
        "OHLCV histories for modelling and charting."
    ),
    172: (
        "pandas and NumPy form the core of my data-processing layer. I use pandas for time-indexed tables, "
        "rolling calculations, date-aware handling and most indicator construction. NumPy becomes "
        "especially important later when I need to shape sequence windows and scaled feature matrices "
        "efficiently for model input. In the separated pipeline it also supports faster window generation "
        "through `sliding_window_view`."
    ),
    173: (
        "I used TensorFlow first because it gave me a straightforward route into stacked LSTM experiments "
        "and TensorBoard logging. I later adopted PyTorch because it gave me more explicit control over "
        "device behaviour, model internals and unusual training or inference logic. That migration was not "
        "just stylistic; it was driven by my need for clearer hardware-aware control and deeper "
        "experimentation."
    ),
    174: (
        "I chose Matplotlib and mplfinance because chart output is essential in a financial project. If my "
        "forecasting system cannot communicate results clearly through charts, it becomes much less useful "
        "to both end users and assessors. Tkinter became important later because it gave me a graphical "
        "launcher for people who may not want to work entirely from the terminal. sqlite3 and Optuna also "
        "appear in the repository because I explored persistence and more formal experiment tracking."
    ),
    176: (
        "I intentionally keep a separate subsection for `yfinance` because my earlier draft was right to "
        "treat it as a major dependency. It became the data foundation for both my TensorFlow and PyTorch "
        "forecasting stages, and its ability to return Yahoo Finance data directly into pandas-compatible "
        "structures made it ideal for rapid experimentation and later pipeline stability."
    ),
    178: (
        "In my sample draft I treated TensorFlow as a major researched library, and I still think that was "
        "appropriate because it genuinely powered the early stages of my implementation. I therefore keep "
        "TensorFlow as a distinct researched technology while also explaining that I later replaced it in "
        "the strongest implementation path with PyTorch for practical engineering reasons."
    ),
    179: HEADING_REPLACEMENT,
    180: (
        "In my earlier research, I also considered how Python automation and supporting scripts could help "
        "me manage code execution, data handling and repeatable workflows. That idea remained useful even "
        "though it did not become the single defining pillar of the project, because it showed that I was "
        "thinking beyond isolated model training and toward a broader software process."
    ),
    183: (
        "SQLite3 remains relevant in my report because my original concept included stored user data, and "
        "some of my later experiment-management work also touched database-backed storage. For that "
        "reason, I still consider it a justified part of the library evaluation and the broader story of "
        "structured persistence in the project."
    ),
    185: (
        "Pandas is one of the most justified libraries in my entire project. My earlier draft emphasised "
        "its value for dataset handling, and the implemented repository fully confirms that view. I rely "
        "on it for historical data processing, rolling indicators, fill operations and many intermediate "
        "transformations required before tensors are built."
    ),
    187: (
        "NumPy deserves separate mention because I repeatedly rely on fast numerical array manipulation. In "
        "the later separated pipeline it also supports more advanced and efficient sequence-window "
        "generation, which shows that it is not just a background dependency but an important performance "
        "enabler in my workflow."
    ),
    189: (
        "Matplotlib remains one of the key visualisation libraries in my report because my outputs have to "
        "be interpretable. My sample document was right to identify graphing as essential. I simply extend "
        "that point here by showing how later charts became richer, more polished and more closely linked "
        "to prediction and uncertainty output."
    ),
    191: (
        "My sample document included Pickle because serialization is useful when I want to preserve "
        "processed objects, figures or model-related state. Even though later stages often favour more "
        "explicit artifact files such as CSV outputs, I still consider Pickle a valid part of the wider "
        "research and experimentation context."
    ),
    193: (
        "I keep TensorFlow as a separate library-evaluation subsection because it was not merely researched; "
        "I genuinely used it. For that reason, I do not want the report to pretend the project was "
        "PyTorch-only from the beginning."
    ),
    195: (
        "In my sample draft I had already acknowledged that PyTorch replaced TensorFlow because of "
        "compatibility issues, and my repository strongly supports that summary. I now expand that point "
        "with a fuller explanation of device control, hardware behaviour, experimentation flexibility and "
        "long-term maintainability."
    ),
    221: (
        "This function demonstrates why my project needed robust data libraries. I use pandas for rolling "
        "calculations, exponential moving averages, fill operations and column management, and the "
        "function also shows that I learned from earlier `yfinance` shape problems by flattening "
        "`MultiIndex` columns before model preparation begins."
    ),
    334: (
        "This function shows how my system moves from the last known sequence window through repeated "
        "next-step prediction into unscaled future prices and confidence bounds. It also reveals a "
        "practical modelling assumption: while the close prediction evolves day by day, the other "
        "engineered features are approximated from the last known scaled state. I accepted that as a "
        "sensible coursework trade-off because it keeps the rollout tractable while still producing "
        "meaningful forward scenarios."
    ),
    500: (
        "These excerpts show that my GUI is not decorative. It actively captures user parameters and turns "
        "them into a runnable forecasting command. In other words, it acts as a real interface layer on "
        "top of the analytical engine, which is exactly the kind of usability feature I need to justify in "
        "the project guide."
    ),
    844: (
        "`future-eric-pytorch.py` has a last write date of 26 January 2026, and later dated development "
        "activity around PyTorch and Optuna becomes visible from 22 February 2026 onward. That suggests "
        "the transition began in late January and became central by late February 2026."
    ),
    845: (
        "My migration to PyTorch was a genuine turning point. I needed more explicit control over devices, "
        "clearer forward-pass logic and a framework that made unusual training and inference behaviour "
        "easier to express. The move was therefore justified by practical engineering pressure, not just "
        "preference."
    ),
    1068: (
        "Major dated development activity appears on 16 March 2026, 18 March 2026, 24 March 2026, 27 "
        "March 2026, 29 March 2026, 7 April 2026 and 9 April 2026. This is the densest development period "
        "in my repository and corresponds to the growth of `pytorch_plotted.py` into a full-feature "
        "pipeline."
    ),
    1592: (
        "A final iterative category concerns output and integration. I check that test-period predictions "
        "align with actual values, that future forecasts contain the requested number of days, that "
        "confidence intervals are shaped correctly and, in the separated pipeline, that prediction "
        "artifacts are written where the plotting script expects them. These tests are justified because "
        "they cover exactly the areas that changed repeatedly during development."
    ),
    1611: (
        "Python Software Foundation (2026) subprocess documentation. Available at: "
        "https://docs.python.org/3/library/subprocess.html (Accessed: 13 April 2026)."
    ),
}


TABLE_TEXT_OVERRIDES: dict[str, str] = {
    "Absolute unit, pro, 17  year old male resides in Scotland": (
        "Seventeen-year-old peer tester based in Scotland."
    ),
    "Student for the sciences of computers and also my neighbour in lesson. He will be my primary tester as he is the closest to me and sees a lot of my work.": (
        "He studies computer science with me and is one of the people who sees my progress most often, so "
        "I chose him as a primary peer tester."
    ),
    "Will have hands on access to the website and view the progress being made in person. He will attain second priority to interact with the project": (
        "He can access the website directly, observe my progress in person and give me regular feedback, "
        "which makes him an appropriate secondary tester."
    ),
    "Computer science teacher for the Compton School, tall, middle-aged man, resides in London": (
        "Computer science teacher at The Compton School, based in London."
    ),
    "Monitoring and foreseeing my progress with the website and its contents. Will guide me to finishing the project and will also provide useful insights on how to improve and polish code.": (
        "He monitors my progress with the website and the wider project, guides me toward completion and "
        "provides useful feedback on how I can improve and polish my code."
    ),
    "Will have first priority access to interact with the website and its contents. Will have unlimited access to view source code and provide suggestions that will be effective immediately.": (
        "He has first-priority access to review the website and its contents, and he can inspect my source "
        "code and make suggestions that I can act on immediately."
    ),
    "Anyone who is responsible for using the website for commercial use.": (
        "A representative end user who would use the website in a practical or commercial context."
    ),
    "Will be using the website for their own business and for their own benefit. They will be communicating with me for implementing features that they wish to be added. Suggestions must be communicated with me as soon as possible.": (
        "This stakeholder would use the website for real benefit and would communicate with me about any "
        "features they want added or improved."
    ),
    "John Doe is able to view prototypes of the website and keep track of my progress regarding the implementation of their suggestions and the general completion of the platform": (
        "This stakeholder can review prototypes of the website and track how well I implement requested "
        "features and move the platform toward completion."
    ),
}


GLOBAL_TEXT_REPLACEMENTS = {
    "GitHub REST API": HEADING_REPLACEMENT,
    "https://github.com/ovinokur": "https://pytorch-optimizers.readthedocs.io/en/latest/",
}


PRESERVE_TAGS = {
    f"{W}pPr",
    f"{W}bookmarkStart",
    f"{W}bookmarkEnd",
    f"{W}commentRangeStart",
    f"{W}commentRangeEnd",
    f"{W}permStart",
    f"{W}permEnd",
    f"{W}proofErr",
}


GENERIC_REPLACEMENTS = [
    (r"\bThe project guide asks for\b", "The project guide asks me for"),
    (r"\bThe project guide asks\b", "The project guide asks me"),
    (r"\bThe project's\b", "My project's"),
    (r"\bthe project's\b", "my project's"),
    (r"\bThis project\b", "My project"),
    (r"\bthis project\b", "my project"),
    (r"\bThe project\b", "My project"),
    (r"\bthe project\b", "my project"),
    (r"\bThe system\b", "My system"),
    (r"\bthe system\b", "my system"),
    (r"\bThe repository\b", "My repository"),
    (r"\bthe repository\b", "my repository"),
    (r"\bThe sample draft\b", "My sample draft"),
    (r"\bthe sample draft\b", "my sample draft"),
    (r"\bThe sample document\b", "My sample document"),
    (r"\bthe sample document\b", "my sample document"),
    (r"\bThe earlier draft\b", "My earlier draft"),
    (r"\bthe earlier draft\b", "my earlier draft"),
    (r"\bThe earlier document\b", "In my earlier document"),
    (r"\bthe earlier document\b", "in my earlier document"),
    (r"\bThe continuation version\b", "In this continuation version"),
    (r"\bthe continuation version\b", "in this continuation version"),
    (r"\bThe continuation document\b", "In this continuation document"),
    (r"\bthe continuation document\b", "in this continuation document"),
    (r"\bThe final architecture\b", "My final architecture"),
    (r"\bthe final architecture\b", "my final architecture"),
    (r"\bThe final repository\b", "My final repository"),
    (r"\bthe final repository\b", "my final repository"),
    (r"\bThe developer was\b", "I was"),
    (r"\bthe developer was\b", "I was"),
    (r"\bThe developer became\b", "I became"),
    (r"\bthe developer became\b", "I became"),
    (r"\bThe developer learned\b", "I learned"),
    (r"\bthe developer learned\b", "I learned"),
    (r"\bThe developer recognised\b", "I recognised"),
    (r"\bthe developer recognised\b", "I recognised"),
    (r"\bthe developer\b", "I"),
    (r"\bThe strongest completed outcome\b", "The strongest completed outcome I produced"),
    (r"\bthe strongest completed outcome\b", "the strongest completed outcome I produced"),
    (r"\bthe strongest completed deliverable\b", "the strongest completed deliverable I produced"),
    (r"\bthe final implemented form\b", "the final form I implemented"),
    (r"\bthe final implemented project\b", "the final version I implemented"),
    (r"\bthe final implemented project\b", "the final version I implemented"),
]


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(t.text or "" for t in paragraph.iter(f"{W}t")).strip()


def cell_paragraphs(table: ET.Element) -> list[ET.Element]:
    return table.findall(f".//{W}tc/{W}p")


def has_field_code(paragraph: ET.Element) -> bool:
    return (
        paragraph.find(f".//{W}instrText") is not None
        or paragraph.find(f".//{W}fldChar") is not None
        or paragraph.find(f".//{W}hyperlink") is not None
    )


def paragraph_style(paragraph: ET.Element) -> str:
    style = paragraph.find(f"{W}pPr/{W}pStyle")
    if style is None:
        return ""
    return style.attrib.get(f"{W}val", "")


def looks_like_url(text: str) -> bool:
    return bool(re.fullmatch(r"https?://\S+", text))


def looks_like_bibliography(text: str) -> bool:
    markers = ("Available at:", "[online]", "Accessed:", "documentation.", "documentation)")
    return any(marker in text for marker in markers)


def rewrite_generic(text: str) -> str:
    updated = text

    sentence_starts = [
        (r"^There are multiple examples", "I found multiple examples"),
        (r"^The First example,", "The first example I evaluated was"),
        (r"^The first example,", "The first example I evaluated was"),
        (r"^Another example is", "Another example I evaluated was"),
        (r"^Similar systems are important because they show", "I evaluated similar systems because they show"),
        (r"^The earliest stage was", "My earliest stage was"),
        (r"^This stage is valuable because it identified", "This stage was valuable because it helped me identify"),
        (r"^This stage is important because it shows", "This stage matters because it shows"),
        (r"^The first machine-learning implementation used", "My first machine-learning implementation used"),
        (r"^The phase also exposed immediate weaknesses", "This phase also exposed immediate weaknesses for me"),
    ]
    for pattern, replacement in sentence_starts:
        updated = re.sub(pattern, replacement, updated)

    for pattern, replacement in GENERIC_REPLACEMENTS:
        updated = re.sub(pattern, replacement, updated)

    updated = updated.replace("A second major stakeholder is the developer,", "A second major stakeholder is me as the developer,")
    updated = updated.replace("Someone other than the developer may need", "Someone other than me may need")
    updated = updated.replace("This makes it possible to show not only what the current code does, but also why each stage existed", "This lets me show not only what the current code does, but also why each stage existed")
    updated = updated.replace("The developer was not only trying", "I was not only trying")
    updated = updated.replace("They were imagining", "I was imagining")
    updated = updated.replace("This explains the presence", "That explains the presence")

    updated = re.sub(r"\s{2,}", " ", updated).strip()
    return updated


def set_paragraph_text(paragraph: ET.Element, new_text: str) -> None:
    for child in list(paragraph):
        if child.tag not in PRESERVE_TAGS:
            paragraph.remove(child)

    run = ET.Element(f"{W}r")
    text_node = ET.SubElement(run, f"{W}t")
    if new_text.startswith(" ") or new_text.endswith(" ") or "  " in new_text:
        text_node.set(f"{{{XML_NS}}}space", "preserve")
    text_node.text = new_text
    paragraph.append(run)


def replace_global_text_nodes(root: ET.Element) -> None:
    for text_node in root.iter(f"{W}t"):
        if text_node.text in GLOBAL_TEXT_REPLACEMENTS:
            text_node.text = GLOBAL_TEXT_REPLACEMENTS[text_node.text]


def rewrite_tables(root: ET.Element) -> None:
    for table in root.iter(f"{W}tbl"):
        for paragraph in cell_paragraphs(table):
            text = paragraph_text(paragraph)
            if not text:
                continue
            if text in TABLE_TEXT_OVERRIDES:
                set_paragraph_text(paragraph, TABLE_TEXT_OVERRIDES[text])


def rewrite_body_paragraphs(root: ET.Element) -> None:
    body = root.find(f"{W}body")
    if body is None:
        raise ValueError("Document body not found")

    for idx, child in enumerate(list(body)):
        if child.tag != f"{W}p":
            continue

        text = paragraph_text(child)
        if not text:
            continue

        style = paragraph_style(child)

        if idx in BODY_OVERRIDES:
            set_paragraph_text(child, BODY_OVERRIDES[idx])
            continue

        if has_field_code(child):
            continue

        if style == "CodeBlock":
            continue

        if looks_like_url(text) or looks_like_bibliography(text):
            continue

        if style in {"BodyText", "Normal"} or (not style and len(text.split()) > 12):
            updated = rewrite_generic(text)
            if updated != text:
                set_paragraph_text(child, updated)


def rewrite_docx(doc_path: Path) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = HISTORY_DIR / f"{doc_path.stem}.before_first_person_{timestamp}{doc_path.suffix}"
    shutil.copy2(doc_path, backup_path)

    with zipfile.ZipFile(doc_path) as src:
        xml_bytes = src.read("word/document.xml")
        root = ET.fromstring(xml_bytes)

        replace_global_text_nodes(root)
        rewrite_body_paragraphs(root)
        rewrite_tables(root)

        updated_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        temp_path = doc_path.with_suffix(".tmp.docx")
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename == "word/document.xml":
                    data = updated_xml
                dst.writestr(info, data)

    temp_path.replace(doc_path)
    return backup_path


def main() -> None:
    backup_path = rewrite_docx(DOC_PATH)
    print(f"Rewrote: {DOC_PATH}")
    print(f"Backup:  {backup_path}")


if __name__ == "__main__":
    main()
