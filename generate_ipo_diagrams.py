import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = r'C:\Users\paron\Desktop\Dev\CS_PROJECT\docs\diagrams\ipo'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define IPO specs for each bottom-level node
# Each entry: (module_name, input_label, process_label, output_label, io_table_inputs, io_table_outputs)
modules = [
    # Application Layer
    {
        'name': 'Login/Logout',
        'layer': 'Application',
        'inputs': ['Username', 'Password', 'Action (login/logout)'],
        'process': 'Credential validation against Users table\nSession token generation\nSecure password hashing',
        'outputs': ['Session token', 'Success/failure message', 'Redirect'],
        'io_table': [
            ['Valid creds + login', 'sess_abc123...', 'Login successful → /search.php'],
            ['Wrong password', 'null', 'Invalid credentials → /login.php?error=1'],
            ['Valid session + logout', 'null', 'Session destroyed → /index.php'],
            ['Expired session', 'null', 'Session expired → /login.php?msg=expired'],
        ],
        'io_headers': ['Input', 'Session Token', 'Response'],
        'filename': 'ipo_01_login_logout.png',
    },
    {
        'name': 'Session Management',
        'layer': 'Application',
        'inputs': ['Session token', 'Timestamp'],
        'process': 'Token lookup in session store\nExpiry time validation\nUser ID resolution',
        'outputs': ['User ID', 'Valid/invalid status'],
        'io_table': [
            ['sess_abc + not expired', 'user_id = 42', 'valid'],
            ['sess_abc + expired', 'null', 'expired'],
            ['sess_invalid', 'null', 'invalid_token'],
            ['empty string', 'null', 'no_session'],
        ],
        'io_headers': ['Input', 'Resolved User', 'Status'],
        'filename': 'ipo_02_session_mgmt.png',
    },
    {
        'name': 'Task Scheduling',
        'layer': 'Application',
        'inputs': ['Ticker symbol', 'User ID', 'Forecast params'],
        'process': 'Job record creation in DB\nQueue priority assignment\nWorker notification',
        'outputs': ['Job ID', 'Queued status'],
        'io_table': [
            ['AAPL + user_id=42', 'job_id = 1057', 'queued'],
            ['ZZZZ (invalid)', 'null', 'invalid_ticker'],
            ['No session', 'null', 'unauthorized'],
            ['AAPL + duplicate', 'job_id = 1058', 'queued (duplicate)'],
        ],
        'io_headers': ['Input', 'Job ID', 'Status'],
        'filename': 'ipo_03_task_sched.png',
    },
    {
        'name': 'Job Tracking',
        'layer': 'Application',
        'inputs': ['Job ID'],
        'process': 'Query prediction_jobs table\nParse status field\nFormat timestamp',
        'outputs': ['Status', 'Progress info', 'Error message'],
        'io_table': [
            ['job_id = 1057', 'completed', '100% (graph_id=88)', 'null'],
            ['job_id = 1060', 'running', '45% (training epoch 12/30)', 'null'],
            ['job_id = 9999', 'not_found', 'null', 'No job exists with that ID'],
            ['job_id = 1055', 'failed', 'null', 'yfinance returned empty data'],
        ],
        'io_headers': ['Input', 'Status', 'Progress', 'Error'],
        'filename': 'ipo_04_job_tracking.png',
    },
    {
        'name': 'Model Inference',
        'layer': 'Application',
        'inputs': ['Normalized feature tensor', 'Model weights (.pt)'],
        'process': 'Load BiLSTM model state_dict\nForward pass on input tensor\nExtract prediction values',
        'outputs': ['Raw price predictions', 'Hidden states'],
        'io_table': [
            ['tensor[30,12] + model.pt', '[142.5, 143.1, 144.0, ...]', 'tensor[30,256]'],
            ['tensor[15,8] (wrong dims)', 'error: size mismatch', 'null'],
            ['corrupted model.pt', 'error: invalid file', 'null'],
            ['tensor[30,12] only', '[142.5, 143.1, 144.0, ...]', 'null (no MC dropout)'],
        ],
        'io_headers': ['Input', 'Predictions (next 30 days)', 'Hidden States'],
        'filename': 'ipo_05_model_inference.png',
    },
    {
        'name': 'Result Formatting',
        'layer': 'Application',
        'inputs': ['Raw predictions', 'Confidence intervals', 'Ticker metadata'],
        'process': 'Merge predictions with dates\nAppend uncertainty bands\nFormat for display/chart',
        'outputs': ['Formatted chart data', 'Graph image file', 'Summary text'],
        'io_table': [
            ['All inputs valid', '30 rows: date/pred/low/high', 'AAPL_forecast_2025.png', 'Pred: +2.3% ± 1.1%'],
            ['Missing metadata', '30 rows: date/pred/low/high', 'forecast_2025.png', 'Pred: +2.3% ± 1.1%'],
            ['Missing CI bands', '30 rows: date/pred', 'AAPL_forecast_2025.png', 'Pred: +2.3% (no CI)'],
            ['Missing predictions', 'null', 'null', 'error: no data to plot'],
        ],
        'io_headers': ['Input', 'Chart Data', 'Graph File', 'Summary'],
        'filename': 'ipo_06_result_format.png',
    },
    {
        'name': 'yfinance Integration',
        'layer': 'Application',
        'inputs': ['Ticker symbol', 'Date range (start/end)'],
        'process': 'Construct yfinance API request\nSend HTTP request to Yahoo Finance\nParse JSON response',
        'outputs': ['OHLCV DataFrame', 'Stock metadata'],
        'io_table': [
            ['AAPL + 2024-01 to 2025-01', '252 rows (Open/High/Low/Close/Vol)', 'Apple Inc., Tech, NASDAQ'],
            ['ZZZZ (invalid)', 'empty DataFrame (0 rows)', 'null'],
            ['end < start', 'empty DataFrame (0 rows)', 'null'],
            ['Network timeout', 'null (connection error)', 'null'],
        ],
        'io_headers': ['Input', 'OHLCV Data', 'Company Info'],
        'filename': 'ipo_07_yfinance.png',
    },
    {
        'name': 'Data Cleaning',
        'layer': 'Application',
        'inputs': ['Raw OHLCV DataFrame'],
        'process': 'Remove NaN rows\nHandle zero-volume days\nSort by date ascending',
        'outputs': ['Cleaned OHLCV DataFrame'],
        'io_table': [
            ['252 rows (3 NaN)', '249 rows, no NaN, sorted'],
            ['All NaN (252 rows)', 'empty DataFrame (0 rows)'],
            ['Empty input', 'empty DataFrame (0 rows)'],
            ['252 rows (1 zero vol)', '252 rows, vol=0 flagged'],
        ],
        'io_headers': ['Input', 'Cleaned DataFrame'],
        'filename': 'ipo_08_data_cleaning.png',
    },
    {
        'name': 'Technical Indicators',
        'layer': 'Application',
        'inputs': ['Cleaned OHLCV data'],
        'process': 'Calculate Moving Averages (SMA/EMA)\nCompute RSI (14-period)\nCompute MACD (12,26,9)',
        'outputs': ['Indicator DataFrame (MA, RSI, MACD)'],
        'io_table': [
            ['252 rows', '227 rows with SMA, EMA, RSI, MACD columns'],
            ['20 rows (< 26 needed)', 'NaN for MACD, SMA(26) computed'],
            ['10 rows (< 14 needed)', 'NaN for RSI, partial MA only'],
            ['252 rows, valid prices', '227 rows, all indicators valid'],
        ],
        'io_headers': ['Input', 'Indicator Output'],
        'filename': 'ipo_09_indicators.png',
    },
    {
        'name': 'Normalization',
        'layer': 'Application',
        'inputs': ['Indicator DataFrame', 'Scaling parameters (min/max)'],
        'process': 'Apply MinMax scaling per column\nReshape to sequence tensor format\nConvert to PyTorch tensor',
        'outputs': ['Normalized tensor [seq_len, features]'],
        'io_table': [
            ['227 rows + min/max', 'tensor[227, 12] values in [0.0, 1.0]'],
            ['227 rows (compute params)', 'tensor[227, 12] + new min/max dict'],
            ['Empty input', 'error: no data to normalize'],
            ['Single row', 'tensor[1, 12] (all values = 0.5)'],
        ],
        'io_headers': ['Input', 'Output Tensor'],
        'filename': 'ipo_10_normalization.png',
    },
]

# Computation Layer
modules_comp = [
    {
        'name': 'Input Processing (BiLSTM)',
        'layer': 'Computation',
        'inputs': ['Normalized tensor [seq_len, features]'],
        'process': 'Reshape to [batch=1, seq, features]\nAdd batch dimension\nVerify tensor device (CPU/GPU)',
        'outputs': ['Batched tensor ready for LSTM'],
        'io_table': [
            ['tensor[30,12] (2D)', 'tensor[1,30,12] (batched)'],
            ['tensor[1,30,12] (already 3D)', 'tensor[1,30,12] (passed through)'],
            ['tensor[12] (1D)', 'error: expected 2D or 3D input'],
            ['tensor[str, ...]', 'error: non-numeric dtype'],
        ],
        'io_headers': ['Input', 'Output'],
        'filename': 'ipo_11_input_processing.png',
    },
    {
        'name': 'LSTM Layers',
        'layer': 'Computation',
        'inputs': ['Batched tensor', 'Model config (hidden_dim, layers, dropout)'],
        'process': 'Forward LSTM pass (t=1..seq)\nBackward LSTM pass (t=seq..1)\nConcatenate forward+backward states',
        'outputs': ['BiLSTM hidden states', 'Final hidden vector'],
        'io_table': [
            ['tensor[1,30,12] + hidden=256', 'states[1,30,512]', 'vector[1,512]'],
            ['hidden=0 (invalid)', 'error: hidden_dim must be > 0', 'null'],
            ['seq_len=0 (empty)', 'error: sequence length zero', 'null'],
            ['tensor[1,1,12] (single step)', 'states[1,1,512]', 'vector[1,512]'],
        ],
        'io_headers': ['Input', 'Hidden States', 'Final Vector'],
        'filename': 'ipo_12_lstm_layers.png',
    },
    {
        'name': 'Hyperparameter Tuning',
        'layer': 'Computation',
        'inputs': ['Training data', 'Parameter search space'],
        'process': 'Iterate over param combinations\nTrain model for each config\nTrack validation loss per run',
        'outputs': ['Best parameter set', 'Validation loss history'],
        'io_table': [
            ['data + lr=[0.001,0.01]', 'best: lr=0.001, hidden=256', '[0.045, 0.032, 0.028, ...]'],
            ['empty search space', 'error: no configs to test', '[]'],
            ['No training data', 'error: train split is empty', '[]'],
            ['Single config', 'lr=0.01, hidden=128', '[0.052]'],
        ],
        'io_headers': ['Input', 'Best Config', 'Loss History'],
        'filename': 'ipo_13_hyperparam.png',
    },
    {
        'name': 'GPU Acceleration',
        'layer': 'Computation',
        'inputs': ['PyTorch tensors', 'Model weights'],
        'process': 'Check CUDA availability\nTransfer tensors to GPU device\nExecute on GPU compute cores',
        'outputs': ['GPU-computed results', 'Device location'],
        'io_table': [
            ['CUDA available', 'results on cuda:0', 'cuda:0'],
            ['No CUDA', 'results on cpu (fallback)', 'cpu'],
            ['Out of GPU memory', 'error: CUDA out of memory', 'cpu (after fallback)'],
            ['Already on cpu', 'results on cpu', 'cpu'],
        ],
        'io_headers': ['Input', 'Results', 'Device'],
        'filename': 'ipo_14_gpu_accel.png',
    },
    {
        'name': 'Uncertainty Quantification',
        'layer': 'Computation',
        'inputs': ['Model with MC Dropout enabled', 'Input tensor', 'Number of forward passes (N)'],
        'process': 'Run N stochastic forward passes\nCollect prediction samples\nCompute variance across samples',
        'outputs': ['Prediction variance', 'Mean prediction', 'Sample distribution'],
        'io_table': [
            ['N=30 + valid tensor', 'var=0.0023', 'mean=[142.5, 143.1, ...]', '30 sample arrays'],
            ['N=3 (too few)', 'error: N must be >= 10', 'null', 'null'],
            ['No dropout layers', 'var=0.0 (deterministic)', 'mean=[142.5, 143.1, ...]', '1 sample (same each time)'],
            ['N=1', 'null (variance needs N>1)', 'mean=[142.5, 143.1, ...]', '1 sample'],
        ],
        'io_headers': ['Input', 'Variance', 'Mean', 'Samples'],
        'filename': 'ipo_15_uncertainty.png',
    },
    {
        'name': 'Confidence Intervals',
        'layer': 'Computation',
        'inputs': ['Mean prediction', 'Variance / standard deviation'],
        'process': 'Compute upper bound = mean + 1.96 * std\nCompute lower bound = mean - 1.96 * std\nFormat as percentage band',
        'outputs': ['Upper bound', 'Lower bound', 'Confidence level'],
        'io_table': [
            ['mean=142.5, std=1.2', 'upper: 144.85', 'lower: 140.15', '95% (±1.65%)'],
            ['std=0 (deterministic)', 'upper: 142.50', 'lower: 142.50', '95% (±0.00%)'],
            ['std=NaN', 'upper: NaN', 'lower: NaN', 'null (invalid std)'],
            ['var < 0 (impossible)', 'error: negative variance', 'null', 'null'],
        ],
        'io_headers': ['Input', 'Upper Bound', 'Lower Bound', 'Confidence'],
        'filename': 'ipo_16_confidence.png',
    },
]

# Data Layer
modules_data = [
    {
        'name': 'Users Table (MySQL)',
        'layer': 'Data',
        'inputs': ['User data (username, hashed_password)', 'Query type (INSERT/SELECT)'],
        'process': 'Execute SQL statement via PDO\nApply parameter binding (prevent injection)\nCommit transaction',
        'outputs': ['Query result', 'Affected row count'],
        'io_table': [
            ['INSERT + demo:$2y$10$...', 'null (insert)', '1 row inserted'],
            ['SELECT + demo', 'row: id=1, username=demo, ...', '1 row returned'],
            ['INSERT + duplicate demo', 'null (constraint error)', '0 rows (duplicate key)'],
            ['SELECT + nonexistent', 'empty result set', '0 rows returned'],
        ],
        'io_headers': ['Input', 'Query Result', 'Rows Affected'],
        'filename': 'ipo_17_users_table.png',
    },
    {
        'name': 'Jobs Table (MySQL)',
        'layer': 'Data',
        'inputs': ['Job metadata (ticker, user_id, params)', 'Status update (running/completed)'],
        'process': 'INSERT new job into prediction_jobs\nUPDATE status field on state change\nSELECT for queue worker polling',
        'outputs': ['Job record / status update result'],
        'io_table': [
            ['INSERT AAPL + user_id=1', 'row: id=1057, status=queued', '1 row inserted'],
            ['UPDATE id=1057 status=running', 'status changed: queued → running', '1 row updated'],
            ['UPDATE id=9999 (not found)', 'null (no matching row)', '0 rows updated'],
            ['SELECT WHERE status=queued', 'rows: [{id:1058}, {id:1059}]', '2 rows returned'],
        ],
        'io_headers': ['Input', 'Result', 'Rows Affected'],
        'filename': 'ipo_18_jobs_table.png',
    },
    {
        'name': 'Model Files (.pt)',
        'layer': 'Data',
        'inputs': ['Trained model state_dict', 'File path'],
        'process': 'Serialize PyTorch model via torch.save\nWrite to disk as .pt file\nVerify file integrity on load',
        'outputs': ['File path', 'Load verification'],
        'io_table': [
            ['state_dict + /models/aapl.pt', '/models/aapl.pt (34.2 MB)', 'loaded successfully'],
            ['corrupted state_dict', 'null (save failed)', 'error on load attempt'],
            ['/models/nonexistent.pt', 'null (file not found)', 'null (load failed)'],
            ['Load /models/aapl.pt', 'null (load operation)', 'loaded: 3 layers, hidden=256'],
        ],
        'io_headers': ['Input', 'File Path', 'Verification'],
        'filename': 'ipo_19_model_files.png',
    },
    {
        'name': 'CSV Exports',
        'layer': 'Data',
        'inputs': ['Prediction DataFrame', 'Output file path'],
        'process': 'Format DataFrame to CSV structure\nWrite rows to .csv file\nOptionally upload to remote via SFTP',
        'outputs': ['CSV file', 'File size'],
        'io_table': [
            ['30 rows + /exports/aapl.csv', '/exports/aapl.csv', '2.4 KB'],
            ['Empty DataFrame + path', '/exports/empty.csv', '0 bytes (headers only)'],
            ['30 rows + invalid path', 'null (write failed)', '0 bytes'],
            ['30 rows + no write perms', 'null (permission denied)', '0 bytes'],
        ],
        'io_headers': ['Input', 'CSV File', 'File Size'],
        'filename': 'ipo_20_csv_exports.png',
    },
]

# External Services
modules_ext = [
    {
        'name': 'Historical Data Retrieval',
        'layer': 'External',
        'inputs': ['Ticker symbol', 'Period (start_date, end_date)'],
        'process': 'Call yfinance.download(ticker, start, end)\nHandle rate limiting / retries\nParse returned DataFrame',
        'outputs': ['OHLCV DataFrame', 'Download status'],
        'io_table': [
            ['AAPL + 2024-01-01 to 2025-01-01', '252 rows (O/H/L/C/Volume)', 'success: 252 trading days'],
            ['DELL (delisted)', 'empty DataFrame (0 rows)', 'warning: no data for ticker'],
            ['AAPL + 2025-12-01 to 2025-01-01', 'empty DataFrame (0 rows)', 'error: start > end'],
            ['AAPL (rate limited)', 'null (retry after 5s)', 'error: HTTP 429 Too Many Requests'],
        ],
        'io_headers': ['Input', 'OHLCV Data', 'Status'],
        'filename': 'ipo_21_hist_data.png',
    },
    {
        'name': 'Stock Metadata',
        'layer': 'External',
        'inputs': ['Ticker symbol'],
        'process': 'Call yfinance.Ticker(ticker).info\nExtract company name, sector, market cap\nHandle missing fields gracefully',
        'outputs': ['Company info dict', 'Sector / industry'],
        'io_table': [
            ['AAPL', 'name=Apple Inc, mktCap=2.8T', 'Technology, Consumer Electronics'],
            ['ZZZZ (invalid)', 'empty dict {}', 'null (ticker not found)'],
            ['AAPL (partial info)', 'name=Apple Inc, mktCap=null', 'null (sector field missing)'],
            ['AAPL (API down)', 'null (request failed)', 'null (HTTP 503)'],
        ],
        'io_headers': ['Input', 'Company Info', 'Sector'],
        'filename': 'ipo_22_stock_meta.png',
    },
    {
        'name': 'Remote Commands (SSH)',
        'layer': 'External',
        'inputs': ['SSH credentials', 'Command string'],
        'process': 'Establish SSH connection via paramiko\nExecute command on remote host\nCapture stdout/stderr streams',
        'outputs': ['Stdout output', 'Exit code', 'Stderr output'],
        'io_table': [
            ['user@remote + python3 train.py', 'Epoch 30/30, loss: 0.021', '0', ''],
            ['wrong password', '', '-1 (auth failed)', 'paramiko.AuthenticationException'],
            ['Connection timeout (30s)', '', '-2 (timeout)', 'paramiko.SSHException: timeout'],
            ['python3 missing_script.py', '', '127', 'bash: python3: command not found'],
        ],
        'io_headers': ['Input', 'Stdout', 'Exit Code', 'Stderr'],
        'filename': 'ipo_23_remote_cmds.png',
    },
    {
        'name': 'Secure File Transfer (SFTP)',
        'layer': 'External',
        'inputs': ['Local file path', 'Remote file path', 'SSH credentials'],
        'process': 'Open SFTP session over SSH\nRead local file in binary mode\nWrite to remote path via SFTP put',
        'outputs': ['Transfer status', 'Remote file path'],
        'io_table': [
            ['/local/aapl.pt → /remote/', 'success (34.2 MB transferred)', '/remote/aapl.pt'],
            ['/local/missing.pt → /remote/', 'error: local file not found', 'null'],
            ['... → /invalid/path/', 'error: remote path invalid', 'null'],
            ['... (no write perms)', 'error: permission denied', 'null'],
        ],
        'io_headers': ['Input', 'Status', 'Remote Path'],
        'filename': 'ipo_24_sftp.png',
    },
]

# UI Layer
modules_ui = [
    {
        'name': 'PHP Backend',
        'layer': 'UI',
        'inputs': ['HTTP request (GET/POST)', 'Route path', 'Session cookies'],
        'process': 'Parse request parameters\nRoute to appropriate handler\nRender HTML template or redirect',
        'outputs': ['HTML response', 'HTTP status code'],
        'io_table': [
            ['GET /search.php + valid session', '<!DOCTYPE html>...', '200 OK'],
            ['GET /index.php (no session)', '<!DOCTYPE html> (public)', '200 OK'],
            ['GET /nonexistent.php', '<!DOCTYPE html> (404 page)', '404 Not Found'],
            ['DB unavailable', '<!DOCTYPE html> (db warning)', '200 OK (degraded)'],
        ],
        'io_headers': ['Input', 'Response', 'HTTP Status'],
        'filename': 'ipo_25_php_backend.png',
    },
    {
        'name': 'JavaScript Frontend',
        'layer': 'UI',
        'inputs': ['User interaction (click/submit)', 'DOM state'],
        'process': 'Capture form inputs\nConstruct AJAX/fetch request\nUpdate DOM with response data',
        'outputs': ['Updated page content', 'User feedback'],
        'io_table': [
            ['Submit ticker form', 'Results table rendered', 'Green flash: "Forecast queued"'],
            ['Empty ticker field', 'Form unchanged', 'Red flash: "Enter a ticker"'],
            ['Network timeout', 'Loading spinner stays', 'Red flash: "Connection failed"'],
            ['Page load', 'DOM initialized', 'No notification (normal load)'],
        ],
        'io_headers': ['Input', 'Page Content', 'User Feedback'],
        'filename': 'ipo_26_js_frontend.png',
    },
    {
        'name': 'System Logs Viewer',
        'layer': 'UI',
        'inputs': ['User request', 'Filter params (date, severity)'],
        'process': 'Query log entries from DB/files\nApply filters\nPaginate results for display',
        'outputs': ['Filtered log entries', 'Pagination info'],
        'io_table': [
            ['2025-01-01 to 2025-01-31 + error', '12 entries (timestamp, level, msg)', 'page 1 of 1, 12 total'],
            ['No filters (all logs)', '487 entries (all levels)', 'page 1 of 20, 487 total'],
            ['2025-13-01 (invalid date)', 'null (query error)', 'null (invalid date range)'],
            ['No logs available', '0 entries', 'page 0 of 0, 0 total'],
        ],
        'io_headers': ['Input', 'Log Entries', 'Pagination'],
        'filename': 'ipo_27_logs_viewer.png',
    },
    {
        'name': 'Job Status Monitor',
        'layer': 'UI',
        'inputs': ['User session', 'Polling interval'],
        'process': 'Fetch job statuses at interval\nRender status badges (queued/running/done)\nAuto-refresh on state change',
        'outputs': ['Status display', 'Auto-refresh trigger'],
        'io_table': [
            ['3 active jobs, 5s interval', '3 badges: 1 running, 2 queued', 'poll again in 5s'],
            ['No active jobs', 'Message: "No jobs in queue"', 'stop polling'],
            ['Session expired', 'Redirect to login page', 'stop polling'],
            ['Worker offline', '3 badges: 1 running (stale)', 'poll + warning: "worker not responding"'],
        ],
        'io_headers': ['Input', 'Status Display', 'Refresh Action'],
        'filename': 'ipo_28_job_monitor.png',
    },
]

all_modules = modules + modules_comp + modules_data + modules_ext + modules_ui


def draw_ipo_diagram(mod, save_path):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_facecolor('#fff')
    ax.axis('off')

    # Layer color
    layer_colors = {
        'Application': '#F59E0B',
        'Computation': '#8B5CF6',
        'Data': '#10B981',
        'External': '#EF4444',
        'UI': '#3B82F6',
    }
    layer_color = layer_colors.get(mod['layer'], '#666')

    # Title
    ax.text(80, 87, 'IPO Module Diagram', fontsize=12, fontweight='bold', color='#1e3a8a', ha='center')
    ax.text(80, 83, 'Layer: %s' % mod['layer'], fontsize=9, color='#666', ha='center')

    # Module box
    ax.add_patch(patches.Rectangle((30, 55), 30, 18, linewidth=2, edgecolor='#333', facecolor=layer_color, zorder=1))
    ax.text(45, 64, mod['name'], fontsize=10, fontweight='bold', ha='center', va='center', zorder=2, color='#fff')

    # Input arrow (left)
    ax.annotate('', xy=(30, 64), xytext=(8, 64),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.text(18, 68, 'INPUT', fontsize=9, fontweight='bold', ha='center', color='#333')
    for i, inp in enumerate(mod['inputs']):
        ax.text(18, 61 - i*3.5, inp, fontsize=7, ha='center', color='#555')

    # Process box
    ax.add_patch(patches.Rectangle((65, 50), 30, 28, linewidth=1.5, edgecolor='#333', facecolor='#f0f0f0', zorder=1))
    ax.text(80, 74, 'PROCESS', fontsize=9, fontweight='bold', ha='center', color='#333')
    for i, line in enumerate(mod['process'].split('\n')):
        ax.text(80, 69 - i*4, line, fontsize=7, ha='center', color='#555')

    # Arrow: Module -> Process
    ax.annotate('', xy=(65, 64), xytext=(60, 64),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Output arrow (right)
    ax.annotate('', xy=(150, 64), xytext=(125, 64),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.text(138, 68, 'OUTPUT', fontsize=9, fontweight='bold', ha='center', color='#333')
    for i, out in enumerate(mod['outputs']):
        ax.text(138, 61 - i*3.5, out, fontsize=7, ha='center', color='#555')

    # Arrow: Process -> Output
    ax.annotate('', xy=(125, 64), xytext=(95, 64),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # I/O Truth Table
    table_y = 40
    num_rows = len(mod['io_table'])
    table_h = num_rows * 4.5 + 10
    # Outer border intentionally removed to eliminate the grey looping line
    # that previously started at the left, wrapped the table and ended on the
    # right. The individual header/data cell borders remain.
    # (Previously: ax.add_patch(patches.Rectangle((12, table_y - table_h + 4), 136, table_h, linewidth=1, edgecolor='#aaa', facecolor='none', zorder=1)))
    ax.text(80, table_y + 14, 'Input / Output Truth Table', fontsize=10, fontweight='bold', ha='center', color='#333')

    headers = mod['io_headers']
    ncols = len(headers)
    all_cell_lens = []
    for j in range(ncols):
        lengths = [len(h) for h in headers]
        for row in mod['io_table']:
            lengths.append(len(str(row[j])))
        all_cell_lens.append(max(lengths))
    total_len = sum(all_cell_lens)
    avail_width = 130
    col_widths = [max(20, int(avail_width * l / total_len)) for l in all_cell_lens]
    col_starts = [15]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    # Header row
    for j, hdr in enumerate(headers):
        ax.add_patch(patches.Rectangle((col_starts[j], table_y + 6), col_widths[j], 5, linewidth=0.8, edgecolor='#333', facecolor='#e8e8e8', zorder=1))
        ax.text(col_starts[j] + 1.5, table_y + 8.5, hdr, fontsize=7, fontweight='bold', va='center', zorder=2)

    # Data rows
    row_h = 4.5
    for i, row in enumerate(mod['io_table']):
        ry = table_y + 6 - (i+1) * row_h
        for j, cell in enumerate(row):
            cell_text = str(cell)
            ax.add_patch(patches.Rectangle((col_starts[j], ry), col_widths[j], row_h, linewidth=0.5, edgecolor='#ccc', facecolor='#fafafa' if i % 2 == 0 else '#fff', zorder=1))
            ax.text(col_starts[j] + 1.5, ry + row_h/2, cell_text, fontsize=6, va='center', zorder=2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


for mod in all_modules:
    path = os.path.join(OUTPUT_DIR, mod['filename'])
    draw_ipo_diagram(mod, path)
    print('Generated: %s' % mod['filename'])

print('\nAll IPO diagrams generated in: %s' % OUTPUT_DIR)
