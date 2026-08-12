import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = r'C:\Users\paron\Desktop\Dev\CS_PROJECT\docs\diagrams'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Slide 19: Detailed Algorithms & Data Structures ----

# Layout constants to avoid borders cutting into content
MARGIN = 2.5
INNER_PADDING = 1.2

fig, axes = plt.subplots(2, 3, figsize=(30, 20))
fig.suptitle('Design: Detailed Algorithms - Variables, Data Structures, Validation', fontsize=16, fontweight='bold', color='#1e3a8a', y=0.98)

def setup_ax(ax, title, x_lim, y_lim):
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_aspect('equal')
    ax.set_facecolor('#fafafa')
    ax.axis('off')
    ax.text(x_lim/2, y_lim - MARGIN/2, title, fontsize=10, fontweight='bold', ha='center', color='#1e3a8a', zorder=10)
    # Outer background card with small margin so borders don't touch axis edges
    card_x = MARGIN - INNER_PADDING
    card_y = MARGIN - INNER_PADDING
    card_w = x_lim - 2 * (MARGIN - INNER_PADDING)
    card_h = y_lim - 2 * (MARGIN - INNER_PADDING) - 0.5
    rect = patches.Rectangle((card_x, card_y), card_w, card_h, linewidth=0.9, edgecolor='#e6e6e6', facecolor='#ffffff', zorder=0)
    rect.set_clip_on(False)
    ax.add_patch(rect)

# ===== 1. Key Variables =====
ax = axes[0, 0]
setup_ax(ax, 'Key Variables', 50, 50)
vars_app = [
    ('$_SESSION["user_id"]', 'int', 'Current logged-in user ID', 'App'),
    ('$_SESSION["username"]', 'string', 'Username for display', 'App'),
    ('$db', 'PDO', 'Database connection handle', 'App'),
    ('$job_id', 'int', 'Auto-increment prediction_jobs.id', 'App'),
    ('$ticker_symbol', 'string', 'User-entered stock ticker', 'App'),
]
vars_ml = [
    ('model', 'nn.Module', 'BiLSTM PyTorch model instance', 'ML'),
    ('optimizer', 'Adam', 'torch.optim.Adam(model.parameters())', 'ML'),
    ('criterion', 'MSELoss', 'torch.nn.MSELoss() for regression', 'ML'),
    ('device', 'torch.device', '"cuda" if available else "cpu"', 'ML'),
    ('X_train', 'tensor', 'Training sequences [N, seq, features]', 'ML'),
]
vars_ssh = [
    ('ssh_client', 'SSHClient', 'paramiko SSH connection object', 'SSH'),
    ('sftp', 'SFTPClient', 'paramiko SFTP file transfer handle', 'SSH'),
    ('remote_cmd', 'str', 'Command string for remote execution', 'SSH'),
    ('exit_status', 'int', 'Remote command exit code (0=success)', 'SSH'),
    ('transfer_size', 'int', 'Bytes transferred via SFTP', 'SSH'),
]
sections = [
    ('Application (PHP)', 46, vars_app),
    ('ML Computation (Python)', 25, vars_ml),
    ('SSH Execution (Python)', 4, vars_ssh),
]
y = 47
for label, start_y, var_list in sections:
    ax.text(3, start_y, label, fontsize=8, fontweight='bold', color='#333')
    ax.add_patch(patches.Rectangle((2, start_y - len(var_list)*3.2 - 1), 46, len(var_list)*3.2 + 2, linewidth=0.5, edgecolor='#ddd', facecolor='none'))
    for i, (name, dtype, desc, layer) in enumerate(var_list):
        ry = start_y - 2 - i*3.2
        colors = {'App': '#FEF3C7', 'ML': '#EDE9FE', 'SSH': '#FEE2E2'}
        ax.add_patch(patches.Rectangle((3, ry - 1.2), 44, 3, facecolor=colors.get(layer, '#f9f9f9')))
        ax.text(4, ry, name, fontsize=6.5, fontweight='bold', color='#333')
        ax.text(18, ry, dtype, fontsize=6.5, color='#666', style='italic')
        ax.text(30, ry, desc, fontsize=6, color='#555')

# ===== 2. Data Dictionary (MySQL) =====
ax = axes[0, 1]
setup_ax(ax, 'Data Dictionary: MySQL Tables', 50, 50)
users_cols = [
    ('id', 'INT PK', 'AUTO_INCREMENT', '1'),
    ('username', 'VARCHAR(50)', 'UNIQUE, NOT NULL', '"demo"'),
    ('password_hash', 'VARCHAR(255)', 'NOT NULL (bcrypt)', '"$2y$10$..."'),
    ('created_at', 'TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP', '"2025-01-01 00:00:00"'),
]
jobs_cols = [
    ('id', 'INT PK', 'AUTO_INCREMENT', '1057'),
    ('user_id', 'INT FK', 'REFERENCES users(id)', '1'),
    ('ticker_symbol', 'VARCHAR(10)', 'NOT NULL', '"AAPL"'),
    ('status', 'ENUM', 'queued/running/completed/failed', '"running"'),
    ('params_json', 'TEXT', 'Forecast parameters (nullable)', '{"epochs":30}'),
    ('created_at', 'TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP', '"2025-01-15 14:30:00"'),
    ('completed_at', 'TIMESTAMP', 'NULLABLE', 'NULL'),
]
graphs_cols = [
    ('id', 'INT PK', 'AUTO_INCREMENT', '88'),
    ('job_id', 'INT FK', 'REFERENCES prediction_jobs(id)', '1057'),
    ('ticker_symbol', 'VARCHAR(10)', 'NOT NULL', '"AAPL"'),
    ('title', 'VARCHAR(255)', 'Chart title string', '"AAPL Forecast (2025-01-15)"'),
    ('graph_image', 'LONGBLOB', 'PNG image binary data', '<binary: 245KB>'),
    ('created_at', 'TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP', '"2025-01-15 14:45:00"'),
]

def draw_table(ax, x, y, title, cols, col_widths):
    ax.text(x + MARGIN/4, y, title, fontsize=7.5, fontweight='bold', color='#1e3a8a')
    headers = ['Column', 'Type', 'Constraints', 'Example']
    hdr_y = y - 2.5
    cx = x
    # scale column widths to fit inside axis limits with margins
    x_max = ax.get_xlim()[1]
    available = x_max - x - MARGIN
    total = sum(col_widths)
    scale = available / total if total > 0 else 1.0
    scaled = [w * scale for w in col_widths]
    for j, h in enumerate(headers):
        wj = scaled[j]
        patch = patches.Rectangle((cx, hdr_y - 1.8), wj, 2, facecolor='#e8e8e8', edgecolor='#999', lw=0.5)
        patch.set_clip_on(False)
        ax.add_patch(patch)
        ax.text(cx + wj/2, hdr_y - 0.8, h, fontsize=6, fontweight='bold', ha='center', color='#333', zorder=11)
        cx += wj
    row_y = hdr_y - 1.8
    for i, row in enumerate(cols):
        cx = x
        bg = '#f9f9f9' if i % 2 == 0 else '#fff'
        for j, val in enumerate(row):
            wj = scaled[j]
            patch = patches.Rectangle((cx, row_y - 1.6), wj, 1.8, facecolor=bg, edgecolor='#ddd', lw=0.3)
            patch.set_clip_on(False)
            ax.add_patch(patch)
            ax.text(cx + wj/2, row_y - 0.7, val, fontsize=5.5, ha='center', color='#444', zorder=11)
            cx += wj
        row_y -= 1.8

draw_table(ax, 2, 47, 'Table: users', users_cols, [9, 9, 16, 12])
draw_table(ax, 2, 28, 'Table: prediction_jobs', jobs_cols, [9, 9, 16, 12])
draw_table(ax, 2, 9, 'Table: saved_graphs', graphs_cols, [9, 9, 16, 12])

# ===== 3. ERD =====
ax = axes[0, 2]
setup_ax(ax, 'Entity Relationship Diagram', 50, 50)
# Users entity
ax.add_patch(patches.Rectangle((4, 35), 16, 12, linewidth=1.5, edgecolor='#333', facecolor='#FEF3C7'))
ax.text(12, 45, 'users', fontsize=8, fontweight='bold', ha='center')
ax.plot([4, 20], [42, 42], color='#333', lw=0.5)
for i, col in enumerate(['id (PK, INT)', 'username (VARCHAR)', 'password_hash (VARCHAR)', 'created_at (TIMESTAMP)']):
    ax.text(5, 40 - i*1.8, col, fontsize=5.5, color='#555')
# Jobs entity
ax.add_patch(patches.Rectangle((22, 20), 18, 14, linewidth=1.5, edgecolor='#333', facecolor='#DBEAFE'))
ax.text(31, 32, 'prediction_jobs', fontsize=8, fontweight='bold', ha='center')
ax.plot([22, 40], [29, 29], color='#333', lw=0.5)
for i, col in enumerate(['id (PK, INT)', 'user_id (FK -> users.id)', 'ticker_symbol (VARCHAR)', 'status (ENUM)', 'params_json (TEXT)', 'created_at (TIMESTAMP)']):
    ax.text(23, 27.5 - i*1.5, col, fontsize=5.5, color='#555')
# Graphs entity
ax.add_patch(patches.Rectangle((24, 2), 18, 14, linewidth=1.5, edgecolor='#333', facecolor='#D1FAE5'))
ax.text(33, 14, 'saved_graphs', fontsize=8, fontweight='bold', ha='center')
ax.plot([24, 42], [11, 11], color='#333', lw=0.5)
for i, col in enumerate(['id (PK, INT)', 'job_id (FK -> prediction_jobs.id)', 'ticker_symbol (VARCHAR)', 'title (VARCHAR)', 'graph_image (LONGBLOB)', 'created_at (TIMESTAMP)']):
    ax.text(25, 9.5 - i*1.5, col, fontsize=5.5, color='#555')
# Relationships
ax.annotate('', xy=(22, 30), xytext=(20, 40), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
ax.text(18, 37, '1:N', fontsize=6, fontweight='bold', color='#333', rotation=-55)
ax.annotate('', xy=(24, 9), xytext=(40, 20), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
ax.text(28, 12, '1:1', fontsize=6, fontweight='bold', color='#333', rotation=40)

# ===== 4. Class Diagrams (OOP) =====
ax = axes[1, 0]
setup_ax(ax, 'Class Diagrams (OOP - Python)', 50, 50)
def draw_class(ax, x, y, name, attrs, methods, color='#e8e8e8'):
    # ensure class box fits within axis limits
    x_lim = ax.get_xlim()[1]
    w, h = 16, len(attrs)*1.8 + len(methods)*1.8 + 6
    if x + w > x_lim - MARGIN:
        w = max(10, x_lim - MARGIN - x)
    ax.add_patch(patches.Rectangle((x, y-h), w, h, linewidth=1.2, edgecolor='#333', facecolor=color))
    ax.text(x + w/2, y - 2, name, fontsize=7, fontweight='bold', ha='center')
    ax.plot([x, x+w], [y-4, y-4], color='#333', lw=0.5)
    for i, a in enumerate(attrs):
        ax.text(x+1, y-6 - i*1.8, a, fontsize=5.5, color='#555')
    ay = y - 6 - len(attrs)*1.8
    ax.plot([x, x+w], [ay, ay], color='#333', lw=0.5)
    for i, m in enumerate(methods):
        ax.text(x+1, ay - 1.5 - i*1.8, m, fontsize=5.5, color='#555')

draw_class(ax, 3, 48, 'BiLSTMModel', ['+ hidden_dim: int', '+ num_layers: int', '+ dropout: float', '+ fc: nn.Linear'], ['+ forward(x) -> tensor', '+ predict(seq) -> float[]', '+ save(path: str) -> None', '+ load(path: str) -> None'], '#EDE9FE')
draw_class(ax, 22, 48, 'StockPredictor', ['+ model: BiLSTMModel', '+ device: torch.device', '+ scaler: MinMaxScaler', '+ seq_len: int'], ['+ train(data, epochs)', '+ forecast(ticker, days)', '+ calculate_features(df)', '+ normalize(data)'], '#FEF3C7')
draw_class(ax, 32, 48, 'JobQueue', ['+ db: PDO', '+ ssh: SSHClient'], ['+ queue_job(ticker, params)', '+ poll_pending() -> Job[]', '+ update_status(id, status)', '+ get_results(id) -> dict'], '#DBEAFE')
# Inheritance/uses arrows with controlled shrink and mutation to avoid overlapping edges
ax.annotate('', xy=(19, 30), xytext=(19, 48-10), arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.0, shrinkA=6, shrinkB=6, mutation_scale=10))
ax.text(19, 37, 'uses', fontsize=5, ha='center', style='italic', zorder=12)
ax.annotate('', xy=(32, 30), xytext=(32, 48-10), arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.0, shrinkA=6, shrinkB=6, mutation_scale=10))

# ===== 5. Input Validation =====
ax = axes[1, 1]
setup_ax(ax, 'Input Validation Rules', 50, 50)
validations = [
    ('Ticker Symbol', 'String, 1-5 chars', 'is_string && len >= 1 && <= 5', 'regex ^[A-Z]{1,5}$ pattern', 'Invalid ticker format. Use 1-5 uppercase letters (e.g. AAPL)'),
    ('Date Range', 'start < end', 'strtotime($start) < strtotime($end)', 'start_date < end_date AND both are valid dates', 'Start date must be before end date'),
    ('Epochs (Training)', 'Integer, 1-500', 'is_int($epochs) && $epochs >= 1 && <= 500', 'epochs > 0 AND epochs <= 500', 'Epochs must be between 1 and 500'),
    ('Learning Rate', 'Float, 0.0001-0.1', 'is_float($lr) && $lr > 0 && $lr <= 0.1', 'lr >= 0.0001 AND lr <= 0.1', 'Learning rate must be between 0.0001 and 0.1'),
    ('Username', 'String, 3-50 chars', 'ctype_alnum($user) && length >= 3', 'username UNIQUE and matches pattern ^[a-zA-Z0-9_]{3,50}$', 'Username must be 3-50 alphanumeric characters'),
    ('Password', 'String, min 8 chars', 'strlen($pw) >= 8', 'password must not be common (check against breach list)', 'Password must be at least 8 characters'),
    ('Sequence Length', 'Integer, 5-120', 'is_int($seq) && $seq >= 5', 'seq_len >= 5 (minimum for MA calculation)', 'Sequence length must be at least 5 days'),
    ('Forecast Days', 'Integer, 1-90', 'is_int($days) && $days >= 1 && $days <= 90', 'forecast_days >= 1 AND <= 90', 'Forecast period must be between 1 and 90 days'),
]
headers2 = ['Field', 'Type Constraint', 'PHP Validation', 'DB Constraint', 'Error Message']
col_ws = [10, 10, 12, 12, 6]
cx = 2
hdr_y = 48
for j, h in enumerate(headers2):
    ax.add_patch(patches.Rectangle((cx, hdr_y - 2), col_ws[j], 2.2, facecolor='#e8e8e8', edgecolor='#999', lw=0.5))
    ax.text(cx + col_ws[j]/2, hdr_y - 0.9, h, fontsize=6, fontweight='bold', ha='center')
    cx += col_ws[j]
row_y = hdr_y - 2
for i, row in enumerate(validations):
    bg = '#f9f9f9' if i % 2 == 0 else '#fff'
    cx = 2
    for j, val in enumerate(row):
        ax.add_patch(patches.Rectangle((cx, row_y - 2), col_ws[j], 2.2, facecolor=bg, edgecolor='#ddd', lw=0.3))
        ax.text(cx + col_ws[j]/2, row_y - 0.9, val, fontsize=5, ha='center', color='#444')
        cx += col_ws[j]
    row_y -= 2.2

# ===== 6. Data Structures Summary =====
ax = axes[1, 2]
setup_ax(ax, 'Data Structures Used', 50, 50)
structures = [
    ('PyTorch Tensor', 'torch.Tensor', 'Multi-dimensional arrays for model input/output. Shape: [batch, seq_len, features]. Stores normalized stock data and predictions.', 'ML Input/Output'),
    ('Pandas DataFrame', 'pd.DataFrame', 'Tabular data structure for OHLCV prices, indicators, and results. Supports date indexing, column operations, and CSV export.', 'Data Processing'),
    ('PHP Associative Array', 'array<string, mixed>', 'Stores user session data, job parameters (JSON decoded), and template rendering context. Key-value pairs.', 'Web Layer'),
    ('Queue (FIFO)', 'Array/DB table', 'prediction_jobs table acts as a FIFO queue. Worker polls WHERE status="queued" ORDER BY created_at ASC LIMIT 1.', 'Job Scheduling'),
    ('Dictionary (Config)', 'dict / JSON', 'Model hyperparameters stored as JSON in params_json column. Loaded as Python dict. Keys: epochs, lr, hidden_dim, etc.', 'Configuration'),
    ('Binary BLOB', 'LONGBLOB (MySQL)', 'Stores PNG graph images as raw binary in saved_graphs.graph_image. Max 4GB. Retrieved and sent as image/jpeg response.', 'Graph Storage'),
    ('SSH Channel', 'paramiko.Channel', 'Bidirectional pipe for remote command execution. Captures stdout, stderr streams. Used to trigger training on GPU machine.', 'Remote Execution'),
    ('SFTP File Handle', 'paramiko.SFTPFile', 'Stream-based file transfer object. Supports read/write in binary mode. Used to transfer .pt models and .csv results.', 'File Transfer'),
]
for i, (name, dtype, desc, context) in enumerate(structures):
    y_pos = 48 - i * 5.5
    ax.add_patch(patches.Rectangle((2, y_pos - 3.5), 46, 4.5, facecolor='#f9f9f9' if i % 2 == 0 else '#fff', edgecolor='#ddd', lw=0.5))
    ax.text(3, y_pos, name, fontsize=7, fontweight='bold', color='#1e3a8a')
    ax.text(3, y_pos - 1.5, dtype, fontsize=6, color='#666', style='italic')
    ax.text(3, y_pos - 3, desc[:85] + ('...' if len(desc) > 85 else ''), fontsize=5.5, color='#555')
    ax.text(44, y_pos, context, fontsize=5.5, color='#888', ha='right', style='italic')

plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.15, hspace=0.15)
fig.savefig(os.path.join(OUTPUT_DIR, 'slide19_design_detailed.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close(fig)
print('Generated: slide19_design_detailed.png')
