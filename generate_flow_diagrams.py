import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = r'C:\Users\paron\Desktop\Dev\CS_PROJECT\docs\diagrams\flow'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_flow_diagram_1():
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 160)
    ax.set_aspect('equal')
    ax.set_facecolor('#fff')
    ax.axis('off')

    # Title
    ax.text(120, 157, 'Flow Diagram: Prediction Request Pipeline', fontsize=14, fontweight='bold', color='#1e3a8a', ha='center')
    ax.text(120, 153, 'Shows how module inputs and outputs pipeline to form a complete system (Slide 16)', fontsize=10, color='#666', ha='center')

    layer_colors = {
        'User': '#3B82F6',
        'App': '#F59E0B',
        'Data': '#10B981',
        'Computation': '#8B5CF6',
        'External': '#EF4444',
    }

    # Define nodes with positions
    nodes = {
        'User': (12, 120, 'User', 'User', 'User'),
        'PHP Backend': (35, 120, 'PHP Backend', 'Route request, render HTML', 'App'),
        'Auth Check': (58, 120, 'Auth Service', 'Validate session, check login', 'App'),
        'Search/Ticker': (81, 120, 'Search Page', 'User enters ticker symbol', 'App'),
        'Job Queue': (104, 120, 'Job Queue', 'Create job record in DB', 'App'),
        'yfinance': (127, 120, 'yfinance API', 'Fetch historical price data', 'External'),
        'Data Clean': (150, 120, 'Data Cleaning', 'Remove NaN, handle outliers', 'App'),
        'Indicators': (173, 120, 'Tech Indicators', 'Calculate MA, RSI, MACD', 'App'),
        'Normalize': (196, 120, 'Normalization', 'Scale to [0,1], tensor conv', 'App'),
        'Input Proc': (219, 120, 'Input Processing', 'Reshape to batched tensor', 'Computation'),
        'LSTM': (219, 95, 'BiLSTM Model', 'Forward/backward LSTM pass', 'Computation'),
        'MC Dropout': (196, 95, 'MC Dropout', 'N stochastic passes', 'Computation'),
        'Uncertainty': (173, 95, 'Uncertainty', 'Compute variance across samples', 'Computation'),
        'Confidence': (150, 95, 'Confidence Intervals', 'Mean +/- 1.96*std bands', 'Computation'),
        'Format': (127, 95, 'Result Formatting', 'Merge predictions with dates', 'App'),
        'Graph': (104, 95, 'Generate Graph', 'Render prediction chart', 'App'),
        'CSV Export': (81, 95, 'CSV Export', 'Write results to .csv file', 'Data'),
        'SFTP': (58, 95, 'Secure File Transfer', 'Transfer files to remote', 'External'),
        'Remote': (35, 95, 'Remote Storage', 'Store on remote machine', 'External'),
        'DB Users': (35, 70, 'Users Table (MySQL)', 'Store user credentials', 'Data'),
        'DB Jobs': (58, 70, 'Jobs Table (MySQL)', 'Store job records/status', 'Data'),
        'Models': (81, 70, 'Model Files (.pt)', 'Load trained model weights', 'Data'),
        'Training': (104, 70, 'Training Engine', 'Train BiLSTM on data', 'Computation'),
        'GPU': (127, 70, 'GPU Acceleration', 'CUDA compute execution', 'Computation'),
        'Hyper': (150, 70, 'Hyperparameter Tuning', 'Grid search for optimal params', 'Computation'),
    }

    # Draw nodes
    for name, (x, y, title, desc, layer) in nodes.items():
        color = layer_colors.get(layer, '#666')
        ax.add_patch(patches.FancyBboxPatch((x-10, y-8), 20, 16, boxstyle="round,pad=1",
                                             linewidth=1.5, edgecolor='#333', facecolor=color, alpha=0.85, zorder=1))
        ax.text(x, y+3, title, fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=2, color='#fff')
        ax.text(x, y-3, desc, fontsize=6, ha='center', va='center', zorder=2, color='#eee')

    # Draw pipelining arrows (main flow - top row left to right)
    main_flow = [
        ('User', 'PHP Backend', 'HTTP request'),
        ('PHP Backend', 'Auth Check', 'session cookie'),
        ('Auth Check', 'Search/Ticker', 'valid session'),
        ('Search/Ticker', 'Job Queue', 'ticker symbol'),
        ('Job Queue', 'yfinance', 'job created'),
        ('yfinance', 'Data Clean', 'raw OHLCV'),
        ('Data Clean', 'Indicators', 'cleaned data'),
        ('Indicators', 'Normalize', 'indicator values'),
        ('Normalize', 'Input Proc', 'normalized tensor'),
    ]

    for src, dst, label in main_flow:
        sx, sy = nodes[src][0], nodes[src][1]
        dx, dy = nodes[dst][0], nodes[dst][1]
        ax.annotate('', xy=(dx-10, dy), xytext=(sx+10, sy),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
        mx, my = (sx+dx)/2, sy + 4
        ax.text(mx, my, label, fontsize=5.5, ha='center', color='#555', style='italic')

    # Down arrows
    down_flow = [
        ('Input Proc', 'LSTM', 'batched tensor'),
        ('LSTM', 'MC Dropout', 'hidden states'),
        ('MC Dropout', 'Uncertainty', 'prediction samples'),
        ('Uncertainty', 'Confidence', 'variance'),
        ('Confidence', 'Format', 'bounds + mean'),
    ]

    for src, dst, label in down_flow:
        sx, sy = nodes[src][0], nodes[src][1] - 8
        dx, dy = nodes[dst][0], nodes[dst][1] + 8
        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#6366f1', lw=1.5))
        ax.text(sx + 5, (sy+dy)/2, label, fontsize=5.5, color='#6366f1', style='italic')

    # Return flow
    return_flow = [
        ('Format', 'Graph', 'formatted data'),
        ('Graph', 'CSV Export', 'graph image'),
        ('CSV Export', 'SFTP', 'csv + image'),
        ('SFTP', 'Remote', 'transferred files'),
    ]

    for src, dst, label in return_flow:
        sx, sy = nodes[src][0], nodes[src][1]
        dx, dy = nodes[dst][0], nodes[dst][1]
        ax.annotate('', xy=(dx+10, dy), xytext=(sx-10, sy),
                    arrowprops=dict(arrowstyle='->', color='#10b981', lw=1.5))
        mx, my = (sx+dx)/2, sy - 12
        ax.text(mx, my, label, fontsize=5.5, ha='center', color='#10b981', style='italic')

    # Data connections
    ax.annotate('', xy=(58-10, 70), xytext=(58, 95-8),
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1, linestyle='--'))
    ax.text(45, 82, 'job records', fontsize=5, color='#10b981', style='italic')

    ax.annotate('', xy=(81-10, 70), xytext=(104, 95-8),
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1, linestyle='--'))
    ax.text(85, 82, 'graph archive', fontsize=5, color='#10b981', style='italic')

    # Training flow
    ax.annotate('', xy=(104-10, 70), xytext=(127, 120-8),
                arrowprops=dict(arrowstyle='->', color='#8b5cf6', lw=1, linestyle='--'))
    ax.text(122, 95, 'data for training', fontsize=5, color='#8b5cf6', style='italic')

    ax.annotate('', xy=(127, 70+8), xytext=(127, 95-8),
                arrowprops=dict(arrowstyle='->', color='#8b5cf6', lw=1, linestyle='--'))
    ax.text(132, 83, 'model weights', fontsize=5, color='#8b5cf6', style='italic')

    # GPU/Hyper connections
    ax.annotate('', xy=(127-10, 70), xytext=(104+10, 70),
                arrowprops=dict(arrowstyle='->', color='#8b5cf6', lw=1, linestyle='--'))
    ax.text(115, 67, 'GPU tensors', fontsize=5, color='#8b5cf6', style='italic')

    ax.annotate('', xy=(150-10, 70), xytext=(127+10, 70),
                arrowprops=dict(arrowstyle='->', color='#8b5cf6', lw=1, linestyle='--'))
    ax.text(138, 67, 'hyperparams', fontsize=5, color='#8b5cf6', style='italic')

    ax.annotate('', xy=(81, 70+8), xytext=(104, 95-8),
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1, linestyle='--'))
    ax.text(92, 82, 'model .pt', fontsize=5, color='#10b981', style='italic')

    # Legend
    legend_x = 175
    legend_y = 65
    for i, (layer, color) in enumerate(layer_colors.items()):
        ax.add_patch(patches.Rectangle((legend_x, legend_y - i*6), 4, 4, linewidth=0.5, edgecolor='#333', facecolor=color, alpha=0.85))
        ax.text(legend_x + 6, legend_y - i*6 + 2, layer + ' Layer', fontsize=7, va='center', color='#333')

    # Arrow legend
    ax.annotate('', xy=(legend_x, legend_y - 35), xytext=(legend_x - 12, legend_y - 35),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text(legend_x + 2, legend_y - 34, 'Main pipeline flow', fontsize=6.5, va='center')

    ax.annotate('', xy=(legend_x, legend_y - 41), xytext=(legend_x - 12, legend_y - 41),
                arrowprops=dict(arrowstyle='->', color='#6366f1', lw=1.5))
    ax.text(legend_x + 2, legend_y - 40, 'Computation flow', fontsize=6.5, va='center')

    ax.annotate('', xy=(legend_x, legend_y - 47), xytext=(legend_x - 12, legend_y - 47),
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1.5))
    ax.text(legend_x + 2, legend_y - 46, 'Data/Storage flow', fontsize=6.5, va='center')

    ax.annotate('', xy=(legend_x, legend_y - 53), xytext=(legend_x - 12, legend_y - 53),
                arrowprops=dict(arrowstyle='->', color='#8b5cf6', lw=1, linestyle='--'))
    ax.text(legend_x + 2, legend_y - 52, 'Training pipeline', fontsize=6.5, va='center')

    # Pipelining explanation
    expl_x = 15
    expl_y = 55
    ax.add_patch(patches.Rectangle((expl_x, expl_y - 35), 70, 38, linewidth=1, edgecolor='#aaa', facecolor='#fafafa', zorder=1))
    ax.text(expl_x + 3, expl_y - 1, 'Pipelining Explanation', fontsize=9, fontweight='bold', zorder=2)
    expl_lines = [
        '1. User request -> PHP routes to Auth',
        '   Auth validates session -> Search page',
        '2. Ticker submitted -> Job queued in DB',
        '3. Data fetched (yfinance) -> cleaned',
        '   -> indicators calculated -> normalized',
        '4. Tensor passed to BiLSTM -> MC Dropout',
        '   generates N samples for uncertainty',
        '5. Confidence intervals computed',
        '   -> results formatted -> graph generated',
        '6. Graph + CSV exported via SFTP to remote',
        '7. Job status updated in DB throughout',
    ]
    for i, line in enumerate(expl_lines):
        ax.text(expl_x + 3, expl_y - 5 - i*3, line, fontsize=6, zorder=2, color='#444')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'flow_prediction_pipeline.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print('Generated: flow_prediction_pipeline.png')


def draw_flow_diagram_2():
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 160)
    ax.set_aspect('equal')
    ax.set_facecolor('#fff')
    ax.axis('off')

    ax.text(120, 157, 'Flow Diagram: Training & SSH Execution Pipeline', fontsize=14, fontweight='bold', color='#1e3a8a', ha='center')
    ax.text(120, 153, 'Shows how the training engine connects to remote GPU execution via SSH', fontsize=10, color='#666', ha='center')

    layer_colors = {
        'App': '#F59E0B',
        'Data': '#10B981',
        'Computation': '#8B5CF6',
        'External': '#EF4444',
    }

    nodes = {
        'Queue Worker': (12, 120, 'Queue Worker', 'PHP polls jobs table', 'App'),
        'SSH Connect': (35, 120, 'SSH Connection', 'Establish paramiko session', 'External'),
        'Remote Cmd': (58, 120, 'Remote Command', 'Send training script execution', 'External'),
        'Fetch Data': (81, 120, 'yfinance Fetch', 'Retrieve training data', 'External'),
        'Clean/Prep': (104, 120, 'Clean + Preprocess', 'NaN removal, feature engineering', 'App'),
        'Hyper Search': (127, 120, 'Hyperparameter Search', 'Grid/random search over params', 'Computation'),
        'GPU Train': (150, 120, 'GPU Training', 'CUDA-accelerated BiLSTM training', 'Computation'),
        'LSTM Layers': (173, 120, 'LSTM Forward/Back', 'Bidirectional sequence processing', 'Computation'),
        'MC Dropout': (196, 120, 'MC Dropout', 'Stochastic forward passes', 'Computation'),
        'Evaluate': (219, 120, 'Model Evaluation', 'Compute loss, save best model', 'Computation'),
        'Save Model': (219, 95, 'Save Model (.pt)', 'Serialize state_dict to disk', 'Data'),
        'SFTP Back': (196, 95, 'SFTP Transfer', 'Transfer model/graph back', 'External'),
        'Import DB': (173, 95, 'Import to Database', 'Store graph in DB archive', 'Data'),
        'Update Job': (150, 95, 'Update Job Status', 'Mark completed in jobs table', 'App'),
        'Graph Archive': (127, 95, 'Graph Archive', 'Store in MySQL longblob', 'Data'),
        'Notify User': (104, 95, 'User Notification', 'Flash message, status update', 'App'),
    }

    for name, (x, y, title, desc, layer) in nodes.items():
        color = layer_colors.get(layer, '#666')
        ax.add_patch(patches.FancyBboxPatch((x-10, y-8), 20, 16, boxstyle="round,pad=1",
                                             linewidth=1.5, edgecolor='#333', facecolor=color, alpha=0.85, zorder=1))
        ax.text(x, y+3, title, fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=2, color='#fff')
        ax.text(x, y-3, desc, fontsize=6, ha='center', va='center', zorder=2, color='#eee')

    # Main flow left to right
    main_flow = [
        ('Queue Worker', 'SSH Connect', 'poll jobs'),
        ('SSH Connect', 'Remote Cmd', 'SSH session'),
        ('Remote Cmd', 'Fetch Data', 'execute script'),
        ('Fetch Data', 'Clean/Prep', 'raw OHLCV'),
        ('Clean/Prep', 'Hyper Search', 'cleaned data'),
        ('Hyper Search', 'GPU Train', 'best params'),
        ('GPU Train', 'LSTM Layers', 'batched tensor'),
        ('LSTM Layers', 'MC Dropout', 'hidden states'),
        ('MC Dropout', 'Evaluate', 'prediction samples'),
    ]

    for src, dst, label in main_flow:
        sx, sy = nodes[src][0], nodes[src][1]
        dx, dy = nodes[dst][0], nodes[dst][1]
        ax.annotate('', xy=(dx-10, dy), xytext=(sx+10, sy),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
        mx, my = (sx+dx)/2, sy + 4
        ax.text(mx, my, label, fontsize=5.5, ha='center', color='#555', style='italic')

    # Down flow
    down_flow = [
        ('Evaluate', 'Save Model', 'best weights'),
        ('Save Model', 'SFTP Back', '.pt file'),
        ('SFTP Back', 'Import DB', 'graph images'),
        ('Import DB', 'Update Job', 'import result'),
        ('Update Job', 'Graph Archive', 'completed job'),
        ('Graph Archive', 'Notify User', 'archive updated'),
    ]

    for src, dst, label in down_flow:
        sx, sy = nodes[src][0], nodes[src][1] - 8
        dx, dy = nodes[dst][0], nodes[dst][1] + 8
        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#6366f1', lw=1.5))
        ax.text(sx + 8, (sy+dy)/2, label, fontsize=5.5, color='#6366f1', style='italic')

    # DB connections
    ax.annotate('', xy=(150-10, 95), xytext=(150, 120-8),
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1, linestyle='--'))
    ax.text(142, 110, 'job record', fontsize=5, color='#10b981', style='italic')

    ax.annotate('', xy=(127-10, 95), xytext=(173, 120-8),
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1, linestyle='--'))
    ax.text(142, 100, 'graph data', fontsize=5, color='#10b981', style='italic')

    # Legend
    legend_x = 15
    legend_y = 70
    ax.add_patch(patches.Rectangle((legend_x, legend_y - 25), 65, 28, linewidth=1, edgecolor='#aaa', facecolor='#fafafa', zorder=1))
    ax.text(legend_x + 3, legend_y - 1, 'Pipelining Explanation', fontsize=9, fontweight='bold', zorder=2)
    expl_lines = [
        '1. Queue Worker polls jobs table for pending',
        '2. SSH connects to remote GPU machine',
        '3. Remote command executes training script',
        '4. Data fetched via yfinance, cleaned locally',
        '5. Hyperparameter search finds best config',
        '6. GPU training runs BiLSTM on CUDA device',
        '7. MC Dropout generates uncertainty estimates',
        '8. Model evaluated, best weights saved',
        '9. Model + graphs transferred back via SFTP',
        '10. Results imported to DB, job marked done',
        '11. User notified on next page visit',
    ]
    for i, line in enumerate(expl_lines):
        ax.text(legend_x + 3, legend_y - 5 - i*2.2, line, fontsize=6, zorder=2, color='#444')

    # Arrow legend
    alx = 85
    aly = 70
    ax.annotate('', xy=(alx, aly - 2), xytext=(alx - 10, aly - 2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text(alx + 2, aly - 1, 'Main execution flow', fontsize=6.5, va='center')

    ax.annotate('', xy=(alx, aly - 8), xytext=(alx - 10, aly - 8),
                arrowprops=dict(arrowstyle='->', color='#6366f1', lw=1.5))
    ax.text(alx + 2, aly - 7, 'Result/Storage flow', fontsize=6.5, va='center')

    ax.annotate('', xy=(alx, aly - 14), xytext=(alx - 10, aly - 14),
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1, linestyle='--'))
    ax.text(alx + 2, aly - 13, 'Database connections', fontsize=6.5, va='center')

    # Layer legend
    lx = 85
    ly = 55
    for i, (layer, color) in enumerate(layer_colors.items()):
        ax.add_patch(patches.Rectangle((lx, ly - i*5), 4, 4, linewidth=0.5, edgecolor='#333', facecolor=color, alpha=0.85))
        ax.text(lx + 6, ly - i*5 + 2, layer + ' Layer', fontsize=7, va='center', color='#333')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'flow_training_ssh.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print('Generated: flow_training_ssh.png')


draw_flow_diagram_1()
draw_flow_diagram_2()
print('\nAll flow diagrams generated in: %s' % OUTPUT_DIR)
