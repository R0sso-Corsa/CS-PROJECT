import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUTPUT_DIR = r'C:\Users\paron\Desktop\Dev\CS_PROJECT\docs\diagrams'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rounded_box(ax, xy, w, h, text, facecolor='#fff', edgecolor='#444', fontsize=9, title=False):
    box = FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.6,rounding_size=6', linewidth=1.0, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(box)
    tx = xy[0] + 8
    ty = xy[1] + h - (14 if title else 10)
    ax.text(tx, ty, text, fontsize=fontsize, fontweight='bold' if title else 'normal', color='#1f2937')
    return box

def draw_redesign(path_out):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 75)
    ax.axis('off')
    ax.set_facecolor('#f7fafc')

    # Header
    ax.text(6, 71, 'System Design — Databases & Key Variables', fontsize=16, fontweight='bold', color='#0f172a')
    ax.add_patch(patches.Rectangle((4, 68), 132, 0.6, facecolor='#e6eef8', edgecolor='none'))

    # Left: ERD group with stylized boxes (repositioned to avoid overlap)
    # users (top-left)
    rounded_box(ax, (6, 46), 36, 16, 'users', facecolor='#fff7ed', edgecolor='#d97706', title=True)
    ax.text(10, 56, 'id (PK)  •  username', fontsize=9, color='#374151')

    # prediction_jobs (center-left)
    rounded_box(ax, (46, 30), 40, 20, 'prediction_jobs', facecolor='#eef2ff', edgecolor='#3742b3', title=True)
    ax.text(50, 46, 'id (PK)  •  user_id (FK)  •  ticker_symbol', fontsize=9, color='#374151')

    # saved_graphs (lower-right of ERD)
    rounded_box(ax, (90, 12), 36, 18, 'saved_graphs', facecolor='#ecfdf5', edgecolor='#059669', title=True)
    ax.text(94, 26, 'id (PK)  •  job_id (FK)  •  graph_image', fontsize=9, color='#374151')

    # Connectors with clearer arrows and no weird pointers (start/end outside box edges)
    arr1 = FancyArrowPatch((42, 52), (46, 46), connectionstyle='arc3,rad=0.0', arrowstyle='-|>', mutation_scale=12, linewidth=1.1, color='#111827')
    ax.add_patch(arr1)
    arr2 = FancyArrowPatch((78, 38), (90, 28), connectionstyle='arc3,rad=-0.08', arrowstyle='-|>', mutation_scale=12, linewidth=1.1, color='#111827')
    ax.add_patch(arr2)

    # Middle: Data flow lane
    ax.add_patch(patches.Rectangle((6, 4), 128, 6, facecolor='#fff', edgecolor='#e2e8f0'))
    ax.text(10, 8, 'Data Flow: user -> job queued -> ML processes -> saved graphs', fontsize=9, color='#0f172a')

    # Right: Key variables card (compact and focused)
    kv_x, kv_y, kv_w, kv_h = 90, 46, 40, 24
    rounded_box(ax, (kv_x, kv_y), kv_w, kv_h, 'Key Variables (Top)', facecolor='#ffffff', edgecolor='#94a3b8', title=True)
    key_vars = [
        ('user_id', 'int — session identifier'),
        ('username', 'string — display'),
        ('job_id', 'int — queue identifier'),
        ('ticker_symbol', 'str — user input'),
        ('model', 'BiLSTM — PyTorch'),
    ]
    vy = kv_y + kv_h - 6
    for name, desc in key_vars:
        ax.text(kv_x + 3.5, vy, '• ' + name + ': ' + desc, fontsize=8.5, color='#1f2937')
        vy -= 3.8

    # Bottom-left: small table summary with subtle grid
    bx, by = 6, 14
    bw, bh = 60, 14
    rounded_box(ax, (bx, by), bw, bh, 'Table Summary', facecolor='#ffffff', edgecolor='#cbd5e1', title=True)
    ax.text(bx + 4, by + bh - 7, 'users(id, username)  |  prediction_jobs(id, user_id, ticker_symbol)  |  saved_graphs(id, job_id)', fontsize=8, color='#334155')

    # Legend / footnote
    ax.text(6, 2.8, 'Note: Diagram highlights only the most important variables and relationships for clarity.', fontsize=7.5, color='#475569')

    fig.savefig(path_out, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

if __name__ == '__main__':
    out = os.path.join(OUTPUT_DIR, 'slide19_redesign.png')
    draw_redesign(out)
    print('Generated:', out)
