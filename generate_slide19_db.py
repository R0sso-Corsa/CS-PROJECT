import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = r'C:\Users\paron\Desktop\Dev\CS_PROJECT\docs\diagrams'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MARGIN = 2.5
INNER_PADDING = 1.2

def setup_ax(ax, title, x_lim, y_lim):
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_aspect('equal')
    ax.set_facecolor('#fafafa')
    ax.axis('off')
    ax.text(x_lim/2, y_lim - MARGIN/2, title, fontsize=12, fontweight='bold', ha='center', color='#1e3a8a', zorder=10)
    card_x = MARGIN - INNER_PADDING
    card_y = MARGIN - INNER_PADDING
    card_w = x_lim - 2 * (MARGIN - INNER_PADDING)
    card_h = y_lim - 2 * (MARGIN - INNER_PADDING) - 0.5
    rect = patches.Rectangle((card_x, card_y), card_w, card_h, linewidth=0.9, edgecolor='#e6e6e6', facecolor='#ffffff', zorder=0)
    rect.set_clip_on(False)
    ax.add_patch(rect)

def draw_small_table(ax, x, y, title, cols, width=16):
    header_h = 2.2
    row_h = 1.6
    num_rows = len(cols)
    total_h = header_h + num_rows * row_h + 0.6
    # card background
    ax.add_patch(patches.Rectangle((x, y - total_h), width, total_h, linewidth=0.9, edgecolor='#ccc', facecolor='#fff', zorder=1))
    ax.text(x + width/2, y - 0.6, title, fontsize=9, fontweight='bold', ha='center', color='#1e3a8a', zorder=3)
    # header band
    ax.add_patch(patches.Rectangle((x + 0.2, y - 2.6), width - 0.4, header_h, facecolor='#e8e8e8', edgecolor='#999', zorder=2))
    # rows
    ry = y - 2.6 - header_h
    for i, row in enumerate(cols):
        bg = '#f9f9f9' if i % 2 == 0 else '#fff'
        ax.add_patch(patches.Rectangle((x + 0.2, ry - row_h), width - 0.4, row_h, facecolor=bg, edgecolor='#ddd', zorder=1))
        txt = f"{row[0]} — {row[1]}"
        ax.text(x + 0.4, ry - row_h/2 - 0.2, txt, fontsize=7.5, color='#444', zorder=3)
        ry -= row_h

def main():
    users_cols = [
        ('id', 'INT PK', 'AUTO_INCREMENT', '1'),
        ('username', 'VARCHAR(50)', 'UNIQUE, NOT NULL', 'demo'),
        ('password_hash', 'VARCHAR(255)', 'NOT NULL (bcrypt)', '<hash>'),
        ('created_at', 'TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP', '2025-01-01 00:00:00'),
    ]
    jobs_cols = [
        ('id', 'INT PK', 'AUTO_INCREMENT', '1057'),
        ('user_id', 'INT FK', 'REFERENCES users(id)', '1'),
        ('ticker_symbol', 'VARCHAR(10)', 'NOT NULL', 'AAPL'),
        ('status', 'ENUM', 'queued/running/completed/failed', 'running'),
        ('params_json', 'TEXT', 'Forecast parameters (nullable)', '{"epochs":30}'),
        ('created_at', 'TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP', '2025-01-15 14:30:00'),
        ('completed_at', 'TIMESTAMP', 'NULLABLE', 'NULL'),
    ]
    graphs_cols = [
        ('id', 'INT PK', 'AUTO_INCREMENT', '88'),
        ('job_id', 'INT FK', 'REFERENCES prediction_jobs(id)', '1057'),
        ('ticker_symbol', 'VARCHAR(10)', 'NOT NULL', 'AAPL'),
        ('title', 'VARCHAR(255)', 'Chart title', 'AAPL Forecast'),
        ('graph_image', 'LONGBLOB', 'PNG image binary', '<binary>'),
        ('created_at', 'TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP', '2025-01-15 14:45:00'),
    ]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(22, 10))
    plt.subplots_adjust(wspace=0.18, left=0.03, right=0.98, top=0.95, bottom=0.03)

    # ERD on the left
    setup_ax(ax_left, 'Entity Relationship Diagram', 60, 50)
    ax_left.add_patch(patches.Rectangle((4, 33), 18, 10, linewidth=1.2, edgecolor='#333', facecolor='#FEF3C7', zorder=2))
    ax_left.text(13, 41, 'users', fontsize=10, fontweight='bold', ha='center')
    ax_left.plot([4, 22], [38.8, 38.8], color='#333', lw=0.6)
    for i, col in enumerate(['id (PK, INT)', 'username (VARCHAR)', 'password_hash', 'created_at (TIMESTAMP)']):
        ax_left.text(5, 37.5 - i*1.4, col, fontsize=8, color='#555')

    ax_left.add_patch(patches.Rectangle((26, 22), 22, 12, linewidth=1.2, edgecolor='#333', facecolor='#DBEAFE', zorder=2))
    ax_left.text(37, 33, 'prediction_jobs', fontsize=9, fontweight='bold', ha='center')
    ax_left.plot([26, 48], [30.5, 30.5], color='#333', lw=0.6)
    for i, col in enumerate(['id (PK, INT)', 'user_id (FK -> users.id)', 'ticker_symbol', 'status', 'params_json', 'created_at']):
        ax_left.text(27, 29 - i*1.2, col, fontsize=7.5, color='#555')

    ax_left.add_patch(patches.Rectangle((26, 4), 22, 12, linewidth=1.2, edgecolor='#333', facecolor='#D1FAE5', zorder=2))
    ax_left.text(37, 15, 'saved_graphs', fontsize=9, fontweight='bold', ha='center')
    ax_left.plot([26, 48], [11.8, 11.8], color='#333', lw=0.6)
    for i, col in enumerate(['id (PK, INT)', 'job_id (FK -> prediction_jobs.id)', 'ticker_symbol', 'title', 'graph_image', 'created_at']):
        ax_left.text(27, 10.5 - i*1.2, col, fontsize=7.5, color='#555')

    ax_left.annotate('', xy=(26, 31), xytext=(22, 38), arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.2, shrinkA=6, shrinkB=6, mutation_scale=12))
    ax_left.text(23, 36, '1:N', fontsize=8, fontweight='bold', color='#333', rotation=-55)
    ax_left.annotate('', xy=(28, 14), xytext=(46, 24), arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.2, shrinkA=6, shrinkB=6, mutation_scale=12))
    ax_left.text(32, 18, '1:1', fontsize=8, fontweight='bold', color='#333', rotation=38)

    # Tables side-by-side on the right
    setup_ax(ax_right, 'Data Dictionary (tables)', 60, 50)
    xs = [4, 22, 40]
    draw_small_table(ax_right, xs[0], 44, 'users', users_cols, width=16)
    draw_small_table(ax_right, xs[1], 44, 'prediction_jobs', jobs_cols, width=16)
    draw_small_table(ax_right, xs[2], 44, 'saved_graphs', graphs_cols, width=16)

    fig.savefig(os.path.join(OUTPUT_DIR, 'slide19_databases.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
    print('Generated: slide19_databases.png')

if __name__ == '__main__':
    main()
