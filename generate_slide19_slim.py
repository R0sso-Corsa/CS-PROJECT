import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = r'C:\Users\paron\Desktop\Dev\CS_PROJECT\docs\diagrams'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Slim slide 19: ERD + Most important variables only
def draw_erd(ax):
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#ffffff')

    # users
    ax.add_patch(patches.Rectangle((4, 33), 18, 10, linewidth=1.2, edgecolor='#333', facecolor='#FEF3C7'))
    ax.text(13, 41, 'users', fontsize=10, fontweight='bold', ha='center')
    ax.text(6, 37.8, 'id (PK)', fontsize=8, color='#333')
    ax.text(6, 36.2, 'username', fontsize=8, color='#333')

    # prediction_jobs
    ax.add_patch(patches.Rectangle((26, 22), 24, 12, linewidth=1.2, edgecolor='#333', facecolor='#DBEAFE'))
    ax.text(38, 33, 'prediction_jobs', fontsize=9, fontweight='bold', ha='center')
    ax.text(28, 29.5, 'id (PK)', fontsize=8, color='#333')
    ax.text(28, 28.0, 'user_id (FK)', fontsize=8, color='#333')
    ax.text(28, 26.5, 'ticker_symbol', fontsize=8, color='#333')

    # saved_graphs
    ax.add_patch(patches.Rectangle((26, 6), 24, 10, linewidth=1.2, edgecolor='#333', facecolor='#D1FAE5'))
    ax.text(38, 14, 'saved_graphs', fontsize=9, fontweight='bold', ha='center')
    ax.text(28, 11.2, 'id (PK)', fontsize=8, color='#333')
    ax.text(28, 9.6, 'job_id (FK)', fontsize=8, color='#333')

    # relationships
    ax.annotate('', xy=(26, 31), xytext=(22, 38), arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.0, shrinkA=6, shrinkB=6, mutation_scale=10))
    ax.text(23, 36, '1:N', fontsize=8, color='#333', rotation=-55)
    ax.annotate('', xy=(28, 12), xytext=(46, 22), arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.0, shrinkA=6, shrinkB=6, mutation_scale=10))

def draw_key_vars(ax):
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')
    ax.axis('off')

    # card
    x, y, w, h = 4, 40, 32, 28
    ax.add_patch(patches.Rectangle((x, 12), w, h, linewidth=0.9, edgecolor='#bbb', facecolor='#fff', zorder=1))
    ax.text(x + w/2, 36, 'Key Variables (Most Important)', fontsize=11, fontweight='bold', ha='center', color='#1e3a8a')

    vars_grouped = [
        ('Application', ['user_id', 'username', 'db (connection)']),
        ('Job Queue', ['job_id', 'ticker_symbol', 'status']),
        ('ML', ['model (BiLSTM)', 'device (cuda/cpu)']),
    ]

    gy = 32
    for title, items in vars_grouped:
        ax.text(x + 1.2, gy, title, fontsize=9, fontweight='bold', color='#333')
        gy -= 1.8
        for it in items:
            ax.text(x + 2.6, gy, '• ' + it, fontsize=8.5, color='#444')
            gy -= 1.6
        gy -= 0.8

def main():
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    draw_erd(ax0)
    draw_key_vars(ax1)
    out = os.path.join(OUTPUT_DIR, 'slide19_slim.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print('Generated:', out)

if __name__ == '__main__':
    main()
