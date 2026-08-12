import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 16))
ax.set_xlim(0, 110)
ax.set_ylim(0, 140)
ax.set_aspect('equal')
ax.set_facecolor('#fff')
ax.axis('off')

# Helper for inner section boxes
def inner_box(x, y, w, h, **kwargs):
    kwargs.setdefault('linewidth', 0.8)
    kwargs.setdefault('edgecolor', '#aaa')
    kwargs.setdefault('facecolor', 'none')
    kwargs.setdefault('linestyle', '--')
    return patches.Rectangle((x, y), w, h, **kwargs)

# HEADER
header_y = 132
ax.add_patch(patches.Rectangle((7, header_y - 12), 96, 16, linewidth=1, edgecolor='#555', facecolor='#e8e8e8', zorder=1))
ax.text(10, header_y + 1, 'Stock Price Prediction System', fontsize=12, fontweight='bold', zorder=2)
ax.text(10, header_y - 6, 'Queued ticker forecasting website.', fontsize=8, color='#666', zorder=2)

# Nav bar
nav_y = header_y - 16
ax.add_patch(patches.Rectangle((7, nav_y - 5), 96, 8, linewidth=1, edgecolor='#555', facecolor='#f0f0f0', zorder=1))
nav_items = ['Home', 'Search', 'Log In', 'Create Account']
x = 10
for i, item in enumerate(nav_items):
    ax.text(x, nav_y - 1, item, fontsize=8, zorder=2, style='italic')
    x += len(item) * 1.0 + 3
    if i < len(nav_items) - 1:
        ax.text(x - 1.5, nav_y - 1, '|', fontsize=8, color='#999', zorder=2)

# Separator
sep1 = header_y - 22
ax.plot([7, 103], [sep1, sep1], color='#999', linewidth=1, linestyle='-', zorder=1)

# MAIN CONTENT
main_y = sep1 - 2

# Title
ax.text(10, main_y, 'Forecast queue and graph archive', fontsize=11, fontweight='bold', zorder=2)

# Description
desc = 'This site lets a user log in, search for a ticker, queue a new training run,\nand reopen older saved graphs.'
ax.text(10, main_y - 5, desc, fontsize=7.5, color='#444', zorder=2)

# Quick Links section
section_y = main_y - 14
ax.add_patch(inner_box(7, section_y - 20, 96, 22))
ax.text(10, section_y, 'Quick Links', fontsize=9, fontweight='bold', zorder=2)
ax.text(10, section_y - 5, '[ Open Search Page ]    [ Log In ]    [ Create Account ]', fontsize=8, style='italic', color='#2563eb', zorder=2)

# Current Totals section
stats_y = section_y - 25
ax.add_patch(inner_box(7, stats_y - 18, 96, 20))
ax.text(10, stats_y, 'Current Totals', fontsize=9, fontweight='bold', zorder=2)
stat_boxes = ['Queued Jobs: __', 'Running Jobs: __', 'Saved Graphs: __', 'Tracked Tickers: __']
for i, stat in enumerate(stat_boxes):
    x_pos = 10 + (i % 2) * 46
    y_pos = stats_y - 5 - (i // 2) * 7
    ax.add_patch(patches.Rectangle((x_pos, y_pos - 3), 42, 7, linewidth=0.5, edgecolor='#bbb', facecolor='#fafafa', zorder=1))
    ax.text(x_pos + 2, y_pos - 0.5, stat, fontsize=7.5, zorder=2)

# How it Works section
how_y = stats_y - 22
ax.add_patch(inner_box(7, how_y - 28, 96, 30))
ax.text(10, how_y, 'How the site works', fontsize=9, fontweight='bold', zorder=2)
steps = [
    '1. Log in and search for a ticker.',
    '2. Open an older saved graph or request a new forecast.',
    '3. Request is written into the prediction_jobs queue table.',
    '4. PHP worker starts a single remote training run over SSH.',
    '5. Generated graph files are imported back and stored in DB.',
]
for i, step in enumerate(steps):
    ax.text(12, how_y - 5 - i * 4, step, fontsize=7, color='#444', zorder=2)

# Demo Account section
demo_y = how_y - 33
ax.add_patch(inner_box(7, demo_y - 10, 96, 12))
ax.text(10, demo_y, 'Demo Account', fontsize=9, fontweight='bold', zorder=2)
ax.text(12, demo_y - 4, 'Username: demo    |    Password: ********', fontsize=7.5, color='#888', zorder=2)

# Recent Jobs section
jobs_y = demo_y - 14
ax.add_patch(inner_box(7, jobs_y - 22, 96, 24))
ax.text(10, jobs_y, 'Recent Jobs', fontsize=9, fontweight='bold', zorder=2)
for i in range(4):
    ax.text(12, jobs_y - 4 - i * 4.5, f'[ AAPL - completed - 2025-12-15 14:30:00 ]', fontsize=7, color='#444', zorder=2)
    if i == 3:
        ax.text(12, jobs_y - 4 - (i+1) * 4.5, '[ ... ]', fontsize=7, color='#999', zorder=2)

# Recent Graphs section
graphs_y = jobs_y - 26
ax.add_patch(inner_box(7, graphs_y - 22, 96, 24))
ax.text(10, graphs_y, 'Recent Graphs', fontsize=9, fontweight='bold', zorder=2)
for i in range(3):
    ax.text(12, graphs_y - 4 - i * 4.5, f'[ TSLA - AAPL Forecast (2025-12-15 14:30) ]', fontsize=7, color='#444', zorder=2)

# FOOTER
footer_y = graphs_y - 28
ax.plot([7, 103], [footer_y, footer_y], color='#999', linewidth=1, linestyle='-', zorder=1)
ax.text(55, footer_y - 4, 'Built for Linux XAMPP deployment with PHP queueing, SSH-triggered training, and database-backed graph history.', fontsize=6, color='#888', ha='center', zorder=2)

# Wireframe label
ax.text(55, 139, 'WIREFRAME - Home Page (index.php)', fontsize=10, fontweight='bold', color='#1e3a8a', ha='center', zorder=3)

# Save with fixed bbox to ensure full border is visible
fig.savefig(r'C:\Users\paron\Desktop\Dev\CS_PROJECT\docs\diagrams\wireframe_home_v2.png', dpi=150, bbox_inches=None, pad_inches=0.5)
print('Wireframe saved.')
