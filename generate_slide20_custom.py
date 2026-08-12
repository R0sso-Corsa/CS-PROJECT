import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = r'C:\Users\paron\Desktop\Dev\CS_PROJECT\docs\diagrams'
os.makedirs(OUT, exist_ok=True)

def box(ax, x, y, w, h, title, lines=None, face='#fff', edge='#444'):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=6', linewidth=0.9, edgecolor=edge, facecolor=face)
    ax.add_patch(b)
    ax.text(x + 6, y + h - 12, title, fontsize=9.5, fontweight='bold', color='#0f172a')
    if lines:
        ly = y + h - 26
        for L in lines:
            ax.text(x + 8, ly, L, fontsize=8.2, color='#1f2937')
            ly -= 12
    return b

def draw():
    fig = plt.figure(figsize=(13,7))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,130); ax.set_ylim(0,80)
    ax.axis('off')
    ax.set_facecolor('#ffffff')

    # Title
    ax.text(6, 74, 'Design: Detailed algorithms', fontsize=30, color='#0b1220')
    ax.text(6, 66, 'Marksheet requirement: Identified and justified the key variables / data structures / classes (explain validation).', fontsize=10.5, color='#111827')

    # Left bullets (concise)
    bullets = [
        'List the key (important) variables you will need in code',
        'You can add missing variables later',
        'Give details of data structures: ERD, data dictionary, example records',
        'Validation: how you will check user-entered data',
    ]
    by = 52
    for b in bullets:
        ax.text(10, by, '• ' + b, fontsize=11, color='#1f2937')
        by -= 9

    # Right: ERD boxes
    # student
    box(ax, 78, 52, 42, 16, 'STUDENT', lines=['*student_id', 'student_name', 'student_address'], face='#f8fafc', edge='#0ea5a4')
    # course
    box(ax, 98, 34, 42, 16, 'COURSE', lines=['*course_name', '*course_number'], face='#f8fafc', edge='#3b82f6')
    # instructor
    box(ax, 118, 56, 34, 14, 'INSTRUCTOR', lines=['*instructor_no', 'instructor_name'], face='#fff7ed', edge='#d97706')
    # seat
    box(ax, 82, 34, 34, 12, 'SEAT', lines=['*seat_no', 'seat_position'], face='#fff7ed', edge='#94a3b8')
    # class
    box(ax, 96, 14, 44, 16, 'CLASS', lines=['*course_name', '*section_number', 'num_registered'], face='#f8fafc', edge='#7c3aed')
    # section
    box(ax, 88, 6, 36, 10, 'SECTION', lines=['*section_number'], face='#ecfeff', edge='#06b6d4')
    # professor
    box(ax, 120, 8, 32, 12, 'PROFESSOR', lines=['*professor_id', 'professor_name'], face='#ecfeff', edge='#06b6d4')

    # Arrows (relationships)
    arr = FancyArrowPatch((100, 52), (118, 60), arrowstyle='->', mutation_scale=12, linewidth=1.0, color='#374151')
    ax.add_patch(arr)
    arr2 = FancyArrowPatch((110, 34), (102, 44), arrowstyle='->', mutation_scale=12, linewidth=1.0, color='#374151')
    ax.add_patch(arr2)
    arr3 = FancyArrowPatch((118, 20), (126, 14), arrowstyle='->', mutation_scale=12, linewidth=1.0, color='#374151')
    ax.add_patch(arr3)

    # small ERD note
    ax.text(78, 28, 'ERD: relationships shown (simplified)', fontsize=9, color='#475569')

    out = os.path.join(OUT, 'slide20_custom.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print('Generated:', out)

if __name__ == '__main__':
    draw()
