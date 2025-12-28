#!/usr/bin/env python3
"""
Extract score progression from ML-Master log file and generate visualization.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_path):
    """Parse log file to extract step numbers and corresponding scores."""
    scores = []
    steps = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            # Match pattern: "[timestamp] INFO: - Score: ... metric = <score>"
            # The score may have ANSI color codes around it like [96m0.06495[0m
            # Need to handle \x1b[96m format (ESC character)
            match = re.search(r'INFO: - Score:.*?metric =.*?([0-9]+\.[0-9]+)', line)
            if match:
                score = float(match.group(1))
                steps.append(line_num)
                scores.append(score)

    return steps, scores


def plot_score_trend(steps, scores, output_path):
    """Plot score progression with medal zone background colors."""
    # Baseline values from log
    top_score = 0.051
    gold = 0.05589
    silver = 0.06229
    bronze = 0.06582
    median = 0.06988

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define zones with background colors (from top to bottom)
    # Using transparent colors for better visibility
    zones = [
        (0, gold, '#FFD700', 'Gold Zone'),      # Gold - yellow
        (gold, silver, '#C0C0C0', 'Silver Zone'),  # Silver - silver/gray
        (silver, bronze, '#CD7F32', 'Bronze Zone'),  # Bronze - bronze/copper
        (bronze, median, '#90EE90', 'Above Median'),  # Above median - light green
        (median, max(scores) + 0.005, '#FFCCCB', 'Below Median'),  # Below median - light red
    ]

    # Plot background zones
    for lower, upper, color, label in zones:
        ax.axhspan(lower, upper, alpha=0.2, color=color, label=label)

    # Plot the score line
    ax.plot(steps, scores, linewidth=2, color='#1f77b4', marker='o',
            markersize=3, markevery=max(1, len(steps) // 50), label='Score Progression')

    # Add horizontal reference lines
    ax.axhline(y=gold, color='#FFD700', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=silver, color='#808080', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=bronze, color='#CD7F32', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=median, color='green', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    ax.set_xlabel('Log Line Number', fontsize=12)
    ax.set_ylabel('Score (MAE - lower is better)', fontsize=12)
    ax.set_title('ML-Master Score Progression on nomad2018 Dataset\n'
                 f'Top: {top_score} | Gold: {gold} | Silver: {silver} | '
                 f'Bronze: {bronze} | Median: {median}',
                 fontsize=14, fontweight='bold')

    # Set y-axis limits
    ax.set_ylim(min(scores) - 0.002, max(scores) + 0.002)
    ax.invert_yaxis()  # Lower score is better, so invert

    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Add best and final score annotations
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_step = steps[best_idx]
    final_score = scores[-1]
    final_step = steps[-1]

    ax.annotate(f'Best: {best_score:.5f}',
                xy=(best_step, best_score),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=9, fontweight='bold')

    ax.annotate(f'Final: {final_score:.5f}',
                xy=(final_step, final_score),
                xytext=(-80, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                fontsize=9, fontweight='bold')

    # Tight layout
    plt.tight_layout()

    # Save as PDF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    # Paths
    log_path = "backup/original_ml_master/logs/run/nomad2018-predict-transparent-conductors_mcts_comp_validcheck_[cpu-0-23]/ml-master.log"
    output_path = "images/score_progression.pdf"

    # Parse log
    print(f"Parsing log file: {log_path}")
    steps, scores = parse_log_file(log_path)

    print(f"Found {len(scores)} score entries")
    print(f"Score range: {min(scores):.5f} - {max(scores):.5f}")
    print(f"Best score: {min(scores):.5f} at line {steps[np.argmin(scores)]}")
    print(f"Final score: {scores[-1]:.5f} at line {steps[-1]}")

    # Plot
    plot_score_trend(steps, scores, output_path)


if __name__ == "__main__":
    main()
