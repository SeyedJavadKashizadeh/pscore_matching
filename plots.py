import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mirrored_histogram(data, treat_col, score_col, bins, name, saving_path):
    treatment_group = data[data[treat_col] == 1][score_col]
    control_group = data[data[treat_col] == 0][score_col]

    # Histogram for Treatment Group
    plt.hist(treatment_group, bins=bins, density=True, edgecolor='black', color='green', alpha=0.6)

    # Histogram for Control Group (mirrored)
    heights, bin_edges = np.histogram(control_group, bins=bins, density=True)
    heights *= -1  # Reverse direction for mirroring
    bin_width = np.diff(bin_edges)[0]
    bin_pos = (bin_edges[:-1] + bin_width / 2)

    plt.bar(bin_pos, heights, width=bin_width, edgecolor='black', color='red', alpha=0.6)

    # Add horizontal line at y=0
    plt.axhline(0, color='blue', linewidth=1)

    # Labels
    plt.xlabel(score_col)
    plt.ylabel('Density')
    # plt.title(f'Mirrored Histogram of {score_col} by Treatment Group')

    # Customize y-axis ticks to show positive numbers on both sides
    yticks = plt.yticks()[0]
    plt.yticks(yticks, [f'{abs(y):.2f}' for y in yticks])

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = f"{saving_path}/pscore_{name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
