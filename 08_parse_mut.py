import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob

DEFAULT_MUTATION = "S189D"
INPUT_PATTERN = "logs/mutations_full_*.out"
OUTPUT_DIR = "figures"


# ---------- PARSING ----------
def parse_mutation_file(file_path):
    subalignments = []
    current_mutations = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("ALIGNMENT"):
                if current_mutations:
                    subalignments.append(build_ordered_mutations(current_mutations))
                    current_mutations = {}

            match = re.search(r"Mutation (\d+): ([A-Z0-9\-]+)", line)
            if match:
                idx = int(match.group(1))
                mutation = match.group(2)
                current_mutations[idx] = mutation

        if current_mutations:
            subalignments.append(build_ordered_mutations(current_mutations))

    return subalignments


def build_ordered_mutations(mutation_dict):
    ordered = [mutation_dict[k] for k in sorted(mutation_dict.keys())]
    return [DEFAULT_MUTATION] + ordered


# ---------- GRID ----------
def build_grid(subalignments):
    max_len = max(len(s) for s in subalignments)
    return np.array([s + [""] * (max_len - len(s)) for s in subalignments])


# ---------- COLORS ----------
def assign_colors(grid):
    unique_mutations = sorted(set(m for row in grid for m in row if m))
    cmap = plt.cm.get_cmap("tab20", len(unique_mutations))
    return {m: cmap(i) for i, m in enumerate(unique_mutations)}


# ---------- PLOT ----------
def plot_grid(grid, color_map, output_path):
    n_rows, n_cols = grid.shape

    fig, ax = plt.subplots(figsize=(n_cols * 1.2, n_rows * 0.35))

    xs, ys, colors = [], [], []
    labels = []

    for i in range(n_rows):
        for j in range(n_cols):
            mutation = grid[i, j]

            if mutation:
                xs.append(j + 0.25)   # left position
                ys.append(i + 0.5)
                colors.append(color_map[mutation])
                labels.append((j, i, mutation))

    # Draw points (true circles, size in points^2)
    ax.scatter(xs, ys, s=60, c=colors)

    # Add text
    for j, i, mutation in labels:
        ax.text(
            j + 0.45, i + 0.5,
            mutation,
            ha='left', va='center',
            fontsize=6
        )

    # Limits
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    # Center ticks
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)

    ax.set_xticklabels(range(1, n_cols + 1), fontsize=8)
    ax.set_yticklabels(range(1, n_rows + 1), fontsize=8)

    ax.set_xlabel("Mutation Order", fontsize=9)
    ax.set_ylabel("Subalignment", fontsize=9)

    # Clean look
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(length=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---------- MAIN ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(INPUT_PATTERN))

    for file_path in files:
        match = re.search(r"mutations_full_(\d+)\.out", file_path)
        step = match.group(1) if match else "unknown"

        output_path = os.path.join(
            OUTPUT_DIR,
            f"mutation_grid_step_{step}.png"
        )

        print(f"Processing: {file_path} → {output_path}")

        subalignments = parse_mutation_file(file_path)
        grid = build_grid(subalignments)
        color_map = assign_colors(grid)
        plot_grid(grid, color_map, output_path)


if __name__ == "__main__":
    main()
