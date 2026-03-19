import numpy as np
import matplotlib.pyplot as plt


def load_sector_data(prefix: str = "./data"):
    red_vals = np.load(f"{prefix}/red_sector_eigvals.npy")
    green_vals = np.load(f"{prefix}/green_sector_eigvals.npy")
    blue_vals = np.load(f"{prefix}/blue_sector_eigvals.npy")

    red_idx = np.load(f"{prefix}/red_sector_indices.npy")
    green_idx = np.load(f"{prefix}/green_sector_indices.npy")
    blue_idx = np.load(f"{prefix}/blue_sector_indices.npy")

    return (red_vals, green_vals, blue_vals, red_idx, green_idx, blue_idx)


def plot_pairwise_scatter(red_vals, green_vals, blue_vals, out_prefix: str = "./figures"):
    # Ensure output directory exists
    import os

    os.makedirs(out_prefix, exist_ok=True)

    plt.figure(figsize=(12, 4))

    # Red vs Blue
    plt.subplot(1, 3, 1)
    plt.scatter(red_vals, blue_vals, c="black", alpha=0.6, edgecolors="none")
    plt.xlabel("Red sector eigenvalues")
    plt.ylabel("Blue sector eigenvalues")
    plt.title("Red vs Blue")

    # Red vs Green
    plt.subplot(1, 3, 2)
    plt.scatter(red_vals, green_vals, c="black", alpha=0.6, edgecolors="none")
    plt.xlabel("Red sector eigenvalues")
    plt.ylabel("Green sector eigenvalues")
    plt.title("Red vs Green")

    # Green vs Blue
    plt.subplot(1, 3, 3)
    plt.scatter(green_vals, blue_vals, c="black", alpha=0.6, edgecolors="none")
    plt.xlabel("Green sector eigenvalues")
    plt.ylabel("Blue sector eigenvalues")
    plt.title("Green vs Blue")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}/sector_eigvals_pairwise_scatter.png", dpi=300)


def plot_sorted_sector_eigvals(
    red_vals, green_vals, blue_vals, red_idx, green_idx, blue_idx, out_prefix: str = "./figures"
):
    import os

    os.makedirs(out_prefix, exist_ok=True)

    # Sort full eigenvalue spectra for each sector (no absolute value),
    # so the ranking plot reflects the true sign structure.
    red_sector_vals = np.sort(red_vals)[::-1]
    green_sector_vals = np.sort(green_vals)[::-1]
    blue_sector_vals = np.sort(blue_vals)[::-1]

    plt.figure(figsize=(6, 4))

    plt.plot(
        np.arange(1, len(red_sector_vals) + 1),
        red_sector_vals,
        color="red",
        marker=".",
        linestyle="-",
        label="Red sector (D189)",
    )
    plt.plot(
        np.arange(1, len(green_sector_vals) + 1),
        green_sector_vals,
        color="green",
        marker=".",
        linestyle="-",
        label="Green sector (S195)",
    )
    plt.plot(
        np.arange(1, len(blue_sector_vals) + 1),
        blue_sector_vals,
        color="blue",
        marker=".",
        linestyle="-",
        label="Blue sector",
    )

    plt.xlabel("Rank (1 = largest eigenvalue)")
    plt.ylabel("Eigenvalue")
    plt.title("Sorted sector eigenvalues")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}/sector_eigvals_sorted.png", dpi=300)


if __name__ == "__main__":
    (
        red_vals,
        green_vals,
        blue_vals,
        red_idx,
        green_idx,
        blue_idx,
    ) = load_sector_data()

    plot_pairwise_scatter(red_vals, green_vals, blue_vals)
    plot_sorted_sector_eigvals(red_vals, green_vals, blue_vals, red_idx, green_idx, blue_idx)

