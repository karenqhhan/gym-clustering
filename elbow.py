"""Fit K-means on scaled training data and plot the elbow curve."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from prepare_data import load_scaled_features

OUTPUT_PATH = Path(__file__).resolve().parent / "elbow.png"
K_VALUES = range(1, 9)


def elbow_k(k_values, inertias):
    """Return the K farthest from the line between the first and last points."""
    x = np.asarray(k_values, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line = end - start
    distances = np.abs(line[0] * (start[1] - y_norm) - line[1] * (start[0] - x_norm))
    distances /= np.linalg.norm(line)
    return int(x[np.argmax(distances)])


def main():
    train_scaled, _ = load_scaled_features()

    inertias = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(train_scaled)
        inertias.append(model.inertia_)

    chosen_k = elbow_k(list(K_VALUES), inertias)

    print("Training inertia by K:")
    for k, inertia in zip(K_VALUES, inertias):
        print(f"K={k}: {inertia:.2f}")

    print(
        f"\nThe elbow appears at K={chosen_k}: "
        "inertia drops sharply up to that point, then flattens."
    )

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_VALUES), inertias, marker="o")
    plt.axvline(chosen_k, linestyle="--", color="gray", label=f"Elbow at K={chosen_k}")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Training inertia")
    plt.title("K-means elbow curve (training data)")
    plt.xticks(list(K_VALUES))
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Saved elbow curve to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
