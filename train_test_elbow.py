"""Compare training and test inertia for K-means fit on the training set only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from prepare_data import load_scaled_features

OUTPUT_PATH = Path(__file__).resolve().parent / "train_test_elbow.png"
K_VALUES = range(1, 9)


def test_inertia(model, test_scaled):
    """Total squared distance from test rows to the nearest training centroid."""
    labels = model.predict(test_scaled)
    nearest_centers = model.cluster_centers_[labels]
    distances = test_scaled.to_numpy() - nearest_centers
    return float(np.sum(distances**2))


def main():
    train_scaled, test_scaled = load_scaled_features()

    train_inertias = []
    test_inertias = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(train_scaled)
        train_inertias.append(model.inertia_)
        test_inertias.append(test_inertia(model, test_scaled))

    print("Inertia by K (models fit on training data only):")
    for k, train_value, test_value in zip(K_VALUES, train_inertias, test_inertias):
        print(f"K={k}: training={train_value:.2f}, test={test_value:.2f}")

    test_by_k = dict(zip(K_VALUES, test_inertias))
    print(f"\nTest inertia at K=1: {test_by_k[1]:.2f}")
    print(f"Test inertia at K=4: {test_by_k[4]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        (axes[0], train_inertias, "Training inertia", "Training elbow"),
        (axes[1], test_inertias, "Test inertia", "Test elbow"),
    )
    for axis, values, ylabel, title in panels:
        axis.plot(list(K_VALUES), values, marker="o")
        axis.axvline(4, linestyle="--", color="gray", label="K=4")
        axis.set_xlabel("Number of clusters (K)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xticks(list(K_VALUES))
        axis.legend()

    fig.suptitle("K-means inertia on training and held-out members")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved comparison to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
