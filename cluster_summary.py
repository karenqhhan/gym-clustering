"""Fit K=4 on scaled training members and summarize every member in original units."""

import pandas as pd
from sklearn.cluster import KMeans

from prepare_data import FEATURE_COLUMNS, load_member_splits


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", lambda value: f"{value:.1f}")

    train_features, test_features, train_scaled, test_scaled, _ = load_member_splits()

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    model.fit(train_scaled)

    train_labeled = train_features.copy()
    test_labeled = test_features.copy()
    train_labeled["cluster"] = model.labels_
    test_labeled["cluster"] = model.predict(test_scaled)

    members = pd.concat([train_labeled, test_labeled], axis=0)
    counts = members["cluster"].value_counts().sort_index()
    averages = members.groupby("cluster")[FEATURE_COLUMNS].mean().sort_index()

    print("Members in each cluster (training and test combined):")
    for cluster, count in counts.items():
        print(f"Cluster {cluster}: {count}")

    print("\nAverage original feature values:")
    print(averages)

    print(f"\nMembers summarized: {len(members)}")


if __name__ == "__main__":
    main()
