"""Split gym member data and scale clustering features on the training set."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "gym_members.csv"
FEATURE_COLUMNS = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]


def load_member_splits():
    """Return original and scaled features for the shared 70/30 split.

    The scaler is fit on the training rows only. Indexes match the source CSV
    so cluster labels can be joined back to the original feature values.
    """
    members = pd.read_csv(DATA_PATH)
    features = members[FEATURE_COLUMNS]
    train_features, test_features = train_test_split(
        features, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        columns=FEATURE_COLUMNS,
        index=train_features.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_features),
        columns=FEATURE_COLUMNS,
        index=test_features.index,
    )
    return train_features, test_features, train_scaled, test_scaled, scaler


def load_scaled_features():
    """Return training and test features scaled from the training set only."""
    _, _, train_scaled, test_scaled = load_member_splits()
    return (
        train_scaled.reset_index(drop=True),
        test_scaled.reset_index(drop=True),
    )


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    train_scaled, test_scaled = load_scaled_features()

    print("Feature names:")
    print(FEATURE_COLUMNS)

    print("\nFirst five scaled training rows:")
    print(train_scaled.head())

    print(f"\nScaled training rows: {len(train_scaled)}")
    print(f"Scaled testing rows: {len(test_scaled)}")


if __name__ == "__main__":
    main()
