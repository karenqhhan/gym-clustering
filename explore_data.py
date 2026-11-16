"""Explore gym member data and split it into training and testing sets."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "gym_members.csv"

members = pd.read_csv(DATA_PATH)

print(f"Rows: {members.shape[0]}")
print(f"Columns: {members.shape[1]}")

print("\nFirst five rows:")
print(members.head())

print("\nMissing values:")
print(members.isnull().sum())

train_members, test_members = train_test_split(
    members, test_size=0.3, random_state=42
)

print(f"\nTraining members: {len(train_members)}")
print(f"Testing members: {len(test_members)}")
