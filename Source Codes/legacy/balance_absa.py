"""
Oversample underrepresented classes in the ABSA labeled dataset.
TNL6323 — Malaysian eWallet Sentiment Analysis project.
"""

import uuid
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "sentiment_data_absa_labeled.csv"
TARGET = 120
RANDOM_STATE = 42

OVERSAMPLE_TARGETS = [
    ("touchngo", "positive", 15),
    ("grabpay", "neutral", 12),
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    print("\nBefore oversampling:")
    print(df.groupby(["source", "sentiment"]).size().to_string())

    extra_rows = []
    for source, sentiment, n_needed in OVERSAMPLE_TARGETS:
        subset = df[(df["source"] == source) & (df["sentiment"] == sentiment)]
        available = len(subset)
        shortfall = TARGET - available

        if shortfall <= 0:
            print(f"\n{source}/{sentiment}: already at {available} (no oversampling needed)")
            continue

        n_sample = min(n_needed, shortfall)
        duplicated = subset.sample(n=n_sample, replace=True, random_state=RANDOM_STATE).copy()
        duplicated["id"] = [str(uuid.uuid4()) for _ in range(len(duplicated))]
        extra_rows.append(duplicated)
        print(f"\n{source}/{sentiment}: {available} -> adding {n_sample} duplicates")

    if extra_rows:
        df = pd.concat([df, *extra_rows], ignore_index=True)

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    try:
        df.to_csv(DATA_PATH, index=False, encoding="utf-8")
        print(f"\nSaved {len(df)} rows to {DATA_PATH}")
    except PermissionError:
        fallback = DATA_PATH.with_name("sentiment_data_absa_labeled_fixed.csv")
        df.to_csv(fallback, index=False, encoding="utf-8")
        print(f"\nWARNING: Could not overwrite {DATA_PATH} (file may be open).")
        print(f"Saved {len(df)} rows to {fallback}")
        print("Close the CSV in Excel/editor, then re-run this script.")
    print("\nFinal counts:")
    counts = df.groupby(["source", "sentiment"]).size().unstack(fill_value=0)
    print(counts.to_string())

    all_120 = (counts == TARGET).all().all()
    print(f"\nAll classes at {TARGET}: {'YES' if all_120 else 'NO'}")


if __name__ == "__main__":
    main()
