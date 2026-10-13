#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


BASE_DIR = Path.home() / "sdn_dataset"
TS_DIR = BASE_DIR / "timeseries"
FINAL_DIR = BASE_DIR / "final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = FINAL_DIR / "balanced_large_congestion_dataset.csv"


def main():

    files = sorted(TS_DIR.glob("*_timeseries.csv"))

    print(f"[INFO] Found {len(files)} time-series files.")

    dfs = []

    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    merged = pd.concat(
        dfs,
        ignore_index=True
    )

    print("[INFO] Merged shape:", merged.shape)

    # Remove extremely tiny remaining rows
    merged = merged[
        (
            (merged["packet_count"] > 1) |
            (merged["byte_count"] > 64) |
            (merged["throughput"] > 0.001)
        )
    ].copy()

    print("[INFO] After final filtering:", merged.shape)

    print("\n[INFO] Label distribution before balancing:")
    print(merged["label"].value_counts().sort_index())

    min_count = merged["label"].value_counts().min()

    balanced_parts = []

    for label in sorted(merged["label"].unique()):

        class_df = merged[
            merged["label"] == label
        ].copy()

        sampled = class_df.sample(
            n=min_count,
            random_state=42
        )

        balanced_parts.append(sampled)

    balanced = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    balanced = balanced.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("\n[INFO] Label distribution after balancing:")
    print(balanced["label"].value_counts().sort_index())

    balanced.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n[INFO] Saved:", OUTPUT_FILE)
    print("[INFO] Final shape:", balanced.shape)

    important = [
        "packet_count",
        "byte_count",
        "throughput",
        "packet_rate",
        "utilization",
        "delay",
        "packet_loss"
    ]

    print("\n[INFO] Mean features by label:")
    print(
        balanced.groupby("label")[important].mean()
    )


if __name__ == "__main__":
    main()
