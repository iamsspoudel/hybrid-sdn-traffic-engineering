import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class SDNLargeDatasetPreprocessor:
    

    SPLIT_NAMES = ("train", "val", "test")

    def __init__(
        self,
        input_file="dataset/raw/balanced_large_congestion_dataset.csv",
        output_dir="dataset/processed",
        window_size=20,
        gap_threshold=5.0,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        random_state=42,
        strict_hash_check=True,
    ):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)

        self.window_size = int(window_size)
        self.gap_threshold = float(gap_threshold)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.random_state = int(random_state)
        self.strict_hash_check = bool(strict_hash_check)

        self.scaler = MinMaxScaler()
        self.split_manifest = None
        self.flow_count_training_max = None
        self.raw_shape = None

        self._validate_configuration()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_configuration(self):
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1.")

        if self.gap_threshold < 0:
            raise ValueError("gap_threshold cannot be negative.")

        ratios = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(ratios, 1.0):
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0. "
                f"Received {ratios:.6f}."
            )

        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0:
            raise ValueError("All split ratios must be greater than zero.")

    @property
    def split_ratios(self):
        return {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }

    def load_dataset(self):
        print("[INFO] Loading dataset...")

        if not self.input_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.input_file}")

        df = pd.read_csv(self.input_file)
        self.raw_shape = tuple(df.shape)

        print("[INFO] Dataset shape:", df.shape)
        print("[INFO] Columns:", df.columns.tolist())

        required_columns = {"timestamp", "label"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        if df.columns.duplicated().any():
            duplicated = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names detected: {duplicated}")

        print("\n[INFO] Raw label distribution:")
        print(df["label"].value_counts(dropna=False).sort_index())

        return df

    def clean_dataset(self, df):
        print("\n[INFO] Cleaning dataset...")

        df = df.copy()
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] Removed duplicates: {before - len(df)}")

        df = df.replace([np.inf, -np.inf], np.nan)

        # Convert timestamp and label first because both are required for
        # chronological segmentation and class-aware splitting.
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["label"] = pd.to_numeric(df["label"], errors="coerce")

        invalid_required = df["timestamp"].isna() | df["label"].isna()
        invalid_count = int(invalid_required.sum())
        if invalid_count:
            warnings.warn(
                f"Dropping {invalid_count} rows with invalid timestamp or label."
            )
            df = df.loc[~invalid_required].copy()

        df["label"] = df["label"].astype(np.int64)

        # The project uses three congestion classes.
        expected_labels = {0, 1, 2}
        observed_labels = set(df["label"].unique().tolist())
        unexpected_labels = observed_labels.difference(expected_labels)
        if unexpected_labels:
            raise ValueError(
                f"Unexpected label values detected: {sorted(unexpected_labels)}"
            )

        feature_columns = [
            col for col in df.columns if col not in {"timestamp", "label"}
        ]

        if not feature_columns:
            raise ValueError("No traffic feature columns were found.")

        # Ensure all traffic features are numeric.
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values(["label", "timestamp"]).reset_index(drop=True)

        print("[INFO] Rows after required-field cleaning:", len(df))
        print(
            "[INFO] Missing traffic values before segment-wise imputation:",
            int(df[feature_columns].isna().sum().sum()),
        )

        return df

    def sort_and_segment_dataset(self, df):
        print("\n[INFO] Sorting and segmenting time-series data...")

        df = df.sort_values(["label", "timestamp"]).reset_index(drop=True)
        df["segment_id"] = np.int64(0)

        for label_value, group in df.groupby("label", sort=True):
            time_gap = group["timestamp"].diff().fillna(0.0)

            # A new segment begins when the chronological gap exceeds the
            # configured threshold.
            local_segment = (
                time_gap.gt(self.gap_threshold).cumsum().astype(np.int64)
            )

            df.loc[group.index, "segment_id"] = local_segment.to_numpy()

        df["segment_id"] = df["segment_id"].astype(np.int64)

        print("[INFO] Segments per label:")
        print(df.groupby("label")["segment_id"].nunique())

        return df

    def impute_features_within_segments(self, df):
        """
        Impute traffic features independently inside each complete segment.

        ``groupby.apply`` can remove the grouping columns (``label`` and
        ``segment_id``) in newer pandas versions.  This implementation uses
        ``groupby.transform`` so every original column and row index is
        preserved while the feature values are imputed segment by segment.
        """
        print("\n[INFO] Imputing traffic features within each segment...")

        df = (
            df.copy()
            .sort_values(["label", "segment_id", "timestamp"])
            .reset_index(drop=True)
        )

        required_columns = {"timestamp", "label", "segment_id"}
        missing_required = required_columns.difference(df.columns)
        if missing_required:
            raise ValueError(
                "Cannot perform segment-wise imputation because these "
                "columns are missing: " + ", ".join(sorted(missing_required))
            )

        feature_columns = [
            col for col in df.columns if col not in required_columns
        ]
        if not feature_columns:
            raise ValueError("No traffic feature columns are available to impute.")

        group_keys = [df["label"], df["segment_id"]]

        def impute_feature(series):
            return (
                series.interpolate(
                    method="linear",
                    limit_direction="both",
                )
                .ffill()
                .bfill()
                .fillna(0.0)
            )

        # transform returns an object aligned to the original dataframe index,
        # so timestamp, label and segment_id cannot disappear.
        df[feature_columns] = (
            df[feature_columns]
            .groupby(group_keys, sort=False)
            .transform(impute_feature)
        )

        remaining_missing = int(df[feature_columns].isna().sum().sum())
        feature_values = df[feature_columns].to_numpy(dtype=np.float64)
        remaining_nonfinite = int((~np.isfinite(feature_values)).sum())

        print("[INFO] Missing values after imputation:", remaining_missing)
        print("[INFO] Non-finite values after imputation:", remaining_nonfinite)

        if remaining_missing or remaining_nonfinite:
            raise ValueError(
                "Cleaning failed: missing or non-finite traffic values remain."
            )

        return df

    def build_segment_summary(self, df):
        summary = (
            df.groupby(["label", "segment_id"], as_index=False)
            .size()
            .rename(columns={"size": "row_count"})
        )

        summary["expected_sequences"] = (
            summary["row_count"] - self.window_size
        ).clip(lower=0).astype(np.int64)

        summary["usable_for_sequences"] = (
            summary["expected_sequences"] > 0
        )

        return summary

    def _weighted_split_for_one_label(self, label_summary, label_value):
        """
        Assign whole segments for one class to train/val/test.

        Segments are weighted by their expected number of sliding-window
        sequences, so the final sequence ratios stay close to 70/10/20 while
        no segment can cross partition boundaries.
        """
        rng = np.random.default_rng(self.random_state + int(label_value) * 1009)

        usable = label_summary[
            label_summary["expected_sequences"] > 0
        ].copy()

        short = label_summary[
            label_summary["expected_sequences"] == 0
        ].copy()

        if len(usable) < 3:
            raise ValueError(
                f"Label {label_value} has only {len(usable)} usable segments. "
                "At least three usable segments are required for leakage-safe "
                "train/validation/test splitting."
            )

        # Random tie-breaker, followed by largest-first assignment.
        usable["_tie_breaker"] = rng.random(len(usable))
        usable = usable.sort_values(
            ["expected_sequences", "_tie_breaker"],
            ascending=[False, True],
        ).reset_index(drop=True)

        total_weight = float(usable["expected_sequences"].sum())
        targets = {
            split: self.split_ratios[split] * total_weight
            for split in self.SPLIT_NAMES
        }
        assigned_weights = {split: 0.0 for split in self.SPLIT_NAMES}
        assignments = []

        for _, row in usable.iterrows():
            weight = float(row["expected_sequences"])

            candidate_costs = {}
            for split in self.SPLIT_NAMES:
                projected = assigned_weights.copy()
                projected[split] += weight

                # Absolute squared target error works well for weighted
                # partitioning because larger target sets should absorb more
                # of the largest segments.
                candidate_costs[split] = sum(
                    (projected[name] - targets[name]) ** 2
                    for name in self.SPLIT_NAMES
                )

            best_split = min(
                self.SPLIT_NAMES,
                key=lambda name: (
                    candidate_costs[name],
                    -(
                        targets[name] - assigned_weights[name]
                    ),
                    self.SPLIT_NAMES.index(name),
                ),
            )

            assignments.append(best_split)
            assigned_weights[best_split] += weight

        usable["split"] = assignments
        usable = usable.drop(columns=["_tie_breaker"])

        # Guarantee that every partition contains at least one usable segment.
        for missing_split in self.SPLIT_NAMES:
            if (usable["split"] == missing_split).any():
                continue

            donor_counts = usable["split"].value_counts()
            donor_candidates = [
                split
                for split in self.SPLIT_NAMES
                if donor_counts.get(split, 0) > 1
            ]

            if not donor_candidates:
                raise ValueError(
                    f"Could not place a usable label-{label_value} segment in "
                    f"the {missing_split} partition."
                )

            donor = max(
                donor_candidates,
                key=lambda name: assigned_weights[name] - targets[name],
            )

            donor_rows = usable[usable["split"] == donor]
            move_index = donor_rows["expected_sequences"].idxmin()
            moved_weight = float(
                usable.loc[move_index, "expected_sequences"]
            )

            usable.loc[move_index, "split"] = missing_split
            assigned_weights[donor] -= moved_weight
            assigned_weights[missing_split] += moved_weight

        # Zero-sequence segments do not affect model sample ratios. Assign them
        # reproducibly according to the requested split probabilities.
        if not short.empty:
            short = short.copy()
            short["split"] = rng.choice(
                self.SPLIT_NAMES,
                size=len(short),
                p=[
                    self.train_ratio,
                    self.val_ratio,
                    self.test_ratio,
                ],
            )

        combined = pd.concat([usable, short], ignore_index=True)
        return combined

    def split_segments(self, df):
        print(
            "\n[INFO] Splitting complete segments into "
            f"{self.train_ratio:.0%}/{self.val_ratio:.0%}/{self.test_ratio:.0%}..."
        )

        summary = self.build_segment_summary(df)
        assigned_parts = []

        for label_value, label_summary in summary.groupby(
            "label", sort=True
        ):
            assigned_parts.append(
                self._weighted_split_for_one_label(
                    label_summary.copy(),
                    int(label_value),
                )
            )

        manifest = (
            pd.concat(assigned_parts, ignore_index=True)
            .sort_values(["label", "segment_id"])
            .reset_index(drop=True)
        )

        if manifest["split"].isna().any():
            raise ValueError("Some segments were not assigned to a split.")

        # Merge the split assignment back to every raw record.
        assigned_df = df.merge(
            manifest[["label", "segment_id", "split"]],
            on=["label", "segment_id"],
            how="left",
            validate="many_to_one",
        )

        if assigned_df["split"].isna().any():
            raise ValueError("Some dataset rows have no split assignment.")

        train_df = assigned_df[assigned_df["split"] == "train"].drop(
            columns=["split"]
        )
        val_df = assigned_df[assigned_df["split"] == "val"].drop(
            columns=["split"]
        )
        test_df = assigned_df[assigned_df["split"] == "test"].drop(
            columns=["split"]
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        self._verify_segment_isolation(train_df, val_df, test_df)

        print("\n[INFO] Segment split summary:")
        split_summary = (
            manifest.groupby(["split", "label"])
            .agg(
                segments=("segment_id", "size"),
                usable_segments=("usable_for_sequences", "sum"),
                rows=("row_count", "sum"),
                expected_sequences=("expected_sequences", "sum"),
            )
            .reset_index()
        )
        print(split_summary.to_string(index=False))

        print("\n[INFO] Expected sequence totals by split:")
        print(
            manifest.groupby("split")["expected_sequences"]
            .sum()
            .reindex(self.SPLIT_NAMES)
        )

        self.split_manifest = manifest

        return train_df, val_df, test_df

    @staticmethod
    def _segment_key_set(df):
        return set(
            map(
                tuple,
                df[["label", "segment_id"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )

    def _verify_segment_isolation(self, train_df, val_df, test_df):
        train_keys = self._segment_key_set(train_df)
        val_keys = self._segment_key_set(val_df)
        test_keys = self._segment_key_set(test_df)

        train_val = train_keys.intersection(val_keys)
        train_test = train_keys.intersection(test_keys)
        val_test = val_keys.intersection(test_keys)

        print("\n[INFO] Segment leakage check:")
        print("[INFO] Train-Val shared segments :", len(train_val))
        print("[INFO] Train-Test shared segments:", len(train_test))
        print("[INFO] Val-Test shared segments  :", len(val_test))

        if train_val or train_test or val_test:
            raise ValueError(
                "Segment leakage detected between dataset partitions."
            )

    def add_engineered_features(
        self,
        df,
        flow_count_scale,
        partition_name,
    ):
        print(
            f"\n[INFO] Adding engineered features to {partition_name} split..."
        )

        df = df.copy()
        group_columns = ["label", "segment_id"]

        if "throughput" in df.columns:
            df["throughput_change_rate"] = (
                df.groupby(group_columns, sort=False)["throughput"]
                .diff()
                .fillna(0.0)
            )
        else:
            df["throughput_change_rate"] = 0.0

        if "utilization" in df.columns:
            df["utilization_gradient"] = (
                df.groupby(group_columns, sort=False)["utilization"]
                .diff()
                .fillna(0.0)
            )
        else:
            df["utilization_gradient"] = 0.0

        if "flow_count" in df.columns and flow_count_scale > 0:
            df["flow_density"] = (
                df["flow_count"].astype(np.float64) / flow_count_scale
            )
        else:
            df["flow_density"] = 0.0

        engineered = [
            "throughput_change_rate",
            "utilization_gradient",
            "flow_density",
        ]

        if not np.isfinite(
            df[engineered].to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                f"Non-finite engineered feature detected in {partition_name}."
            )

        return df

    def create_sequences(self, df, partition_name):
        print(
            f"\n[INFO] Creating sliding-window sequences for "
            f"{partition_name}..."
        )

        feature_columns = [
            col
            for col in df.columns
            if col not in {"timestamp", "label", "segment_id"}
        ]

        print("[INFO] Number of input features:", len(feature_columns))
        print("[INFO] Feature columns:")
        print(feature_columns)

        X_all = []
        y_all = []
        skipped_segments = 0

        for (label_value, segment_id), group in df.groupby(
            ["label", "segment_id"],
            sort=True,
        ):
            group = group.sort_values("timestamp")

            if len(group) <= self.window_size:
                skipped_segments += 1
                continue

            features = group[feature_columns].to_numpy(
                dtype=np.float32,
                copy=True,
            )

            if not np.isfinite(features).all():
                raise ValueError(
                    "Non-finite values found before sequence generation in "
                    f"{partition_name}, label={label_value}, "
                    f"segment_id={segment_id}."
                )

            # The sequence uses positions [i, i + window_size - 1].
            # Its target is the congestion label at position
            # i + window_size.
            number_of_sequences = len(features) - self.window_size

            for i in range(number_of_sequences):
                X_all.append(features[i : i + self.window_size])
                y_all.append(int(label_value))

        if not X_all:
            raise ValueError(
                f"No sequences were generated for the {partition_name} split."
            )

        X = np.asarray(X_all, dtype=np.float32)
        y = np.asarray(y_all, dtype=np.int64)

        print("[INFO] Skipped short segments:", skipped_segments)
        print("[INFO] X shape:", X.shape)
        print("[INFO] y shape:", y.shape)
        print("[INFO] Sequence label distribution:")
        print(pd.Series(y).value_counts().sort_index())

        return X, y, feature_columns

    def normalize(self, X_train, X_val, X_test):
        print("\n[INFO] Applying MinMax normalization...")
        print("[INFO] Scaler fit source: training sequences only")

        if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
            raise ValueError("All X arrays must have shape (N, T, F).")

        n_features = X_train.shape[-1]

        if X_val.shape[-1] != n_features or X_test.shape[-1] != n_features:
            raise ValueError(
                "Train, validation, and test feature dimensions do not match."
            )

        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)

        self.scaler.fit(X_train_2d)

        X_train_scaled = self.scaler.transform(X_train_2d)
        X_val_scaled = self.scaler.transform(X_val_2d)
        X_test_scaled = self.scaler.transform(X_test_2d)

        X_train_scaled = X_train_scaled.reshape(
            X_train.shape
        ).astype(np.float32)

        X_val_scaled = X_val_scaled.reshape(
            X_val.shape
        ).astype(np.float32)

        X_test_scaled = X_test_scaled.reshape(
            X_test.shape
        ).astype(np.float32)

        for name, array in (
            ("X_train", X_train_scaled),
            ("X_val", X_val_scaled),
            ("X_test", X_test_scaled),
        ):
            if not np.isfinite(array).all():
                raise ValueError(
                    f"Non-finite values found after normalization in {name}."
                )

        return X_train_scaled, X_val_scaled, X_test_scaled

    @staticmethod
    def _sequence_hashes(X):
        hashes = set()

        for sequence in X:
            contiguous = np.ascontiguousarray(sequence)
            digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
            hashes.add(digest)

        return hashes

    def verify_exact_sequence_overlap(self, X_train, X_val, X_test):
        print("\n[INFO] Exact sequence hash verification...")

        train_hashes = self._sequence_hashes(X_train)
        val_hashes = self._sequence_hashes(X_val)
        test_hashes = self._sequence_hashes(X_test)

        overlap = {
            "train_val_overlap": len(
                train_hashes.intersection(val_hashes)
            ),
            "train_test_overlap": len(
                train_hashes.intersection(test_hashes)
            ),
            "val_test_overlap": len(
                val_hashes.intersection(test_hashes)
            ),
            "train_unique_hashes": len(train_hashes),
            "val_unique_hashes": len(val_hashes),
            "test_unique_hashes": len(test_hashes),
            "train_internal_duplicates": int(
                len(X_train) - len(train_hashes)
            ),
            "val_internal_duplicates": int(
                len(X_val) - len(val_hashes)
            ),
            "test_internal_duplicates": int(
                len(X_test) - len(test_hashes)
            ),
        }

        print("[INFO] Train-Val overlap :", overlap["train_val_overlap"])
        print("[INFO] Train-Test overlap:", overlap["train_test_overlap"])
        print("[INFO] Val-Test overlap  :", overlap["val_test_overlap"])

        cross_split_overlap = (
            overlap["train_val_overlap"]
            + overlap["train_test_overlap"]
            + overlap["val_test_overlap"]
        )

        if cross_split_overlap:
            message = (
                "Exact duplicate sequences were detected across partitions. "
                f"Details: {overlap}"
            )
            if self.strict_hash_check:
                raise ValueError(message)
            warnings.warn(message)

        return overlap

    @staticmethod
    def _class_count_dict(y):
        labels, counts = np.unique(y, return_counts=True)
        return {
            str(int(label)): int(count)
            for label, count in zip(labels, counts)
        }

    def save_processed_data(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_columns,
        leakage_results,
    ):
        print("\n[INFO] Saving processed files...")

        np.save(self.output_dir / "X_train.npy", X_train)
        np.save(self.output_dir / "X_val.npy", X_val)
        np.save(self.output_dir / "X_test.npy", X_test)

        np.save(self.output_dir / "y_train.npy", y_train)
        np.save(self.output_dir / "y_val.npy", y_val)
        np.save(self.output_dir / "y_test.npy", y_test)

        joblib.dump(self.scaler, self.output_dir / "scaler.pkl")

        with open(
            self.output_dir / "feature_columns.txt",
            "w",
            encoding="utf-8",
        ) as file:
            for column in feature_columns:
                file.write(column + "\n")

        if self.split_manifest is None:
            raise RuntimeError("Split manifest is unavailable.")

        self.split_manifest.to_csv(
            self.output_dir / "split_manifest.csv",
            index=False,
        )

        with open(
            self.output_dir / "leakage_check.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(leakage_results, file, indent=4)

        total_sequences = int(
            len(X_train) + len(X_val) + len(X_test)
        )

        metadata = {
            "input_file": str(self.input_file),
            "raw_dataset_shape": (
                list(self.raw_shape) if self.raw_shape is not None else None
            ),
            "window_size": self.window_size,
            "gap_threshold": self.gap_threshold,
            "random_state": self.random_state,
            "split_method": (
                "weighted complete-segment split before sequence generation"
            ),
            "requested_split_ratios": {
                "train": self.train_ratio,
                "validation": self.val_ratio,
                "test": self.test_ratio,
            },
            "actual_sequence_ratios": {
                "train": len(X_train) / total_sequences,
                "validation": len(X_val) / total_sequences,
                "test": len(X_test) / total_sequences,
            },
            "normalization": "MinMaxScaler fitted on training sequences only",
            "flow_density_scale_source": "maximum flow_count in training split",
            "flow_count_training_max": self.flow_count_training_max,
            "input_features": len(feature_columns),
            "feature_columns": feature_columns,
            "num_classes": 3,
            "total_sequences": total_sequences,
            "X_train_shape": list(X_train.shape),
            "X_val_shape": list(X_val.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_val_shape": list(y_val.shape),
            "y_test_shape": list(y_test.shape),
            "class_distribution": {
                "train": self._class_count_dict(y_train),
                "validation": self._class_count_dict(y_val),
                "test": self._class_count_dict(y_test),
            },
            "leakage_verification": leakage_results,
            "class_labels": {
                "0": "Normal Traffic",
                "1": "Moderate Congestion",
                "2": "Heavy Congestion",
            },
        }

        with open(
            self.output_dir / "preprocessing_metadata.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=4)

        print("[INFO] Saved:")
        print("       X_train.npy, X_val.npy, X_test.npy")
        print("       y_train.npy, y_val.npy, y_test.npy")
        print("       scaler.pkl")
        print("       feature_columns.txt")
        print("       split_manifest.csv")
        print("       leakage_check.json")
        print("       preprocessing_metadata.json")

    def run(self):
        df = self.load_dataset()
        df = self.clean_dataset(df)
        df = self.sort_and_segment_dataset(df)
        df = self.impute_features_within_segments(df)

        # Critical leakage-prevention step:
        # split complete segments before any sliding-window sequence is created.
        train_df, val_df, test_df = self.split_segments(df)

        # Fit the scale used by flow_density on the training split only.
        if "flow_count" in train_df.columns:
            max_flow = float(train_df["flow_count"].max())
            if not np.isfinite(max_flow) or max_flow <= 0:
                max_flow = 1.0
        else:
            max_flow = 1.0

        self.flow_count_training_max = max_flow

        train_df = self.add_engineered_features(
            train_df,
            flow_count_scale=max_flow,
            partition_name="training",
        )
        val_df = self.add_engineered_features(
            val_df,
            flow_count_scale=max_flow,
            partition_name="validation",
        )
        test_df = self.add_engineered_features(
            test_df,
            flow_count_scale=max_flow,
            partition_name="test",
        )

        X_train, y_train, feature_columns = self.create_sequences(
            train_df,
            "training",
        )
        X_val, y_val, val_feature_columns = self.create_sequences(
            val_df,
            "validation",
        )
        X_test, y_test, test_feature_columns = self.create_sequences(
            test_df,
            "test",
        )

        if feature_columns != val_feature_columns:
            raise ValueError(
                "Training and validation feature columns do not match."
            )

        if feature_columns != test_feature_columns:
            raise ValueError(
                "Training and test feature columns do not match."
            )

        total_sequences = len(X_train) + len(X_val) + len(X_test)

        print("\n========== LEAKAGE-SAFE DATASET SUMMARY ==========")
        print("[INFO] Train:", X_train.shape, y_train.shape)
        print("[INFO] Val  :", X_val.shape, y_val.shape)
        print("[INFO] Test :", X_test.shape, y_test.shape)
        print("[INFO] Total sequences:", total_sequences)
        print(
            "[INFO] Actual sequence ratios:",
            {
                "train": round(len(X_train) / total_sequences, 6),
                "val": round(len(X_val) / total_sequences, 6),
                "test": round(len(X_test) / total_sequences, 6),
            },
        )

        X_train, X_val, X_test = self.normalize(
            X_train,
            X_val,
            X_test,
        )

        leakage_results = self.verify_exact_sequence_overlap(
            X_train,
            X_val,
            X_test,
        )

        self.save_processed_data(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            feature_columns,
            leakage_results,
        )

        print("\n[INFO] Preprocessing completed successfully!")
        print(
            "[INFO] The saved dataset is split by complete time-series "
            "segments before sequence generation."
        )


if __name__ == "__main__":
    processor = SDNLargeDatasetPreprocessor(
        input_file="dataset/raw/balanced_large_congestion_dataset.csv",
        output_dir="dataset/processed",
        window_size=20,
        gap_threshold=5,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        random_state=42,
        strict_hash_check=False,
    )

    processor.run()