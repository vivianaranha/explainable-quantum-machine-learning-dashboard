"""Leakage-safe dataset loading and deterministic splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

FEATURE_NAMES = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)
TARGET_NAMES = ("setosa", "versicolor", "virginica")


@dataclass
class AnglePreprocessor:
    """Train-only standardization followed by an angle map to [0, pi]."""

    standardizer: StandardScaler
    angle_scaler: MinMaxScaler
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit(cls, x_train: np.ndarray) -> AnglePreprocessor:
        standardizer = StandardScaler().fit(x_train)
        standardized = standardizer.transform(x_train)
        angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi), clip=True).fit(standardized)
        return cls(standardizer=standardizer, angle_scaler=angle_scaler)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected a 2D array with {len(self.feature_names)} columns.")
        return self.angle_scaler.transform(self.standardizer.transform(values))

    def inverse_transform(self, angles: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, dtype=float)
        return self.standardizer.inverse_transform(self.angle_scaler.inverse_transform(angles))


@dataclass(frozen=True)
class DatasetBundle:
    x_raw: np.ndarray
    y: np.ndarray
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class SplitBundle:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    def as_frame(self, y: np.ndarray) -> pd.DataFrame:
        rows: list[dict[str, int | str]] = []
        for split_name, indices in (
            ("train", self.train_indices),
            ("validation", self.validation_indices),
            ("test", self.test_indices),
        ):
            rows.extend(
                {"row_id": int(index), "split": split_name, "target": int(y[index])}
                for index in indices
            )
        return pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)


def load_dataset() -> DatasetBundle:
    """Load the versioned scikit-learn Iris dataset without network access."""

    bunch = load_iris(as_frame=True)
    frame = bunch.frame.copy()
    x_raw = frame.loc[:, list(bunch.feature_names)].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=int)
    return DatasetBundle(
        x_raw=x_raw,
        y=y,
        frame=frame,
        feature_names=tuple(bunch.feature_names),
        target_names=tuple(bunch.target_names),
    )


def make_splits(
    y: np.ndarray,
    *,
    test_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> SplitBundle:
    """Create deterministic 60/20/20-style stratified partitions."""

    all_indices = np.arange(len(y))
    train_val, test = train_test_split(
        all_indices,
        test_size=test_fraction,
        stratify=y,
        random_state=random_seed,
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        stratify=y[train_val],
        random_state=random_seed,
    )
    return SplitBundle(
        train_indices=np.sort(train),
        validation_indices=np.sort(validation),
        test_indices=np.sort(test),
    )


def dataset_summary(dataset: DatasetBundle, splits: SplitBundle) -> dict[str, object]:
    """Build audit metadata for the bundled dataset and frozen split."""

    counts = np.bincount(dataset.y, minlength=len(dataset.target_names))
    return {
        "name": "scikit-learn Iris",
        "rows": int(len(dataset.y)),
        "features": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "class_counts": {
            name: int(count) for name, count in zip(dataset.target_names, counts, strict=True)
        },
        "split_counts": {
            "train": int(len(splits.train_indices)),
            "validation": int(len(splits.validation_indices)),
            "test": int(len(splits.test_indices)),
        },
        "data_origin": "Fisher's Iris dataset bundled with scikit-learn",
        "network_required": False,
    }
