"""Shared deterministic test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from explainable_qml_dashboard.data import AnglePreprocessor, load_dataset, make_splits
from explainable_qml_dashboard.qvc import VariationalQuantumClassifier


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def splits(dataset):
    return make_splits(dataset.y, test_fraction=0.2, validation_fraction=0.2, random_seed=42)


@pytest.fixture(scope="session")
def prepared(dataset, splits):
    preprocessor = AnglePreprocessor.fit(dataset.x_raw[splits.train_indices])
    return preprocessor, preprocessor.transform(dataset.x_raw)


@pytest.fixture()
def qvc():
    return VariationalQuantumClassifier(layers=2, logit_scale=2.0, random_seed=42)


@pytest.fixture()
def random_angles():
    return np.random.default_rng(7).uniform(0.0, np.pi, size=(6, 4))
