import numpy as np
import pytest

from explainable_qml_dashboard.data import (
    FEATURE_NAMES,
    TARGET_NAMES,
    AnglePreprocessor,
    dataset_summary,
    make_splits,
)


def test_dataset_shape(dataset):
    assert dataset.x_raw.shape == (150, 4)
    assert dataset.y.shape == (150,)


def test_dataset_names(dataset):
    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.target_names == TARGET_NAMES


def test_class_balance(dataset):
    assert np.bincount(dataset.y).tolist() == [50, 50, 50]


def test_split_sizes(splits):
    assert len(splits.train_indices) == 90
    assert len(splits.validation_indices) == 30
    assert len(splits.test_indices) == 30


def test_split_disjoint_and_complete(splits):
    groups = [set(splits.train_indices), set(splits.validation_indices), set(splits.test_indices)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set.union(*groups) == set(range(150))


def test_split_stratified(dataset, splits):
    for indices, expected in (
        (splits.train_indices, [30, 30, 30]),
        (splits.validation_indices, [10, 10, 10]),
        (splits.test_indices, [10, 10, 10]),
    ):
        assert np.bincount(dataset.y[indices]).tolist() == expected


def test_split_is_deterministic(dataset):
    left = make_splits(dataset.y, test_fraction=0.2, validation_fraction=0.2, random_seed=17)
    right = make_splits(dataset.y, test_fraction=0.2, validation_fraction=0.2, random_seed=17)
    np.testing.assert_array_equal(left.train_indices, right.train_indices)


def test_preprocessor_angle_range(prepared):
    _, angles = prepared
    assert np.min(angles) >= 0.0
    assert np.max(angles) <= np.pi


def test_preprocessor_inverse_train(dataset, splits, prepared):
    preprocessor, angles = prepared
    recovered = preprocessor.inverse_transform(angles[splits.train_indices])
    np.testing.assert_allclose(recovered, dataset.x_raw[splits.train_indices], atol=1e-10)


def test_preprocessor_clips_outliers(dataset, splits):
    preprocessor = AnglePreprocessor.fit(dataset.x_raw[splits.train_indices])
    extreme = np.array([[-1e6, 1e6, -1e6, 1e6]])
    angles = preprocessor.transform(extreme)
    assert set(angles.ravel()).issubset({0.0, np.pi})


def test_preprocessor_rejects_wrong_shape(dataset, splits):
    preprocessor = AnglePreprocessor.fit(dataset.x_raw[splits.train_indices])
    with pytest.raises(ValueError, match="2D array"):
        preprocessor.transform(np.zeros((2, 3)))


def test_dataset_summary(dataset, splits):
    summary = dataset_summary(dataset, splits)
    assert summary["rows"] == 150
    assert summary["network_required"] is False
    assert summary["split_counts"] == {"train": 90, "validation": 30, "test": 30}
