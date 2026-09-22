import numpy as np
import pandas as pd
import pytest

from explainable_qml_dashboard.data import FEATURE_NAMES
from explainable_qml_dashboard.inference import MAX_UPLOAD_ROWS, validate_input_frame


def valid_frame(rows=2):
    return pd.DataFrame(np.ones((rows, 4)), columns=FEATURE_NAMES)


def test_valid_input():
    assert validate_input_frame(valid_frame()).shape == (2, 4)


def test_extra_columns_are_ignored():
    frame = valid_frame()
    frame["unused"] = "ok"
    assert list(validate_input_frame(frame).columns) == list(FEATURE_NAMES)


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="Missing"):
        validate_input_frame(valid_frame().drop(columns=[FEATURE_NAMES[0]]))


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="no rows"):
        validate_input_frame(valid_frame(0))


def test_large_input_rejected():
    with pytest.raises(ValueError, match="safety limit"):
        validate_input_frame(valid_frame(MAX_UPLOAD_ROWS + 1))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(value):
    frame = valid_frame()
    frame.iloc[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        validate_input_frame(frame)


def test_nonnumeric_rejected():
    frame = valid_frame().astype(object)
    frame.iloc[0, 0] = "flower"
    with pytest.raises(ValueError):
        validate_input_frame(frame)
