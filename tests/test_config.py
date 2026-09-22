import pytest

from explainable_qml_dashboard.config import ExperimentConfig


def test_default_parameter_count():
    assert ExperimentConfig().parameter_count == 16


def test_round_trip_dict():
    config = ExperimentConfig(random_seed=9, layers=3)
    assert ExperimentConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"qubits": 3}, "four qubits"),
        ({"layers": 0}, "layers"),
        ({"epochs": 0}, "epochs"),
        ({"patience": 0}, "epochs"),
        ({"test_fraction": 0.0}, "test_fraction"),
        ({"validation_fraction": 0.6}, "validation_fraction"),
        ({"test_fraction": 0.4, "validation_fraction": 0.4}, "training fraction"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"logit_scale": -1.0}, "logit_scale"),
        ({"saliency_step": 0.0}, "saliency_step"),
        ({"counterfactual_grid_points": 2}, "counterfactual"),
    ],
)
def test_invalid_config(values, message):
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**values)


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        ExperimentConfig.from_dict({"mystery": 1})
