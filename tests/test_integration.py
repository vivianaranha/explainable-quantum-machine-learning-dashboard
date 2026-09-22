import json

from explainable_qml_dashboard.artifacts import verify_manifest
from explainable_qml_dashboard.config import ExperimentConfig
from explainable_qml_dashboard.experiment import run_experiment
from explainable_qml_dashboard.inference import load_inference_bundle, predict_frame


def test_small_end_to_end_run(tmp_path, dataset):
    output = tmp_path / "run"
    config = ExperimentConfig(layers=1, epochs=1, patience=1, counterfactual_grid_points=5)
    result = run_experiment(config, output, make_plots=False)
    assert result.output_directory == output
    assert len(result.metrics) == 4
    assert (output / "models" / "quantum_model.json").is_file()
    assert verify_manifest(output) == []

    bundle = load_inference_bundle(output)
    predictions = predict_frame(bundle, dataset.frame.iloc[:2])
    assert len(predictions) == 2
    assert predictions["confidence"].between(0, 1).all()

    conclusion = json.loads((output / "study_conclusion.json").read_text())
    assert conclusion["quantum_advantage_observed"] is False
