"""End-to-end reproducible experiment orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import (
    build_manifest,
    environment_metadata,
    save_joblib,
    write_json,
    write_text,
)
from .config import ExperimentConfig
from .data import AnglePreprocessor, dataset_summary, load_dataset, make_splits
from .evaluation import (
    ModelEvaluation,
    calibration_frame,
    fit_classical_baselines,
    metrics_frame,
    predictions_frame,
    timed_evaluation,
)
from .explain import (
    circuit_trace_frames,
    encoding_frame,
    global_input_saliency,
    local_input_saliency,
    nearest_single_feature_counterfactual,
)
from .quantum import build_qiskit_circuit, qiskit_state, simulate, state_fidelity
from .qvc import VariationalQuantumClassifier
from .visualization import create_all_plots


@dataclass(frozen=True)
class ExperimentResult:
    output_directory: Path
    metrics: pd.DataFrame
    best_model: str
    quantum_best_epoch: int
    quantum_stopped_epoch: int


def _preprocessor_metadata(preprocessor: AnglePreprocessor) -> dict[str, object]:
    return {
        "pipeline": ["StandardScaler", "MinMaxScaler(feature_range=[0, pi], clip=true)"],
        "fit_partition": "train only",
        "feature_names": list(preprocessor.feature_names),
        "standardizer_mean": preprocessor.standardizer.mean_.tolist(),
        "standardizer_scale": preprocessor.standardizer.scale_.tolist(),
        "standardized_train_min": preprocessor.angle_scaler.data_min_.tolist(),
        "standardized_train_max": preprocessor.angle_scaler.data_max_.tolist(),
        "angle_range": [0.0, float(np.pi)],
        "minmax_scale": preprocessor.angle_scaler.scale_.tolist(),
        "minmax_offset": preprocessor.angle_scaler.min_.tolist(),
    }


def _circuit_specification(config: ExperimentConfig) -> dict[str, object]:
    return {
        "qubits": config.qubits,
        "layers": config.layers,
        "trainable_parameters": config.parameter_count,
        "feature_map": [
            "H on every qubit",
            "RY(x_i) then RZ(x_i / 2) on qubit i",
            "RZZ(0.5 * (pi-x_i) * (pi-x_j) / pi) on adjacent pairs",
        ],
        "ansatz_layer": [
            "trainable RY and RZ on every qubit",
            "CZ ring: (0,1), (1,2), (2,3), (3,0)",
        ],
        "readout": "Exact Z expectation on qubits 0, 1, and 2; scaled softmax logits",
        "training_gradient": "Two-evaluation parameter-shift rule for every trainable angle",
        "execution": "Exact noiseless local statevector; no hardware claim",
    }


def _conclusion(metrics: pd.DataFrame) -> dict[str, object]:
    ordered = metrics.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    quantum = metrics.loc[metrics["model"] == "variational_quantum_classifier"].iloc[0]
    best = ordered.iloc[0]
    classical = metrics.loc[metrics["model"] != "variational_quantum_classifier"]
    best_classical = classical.sort_values("macro_f1", ascending=False).iloc[0]
    delta = float(quantum["macro_f1"] - best_classical["macro_f1"])
    return {
        "best_model": str(best["model"]),
        "best_macro_f1": float(best["macro_f1"]),
        "quantum_macro_f1": float(quantum["macro_f1"]),
        "best_classical_model": str(best_classical["model"]),
        "best_classical_macro_f1": float(best_classical["macro_f1"]),
        "quantum_minus_best_classical_macro_f1": delta,
        "quantum_advantage_observed": False,
        "interpretation": (
            "This single small, noiseless simulation is an explainability case study, not evidence "
            "of quantum advantage. Scores are descriptive and should not be generalized."
        ),
    }


def _report_markdown(
    metrics: pd.DataFrame,
    conclusion: dict[str, object],
    *,
    sample_row_id: int,
    best_epoch: int,
    stopped_epoch: int,
    parity_infidelity: float,
) -> str:
    lines = [
        "# Experiment report",
        "",
        "Created by School of AI and School of QC.",
        "",
        "## Protocol",
        "",
        "Iris was split once with stratification into train, validation, and test partitions. "
        "Both preprocessing stages were fitted on train only. All four models saw the same "
        "angle-encoded inputs. The test partition stayed untouched until final evaluation.",
        "",
        "## Frozen-test results",
        "",
    ]
    for row in metrics.sort_values("macro_f1", ascending=False).itertuples(index=False):
        lines.append(
            f"- **{row.model.replace('_', ' ').title()}** — accuracy {row.accuracy:.3f}, "
            f"macro F1 {row.macro_f1:.3f}, log loss {row.log_loss:.3f}, "
            f"ECE {row.expected_calibration_error:.3f}."
        )
    lines.extend(
        [
            "",
            "## Quantum audit",
            "",
            f"- Validation-selected epoch: {best_epoch}; training stopped at epoch "
            f"{stopped_epoch}.",
            f"- Independent NumPy/Qiskit maximum parity infidelity: {parity_infidelity:.3e}.",
            f"- Reference explanation row: dataset row {sample_row_id}.",
            "- Gradients are exact parameter-shift derivatives of the noiseless simulator.",
            "- Input saliency is a local finite-difference sensitivity, not a causal effect.",
            "- The counterfactual changes one feature inside the train-fitted angle range; it is "
            "a model probe, not a real-world recommendation.",
            "",
            "## Conclusion",
            "",
            "The best frozen-test macro F1 came from "
            f"**{str(conclusion['best_model']).replace('_', ' ')}** "
            f"at {float(conclusion['best_macro_f1']):.3f}. The quantum-minus-best-classical "
            f"difference was {float(conclusion['quantum_minus_best_classical_macro_f1']):+.3f}. "
            "This run does not establish quantum advantage; its value is the linked audit trail "
            "from input encoding through circuit behavior, optimization, and predictions.",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    config: ExperimentConfig,
    output_directory: str | Path,
    *,
    make_plots: bool = True,
) -> ExperimentResult:
    """Run the benchmark and persist a self-contained, auditable artifact bundle."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    models_directory = output / "models"
    plots_directory = output / "plots"
    models_directory.mkdir(exist_ok=True)

    dataset = load_dataset()
    splits = make_splits(
        dataset.y,
        test_fraction=config.test_fraction,
        validation_fraction=config.validation_fraction,
        random_seed=config.random_seed,
    )
    train = splits.train_indices
    validation = splits.validation_indices
    test = splits.test_indices
    preprocessor = AnglePreprocessor.fit(dataset.x_raw[train])
    x_angles = preprocessor.transform(dataset.x_raw)

    quantum_model = VariationalQuantumClassifier(
        layers=config.layers,
        logit_scale=config.logit_scale,
        random_seed=config.random_seed,
    )
    quantum_started = time.perf_counter()
    training = quantum_model.fit(
        x_angles[train],
        dataset.y[train],
        x_angles[validation],
        dataset.y[validation],
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        patience=config.patience,
        min_delta=config.min_delta,
        gradient_clip=config.gradient_clip,
    )
    quantum_fit_seconds = time.perf_counter() - quantum_started

    baselines, baseline_fit_times = fit_classical_baselines(
        x_angles[train], dataset.y[train], config.random_seed
    )
    evaluations: dict[str, ModelEvaluation] = {
        "variational_quantum_classifier": timed_evaluation(
            quantum_model, x_angles[test], dataset.y[test], fit_seconds=quantum_fit_seconds
        )
    }
    for name, model in baselines.items():
        evaluations[name] = timed_evaluation(
            model, x_angles[test], dataset.y[test], fit_seconds=baseline_fit_times[name]
        )

    metrics = metrics_frame(evaluations)
    predictions = predictions_frame(test, dataset.y[test], dataset.target_names, evaluations)
    calibration = calibration_frame(dataset.y[test], evaluations)
    split_frame = splits.as_frame(dataset.y)
    reference_index = int(test[0])
    encoding = encoding_frame(
        dataset.x_raw[reference_index], x_angles[reference_index], dataset.feature_names
    )
    trace, basis = circuit_trace_frames(x_angles[reference_index], quantum_model)
    local_saliency = local_input_saliency(
        quantum_model,
        x_angles[reference_index],
        dataset.feature_names,
        step=config.saliency_step,
    )
    global_saliency = global_input_saliency(
        quantum_model, x_angles[test], dataset.feature_names, step=config.saliency_step
    )
    counterfactual = nearest_single_feature_counterfactual(
        quantum_model,
        x_angles[reference_index],
        preprocessor,
        grid_points=config.counterfactual_grid_points,
    )

    parity_fidelities = []
    for index in test[:3]:
        numpy_state = simulate(x_angles[index], quantum_model.parameters)[0]
        parity_fidelities.append(
            state_fidelity(numpy_state, qiskit_state(x_angles[index], quantum_model.parameters))
        )
    parity_infidelity = float(max(0.0, 1.0 - min(parity_fidelities)))
    quantum_diagnostics = {
        "numpy_qiskit_state_fidelities": parity_fidelities,
        "maximum_infidelity": parity_infidelity,
        "final_state_norm": float(
            np.linalg.norm(simulate(x_angles[reference_index], quantum_model.parameters)[0])
        ),
        "best_epoch": training.best_epoch,
        "stopped_epoch": training.stopped_epoch,
        "parameter_count": quantum_model.parameter_count,
        "reference_row_id": reference_index,
    }
    conclusion = _conclusion(metrics)

    write_json(output / "config.json", config.to_dict())
    write_json(output / "dataset_summary.json", dataset_summary(dataset, splits))
    write_json(output / "environment.json", environment_metadata())
    write_json(output / "preprocessing.json", _preprocessor_metadata(preprocessor))
    write_json(output / "circuit_specification.json", _circuit_specification(config))
    write_json(
        output / "metrics.json", {name: value.metrics for name, value in evaluations.items()}
    )
    write_json(output / "quantum_diagnostics.json", quantum_diagnostics)
    write_json(output / "counterfactual.json", counterfactual)
    write_json(output / "study_conclusion.json", conclusion)
    write_json(
        output / "run_metadata.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "project_version": "1.0.0",
            "random_seed": config.random_seed,
            "reference_row_id": reference_index,
            "branding": "Created by School of AI and School of QC",
        },
    )
    dataset.frame.assign(row_id=np.arange(len(dataset.frame))).to_csv(
        output / "dataset_snapshot.csv", index=False
    )
    split_frame.to_csv(output / "fixed_split.csv", index=False)
    pd.DataFrame(x_angles, columns=dataset.feature_names).assign(
        row_id=np.arange(len(x_angles)), target=dataset.y
    ).to_csv(output / "angle_encoded_dataset.csv", index=False)
    training.history.to_csv(output / "training_history.csv", index=False)
    training.gradients.to_csv(output / "parameter_gradients.csv", index=False)
    metrics.to_csv(output / "summary_metrics.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    calibration.to_csv(output / "calibration_bins.csv", index=False)
    encoding.to_csv(output / "reference_encoding.csv", index=False)
    trace.to_csv(output / "reference_circuit_trace.csv", index=False)
    basis.to_csv(output / "reference_basis_probabilities.csv", index=False)
    local_saliency.to_csv(output / "reference_local_saliency.csv", index=False)
    global_saliency.to_csv(output / "global_saliency.csv", index=False)

    quantum_model.save(models_directory / "quantum_model.json")
    save_joblib(models_directory / "preprocessor.joblib", preprocessor)
    save_joblib(models_directory / "classical_baselines.joblib", baselines)
    circuit_text = str(
        build_qiskit_circuit(x_angles[reference_index], quantum_model.parameters).draw(
            output="text", fold=120
        )
    )
    write_text(output / "reference_circuit.txt", circuit_text)
    write_text(
        output / "experiment_report.md",
        _report_markdown(
            metrics,
            conclusion,
            sample_row_id=reference_index,
            best_epoch=training.best_epoch,
            stopped_epoch=training.stopped_epoch,
            parity_infidelity=parity_infidelity,
        ),
    )

    if make_plots:
        create_all_plots(
            output_directory=plots_directory,
            split_frame=split_frame,
            encoding=encoding,
            history=training.history,
            gradients=training.gradients,
            saliency=global_saliency,
            trace=trace,
            basis=basis,
            metrics=metrics,
            evaluations=evaluations,
            calibration=calibration,
            predictions=predictions,
            quantum_model=quantum_model,
            x_train=x_angles[train],
            x_test=x_angles[test],
            y_test=dataset.y[test],
            feature_names=dataset.feature_names,
            counterfactual=counterfactual,
        )

    write_text(
        output / "README.md",
        """# Reference run

This directory is a seed-locked, fully generated evidence bundle. Start with
`experiment_report.md`, `summary_metrics.csv`, and the four explanation families under `plots/`.
Binary joblib files are trusted local artifacts generated by this repository; never load joblib
files from an untrusted source.

Created by School of AI and School of QC.
""",
    )
    write_json(output / "artifact_manifest.json", build_manifest(output))
    best_model = str(metrics.iloc[0]["model"])
    return ExperimentResult(
        output_directory=output,
        metrics=metrics,
        best_model=best_model,
        quantum_best_epoch=training.best_epoch,
        quantum_stopped_epoch=training.stopped_epoch,
    )
