"""Deterministic visual evidence for the report and dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import TARGET_NAMES
from .evaluation import ModelEvaluation
from .qvc import VariationalQuantumClassifier

COLORS = ("#5B4BDB", "#14A38B", "#F59E0B", "#D14A72")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dataset_split(split_frame: pd.DataFrame, path: Path) -> None:
    counts = split_frame.groupby(["split", "target"]).size().unstack(fill_value=0)
    counts = counts.reindex(["train", "validation", "test"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(counts))
    for target, color in zip(range(3), COLORS, strict=False):
        values = counts[target].to_numpy()
        ax.bar(counts.index, values, bottom=bottom, label=TARGET_NAMES[target], color=color)
        bottom += values
    ax.set(title="Frozen stratified split", ylabel="Samples")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_feature_encoding(encoding: pd.DataFrame, path: Path) -> None:
    ordered = encoding.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    bars = ax.barh(ordered["feature"], ordered["angle_fraction_of_pi"], color=COLORS[0], alpha=0.88)
    ax.set(
        xlabel="Encoded angle / π",
        xlim=(0, 1.05),
        title="Raw features mapped to circuit angles",
    )
    for bar, raw in zip(bars, ordered["raw_value"], strict=True):
        ax.text(
            min(bar.get_width() + 0.02, 0.94),
            bar.get_y() + bar.get_height() / 2,
            f"raw {raw:.2f}",
            va="center",
            fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_training_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(history["epoch"], history["train_loss"], label="train", color=COLORS[0])
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation", color=COLORS[1])
    axes[0].set(title="Cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train", color=COLORS[0])
    axes[1].plot(
        history["epoch"],
        history["validation_accuracy"],
        label="validation",
        color=COLORS[1],
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Score", ylim=(0, 1.03))
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Quantum classifier training trace")
    _save(fig, path)


def plot_gradient_heatmap(gradients: pd.DataFrame, path: Path) -> None:
    matrix = gradients.pivot(index="parameter", columns="epoch", values="gradient")
    bound = max(float(np.max(np.abs(matrix.to_numpy()))), 1e-8)
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    ax.set(
        title="Exact parameter-shift gradients",
        xlabel="Epoch",
        ylabel="Trainable circuit parameter",
        yticks=np.arange(len(matrix)),
        yticklabels=matrix.index,
    )
    tick_positions = np.linspace(0, len(matrix.columns) - 1, min(8, len(matrix.columns))).astype(
        int
    )
    ax.set_xticks(tick_positions, matrix.columns[tick_positions])
    fig.colorbar(image, ax=ax, label="∂ loss / ∂ parameter")
    _save(fig, path)


def plot_gradient_norm(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(history["epoch"], history["gradient_norm"], color=COLORS[3], linewidth=2)
    ax.set(title="Gradient norm by epoch", xlabel="Epoch", ylabel="L2 norm")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_global_saliency(saliency: pd.DataFrame, path: Path) -> None:
    values = saliency.sort_values("mean_absolute_sensitivity")
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.barh(
        values["feature"],
        values["mean_absolute_sensitivity"],
        xerr=values["std_absolute_sensitivity"],
        color=COLORS[1],
        alpha=0.88,
    )
    ax.set(
        title="Global quantum input sensitivity",
        xlabel="Mean |∂ predicted-class probability / ∂ angle|",
    )
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_circuit_dynamics(trace: pd.DataFrame, path: Path) -> None:
    stages = trace["stage"].str.replace("_", " ").tolist()
    x_axis = np.arange(len(trace))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for class_index, name in enumerate(TARGET_NAMES):
        axes[0].plot(
            x_axis,
            trace[f"probability_{name}"],
            marker="o",
            label=name,
            color=COLORS[class_index],
        )
    axes[0].set(
        title="Prediction evolution",
        ylabel="Class probability",
        ylim=(0, 1.03),
        xticks=x_axis,
        xticklabels=stages,
    )
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].legend(frameon=False)
    axes[1].plot(
        x_axis,
        trace["mean_single_qubit_entropy"],
        marker="o",
        color=COLORS[3],
    )
    axes[1].set(
        title="Mean single-qubit entropy",
        ylabel="Entropy (bits)",
        ylim=(0, 1.03),
        xticks=x_axis,
        xticklabels=stages,
    )
    axes[1].tick_params(axis="x", rotation=18)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_basis_probabilities(basis: pd.DataFrame, path: Path) -> None:
    matrix = basis.pivot(index="stage", columns="basis_state", values="probability")
    fig, ax = plt.subplots(figsize=(10.8, 3.7))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Purples", vmin=0.0)
    ax.set(
        title="Basis-state probability through the circuit",
        xlabel="Computational basis |q3q2q1q0⟩",
        ylabel="Circuit stage",
        xticks=np.arange(len(matrix.columns)),
        xticklabels=matrix.columns,
        yticks=np.arange(len(matrix.index)),
        yticklabels=matrix.index.str.replace("_", " "),
    )
    ax.tick_params(axis="x", rotation=55)
    fig.colorbar(image, ax=ax, label="Probability")
    _save(fig, path)


def plot_model_metrics(metrics: pd.DataFrame, path: Path) -> None:
    frame = metrics.set_index("model").sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y_axis = np.arange(len(frame))
    width = 0.34
    ax.barh(y_axis - width / 2, frame["accuracy"], height=width, label="accuracy", color=COLORS[0])
    ax.barh(y_axis + width / 2, frame["macro_f1"], height=width, label="macro F1", color=COLORS[1])
    ax.set(
        title="Frozen-test comparison",
        xlabel="Score",
        xlim=(0, 1.03),
        yticks=y_axis,
        yticklabels=frame.index.str.replace("_", " "),
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_confusion_matrices(evaluations: dict[str, ModelEvaluation], path: Path) -> None:
    model_names = list(evaluations)
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.0 * len(model_names), 3.7))
    for ax, model_name in zip(np.atleast_1d(axes), model_names, strict=True):
        matrix = evaluations[model_name].confusion
        ax.imshow(matrix, cmap="Purples", vmin=0, vmax=max(1, int(matrix.max())))
        for row in range(3):
            for column in range(3):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
        ax.set(
            title=model_name.replace("_", " "),
            xlabel="Predicted",
            ylabel="Actual",
            xticks=range(3),
            xticklabels=TARGET_NAMES,
            yticks=range(3),
            yticklabels=TARGET_NAMES,
        )
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Frozen-test confusion matrices")
    _save(fig, path)


def plot_calibration(calibration: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#697386", label="perfect calibration")
    for index, (model_name, group) in enumerate(calibration.groupby("model")):
        ax.plot(
            group["mean_confidence"],
            group["empirical_accuracy"],
            marker="o",
            label=model_name.replace("_", " "),
            color=COLORS[index % len(COLORS)],
        )
    ax.set(
        title="Top-label reliability",
        xlabel="Mean confidence",
        ylabel="Empirical accuracy",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_confidence_distribution(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for index, (model_name, group) in enumerate(predictions.groupby("model")):
        ax.hist(
            group["confidence"],
            bins=np.linspace(0.3, 1.0, 10),
            alpha=0.42,
            label=model_name.replace("_", " "),
            color=COLORS[index % len(COLORS)],
        )
    ax.set(title="Prediction confidence", xlabel="Maximum class probability", ylabel="Count")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_decision_surface(
    model: VariationalQuantumClassifier,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: tuple[str, ...],
    saliency: pd.DataFrame,
    path: Path,
) -> None:
    top_features = [feature_names.index(name) for name in saliency["feature"].iloc[:2]]
    first, second = top_features
    grid = np.linspace(0.0, np.pi, 55)
    xx, yy = np.meshgrid(grid, grid)
    base = np.median(x_train, axis=0)
    points = np.repeat(base[None, :], xx.size, axis=0)
    points[:, first] = xx.ravel()
    points[:, second] = yy.ravel()
    zz = model.predict(points).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.contourf(
        xx / np.pi,
        yy / np.pi,
        zz,
        levels=[-0.5, 0.5, 1.5, 2.5],
        alpha=0.2,
        colors=COLORS[:3],
    )
    for class_index, name in enumerate(TARGET_NAMES):
        include = y_test == class_index
        ax.scatter(
            x_test[include, first] / np.pi,
            x_test[include, second] / np.pi,
            label=name,
            color=COLORS[class_index],
            edgecolor="white",
            linewidth=0.7,
        )
    ax.set(
        title="Quantum decision surface (other angles held at train median)",
        xlabel=f"{feature_names[first]} / π",
        ylabel=f"{feature_names[second]} / π",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_counterfactual(counterfactual: dict[str, object], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    if bool(counterfactual.get("found")):
        values = [
            float(counterfactual["original_raw_value"]),
            float(counterfactual["counterfactual_raw_value"]),
        ]
        ax.bar(["original", "counterfactual"], values, color=[COLORS[0], COLORS[3]])
        ax.set(
            title=f"Nearest one-feature prediction flip: {counterfactual['feature']}",
            ylabel="Raw feature value",
        )
    else:
        ax.text(0.5, 0.5, "No one-feature flip found", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def create_all_plots(
    *,
    output_directory: Path,
    split_frame: pd.DataFrame,
    encoding: pd.DataFrame,
    history: pd.DataFrame,
    gradients: pd.DataFrame,
    saliency: pd.DataFrame,
    trace: pd.DataFrame,
    basis: pd.DataFrame,
    metrics: pd.DataFrame,
    evaluations: dict[str, ModelEvaluation],
    calibration: pd.DataFrame,
    predictions: pd.DataFrame,
    quantum_model: VariationalQuantumClassifier,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: tuple[str, ...],
    counterfactual: dict[str, object],
) -> list[Path]:
    """Render the complete reference evidence suite."""

    output_directory.mkdir(parents=True, exist_ok=True)
    jobs = {
        "dataset_split.png": lambda path: plot_dataset_split(split_frame, path),
        "feature_encoding.png": lambda path: plot_feature_encoding(encoding, path),
        "training_history.png": lambda path: plot_training_history(history, path),
        "gradient_heatmap.png": lambda path: plot_gradient_heatmap(gradients, path),
        "gradient_norm.png": lambda path: plot_gradient_norm(history, path),
        "global_saliency.png": lambda path: plot_global_saliency(saliency, path),
        "circuit_dynamics.png": lambda path: plot_circuit_dynamics(trace, path),
        "basis_probabilities.png": lambda path: plot_basis_probabilities(basis, path),
        "model_metrics.png": lambda path: plot_model_metrics(metrics, path),
        "confusion_matrices.png": lambda path: plot_confusion_matrices(evaluations, path),
        "calibration.png": lambda path: plot_calibration(calibration, path),
        "confidence_distribution.png": lambda path: plot_confidence_distribution(predictions, path),
        "decision_surface.png": lambda path: plot_decision_surface(
            quantum_model, x_train, x_test, y_test, feature_names, saliency, path
        ),
        "counterfactual.png": lambda path: plot_counterfactual(counterfactual, path),
    }
    paths = []
    for filename, plotter in jobs.items():
        target = output_directory / filename
        plotter(target)
        paths.append(target)
    return paths
