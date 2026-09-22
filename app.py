"""Interactive Streamlit interface for the Explainable QML Dashboard."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from explainable_qml_dashboard.artifacts import verify_manifest
from explainable_qml_dashboard.config import ExperimentConfig
from explainable_qml_dashboard.data import FEATURE_NAMES, TARGET_NAMES
from explainable_qml_dashboard.experiment import run_experiment
from explainable_qml_dashboard.explain import (
    circuit_trace_frames,
    encoding_frame,
    local_input_saliency,
)
from explainable_qml_dashboard.inference import load_inference_bundle, predict_frame
from explainable_qml_dashboard.quantum import build_qiskit_circuit

ROOT = Path(__file__).resolve().parent
REFERENCE_RUN = ROOT / "examples" / "reference-run"
COLORS = ("#5B4BDB", "#14A38B", "#F59E0B")

st.set_page_config(
    page_title="Explainable QML Dashboard",
    page_icon="⚛️",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {background: linear-gradient(120deg,#f1efff,#eefbf8); border:1px solid #dedcf7;
           border-radius:18px; padding:1.45rem 1.6rem; margin-bottom:1rem;}
    .hero h1 {margin:0; color:#172033; font-size:2.15rem;}
    .hero p {margin:.4rem 0 0; color:#4b5568;}
    .callout {border-left:4px solid #5B4BDB; padding:.7rem 1rem; background:#f8f8fd;}
    </style>
    <div class="hero">
      <h1>Explainable Quantum Machine Learning Dashboard</h1>
      <p>Trace one prediction from raw features to angles, quantum state dynamics,
      optimization gradients, and frozen-test evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle(run_directory: str):
    return load_inference_bundle(run_directory)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


if not (REFERENCE_RUN / "models" / "quantum_model.json").exists():
    st.error("Reference artifacts are missing. Run `make reference` first.")
    st.stop()

bundle = load_bundle(str(REFERENCE_RUN))
dataset = load_csv(str(REFERENCE_RUN / "dataset_snapshot.csv"))
split = load_csv(str(REFERENCE_RUN / "fixed_split.csv"))
metrics = load_csv(str(REFERENCE_RUN / "summary_metrics.csv"))
predictions = load_csv(str(REFERENCE_RUN / "test_predictions.csv"))
history = load_csv(str(REFERENCE_RUN / "training_history.csv"))
gradients = load_csv(str(REFERENCE_RUN / "parameter_gradients.csv"))
test_ids = split.loc[split["split"] == "test", "row_id"].astype(int).tolist()

with st.sidebar:
    st.subheader("Audit controls")
    selected_row = st.selectbox("Test sample row", test_ids, index=0)
    st.caption("Changing the row updates every local explanation from the same saved model.")
    st.divider()
    st.subheader("Reproduce")
    epochs = st.slider("Maximum epochs", 5, 60, 20, 5)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    if st.button("Run a new local experiment", use_container_width=True):
        config = ExperimentConfig.from_json(ROOT / "configs" / "default.json")
        values = config.to_dict()
        values.update({"epochs": epochs, "random_seed": int(seed)})
        destination = ROOT / "artifacts" / f"dashboard-seed-{int(seed)}"
        with st.spinner("Training the quantum model and baselines…"):
            result = run_experiment(ExperimentConfig.from_dict(values), destination)
        st.success(f"Completed. Best frozen-test model: {result.best_model.replace('_', ' ')}")
    st.divider()
    integrity_failures = verify_manifest(REFERENCE_RUN)
    st.success("Reference manifest verified" if not integrity_failures else "Integrity warning")

raw_sample = dataset.loc[dataset["row_id"] == selected_row, list(FEATURE_NAMES)].to_numpy(
    dtype=float
)[0]
angles = bundle.preprocessor.transform(raw_sample[None, :])[0]
probabilities = bundle.quantum_model.predict_proba(angles[None, :])[0]
predicted_class = int(np.argmax(probabilities))

overview, encoding_tab, circuit_tab, gradients_tab, predictions_tab, upload_tab = st.tabs(
    ["Overview", "Encoding", "Circuit", "Gradients", "Predictions", "CSV inference"]
)

with overview:
    best = metrics.sort_values("macro_f1", ascending=False).iloc[0]
    quantum = metrics.loc[metrics["model"] == "variational_quantum_classifier"].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(dataset))
    col2.metric("Qubits", 4)
    col3.metric("Trainable angles", bundle.quantum_model.parameter_count)
    col4.metric("Best macro F1", f"{best['macro_f1']:.3f}")
    st.markdown(
        '<div class="callout"><b>Evidence boundary.</b> This is a noiseless, exact-statevector '
        "Iris case study. It visualizes mechanisms and checks implementation parity; it does not "
        "claim quantum advantage or hardware speedup.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        image(REFERENCE_RUN / "plots" / "model_metrics.png", "Four-model frozen-test comparison")
    with right:
        image(REFERENCE_RUN / "plots" / "calibration.png", "Confidence reliability")
    st.caption(
        f"Reference quantum macro F1: {quantum['macro_f1']:.3f}. "
        "All preprocessing was fitted on train only; the test split stayed frozen."
    )

with encoding_tab:
    st.subheader(f"Row {selected_row}: raw measurement → quantum angle")
    encoding = encoding_frame(raw_sample, angles, FEATURE_NAMES)
    left, right = st.columns([1.15, 1])
    with left:
        chart = encoding.set_index("feature")[["angle_fraction_of_pi"]]
        st.bar_chart(chart, color=COLORS[0], horizontal=True)
        st.dataframe(
            encoding[["feature", "raw_value", "angle_radians", "angle_fraction_of_pi"]],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        bloch_columns = [
            "feature",
            "bloch_x_after_encoding",
            "bloch_y_after_encoding",
            "bloch_z_after_encoding",
        ]
        st.write("Bloch-vector components after the full feature map")
        st.dataframe(encoding[bloch_columns], hide_index=True, use_container_width=True)
        st.caption(
            "The angle mapping is fitted only on the training partition and clipped to [0, π]. "
            "Bloch components describe local qubit marginals after data-dependent entanglement."
        )

with circuit_tab:
    st.subheader("Layer-by-layer circuit behavior")
    trace, basis = circuit_trace_frames(angles, bundle.quantum_model)
    left, right = st.columns(2)
    with left:
        probability_columns = [f"probability_{name}" for name in TARGET_NAMES]
        probability_trace = trace.set_index("stage")[probability_columns]
        probability_trace.columns = TARGET_NAMES
        st.line_chart(probability_trace, color=list(COLORS))
        st.caption("Softmax probabilities recomputed after the feature map and each ansatz layer.")
    with right:
        entropy = trace.set_index("stage")[["mean_single_qubit_entropy"]]
        st.line_chart(entropy, color="#D14A72")
        st.caption("Mean one-qubit von Neumann entropy is a diagnostic, not an importance score.")
    st.dataframe(trace, hide_index=True, use_container_width=True)
    with st.expander("Qiskit circuit for this sample"):
        circuit = build_qiskit_circuit(angles, bundle.quantum_model.parameters)
        st.code(str(circuit.draw(output="text", fold=110)), language="text")
    with st.expander("All 16 basis-state probabilities"):
        st.dataframe(basis, hide_index=True, use_container_width=True)

with gradients_tab:
    st.subheader("Optimization gradients and input sensitivity")
    image(REFERENCE_RUN / "plots" / "gradient_heatmap.png", "Exact parameter-shift gradient trace")
    left, right = st.columns(2)
    with left:
        st.line_chart(history.set_index("epoch")[["train_loss", "validation_loss"]])
        st.caption(
            "Validation loss selects the saved epoch; the test labels never select parameters."
        )
    with right:
        local = local_input_saliency(bundle.quantum_model, angles, FEATURE_NAMES)
        local_chart = local.set_index("feature")[["signed_sensitivity_per_radian"]]
        st.bar_chart(local_chart, horizontal=True, color="#14A38B")
        st.caption(
            "Signed local derivative of the predicted-class probability with respect to each "
            "encoded angle. It is local sensitivity, not causality."
        )
    with st.expander("Raw gradient log"):
        st.dataframe(gradients, hide_index=True, use_container_width=True)

with predictions_tab:
    st.subheader(f"Saved prediction for row {selected_row}")
    columns = st.columns(3)
    for index, name in enumerate(TARGET_NAMES):
        columns[index].metric(name.title(), f"{probabilities[index]:.3f}")
    st.success(
        f"Quantum prediction: {TARGET_NAMES[predicted_class]} "
        f"({probabilities[predicted_class]:.1%})"
    )
    selected_predictions = predictions.loc[predictions["row_id"] == selected_row]
    st.dataframe(
        selected_predictions[["model", "target_name", "prediction_name", "correct", "confidence"]],
        hide_index=True,
        use_container_width=True,
    )
    left, right = st.columns(2)
    with left:
        image(REFERENCE_RUN / "plots" / "confusion_matrices.png", "Frozen-test confusion matrices")
    with right:
        image(REFERENCE_RUN / "plots" / "decision_surface.png", "Two-feature model slice")

with upload_tab:
    st.subheader("Classify a local CSV with saved models")
    st.write("Required columns: " + ", ".join(FEATURE_NAMES))
    model_name = st.selectbox("Saved model", bundle.model_names)
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            upload_frame = pd.read_csv(uploaded)
            inferred = predict_frame(bundle, upload_frame, model_name=model_name)
            st.dataframe(inferred, hide_index=True, use_container_width=True)
            buffer = io.StringIO()
            inferred.to_csv(buffer, index=False)
            st.download_button(
                "Download predictions",
                data=buffer.getvalue(),
                file_name="xqml_predictions.csv",
                mime="text/csv",
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))

st.divider()
st.caption("Created by School of AI and School of QC")
