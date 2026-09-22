# Model card

## Model details

Name: Explainable four-qubit variational quantum classifier

Version: 1.0.0

Developed by: School of AI and School of QC

Model family: Hybrid quantum-classical multiclass classifier using an exact local statevector
simulator and a classical Adam optimization loop

License: MIT

## Intended use

The model is intended for education, portfolio demonstration, numerical testing, and research
prototyping of QML explanation workflows. It shows how to connect circuit internals with model-level
evidence in a fully reproducible repository.

It is not intended for scientific species identification, safety-critical decisions, commercial
automation, claims of quantum advantage, or estimates of quantum-hardware performance.

## Inputs and outputs

Inputs are four finite numeric Iris measurements in centimeters. A train-fitted pipeline converts
them to four angles in `[0, π]`. Outputs are probabilities for setosa, versicolor, and virginica,
plus a maximum-probability class.

## Architecture

The feature map applies per-qubit H, RY, and RZ gates plus three data-dependent adjacent RZZ gates.
Two trainable layers each use eight rotation angles and a four-edge CZ ring. Z expectations on the
first three qubits become scaled softmax logits.

## Training data

The saved reference model uses 90 stratified training rows and 30 validation rows from scikit-learn's
bundled Iris dataset. Preprocessing is fitted only on training rows. Validation loss selects the
checkpoint. Thirty frozen rows form the test partition.

## Reference performance

- Accuracy: 0.767
- Balanced accuracy: 0.767
- Macro F1: 0.764
- One-vs-rest ROC AUC: 0.892
- Log loss: 0.716
- Multiclass Brier score: 0.418
- Expected calibration error: 0.313

Random forest and RBF SVC each achieved 0.933 macro F1 on the same split. No advantage claim is
supported.

## Explainability

The model exposes angle mapping, Bloch components, basis probabilities, intermediate readouts,
single-qubit entropy, full parameter-gradient history, local and global input sensitivity, a
two-feature decision slice, and a bounded one-feature counterfactual. These quantities have different
semantics and are not interchangeable.

## Limitations and risks

- Results are high variance because the dataset and test partition are small.
- Exact simulation does not model measurements or hardware errors.
- The probability head is a design choice, not a physical requirement.
- Explanations are local to one saved model and may change across seeds or parameterizations.
- Counterfactual raw values may violate real botanical correlations.
- The model may be overconfident or miscalibrated; its reference ECE is 0.313.

## Verification

The NumPy simulator matches independently constructed Qiskit statevectors within
`1.6e-15` maximum infidelity on three reference samples. Unit tests also compare analytic
parameter-shift loss gradients with central finite differences.

Created by School of AI and School of QC.

