# Evaluation protocol

## Question

Can a compact variational quantum classifier be made transparent enough to audit its data encoding,
state evolution, optimization, and predictions while remaining comparable to ordinary classifiers?

## Pre-registered choices

- Dataset: bundled Iris, all four features, all three classes
- Seed: 42
- Split: 60% train, 20% validation, 20% test, stratified
- Representation: train-fitted standardization followed by `[0, π]` scaling
- Quantum model: four qubits, two layers, 16 angles, exact statevector
- Optimizer: full-batch Adam, learning rate 0.08, at most 45 epochs
- Selection: validation loss with patience 10 and minimum improvement 0.0001
- Baselines: logistic regression, RBF SVC, random forest with fixed settings
- Primary descriptive metric: frozen-test macro F1
- No hypothesis test and no quantum-advantage claim

## Metrics

Discrimination metrics are accuracy, balanced accuracy, macro precision, macro recall, macro F1, and
one-vs-rest ROC AUC. Probability metrics are multiclass log loss, multiclass Brier score, and
top-label expected calibration error over eight bins. Runtime metrics are fit seconds and inference
milliseconds per sample.

Timing is recorded for transparency but not treated as a fair performance benchmark: implementations
use different languages and algorithms, the QVC uses exact simulation, and the dataset is too small
for stable latency measurements.

## Integrity checks

- NumPy statevectors are checked against independently constructed Qiskit statevectors.
- Parameter-shift loss gradients are checked against central finite differences.
- State norms, probability sums, expectation ranges, and entropy bounds are tested.
- The generated run is protected with a SHA-256 manifest.
- The release is retested after extraction into a fresh directory.

## What would be needed for stronger conclusions

A comparative research claim would need repeated nested cross-validation, predeclared model budgets,
hyperparameter searches of comparable scope, uncertainty intervals, multiple datasets, ablations,
shot/noise analysis, hardware execution, resource accounting, and a definition of advantage that is
not merely one favorable accuracy result.

Created by School of AI and School of QC.

