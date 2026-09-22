# Portfolio guide

## One-sentence pitch

I built a four-qubit classification system that lets a reviewer trace one prediction from raw
features through quantum encoding, intermediate state behavior, exact gradients, calibration, and a
bounded counterfactual—then compare it honestly with three classical models.

## Five-minute demo

1. Open **Overview** and state the evidence boundary: exact noiseless simulation, 150 Iris rows, no
   quantum-advantage claim.
2. Show the frozen-test chart. Point out that random forest and RBF SVC beat the QVC, 0.933 versus
   0.764 macro F1.
3. Select test row 7. In **Encoding**, connect the four raw measurements to four circuit angles and
   explain why preprocessing is train-only.
4. In **Circuit**, show how the feature-map preference changes after layer one and how entropy
   changes without calling it importance.
5. In **Gradients**, show the 16-parameter heatmap and the local signed sensitivity chart.
6. In **Predictions**, compare all four saved models, then show the decision slice.
7. In **CSV inference**, upload a local Iris CSV and download saved-model predictions.
8. Close with the parity check and the honest result: interpretability was delivered; advantage was
   not demonstrated.

## Resume bullet

Built and validated an explainable four-qubit QML platform with exact parameter-shift training,
independent Qiskit parity checks, four-layer explanation evidence, three classical baselines,
saved-model inference, Streamlit, CI, and 69 automated tests; documented a −0.169 macro-F1 gap to
the best classical control without overstating quantum performance.

## Interview talking points

### Why a custom simulator and Qiskit?

The custom 16-amplitude engine exposes every intermediate state and makes batched shifted evaluations
straightforward. Qiskit acts as an independent implementation check rather than sharing the same
code path.

### How did you avoid leakage?

The split occurs before preprocessing. Both scalers fit on training rows only. Validation labels
select the QVC checkpoint, while test labels are used only for final evidence.

### Why were the classical baselines important?

They establish whether the quantum model adds predictive value under the same representation and
split. Here they showed that it did not, which is a valuable engineering finding.

### What makes the dashboard explainable?

It connects four levels that are often shown separately: the encoding transformation, internal state
dynamics, optimization gradients, and prediction-level evidence. Selecting a row updates all local
views from the same model.

### What would you do next?

Measure stability across seeds, add finite-shot and noise studies, compare explanation methods,
pre-register compute-matched hyperparameter budgets, and test on multiple datasets before considering
hardware claims.

## Demo checklist

- Start from a fresh environment and verify the reference manifest.
- Keep the exact version and seed visible.
- Demonstrate one correct and one incorrect test prediction.
- Define entropy, parameter gradients, and saliency separately.
- State that the counterfactual can be off-manifold.
- Show the classical winner and the calibration gap.
- Keep a terminal ready with `pytest` and the parity diagnostic.
- End on limitations and next experiments.

Created by School of AI and School of QC.

