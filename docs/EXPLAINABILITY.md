# Explainability methods

The project uses four linked explanation layers. Each answers a different question and carries a
different limitation.

## 1. Feature encoding

The encoding view reports the raw measurement, exact angle, fraction of π, and post-feature-map
Bloch components for each qubit. The Bloch vector is obtained from local Pauli expectations:

\[
\vec r_i = (\langle X_i\rangle,\langle Y_i\rangle,\langle Z_i\rangle).
\]

Because the feature map contains `RZZ` gates, a local Bloch vector can shrink even though the global
state remains pure. This is a state description, not an importance ranking.

## 2. Circuit behavior

After the feature map and each variational layer, the project records:

- All 16 computational-basis probabilities
- The three Z expectations used as logits
- The resulting softmax class probabilities
- Mean single-qubit von Neumann entropy

For reduced state \(\rho_i\), entropy is

\[
S(\rho_i)=-\mathrm{Tr}(\rho_i\log_2\rho_i).
\]

For the pure four-qubit global state, nonzero local entropy indicates entanglement with the other
qubits. It does not say that a layer or feature is useful.

## 3. Gradients

The optimization view saves the exact cross-entropy derivative for every trainable rotation at
every epoch. Each Pauli-generated rotation uses two shifted expectation evaluations. The chain rule
propagates those readout derivatives through the scaled softmax loss.

The heatmap can reveal active parameters, sign reversals, convergence, or persistently small
directions. A small gradient at one point is not proof of an architecture-wide barren plateau.

Input saliency uses a symmetric finite difference of the saved model's predicted-class probability:

\[
g_i(x)=\frac{p_{\hat y}(x_i+h)-p_{\hat y}(x_i-h)}{2h}.
\]

At angle boundaries the denominator uses the actual clipped interval. Global saliency is the mean
absolute local value across the frozen test partition. These are sensitivities per radian, not causal
effects and not directly comparable to raw-unit feature effects.

## 4. Predictions

Prediction explanations include probabilities, confidence, confusion matrices, probability scores,
calibration bins, a two-feature decision slice, and a nearest one-feature counterfactual.

The counterfactual tests 121 values from 0 to π for each feature, holds other angles fixed, and keeps
the closest change that flips the quantum prediction. The raw value is recovered through the fitted
inverse transform. It may not be a botanically plausible flower.

The decision surface chooses the two features with highest global sensitivity and holds the other
features at their training medians. It is a slice, not a projection or complete boundary.

## Stability and trust

An explanation is evidence about a particular model, input, checkpoint, and numerical procedure.
Before using one operationally, test it across seeds, nearby inputs, shot budgets, noise models,
alternative explainers, and domain constraints. Agreement among multiple views is more informative
than a single attractive chart.

Created by School of AI and School of QC.

