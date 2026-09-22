# Responsible use

## Use this project to

- Learn how classical data becomes a parameterized quantum state.
- Audit a small variational circuit and its optimization trace.
- Compare a QML model with competent classical baselines.
- Prototype explanation interfaces and reproducibility contracts.
- Demonstrate honest negative or mixed results in a portfolio.

## Do not use this project to

- Claim quantum advantage, supremacy, or hardware speedup.
- Infer causality from saliency, entropy, gradients, or counterfactuals.
- Make consequential decisions about people, health, finance, employment, or safety.
- Treat a single seed, dataset, split, or simulator as general evidence.
- Load joblib artifacts obtained from an untrusted party.
- Present clipped out-of-range inputs as validated domain behavior.

## Communication checklist

- Say “exact noiseless statevector simulation” wherever runtime or circuit outcomes are discussed.
- Report the strongest classical baseline, even when it outperforms the quantum model.
- Give denominators and split sizes with metrics.
- Distinguish local model sensitivity from real-world importance.
- Call the decision surface a held-feature slice.
- State that counterfactuals may be off-manifold.
- Preserve configurations, raw predictions, and failed experiments.

## Environmental and compute considerations

The four-qubit state has only 16 amplitudes and the dataset has 150 rows. This makes the educational
workflow inexpensive. Statevector cost grows exponentially with qubit count; extensions should
measure memory and energy rather than extrapolating from this toy scale.

Created by School of AI and School of QC.

