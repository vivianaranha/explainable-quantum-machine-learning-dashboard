# Experiment guide

## Reproduce the reference run

```bash
python -m pip install -e ".[dev]"
xqml-dashboard run --config configs/default.json --output artifacts/reproduction
xqml-dashboard verify --run artifacts/reproduction
```

Compare `summary_metrics.csv`, `quantum_diagnostics.json`, `training_history.csv`, and
`study_conclusion.json` with `examples/reference-run/`. Timings and creation timestamps may differ.

## Run a fast smoke experiment

```bash
xqml-dashboard run \
  --config configs/default.json \
  --epochs 3 \
  --output artifacts/smoke
```

This verifies the full data, training, explanation, plotting, and serialization path but should not
be interpreted as a trained benchmark.

## Change the protocol

Copy `configs/default.json`, change one independent variable, and write to a new output directory.
Do not overwrite the checked-in reference evidence when exploring. Record the research question and
expected direction before looking at test results.

Useful ablations include:

- One versus two variational layers
- Feature-map entanglement on versus off
- Exact versus finite-shot readouts
- Alternate logit scales
- Different train/validation seeds with the test protocol re-frozen
- Local saliency step sizes
- Median-held decision slices for different feature pairs

## Inspect a single prediction

```bash
xqml-dashboard explain \
  --run examples/reference-run \
  --x0 5.1 --x1 3.5 --x2 1.4 --x3 0.2
```

The command reports the saved quantum prediction, class probabilities, encoded angles, signed local
sensitivities, and nearest bounded one-feature counterfactual.

## Research log checklist

- State the question and success criterion.
- Save the complete configuration and environment.
- Keep split assignments fixed for the comparison.
- Record failed as well as successful runs.
- Inspect probability metrics, not only accuracy.
- Separate simulator observations from hardware expectations.
- Report when a classical baseline wins.
- Note every post-hoc choice.

Created by School of AI and School of QC.

