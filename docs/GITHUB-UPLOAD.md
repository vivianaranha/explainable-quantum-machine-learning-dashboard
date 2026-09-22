# GitHub upload and release guide

## 1. Create the repository

Use the name `explainable-quantum-machine-learning-dashboard` and this description:

> Trace a four-qubit classifier from feature encoding through circuit dynamics, gradients, and
> calibrated predictions—with classical controls and reproducible evidence.

Recommended topics are `quantum-machine-learning`, `explainable-ai`, `qiskit`, `streamlit`,
`variational-quantum-circuit`, `parameter-shift`, `python`, and `portfolio-project`.

## 2. Initialize and push

```bash
git init
git add .
git commit -m "feat: release explainable QML dashboard v1.0.0"
git branch -M main
git remote add origin https://github.com/vivianaranha/explainable-quantum-machine-learning-dashboard.git
git push -u origin main
```

Review `CITATION.cff` and confirm the repository URL before pushing.

## 3. Protect quality

Enable branch protection for `main` and require the `test` workflow. Turn on dependency alerts and
secret scanning. Keep security reports private as described in `SECURITY.md`.

## 4. Add showcase media

The README already embeds generated evidence. For a social preview or release notes, use
`docs/assets/model_metrics.png`, `docs/assets/gradient_heatmap.png`, and
`docs/assets/circuit_dynamics.png`. Do not crop away axes or the evidence-boundary language.

## 5. Create the release

```bash
git tag -a v1.0.0 -m "Explainable QML Dashboard v1.0.0"
git push origin v1.0.0
```

Create a GitHub release from the tag and attach the verified project ZIP plus its SHA-256 checksum.
Summarize the measured classical win and the scope of the explanation dashboard. Do not frame the
release as evidence of quantum advantage.

## 6. First issues

Good first issues include adding bootstrap intervals, explanation-stability tests, and shot-based
readout simulation. A research issue can propose a noise ablation only if it defines controls and
acceptance criteria before results are collected.

Created by School of AI and School of QC.
