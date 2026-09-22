# Project manifest

Version 1.0.0 is a standalone GitHub-ready repository.

## Product surfaces

- `app.py` — six-tab Streamlit dashboard with sample-linked explanations and CSV inference
- `xqml-dashboard` — run, predict, explain, and verify CLI commands
- `notebooks/quickstart.ipynb` — executable guided analysis
- `scripts/` — dataset export, notebook verification, inference wrapper, and safe cleanup

## Core package

The `src/explainable_qml_dashboard/` package contains 12 modules covering configuration, data,
statevector simulation, Qiskit parity, QVC training, evaluation, explanations, visualization,
inference, artifact integrity, orchestration, and CLI behavior.

## Evidence

`examples/reference-run/` contains 44 generated files, including 14 plots, three saved-model files,
raw predictions, calibration bins, all parameter gradients, local/global explanations, the frozen
split, environment and protocol metadata, and an internal SHA-256 manifest.

`docs/assets/` contains eight curated README/report figures. The source evidence remains in the
reference bundle.

## Quality and governance

- 69 automated tests across eight test modules plus shared fixtures
- GitHub Actions on Python 3.11 and 3.12
- Ruff lint and format configuration
- Wheel and source build configuration
- MIT license, citation metadata, changelog, roadmap, contribution guide, security policy, code of
  conduct, issue forms, and pull-request checklist
- Architecture, data, explainability, evaluation, experiment, results, model-card, responsible-use,
  verification, portfolio, self-assessment, and GitHub-upload documentation

## Release exclusions

The curated ZIP excludes virtual environments, bytecode, local caches, transient `artifacts/` runs,
editable-install metadata, and build work directories. It includes the distributable wheel and
source archive under `dist/`.

Created by School of AI and School of QC.

