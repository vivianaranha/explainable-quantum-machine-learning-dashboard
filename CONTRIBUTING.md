# Contributing

Thank you for improving the Explainable Quantum Machine Learning Dashboard.

## Development workflow

1. Create a focused branch from `main`.
2. Install with `python -m pip install -e ".[dev,notebook]"`.
3. Add tests for every behavior change.
4. Run `make quality` and `make notebook`.
5. Explain statistical, circuit, and compatibility effects in the pull request.

## Evidence rules

- Keep the test partition frozen and fit preprocessing on training data only.
- Label simulator, shot-based, noisy, and hardware evidence explicitly.
- Never describe a score increase as quantum advantage without a defensible resource-matched study.
- Preserve raw run artifacts and configuration alongside reported summaries.
- Define every explanation quantity and its limitations.

## Pull request checklist

- The change is scoped and documented.
- Ruff, tests, builds, and notebook execution pass.
- New plots have been visually inspected.
- Saved-artifact compatibility is tested when applicable.
- Security, data, and responsible-use implications are addressed.
- Branding remains “Created by School of AI and School of QC.”

Created by School of AI and School of QC.

