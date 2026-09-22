# Verification record

Release: 1.0.0

Verification date: 2026-09-22

## Automated checks

- Ruff lint: passed in the source tree and clean extracted archive
- Ruff format check: passed across 50 extracted files
- Pytest: 69 tests passed in 3.05 seconds after clean extraction
- Wheel and source build: passed from the clean extracted archive
- Notebook execution: all standard-Python cells passed in the sandbox-safe in-process runner
- Artifact manifest: all 43 listed reference artifacts verified by size and SHA-256 digest
- Streamlit smoke test: server started successfully from the clean extracted archive

## Scientific checks

- All 150 dataset rows belong to exactly one frozen partition.
- Each class contributes 30/10/10 rows to train/validation/test.
- Preprocessing fits only training rows and produces angles in `[0, π]`.
- Exact NumPy statevectors match Qiskit with maximum reference infidelity `1.55e-15`.
- Parameter-shift loss derivatives match central finite differences in tests.
- Probabilities, state norms, Bloch vectors, expectations, and entropies satisfy numerical bounds.
- Test metrics are computed from saved probabilities and fixed labels.
- The conclusion records `quantum_advantage_observed: false`.

## Visual review

All 14 reference plots were reviewed together at full generation resolution. Labels, legends,
scales, class colors, and cropping were checked. The calibration plot is intentionally jagged because
the frozen test set contains only 30 rows; it is not smoothed.

## Clean-archive verification

The curated ZIP was extracted into a new temporary directory. The release workflow then reran Ruff,
all tests, reference-manifest verification, notebook execution, source/wheel builds, and a headless
Streamlit launch against the extracted source. The archive scan found no bytecode, test/lint caches,
editable-install metadata, transient run directories, or build work trees.

The notebook uses a normal Jupyter kernel by default. Verification used its documented in-process
fallback because the packaging sandbox blocks local Jupyter kernel sockets; the same code cells ran
sequentially in one isolated Python process.

Created by School of AI and School of QC.
