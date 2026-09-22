# Self-assessment

## 100-point release rubric

### Scientific protocol — 20 points

- 5: Split is frozen, stratified, disjoint, and persisted.
- 5: Preprocessing is fitted on training data only.
- 5: Classical controls receive the same representation and training rows.
- 5: Claims distinguish simulator evidence from hardware evidence.

### Quantum implementation — 20 points

- 5: Circuit definition and readout are documented.
- 5: States, expectations, and probabilities satisfy numerical invariants.
- 5: Independent Qiskit parity passes within declared tolerance.
- 5: Analytic gradients agree with finite differences.

### Explainability — 20 points

- 5: Raw-to-angle encoding is visible.
- 5: Layer-level state/readout behavior is traceable.
- 5: Parameter and input gradients are distinguished and defined.
- 5: Prediction explanations include calibration and explicit limitations.

### Software quality — 20 points

- 5: Package, CLI, dashboard, notebook, and saved inference work.
- 5: Lint, format, tests, notebook, and builds pass.
- 5: Inputs and serialized artifacts have documented security boundaries.
- 5: A clean extracted archive reproduces the quality gates.

### Communication — 20 points

- 5: README explains the problem, architecture, setup, usage, and evidence.
- 5: Results report classical wins and negative findings honestly.
- 5: Model card, responsible-use guidance, and verification record are present.
- 5: Demo, resume, interview, contribution, and GitHub-release material is ready.

The release target is at least 90 points with no zero in any subsection.

## Knowledge check

1. Why must the angle scaler fit on the training partition only?
2. What quantity does a Z expectation represent in this classifier?
3. Why does the parameter-shift rule use two shifted circuit evaluations?
4. What does nonzero single-qubit entropy mean for a pure global state?
5. Why is input saliency not causal feature importance?
6. Why can a one-feature counterfactual be invalid in the real domain?
7. What does global-phase-invariant state fidelity protect against in parity testing?
8. Why are runtime numbers not a fair classical-versus-quantum speed benchmark here?
9. What additional evidence would be required for a quantum-advantage claim?
10. Why can reporting a classical win strengthen a QML portfolio project?

## Answer guide

1. Fitting it on validation or test values would leak evaluation-distribution information.
2. The difference between probabilities of measuring that qubit in zero and one.
3. Pauli-generated rotations have a sinusoidal expectation structure whose exact derivative is the
   half-difference at shifts of plus and minus π/2.
4. That qubit is entangled with the remaining subsystem; its reduced state is mixed.
5. It is a local model derivative and does not identify real-world interventions or causation.
6. Holding correlated features fixed may produce a combination absent from the data manifold.
7. Physically equivalent states can differ by a global complex phase even when amplitudes do not
   match element by element.
8. Implementations, languages, workloads, and resources are not matched; the QVC is simulated.
9. Repeated pre-registered comparisons, uncertainty, resource matching, multiple datasets,
   noise/hardware evidence, and a precise advantage definition.
10. It demonstrates disciplined controls, reproducibility, and the ability to learn from evidence
    instead of optimizing the story.

Created by School of AI and School of QC.

