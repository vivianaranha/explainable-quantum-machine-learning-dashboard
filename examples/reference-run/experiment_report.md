# Experiment report

Created by School of AI and School of QC.

## Protocol

Iris was split once with stratification into train, validation, and test partitions. Both preprocessing stages were fitted on train only. All four models saw the same angle-encoded inputs. The test partition stayed untouched until final evaluation.

## Frozen-test results

- **Random Forest** — accuracy 0.933, macro F1 0.933, log loss 0.216, ECE 0.071.
- **Rbf Svc** — accuracy 0.933, macro F1 0.933, log loss 0.147, ECE 0.116.
- **Logistic Regression** — accuracy 0.900, macro F1 0.900, log loss 0.216, ECE 0.126.
- **Variational Quantum Classifier** — accuracy 0.767, macro F1 0.764, log loss 0.716, ECE 0.313.

## Quantum audit

- Validation-selected epoch: 33; training stopped at epoch 43.
- Independent NumPy/Qiskit maximum parity infidelity: 1.554e-15.
- Reference explanation row: dataset row 7.
- Gradients are exact parameter-shift derivatives of the noiseless simulator.
- Input saliency is a local finite-difference sensitivity, not a causal effect.
- The counterfactual changes one feature inside the train-fitted angle range; it is a model probe, not a real-world recommendation.

## Conclusion

The best frozen-test macro F1 came from **random forest** at 0.933. The quantum-minus-best-classical difference was -0.169. This run does not establish quantum advantage; its value is the linked audit trail from input encoding through circuit behavior, optimization, and predictions.
