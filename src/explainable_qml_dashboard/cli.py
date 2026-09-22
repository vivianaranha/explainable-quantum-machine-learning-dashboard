"""Command-line interface for experiments, explanations, and saved inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .artifacts import verify_manifest
from .config import ExperimentConfig
from .data import FEATURE_NAMES
from .experiment import run_experiment
from .inference import explain_raw_row, load_inference_bundle, predict_frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xqml-dashboard",
        description="Train, inspect, and use the Explainable QML Dashboard models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the complete benchmark.")
    run.add_argument("--config", default="configs/default.json", type=Path)
    run.add_argument("--output", default="artifacts/run", type=Path)
    run.add_argument("--no-plots", action="store_true")
    run.add_argument("--epochs", type=int, help="Override the configured epoch limit.")
    run.add_argument("--seed", type=int, help="Override the configured random seed.")

    predict = subparsers.add_parser("predict", help="Classify a CSV with saved models.")
    predict.add_argument("--run", default="examples/reference-run", type=Path)
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--model", default="variational_quantum_classifier")

    explain = subparsers.add_parser("explain", help="Explain one raw Iris row.")
    explain.add_argument("--run", default="examples/reference-run", type=Path)
    for index, name in enumerate(FEATURE_NAMES):
        explain.add_argument(f"--x{index}", type=float, required=True, help=name)

    verify = subparsers.add_parser("verify", help="Verify a run artifact manifest.")
    verify.add_argument("--run", default="examples/reference-run", type=Path)
    return parser


def _config_with_overrides(args: argparse.Namespace) -> ExperimentConfig:
    values = ExperimentConfig.from_json(args.config).to_dict()
    if args.epochs is not None:
        values["epochs"] = args.epochs
    if args.seed is not None:
        values["random_seed"] = args.seed
    return ExperimentConfig.from_dict(values)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_experiment(
                _config_with_overrides(args), args.output, make_plots=not args.no_plots
            )
            print(f"Run complete: {result.output_directory}")
            print(result.metrics.to_string(index=False))
            return 0
        if args.command == "predict":
            bundle = load_inference_bundle(args.run)
            input_frame = pd.read_csv(args.input)
            output_frame = predict_frame(bundle, input_frame, model_name=args.model)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_frame.to_csv(args.output, index=False)
            print(f"Wrote {len(output_frame)} predictions to {args.output}")
            return 0
        if args.command == "explain":
            bundle = load_inference_bundle(args.run)
            explanation = explain_raw_row(bundle, [args.x0, args.x1, args.x2, args.x3])
            serializable = {
                "prediction": explanation["prediction"],
                "prediction_name": explanation["prediction_name"],
                "probabilities": explanation["probabilities"].tolist(),
                "angles": explanation["angles"].tolist(),
                "saliency": explanation["saliency"].to_dict(orient="records"),
                "counterfactual": explanation["counterfactual"],
            }
            print(json.dumps(serializable, indent=2))
            return 0
        failures = verify_manifest(args.run)
        if failures:
            print("Manifest verification failed:", file=sys.stderr)
            print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
            return 1
        print(f"Verified all listed artifacts in {args.run}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
