"""Export the bundled Iris dataset as a local CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from explainable_qml_dashboard.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/iris.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_dataset().frame.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
