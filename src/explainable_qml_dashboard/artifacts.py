"""Artifact serialization and integrity helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy
import pandas
import qiskit
import sklearn


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value.rstrip() + "\n", encoding="utf-8")


def save_joblib(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, target)


def environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
        "matplotlib": matplotlib.__version__,
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: str | Path) -> list[dict[str, str | int]]:
    base = Path(root)
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def verify_manifest(root: str | Path) -> list[str]:
    base = Path(root)
    manifest_path = base / "artifact_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        entries = json.load(handle)
    failures = []
    for entry in entries:
        path = base / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            failures.append(f"size mismatch: {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            failures.append(f"checksum mismatch: {entry['path']}")
    return failures
