"""Remove only known local build and run outputs."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    targets = [ROOT / "build", ROOT / "dist", ROOT / ".pytest_cache", ROOT / ".ruff_cache"]
    for target in targets:
        if target.exists() and target.parent == ROOT:
            shutil.rmtree(target)
            print(f"Removed {target.relative_to(ROOT)}")
    artifacts = ROOT / "artifacts"
    for target in artifacts.iterdir():
        if target.name != ".gitkeep":
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            print(f"Removed artifacts/{target.name}")


if __name__ == "__main__":
    main()
