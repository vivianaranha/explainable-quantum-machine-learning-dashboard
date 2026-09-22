"""Execute and persist the quickstart notebook for release verification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "quickstart.ipynb"


def _execute_in_process(notebook: nbformat.NotebookNode) -> None:
    """Execute standard-Python cells when a sandbox blocks Jupyter kernel sockets."""

    namespace = {"__name__": "__main__"}
    original_directory = Path.cwd()
    os.chdir(ROOT)
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                exec(compile(cell.source, f"quickstart-cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Execute code cells in one Python process when Jupyter sockets are unavailable.",
    )
    args = parser.parse_args()
    with NOTEBOOK.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    for index, cell in enumerate(notebook.cells):
        cell["id"] = cell.get("id", f"xqml-{index:02d}")
    if args.in_process:
        _execute_in_process(notebook)
    else:
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
    with NOTEBOOK.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    print(f"Executed {NOTEBOOK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
