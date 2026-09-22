"""Small wrapper around the package's saved-inference API."""

from explainable_qml_dashboard.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["predict", *__import__("sys").argv[1:]]))
