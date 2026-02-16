#!/usr/bin/env python3
"""Validate Jupyter notebooks have required metadata and nbformat fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_notebook(path: Path) -> list[str]:
    """Return list of validation errors for a notebook, empty if valid."""
    errors: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    except OSError as e:
        return [f"Cannot read: {e}"]

    if "metadata" not in nb:
        errors.append("Missing top-level 'metadata'")
    elif not isinstance(nb["metadata"], dict):
        errors.append("'metadata' must be a dict")

    if "nbformat" not in nb:
        errors.append("Missing 'nbformat'")
    elif nb["nbformat"] not in (4, 5):
        errors.append(f"'nbformat' must be 4 or 5, got {nb['nbformat']}")

    if "nbformat_minor" not in nb:
        errors.append("Missing 'nbformat_minor'")
    elif not isinstance(nb["nbformat_minor"], int):
        errors.append("'nbformat_minor' must be an integer")

    if "cells" not in nb:
        errors.append("Missing 'cells'")
    elif not isinstance(nb["cells"], list):
        errors.append("'cells' must be a list")

    return errors


def main() -> int:
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["."]
    all_errors: dict[Path, list[str]] = {}

    for p in paths:
        path = Path(p).resolve()
        if path.is_dir():
            for nb_path in path.rglob("*.ipynb"):
                errs = check_notebook(nb_path)
                if errs:
                    all_errors[nb_path] = errs
        elif path.suffix == ".ipynb":
            errs = check_notebook(path)
            if errs:
                all_errors[path] = errs
        else:
            print(f"Warning: skipping non-notebook {path}", file=sys.stderr)

    if not all_errors:
        return 0

    for nb_path, errs in sorted(all_errors.items()):
        rel = nb_path.relative_to(Path.cwd()) if nb_path.is_relative_to(Path.cwd()) else nb_path
        print(f"{rel}:")
        for e in errs:
            print(f"  - {e}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
