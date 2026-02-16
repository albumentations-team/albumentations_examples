---
name: notebook-metadata
description: Ensures Jupyter notebooks have required metadata and nbformat. Use when adding, creating, or modifying .ipynb files, or when the user asks about notebook structure or validation.
---

# Notebook Metadata and nbformat

## When to Apply

- Adding or creating new Jupyter notebooks
- Modifying notebook structure
- User asks about notebook validation or metadata

## Required Structure

Every `.ipynb` file must have these top-level keys:

| Key | Type | Notes |
|-----|------|-------|
| `metadata` | object | Can be empty `{}` but must exist |
| `nbformat` | int | Must be 4 or 5 |
| `nbformat_minor` | int | Minor version |
| `cells` | list | Array of cell objects |

## Minimal Valid Notebook

```json
{
  "cells": [],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 4
}
```

## Validation

Run the project script:

```bash
python scripts/check_notebook_metadata.py [path]
```

Omit path to check current directory recursively. Pre-commit runs this automatically on staged `.ipynb` files.

## Fixing Invalid Notebooks

If a notebook is missing fields, add them at the root level:

```python
import json
with open("notebook.ipynb") as f:
    nb = json.load(f)
nb.setdefault("metadata", {})
nb.setdefault("nbformat", 4)
nb.setdefault("nbformat_minor", 4)
nb.setdefault("cells", [])
with open("notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
```
