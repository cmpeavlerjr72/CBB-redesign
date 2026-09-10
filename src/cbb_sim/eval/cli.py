"""cli.py -- tiny shared helpers for the eval/market/props CLI scripts."""

from __future__ import annotations

from pathlib import Path

import yaml


def tag_from_results_dir(results_dir: Path) -> str:
    """`results/control/F2_A_own` -> `control_F2_A_own`, used to name the
    output report file `docs/tests/<report>_<tag>_<date>.md`."""
    results_dir = Path(results_dir)
    try:
        rel = results_dir.resolve().relative_to(Path("results").resolve())
        return str(rel).replace("\\", "_").replace("/", "_")
    except ValueError:
        return results_dir.name


def load_tolerances(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
