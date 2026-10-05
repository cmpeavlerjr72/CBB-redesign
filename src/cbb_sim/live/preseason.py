"""Single source of the active 2026-27 preseason pull directory (`data/overrides/preseason_dir.json`)."""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
CONFIG = _REPO / "data/overrides/preseason_dir.json"


def preseason_dir() -> Path:
    name = json.loads(CONFIG.read_text(encoding="utf-8"))["preseason_dir"]
    return _REPO / "data/raw/preseason" / name


def preseason_rel() -> str:
    """Repo-relative posix path, for command lines."""
    return "data/raw/preseason/" + preseason_dir().name
