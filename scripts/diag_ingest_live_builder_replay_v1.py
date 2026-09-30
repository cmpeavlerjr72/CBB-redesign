#!/usr/bin/env python
"""
diag_ingest_live_builder_replay_v1.py -- run scripts/build_engine_inputs_live.py against a DATA ROOT and compare.

    # build from a root (scratch or the repo) ; cwd is switched to the root so every relative data path follows it
    .venv/Scripts/python.exe scripts/diag_ingest_live_builder_replay_v1.py run --root <root> --slate-date 2025-01-16 \
        --as-of 2025-01-16T12:00:00Z --out-dir <dir> [--season 2025 --fold F2]
    # compare two output directories array by array
    .venv/Scripts/python.exe scripts/diag_ingest_live_builder_replay_v1.py compare --a <dirA> --b <dirB> --tag LIVE_F2_2025_2025-01-16

The builder module keeps four table paths as absolute constants under the repo (FG_EVENTS, RB_EVENTS, FT_ATTEMPTS,
USAGE_EVENTS_V2) plus UNIVERSE; they are re-pointed to the root here (module attributes only, no file edited).
Every file the run opens under the repo's data/ tree but outside the root is reported (static inputs only expected).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def cmd_run(a) -> int:
    root = Path(a.root).resolve()
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    opened: set = set()

    def hook(ev, args):
        if ev == "open" and isinstance(args[0], str):
            p = os.path.abspath(args[0])
            if ("\\data\\" in p or "/data/" in p) and (args[1] in (None, "r", "rb")):
                opened.add(p)

    sys.addaudithook(hook)
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    a.schedule_path, a.crosswalk = str(Path(a.schedule_path).resolve()), str(Path(a.crosswalk).resolve())
    os.chdir(root)
    import build_engine_inputs_live as BL
    if root != REPO:
        BL.FG_EVENTS = root / "data/processed/models/fg_make/events_v2_shotshooter.parquet"
        BL.RB_EVENTS = root / "data/processed/models/rebound/events_v1.parquet"
        BL.FT_ATTEMPTS = root / "data/processed/models/free_throw/attempts_v1_era.parquet"
        BL.USAGE_EVENTS_V2 = root / "data/processed/models/usage_v2/events_v2_shotshooter.parquet"
        BL.UNIVERSE = root / "data/processed/games_universe.parquet"
    sys.argv = ["build_engine_inputs_live.py", "--slate-date", a.slate_date, "--as-of", a.as_of, "--season", str(a.season),
                "--fold", a.fold, "--schedule-source", "cbbd", "--schedule-path", str(Path(a.schedule_path).resolve()),
                "--crosswalk", str(Path(a.crosswalk).resolve()), "--out-dir", str(out_dir), "--created-at", a.created_at]
    rc = BL.main()
    # the slate itself comes from the games universe; under the root the slate day is not in the universe, so the
    # caller passes --slate-from-repo-universe (games table from the repo universe, result columns dropped by the builder)
    outside = sorted(p for p in opened if not p.lower().startswith(str(root).lower()) and "CBB-clean-sheet" in p
                     and "\\src\\" not in p and ".venv" not in p)
    (out_dir / "files_opened_outside_root.json").write_text(json.dumps(outside, indent=2), encoding="utf-8")
    return rc


def cmd_compare(a) -> int:
    import numpy as np
    import pandas as pd
    A, B = Path(a.a), Path(a.b)
    res: dict = {}
    g1, g2 = pd.read_parquet(A / f"games_{a.tag}.parquet"), pd.read_parquet(B / f"games_{a.tag}.parquet")
    res["games_equal"] = bool(g1.drop(columns=["created_at"], errors="ignore").equals(g2.drop(columns=["created_at"], errors="ignore")))
    z1, z2 = np.load(A / f"arrays_{a.tag}.npz", allow_pickle=True), np.load(B / f"arrays_{a.tag}.npz", allow_pickle=True)
    diffs, n_cells = {}, 0
    for k in sorted(set(z1.files) | set(z2.files)):
        if k not in z1.files or k not in z2.files:
            diffs[k] = "missing in one side"
            continue
        x, y = z1[k], z2[k]
        if x.shape != y.shape:
            diffs[k] = f"shape {x.shape} vs {y.shape}"
            continue
        n_cells += x.size
        if x.dtype.kind in "fc":
            ne = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        else:
            ne = x != y
        if ne.any():
            diffs[k] = int(ne.sum())
    res["arrays_compared"] = len(z1.files)
    res["cells_compared"] = int(n_cells)
    res["arrays_differing"] = diffs
    n1, n2 = json.loads((A / f"names_{a.tag}.json").read_text(encoding="utf-8")), json.loads((B / f"names_{a.tag}.json").read_text(encoding="utf-8"))
    res["names_equal"] = n1 == n2
    e1, e2 = np.load(A / f"event_block_{a.tag}.npz", allow_pickle=True), np.load(B / f"event_block_{a.tag}.npz", allow_pickle=True)
    res["event_block_equal"] = bool(np.array_equal(e1["team_block"], e2["team_block"], equal_nan=True))
    res["identical"] = bool(res["games_equal"] and not diffs and res["names_equal"] and res["event_block_equal"])
    print(json.dumps(res, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True); r.add_argument("--slate-date", required=True); r.add_argument("--as-of", required=True)
    r.add_argument("--out-dir", required=True); r.add_argument("--season", type=int, default=2025); r.add_argument("--fold", default="F2")
    r.add_argument("--created-at", default=None)
    r.add_argument("--schedule-path", default=str(REPO / "data/raw/cbbd/games_2025.parquet"))
    r.add_argument("--crosswalk", default=str(REPO / "data/reference/team_crosswalk.parquet"))
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True); c.add_argument("--b", required=True); c.add_argument("--tag", required=True)
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
