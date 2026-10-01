#!/usr/bin/env python
"""build_foul_state_v4_v1.py -- the corrected team-foul state on the EVENT LAYER v4 machine (full-retrain chain).

Versioned sibling of `scripts/build_foul_accrual_design_v2.py` (lane A, NOT edited) and of
`scripts/build_po_in_bonus_overlay_v1.py` (round 8, NOT edited). Join point:
`docs/tests/event_layer_v4_2026-09-30.md` section 5.

What it does, and nothing else:
  1. runs lane A's replay (`build_foul_accrual_design_v2.main`, imported, unedited) with the
     possession machine built as `_GameMachine(gm, sub, tech_lookahead=..., **VERSION_EVENT_FIXES[v])`
     for `--machine v` (v2 = the served machine, v4 = event layer v4). The machine is swapped by
     subclassing `cbb_sim.pbp.possessions._GameMachine` in every worker process; the replay code
     itself is lane A's;
  2. writes `foul_accrual_poss.parquet` (+ `machine_version`) and `build_report.json` into a NEW
     `--out-dir` (never the round6 directory);
  3. `--overlay-design <PO design parquet>` additionally writes `in_bonus_overlay.parquet`: round 8's
     recipe (open count minus the possession's own pre-open fouls, plus the defence's trip fouls on
     EARLIER chances, >= 6), keyed (game_id, poss_index, chance_number);
  4. runs the join-point checks of section 5 (anti-join both ways against the possessions table of
     the same machine; period / offence / start_clock equal; per-game foul totals equal across
     machines when `--compare-to` names another build).

Identity: `--machine v2` must reproduce `round6/foul_accrual_poss_v2.parquet` (and, with
`--overlay-design round2/design.parquet`, `round8/in_bonus_overlay_v2state_v1.parquet`) exactly:
    .venv/Scripts/python.exe scripts/build_foul_state_v4_v1.py --machine v2 --out-dir <scratch> \
        --overlay-design data/processed/models/possession_outcome/round2/design.parquet --identity
Full-retrain use (v4):
    .venv/Scripts/python.exe scripts/build_foul_state_v4_v1.py --machine v4 --out-dir <root>/foul_state \
        --overlay-design <root>/po_design/design.parquet --workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

SERVED_STATE = ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet"
SERVED_OVERLAY = ROOT / "data/processed/models/possession_outcome/round8/in_bonus_overlay_v2state_v1.parquet"


def _install_machine(machine: str) -> None:
    """Make `cbb_sim.pbp.possessions._GameMachine` build the `machine` version (in THIS process)."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    from cbb_sim.pbp import possessions as PZ
    if getattr(PZ, "_fsv4_machine", None) == machine:
        return
    base = getattr(PZ, "_fsv4_base", PZ._GameMachine)
    kw = dict(tech_lookahead=PZ.VERSION_TECH_LOOKAHEAD[machine], **PZ.VERSION_EVENT_FIXES[machine])

    class _Versioned(base):                                      # noqa: D401
        def __init__(self, game_meta, ev, **over):
            super().__init__(game_meta, ev, **{**kw, **over})

    _Versioned.__name__ = "_GameMachine"
    PZ._fsv4_base = base
    PZ._fsv4_machine = machine
    PZ._GameMachine = _Versioned


def _pool_factory(machine: str, workers: int):
    def _ppe(max_workers=None, **_kw):                            # lane A passes max_workers=4
        return ProcessPoolExecutor(max_workers=workers, initializer=_install_machine,
                                   initargs=(machine,))
    return _ppe


def build_state(machine: str, out_dir: Path, seasons: list[int], workers: int) -> Path:
    import pandas as pd
    import build_foul_accrual_design_v2 as A

    out_dir.mkdir(parents=True, exist_ok=True)
    A.OUT_DIR = out_dir
    A.SEASONS = list(seasons)
    A.ProcessPoolExecutor = _pool_factory(machine, workers)
    _install_machine(machine)                      # the parent too (main() imports thresholds only)
    A.main()
    src = out_dir / "foul_accrual_poss_v2.parquet"   # lane A's fixed file name: the DESIGN version
    df = pd.read_parquet(src)
    df["machine_version"] = machine
    dst = out_dir / "foul_accrual_poss.parquet"
    df.to_parquet(dst, index=False)
    src.unlink()
    rep_src = out_dir / "build_report_v2.json"
    rep = json.loads(rep_src.read_text())
    rep_src.unlink()
    rep.update({"machine_version": machine, "builder": "build_foul_state_v4_v1.py",
                "wrapped": "build_foul_accrual_design_v2.py (unedited)"})
    (out_dir / "build_report.json").write_text(json.dumps(rep, indent=1))
    return dst


def build_overlay(state_path: Path, design_path: Path, out_path: Path) -> dict:
    """Round 8's recipe (`build_po_in_bonus_overlay_v1.py`), with the paths as parameters."""
    import pandas as pd
    ch = pd.read_parquet(design_path, columns=["game_id", "season", "poss_index", "chance_number",
                                               "terminal_event", "and_one", "in_bonus"])
    st = pd.read_parquet(state_path, columns=["game_id", "season", "poss_index", "def_team_fouls_true"])
    d = ch.merge(st, on=["game_id", "season", "poss_index"], how="left", validate="many_to_one")
    d = d.sort_values(["game_id", "poss_index", "chance_number"])
    trip = (d["terminal_event"].isin(["FT_trip_shooting", "FT_trip_bonus"]).astype(int)
            + d["and_one"].astype(int))
    prev = trip.groupby([d["game_id"], d["poss_index"]]).cumsum() - trip
    live = d["def_team_fouls_true"] + prev
    d["in_bonus_new"] = (live >= 6).astype("float32").where(d["def_team_fouls_true"].notna())
    o = d[["game_id", "poss_index", "chance_number"]].copy()
    o["in_bonus"] = d["in_bonus_new"]
    assert not o.duplicated(["game_id", "poss_index", "chance_number"]).any()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out_path, index=False)
    chg = (d["in_bonus_new"] != d["in_bonus"]) & d["in_bonus_new"].notna()
    return {"rows": int(len(o)), "unmatched_nan": int(o["in_bonus"].isna().sum()),
            "changed": int(chg.sum()), "changed_pct": round(100 * float(chg.mean()), 3),
            "one_to_zero": int(((d["in_bonus"] == 1) & (d["in_bonus_new"] == 0)).sum()),
            "zero_to_one": int(((d["in_bonus"] == 0) & (d["in_bonus_new"] == 1)).sum())}


def join_checks(state_path: Path, machine: str, seasons: list[int], compare_to: Path | None) -> dict:
    import pandas as pd
    from cbb_sim.pbp.possessions import possessions_dir
    st = pd.read_parquet(state_path)
    pz = pd.concat([pd.read_parquet(possessions_dir(machine) / f"possessions_{s}.parquet",
                                    columns=["game_id", "season", "poss_index", "period", "start_clock",
                                             "offense_team_id"]) for s in seasons], ignore_index=True)
    # the possessions table is restricted to the same universe filter as the replay (D-I, not truncated)
    k = ["game_id", "season", "poss_index"]
    m = st[k + ["period", "start_clock", "offense_team_id"]].merge(pz, on=k, how="outer",
                                                                    indicator=True, suffixes=("", "_p"))
    both = m["_merge"] == "both"
    out = {"machine": machine, "state_rows": int(len(st)), "poss_rows": int(len(pz)),
           "state_only": int((m["_merge"] == "left_only").sum()),
           "poss_only": int((m["_merge"] == "right_only").sum()),
           "poss_only_games": int(m.loc[m["_merge"] == "right_only", "game_id"].nunique()),
           "period_mismatch": int((m.loc[both, "period"] != m.loc[both, "period_p"]).sum()),
           "start_clock_mismatch": int((m.loc[both, "start_clock"] != m.loc[both, "start_clock_p"]).sum()),
           "offense_mismatch": int((m.loc[both, "offense_team_id"] != m.loc[both, "offense_team_id_p"]).sum())}
    if compare_to is not None:
        o = pd.read_parquet(compare_to)
        cols = ["def_silent", "def_trip", "off_silent", "off_trip"]
        a = st.groupby("game_id")[cols].sum()
        b = o.groupby("game_id")[cols].sum()
        j = a.join(b, rsuffix="_o", how="inner")
        out["per_game_foul_totals_equal_share"] = round(float(
            (j[cols].to_numpy() == j[[c + "_o" for c in cols]].to_numpy()).all(axis=1).mean()), 6)
        out["per_game_total_fouls_equal_share"] = round(float(
            (j[cols].sum(axis=1).to_numpy() == j[[c + "_o" for c in cols]].sum(axis=1).to_numpy()).mean()), 6)
        out["games_compared"] = int(len(j))
    return out


def identity(out_dir: Path, overlay_out: Path | None) -> dict:
    import pandas as pd
    new = pd.read_parquet(out_dir / "foul_accrual_poss.parquet").drop(columns=["machine_version"])
    old = pd.read_parquet(SERVED_STATE)
    res = {"state_equal": bool(new.equals(old)), "state_rows": [int(len(new)), int(len(old))]}
    if not res["state_equal"] and list(new.columns) == list(old.columns) and len(new) == len(old):
        res["state_cols_differing"] = [c for c in new.columns if not new[c].equals(old[c])]
    if overlay_out is not None:
        a, b = pd.read_parquet(overlay_out), pd.read_parquet(SERVED_OVERLAY)
        res["overlay_equal"] = bool(a.equals(b))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["v2", "v4", "v4otc"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seasons", type=int, nargs="*", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--overlay-design", type=Path, default=None,
                    help="PO design parquet (same machine) to key the in_bonus overlay on")
    ap.add_argument("--compare-to", type=Path, default=None,
                    help="another machine's foul_accrual_poss parquet: per-game foul totals must match")
    ap.add_argument("--identity", action="store_true", help="--machine v2 only: compare to the served files")
    a = ap.parse_args()
    if a.out_dir.resolve() in (SERVED_STATE.parent.resolve(), SERVED_OVERLAY.parent.resolve()):
        raise SystemExit("refusing to write into a served round directory")
    if a.out_dir.exists() and any(a.out_dir.iterdir()) and not (a.out_dir / "build_report.json").exists():
        raise SystemExit(f"{a.out_dir} exists and is not a build of this script")
    t0 = time.time()
    st = build_state(a.machine, a.out_dir, a.seasons, a.workers)
    rep = {"machine": a.machine, "state": str(st), "seconds_state": round(time.time() - t0, 1)}
    ov = None
    if a.overlay_design is not None:
        ov = a.out_dir / "in_bonus_overlay.parquet"
        rep["overlay"] = build_overlay(st, a.overlay_design, ov)
        rep["overlay_design"] = str(a.overlay_design)
    rep["join_checks"] = join_checks(st, a.machine, a.seasons, a.compare_to)
    if a.identity:
        if a.machine != "v2":
            raise SystemExit("--identity compares against the served v2 files; use --machine v2")
        rep["identity"] = identity(a.out_dir, ov)
    rep["seconds_total"] = round(time.time() - t0, 1)
    (a.out_dir / "foul_state_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    if a.identity and not (rep["identity"]["state_equal"] and rep["identity"].get("overlay_equal", True)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
