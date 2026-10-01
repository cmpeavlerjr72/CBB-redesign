#!/usr/bin/env python
"""
build_ot_carry_designs_v1.py -- the fg_make, rebound and free-throw training tables on the OVERTIME TEAM-FOUL CARRY
(lane F, 2026-10-01). Versioned siblings under --out-root; nothing existing is overwritten; 2026 (sealed) is never built.

    .venv/Scripts/python.exe scripts/build_ot_carry_designs_v1.py --out-root data/processed/models/otc_designs [--identity]

Per consumer (event layer builders are the canonical ones `pull_daily_ingest_v1.rebuild_derived` calls):
  fg_make   FG.build_fg_events(version v2, shooter_key shot_shooter_id)  -> fg_make/events_v2_shotshooter.parquet
            FG.build_design(...)                                          -> fg_make/design_v2_shotshooter.parquet
  rebound   RB.build_rebound_events(version v1)                           -> rebound/events_v1.parquet
            scripts/build_rebound_round3_design_v1.py (env-redirected)    -> rebound/round3/design_round3.parquet
  FT        FT.build_trips_and_attempts(version v1)                       -> free_throw/attempts_v1_era.parquet
The carry is `CBB_OT_FOUL_CARRY=1` in `cbb_sim.models.event_stream._attach_team_fouls` (default off).

Proof written to <out-root>/build_report.json:
  * IDENTITY (default path, env unset; --identity runs only this): the rebuilt events / attempts equal the stored files on the
    2022-2025 rows (DataFrame.equals), so the carry run starts from a reproduction of what the served trainers read;
  * REGULATION: carry vs stored rows with period <= 2 are bit-identical; OT rows differing are counted per column.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
M = ROOT / "data/processed/models"
STORED = {"fg_events": M / "fg_make/events_v2_shotshooter.parquet", "fg_design": M / "fg_make/design_v2_shotshooter.parquet",
          "rb_events": M / "rebound/events_v1.parquet", "rb_design": M / "rebound/round3/design_round3.parquet",
          "ft_attempts": M / "free_throw/attempts_v1_era.parquet"}
FOUL_COLS = ("off_in_bonus", "off_in_double_bonus", "in_bonus", "prior_fouls", "trip_prior_fouls", "foul_class", "n_ft", "is_bonus", "bonus_flag")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def seasons_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["season"].isin(SEASONS)].reset_index(drop=True)


def compare(stored: pd.DataFrame, new: pd.DataFrame) -> dict:
    a, b = seasons_only(stored), new.reset_index(drop=True)
    out = {"rows_stored": int(len(a)), "rows_new": int(len(b)), "columns_equal": list(a.columns) == list(b.columns)}
    if not out["columns_equal"]:
        out["only_stored"] = sorted(set(a.columns) - set(b.columns)); out["only_new"] = sorted(set(b.columns) - set(a.columns))
        common = [c for c in a.columns if c in b.columns]
        a, b = a[common], b[common]
    out["bit_identical"] = bool(len(a) == len(b) and a.equals(b))
    if len(a) == len(b) and not out["bit_identical"]:        # (equals() is also False on a dtype-only difference: see differing_columns)
        diff = {}
        for c in a.columns:
            x, y = a[c], b[c]
            ne = ~((x == y) | (x.isna() & y.isna())).to_numpy()
            if ne.any():
                diff[c] = int(ne.sum())
        out["differing_columns"] = diff
        out["value_identical"] = not diff
        per = a["period"].to_numpy() if "period" in a.columns else None
        if per is not None:
            anyd = np.zeros(len(a), bool)
            for c in diff:
                x, y = a[c], b[c]
                anyd |= ~((x == y) | (x.isna() & y.isna())).to_numpy()
            out["regulation_rows"] = int((per <= 2).sum())
            out["regulation_rows_differing"] = int((anyd & (per <= 2)).sum())
            out["ot_rows"] = int((per > 2).sum())
            out["ot_rows_differing"] = int((anyd & (per > 2)).sum())
    return out


def build_all(carry: bool, t0: float) -> dict:
    if carry:
        os.environ[ES.OT_FOUL_CARRY_ENV] = "1"
    else:
        os.environ.pop(ES.OT_FOUL_CARRY_ENV, None)
    u_pc = ES.load_universe(require_pbp_complete=True)
    u = ES.load_universe()
    res = {}
    res["fg_events"] = FG.build_fg_events(SEASONS, universe=u_pc, version="v2", shooter_key="shot_shooter_id")
    log(f"carry={carry}: fg events {len(res['fg_events']):,}", t0)
    res["rb_events"] = RB.build_rebound_events(SEASONS, universe=u, version="v1")
    log(f"carry={carry}: rebound events {len(res['rb_events']):,}", t0)
    _, att = FT.build_trips_and_attempts(SEASONS, universe=u, version="v1")
    res["ft_attempts"] = att
    log(f"carry={carry}: ft attempts {len(att):,}", t0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--identity", action="store_true", help="only the default-path identity check (env unset), write nothing")
    ap.add_argument("--only", default="", help="comma subset of fg,rb,ft")
    a = ap.parse_args()
    t0 = time.time()
    out = a.out_root if a.out_root.is_absolute() else ROOT / a.out_root
    rep = {"seasons": SEASONS, "built_at": time.strftime("%Y-%m-%d %H:%M")}
    base = build_all(False, t0)
    rep["identity_default_path_vs_stored"] = {k: compare(pd.read_parquet(STORED[k]), v) for k, v in base.items()}
    for k, v in rep["identity_default_path_vs_stored"].items():
        log(f"IDENTITY {k}: bit_identical={v['bit_identical']} rows {v['rows_stored']}/{v['rows_new']}", t0)
    if a.identity:
        (ROOT / "results/ot_carry_designs_identity.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        return 0
    new = build_all(True, t0)
    rep["carry_vs_default_rebuild"] = {k: compare(base[k], new[k]) for k in new}
    for sub in ("fg_make", "rebound/round3", "free_throw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    new["fg_events"].to_parquet(out / "fg_make/events_v2_shotshooter.parquet", index=False)
    new["rb_events"].to_parquet(out / "rebound/events_v1.parquet", index=False)
    new["ft_attempts"].to_parquet(out / "free_throw/attempts_v1_era.parquet", index=False)
    os.environ[ES.OT_FOUL_CARRY_ENV] = "1"
    uni = ES.load_universe(require_pbp_complete=True)
    d = FG.build_design(SEASONS, universe=uni, version="v2", events=new["fg_events"])
    d.to_parquet(out / "fg_make/design_v2_shotshooter.parquet", index=False)
    log(f"fg design {len(d):,} rows", t0)
    env = {**os.environ, "CBB_RB_R3_IN_EVENTS": str(out / "rebound/events_v1.parquet"), "CBB_RB_R3_OUT_DIR": str(out / "rebound/round3")}
    subprocess.run([sys.executable, str(ROOT / "scripts/build_rebound_round3_design_v1.py")], check=True, env=env, cwd=ROOT)
    log("rebound round3 design built", t0)
    for key, p in (("fg_design", out / "fg_make/design_v2_shotshooter.parquet"), ("rb_design", out / "rebound/round3/design_round3.parquet")):
        rep.setdefault("carry_design_vs_stored", {})[key] = compare(pd.read_parquet(STORED[key]), pd.read_parquet(p))
    (out / "build_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log("done", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
