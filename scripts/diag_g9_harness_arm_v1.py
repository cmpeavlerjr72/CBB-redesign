"""
diag_g9_harness_arm_v1.py -- lane A day 2026-10-01 (docs/models/aggregation/experiments.md section 3):
the deterministic expected-points harness of `diag_aggregation_harness_v1.py` with fg_make served from an
ARM's artifact directory (a `train_fg_make_v4_par_g9_v1.py` output), every other sub-model as served.

Read-only on the engine: the inputs are loaded from a tagged input dir (`engine_v3_S0_laneA` for served-feature
arms = served stack v2's arrays; `engine_v3_X_Tfix_laneA` for E3-feature arms), the arm's fg_make is loaded through
the engine's own `FgMakeAdapter._load_dated`, and any extra team feature the arm was trained on is appended
IN MEMORY to this process's `team_static` (never written to disk) with the class alias the adapter already uses
for `off_make_c` (`<name>__<class>`). Train/serve parity of every team and slot feature the arm reads is checked
against the arm's own test-row design values before the harness runs (`--parity`), and the run refuses on skew.

    .venv/Scripts/python.exe scripts/diag_g9_harness_arm_v1.py --arm-dir data/processed/models/fg_make/round_g9/G3R_s0_F2/B1 \
        --base S0 --name G3R_s0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diag_aggregation_harness_v1 as H  # noqa: E402

OUT = ROOT / "results" / "g9_team_response_v1"
BASES = {"S0": "data/processed/models/engine_v3_S0_laneA", "X_Tfix": "data/processed/models/engine_v3_X_Tfix_laneA"}
CLS_KEY = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}
E3_PREFIX = {"rim": "make_rim", "jump2": "make_jump", "three": "make3"}


def extra_columns(inp, needed: list[str], e3_table: Path | None, design_preds: Path | None) -> dict:
    """values per (game, side, class) of each extra team feature, from the same source the trainer used."""
    gid = inp.games["game_id"].to_numpy()
    G = len(gid)
    home = inp.games["home_team_id"].to_numpy() if "home_team_id" in inp.games else None
    away = inp.games["away_team_id"].to_numpy() if "away_team_id" in inp.games else None
    out = {}
    if {"off_make_v", "def_allow_v"} & set(needed):
        t = pd.read_parquet(e3_table)
        t = t[t["fold"] == "F2"].set_index(["game_id", "team_id"])
        for k, pre in E3_PREFIX.items():
            for name, col, own in (("off_make_v", f"{pre}_off_v", True), ("def_allow_v", f"{pre}_def_v", False)):
                arr = np.full((G, 2), np.nan)
                for side in (0, 1):
                    # side s is the OFFENCE in its own row; the defence is the other team
                    team = (home if side == 0 else away) if own else (away if side == 0 else home)
                    idx = pd.MultiIndex.from_arrays([gid, team])
                    arr[:, side] = t[col].reindex(idx).to_numpy(float)
                out[f"{name}__{k}"] = arr
    if {"off_att_prior", "def_att_prior"} & set(needed):
        d = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet",
                            columns=["game_id", "off_team_id", "def_team_id", "shot_class", "off_att_prior",
                                     "def_att_prior", "season"])
        d = d[d["season"] == 2025]
        for cls, k in CLS_KEY.items():
            s = d[d["shot_class"] == cls]
            off = s.groupby(["game_id", "off_team_id"])["off_att_prior"].first()
            dfn = s.groupby(["game_id", "def_team_id"])["def_att_prior"].first()
            for name, ser, own in (("off_att_prior", off, True), ("def_att_prior", dfn, False)):
                arr = np.full((G, 2), np.nan)
                for side in (0, 1):
                    team = (home if side == 0 else away) if own else (away if side == 0 else home)
                    arr[:, side] = ser.reindex(pd.MultiIndex.from_arrays([gid, team])).to_numpy(float)
                out[f"{name}__{k}"] = arr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True, type=Path)
    ap.add_argument("--base", choices=list(BASES), default="S0")
    ap.add_argument("--name", required=True)
    ap.add_argument("--arms", default="FULL,FG,RAT")
    ap.add_argument("--e3-table", type=Path, default=ROOT / "data/processed/team_rate_features_E3_v4.parquet")
    a = ap.parse_args()
    t0 = time.time()
    from run_engine_live import prepare_from_overlay
    from cbb_sim.engine import adapters as AD
    from cbb_sim.engine.inputs import EngineInputs
    sdir = ROOT / BASES[a.base]
    over = prepare_from_overlay(sdir / "overlay", "F2", 2025, ROOT / "results/aggregation_v1/_adapter_dirs" / f"harness_{a.base}")
    for k, v in over.items():
        setattr(AD, k, Path(v))
    inp = EngineInputs.load(sdir, "F2_2025")
    feats = json.loads((a.arm_dir / "manifest_FGA_rim.json").read_text(encoding="utf-8"))["features"]
    known = set(inp.team_names) | set(inp.slot_names) | set(AD.STATE_INDEX) | set(AD.FG_TEAM_ALIAS) | set(AD.FG_SLOT_ALIAS)
    needed = [f for f in feats if f not in known]
    report = {"name": a.name, "arm_dir": str(a.arm_dir), "base": a.base, "features": feats, "extra": needed}
    if needed:
        ex = extra_columns(inp, needed, a.e3_table, None)
        missing = {k: int(np.isnan(v).sum()) for k, v in ex.items()}
        report["extra_missing_cells"] = missing
        ts = inp.team_static
        names = dict(inp.team_names)
        add = []
        for k, v in ex.items():
            names[k] = ts.shape[2] + len(add)
            add.append(np.nan_to_num(v, nan=0.0))
        inp.team_static = np.concatenate([ts, np.stack(add, axis=2).astype(ts.dtype)], axis=2)
        inp.team_names = names
        for f in needed:
            AD.FG_TEAM_ALIAS[f] = f + "__{k}"
        print(f"appended {len(add)} extra team columns in memory; NaN cells {missing}", flush=True)
    # ---- train/serve parity: every team feature and the shooter dev, engine value vs the arm's design value
    pp = sorted(a.arm_dir.parent.glob("preds_F2_s*.parquet"))
    if not pp:
        raise SystemExit(f"no preds_F2_s*.parquet next to {a.arm_dir}; parity cannot be checked")
    pr = pd.read_parquet(pp[0])
    gpos = {int(g): i for i, g in enumerate(inp.games["game_id"].to_numpy())}
    hid = inp.games["home_team_id"].to_numpy()
    par = {}
    team_feats = [f for f in feats if f in ("off_make_c", "def_allow_c") or f in needed]
    ros = inp.roster_cbbd
    for cls, k in CLS_KEY.items():
        s = pr[(pr["shot_class"] == cls) & pr["game_id"].isin(gpos)]
        gi = s["game_id"].map(gpos).to_numpy()
        side = np.where(s["off_team_id"].to_numpy() == hid[gi], 0, 1)
        for f in team_feats:
            col = inp.team_names[f + "__" + k] if (f + "__" + k) in inp.team_names else inp.team_names[f]
            ev = inp.team_static[gi, side, col].astype(float)
            dv = s[f].to_numpy(float)
            dif = np.abs(ev - dv)
            par[f"{f}|{k}"] = {"n": int(len(s)), "max_abs_diff": float(np.nanmax(dif)),
                               "share_gt_1e-4": float(np.mean(dif > 1e-4))}
        # shooter dev: slot value of the shooter's roster slot
        j = inp.slot_names[f"shooter_shrunk_dev_c__{k}"]
        sh = s["shooter_id"].to_numpy()
        ev = np.full(len(s), np.nan)
        for r_ in range(len(s)):
            hit = np.flatnonzero(ros[gi[r_], side[r_]] == sh[r_])
            if len(hit):
                ev[r_] = inp.slot_static[gi[r_], side[r_], hit[0], j]
        ok = np.isfinite(ev)
        dif = np.abs(ev[ok] - s["shooter_shrunk_dev_c"].to_numpy(float)[ok])
        par[f"shooter_shrunk_dev_c|{k}"] = {"n_matched": int(ok.sum()), "n": int(len(s)),
                                            "max_abs_diff": float(dif.max()), "share_gt_1e-4": float(np.mean(dif > 1e-4))}
    report["parity"] = par
    bad = {k: v for k, v in par.items() if v["share_gt_1e-4"] > 0.001}
    print("parity:", json.dumps(par), flush=True)
    if bad:
        (OUT / f"harness_{a.name}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        raise SystemExit(f"TRAIN/SERVE SKEW: {list(bad)}; harness refused")
    # fg_make from the arm dir through the engine's own dated loader
    rel = a.arm_dir.resolve().relative_to((ROOT / "data/processed/models/fg_make").resolve())
    round_dir, arm = rel.parts[0], "/".join(rel.parts[1:])
    old_dir = AD.FG_DIR
    AD.FG_DIR = ROOT / "data/processed/models/fg_make"
    AD._FG_ROUND_NOTE.setdefault(round_dir, "lane A g9 round arm (in-process only, fg_make experiments.md s23)")
    fg_arm = AD.FgMakeAdapter._load_dated(inp, "F2", f"g9_{arm}", "g9_", round_dir, "train_fg_make_v4_par_g9_v1.py")
    AD.FG_DIR = old_dir
    orig_load = AD.FgMakeAdapter.load
    AD.FgMakeAdapter.load = classmethod(lambda cls, *x, **y: fg_arm)
    # run the stock harness main on this stack with the patched adapter
    sys.argv = [sys.argv[0], "--stack", f"g9_{a.name}", "--arms", a.arms]
    H.EXTRA_STACKS[f"g9_{a.name}"] = BASES[a.base]
    _orig_load_inputs = EngineInputs.load
    EngineInputs.load = staticmethod(lambda *x, **y: inp)
    try:
        rc = H.main()
    finally:
        EngineInputs.load = _orig_load_inputs
        AD.FgMakeAdapter.load = orig_load
    src = ROOT / "results/aggregation_v1" / f"harness_g9_{a.name}.parquet"
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"harness_{a.name}.parquet"
    os.replace(src, dst)
    report["runtime_s"] = round(time.time() - t0, 1)
    (OUT / f"harness_{a.name}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("moved to", dst, report["runtime_s"], "s", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
