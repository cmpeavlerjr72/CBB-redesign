#!/usr/bin/env python
"""
diag_ot_carry_read_v1.py -- OVERTIME block of the carry-vs-no-carry read (lane F, 2026-10-01).

Reads two (or more) engine runs made with ENGINE_OT_STATS=1 (extra game columns `{home,away}_ot_{pts,fta,fga,poss}`; 0 in
regulation games) on the SAME games and seeds, and reports per arm and as a paired difference (game-clustered bootstrap, seeds
averaged within game):
  * OT rate (share of sims with n_periods > 2)
  * per OT sim: OT points (both teams), OT free-throw attempts, OT field-goal attempts, OT possessions (per team), and per
    extra period (divide by n_periods - 2), PPP and FTA per possession in OT
  * the REAL fold-2 (2025) overtime reference from the event layer: points per possession and FTA per possession in periods >= 3
  * floor: the same paired difference between two no-carry draws when `--floor-a/--floor-b` name them (Decision 12 floor),
    otherwise the bootstrap CI only, labelled.

    python scripts/diag_ot_carry_read_v1.py --ref results/engine_v0/<no-carry> --arm results/engine_v0/<carry> \
        [--floor-a ... --floor-b ...] --out-md results/laneF/ot_read.md --out-json results/laneF/ot_read.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["ot_pts", "ot_fta", "ot_fga", "ot_poss"]


def load(path: Path) -> pd.DataFrame:
    g = pd.read_parquet(path / "games.parquet")
    need = [f"{s}_{c}" for s in ("home", "away") for c in COLS]
    miss = [c for c in need if c not in g.columns]
    if miss:
        raise SystemExit(f"{path}: no OT columns {miss}; the run needs ENGINE_OT_STATS=1")
    g["is_ot"] = (g["n_periods"] > 2).astype(float)
    g["ot_periods"] = np.maximum(g["n_periods"] - 2, 0).astype(float)
    for c in COLS:
        g[c] = g[f"home_{c}"].astype(float) + g[f"away_{c}"].astype(float)       # both teams
    return g


def per_game(g: pd.DataFrame) -> pd.DataFrame:
    """sum over seeds within game (so ratios are pooled ratios of sums), keyed by game_id"""
    a = g.groupby("game_id")[["is_ot", "ot_periods", *COLS]].sum()
    a["n"] = g.groupby("game_id").size()
    return a


def ratios(a: pd.DataFrame) -> dict:
    n_sim, n_ot = a["n"].sum(), a["is_ot"].sum()
    out = {"n_sims": int(n_sim), "ot_sims": int(n_ot), "ot_rate": float(n_ot / n_sim)}
    if n_ot:
        out.update({"ot_pts_per_ot_sim": float(a["ot_pts"].sum() / n_ot), "ot_fta_per_ot_sim": float(a["ot_fta"].sum() / n_ot),
                    "ot_fga_per_ot_sim": float(a["ot_fga"].sum() / n_ot), "ot_poss_per_ot_sim": float(a["ot_poss"].sum() / n_ot),
                    "ot_periods_per_ot_sim": float(a["ot_periods"].sum() / n_ot),
                    "ppp_ot": float(a["ot_pts"].sum() / max(a["ot_poss"].sum(), 1)),          # ot_poss = the two teams possessions summed, as the event-layer rows count them
                    "fta_per_poss_ot": float(a["ot_fta"].sum() / max(a["ot_poss"].sum(), 1))})
    return out


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = 20261001) -> dict:
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    rng = np.random.default_rng(seed)
    keys = list(ratios(a).keys())
    base = {k: ratios(b)[k] - ratios(a)[k] for k in keys if k in ratios(b) and k in ratios(a)}
    draws = {k: [] for k in base}
    n = len(idx)
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        ra, rb = ratios(a.iloc[s]), ratios(b.iloc[s])
        for k in base:
            if k in ra and k in rb:
                draws[k].append(rb[k] - ra[k])
    return {k: {"diff": base[k], "boot_sd": float(np.std(draws[k])), "ci95": [float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))]}
            for k in base if draws[k]}


def real_ot_reference() -> dict:
    out = {}
    for s in (2025,):
        c = pd.read_parquet(ROOT / f"data/processed/possessions_v4/possessions_{s}.parquet", columns=["period", "points", "fta"])
        o = c[c["period"] >= 3]
        out[str(s)] = {"ot_possessions": int(len(o)), "ppp_ot": float(o["points"].sum() / len(o)), "fta_per_poss_ot": float(o["fta"].sum() / len(o))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--floor-a", type=Path, default=None)
    ap.add_argument("--floor-b", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()
    ga, gb = per_game(load(a.ref)), per_game(load(a.arm))
    rep = {"ref": str(a.ref), "arm": str(a.arm), "ref_ratios": ratios(ga), "arm_ratios": ratios(gb), "paired_arm_minus_ref": boot_diff(ga, gb),
           "real_2025_ot": real_ot_reference()}
    if a.floor_a and a.floor_b:
        rep["floor_ref_draws"] = boot_diff(per_game(load(a.floor_a)), per_game(load(a.floor_b)))
    lines = ["# OT block, carry vs no carry", "", f"ref `{a.ref}` / arm `{a.arm}`", "",
             "| quantity | ref | arm | arm - ref | boot SD | CI95 |" + (" floor (two ref draws) |" if "floor_ref_draws" in rep else ""),
             "|---|--:|--:|--:|--:|---|" + ("--:|" if "floor_ref_draws" in rep else "")]
    for k, d in rep["paired_arm_minus_ref"].items():
        fl = rep.get("floor_ref_draws", {}).get(k, {}).get("diff")
        lines.append(f"| {k} | {rep['ref_ratios'].get(k, float('nan')):.4f} | {rep['arm_ratios'].get(k, float('nan')):.4f} | {d['diff']:+.4f} | "
                     f"{d['boot_sd']:.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |" + (f" {fl:+.4f} |" if fl is not None else (" |" if "floor_ref_draws" in rep else "")))
    r = rep["real_2025_ot"]["2025"]
    lines += ["", f"Real 2025 overtime (event layer v4): PPP {r['ppp_ot']:.4f}, FTA per possession {r['fta_per_poss_ot']:.4f} over {r['ot_possessions']:,} possessions."]
    md = "\n".join(lines)
    print(md)
    if a.out_md:
        a.out_md.parent.mkdir(parents=True, exist_ok=True)
        a.out_md.write_text(md, encoding="utf-8")
    if a.out_json:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
