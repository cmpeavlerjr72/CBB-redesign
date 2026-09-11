"""diag_clk4_composition.py -- law versus composition in the engine's clock.

`scripts/diag_clock_v4_shortfall.py` measures the served arm's conditional law
on the REAL possessions. This script measures the other half: the distribution
of STATES the engine visits, accumulated live by the clock adapter
(`ENGINE_CLOCK_DIAG=1`), against the same cut of the real possessions on the
same round-2 cell grid.

The identity it reports, over cells c with share w and mean consumed duration d:

    mean_sim - mean_act = SUM_c w_act(c) [d_sim(c) - d_act(c)]     <- LAW
                        + SUM_c [w_sim(c) - w_act(c)] d_act(c)     <- COMPOSITION
                        + SUM_c [w_sim(c) - w_act(c)][d_sim(c) - d_act(c)]

A shortfall that is LAW is a clock-model defect and round 4's to fix. A
shortfall that is COMPOSITION is the engine visiting different states -- which
for a state the clock conditions on (`prev_end` above all, i.e. how the previous
possession ended) is an UPSTREAM defect, and fixing it inside the clock would be
a downstream stage compensating for a known upstream bias.

Usage:
    .venv/Scripts/python.exe scripts/diag_clk4_composition.py \
        --run results/engine_v0/clock4_REF_cc_diag
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import clock as CK  # noqa: E402

CK_DIR = ROOT / "data" / "processed" / "models" / "clock"
UNIV_V2 = ROOT / "data" / "processed" / "games_universe_v2.parquet"
DESIGN = CK_DIR / "design_v2.parquet"
PERIOD_KEYS = ("H1", "H2", "OT")
DIMS = ["prev_end", "r2_bucket", "period_group", "bonus", "tempo_tercile"]


def actual_cells(game_ids: set[int], tempo_edges) -> pd.DataFrame:
    d = pd.read_parquet(DESIGN)
    d = d[(d["season"] == 2025) & (d["game_id"].isin(game_ids))].copy()
    d["r2_bucket"] = np.asarray(CK.R2_SR_LABELS, dtype=object)[
        CK.r2_bucket_id(d["seconds_remaining"].to_numpy())]
    per = d["period"].to_numpy()
    d["period_group"] = np.asarray(PERIOD_KEYS, dtype=object)[
        np.where(per <= 1.0, 0, np.where(per <= 2.0, 1, 2))]
    d["bonus"] = (d["in_bonus"].to_numpy() > 0).astype(np.int64)
    d["tempo_tercile"] = np.searchsorted(np.asarray(tempo_edges),
                                         d["tempo_prior_game"].to_numpy(), side="right")
    g = d.groupby(DIMS, dropna=False)["duration_s"]
    return g.agg(n="size", mean_consumed="mean").reset_index()


def decompose(sim: pd.DataFrame, act: pd.DataFrame) -> dict:
    m = sim.merge(act, on=DIMS, how="outer", suffixes=("_sim", "_act"))
    for c in ("n_sim", "n_act"):
        m[c] = m[c].fillna(0.0)
    m["w_sim"] = m["n_sim"] / m["n_sim"].sum()
    m["w_act"] = m["n_act"] / m["n_act"].sum()
    ds = m["mean_consumed_sim"].fillna(0.0)
    da = m["mean_consumed_act"].fillna(0.0)
    # cells only one side reaches carry their whole weight into composition
    law = float((m["w_act"] * (ds - da) * (m["n_sim"] > 0) * (m["n_act"] > 0)).sum())
    comp = float(((m["w_sim"] - m["w_act"]) * da).sum())
    inter = float(((m["w_sim"] - m["w_act"]) * (ds - da)
                   * (m["n_sim"] > 0) * (m["n_act"] > 0)).sum())
    mean_sim = float((m["w_sim"] * ds).sum())
    mean_act = float((m["w_act"] * da).sum())
    # cells the engine reaches and the real games never did carry their whole
    # contribution here rather than into any of the three named terms
    unmatched = float((m["w_sim"] * ds * (m["n_act"] == 0)).sum())
    resid = (mean_sim - mean_act) - (law + comp + inter + unmatched)
    return {"mean_sim": mean_sim, "mean_act": mean_act, "total_gap": mean_sim - mean_act,
            "law": law, "composition": comp, "interaction": inter,
            "unmatched_sim_only_cells": unmatched, "identity_residual": float(resid),
            "n_cells": int(len(m)), "cells_sim_only": int((m["n_act"] == 0).sum()),
            "cells_act_only": int((m["n_sim"] == 0).sum()),
            "possessions_sim": float(m["n_sim"].sum()),
            "possessions_act": float(m["n_act"].sum())}


def marginal(sim: pd.DataFrame, act: pd.DataFrame, by: str) -> pd.DataFrame:
    s = sim.groupby(by, dropna=False).apply(
        lambda g: pd.Series({"n_sim": g["n"].sum(),
                             "mean_sim": np.average(g["mean_consumed"], weights=g["n"])}),
        include_groups=False).reset_index()
    a = act.groupby(by, dropna=False).apply(
        lambda g: pd.Series({"n_act": g["n"].sum(),
                             "mean_act": np.average(g["mean_consumed"], weights=g["n"])}),
        include_groups=False).reset_index()
    t = s.merge(a, on=by, how="outer").fillna({"n_sim": 0.0, "n_act": 0.0})
    t["share_sim"] = t["n_sim"] / t["n_sim"].sum()
    t["share_act"] = t["n_act"] / t["n_act"].sum()
    t["share_gap_pp"] = 100.0 * (t["share_sim"] - t["share_act"])
    t["dur_gap_s"] = t["mean_sim"] - t["mean_act"]
    ma = float(np.average(t["mean_act"].fillna(0.0), weights=t["share_act"]))
    t["comp_contrib_s"] = (t["share_sim"] - t["share_act"]) * (t["mean_act"].fillna(ma) - ma)
    t["law_contrib_s"] = t["share_act"] * t["dur_gap_s"].fillna(0.0)
    t["underpowered"] = (t["n_sim"] < 300) | (t["n_act"] < 300)
    return t.sort_values(by)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="v3c_srfloor_P3_s1")
    ap.add_argument("--out", default="data/processed/models/clock/v4_diag")
    a = ap.parse_args()

    run = ROOT / a.run
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    sim = pd.read_parquet(run / "clock_cells.parquet")
    games = pd.read_parquet(run / "games.parquet")
    gids = set(int(g) for g in meta["game_ids"])

    from cbb_sim.engine.clock_adapter_v3 import V3C_MODES
    doc = json.loads((CK_DIR / V3C_MODES[a.mode]["manifest"]).read_text(encoding="utf-8"))
    with open(CK_DIR / doc["months"][0]["model_file"], "rb") as f:
        arm0 = pickle.load(f)
    from cbb_sim.engine.clock_adapter_v3 import tempo_edges_of
    edges = tempo_edges_of(arm0)
    if edges is None:
        raise SystemExit(f"{a.mode} exposes no tempo_edges; cell coding is undefined")

    act = actual_cells(gids, edges)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {"run": str(run), "mode": a.mode, "n_games": len(gids),
                    "n_seeds": meta["n_seeds"], "loop_commit": meta.get("loop_commit"),
                    "adapter_flags": meta["adapter_flags"],
                    "tempo_edges": [float(x) for x in edges]}

    for label, keep in (("all_periods", set(PERIOD_KEYS)), ("regulation", {"H1", "H2"})):
        s = sim[sim["period_group"].isin(keep)]
        t = act[act["period_group"].isin(keep)]
        report[f"decomposition_{label}"] = decompose(s, t)
        for by in ("prev_end", "r2_bucket", "period_group", "bonus", "tempo_tercile"):
            marginal(s, t, by).to_csv(out / f"v4_engine_vs_actual_{label}_{by}.csv",
                                      index=False)

    # engine G1 on these games, for the record
    # `possessions` on the engine's game row is already PER TEAM-GAME
    report["engine_possessions_per_team_game"] = float(games["possessions"].mean())
    report["engine_possessions_sd"] = float(
        games.groupby("seed")["possessions"].mean().std(ddof=1))
    (out / "v4_composition_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
