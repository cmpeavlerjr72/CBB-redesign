"""diag_clk5_mc_check.py -- the pre-registered Monte-Carlo cross-check of round 5's
primary metric (`docs/models/clock/experiments.md` section 16.5).

The bake-off computes each arm's `Var(Dbar)` in closed form by conditioning on
the latent. Section 16.5 requires the WINNING arm's number to be cross-checked
by a direct resample of the real 2025 state sequences, because a closed form
that is wrong is wrong silently. This script does exactly that and nothing else:
it draws durations the way `clock.sample_from_pmf` draws them, applies the arm's
own latent, and measures the across-replicate SD of each game's realised mean
duration and of the possession count the bridge implies from it.

It fits nothing. Sigma is read from the bake-off report, which fitted it on
training rows only.

Usage:
    .venv/Scripts/python.exe scripts/diag_clk5_mc_check.py --reps 25
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

from cbb_sim.models import clock as ck  # noqa: E402

CK_DIR = ROOT / "data" / "processed" / "models" / "clock"
UNIV_V2 = ROOT / "data" / "processed" / "games_universe_v2.parquet"
DESIGN = CK_DIR / "design_v2.parquet"
GRID = np.arange(ck.DURATION_CAP + 1, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--bakeoff", default="data/processed/models/clock/v5_bakeoff")
    a = ap.parse_args()
    rep = json.loads((ROOT / a.bakeoff / "v5_bakeoff_report.json").read_text("utf-8"))
    sigma_a1 = float(rep["params"]["F2"]["A1_sigma"])
    sigma_a2 = float(rep["params"]["F2"]["A2_sigma"])

    univ = pd.read_parquet(UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    d = pd.read_parquet(DESIGN)
    d = d[(d["season"] == 2025) & (d["period"] <= 2.0) & (d["game_id"].isin(cc))]
    d = d.sort_values(["game_id", "period", "poss_index"],
                      kind="stable").reset_index(drop=True)

    from cbb_sim.engine.clock_adapter_v3 import V3C_MODES
    doc = json.loads((CK_DIR / V3C_MODES["v3c_srfloor_P3_s1"]["manifest"]).read_text("utf-8"))
    months = sorted(doc["months"], key=lambda m: m["refit_date"])
    cuts = np.array([np.datetime64(pd.Timestamp(m["refit_date"]), "ns") for m in months])
    seg = np.searchsorted(cuts, pd.to_datetime(d["game_date"]).to_numpy(), side="right") - 1

    pmf = np.empty((len(d), ck.DURATION_CAP + 1))
    for k in np.unique(seg):
        with open(CK_DIR / months[int(k)]["model_file"], "rb") as f:
            arm = pickle.load(f)
        r = np.flatnonzero(seg == k)
        for lo in range(0, len(r), 40_000):
            rr = r[lo:lo + 40_000]
            pmf[rr] = arm.pmf(d.iloc[rr].reset_index(drop=True))

    R = d["seconds_remaining"].to_numpy(dtype=np.float64)
    gid = d["game_id"].to_numpy()
    off_h = (d["offense_is_home"].to_numpy() > 0)
    codes, first = pd.factorize(gid)
    ng = len(first)
    M = np.bincount(codes).astype(np.float64)
    rng = np.random.default_rng(a.seed)
    F = np.cumsum(pmf, axis=1)
    F[:, -1] = 1.0

    out = {}
    for name, sig, per_offence in (("R", 0.0, False), ("A1", sigma_a1, False),
                                   ("A2", sigma_a2, True)):
        dbar = np.empty((a.reps, ng))
        for t in range(a.reps):
            u = rng.random(len(d))
            idx = (u[:, None] > F).sum(axis=1)
            T = GRID[np.minimum(idx, ck.DURATION_CAP)]
            if sig > 0:
                if per_offence:
                    zh = rng.normal(size=ng); za = rng.normal(size=ng)
                    Ah = np.exp(sig * zh - 0.5 * sig ** 2)
                    Aa = np.exp(sig * za - 0.5 * sig ** 2)
                    A = np.where(off_h, Ah[codes], Aa[codes])
                else:
                    z = rng.normal(size=ng)
                    A = np.exp(sig * z - 0.5 * sig ** 2)[codes]
                T = np.rint(A * T)
            D = np.minimum(T, R)
            dbar[t] = np.bincount(codes, weights=D) / M
        var_within = dbar.var(axis=0, ddof=1).mean()
        Pbar = 1200.0 / (d["duration_s"].sum() / len(d))
        k = Pbar / float(d["duration_s"].mean())
        out[name] = {
            "sigma": sig, "reps": a.reps,
            "mc_var_dbar": float(var_within),
            "mc_sd_dbar": float(np.sqrt(var_within)),
            "mc_P_sd": float(k * np.sqrt(var_within)),
            "mc_mean_consumed": float(dbar.mean()),
        }
        print(name, json.dumps(out[name]))
    (ROOT / a.bakeoff / "v5_mc_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
