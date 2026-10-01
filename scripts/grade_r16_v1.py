"""grade_r16_v1.py -- round 16 closed-loop primary lines from full-size runs (lane B, 2026-10-01).

Spec: docs/models/shared_shooting/experiments.md section 5.2. For each run (arm, reference, floor draws):
  1. FT x opponent-FG make covariance (pts^2, both directions, rim / jump / three),
  2. FT x own-FG make covariance (pts^2, both sides summed),
  3. G5 home/away correlation, 4. total SD ratio (sqrt variance ratio), 5. margin SD ratio,
all from `diag_g5_channels_v1.run_read` (actual = residual against each run's own per-game sim mean).
Floor per line = SD across the floor draws (Decision 12 also takes the paired game bootstrap from
`ops_pair_bootstrap_v1.py`; the doc reports the max).

    python scripts/grade_r16_v1.py --ref v3full_COMB9GCTKD_s200_o0 --arms d1001B_FL1_s200_o0 ... --floors d1001D_S2f1_s200_o1000 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diag_g5_channels_v1 as GC  # noqa: E402

CH = GC.CH
N = len(CH)
FG = [CH.index(k) for k in ("rim", "jump", "three")]
FTI = CH.index("ft")


def lines(run: str) -> dict:
    r = GC.run_read(ROOT / "results/engine_v0" / run, 2025, segments=False, boot=0)
    Ca, Cs = np.array(r["Ca"]), np.array(r["Cs"])
    out = {}
    for lab, C in (("act", Ca), ("sim", Cs)):
        opp = sum(C[FTI, N + k] + C[k, N + FTI] for k in FG)
        own = sum(C[FTI, k] + C[N + FTI, N + k] for k in FG)
        out[f"ft_x_opp_fg_{lab}"] = float(opp)
        out[f"ft_x_own_fg_{lab}"] = float(own)
    ch = r["chain"]
    out["corr_sim"], out["corr_act"] = ch["corr_sim"], ch["corr_act"]
    out["total_sd_ratio"] = r["var_total"]["ratio_sqrt"]
    va = Ca[:N, :N].sum() + Ca[N:, N:].sum() - Ca[:N, N:].sum() - Ca[N:, :N].sum()
    vs = Cs[:N, :N].sum() + Cs[N:, N:].sum() - Cs[:N, N:].sum() - Cs[N:, :N].sum()
    out["margin_sd_ratio"] = float(np.sqrt(vs / va))
    out["n_games"] = ch["n_games"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--floors", nargs="*", default=[])
    ap.add_argument("--out", default="results/team_form/grade_r16_v1.json")
    a = ap.parse_args()
    runs = [a.ref] + a.arms + a.floors
    L = {r: lines(r) for r in runs if (ROOT / "results/engine_v0" / r / "games.parquet").exists()}
    keys = ["ft_x_opp_fg_sim", "ft_x_own_fg_sim", "corr_sim", "total_sd_ratio", "margin_sd_ratio"]
    fl = [r for r in a.floors if r in L] + ([a.ref] if a.ref in L else [])
    floors = {k: float(np.std([L[r][k] for r in fl], ddof=1)) if len(fl) >= 2 else None for k in keys}
    rep = {"lines": L, "floor_draws": fl, "floors_sd": floors, "deltas": {}}
    for arm in a.arms:
        if arm not in L or a.ref not in L:
            continue
        rep["deltas"][arm] = {k: {"ref": L[a.ref][k], "arm": L[arm][k], "delta": L[arm][k] - L[a.ref][k],
                                  "floors": ((L[arm][k] - L[a.ref][k]) / floors[k]) if floors[k] else None}
                              for k in keys}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"floors_sd": floors, "deltas": rep["deltas"],
                      "actual": {k: L[a.ref][k] for k in ("ft_x_opp_fg_act", "ft_x_own_fg_act", "corr_act")}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
