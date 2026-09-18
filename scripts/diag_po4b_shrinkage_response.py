"""
diag_po4b_shrinkage_response.py -- the two multi-level evidence cells section
12.4 asks for that the gate table does not carry.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 12.4
("Multi-level evidence").

  1. **Per-team, against the thing the arm actually does.** An arm changes a
     team's eight style columns by some amount; the pre-registered question is
     whether that team's simulated scoring moves with it. Per team: the mean
     absolute shrinkage the arm applies to that team's own style columns over
     its games in the subset, against the change in that team's simulated
     points bias. A team with fewer than 3 games in the subset is UNDERPOWERED
     and is labelled, never dropped.
  2. **Per possession type.** The realised terminal-event mix per possession --
     3PA, rim FGA, jump-2 FGA, turnovers, FTA -- against the event layer's own
     actuals on the SAME team-games, arm by arm.

Read-only. Nothing here adjusts anything.

    .venv/Scripts/python.exe scripts/diag_po4b_shrinkage_response.py \
        --ref po4b_R_s25 --floor po4b_R_s25_floor --arms po4b_G2_s25 po4b_G3_s25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import grade_po4b_closed_loop as GR  # noqa: E402

ENGINE_DIR = Path("data/processed/models/engine")
RESULTS = Path("results/engine_v0")
ARM_OF = {"po4b_G2_s25": "G2", "po4b_G3_s25": "G3"}


def shrinkage_per_team(arm: str, game_ids: np.ndarray, allg: pd.DataFrame) -> pd.DataFrame:
    """Mean |arm style column - served style column| per team over its games in
    the subset, averaged over the eight style columns."""
    ref = ENGINE_DIR / "event_round2_s1_F2_2025"
    a = ENGINE_DIR / f"event_round4b_{arm}_F2_2025"
    ri = json.loads((ref / "index.json").read_text(encoding="utf-8"))
    style_pos = [i for i, c in enumerate(ri["team_cols"])
                 if (c.startswith("off_") and c.endswith("_c")
                     and not c.startswith("off_rating_"))
                 or (c.startswith("opp_def_") and c.endswith("_c"))]
    rb = np.load(ref / "team_block.npz")["team_block"]
    ab = np.load(a / "team_block.npz")["team_block"]
    dev = np.abs(ab[:, :, style_pos] - rb[:, :, style_pos]).mean(axis=2)   # (G, 2)
    pos = {int(g): i for i, g in enumerate(allg["game_id"].to_numpy())}
    rows = []
    sub = allg[allg["game_id"].isin(game_ids)]
    for _, r in sub.iterrows():
        i = pos[int(r["game_id"])]
        rows.append({"game_id": int(r["game_id"]), "team_id": int(r["home_team_id"]),
                     "shrink": float(dev[i, 0])})
        rows.append({"game_id": int(r["game_id"]), "team_id": int(r["away_team_id"]),
                     "shrink": float(dev[i, 1])})
    d = pd.DataFrame(rows)
    return d.groupby("team_id", as_index=False).agg(n=("game_id", "size"),
                                                    shrink=("shrink", "mean"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--out", default="results/engine_v0/po4b_grade")
    args = ap.parse_args()

    meta = json.loads((RESULTS / args.ref / "run_meta.json").read_text(encoding="utf-8"))
    game_ids = np.array(sorted(int(x) for x in meta["game_ids"]))
    allg = GR.engine_games()
    act = GR.actual_team_long(args.season, game_ids)
    tags = [args.ref, args.floor, *args.arms]
    sims = {t: GR.sim_team_long(RESULTS / t, game_ids, allg) for t in tags}

    # ---- 1. per-team response against the shrinkage the arm applies --------
    def team_bias(t):
        m = sims[t].merge(act, on=["game_id", "team_id"], how="inner")
        m["e"] = m["pts"] - m["a_pts"]
        return m.groupby("team_id", as_index=False).agg(n=("game_id", "size"),
                                                        bias=("e", "mean"))

    ref_b = team_bias(args.ref).rename(columns={"bias": "bias_ref"})
    fl_b = team_bias(args.floor).rename(columns={"bias": "bias_floor"})[["team_id", "bias_floor"]]
    out_rows = []
    print("=== per-team response vs the shrinkage the arm applies ===")
    for t in args.arms:
        arm = ARM_OF.get(t, t)
        sh = shrinkage_per_team(arm, game_ids, allg)
        b = team_bias(t).rename(columns={"bias": "bias_arm"})[["team_id", "bias_arm"]]
        m = ref_b.merge(b, on="team_id").merge(fl_b, on="team_id").merge(
            sh[["team_id", "shrink"]], on="team_id", how="left")
        m["d_arm"] = m["bias_arm"] - m["bias_ref"]
        m["d_floor"] = m["bias_floor"] - m["bias_ref"]
        pw = m[m["n"] >= 3]
        print(f"\n{t} ({arm}): {len(m)} teams, {len(pw)} with >=3 subset games "
              f"({len(m) - len(pw)} UNDERPOWERED and excluded from the correlations)")
        print(f"  mean shrinkage applied      {m['shrink'].mean():.4f} "
              f"(p10 {m['shrink'].quantile(.1):.4f}, p90 {m['shrink'].quantile(.9):.4f})")
        print(f"  corr(shrinkage, |d_arm|)    {pw['shrink'].corr(pw['d_arm'].abs()):+.4f}   "
              f"floor: corr(shrinkage, |d_floor|) {pw['shrink'].corr(pw['d_floor'].abs()):+.4f}")
        print(f"  SD of per-team bias change  arm {pw['d_arm'].std(ddof=1):.4f}   "
              f"floor {pw['d_floor'].std(ddof=1):.4f}")
        q = pd.qcut(pw["shrink"], 5, labels=False, duplicates="drop")
        for k in sorted(pd.Series(q).dropna().unique()):
            c = pw[q == k]
            print(f"    shrink Q{int(k) + 1}: n {len(c):3d}  mean shrink {c['shrink'].mean():.4f}  "
                  f"mean d_arm {c['d_arm'].mean():+.4f}  mean d_floor {c['d_floor'].mean():+.4f}"
                  f"{'   UNDERPOWERED' if len(c) < 30 else ''}")
        m.insert(0, "run", t)
        out_rows.append(m)

    # ---- 2. per-possession terminal mix ------------------------------------
    print("\n=== per-possession terminal mix, sim vs the event layer's actuals ===")
    mix = []
    for t in tags:
        m = sims[t].merge(act, on=["game_id", "team_id"], how="inner")
        row = {"run": t, "n_team_games": int(len(m))}
        row["fga3/poss"] = float((m["fga3"] / m["poss"]).mean())
        row["fga_rim/poss"] = float((m["fga2_rim"] / m["poss"]).mean())
        row["fga_jump2/poss"] = float((m["fga2_jump"] / m["poss"]).mean())
        row["tov/poss"] = float((m["tov"] / m["poss"]).mean())
        row["fta/poss"] = float((m["fta"] / m["poss"]).mean())
        row["oreb/poss"] = float((m["oreb"] / m["poss"]).mean())
        mix.append(row)
    a = sims[args.ref].merge(act, on=["game_id", "team_id"], how="inner")
    mix.append({"run": "ACTUAL", "n_team_games": int(len(a)),
                "fga3/poss": float((a["a_tpa"] / a["a_poss"]).mean()),
                "fga_rim/poss": float((a["ev_fga_rim"] / a["a_poss"]).mean()),
                "fga_jump2/poss": float(((a["a_fga"] - a["a_tpa"] - a["ev_fga_rim"])
                                         / a["a_poss"]).mean()),
                "tov/poss": float((a["a_tov"] / a["a_poss"]).mean()),
                "fta/poss": float((a["a_fta"] / a["a_poss"]).mean()),
                "oreb/poss": float((a["a_oreb"] / a["a_poss"]).mean())})
    mx = pd.DataFrame(mix)
    print(mx.round(5).to_string(index=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(out_rows, ignore_index=True).to_csv(out / "per_team_response.csv", index=False)
    mx.to_csv(out / "possession_mix.csv", index=False)
    print(f"\nwrote {out}/per_team_response.csv and {out}/possession_mix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
