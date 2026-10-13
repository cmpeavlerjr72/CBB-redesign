"""diag_endgame_multilevel_v1.py -- multi-level evidence for the late-game round 7 closed loop (CLAUDE.md multi-level
rule). DIAGNOSTIC ONLY; reads the `diag_c4_extract_v1.py` caches of each run (results/diag_c4/<tag>_*.parquet).

Levels (the per-state and per-possession-type tables come from `diag_c4_d34_endgame_v1.py` on the same taps):
  per team quintile  offence pregame net rating (relative to its snapshot's league mean; `home_net`/`away_net`),
                     H2 last-4:00 window (possession start <= 240 s): seconds per possession of leaders by >= 8 and of
                     trailers by >= 8, leader TOV rate (>= 8), offence in-bonus share (last 2:00, leading 1-6).
  per game           per-game sim mean minus real for H2 possessions ending <= 240 s (MD4) and last-2:00 points
                     (MD5 state), mean abs error and correlation across games.
Cells with fewer than 60 real possessions (or games) are labelled UNDERPOWERED.

    .venv/Scripts/python.exe scripts/diag_endgame_multilevel_v1.py --tags eg_ctrl_o0_s25 eg_LF2_o0_s25 --out results/endgame/multilevel.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_c4_lib_v1 as L  # noqa: E402

MIN_N = 60


def team_q(pg: pd.DataFrame) -> pd.DataFrame:
    """(game_id, off_side) -> offence net-rating quintile (edges over all sample team-games)."""
    h = pg[["game_id", "home_net"]].rename(columns={"home_net": "net"}).assign(off_side=0)
    a = pg[["game_id", "away_net"]].rename(columns={"away_net": "net"}).assign(off_side=1)
    t = pd.concat([h, a], ignore_index=True)
    e = np.nanquantile(t["net"], [0.2, 0.4, 0.6, 0.8])
    t["q"] = np.where(t["net"].isna(), -1, np.searchsorted(e, t["net"].to_numpy(), side="right"))
    return t[["game_id", "off_side", "q"]]


def lines_by_q(late: pd.DataFrame, tq: pd.DataFrame, S: int) -> pd.DataFrame:
    x = late[(late["period"] == 2) & (late["cs"] <= 240)].merge(tq, on=["game_id", "off_side"], how="left")
    rows = []
    for q in range(5):
        y = x[x["q"] == q]
        ld, tr = y[y["moff"] >= 8], y[y["moff"] <= -8]
        b = y[(y["cs"] <= 120) & (y["moff"] >= 1) & (y["moff"] <= 6)]
        rows.append({"q": q + 1, "n_lead": len(ld) / S, "sec_lead": ld["dur"].mean(), "tov_lead": ld["tov"].mean(),
                     "n_trail": len(tr) / S, "sec_trail": tr["dur"].mean(), "n_bonus": len(b) / S,
                     "bonus": b["bonus"].mean()})
    return pd.DataFrame(rows)


def per_game(late: pd.DataFrame, S: int) -> pd.DataFrame:
    x = late[late["period"] == 2]
    md4 = x[x["ce"] <= 240].groupby("game_id").size() / S
    last2 = x[x["cs"] <= 120].groupby("game_id")["pts"].sum() / S
    return pd.DataFrame({"md4": md4, "last2": last2})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = lambda t, n: pd.read_parquet(L.OUT / f"{t}_{n}.parquet")  # noqa: E731
    pg = D(a.tags[0], "pregame")
    pg["fin_margin"] = pg["fin_h"] - pg["fin_a"]
    rgs = D(a.tags[0], "real_gs")
    ok = pg.merge(rgs[["game_id", "final_margin"]], on="game_id")
    G = set(ok.loc[ok["final_margin"] == ok["fin_margin"], "game_id"])
    tq = team_q(pg)
    rl = D(a.tags[0], "real_late")
    rl = rl[rl["game_id"].isin(G)]
    rq = lines_by_q(rl, tq, 1)
    rpg = per_game(rl, 1)
    W = []
    w = W.append
    w("# Endgame round 7 closed loop: multi-level evidence (per team quintile, per game)\n")
    w(f"DIAGNOSTIC ONLY. 500-game verified fold-2 sample, {len(G)} games with a pbp path ending at the verified final. "
      "Team quintile = offence pregame net rating (relative to its snapshot's league mean). `U` = underpowered "
      f"(< {MIN_N} real possessions in the cell).\n")
    w("## Per offence team-strength quintile, H2 last 4:00 (leaders / trailers by 8+), last 2:00 bonus (leading 1-6)\n")
    w("| run | Q | lead poss | sec/poss lead (real / sim) | TOV lead (real / sim) | trail poss | sec/poss trail (real / sim) "
      "| in-bonus lead 1-6 (real / sim) |")
    w("|---|---|---:|---|---|---:|---|---|")
    pg_rows = []
    for t in a.tags:
        sl = D(t, "sim_late")
        sl = sl[sl["game_id"].isin(G)]
        S = int(sl["seed"].nunique())
        sq = lines_by_q(sl, tq, S)
        for i in range(5):
            r, s = rq.iloc[i], sq.iloc[i]
            u1 = " U" if r["n_lead"] < MIN_N else ""
            u2 = " U" if r["n_trail"] < MIN_N else ""
            u3 = " U" if r["n_bonus"] < MIN_N else ""
            w(f"| {t} | Q{i + 1} | {r['n_lead']:.0f}{u1} | {r['sec_lead']:.2f} / {s['sec_lead']:.2f} | "
              f"{r['tov_lead']:.3f} / {s['tov_lead']:.3f} | {r['n_trail']:.0f}{u2} | {r['sec_trail']:.2f} / {s['sec_trail']:.2f} | "
              f"{r['bonus']:.3f}{u3} / {s['bonus']:.3f} |")
        # slopes across quintiles (sim on real), the responsiveness read
        sl_lead = np.polyfit(rq["sec_lead"], sq["sec_lead"], 1)[0]
        sl_trail = np.polyfit(rq["sec_trail"], sq["sec_trail"], 1)[0]
        spg = per_game(sl, S).reindex(rpg.index)
        d4, d5 = spg["md4"] - rpg["md4"], spg["last2"] - rpg["last2"]
        pg_rows.append((t, sl_lead, sl_trail, d4.mean(), d4.abs().mean(), np.corrcoef(spg["md4"].fillna(0), rpg["md4"].fillna(0))[0, 1],
                        d5.mean(), d5.abs().mean(), np.corrcoef(spg["last2"].fillna(0), rpg["last2"].fillna(0))[0, 1]))
    w("\n## Responsiveness across team quintiles and per-game errors\n")
    w("| run | slope sim on real, lead sec/poss (5 quintiles) | same, trail | per-game H2 poss ending <= 240 s: mean (sim - real) | "
      "MAE | corr | per-game last-2:00 points: mean (sim - real) | MAE | corr |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in pg_rows:
        w(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:+.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:+.3f} | {r[7]:.3f} | {r[8]:.3f} |")
    w("\nQuintile slopes rest on 5 points each and are noisy at this sample size; read as direction.\n")
    Path(a.out).write_text("\n".join(W) + "\n", encoding="utf-8")
    print("\n".join(W))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
