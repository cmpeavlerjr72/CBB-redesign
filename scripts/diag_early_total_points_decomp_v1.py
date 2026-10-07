"""Early-season total gap in POINTS (diagnostic only). Served v3 50-seed control taps vs verified actual.

Reuses scripts/diag_total_bias_decomp_v1 (identity + exact Shapley over 10 factors, possession-replacement
built in: with P fixed, FGA = P(1-tov+oreb-0.44 fta), so extra FTA/TOV/OREB displace shots).
Adds: v3 control runs, grouped channels, one-at-a-time swap check, per-team cut, writes results/early_total_points/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_total_bias_decomp_v1 as D
from cbb_sim.eval import reference as R

ROOT = D.ROOT
D.RUNS = {2024: ROOT / "results/engine_v0/fcal_F1_ctrl_s50/games.parquet",
          2025: ROOT / "results/engine_v0/fcal_F2_ctrl_s50/games.parquet"}
OUT = ROOT / "results/early_total_points"; OUT.mkdir(parents=True, exist_ok=True)
GROUP = {"possessions": ["P"], "TOV": ["tov_r"], "OREB": ["oreb_r"], "FTA rate (net)": ["fta_r"],
         "FT%": ["p_ft"], "shot mix 3PA share": ["s3"], "shot mix rim share": ["rim_sh"],
         "rim make": ["p_rim"], "jump2 make": ["p_jump"], "3 make": ["p3"]}

def swap_one(fs, fa):  # sim->actual one at a time, from sim base (order dependent; check only)
    base = D.ident(fs); return {k: D.ident({**fs, k: fa[k]}) - base for k in D.FACT}

def main():
    rows, teamrows = [], []
    for season in (2024, 2025):
        games = R.load_actual_games(season)
        sim = D.sim_team_games(season); tru = D.truth_team_games(season, games)
        common = set(sim.game_id) & set(tru.groupby("game_id").filter(lambda x: len(x) == 2).game_id)
        games = games[games.game_id.isin(common)].copy()
        games["date"] = pd.to_datetime(games["game_date"]); games["dss"] = (games["date"] - games["date"].min()).dt.days
        games["b"] = pd.cut(games["dss"], [-1, 14, 45, 10000], labels=["d0-14", "d15-45", "d46+"]).astype(str)
        sim = sim[sim.game_id.isin(common)].merge(games[["game_id", "b"]], on="game_id")
        tru = tru[tru.game_id.isin(common)].merge(games[["game_id", "b"]], on="game_id")
        prev = R.load_actual_games(season - 1)
        pl = pd.concat([prev[["home_team_id", "total"]].rename(columns={"home_team_id": "t"}),
                        prev[["away_team_id", "total"]].rename(columns={"away_team_id": "t"})])
        penv = pl.groupby("t")["total"].mean()
        for b in ("d0-14", "d15-45", "d46+", "all"):
            s_ = sim if b == "all" else sim[sim.b == b]; t_ = tru if b == "all" else tru[tru.b == b]
            n = s_.game_id.nunique(); r = D.decompose(s_, t_, n)
            fs, fa = D.factors(s_), D.factors(t_)
            sw = {k: v / n for k, v in swap_one(fs, fa).items()}
            row = dict(season=season, bucket=b, n=n, total_bias=r["total_bias"], ident=r["ident_bias"], resid=r["resid"])
            for g, ks in GROUP.items(): row[g] = sum(r["c_" + k] for k in ks)
            for k in D.FACT: row["swap_" + k] = sw[k]
            for k in D.FACT:
                if k != "P": row["sim_" + k] = fs[k]; row["act_" + k] = fa[k]
            rows.append(row)
        # per-team, d0-14 and d46+: team bucket = all games it played (both sides)
        gm = games.set_index("game_id")
        for b in ("d0-14", "d46+"):
            gids = games.loc[games.b == b, "game_id"]
            tg = pd.concat([games.loc[games.b == b, ["game_id", "home_team_id"]].rename(columns={"home_team_id": "t"}),
                            games.loc[games.b == b, ["game_id", "away_team_id"]].rename(columns={"away_team_id": "t"})])
            for t, g in tg.groupby("t"):
                if len(g) < 2: continue
                ids = set(g.game_id)
                s_ = sim[sim.game_id.isin(ids)]; t_ = tru[tru.game_id.isin(ids)]
                try: r = D.decompose(s_, t_, len(ids))
                except Exception: continue
                o = dict(season=season, b=b, team=t, n=len(ids), total_bias=r["total_bias"], resid=r["resid"], prior=penv.get(t, np.nan) * 1.0)
                for gk, ks in GROUP.items(): o[gk] = sum(r["c_" + k] for k in ks)
                teamrows.append(o)
    A = pd.DataFrame(rows); A.to_csv(OUT / "channels.csv", index=False)
    T = pd.DataFrame(teamrows); T.to_csv(OUT / "teams.csv", index=False)
    pd.set_option("display.width", 250)
    print(A[["season", "bucket", "n", "total_bias", "ident", "resid"] + list(GROUP)].round(2).to_string(index=False))
    print(A[["season", "bucket"] + ["swap_" + k for k in D.FACT]].round(2).to_string(index=False))
    print(A[["season", "bucket"] + [c for c in A.columns if c.startswith(("sim_", "act_"))]].round(4).to_string(index=False))
    # team stats
    for (s, b), g in T.groupby(["season", "b"]):
        g = g.dropna(subset=["prior"]); q = pd.qcut(g.prior, 5, labels=False)
        print(f"\nseason {s} {b} teams {len(g)}")
        for c in ["total_bias"] + list(GROUP):
            sl = np.polyfit(g.prior, g[c], 1)[0]
            print(f"  {c:20s} mean {g[c].mean():+.2f} sd {g[c].std():.2f} neg {np.mean(g[c]<0):.0%} slope {sl:+.3f} corr {np.corrcoef(g.prior,g[c])[0,1]:+.2f} Q:",
                  " ".join(f"{g[c][q==i].mean():+.2f}" for i in range(5)))
if __name__ == "__main__": main()
