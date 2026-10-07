"""FT% channel in points by player group (diagnostic only): anonymous slots vs named returners vs named newcomers.
Sim: 200-seed served-v2 FT/player taps (FT sub-model identical in served v3; v3 control taps carry no players.parquet).
Benchmark for named = shrunk (20 att) full-season actual FT% talent; anonymous benchmark = actual window FT%.
Points/game = sim FTA per game in group x (sim FT% - benchmark). Returner = athlete has FTA in prior-season truth.
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RUNS = {2024: "f1c_V2_full_s200_o0", 2025: "v3full_COMB9GCTKD_s200_o0"}
gl = pd.read_parquet(ROOT / "results/total_bias_decomp/game_level_v1.parquet")
pg = pd.read_parquet(ROOT / "data/processed/truth/player_game_v2.parquet", columns=["game_id","season","athlete_id","team_id","ftm","fta","fta_tech","ftm_tech"])
pg["fta"] -= pg.fta_tech.fillna(0); pg["ftm"] -= pg.ftm_tech.fillna(0)
out = []
for season, tag in RUNS.items():
    g = gl[gl.season == season][["game_id","dss_b"]]
    p = pd.read_parquet(ROOT / f"results/engine_v0/{tag}/players.parquet", columns=["game_id","athlete_id","team_id","pts","fta","fgm2_rim","fgm2_jump","fgm3"])
    p["ftm"] = p.pts - 2*(p.fgm2_rim+p.fgm2_jump) - 3*p.fgm3
    s = p.groupby(["game_id","athlete_id"], as_index=False)[["fta","ftm"]].sum(); s[["fta","ftm"]] /= 200
    s = s.merge(g, on="game_id")
    # anonymous slots are not in players.parquet: anon = game total (games.parquet) - named
    gm = pd.read_parquet(ROOT / f"results/engine_v0/{tag}/games.parquet", columns=["game_id","seed","home_fta","away_fta","home_ftm","away_ftm"])
    gm = gm.groupby("game_id")[["home_fta","away_fta","home_ftm","away_ftm"]].mean()
    gm["fta"] = gm.home_fta+gm.away_fta; gm["ftm"] = gm.home_ftm+gm.away_ftm
    nm = s.groupby("game_id")[["fta","ftm"]].sum()
    an = (gm[["fta","ftm"]] - nm.reindex(gm.index).fillna(0)).clip(lower=0).reset_index().merge(g, on="game_id")
    an["athlete_id"] = -1
    s = pd.concat([s, an], ignore_index=True)
    cur = pg[pg.season == season]; lg = cur.ftm.sum()/cur.fta.sum()
    tal = cur.groupby("athlete_id")[["ftm","fta"]].sum(); tal = ((tal.ftm+20*lg)/(tal.fta+20)).rename("talent")
    prev = set(pg[(pg.season == season-1) & (pg.fta > 0)].athlete_id)
    s = s.merge(tal, left_on="athlete_id", right_index=True, how="left")
    s["grp"] = np.where(s.athlete_id <= 0, "anonymous", np.where(s.athlete_id.isin(prev), "returner", "newcomer"))
    t = cur.merge(g, on="game_id")
    for b in ("d00-14","d15-45","d46+"):
        sb, tb = s[s.dss_b == b], t[t.dss_b == b]; n = sb.game_id.nunique(); act = tb.ftm.sum()/tb.fta.sum()
        tot = 0
        for grp, x in sb.groupby("grp"):
            bench = act if grp == "anonymous" else (x.fta*x.talent.fillna(lg)).sum()/x.fta.sum()
            pts = x.fta.sum()/n*(x.ftm.sum()/x.fta.sum()-bench)*1.0
            tot += pts
            out.append(dict(season=season,bucket=b,grp=grp,fta_pg_team_sum=x.fta.sum()/n, fta_share=x.fta.sum()/sb.fta.sum(), sim_ft=x.ftm.sum()/x.fta.sum(), bench=bench, pts_per_game=pts))
        out.append(dict(season=season,bucket=b,grp="ALL (sum of groups)",fta_pg_team_sum=sb.fta.sum()/n,fta_share=1,sim_ft=sb.ftm.sum()/sb.fta.sum(),bench=act,pts_per_game=tot))
d = pd.DataFrame(out); d.to_csv(ROOT/"results/early_total_points/ft_player_cut.csv", index=False)
pd.set_option("display.width",200); print(d.round(4).to_string(index=False))
