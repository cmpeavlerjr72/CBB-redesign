"""FT channel of the total-bias decomposition: named vs anonymous slots, and
make-probability bias vs who-shoots mix, by days-since-season-start bucket (diagnostic only).

talent proxy = the player's full-season actual FT% (grading only, shrunk 20 attempts toward the
league mean). 'make bias' = sim FT% - talent-weighted FT% on the sim's own FTA mix (named players).
Run: .venv/Scripts/python.exe scripts/diag_total_bias_ft_slots_v1.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {2024: "f1c_V2_full_s200_o0", 2025: "v3full_COMB9GCTKD_s200_o0"}
gl = pd.read_parquet(ROOT / "results/total_bias_decomp/game_level_v1.parquet")
pg = pd.read_parquet(ROOT / "data/processed/truth/player_game_v2.parquet",
                     columns=["game_id", "season", "athlete_id", "team_id", "ftm", "fta", "fta_tech", "ftm_tech"])
rows = []
for season, tag in RUNS.items():
    g = gl[gl.season == season][["game_id", "dss_b", "opener"]]
    p = pd.read_parquet(ROOT / f"results/engine_v0/{tag}/players.parquet",
                        columns=["game_id", "athlete_id", "team_id", "pts", "fta", "fgm2_rim", "fgm2_jump", "fgm3"])
    p["ftm"] = p.pts - 2 * (p.fgm2_rim + p.fgm2_jump) - 3 * p.fgm3
    p["anon"] = p.athlete_id <= 0
    s = p.groupby(["game_id", "athlete_id", "anon"], as_index=False)[["fta", "ftm"]].sum()
    nseed = 200
    s[["fta", "ftm"]] /= nseed
    s = s.merge(g, on="game_id")
    t = pg[pg.season == season].copy()
    t["fta"] -= t.fta_tech.fillna(0); t["ftm"] -= t.ftm_tech.fillna(0)
    lg = t.ftm.sum() / t.fta.sum()
    tal = t.groupby("athlete_id")[["ftm", "fta"]].sum()
    tal = ((tal.ftm + 20 * lg) / (tal.fta + 20)).rename("talent")
    s = s.merge(tal, left_on="athlete_id", right_index=True, how="left")
    t = t.merge(g, on="game_id").merge(tal, left_on="athlete_id", right_index=True, how="left")
    for b, sb in list(s.groupby("dss_b")) + [("opener", s[s.opener])]:
        tb = t[t.dss_b == b] if b != "opener" else t[t.opener]
        nm = sb[~sb.anon]
        nmt = nm[nm.talent.notna()]
        rows.append(dict(season=season, bucket=b,
                         sim_ft=sb.ftm.sum() / sb.fta.sum(), act_ft=tb.ftm.sum() / tb.fta.sum(),
                         sim_anon_fta_share=sb[sb.anon].fta.sum() / sb.fta.sum(),
                         sim_anon_ft=sb[sb.anon].ftm.sum() / max(sb[sb.anon].fta.sum(), 1e-9),
                         sim_named_ft=nm.ftm.sum() / nm.fta.sum(),
                         sim_named_talent=(nmt.fta * nmt.talent).sum() / nmt.fta.sum(),
                         sim_named_ft_matched=nmt.ftm.sum() / nmt.fta.sum(),
                         act_talent=(tb.fta * tb.talent).sum() / tb.fta.sum(),
                         sim_fta_pg=sb.fta.sum() / sb.game_id.nunique(), act_fta_pg=tb.fta.sum() / tb.game_id.nunique()))
df = pd.DataFrame(rows)
df["named_make_bias"] = df.sim_named_ft_matched - df.sim_named_talent
df["act_luck"] = df.act_ft - df.act_talent
pd.set_option("display.width", 250)
print(df.round(4).to_string(index=False))
df.to_csv(ROOT / "results/total_bias_decomp/ft_slots_v1.csv", index=False)
