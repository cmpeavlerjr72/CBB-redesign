"""diag_c4_extract_v1.py -- cache per-team-game, per-game-seed and late-game frames for sim parts and for the real side.
DIAGNOSTIC ONLY.  usage: diag_c4_extract_v1.py --tag f2all50 --parts f2all50_p0 f2all50_p1 ...   (or --traj-file path --tag x)
Writes results/diag_c4/<tag>_{sim_tg,sim_gs,sim_late,real_tg,real_gs,real_late,pregame}.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_c4_lib_v1 as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--traj-file", nargs="*", default=[])
    a = ap.parse_args()
    files = [L.ROOT / "results/trajectories/_runs" / p / "trajectory.parquet" for p in a.parts] + [Path(f) for f in a.traj_file]
    tg, gs, late, ids = [], [], [], []
    for f in files:
        t = pd.read_parquet(f)
        s = L.std_from_sim(t)
        del t
        tg.append(L.team_game(s))
        gs.append(L.game_seed(s))
        late.append(L.late_frame(s, within=300))
        ids.append(s.game_id.unique())
        print("sim part", f, len(s), flush=True)
        del s
    import numpy as np
    gids = np.unique(np.concatenate(ids))
    L.OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(tg, ignore_index=True).to_parquet(L.OUT / f"{a.tag}_sim_tg.parquet")
    pd.concat(gs, ignore_index=True).to_parquet(L.OUT / f"{a.tag}_sim_gs.parquet")
    pd.concat(late, ignore_index=True).to_parquet(L.OUT / f"{a.tag}_sim_late.parquet")
    r = L.std_from_real(gids)
    L.team_game(r).to_parquet(L.OUT / f"{a.tag}_real_tg.parquet")
    L.game_seed(r).to_parquet(L.OUT / f"{a.tag}_real_gs.parquet")
    L.late_frame(r, within=300).to_parquet(L.OUT / f"{a.tag}_real_late.parquet")
    import diag_trajectory_vs_pbp_v1 as T
    pg = T.pregame_tables(set(int(g) for g in gids), L.SEASON)
    u = pd.read_parquet(L.ROOT / "data/processed/games_universe_v2.parquet",
                        columns=["game_id", "neutral_site", "season_type", "home_score", "away_score", "n_periods"])
    fin = pd.read_parquet(L.ROOT / "data/processed/truth/game_finals_v2.parquet",
                          columns=["game_id", "home_score", "away_score", "n_periods"]).rename(
        columns={"home_score": "fin_h", "away_score": "fin_a", "n_periods": "fin_np"})
    sc = pd.read_parquet(L.ROOT / f"data/raw/hoopr/schedules/mbb_schedule_{L.SEASON}.parquet",
                         columns=["game_id", "conference_competition", "home_conference_id", "away_conference_id"])
    pg = (pg.merge(u[["game_id", "neutral_site", "season_type"]], on="game_id", how="left")
            .merge(fin, on="game_id", how="left").merge(sc.drop_duplicates("game_id"), on="game_id", how="left"))
    pg["same_conf"] = (pg.home_conference_id == pg.away_conference_id) & pg.home_conference_id.notna()
    pg.to_parquet(L.OUT / f"{a.tag}_pregame.parquet")
    print("done", len(gids), "games", flush=True)


if __name__ == "__main__":
    main()
