"""diag_c4_extract_chunked_v1.py -- versioned sibling of diag_c4_extract_v1.py (unchanged): the same per-team-game, per-game-seed
and late-game frames, written to the same file names, but the ONE input trajectory is read in game-id chunks (row filter), so a
5,710-game x 50-seed trajectory does not have to sit in memory at once. Every frame function is diag_c4_lib_v1's, applied per
chunk; all of them group by (game, seed[, side]) so chunking by game is exact. Real side and pregame exactly as v1.
DIAGNOSTIC ONLY (clock round 10, docs/models/clock/experiments.md section 43, full-size read).

    .venv/Scripts/python.exe scripts/diag_c4_extract_chunked_v1.py --tag <tag> --traj-file <trajectory.parquet> [--seed-max N]
`--seed-max N` keeps seeds < N only (default: all seeds in the file).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_c4_lib_v1 as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--traj-file", required=True)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--seed-max", type=int, default=None)
    a = ap.parse_args()
    dset = ds.dataset(a.traj_file, format="parquet")
    gids_all = np.unique(dset.to_table(columns=["game_id"]).column("game_id").to_numpy())
    tg, gs, late = [], [], []
    for part in np.array_split(gids_all, a.chunks):
        flt = ds.field("game_id").isin(part.tolist())
        if a.seed_max is not None:
            flt = flt & (ds.field("seed") < a.seed_max)
        t = dset.to_table(filter=flt).to_pandas()
        s = L.std_from_sim(t)
        del t
        tg.append(L.team_game(s))
        gs.append(L.game_seed(s))
        late.append(L.late_frame(s, within=300))
        print("sim chunk", len(part), "games", len(s), "rows", flush=True)
        del s
    gids = gids_all
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
