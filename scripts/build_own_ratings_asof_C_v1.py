#!/usr/bin/env python
"""
build_own_ratings_asof_C_v1.py -- daily entry point for the CHAINED arm-C own ratings (Lane N, 2026-09-30).
Sibling of `build_own_ratings_asof_v1.py` (not edited; its team-set and stub helpers are imported).

    .venv/Scripts/python.exe scripts/build_own_ratings_asof_C_v1.py --season 2025 --as-of 2024-11-04 --out-dir <dir>
    .venv/Scripts/python.exe scripts/build_own_ratings_asof_C_v1.py --parity    # 3 dates per season vs data/processed/ratings_C_v1

Same procedure as the R entry point: (1) the prior chain from 2022 to S-1 is refitted in full, each season primed by the
PREVIOUS CHAINED-C final through `build_own_ratings_C_v1.c_prior` (conference = season-S schedule, w_c 0.6419);
(2) season S is fitted on games strictly before D plus stub games at D; (3) the D rows are written. Chaining through the
sealed 2026 season needs CBB_UNSEAL=1 (PM decision). Refuses data/processed/ratings and data/processed/ratings_C_v1 as output.
For a live season whose hoopR schedule is thin, teams with no conference fall back to cm = 0 (league mean).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.ratings import own_ratings as orat  # noqa: E402
import build_own_ratings_asof_v1 as B  # noqa: E402
import build_own_ratings_C_v1 as C  # noqa: E402

SIBLING = REPO / "data/processed/ratings_C_v1"


def asof_ratings_C(season: int, as_of: str, root: Path = REPO, teams_source: str = "tg", chain_start: int = 2022,
                   cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    D = pd.Timestamp(as_of)
    if any(s >= 2026 for s in range(chain_start, season + 1)):
        assert_not_sealed(list(range(max(chain_start, 2026), season + 1)), context="own ratings C chain through 2025-26")
    hoopr_dir = root / "data/raw/hoopr"
    uni = orat.load_universe(root / "data/processed/games_universe.parquet")
    seasons = [s for s in range(chain_start, season + 1)
               if (hoopr_dir / "team_box" / f"team_box_{s}.parquet").exists() and (uni["season"] == s).any()]
    tkey = ("tg", tuple(seasons))
    if cache is not None and tkey in cache:
        tg = cache[tkey]
    else:
        tg = orat.load_team_games(uni, seasons, hoopr_dir)
        if cache is not None:
            cache[tkey] = tg
    chain = [s for s in seasons if s < season]
    if chain and chain != list(range(chain[0], season)):
        raise RuntimeError(f"prior chain has a gap: {chain} before {season}")
    key = ("chain", tuple(chain))
    if cache is not None and key in cache:
        prior_e, prior_t = cache[key]
    else:
        prior_e = prior_t = None
        if chain:
            res = C.chain_C(tg, chain, root)
            last = res[chain[-1]][0]
            prior_e, prior_t = last.final["eff"], last.final["tempo"]
        if cache is not None:
            cache[key] = (prior_e, prior_t)
    cur_all = tg[tg["season"] == season]
    teams = B.season_teams(teams_source, cur_all, season, root)
    cur = cur_all[cur_all["game_date"] < D]                      # STRICTLY BEFORE D
    assert len(cur) == 0 or cur["game_date"].max() < D
    stubs = B.stub_games(teams, D, season)
    cols = [c for c in cur.columns if c in stubs.columns]
    panel = pd.concat([cur[cols], stubs[cols]], ignore_index=True) if len(cur) else stubs
    panel = panel.sort_values(["season", "game_date", "game_id", "team_id"], kind="mergesort").reset_index(drop=True)
    pe = C.c_prior(prior_e, teams, C.conference_map(season, root)) if prior_e is not None else None
    run = orat.fit_season(panel, season, C.LAM, C.LAM, pe, prior_t, 1.0, C.W_TEMPO)
    fr = orat.run_to_frame(run)
    out = fr[fr["as_of_date"] == D].reset_index(drop=True)
    prov = {"season": season, "as_of": str(D.date()), "arm": "C chained", "w_c": C.W_C, "teams_source": teams_source,
            "n_teams": int(len(teams)), "n_source_games_in_season": int(cur["game_id"].nunique()) if len(cur) else 0,
            "prior_chain_seasons": chain, "seconds": round(time.time() - t0, 1)}
    return out, prov


def parity() -> int:
    cache: dict = {}
    rows = []
    num = ["off_c", "def_c", "tempo_rel", "n_games", "league_off_mean", "league_tempo_mean", "home_off_eff",
           "away_off_eff", "neutral_tempo_eff", "n_games_window"]
    for s in C.SEASONS:
        st = pd.read_parquet(SIBLING / f"own_ratings_{s}.parquet")
        dates = sorted(st["as_of_date"].unique())
        for d in (dates[0], dates[len(dates) // 3], dates[(2 * len(dates)) // 3]):
            fr, prov = asof_ratings_C(s, str(pd.Timestamp(d).date()), cache=cache)
            ref = st[st["as_of_date"] == d].set_index("team_id").sort_index()
            got = fr.set_index("team_id").sort_index()
            same_set = ref.index.equals(got.index)
            mx = float(np.nanmax(np.abs(ref[num].to_numpy(float) - got.loc[ref.index, num].to_numpy(float))))
            rows.append({"season": s, "as_of": str(pd.Timestamp(d).date()), "n_teams": len(ref),
                         "season_games_before": prov["n_source_games_in_season"], "team_set_equal": bool(same_set),
                         "max_abs_diff": mx})
            print(rows[-1], flush=True)
    p = REPO / "docs/ops/own_ratings_C_asof_parity_2026-09-30.json"
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = all(r["team_set_equal"] and r["max_abs_diff"] < 1e-9 for r in rows)
    print("PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--as-of")
    ap.add_argument("--teams-source", default="tg")
    ap.add_argument("--out-dir")
    ap.add_argument("--parity", action="store_true")
    a = ap.parse_args()
    if a.parity:
        return parity()
    out_dir = Path(a.out_dir)
    for bad in (REPO / "data/processed/ratings", SIBLING):
        if out_dir.resolve() == bad.resolve():
            raise SystemExit(f"refusing to write into {bad}")
    cache: dict = {}
    frames, provs = [], []
    for d in a.as_of.split(","):
        fr, prov = asof_ratings_C(a.season, d.strip(), REPO, a.teams_source, 2022, cache)
        frames.append(fr)
        provs.append(prov)
    out = pd.concat(frames, ignore_index=True)
    out["created_at"] = pd.Timestamp.now("UTC")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"own_ratings_{a.season}.parquet", index=False)
    (out_dir / f"own_ratings_{a.season}_provenance.json").write_text(json.dumps(provs, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
