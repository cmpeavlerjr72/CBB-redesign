#!/usr/bin/env python
"""
build_own_ratings_C_v1.py -- Lane N (2026-09-30): CHAINED arm-C own ratings, a versioned sibling of the stored ratings.

    .venv/Scripts/python.exe scripts/build_own_ratings_C_v1.py [--out-dir data/processed/ratings_C_v1]

Arm C (docs/models/own_ratings/experiments.md sections 1-2, selected offline by the PM): the efficiency team-effect
prior for season S is

    prior_i = cm_i + w_c * (c_prev_i - cm_i)

  c_prev_i : team i's season S-1 FINAL centred off / def effect, from THIS chain (chained-C, not R);
  cm_i     : mean of c_prev over the members of team i's season-S conference that have an S-1 rating
             (membership = the team's modal conference id on the hoopR season-S schedule, i.e. the NEW conference
             for a team that realigned; a preseason-public attribute; previous-season finals only);
  w_c      : 0.6419 (fitted on the fold-2 training transitions 2022->23, 2023->24);
  new-to-D-I teams (no S-1 rating): prior = cm_i; a team with no conference on the schedule: cm_i = 0 (league mean).

Everything else is the library's `fit_season` unchanged (lambda 5, fixed-term priors from the S-1 final, tempo carried at
w 0.8, re-centring on the season's team set). The prior is handed to `fit_season` as the team-effect vector with w_eff = 1.
Season 2022 has no prior and is identical to R. Season 2026 is SEALED and not built.
Writes own_ratings_{S}.parquet (batch schema) and own_ratings_manifest.json under --out-dir (refuses data/processed/ratings).
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
from cbb_sim.ratings import own_ratings as orat  # noqa: E402

W_C = 0.6419186589419831          # results/own_ratings_day1/params_v1.json F2 w_C
LAM = 5.0
W_TEMPO = 0.8
SEASONS = [2022, 2023, 2024, 2025]


def conference_map(season: int, root: Path = REPO) -> pd.Series:
    s = pd.read_parquet(root / f"data/raw/hoopr/schedules/mbb_schedule_{season}.parquet",
                        columns=["home_id", "away_id", "home_conference_id", "away_conference_id"])
    a = pd.concat([s[["home_id", "home_conference_id"]].set_axis(["team_id", "conf"], axis=1),
                   s[["away_id", "away_conference_id"]].set_axis(["team_id", "conf"], axis=1)]).dropna()
    a["team_id"] = a["team_id"].astype("int64")
    return a.groupby("team_id")["conf"].agg(lambda x: x.mode().iloc[0])


def c_prior(prior_final_eff: dict, teams: np.ndarray, conf: pd.Series, w_c: float = W_C) -> dict:
    """The arm-C prior as a prior dict for fit_season(..., w_eff=1.0)."""
    off_p, def_p = prior_final_eff["team_off"], prior_final_eff["team_def"]
    has = pd.Series(teams).isin(off_p.index).to_numpy()
    o = off_p.reindex(teams).fillna(0.0).to_numpy(float)
    d = def_p.reindex(teams).fillna(0.0).to_numpy(float)
    cf = conf.reindex(teams).to_numpy()
    df = pd.DataFrame({"conf": cf, "o": o, "d": d, "has": has})
    g = df[df["has"] & df["conf"].notna()].groupby("conf")[["o", "d"]].mean()
    cm_o = df["conf"].map(g["o"]).fillna(0.0).to_numpy(float)
    cm_d = df["conf"].map(g["d"]).fillna(0.0).to_numpy(float)
    po = np.where(has, cm_o + w_c * (o - cm_o), cm_o)
    pdf = np.where(has, cm_d + w_c * (d - cm_d), cm_d)
    return {"team_off": pd.Series(po, index=teams), "team_def": pd.Series(pdf, index=teams),
            "intercept": prior_final_eff["intercept"], "home": prior_final_eff["home"], "away": prior_final_eff["away"]}


def chain_C(tg: pd.DataFrame, seasons: list[int], root: Path = REPO, w_c: float = W_C):
    """Yields (season, run, teams, prior_used) with each season's prior from the previous CHAINED-C final."""
    prior_e = prior_t = None
    out = {}
    for s in seasons:
        teams = np.sort(np.union1d(tg.loc[tg["season"] == s, "team_id"].unique(),
                                   tg.loc[tg["season"] == s, "opp_team_id"].unique()))
        pe = c_prior(prior_e, teams, conference_map(s, root), w_c) if prior_e is not None else None
        run = orat.fit_season(tg, s, LAM, LAM, pe, prior_t, 1.0, W_TEMPO)
        out[s] = (run, pe)
        prior_e, prior_t = run.final["eff"], run.final["tempo"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "data/processed/ratings_C_v1"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    if out.resolve() == (REPO / "data/processed/ratings").resolve():
        raise SystemExit("refusing to write into data/processed/ratings")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: never overwrite; use a new version")
    t0 = time.time()
    os.chdir(REPO)
    uni = orat.load_universe()
    uni = uni[uni["season"].isin(SEASONS)]
    tg = orat.load_team_games(uni, SEASONS)
    assert not (tg["season"] >= 2026).any()
    res = chain_C(tg, SEASONS)
    out.mkdir(parents=True, exist_ok=True)
    for s, (run, _) in res.items():
        fr = orat.run_to_frame(run)
        fr.to_parquet(out / f"own_ratings_{s}.parquet", index=False)
        print(f"{s}: {len(fr)} rows, {fr['as_of_date'].nunique()} dates, {time.time()-t0:.0f}s", flush=True)
    man = {"created_at": pd.Timestamp.now("UTC").isoformat(), "seasons": SEASONS, "arm": "C (chained)",
           "hyperparameters": {"lambda_eff": LAM, "lambda_tempo": LAM, "prior_weight_eff": "arm C: cm + w_c (c_prev - cm)",
                               "w_c": W_C, "prior_weight_tempo": W_TEMPO},
           "conference": "modal home/away_conference_id of the team on the hoopR season-S schedule (new conference after realignment)",
           "conference_mean": "mean of the chained season S-1 final off/def over season-S conference members with an S-1 rating",
           "new_d1_prior": "conference mean; no conference -> 0 (league mean)",
           "source": "scripts/build_own_ratings_C_v1.py", "sealed_not_built": [2026],
           "selection": "docs/models/own_ratings/experiments.md sections 1-2 (offline), section 3 closed loop"}
    (out / "own_ratings_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
