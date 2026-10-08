#!/usr/bin/env python
"""
build_own_ratings_asof_v1.py -- daily entry point for our own ridge team ratings, as of any date of any season
(Lane K part 2, 2026-09-30).

    .venv/Scripts/python.exe scripts/build_own_ratings_asof_v1.py --season 2025 --as-of 2025-01-16 --out-dir <dir>
    .venv/Scripts/python.exe scripts/build_own_ratings_asof_v1.py --season 2027 --as-of 2026-11-02 \
        --teams-source schedule --out-dir data/processed/ratings_asof          # day 1 of 2026-27 (see the design note)

WHAT IT DOES. `build_own_ratings.py` is a batch job over whole seasons and keeps no coefficients, so there is no way to
ask "the ratings on date D" for a season with no (or only part of a) file. This script rebuilds exactly what the batch
job would have stored in the row `as_of_date == D`, with the SAME library code (`cbb_sim.ratings.own_ratings`):

  1. the prior chain: every season from --chain-start (2022) up to season-1 is fitted in full, in order, each season's
     FINAL fit being the next season's prior (`fit_season(..., prior_eff, prior_tempo, w)`), hyperparameters from
     `data/processed/ratings/own_ratings_manifest.json` (lambda 5 / 5, prior weight 0.8 / 0.8, selected on 2022-23);
  2. season `--season` is fitted on the games with game_date STRICTLY BEFORE D only (asserted), plus one STUB game at D
     per pair of teams. The fit for date D is taken before D's own rows are folded in, so a stub's content never
     touches the output (the same device the live inputs builder uses); the stubs only put D and every team of the
     season on the grid;
  3. the row(s) with as_of_date == D are written to `<out-dir>/own_ratings_{season}.parquet` with the batch schema, plus a
     provenance json. Nothing under data/processed/ratings is ever overwritten.

THE TEAM SET. The ridge re-centres every rating on the mean over the season's teams, so the set of teams matters.
`--teams-source tg` (parity mode) = the teams of the season's completed games, which is what the batch file used (it saw
the whole season); `schedule` = teams on the hoopR schedule for the season with a conference id on both sides
(the only choice available before a season has games); `file:<path>` = a parquet/csv with a `team_id` column; `cbbd:<games.parquet>|<crosswalk.parquet>` = teams with both sides
mapped in a CBBD /games dump (2027 today: 365 teams from 5,286 games, vs 319 from the thin hoopR schedule).

SEAL. Chaining through season 2026 (the sealed 2025-26 season) is required for any 2027 date. The script refuses unless
CBB_UNSEAL=1 (cbb_sim.data.seal), because that is the PM's decision, not a script default.
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

from cbb_sim.data.seal import assert_not_sealed, assert_not_sealed_serving  # noqa: E402
from cbb_sim.ratings import own_ratings as orat  # noqa: E402


def _hp(root: Path) -> dict:
    m = json.loads((root / "data/processed/ratings/own_ratings_manifest.json").read_text(encoding="utf-8"))
    return m["hyperparameters"]


def season_teams(source: str, tg_s: pd.DataFrame, season: int, root: Path) -> np.ndarray:
    if source == "tg":
        return np.sort(np.union1d(tg_s["team_id"].unique(), tg_s["opp_team_id"].unique()))
    if source == "schedule":
        s = pd.read_parquet(root / f"data/raw/hoopr/schedules/mbb_schedule_{season}.parquet",
                            columns=["home_id", "away_id", "home_conference_id", "away_conference_id"])
        s = s[s["home_conference_id"].notna() & s["away_conference_id"].notna()]
        return np.sort(np.union1d(s["home_id"].dropna().astype("int64").unique(), s["away_id"].dropna().astype("int64").unique()))
    if source.startswith("cbbd:"):              # cbbd:<CBBD /games parquet>:<team crosswalk parquet>
        gp, cp = source[5:].split("|") if "|" in source[5:] else source[5:].rsplit(":", 1)   # "|" first: Windows drive colons break the ":" split
        g = pd.read_parquet(gp)
        cw = pd.read_parquet(cp)
        cb = "cbbd_team_id" if "cbbd_team_id" in cw.columns else "cbbd_id"
        es = "espn_team_id" if "espn_team_id" in cw.columns else "espn_id"
        m = dict(zip(cw[cb].astype("int64"), cw[es].astype("int64")))
        h, a = g["homeTeamId"].astype("int64").map(m), g["awayTeamId"].astype("int64").map(m)
        ok = h.notna() & a.notna()
        return np.sort(np.union1d(h[ok].astype("int64").unique(), a[ok].astype("int64").unique()))
    if source.startswith("file:"):
        p = Path(source[5:])
        t = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        return np.sort(t["team_id"].astype("int64").unique())
    raise ValueError(source)


def stub_games(teams: np.ndarray, date: pd.Timestamp, season: int) -> pd.DataFrame:
    """Two team-game rows per stub game; teams paired consecutively (odd one out paired with the first)."""
    rows = []
    t = list(teams)
    pairs = [(t[i], t[i + 1]) for i in range(0, len(t) - 1, 2)]
    if len(t) % 2:
        pairs.append((t[-1], t[0]))
    for k, (h, a) in enumerate(pairs):
        gid = -(k + 1)
        for team, opp, home in ((h, a, True), (a, h, False)):
            rows.append({"game_id": gid, "season": season, "game_date": date, "team_id": int(team), "opp_team_id": int(opp),
                         "home_team_id": int(h), "away_team_id": int(a), "neutral_site": False,
                         "site_home": float(home), "site_away": float(not home),
                         "off_eff": 0.0, "game_poss": 0.0})
    return pd.DataFrame(rows)


def asof_ratings(season: int, as_of: str, root: Path, teams_source: str = "tg", chain_start: int = 2022,
                 tg_cache: dict | None = None, serving: bool = False) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    D = pd.Timestamp(as_of)
    hp = _hp(root)
    if any(s >= 2026 for s in range(chain_start, season + 1)):
        # serving=True only from the daily 2027 ratings stage
        _ctx = "own ratings chain through the sealed 2025-26 season"
        _yrs = list(range(max(chain_start, 2026), season + 1))
        assert_not_sealed_serving(_yrs, season, context=_ctx) if serving else assert_not_sealed(_yrs, context=_ctx)
    hoopr_dir = root / "data/raw/hoopr"
    uni = orat.load_universe(root / "data/processed/games_universe.parquet")
    seasons = [s for s in range(chain_start, season + 1)
               if (hoopr_dir / "team_box" / f"team_box_{s}.parquet").exists() and (uni["season"] == s).any()]
    if tg_cache is not None and "tg" in tg_cache:
        tg = tg_cache["tg"]
    else:
        tg = orat.load_team_games(uni, seasons, hoopr_dir)
        if tg_cache is not None:
            tg_cache["tg"] = tg
    chain = [s for s in seasons if s < season]
    if chain and chain != list(range(chain[0], season)):
        raise RuntimeError(f"prior chain has a gap: {chain} before {season}")
    prior_e = prior_t = None
    finals_seasons = []
    for s in chain:
        run = orat.fit_season(tg, s, hp["lambda_eff"], hp["lambda_tempo"], prior_e, prior_t,
                              hp["prior_weight_eff"], hp["prior_weight_tempo"])
        prior_e, prior_t = run.final["eff"], run.final["tempo"]
        finals_seasons.append(s)
    cur_all = tg[tg["season"] == season]
    teams = season_teams(teams_source, cur_all, season, root)
    cur = cur_all[cur_all["game_date"] < D]                      # STRICTLY BEFORE D
    assert len(cur) == 0 or cur["game_date"].max() < D
    stubs = stub_games(teams, D, season)
    cols = [c for c in cur.columns if c in stubs.columns]
    panel = pd.concat([cur[cols], stubs[cols]], ignore_index=True) if len(cur) else stubs
    panel = panel.sort_values(["season", "game_date", "game_id", "team_id"], kind="mergesort").reset_index(drop=True)
    # stubs carry game_id < 0 and sort first within their date; the real rows of earlier dates keep the batch order
    run = orat.fit_season(panel, season, hp["lambda_eff"], hp["lambda_tempo"], prior_e, prior_t,
                          hp["prior_weight_eff"], hp["prior_weight_tempo"])
    fr = orat.run_to_frame(run)
    out = fr[fr["as_of_date"] == D].reset_index(drop=True)
    prov = {"season": season, "as_of": str(D.date()), "hyperparameters": hp, "teams_source": teams_source,
            "n_teams": int(len(teams)), "n_source_games_in_season": int(cur["game_id"].nunique()) if len(cur) else 0,
            "latest_source_game_date": str(cur["game_date"].max().date()) if len(cur) else None,
            "prior_chain_seasons": finals_seasons, "chain_start": chain_start,
            "prior_carry": ("season-1 FINAL fit, weight %.2f (eff) / %.2f (tempo)" % (hp["prior_weight_eff"], hp["prior_weight_tempo"]))
            if prior_e is not None else "none (no previous season on disk): zero prior, intercept from the window mean",
            "seconds": round(time.time() - t0, 1)}
    return out, prov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--as-of", required=True, help="comma-separated dates")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--teams-source", default="tg")
    ap.add_argument("--chain-start", type=int, default=2022)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out_dir = Path(a.out_dir)
    if out_dir.resolve() == (REPO / "data/processed/ratings").resolve():
        raise SystemExit("refusing to write into data/processed/ratings (the stored batch ratings); use a sibling dir")
    cache: dict = {}
    frames, provs = [], []
    for d in a.as_of.split(","):
        fr, prov = asof_ratings(a.season, d.strip(), root, a.teams_source, a.chain_start, cache)
        frames.append(fr)
        provs.append(prov)
        print(f"{d}: {len(fr)} teams, {prov['n_source_games_in_season']} season games before, {prov['seconds']}s")
    out = pd.concat(frames, ignore_index=True)
    out["created_at"] = pd.Timestamp.now("UTC")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"own_ratings_{a.season}.parquet", index=False)
    (out_dir / f"own_ratings_{a.season}_provenance.json").write_text(json.dumps(provs, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
