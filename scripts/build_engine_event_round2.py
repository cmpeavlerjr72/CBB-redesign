"""
build_engine_event_round2.py -- persist the possession-outcome ROUND-2 winners
as engine artifacts, including the S1 monthly refit schedule.

Round 2 (`docs/models/possession_outcome/experiments.md` section 4,
`data/processed/models/possession_outcome/round2/verdict.json`) selected:

    first  ->  arm `lgbm`,    feature set C_plus_state, scheme S1
    cont   ->  arm `cascade`, feature set C_plus_state, scheme S1

The round-2 trainer persists NO booster (`round2/` holds metrics, the verdict
and the scheme ladder, and nothing fitted), exactly as `train_rebound_v1.py`
and `train_free_throw_v1.py` do not. So the engine refits the winners' OWN
specs, through the winners' own module, into the engine's own directory, and
never touches `possession_outcome`'s artifacts. That is the same rule
`build_engine_inputs.py` already applies to rebound and free_throw.

WHAT S1 IS, AND WHY THIS SCRIPT WRITES SEVERAL MODELS PER POPULATION
--------------------------------------------------------------------
S1 is not a model, it is a SCHEDULE. `PO.fit_predict_scheme(..., "S1", ...)`
refits at the first day of every calendar month containing a test-season game,
on all prior seasons plus the test season STRICTLY BEFORE that date, and scores
each test game with the most recent refit AT OR BEFORE its own game date.

A simulator cannot call `fit_predict_scheme`: it has no test frame to partition.
So this script replays the schedule offline and persists one fitted model per
refit date, plus, for every game in the engine's universe, the index of the
refit that game must use. The engine then only indexes. The legality property
S1 exists to give -- no game is ever scored by a model that saw data at or
after its own tipoff -- is re-asserted here per GAME rather than per month
(`_assert_no_leak`), because the engine's universe is not the bake-off's test
frame and the two could in principle disagree.

WHY THE TEAM-FORM BLOCK IS REBUILT TOO
--------------------------------------
Round 2 changed the event layer (the location-based rim override), the style-
rate source (`first_chance` rather than `all_chances`) and the universe
(`pbp_complete`). The 14 team features therefore carry DIFFERENT NUMBERS under
the same names: matched team-game to team-game on season 2025 the correlation
is 0.956-0.978 and the mean absolute difference is about 0.2 of a standard
deviation, with `off_3pa_c`'s own SD shrinking 5.25 -> 4.65. Feeding a round-2
model the round-1 columns would be a silent train/serve skew, so the round-2
adapter carries its own (G, 2, 14) block built from `round2/design.parquet`.
`data/processed/models/engine/arrays_F2_2025.npz` is NOT modified -- the
`reference` arm keeps reading exactly what it read before.

Usage:
    .venv/Scripts/python.exe scripts/build_engine_event_round2.py --fold F2 --season 2025
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.models import possession_outcome as PO

R2_DESIGN = Path("data/processed/models/possession_outcome/round2/design.parquet")
R2_VERDICT = Path("data/processed/models/possession_outcome/round2/verdict.json")
ENGINE_DIR = Path("data/processed/models/engine")

FOLD_TRAIN = {"F1": [2022, 2023], "F2": [2022, 2023, 2024]}
FOLD_TEST = {"F1": 2024, "F2": 2025}

#: The 14 team-level columns of C_plus_state. `season_idx` and
#: `days_since_start` are in here too: they are constant within a game and the
#: engine treats them as static, exactly as the `reference` arm does.
TEAM_COLS: tuple[str, ...] = (
    "off_3pa_c", "off_rim_c", "off_tov_c", "off_ftr_c",
    "opp_def_3pa_c", "opp_def_rim_c", "opp_def_tov_c", "opp_def_ftr_c",
    "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
    "site_home", "site_away", "season_idx", "days_since_start",
)


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def winners_from_verdict() -> dict[str, dict]:
    """The winning (arm, feature set, scheme) per population, READ from round
    2's own verdict file rather than restated here. A verdict that does not say
    S1 is a hard error: this script exists to serve the S1 winners and must not
    quietly serve something else."""
    v = json.loads(R2_VERDICT.read_text(encoding="utf-8").replace("NaN", "null"))
    out = {}
    for pop in ("first", "cont"):
        w = v[pop]["winner"]
        if w["scheme_selected"] != "S1":
            raise SystemExit(
                f"round-2 verdict selects scheme {w['scheme_selected']!r} for population "
                f"{pop!r}, not S1; this builder implements the S1 schedule only")
        out[pop] = {"arm": w["arm"], "feature_set": w["feature_set"],
                    "scheme": w["scheme_selected"], "log_loss": w["log_loss"]}
    return out


def _assert_no_leak(game_dates: pd.Series, seg_of_game: np.ndarray,
                    max_train_date: list[pd.Timestamp], label: str) -> None:
    """Per GAME: the model that game will be scored by was fitted only on games
    strictly earlier than that game's own date. This is the honest-backtest rule
    (`CLAUDE.md`: "every backtest row must satisfy created_at < tipoff") applied
    to model provenance and enforced in code, not assumed from the schedule."""
    d = pd.to_datetime(game_dates).to_numpy()
    mt = np.array([np.datetime64(pd.Timestamp(x), "ns") for x in max_train_date])
    bad = mt[seg_of_game] >= d
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(
            f"{label}: S1 selection would score a game tipping {pd.Timestamp(d[i])} with a "
            f"model trained through {pd.Timestamp(mt[seg_of_game][i])} -- that is a leak")


def build(fold: str, season: int, seed: int = 0) -> Path:
    t0 = time.time()
    train_seasons = FOLD_TRAIN[fold]
    if season != FOLD_TEST[fold]:
        raise SystemExit(f"fold {fold} tests season {FOLD_TEST[fold]}, not {season}")
    for s in [*train_seasons, season]:
        assert_not_sealed(s)

    win = winners_from_verdict()
    log(f"round-2 winners: first={win['first']['arm']}/{win['first']['scheme']}, "
        f"cont={win['cont']['arm']}/{win['cont']['scheme']}", t0)

    out = ENGINE_DIR / f"event_round2_s1_{fold}_{season}"
    out.mkdir(parents=True, exist_ok=True)

    games = pd.read_parquet(ENGINE_DIR / f"games_{fold}_{season}.parquet")
    gdate = pd.to_datetime(games["game_date"] if "game_date" in games
                           else games["date"])
    G = len(games)
    log(f"engine universe: {G} games, {gdate.min().date()} .. {gdate.max().date()}", t0)

    # ---- 1. the round-2 team-form block, on the engine's game order --------
    key = ["game_id", "offense_team_id", "season", "game_date"]
    form = pd.read_parquet(R2_DESIGN, columns=key + list(TEAM_COLS))
    form = form.drop_duplicates(["game_id", "offense_team_id"])
    log(f"round-2 team form: {len(form):,} team-games over {form.season.nunique()} seasons", t0)

    fs = form[form["season"] == season]
    gpos = {int(g): i for i, g in enumerate(games["game_id"].to_numpy())}
    team_block = np.full((G, 2, len(TEAM_COLS)), np.nan, dtype=np.float32)
    gi = fs["game_id"].map(gpos)
    ok = gi.notna().to_numpy()
    gi = gi[ok].to_numpy().astype(np.int64)
    tid = fs.loc[ok, "offense_team_id"].to_numpy()
    home = games["home_team_id"].to_numpy()[gi]
    side = np.where(tid == home, 0, 1).astype(np.int64)
    team_block[gi, side] = fs.loc[ok, list(TEAM_COLS)].to_numpy(dtype=np.float32)

    # A team-game the round-2 design does not cover (pbp-incomplete) takes that
    # team's most recent EARLIER covered game -- at most one of its own games
    # stale, never forward-looking. This is `build_engine_inputs.py`'s own
    # fg_make fallback, reused rather than reinvented. Falling back to the
    # ROUND-1 columns would be worse than stale: it would be a different
    # quantity under the same name, which is the skew this block exists to
    # avoid.
    miss = ~np.isfinite(team_block).all(axis=2)
    n_missing = int(miss.sum())
    n_filled = 0
    if n_missing:
        side_team = np.where(
            np.arange(2)[None, :] == 0,
            games["home_team_id"].to_numpy()[:, None],
            games["away_team_id"].to_numpy()[:, None])
        want = pd.DataFrame({
            "row": np.flatnonzero(miss.ravel()),
            "team_id": side_team.ravel()[np.flatnonzero(miss.ravel())],
            "game_date": np.repeat(gdate.to_numpy(), 2)[np.flatnonzero(miss.ravel())],
        }).sort_values("game_date")
        have = (form.rename(columns={"offense_team_id": "team_id"})
                    .assign(game_date=lambda d: pd.to_datetime(d["game_date"]))
                    .sort_values("game_date"))
        got = pd.merge_asof(want, have[["team_id", "game_date", *TEAM_COLS]],
                            on="game_date", by="team_id", direction="backward")
        vals = got[list(TEAM_COLS)].to_numpy(dtype=np.float32)
        good = np.isfinite(vals).all(axis=1)
        flat = team_block.reshape(G * 2, -1)
        flat[got.loc[good, "row"].to_numpy()] = vals[good]
        n_filled = int(good.sum())
        team_block = flat.reshape(G, 2, -1)
    # anything still unfilled falls to the models' own no-history state: a
    # centred rate of 0.0, which on a league-centred scale IS the league mean.
    n_zero = int((~np.isfinite(team_block).all(axis=2)).sum())
    team_block = np.nan_to_num(team_block, nan=0.0).astype(np.float32)
    log(f"team block: {n_missing} team-games uncovered, {n_filled} filled as-of backward, "
        f"{n_zero} left at the league mean", t0)

    # ---- 2. the S1 refit schedule -----------------------------------------
    design = pd.read_parquet(R2_DESIGN)
    design["game_date"] = pd.to_datetime(design["game_date"])
    cuts = PO.month_boundaries(design.loc[design["season"] == season, "game_date"])
    log(f"S1 refit dates: {[str(c.date()) for c in cuts]}", t0)

    # per engine game: the index of the most recent refit AT OR BEFORE its date
    cut_ns = np.array([np.datetime64(c, "ns") for c in cuts])
    seg_of_game = np.searchsorted(cut_ns, gdate.to_numpy(), side="right") - 1
    if (seg_of_game < 0).any():
        raise AssertionError("a game in the engine universe precedes the first S1 refit date")

    index: dict = {
        "fold": fold, "season": season, "seed": seed,
        "scheme": "S1",
        "source": "possession_outcome round 2 (verdict.json); no booster is persisted by "
                  "scripts/train_possession_outcome_v2.py, so the engine refits the winners' "
                  "own specs through cbb_sim.models.possession_outcome",
        "winners": win,
        "team_cols": list(TEAM_COLS),
        "team_block_provenance": {
            "design": str(R2_DESIGN), "style_source": "first_chance",
            "require_pbp_complete": True, "possessions_version": "v2",
            "n_team_games_uncovered": n_missing, "n_filled_asof_backward": n_filled,
            "n_left_at_league_mean": n_zero,
            "why": "round 2 changed the event layer, the style-rate source and the universe, so "
                   "the same feature NAMES carry different numbers than round 1's design; the "
                   "round-2 arms must be served the columns they were trained on",
        },
        "refit_dates": [str(c.date()) for c in cuts],
        "populations": {},
    }

    for pop in ("first", "cont"):
        arm = win[pop]["arm"]
        feats = PO.feature_set(win[pop]["feature_set"], pop)
        tr = design[(design["season"].isin(train_seasons)) & (design["population"] == pop)]
        te = design[(design["season"] == season) & (design["population"] == pop)]
        te_dates = te["game_date"]
        fit_cols = [*feats, "y", "season"]
        segs = []
        for k, cut in enumerate(cuts):
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, fit_cols]
            rows = tr[fit_cols] if not len(prior) else pd.concat(
                [tr[fit_cols], prior], ignore_index=True)
            model = PO.fit_arm(arm, rows, feats, seed=seed)
            for b in (getattr(model, "clf_", None),):
                try:
                    b.set_params(n_jobs=1)
                except Exception:                                  # noqa: BLE001
                    pass
            p = out / f"{pop}_{cut.date()}.joblib"
            joblib.dump({"model": model, "arm": arm, "features": feats,
                         "refit_date": str(cut.date())}, p, compress=3)
            mt = (tr["game_date"].max() if not before.any()
                  else max(tr["game_date"].max(), te_dates[before].max()))
            segs.append({"refit_date": str(cut.date()), "file": p.name,
                         "n_train": int(len(rows)),
                         "n_train_from_test_season": int(len(prior)),
                         "max_train_date": str(pd.Timestamp(mt).date()),
                         "size_mb": round(p.stat().st_size / 1e6, 2)})
            log(f"  {pop} {cut.date()}: fit {arm} on {len(rows):,} rows "
                f"(through {pd.Timestamp(mt).date()}) -> {p.name}", t0)
        _assert_no_leak(gdate, seg_of_game,
                        [pd.Timestamp(s["max_train_date"]) for s in segs], f"{pop}/S1")
        index["populations"][pop] = {"arm": arm, "feature_set": win[pop]["feature_set"],
                                     "features": feats, "segments": segs}

    np.savez_compressed(out / "team_block.npz", team_block=team_block,
                        seg_of_game=seg_of_game.astype(np.int16))
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    log(f"wrote {out} ({sum(f.stat().st_size for f in out.iterdir()) / 1e6:.1f} MB)", t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2", choices=sorted(FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.fold, a.season, a.seed)


if __name__ == "__main__":
    main()
