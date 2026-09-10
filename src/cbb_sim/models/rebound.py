"""
rebound.py -- L3 REBOUND sub-model: who gets the ball after a miss.

Pre-registration (verbatim, PM, 2026-09-10): `docs/models/rebound/experiments.md`
section 1. Trainer: `scripts/train_rebound_v1.py`. Feature provenance:
`docs/models/rebound/features.md`.

===========================================================================
TARGET
===========================================================================
One row per REBOUND OPPORTUNITY: a missed field-goal attempt, or the missed
LAST free throw of a non-technical trip. (A missed front end of a one-and-one
or a missed first of two is not an opportunity -- the ball is dead and the
same shooter shoots again; `cbb_sim.models.event_stream` resolves trips and
drops ESPN's administrative reset rebounds so that "the next event after the
opportunity is the rebound" is literally true.)

Three classes:

    OREB   the offence secured it            (~27-38% by miss type)
    DREB   the defence secured it            (~61-86% by miss type)
    DEAD   a dead-ball rebound               (~0.4-0.8%)

DEAD is a class and not a nuisance category because the pre-registration says
so, and because it is genuinely a third thing: the ball goes out of bounds off
somebody and is awarded at the sideline. It ends the CHANCE without a live
rebound; measured on 2024, 58.1% of dead-ball rebounds are followed by the next
real action from the SHOOTING team, so it is not a deterministic hand-back to
either side either. Section 4 of `experiments.md` reports whether the class
needs its own model or can be carried as a fixed share by miss type.

A fourth outcome exists in the feed and is NOT modelled: `unresolved`, ~0.7-1.0%
of opportunities where the next event is a foul (a loose-ball foul logged before
the rebound), a shot, or a turnover, and the rebound row never appears. Those
rows are dropped from the target with their count reported, the same treatment
`unknown` gets in L3 round 1. They are a data gap, never imputed.

===========================================================================
LEAK SAFETY
===========================================================================
Every team and player feature is an EXPANDING mean over games STRICTLY BEFORE
the current one, within season, and is then centred on the league's own as-of
mean on the same date, so no feature can carry a level that only exists after
the fact (`CLAUDE.md`: "every rating feature is expressed relative to its own
snapshot's league mean"). Three specific hazards are closed by construction:

  1. The team's own game contributes nothing to its own features
     (`prob_metrics.expanding_asof` is `cumsum() - value`).
  2. The team rates are built from FIRST-CHANCE opportunities only. A team that
     rebounds one miss and then misses again generates a second, easier
     opportunity; pooling them would make "OREB% as-of" partly a function of
     how many second chances a team happened to get, and -- the L16 channel --
     would let a continuation-chance labelling defect into a first-chance
     model's predictors. `chance_index` (below) is what makes the restriction
     possible.
  3. `n_prior` is carried so the "no prior games" case is a feature value of
     exactly 0.0 on a centred scale -- the league mean -- and never a
     fabricated level.

`chance_index` counts how many offensive rebounds this offence has already
secured in the current possession before this opportunity: it is 0 on the first
miss of a possession and increments only when the immediately preceding
opportunity in the same game was an OREB by the same team. It is deliberately
NOT a model feature -- the pre-registration's four bundles do not include it --
and `docs/models/rebound/features.md` records it as a rejected-by-omission
candidate with the effect size it would have had.

===========================================================================
FOLDS AND THE SEAL
===========================================================================
F1 trains {2022, 2023} and tests 2024; F2 trains {2022, 2023, 2024} and tests
2025 and is the selection fold. Season 2026 is sealed and cannot enter a fold:
`fold_slices` calls `cbb_sim.data.seal.assert_not_sealed` on both slices, the
same guard `possession_outcome.fold_slices` uses.

The lineup bundle has its OWN fold, `L2`: train 2024, test 2025. CBBD `onFloor`
is empty at the source before 2024 (L13), so a lineup feature cannot be built
for the F1/F2 training windows at all. C and D are re-scored against each other
on exactly the rows of that fold that carry all ten on-floor ids, so the
comparison is like-for-like.

===========================================================================
THE VERSION ARGUMENT
===========================================================================
`version` selects the possessions build (`cbb_sim.pbp.possessions.
POSSESSION_VERSIONS`). It changes exactly one thing for this model: whether
ESPN's 2025 putback mistag is repaired before a miss is labelled `rim` vs
`jump2` (L16), which is a FEATURE of `B_plus_miss` and above. No rebound
outcome, no opportunity and no team rate depends on it. The threshold is read
out of the chosen build's own `build_report.json`
(`event_stream.rim_override_for_version`), so switching the PM's flag from v1
to v2 needs no edit here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.models import event_stream as ES
from cbb_sim.models import prob_metrics as PM
from cbb_sim.pbp.possessions import DEFAULT_POSSESSION_VERSION
from cbb_sim.ratings import own_ratings as orat

DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_ARTIFACT_DIR = Path("data/processed/models/rebound")

#: FIXED class order. Every probability matrix in this module and every
#: artifact written by the trainer uses it.
CLASSES: tuple[str, ...] = ("OREB", "DREB", "DEAD")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

#: Miss types. `ft` is the missed last free throw of a trip; the three FGA
#: classes are `cbb_sim.pbp.events`' own vocabulary, so the rim/jumper split
#: moves with the possessions version and nothing else does.
MISS_TYPES: tuple[str, ...] = ("rim", "jump2", "three", "ft")
_MISS_FROM_CLS = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}

FOLDS: dict[str, dict[str, list[int]]] = {
    "F1": {"train": [2022, 2023], "test": [2024]},
    "F2": {"train": [2022, 2023, 2024], "test": [2025]},
}
SELECTION_FOLD = "F2"

#: The lineup bundle's own fold. L13: CBBD `onFloor` is empty at the source in
#: 2022-2023, so 2024 is the first season a lineup feature exists for.
LINEUP_FOLDS: dict[str, dict[str, list[int]]] = {
    "L2": {"train": [2024], "test": [2025]},
}
LINEUP_SELECTION_FOLD = "L2"

#: Shrinkage grid for an individual player's as-of rebound rate, in
#: opportunities. FITTED on the lineup fold's TRAIN season, never assumed
#: (L13: "shrinkage strength is a fitted parameter"). 0 means the raw rate.
LINEUP_PRIOR_GRID: tuple[int, ...] = (0, 50, 100, 200, 400)


# ===========================================================================
# 1. Rebound opportunities from the event stream
# ===========================================================================
OPPORTUNITY_COLUMNS: tuple[str, ...] = (
    "game_id", "cbbd_game_id", "season", "game_date", "neutral_site",
    "period", "seconds_remaining", "score_diff", "off_team_id", "def_team_id",
    "offense_is_home", "miss_type", "blocked", "off_in_bonus",
    "off_in_double_bonus", "outcome", "chance_index", "rebounder_id",
)


def build_rebound_events(
    seasons: list[int],
    universe: pd.DataFrame | None = None,
    version: str | None = None,
    poss_dir: Path | str | None = None,
    pbp_dir: Path | str = ES.DEFAULT_PBP_DIR,
    universe_path: Path | str = DEFAULT_UNIVERSE,
    require_pbp_complete: bool = False,
) -> pd.DataFrame:
    """One row per rebound opportunity for `seasons` (module docstring).

    `version` selects the possessions build whose rim/jumper labelling is used
    for `miss_type`; `poss_dir` overrides it with an explicit directory."""
    if universe is None:
        universe = ES.load_universe(universe_path, require_pbp_complete=require_pbp_complete)
    max_ft = ES.rim_override_for_version(version, poss_dir)
    frames = [
        _season_opportunities(int(s), universe, max_ft, pbp_dir)
        for s in seasons
    ]
    out = pd.concat(frames, ignore_index=True)
    out.attrs["possessions_version"] = version or DEFAULT_POSSESSION_VERSION
    out.attrs["rim_override_max_ft"] = float(max_ft)
    return out


def _season_opportunities(season: int, universe: pd.DataFrame, max_ft: float,
                          pbp_dir: Path | str) -> pd.DataFrame:
    st = ES.build_stream(season, universe, rim_override_max_ft=max_ft, pbp_dir=pbp_dir)
    cls = st["cls"].to_numpy(dtype=object)
    made = st["made"].to_numpy()
    g = st["cbbd_game_id"].to_numpy()
    sd = st["side"].to_numpy()

    miss_fga = np.isin(cls, ES.FGA_CLASSES) & ~made
    miss_lastft = ((cls == "FT_missed") & st["trip_last"].to_numpy()
                   & ~st["trip_is_technical"].to_numpy())
    opp = miss_fga | miss_lastft

    same_next = np.concatenate([g[1:] == g[:-1], [False]])
    nxt_cls = np.concatenate([cls[1:], [None]])
    nxt_pid = np.concatenate([st["player_id"].to_numpy()[1:], [np.nan]])
    outcome = np.where(
        ~same_next, "unresolved",
        np.where(nxt_cls == "OREB", "OREB",
                 np.where(nxt_cls == "DREB", "DREB",
                          np.where(nxt_cls == "DeadBallReb", "DEAD", "unresolved"))))

    miss_type = np.where(miss_lastft, "ft",
                         pd.Series(cls).map(_MISS_FROM_CLS).fillna("").to_numpy())

    idx = np.flatnonzero(opp)
    o_out = outcome[idx]
    o_side = sd[idx]
    o_game = g[idx]

    # chance_index: consecutive offensive rebounds by the same offence inside
    # one possession. `link` is "this opportunity continues the previous one".
    link = np.zeros(len(idx), dtype=bool)
    if len(idx) > 1:
        link[1:] = (o_out[:-1] == "OREB") & (o_side[1:] == o_side[:-1]) & (o_game[1:] == o_game[:-1])
    pos = np.arange(len(idx))
    last_reset = np.maximum.accumulate(np.where(~link, pos, -1))
    chance_index = (pos - last_reset).astype("int16")

    hs = st["home_score"].to_numpy()[idx]
    as_ = st["away_score"].to_numpy()[idx]
    off_home = o_side == 0
    out = pd.DataFrame({
        "game_id": st["game_id"].to_numpy()[idx],
        "cbbd_game_id": o_game,
        "season": np.full(len(idx), season, dtype="int16"),
        "game_date": st["game_date"].to_numpy()[idx],
        "neutral_site": st["neutral_site"].to_numpy()[idx],
        "period": st["period"].to_numpy()[idx],
        "seconds_remaining": st["sec"].to_numpy()[idx],
        "score_diff": np.where(off_home, hs - as_, as_ - hs).astype("int32"),
        "off_team_id": st["team_id"].to_numpy()[idx],
        "def_team_id": st["opp_id"].to_numpy()[idx],
        "offense_is_home": off_home,
        "miss_type": miss_type[idx],
        "blocked": st["blocked"].to_numpy()[idx],
        "off_in_bonus": ES.in_bonus(st["fouls_opp_prior"].to_numpy()[idx]),
        "off_in_double_bonus": ES.in_double_bonus(st["fouls_opp_prior"].to_numpy()[idx]),
        "outcome": o_out,
        "chance_index": chance_index,
        "rebounder_id": nxt_pid[idx],
    })
    for c in ES.ON_FLOOR_COLS:
        out[c] = st[c].to_numpy()[idx]
    # the shooter's own on-floor five, resolved from which side has the ball
    return out


# ===========================================================================
# 2. As-of team rebound form
# ===========================================================================
def team_rebound_form(events: pd.DataFrame, universe: pd.DataFrame,
                      first_chance_only: bool = True) -> pd.DataFrame:
    """As-of, league-centred offensive-rebound and defensive-rebound form for
    every (game, team). One row per team-game; both sides of every game.

    `first_chance_only` restricts the source to opportunities with
    `chance_index == 0`. It is the default and the pre-registration's
    "first-chance-safe sources" clause; the pooled variant is kept so
    `experiments.md` can report what the restriction costs.

    Denominator is LIVE opportunities (OREB + DREB). Dead balls and unresolved
    rows are excluded from both numerator and denominator, because OREB% is
    defined on rebounds that were actually contested; their share is reported
    separately rather than folded into a rate that would then mean something
    else."""
    ev = events[events["outcome"].isin(["OREB", "DREB"])]
    if first_chance_only:
        ev = ev[ev["chance_index"] == 0]
    ev = ev.assign(_oreb=(ev["outcome"] == "OREB").astype("int32"), _opp=1)
    box = ev.groupby(["season", "game_id", "off_team_id", "def_team_id"], as_index=False).agg(
        opps=("_opp", "sum"), orebs=("_oreb", "sum"))
    box = box.rename(columns={"off_team_id": "team_id", "def_team_id": "opp_id"})

    dates = universe[["game_id", "game_date"]].copy()
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    box = box.merge(dates, on="game_id", how="left")
    box = box.sort_values(["season", "game_date", "game_id"], kind="stable").reset_index(drop=True)

    cols = ["opps", "orebs"]

    off = box.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    off_asof = PM.expanding_asof(off, ["season", "team_id"], cols)
    off_asof.columns = [f"off_{c}" for c in off_asof.columns]
    off = pd.concat([off[["season", "game_id", "team_id", "opp_id", "game_date"]], off_asof], axis=1)

    dfd = box.rename(columns={"team_id": "_off", "opp_id": "team_id"})
    dfd = dfd.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    def_asof = PM.expanding_asof(dfd, ["season", "team_id"], cols)
    def_asof.columns = [f"def_{c}" for c in def_asof.columns]
    dfd = pd.concat([dfd[["season", "game_id", "team_id"]], def_asof], axis=1)

    form = off.merge(dfd, on=["season", "game_id", "team_id"], how="left")

    day = box.groupby(["season", "game_date"], as_index=False)[cols].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], cols)
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    form = form.merge(day, on=["season", "game_date"], how="left")

    def _rate(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    lg_oreb = _rate(form["lg_orebs"].to_numpy(), form["lg_opps"].to_numpy())
    own_oreb = _rate(form["off_orebs"].to_numpy(), form["off_opps"].to_numpy())
    # the defence's own DREB rate is 1 - (opponents' OREB rate against it)
    opp_oreb_allowed = _rate(form["def_orebs"].to_numpy(), form["def_opps"].to_numpy())
    own_dreb = 1.0 - opp_oreb_allowed
    lg_dreb = 1.0 - lg_oreb

    form["off_oreb_c"] = np.where(np.isnan(own_oreb) | np.isnan(lg_oreb), 0.0,
                                  own_oreb - lg_oreb).astype("float32")
    form["def_dreb_c"] = np.where(np.isnan(own_dreb) | np.isnan(lg_dreb), 0.0,
                                  own_dreb - lg_dreb).astype("float32")
    # A handful of team-games have no row on the defence side of the join (a
    # game in which the other team never generated a live rebound opportunity
    # of its own). They keep `n_prior = 0`, which pairs with the centred
    # feature's own fallback of exactly 0.0 -- the league mean, never a
    # fabricated level -- and the count is reported by the trainer.
    form["n_prior_off"] = form["off_n_prior"].fillna(0).astype("int32")
    form["n_prior_def"] = form["def_n_prior"].fillna(0).astype("int32")
    return form[["season", "game_id", "team_id", "opp_id", "game_date",
                 "off_oreb_c", "def_dreb_c", "n_prior_off", "n_prior_def"]]


# ===========================================================================
# 3. As-of individual rebound rates (lineup bundle, 2024+)
# ===========================================================================
def player_rebound_rates(events: pd.DataFrame, prior_opps: int = 100) -> pd.DataFrame:
    """As-of individual offensive- and defensive-rebound rate for every
    (season, player, game).

    An individual rate here is `rebounds this player was credited with, while
    on the floor / opportunities that occurred while this player was on the
    floor` -- the true share-of-available-rebounds definition, so the five
    on-floor players' rates sum to roughly the team rate and the lineup feature
    is that sum. Team rebounds (a rebound row with no `participant_1_id`,
    12-14% of rebound rows) contribute to the DENOMINATOR and to nobody's
    numerator, which is the honest treatment: the opportunity happened, no
    player was credited.

    `prior_opps` shrinks each rate toward the league as-of rate with that many
    pseudo-opportunities. L13 requires the strength be fitted rather than
    assumed; `scripts/train_rebound_v1.py` fits it on the lineup fold's TRAIN
    season over `LINEUP_PRIOR_GRID` and records the winner."""
    ev = events[events["outcome"].isin(["OREB", "DREB"])].copy()
    on = ev[list(ES.ON_FLOOR_COLS)].to_numpy()
    complete = np.isfinite(on).all(axis=1)
    ev = ev[complete]
    if not len(ev):
        return pd.DataFrame(columns=["season", "player_id", "game_id", "game_date",
                                     "oreb_rate", "dreb_rate"])
    on = ev[list(ES.ON_FLOOR_COLS)].to_numpy().astype("int64")
    off_home = ev["offense_is_home"].to_numpy()
    is_oreb = (ev["outcome"] == "OREB").to_numpy()
    reb_id = ev["rebounder_id"].to_numpy()
    n = len(ev)

    rows = []
    for slot in range(10):
        pid = on[:, slot]
        player_is_home = slot < 5
        on_offense = player_is_home == off_home
        rows.append(pd.DataFrame({
            "season": ev["season"].to_numpy(),
            "game_id": ev["game_id"].to_numpy(),
            "player_id": pid,
            "off_opp": on_offense.astype("int32"),
            "def_opp": (~on_offense).astype("int32"),
            "oreb": (on_offense & is_oreb & (reb_id == pid)).astype("int32"),
            "dreb": ((~on_offense) & (~is_oreb) & (reb_id == pid)).astype("int32"),
        }))
    long = pd.concat(rows, ignore_index=True)
    del rows
    pg = long.groupby(["season", "player_id", "game_id"], as_index=False)[
        ["off_opp", "def_opp", "oreb", "dreb"]].sum()
    dates = ev[["game_id", "game_date"]].drop_duplicates()
    pg = pg.merge(dates, on="game_id", how="left")
    pg = pg.sort_values(["season", "player_id", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    asof = PM.expanding_asof(pg, ["season", "player_id"], ["off_opp", "def_opp", "oreb", "dreb"])
    pg = pd.concat([pg[["season", "player_id", "game_id", "game_date"]], asof], axis=1)

    # league as-of rate on the same date, from the same source
    day = pg.groupby(["season", "game_date"], as_index=False)[["off_opp", "def_opp", "oreb", "dreb"]].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], ["off_opp", "def_opp", "oreb", "dreb"])
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    pg = pg.merge(day, on=["season", "game_date"], how="left")
    pg["prior_opps"] = int(prior_opps)
    return _shrink_player_rates(pg, int(prior_opps))


def _shrink_player_rates(pg: pd.DataFrame, prior_opps: int) -> pd.DataFrame:
    def _rate(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    lg_o = _rate(pg["lg_oreb"].to_numpy(), pg["lg_off_opp"].to_numpy())
    lg_d = _rate(pg["lg_dreb"].to_numpy(), pg["lg_def_opp"].to_numpy())
    # a league rate is a per-PLAYER share, so its neutral value is 1/5 of the
    # team rate; that is exactly what lg_oreb / lg_off_opp already is, because
    # every opportunity contributes five offensive on-floor rows.
    lg_o = np.where(np.isnan(lg_o), 0.0, lg_o)
    lg_d = np.where(np.isnan(lg_d), 0.0, lg_d)
    k = float(prior_opps)
    # A player with no prior opportunities on the side in question falls back
    # to the league's own as-of per-player rate. This matters at k = 0, where
    # the shrunk formula is 0/0: leaving it NaN would drop that opportunity
    # from the lineup subset entirely, so the k = 0 rung of the shrinkage grid
    # would be scored on a DIFFERENT (and smaller) set of rows than every other
    # rung and the grid would not be comparing shrinkage strengths at all.
    den_o = pg["off_opp"].to_numpy() + k
    den_d = pg["def_opp"].to_numpy() + k
    pg["oreb_rate"] = np.where(
        den_o > 0, (pg["oreb"].to_numpy() + k * lg_o) / np.maximum(den_o, 1e-9), lg_o
    ).astype("float32")
    pg["dreb_rate"] = np.where(
        den_d > 0, (pg["dreb"].to_numpy() + k * lg_d) / np.maximum(den_d, 1e-9), lg_d
    ).astype("float32")
    pg["lg_oreb_rate"] = lg_o.astype("float32")
    pg["lg_dreb_rate"] = lg_d.astype("float32")
    return pg[["season", "player_id", "game_id", "game_date", "oreb_rate", "dreb_rate",
               "lg_oreb_rate", "lg_dreb_rate", "off_opp", "def_opp"]]


def attach_lineup_features(design: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Sum of the offence's five on-floor players' as-of OREB rates and of the
    defence's five as-of DREB rates, each centred on five times the league's
    own as-of per-player rate (so a league-average lineup sits at exactly 0.0).

    `lineup_on_floor_ok` marks the rows where all ten ids resolved to a rate;
    the lineup fold is evaluated on those rows only, for BOTH the C and the D
    bundle, so the comparison is like-for-like."""
    d = design.copy()
    key = rates.set_index(["season", "player_id", "game_id"])
    o_sum = np.zeros(len(d), dtype="float64")
    d_sum = np.zeros(len(d), dtype="float64")
    ok = np.ones(len(d), dtype=bool)
    lg_o = np.full(len(d), np.nan)
    lg_d = np.full(len(d), np.nan)
    off_home = d["offense_is_home"].to_numpy()
    for slot, col in enumerate(ES.ON_FLOOR_COLS):
        # The on-floor columns are stored as floats (they carry nulls before
        # 2024), while the rate table is keyed on an int64 player id. Cast
        # explicitly rather than relying on pandas to coerce a float level
        # against an int level: a silent type mismatch here would look exactly
        # like "no lineup data" and would quietly answer the lineup question
        # with an empty feature.
        pid = pd.to_numeric(d[col], errors="coerce").to_numpy()
        pid_i = np.where(np.isfinite(pid), pid, -1).astype("int64")
        idx = pd.MultiIndex.from_arrays(
            [d["season"].to_numpy(), pid_i, d["game_id"].to_numpy()])
        r = key.reindex(idx)
        player_is_home = slot < 5
        on_offense = player_is_home == off_home
        orr = r["oreb_rate"].to_numpy()
        drr = r["dreb_rate"].to_numpy()
        present = np.isfinite(orr) & np.isfinite(drr)
        ok &= present
        o_sum += np.where(on_offense & present, np.nan_to_num(orr), 0.0)
        d_sum += np.where((~on_offense) & present, np.nan_to_num(drr), 0.0)
        lg_o = np.where(np.isfinite(r["lg_oreb_rate"].to_numpy()), r["lg_oreb_rate"].to_numpy(), lg_o)
        lg_d = np.where(np.isfinite(r["lg_dreb_rate"].to_numpy()), r["lg_dreb_rate"].to_numpy(), lg_d)
    lg_o = np.nan_to_num(lg_o)
    lg_d = np.nan_to_num(lg_d)
    d["lineup_oreb_c"] = np.where(ok, o_sum - 5.0 * lg_o, 0.0).astype("float32")
    d["lineup_dreb_c"] = np.where(ok, d_sum - 5.0 * lg_d, 0.0).astype("float32")
    d["lineup_on_floor_ok"] = ok
    return d


# ===========================================================================
# 4. Design matrix and the pre-registered feature bundles
# ===========================================================================
TEAM_FEATURES: tuple[str, ...] = (
    "off_oreb_c",            # offence's own as-of OREB%, league-centred
    "opp_def_dreb_c",        # defence's as-of DREB%, league-centred
    "off_rating_off_c", "off_rating_def_c",
    "def_rating_off_c", "def_rating_def_c",
    "site_home", "site_away",          # neutral is the reference level
)
MISS_FEATURES: tuple[str, ...] = ("miss_rim", "miss_jump2", "miss_three", "blocked_f")
STATE_FEATURES: tuple[str, ...] = ("period", "seconds_remaining", "score_diff", "in_bonus")
LINEUP_FEATURES: tuple[str, ...] = ("lineup_oreb_c", "lineup_dreb_c")

FEATURE_SETS: tuple[str, ...] = ("A_team", "B_plus_miss", "C_plus_state", "D_plus_lineup")


def feature_set(name: str) -> list[str]:
    a = list(TEAM_FEATURES)
    if name == "A_team":
        return a
    b = a + list(MISS_FEATURES)
    if name == "B_plus_miss":
        return b
    c = b + list(STATE_FEATURES)
    if name == "C_plus_state":
        return c
    if name == "D_plus_lineup":
        return c + list(LINEUP_FEATURES)
    raise KeyError(f"unknown feature set {name!r}")


def build_design(
    seasons: list[int],
    universe: pd.DataFrame | None = None,
    version: str | None = None,
    poss_dir: Path | str | None = None,
    ratings_dir: Path | str = "data/processed/ratings",
    universe_path: Path | str = DEFAULT_UNIVERSE,
    pbp_dir: Path | str = ES.DEFAULT_PBP_DIR,
    require_pbp_complete: bool = False,
    first_chance_only: bool = True,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per MODELLED rebound opportunity with every candidate feature.

    `unresolved` opportunities are dropped here; their count is left on
    `df.attrs['n_unresolved']` so the trainer can report the share instead of
    it disappearing silently."""
    seasons = [int(s) for s in seasons]
    if universe is None:
        universe = ES.load_universe(universe_path, require_pbp_complete=require_pbp_complete)
    if events is None:
        events = build_rebound_events(seasons, universe=universe, version=version,
                                      poss_dir=poss_dir, pbp_dir=pbp_dir)
    events = events[events["season"].isin(seasons)]

    form = team_rebound_form(events, universe, first_chance_only=first_chance_only)

    n_all = len(events)
    d = events[events["outcome"].isin(CLASSES)].copy()
    n_unresolved = n_all - len(d)

    off_form = form.rename(columns={"team_id": "off_team_id"})[
        ["season", "game_id", "off_team_id", "off_oreb_c", "n_prior_off"]]
    d = d.merge(off_form, on=["season", "game_id", "off_team_id"], how="left")
    def_form = form.rename(columns={"team_id": "def_team_id", "def_dreb_c": "opp_def_dreb_c"})[
        ["season", "game_id", "def_team_id", "opp_def_dreb_c", "n_prior_def"]]
    d = d.merge(def_form, on=["season", "game_id", "def_team_id"], how="left")

    ratings = orat.load_ratings(sorted(set(seasons)), out_dir=ratings_dir)
    d["game_date"] = pd.to_datetime(d["game_date"])
    d = orat.join_as_of(d, ratings, team_col="off_team_id", date_col="game_date",
                        suffix="__offteam", cols=("off_c", "def_c"))
    d = orat.join_as_of(d, ratings, team_col="def_team_id", date_col="game_date",
                        suffix="__defteam", cols=("off_c", "def_c"))
    d = d.rename(columns={
        "off_c__offteam": "off_rating_off_c", "def_c__offteam": "off_rating_def_c",
        "off_c__defteam": "def_rating_off_c", "def_c__defteam": "def_rating_def_c",
    })

    neutral = d["neutral_site"].to_numpy().astype(bool)
    off_home = d["offense_is_home"].to_numpy().astype(bool)
    d["site_home"] = ((~neutral) & off_home).astype("float32")
    d["site_away"] = ((~neutral) & (~off_home)).astype("float32")

    mt = d["miss_type"].to_numpy()
    d["miss_rim"] = (mt == "rim").astype("float32")
    d["miss_jump2"] = (mt == "jump2").astype("float32")
    d["miss_three"] = (mt == "three").astype("float32")
    d["blocked_f"] = d["blocked"].astype("float32")

    d["period"] = d["period"].astype("float32")
    d["seconds_remaining"] = d["seconds_remaining"].astype("float32")
    d["score_diff"] = d["score_diff"].astype("float32")
    d["in_bonus"] = d["off_in_bonus"].astype("float32")

    fill = ["off_oreb_c", "opp_def_dreb_c", "off_rating_off_c", "off_rating_def_c",
            "def_rating_off_c", "def_rating_def_c"]
    for c in fill:
        d[c] = d[c].astype("float32").fillna(0.0)

    d["y"] = d["outcome"].map(CLASS_INDEX).astype("int8")
    d["season_idx"] = (d["season"] - 2022).astype("float32")
    d.attrs["n_unresolved"] = int(n_unresolved)
    d.attrs["n_opportunities"] = int(n_all)
    d.attrs["possessions_version"] = version or DEFAULT_POSSESSION_VERSION
    d.attrs["rim_override_max_ft"] = float(events.attrs.get("rim_override_max_ft", 0.0))
    d.attrs["first_chance_only"] = bool(first_chance_only)
    return d


def fold_slices(design: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) slices for a fold, with the seal guard on BOTH -- season
    2026 can never enter a fold."""
    spec = FOLDS[fold] if fold in FOLDS else LINEUP_FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"{fold} train seasons")
    assert_not_sealed(spec["test"], context=f"{fold} test seasons")
    tr = design[design["season"].isin(spec["train"])]
    te = design[design["season"].isin(spec["test"])]
    assert_not_sealed(tr, context=f"{fold} train slice")
    assert_not_sealed(te, context=f"{fold} test slice")
    return tr, te


# ===========================================================================
# 5. Model arms
# ===========================================================================
ARMS: tuple[str, ...] = ("baseline", "ridge_logit", "lgbm")


class BaselineArm:
    """The pre-registered floor: league class shares BY MISS TYPE.

    "League-share baseline by miss type" is read the way L3 round 1 read its
    own baseline: the shares of the most recent TRAINING season, because using
    the test season's own shares would be an oracle. The oracle variant is
    reported separately by the trainer and labelled unattainable, so the gap
    between the two is visible as the season-drift cost rather than hidden."""

    def __init__(self, oracle: bool = False):
        self.oracle = oracle
        self.table_: dict[str, np.ndarray] = {}
        self.overall_: np.ndarray | None = None

    def fit(self, y: np.ndarray, miss_type: np.ndarray, seasons: np.ndarray) -> BaselineArm:
        last = int(np.max(seasons))
        sel = seasons == last
        yy, mm = y[sel], miss_type[sel]
        counts = np.bincount(yy, minlength=len(CLASSES)).astype("float64")
        self.overall_ = counts / counts.sum()
        for m in MISS_TYPES:
            s = mm == m
            if s.sum() == 0:
                self.table_[m] = self.overall_
                continue
            c = np.bincount(yy[s], minlength=len(CLASSES)).astype("float64")
            self.table_[m] = c / c.sum()
        return self

    def predict_proba(self, miss_type: np.ndarray) -> np.ndarray:
        out = np.repeat(self.overall_[None, :], len(miss_type), axis=0)
        for m in MISS_TYPES:
            s = miss_type == m
            if s.any():
                out[s] = self.table_[m]
        return out


class RidgeLogitArm:
    """Multinomial ridge logit over the three classes. Features are
    standardised on the TRAIN slice only and the fitted means/SDs travel with
    the model, so the sim applies the identical transform."""

    def __init__(self, C: float = 1.0, max_iter: int = 300, seed: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeLogitArm:
        from sklearn.linear_model import LogisticRegression

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        Z = (X - self.mu_) / self.sd_
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs",
                                       random_state=self.seed).fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu_) / self.sd_
        return _align_classes(self.clf_.predict_proba(Z), self.clf_.classes_)


class LgbmArm:
    """LightGBM multiclass. Hyperparameters are FIXED across every feature set
    and fold, so the grid compares feature bundles and model classes rather
    than tuning effort -- the same discipline L3 round 1 used."""

    PARAMS = dict(
        objective="multiclass", num_class=len(CLASSES), n_estimators=400,
        learning_rate=0.06, num_leaves=63, min_child_samples=400,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        reg_lambda=1.0, verbose=-1,
    )

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> LgbmArm:
        import lightgbm as lgb

        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, **self.PARAMS)
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _align_classes(self.clf_.predict_proba(X), self.clf_.classes_)


def _align_classes(p: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((p.shape[0], len(CLASSES)), dtype="float64")
    for j, c in enumerate(classes):
        out[:, int(c)] = p[:, j]
    s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-12)


def _matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.ascontiguousarray(df[features].to_numpy(dtype="float32"))


def fit_arm(arm: str, tr: pd.DataFrame, features: list[str], seed: int = 0):
    y = tr["y"].to_numpy()
    if arm == "baseline":
        return BaselineArm().fit(y, tr["miss_type"].to_numpy(), tr["season"].to_numpy())
    X = _matrix(tr, features)
    if arm == "ridge_logit":
        return RidgeLogitArm(seed=seed).fit(X, y)
    if arm == "lgbm":
        return LgbmArm(seed=seed).fit(X, y)
    raise KeyError(f"unknown arm {arm!r}")


def predict_arm(arm: str, model, te: pd.DataFrame, features: list[str]) -> np.ndarray:
    if arm == "baseline":
        return model.predict_proba(te["miss_type"].to_numpy())
    return model.predict_proba(_matrix(te, features))


# ===========================================================================
# 6. The dead-ball question, as a diagnostic
# ===========================================================================
def deterministic_dead_share(tr: pd.DataFrame) -> dict[str, float]:
    """Training-fold dead-ball share by miss type -- the "absorbed as a
    deterministic share" alternative the pre-registration asks about."""
    y = tr["y"].to_numpy()
    mt = tr["miss_type"].to_numpy()
    out = {}
    for m in MISS_TYPES:
        s = mt == m
        out[m] = float((y[s] == CLASS_INDEX["DEAD"]).mean()) if s.any() else 0.0
    return out


def compose_binary_plus_fixed_dead(p_binary_oreb: np.ndarray, miss_type: np.ndarray,
                                   dead_share: dict[str, float]) -> np.ndarray:
    """Three-class probabilities from a LIVE-only binary model plus a fixed
    dead-ball share per miss type. This is the arm the dead-ball question is
    decided against: if it matches the full three-class model inside the noise
    floor, dead balls do not need a model of their own."""
    dead = np.array([dead_share.get(m, 0.0) for m in miss_type], dtype="float64")
    live = 1.0 - dead
    out = np.empty((len(miss_type), len(CLASSES)), dtype="float64")
    out[:, CLASS_INDEX["OREB"]] = live * p_binary_oreb
    out[:, CLASS_INDEX["DREB"]] = live * (1.0 - p_binary_oreb)
    out[:, CLASS_INDEX["DEAD"]] = dead
    return out


# ===========================================================================
# 7. Metrics wired to this model's vocabulary
# ===========================================================================
#: The pre-registration's responsiveness drivers: the offence's own as-of OREB%
#: and the defence's as-of DREB%, each against the predicted OREB share.
RESPONSIVENESS_SPECS: tuple[tuple[str, str], ...] = (
    ("off_oreb_c", "OREB"),
    ("opp_def_dreb_c", "OREB"),
)
#: "monotone in 4 of 4 steps" -- five quintiles, four steps, no violation
#: allowed. This is STRICTER than the L3 round-1 reading (3 of 4) because the
#: rebound pre-registration says 4 of 4 explicitly.
RESPONSIVENESS_MIN_STEPS = 4


def responsiveness(te: pd.DataFrame, p: np.ndarray, n_q: int = 5) -> dict[str, dict]:
    out = {}
    y = te["y"].to_numpy()
    for feat, cls in RESPONSIVENESS_SPECS:
        out[f"{feat}->{cls}"] = PM.quintile_responsiveness(
            te[feat].to_numpy(), y, p, CLASS_INDEX[cls], n_q=n_q)
    return out


def score(te: pd.DataFrame, p: np.ndarray) -> dict:
    """The full pre-registered scorecard for one arm on one fold."""
    y = te["y"].to_numpy()
    calib = PM.decile_calibration(y, p, CLASSES)
    calib_ok, calib_worst, calib_who = PM.calibration_verdict(calib)
    resp = responsiveness(te, p)
    resp_ok, resp_worst = PM.responsiveness_verdict(resp, RESPONSIVENESS_MIN_STEPS)
    return {
        "n": int(len(te)),
        "log_loss": PM.log_loss(y, p),
        "brier": PM.multiclass_brier(y, p, CLASSES),
        "brier_by_class": PM.per_class_brier(y, p, CLASSES),
        "calibration": calib,
        "calib_pass": bool(calib_ok),
        "calib_worst_gap_pp": calib_worst,
        "calib_worst_class": calib_who,
        "responsiveness": resp,
        "resp_pass": bool(resp_ok),
        "resp_min_steps": resp_worst,
        "by_miss_type": PM.segment_calibration(te["miss_type"], y, p, CLASSES),
    }


@dataclass
class FittedRebound:
    """What the trainer persists for the sim to load."""
    arm: str
    feature_set: str
    fold: str
    features: list[str]
    model: object
    classes: tuple[str, ...]
    meta: dict


__all__ = [
    "ARMS", "CLASSES", "CLASS_INDEX", "FEATURE_SETS", "FOLDS", "LINEUP_FOLDS",
    "LINEUP_PRIOR_GRID", "MISS_TYPES", "RESPONSIVENESS_MIN_STEPS",
    "SELECTION_FOLD", "BaselineArm", "FittedRebound", "LgbmArm", "RidgeLogitArm",
    "attach_lineup_features", "build_design", "build_rebound_events",
    "compose_binary_plus_fixed_dead", "deterministic_dead_share", "feature_set",
    "fit_arm", "fold_slices", "player_rebound_rates", "predict_arm",
    "responsiveness", "score", "team_rebound_form",
]
