"""
usage.py -- L4 SHOT ALLOCATION (usage): who, of the five on the floor, is
credited with the event.

Pre-registration (verbatim, PM, 2026-09-10):
`docs/models/usage/experiments.md` section 1. Trainer:
`scripts/train_usage_v1.py`. Feature provenance:
`docs/models/usage/features.md`.

===========================================================================
THE TARGET
===========================================================================
One row per credited event of a chance, restricted to events whose OFFENSIVE
on-floor five is fully resolved (L13: CBBD `onFloor` is empty at the source in
2022-2023, so this model lives on 2024+ alone). Five event classes:

  FGA_rim / FGA_jump2 / FGA_3   the shooter of the attempt
  TOV                           the player charged with the turnover
  FT_trip                       the FOULED SHOOTER, i.e. `participant_1_id`
                                on the FIRST attempt of a foul-caused
                                free-throw trip. Technical trips are excluded
                                for the same reason `free_throw` excludes them:
                                the shooter is chosen by the coach, not by who
                                was fouled, so the two populations are drawn
                                from different shooter distributions.

The event is read off `cbb_sim.models.event_stream`, not off the chance table,
for two reasons the rebound and free-throw models already hit: the chance table
carries no `participant` id at all, and the possession table carries the
on-floor five only once per possession (from the possession's first event), so a
lineup change inside a possession would be attributed to the wrong five. The
event stream carries both on the event's own row.

Each surviving row is a choice among FIVE alternatives. The five are sorted
ascending by CBBD player id so the alternative order is a deterministic function
of the lineup and never of the feed's column order.

===========================================================================
THE INPUTS ARE PREGAME, PER-POSSESSION-ON-FLOOR RATES
===========================================================================
For every player, within a season and STRICTLY BEFORE the current game's date:

  exposure_asof   the number of credited events (any of the five classes) that
                  happened while he was one of the offensive five. This is the
                  denominator the pre-registration asks for -- a per-possession
                  ON-FLOOR exposure, not a per-game one -- and it is what makes
                  a bench player's rate comparable to a starter's.
  ev_{class}      the number of those events credited to him, per class.
  rate_{class}    ev_{class} / exposure_asof, shrunk toward a fitted prior with
                  a fitted strength (`shrunk_rate`).
  minutes_asof    hoopR `player_box` minutes to date, joined through the
                  CBBD<->ESPN player crosswalk (`cbb_sim.data.player_ids`).
                  The crosswalk covers 2024-2026, which is exactly this model's
                  window, so unlike at FT-2 it is usable as a feature here.
  prev_rate_{c}   the same rate over the player's COMPLETED previous season.

Every one of those is an expanding sum over GAMES STRICTLY BEFORE the current
one (`prob_metrics.expanding_asof`), the same construction and the same
leak-safety proof shape as `free_throw` and `rebound`.

THE PRIOR-SEASON ASYMMETRY, STATED RATHER THAN HIDDEN. On-floor ids do not
exist before 2024 (L13), so a 2024 row has no prior-season ON-FLOOR rate and a
2025 row does. On fold F1 (train 2024, test 2025) the prior-season features are
therefore identically absent in training and present in test. A feature with no
variation in the training fold has an unidentified coefficient, so
`drop_constant_features` removes it and the trainer RECORDS the drop -- the same
treatment `cbb_sim.models.clock` gives its degenerate chance-number column. The
within-2025 walk-forward fold has them active on both sides and is where their
value is actually measured.

===========================================================================
THE ARMS
===========================================================================
U1 proportional      p_i proportional to the player's shrunk as-of class rate,
                     normalised over the five. Deterministic shares.
U2 dirichlet         one Dirichlet draw per (game, team, class) over the team's
                     as-of rate profile, then U1's normalisation over whichever
                     five is on the floor. Single fitted concentration.
U3 hier_dirichlet    the same, but the CFB two-level form: a between-ROLE draw
                     (handlers / wings / bigs) then a within-role draw, which
                     keeps E[share] = q EXACTLY at every level while letting a
                     big disperse differently from a handler. Ported from
                     `cfb-props-sim/src/cfb_props_sim/sim/usage_alloc.py`.
U4 cond_logit        conditional logit over the five, ridge-penalised.
U5 lgbm              LightGBM scoring each alternative, softmaxed within the
                     five. Parameter search on 2024 only.

WHY U1, U2 AND U3 NEARLY SHARE A LOG LOSS -- AND WHY THEY DO NOT SHARE A
DISPERSION. A Dirichlet centred on q has E[S] = q, so if the allocator ran over
the WHOLE profile the per-event marginal would be exactly U1's and the three
arms would be indistinguishable on log loss by construction. They are not quite,
because the allocation conditions on a SUBSET of the profile (the five on the
floor) and E[S_i / sum_{j in five} S_j] != q_i / sum_{j in five} q_j. The gap is
second order and is computed here by Monte Carlo (`marginal_probs`) rather than
assumed away. It is a COST: the dispersion the Dirichlet buys is paid for in
per-event sharpness. That trade is the entire point of the pre-registered
decision rule, which makes the game-level SD ratio an ELIGIBILITY condition
rather than a tie-break.

`sequential_probs` additionally reports the Polya-urn predictive -- the same
Dirichlet conditioned on the game's own earlier events of that class. It is NOT
a pregame quantity and never enters the decision; it is reported because it is
the cleanest measure of how much within-game usage concentration is real.

===========================================================================
THE SAMPLER
===========================================================================
`new_game_state` draws the per-(game, class) rate realisation once, off the
counter-based `(seed, game_id, "usage")` stream (`cbb_sim.control.rng`), and
`draw_player(five, event_class, state)` credits one event to one of the five
from that realisation. Same stream contract as `pace.py`: paired bake-off arms
line up game for game and seed for seed, and dropping a game from the universe
moves no other game's draws.

TWO SAMPLERS, ONE DISTRIBUTION. `realize_shares` is the engine path: exact
inverse-CDF gamma variates off the counter-based stream, a pure function of
(seed, game_id, draw index), called once per game so its cost is irrelevant.
`realize_batch` is the trainer path: the same generative form over every profile
at once with `numpy`'s own gamma sampler, which is what makes a 200-draw Monte
Carlo over a season affordable. `tests/test_usage.py` asserts the two agree on
mean and SD, the same way `cfb-props-sim/tests/test_usage_alloc.py` pins its
batched twin to the shipped per-call one.

===========================================================================
FOLDS AND THE SEAL
===========================================================================
F1 trains 2024 and tests 2025 and is the selection fold (there is no earlier
fold: 2023 has no lineups). The robustness fold is a within-2025 walk-forward,
train before 2025-01-15, test after. 2026 is sealed; `fold_slices` calls
`assert_not_sealed` on both slices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cbb_sim.control import rng as RNG
from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.models import event_stream as ES
from cbb_sim.models import prob_metrics as PM
from cbb_sim.models.free_throw import load_positions

DEFAULT_ARTIFACT_DIR = Path("data/processed/models/usage")
DEFAULT_PLAYER_BOX_DIR = Path("data/raw/hoopr/player_box")
DEFAULT_CROSSWALK = Path("data/processed/player_crosswalk.parquet")

#: The five event classes of the pre-registration.
EVENT_CLASSES: tuple[str, ...] = ("FGA_rim", "FGA_jump2", "FGA_3", "TOV", "FT_trip")
#: The classes read straight off an event-stream `cls` value.
DIRECT_CLASSES: tuple[str, ...] = ("FGA_rim", "FGA_jump2", "FGA_3", "TOV")

#: Alternatives per choice. Fixed by the sport, not a parameter.
N_ALT = 5

#: Role families for the hierarchical arm (U3), from position group + as-of
#: usage rank (module docstring / features.md).
ROLES: tuple[str, ...] = ("handler", "wing", "big")
N_ROLE = len(ROLES)
HANDLER_RANK = 2

POSITION_LEVELS: tuple[str, ...] = ("G", "F", "C", "UNK")

FOLDS: dict[str, dict[str, list[int]]] = {
    "F1": {"train": [2024], "test": [2025]},
}
SELECTION_FOLD = "F1"
#: The within-season robustness fold (pre-registration): train before this date
#: in 2025, test after it.
WF_SEASON = 2025
WF_SPLIT_DATE = "2025-01-15"

ARMS: tuple[str, ...] = ("proportional", "dirichlet", "hier_dirichlet", "cond_logit", "lgbm")
#: Tie-break order of the pre-registration: U1 < U2 < U3 < U4 < U5.
ARM_ORDER = {a: i for i, a in enumerate(ARMS)}
TREE_ARM = "lgbm"
DIRICHLET_ARMS: tuple[str, ...] = ("dirichlet", "hier_dirichlet")

#: Shrinkage grid in pseudo-exposures (on-floor events), and the three priors
#: the pre-registration names.
SHRINK_GRID: tuple[float, ...] = (5, 10, 25, 50, 100, 200, 400, 800, 1600)
PRIOR_KINDS: tuple[str, ...] = ("league", "position", "prior_season")

#: Concentration grid for U2/U3, fitted by matching the TRAINING fold's own
#: per-player per-game count dispersion (CLAUDE.md: "dispersion comes from the
#: model's own variance function and is validated against realised residual
#: SD"). `inf` is the no-dispersion rung and collapses the arm onto U1.
ALPHA_GRID: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
                                 256.0, 512.0, 1024.0, float("inf"))

#: Pre-registered gates.
CALIB_GATE_PP = 2.0
RESP_MIN_STEPS = 4
SD_RATIO_BAND = (0.9, 1.1)
GT0_TOL = 0.5
TOPSHARE_TOL_PP = 2.0

EPS = 1e-12


# ===========================================================================
# 1. The event table
# ===========================================================================
def _offense_on_floor(stream: pd.DataFrame) -> np.ndarray:
    """(n, 5) array of the OFFENSIVE five's CBBD ids for every stream row."""
    home = stream[[f"home_on_{k}" for k in range(1, 6)]].to_numpy(dtype="float64")
    away = stream[[f"away_on_{k}" for k in range(1, 6)]].to_numpy(dtype="float64")
    side = stream["side"].to_numpy()
    return np.where((side == 0)[:, None], home, away)


def _chance_number(stream: pd.DataFrame) -> np.ndarray:
    """Chance index within the possession, as a PROXY built at the event level.

    A possession ends on a defensive rebound, a turnover, a made field goal or
    the last made free throw of a trip; a new CHANCE inside the same possession
    starts on an offensive rebound. So the proxy is
    `1 + (offensive rebounds since the last possession-ending event)`. It is a
    proxy and is labelled one in `features.md`: the possession state machine
    owns the authoritative `chance_number`, and the chance table it writes
    carries no on-floor five to join on (module docstring)."""
    cls = stream["cls"].to_numpy(dtype=object)
    g = stream["cbbd_game_id"].to_numpy()
    made = stream["made"].to_numpy()
    ends = ((cls == "DREB") | (cls == "TOV")
            | (np.isin(cls, ES.FGA_CLASSES) & made)
            | ((cls == "FT_made") & stream["trip_last"].to_numpy()))
    new_poss = np.concatenate([[True], ends[:-1] | (g[1:] != g[:-1])])
    poss_id = np.cumsum(new_poss)
    oreb = (cls == "OREB").astype("int64")
    df = pd.DataFrame({"poss_id": poss_id, "oreb": oreb})
    cum = df.groupby("poss_id", sort=False)["oreb"].cumsum().to_numpy()
    return (1 + cum - oreb).astype("int16")


def build_usage_events(
    season: int,
    universe: pd.DataFrame,
    rim_override_max_ft: float = 0.0,
    pbp_dir: Path | str = ES.DEFAULT_PBP_DIR,
) -> pd.DataFrame:
    """One row per credited event with its offensive five (module docstring).

    Rows whose five is incomplete, or whose credited player is not one of the
    five, are KEPT with `five_ok` / `in_five` False so the trainer can report
    the coverage instead of silently losing it; `usable_events` applies the
    filter."""
    season = int(season)
    stream = ES.build_stream(season, universe, rim_override_max_ft=rim_override_max_ft,
                             pbp_dir=pbp_dir)
    chance_no = _chance_number(stream)
    off_all = _offense_on_floor(stream)

    cls = stream["cls"].to_numpy(dtype=object)
    is_direct = np.isin(cls, DIRECT_CLASSES)
    is_fttrip = ((stream["trip_pos"].to_numpy() == 1)
                 & (stream["trip_cause"].to_numpy() == "foul"))
    sel = (is_direct | is_fttrip) & (stream["side"].to_numpy() >= 0)

    ev_class = np.where(is_fttrip, "FT_trip", cls)
    s = stream.loc[sel].reset_index(drop=True)
    off = off_all[sel]
    period = s["period"].to_numpy()
    sec = s["sec"].to_numpy()
    off_home = (s["side"].to_numpy() == 0)
    hs = s["home_score"].to_numpy()
    as_ = s["away_score"].to_numpy()

    d = pd.DataFrame({
        "game_id": s["game_id"].to_numpy(),
        "cbbd_game_id": s["cbbd_game_id"].to_numpy(),
        "season": np.full(len(s), season, dtype="int16"),
        "game_date": s["game_date"].to_numpy(),
        "team_id": s["team_id"].to_numpy(),
        "opp_id": s["opp_id"].to_numpy(),
        "offense_is_home": off_home,
        "neutral_site": s["neutral_site"].to_numpy(),
        "event_class": ev_class[sel],
        "player_id": s["player_id"].to_numpy(),
        "period": period.astype("int16"),
        "sec_in_period": sec.astype("int32"),
        "sec_remaining": np.where(period <= 2, (2 - period) * 1200 + sec, sec).astype("int32"),
        "score_diff": np.where(off_home, hs - as_, as_ - hs).astype("int16"),
        "chance_number": chance_no[sel],
    })

    # the five, sorted ascending so the alternative order is a function of the
    # lineup alone
    five_ok = np.isfinite(off).all(axis=1)
    srt = np.sort(np.where(np.isfinite(off), off, np.inf), axis=1)
    distinct = np.zeros(len(d), dtype=bool)
    if five_ok.any():
        distinct[five_ok] = (np.diff(srt[five_ok], axis=1) > 0).all(axis=1)
    for k in range(N_ALT):
        d[f"alt_{k + 1}"] = srt[:, k]
    pid = d["player_id"].to_numpy()
    eq = np.isclose(srt, np.where(np.isfinite(pid), pid, -1.0)[:, None])
    in_five = five_ok & distinct & np.isfinite(pid) & eq.any(axis=1)
    y = np.where(in_five, eq.argmax(axis=1), -1)
    d["five_ok"] = five_ok & distinct
    d["in_five"] = in_five
    d["y"] = y.astype("int8")
    d.attrs["season"] = season
    d.attrs["rim_override_max_ft"] = float(rim_override_max_ft)
    return d


def usable_events(events: pd.DataFrame) -> pd.DataFrame:
    """The modelled universe: a resolved five containing the credited player."""
    out = events[events["in_five"].to_numpy()].reset_index(drop=True)
    for k in range(1, N_ALT + 1):
        out[f"alt_{k}"] = out[f"alt_{k}"].astype("int64")
    return out


def coverage_report(events: pd.DataFrame) -> dict:
    """Per class: rows, the share with a resolved five, the share whose credited
    player is in it, and the modelled share. A reported number, not a filter."""
    out = {}
    for c in EVENT_CLASSES:
        m = events["event_class"].to_numpy() == c
        n = int(m.sum())
        if not n:
            continue
        out[c] = {
            "n": n,
            "five_resolved_pct": round(float(events["five_ok"].to_numpy()[m].mean() * 100), 4),
            "player_id_present_pct": round(
                float(np.isfinite(events["player_id"].to_numpy()[m]).mean() * 100), 4),
            "modelled_pct": round(float(events["in_five"].to_numpy()[m].mean() * 100), 4),
        }
    return out


# ===========================================================================
# 2. As-of player features
# ===========================================================================
EV_COLS: tuple[str, ...] = tuple(f"ev_{c}" for c in EVENT_CLASSES)
RATE_COLS: tuple[str, ...] = tuple(f"rate_{c}" for c in EVENT_CLASSES)
PREV_COLS: tuple[str, ...] = tuple(f"prev_rate_{c}" for c in EVENT_CLASSES)

#: "total" is a pseudo-class: every credited event of any of the five classes.
#: It is what "as-of usage rank" in the pre-registration's role definition is
#: ranked on, and it is built and shrunk by exactly the same code as a real
#: class so the role assignment cannot use a differently-constructed rate.
TOTAL_CLASS = "total"
ALL_RATE_CLASSES: tuple[str, ...] = (*EVENT_CLASSES, TOTAL_CLASS)


def _long_on_floor(ev: pd.DataFrame) -> pd.DataFrame:
    """(event x alternative) long frame: five rows per event, one per on-floor
    offensive player, flagged with whether he was the credited one."""
    n = len(ev)
    alt = ev[[f"alt_{k + 1}" for k in range(N_ALT)]].to_numpy(dtype="int64")
    rep = np.repeat(np.arange(n), N_ALT)
    slot = np.tile(np.arange(N_ALT), n)
    return pd.DataFrame({
        "season": ev["season"].to_numpy()[rep],
        "player_id": alt.reshape(-1),
        "game_id": ev["game_id"].to_numpy()[rep],
        "game_date": ev["game_date"].to_numpy()[rep],
        "team_id": ev["team_id"].to_numpy()[rep],
        "event_class": ev["event_class"].to_numpy()[rep],
        "credited": slot == np.repeat(ev["y"].to_numpy(), N_ALT),
    })


def load_minutes(seasons, player_box_dir: Path | str = DEFAULT_PLAYER_BOX_DIR,
                 crosswalk_path: Path | str = DEFAULT_CROSSWALK) -> pd.DataFrame:
    """(season, cbbd player id, game_id) -> hoopR `player_box` minutes.

    The bridge is `data/processed/player_crosswalk.parquet`
    (`cbb_sim.data.player_ids`), whose coverage is 2024-2026 -- exactly this
    model's window. Unmatched rows are left missing; `build_player_asof` reports
    the join rate rather than imputing a zero, because a zero would read as
    "played no minutes" rather than "not matched"."""
    cwp = Path(crosswalk_path)
    if not cwp.exists():
        return pd.DataFrame(columns=["season", "player_id", "game_id", "minutes"])
    cw = pd.read_parquet(cwp, columns=["season", "cbbd_player_id", "espn_athlete_id",
                                       "match_method"])
    cw = cw[cw["match_method"].to_numpy() != "unmatched"].dropna(subset=["espn_athlete_id"])
    cw["espn_athlete_id"] = cw["espn_athlete_id"].astype("int64")
    frames = []
    for s in sorted({int(x) for x in seasons}):
        p = Path(player_box_dir) / f"player_box_{s}.parquet"
        if not p.exists():
            continue
        pb = pd.read_parquet(p, columns=["game_id", "season", "athlete_id", "minutes"])
        pb = pb[pb["season"].to_numpy() == s].copy()
        pb["minutes"] = pd.to_numeric(pb["minutes"], errors="coerce").fillna(0.0)
        pb["espn_athlete_id"] = pd.to_numeric(pb["athlete_id"], errors="coerce")
        pb = pb.dropna(subset=["espn_athlete_id"])
        pb["espn_athlete_id"] = pb["espn_athlete_id"].astype("int64")
        m = pb.merge(cw.loc[cw["season"].to_numpy() == s,
                            ["cbbd_player_id", "espn_athlete_id"]],
                     on="espn_athlete_id", how="inner")
        m = m.rename(columns={"cbbd_player_id": "player_id"})
        m["game_id"] = pd.to_numeric(m["game_id"], errors="coerce")
        frames.append(m[["season", "player_id", "game_id", "minutes"]].dropna())
    if not frames:
        return pd.DataFrame(columns=["season", "player_id", "game_id", "minutes"])
    out = pd.concat(frames, ignore_index=True)
    out["player_id"] = out["player_id"].astype("int64")
    out["game_id"] = out["game_id"].astype("int64")
    return out.groupby(["season", "player_id", "game_id"], as_index=False)["minutes"].sum()


def build_player_asof(events_by_season: dict[int, pd.DataFrame],
                      minutes: pd.DataFrame | None = None,
                      positions: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (season, player, game) carrying every as-of input.

    Leak safety: every column is an expanding sum over that player's games
    STRICTLY BEFORE the current one within the season
    (`prob_metrics.expanding_asof`), so no row can see its own game. The
    prior-season columns are a COMPLETED season and are in the past by
    construction.

    `events_by_season` maps season -> the USABLE event table of that season
    (`usable_events`). Seasons are pooled so that a season's prior-season
    columns can be built from the season before it when that season is present.
    """
    lg = pd.concat([_long_on_floor(events_by_season[s]) for s in sorted(events_by_season)],
                   ignore_index=True)

    pg = lg.groupby(["season", "player_id", "game_id"], as_index=False).agg(
        exposure=("credited", "size"), game_date=("game_date", "first"),
        team_id=("team_id", "first"))
    cred = lg[lg["credited"].to_numpy()]
    wide = (cred.groupby(["season", "player_id", "game_id", "event_class"])
            .size().unstack("event_class").reindex(columns=list(EVENT_CLASSES))
            .fillna(0.0).reset_index())
    wide.columns = list(wide.columns[:3]) + list(EV_COLS)
    pg = pg.merge(wide, on=["season", "player_id", "game_id"], how="left")
    for c in EV_COLS:
        pg[c] = pg[c].fillna(0.0)
    pg["ev_total"] = pg[list(EV_COLS)].sum(axis=1)

    if minutes is not None and len(minutes):
        pg = pg.merge(minutes, on=["season", "player_id", "game_id"], how="left")
    else:
        pg["minutes"] = np.nan
    minutes_join_pct = round(float(pg["minutes"].notna().mean() * 100), 4)
    pg["minutes_f"] = pg["minutes"].fillna(0.0)
    pg["minutes_known"] = pg["minutes"].notna().astype("float64")

    pg = pg.sort_values(["season", "player_id", "game_date", "game_id"],
                        kind="stable").reset_index(drop=True)
    sum_cols = ["exposure", *EV_COLS, "ev_total", "minutes_f", "minutes_known"]
    asof = PM.expanding_asof(pg, ["season", "player_id"], sum_cols)
    out = pd.concat([pg[["season", "player_id", "game_id", "game_date"]], asof], axis=1)
    out = out.rename(columns={"exposure": "exposure_asof", "minutes_f": "minutes_asof",
                              "minutes_known": "minutes_games_asof",
                              "n_prior": "games_asof"})

    # ---- prior season totals (a COMPLETED season) -------------------------
    tot = pg.groupby(["season", "player_id"], as_index=False)[
        ["exposure", *EV_COLS, "ev_total", "minutes_f"]].sum()
    tot["season"] = tot["season"] + 1
    tot = tot.rename(columns={"exposure": "prev_exposure", "minutes_f": "prev_minutes",
                              **{c: f"prev_{c}" for c in (*EV_COLS, "ev_total")}})
    out = out.merge(tot, on=["season", "player_id"], how="left")

    modal = (pg.groupby(["season", "player_id"])["team_id"]
             .agg(lambda x: x.value_counts().index[0] if len(x.dropna()) else np.nan)
             .reset_index().rename(columns={"team_id": "prev_team_id"}))
    modal["season"] = modal["season"] + 1
    out = out.merge(modal, on=["season", "player_id"], how="left")

    # ---- position group ---------------------------------------------------
    pos = positions if positions is not None else load_positions()
    pos = pos.rename(columns={"shooter_id": "player_id"})
    out = out.merge(pos, on="player_id", how="left")
    out["position_group"] = out["position_group"].fillna("UNK")

    # ---- league and position as-of priors, by calendar date ---------------
    day = pg.groupby(["season", "game_date"], as_index=False)[
        ["exposure", *EV_COLS, "ev_total"]].sum()
    day = day.sort_values(["season", "game_date"], kind="stable").reset_index(drop=True)
    lga = PM.expanding_asof(day, ["season"], ["exposure", *EV_COLS, "ev_total"])
    lga.columns = [f"lg_{c}" for c in lga.columns]
    day = pd.concat([day[["season", "game_date"]], lga], axis=1)
    out = out.merge(day, on=["season", "game_date"], how="left")

    pgp = pg.merge(pos, on="player_id", how="left")
    pgp["position_group"] = pgp["position_group"].fillna("UNK")
    pday = pgp.groupby(["season", "position_group", "game_date"], as_index=False)[
        ["exposure", *EV_COLS, "ev_total"]].sum()
    pday = pday.sort_values(["season", "position_group", "game_date"],
                            kind="stable").reset_index(drop=True)
    pa = PM.expanding_asof(pday, ["season", "position_group"],
                           ["exposure", *EV_COLS, "ev_total"])
    pa.columns = [f"pos_{c}" for c in pa.columns]
    pday = pd.concat([pday[["season", "position_group", "game_date"]], pa], axis=1)
    out = out.merge(pday, on=["season", "position_group", "game_date"], how="left")

    # ---- rates ------------------------------------------------------------
    expo = out["exposure_asof"].to_numpy(dtype="float64")
    for c in ALL_RATE_CLASSES:
        out[f"rate_{c}"] = _safe_div(out[f"ev_{c}"].to_numpy(dtype="float64"), expo)
        out[f"lg_rate_{c}"] = _safe_div(out[f"lg_ev_{c}"].to_numpy(dtype="float64"),
                                        out["lg_exposure"].to_numpy(dtype="float64"))
        out[f"pos_rate_{c}"] = _safe_div(out[f"pos_ev_{c}"].to_numpy(dtype="float64"),
                                         out["pos_exposure"].to_numpy(dtype="float64"))
        out[f"prev_rate_{c}"] = _safe_div(out[f"prev_ev_{c}"].to_numpy(dtype="float64"),
                                          out["prev_exposure"].to_numpy(dtype="float64"))
    # A season's opening day has nothing strictly earlier, so its as-of league
    # and position priors are undefined. They are back-filled from that season's
    # own next available date, computed on the date-sorted table -- the same
    # opening-day rule `free_throw` uses, and the only forward-looking value in
    # the module. It touches opening day alone.
    order = out.sort_values(["season", "game_date"], kind="stable").index
    for c in ALL_RATE_CLASSES:
        for pre, grp in (("lg", ["season"]), ("pos", ["season", "position_group"])):
            col = f"{pre}_rate_{c}"
            filled = out.loc[order].groupby(grp)[col].bfill()
            out[col] = filled.reindex(out.index)
            out[col] = out[col].fillna(out[col].median())

    out["has_prior_season"] = np.isfinite(
        out["prev_exposure"].to_numpy(dtype="float64")).astype("float64")
    out["prev_exposure"] = out["prev_exposure"].fillna(0.0)
    out["prev_minutes"] = out["prev_minutes"].fillna(0.0)
    for c in (*EV_COLS, "ev_total"):
        out[f"prev_{c}"] = out[f"prev_{c}"].fillna(0.0)
    mpg = _safe_div(out["minutes_asof"].to_numpy(dtype="float64"),
                    out["minutes_games_asof"].to_numpy(dtype="float64"))
    out["minutes_per_game_asof"] = np.nan_to_num(mpg, nan=0.0)
    out.attrs["minutes_join_pct"] = minutes_join_pct
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, EPS), np.nan)


# ===========================================================================
# 3. The per-class design
# ===========================================================================
#: Columns of the as-of table carried onto every alternative slot.
SLOT_COLS: tuple[str, ...] = (
    "exposure_asof", "games_asof", "minutes_asof", "minutes_per_game_asof",
    "has_prior_season", "prev_exposure", "position_group",
)


def build_usage_design(events: pd.DataFrame, asof: pd.DataFrame,
                       event_class: str) -> pd.DataFrame:
    """One row per event of `event_class`, with the five alternatives' as-of
    features attached as `*_1 .. *_5` columns.

    The wide form is deliberate: the choice is among exactly five alternatives,
    so an (n, 5) block is the natural shape for the Dirichlet arms and for a
    conditional logit, and it keeps the design one fifth the size of the long
    form the tree arm needs (`long_frame` builds that on demand)."""
    if event_class not in EVENT_CLASSES:
        raise KeyError(f"unknown event class {event_class!r}")
    ev = events[events["event_class"].to_numpy() == event_class].reset_index(drop=True)
    keep = ["season", "player_id", "game_id", f"ev_{event_class}", f"rate_{event_class}",
            f"lg_rate_{event_class}", f"pos_rate_{event_class}",
            f"prev_rate_{event_class}", f"prev_ev_{event_class}", "prev_team_id",
            "ev_total", "lg_rate_total", "pos_rate_total", "prev_rate_total",
            *SLOT_COLS]
    keep = list(dict.fromkeys(keep))
    a = asof[keep].copy()
    a["position_code"] = a["position_group"].map(
        {p: i for i, p in enumerate(POSITION_LEVELS)}).fillna(
        len(POSITION_LEVELS) - 1).astype("int8")
    a = a.drop(columns=["position_group"])

    out = ev
    for k in range(1, N_ALT + 1):
        m = a.rename(columns={"player_id": f"alt_{k}"})
        m = m.rename(columns={c: f"{c}_{k}" for c in m.columns
                              if c not in ("season", "game_id", f"alt_{k}")})
        out = out.merge(m, on=["season", "game_id", f"alt_{k}"], how="left")

    zero_fill = ("exposure_asof", "games_asof", "minutes_asof", "minutes_per_game_asof",
                 f"ev_{event_class}", f"prev_ev_{event_class}", "prev_exposure",
                 "has_prior_season", "ev_total")
    for k in range(1, N_ALT + 1):
        for c in zero_fill:
            out[f"{c}_{k}"] = out[f"{c}_{k}"].fillna(0.0)
        out[f"position_code_{k}"] = out[f"position_code_{k}"].fillna(
            len(POSITION_LEVELS) - 1)
    # A player with no as-of row at all (his first game of the season) has zero
    # exposure, so every rate comes entirely from the prior -- which is what the
    # shrinkage formula does with a zero denominator anyway. The prior itself is
    # taken from the lineup's other members, then from the column median, so a
    # lineup of five debutants still has a defined prior rather than a NaN.
    for c in (f"lg_rate_{event_class}", f"pos_rate_{event_class}",
              "lg_rate_total", "pos_rate_total"):
        cols = [f"{c}_{k}" for k in range(1, N_ALT + 1)]
        ref = out[cols].median(axis=1)
        med = float(np.nanmedian(out[cols].to_numpy(dtype="float64")))
        for col in cols:
            out[col] = out[col].fillna(ref).fillna(med)
    out = out.copy()          # de-fragment before the derived columns go on
    for k in range(1, N_ALT + 1):
        for c in (event_class, TOTAL_CLASS):
            out[f"prev_rate_{c}_{k}"] = out[f"prev_rate_{c}_{k}"].fillna(
                out[f"pos_rate_{c}_{k}"])
        out[f"is_transfer_{k}"] = (
            np.isfinite(out[f"prev_team_id_{k}"].to_numpy(dtype="float64"))
            & (out[f"prev_team_id_{k}"].to_numpy(dtype="float64")
               != out["team_id"].to_numpy())
            & (out[f"has_prior_season_{k}"].to_numpy() > 0)).astype("int8")
    out["event_class"] = event_class
    return out


def _block(d: pd.DataFrame, stem: str) -> np.ndarray:
    return d[[f"{stem}_{k}" for k in range(1, N_ALT + 1)]].to_numpy(dtype="float64")


def shrunk_rate(d: pd.DataFrame, event_class: str, prior_kind: str,
                m: float) -> np.ndarray:
    """(n, 5) shrunk as-of class rate for every alternative.

    `(m * prior + events) / (m + exposure)`: `m` is in pseudo-ON-FLOOR-EVENTS,
    the same unit as the denominator, so "m = 200" literally means "200 events
    of history before a player's own rate outweighs the prior"."""
    num = _block(d, f"ev_{event_class}")
    den = _block(d, "exposure_asof")
    prior = _prior_block(d, event_class, prior_kind)
    return (m * prior + num) / (m + den)


def _prior_block(d: pd.DataFrame, event_class: str, prior_kind: str) -> np.ndarray:
    if prior_kind == "league":
        return _block(d, f"lg_rate_{event_class}")
    if prior_kind == "position":
        return _block(d, f"pos_rate_{event_class}")
    if prior_kind == "prior_season":
        # returning players use their own completed prior season; newcomers have
        # none and fall back to the POSITION prior, which is exactly what
        # "prior-season rates for returning players, position group prior for
        # new players" says in the pre-registration.
        return _block(d, f"prev_rate_{event_class}")
    raise KeyError(f"unknown prior kind {prior_kind!r}")


def normalise(r: np.ndarray) -> np.ndarray:
    """Rows of `r` normalised to sum to 1, with a non-finite or all-zero row
    falling back to the uniform over the five -- the only honest answer when the
    model has no information about a lineup."""
    r = np.asarray(r, dtype="float64")
    r = np.where(np.isfinite(r), np.maximum(r, 0.0), 0.0)
    s = r.sum(axis=1, keepdims=True)
    bad = s <= 0
    return np.where(bad, 1.0 / r.shape[1], r / np.where(bad, 1.0, s))


# ===========================================================================
# 4. U1: proportional
# ===========================================================================
def u1_probs(d: pd.DataFrame, event_class: str, prior_kind: str, m: float) -> np.ndarray:
    return normalise(shrunk_rate(d, event_class, prior_kind, m))


def prior_season_available(d: pd.DataFrame) -> bool:
    """Does this slice carry ANY prior-season on-floor history? False on the F1
    training fold by construction (L13), which is why `fit_shrinkage` reports
    the prior-season rung as unidentified there instead of scoring it."""
    return bool(_block(d, "has_prior_season").max() > 0)


def fit_shrinkage(tr: pd.DataFrame, event_class: str,
                  grid: tuple[float, ...] = SHRINK_GRID,
                  kinds: tuple[str, ...] = PRIOR_KINDS) -> dict:
    """Grid over (prior, strength) on the TRAINING fold's own log loss.

    Honest in-fold, for the reason `free_throw.fit_eb` states: every quantity
    the arm uses is an as-of feature, so a training-fold log loss is already a
    walk-forward number. A prior whose column is identically absent in the
    training fold (prior_season on F1 train, L13) is reported as `unidentified`
    and excluded from the choice rather than silently winning a tie."""
    y = tr["y"].to_numpy()
    have_prev = prior_season_available(tr)
    rows, best = [], None
    for kind in kinds:
        if kind == "prior_season" and not have_prev:
            rows.append({"prior": kind, "m": None, "train_log_loss": None,
                         "status": "unidentified (no prior-season history in this fold)"})
            continue
        for mm in grid:
            p = u1_probs(tr, event_class, kind, float(mm))
            ll = PM.log_loss(y, p)
            rows.append({"prior": kind, "m": float(mm),
                         "train_log_loss": round(ll, 6), "status": "ok"})
            if best is None or ll < best["train_log_loss"]:
                best = rows[-1]
    return {"grid": rows, "best": best, "prior_season_available": have_prev}


# ===========================================================================
# 5. Roles, profiles and the Dirichlet arms (U2, U3)
# ===========================================================================
def assign_roles(d: pd.DataFrame, prior_kind: str, m: float) -> np.ndarray:
    """(n, 5) integer role codes into `ROLES`.

    A centre is a big. Among the non-centres, the two highest as-of TOTAL usage
    rates in the lineup are the primary handlers and the rest are wings --
    "roles from position group and as-of usage rank", with the rank taken over
    the lineup (the candidate set the sampler is handed) and the rate taken
    entirely from pregame history."""
    pos = _block(d, "position_code")
    big = pos == POSITION_LEVELS.index("C")
    usage = shrunk_rate(d, TOTAL_CLASS, prior_kind, m)
    u = np.where(big, -np.inf, usage)
    order = np.argsort(-u, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(N_ALT)[None, :].repeat(len(u), 0), axis=1)
    role = np.full(pos.shape, ROLES.index("wing"), dtype="int8")
    role[~big & (rank < HANDLER_RANK)] = ROLES.index("handler")
    role[big] = ROLES.index("big")
    return role


@dataclass
class Profile:
    """One (game, team, class) allocation profile: the team's players with any
    as-of rate for this class, their normalised weights, and their roles.

    This is the object the Dirichlet draw lives on. Crucially it is the TEAM's
    profile, not the lineup's: the usage realisation has to persist across
    substitutions for a game to carry any game-to-game usage variance at all,
    which is the defect CFB measured as "too narrow"."""
    players: np.ndarray
    weights: np.ndarray
    roles: np.ndarray
    index: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {int(p): i for i, p in enumerate(self.players)}


@dataclass
class ProfileSet:
    """Every (game, team) profile of one design, in flat CSR form.

    `weights[offsets[g]:offsets[g+1]]` is profile g; `codes[i]` is row i's
    profile; `gslot[i, k]` is the FLAT index of row i's k-th alternative. The
    flat form is what makes a season-wide Monte Carlo one vectorised gamma call
    instead of eleven thousand."""
    players: np.ndarray
    weights: np.ndarray
    roles: np.ndarray
    offsets: np.ndarray
    codes: np.ndarray
    gslot: np.ndarray
    game_ids: np.ndarray

    @property
    def n_profiles(self) -> int:
        return len(self.offsets) - 1

    def profile(self, g: int) -> Profile:
        lo, hi = int(self.offsets[g]), int(self.offsets[g + 1])
        return Profile(players=self.players[lo:hi], weights=self.weights[lo:hi],
                       roles=self.roles[lo:hi])

    def single_role(self) -> ProfileSet:
        """The same profiles with every player in ONE role.

        This is how U2 is built: a single flat `Dirichlet(a * w)` over the whole
        profile is the hierarchical form with one role and a deterministic
        between level, so the two arms share one implementation and differ only
        in the role vector. Keeping the roles would make a role of size one
        (a lineup's only centre) deterministic, which is a per-role behaviour,
        not the single-level design the pre-registration names."""
        return ProfileSet(players=self.players, weights=self.weights,
                          roles=np.zeros_like(self.roles), offsets=self.offsets,
                          codes=self.codes, gslot=self.gslot, game_ids=self.game_ids)


def build_profiles(d: pd.DataFrame, rates: np.ndarray, roles: np.ndarray) -> ProfileSet:
    """Group a design's rows into (game, team) profiles.

    A player's profile weight is his as-of rate, which is constant within a game
    by construction (every as-of feature is keyed on the game), so the group
    mean is that constant."""
    gid = d["game_id"].to_numpy(dtype="int64")
    tid = d["team_id"].to_numpy(dtype="int64")
    codes, _ = pd.factorize(pd.MultiIndex.from_arrays([gid, tid]), sort=False)
    alts = d[[f"alt_{k}" for k in range(1, N_ALT + 1)]].to_numpy(dtype="int64")

    flat = pd.DataFrame({
        "g": np.repeat(codes, N_ALT),
        "p": alts.reshape(-1),
        "r": np.asarray(rates, dtype="float64").reshape(-1),
        "role": np.asarray(roles).reshape(-1).astype("int16"),
    })
    agg = flat.groupby(["g", "p"], sort=True, as_index=False).agg(
        r=("r", "mean"), role=("role", "median"))
    agg["role"] = agg["role"].round().astype("int8")
    agg["slot"] = agg.groupby("g").cumcount()
    sizes = agg.groupby("g").size().to_numpy()
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype("int64")
    agg["gslot"] = np.arange(len(agg))

    # normalise the weights inside each profile
    w = np.maximum(agg["r"].to_numpy(dtype="float64"), 0.0)
    tot = np.add.reduceat(w, offsets[:-1])
    tot_rep = np.repeat(np.where(tot > 0, tot, 1.0), sizes)
    w = np.where(np.repeat(tot > 0, sizes), w / tot_rep,
                 np.repeat(1.0 / sizes, sizes))

    lookup = pd.Series(agg["gslot"].to_numpy(),
                       index=pd.MultiIndex.from_arrays([agg["g"], agg["p"]]))
    gslot = np.empty((len(d), N_ALT), dtype="int64")
    for k in range(N_ALT):
        gslot[:, k] = lookup.reindex(
            pd.MultiIndex.from_arrays([codes, alts[:, k]])).to_numpy()

    rep_gid = pd.Series(gid).groupby(codes).first().sort_index().to_numpy()
    return ProfileSet(players=agg["p"].to_numpy(dtype="int64"), weights=w,
                      roles=agg["role"].to_numpy(), offsets=offsets,
                      codes=codes, gslot=gslot, game_ids=rep_gid)


def _gamma_ppf(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Gamma(a, 1) variates by inverse CDF, so a draw is a pure function of its
    uniform and the stream reproduces without carrying Generator state."""
    return stats.gamma.ppf(np.clip(u, 1e-12, 1 - 1e-12), np.maximum(a, 1e-6))


def realize_shares(weights: np.ndarray, roles: np.ndarray, alphas: dict,
                   u: np.ndarray) -> np.ndarray:
    """(R, K) mean-preserving share realisations of `weights` -- THE ENGINE PATH.

    Generative form, ported from `cfb_props_sim.sim.usage_alloc.realize_shares`
    (parameterisation (iv) of that module's fit doc):

        P_f        = sum of weights inside role f
        S_between ~ Dirichlet(a_between * P)            (skip if one role)
        S_within,f ~ Dirichlet(a_f * w_{i|f})           (per role)
        share_i    = S_within,f(i) * S_between,f(i)

    E[share_i] = P_f * (w_i / P_f) = w_i EXACTLY at every level, because both
    Dirichlets are centred on their own conditional means. A single flat
    `a_i = alpha_i * w_i` with a VARYING alpha destroys that -- the measured CFB
    failure (top-1 share 1.04-1.13x real) this form exists to avoid.

    `u` is (R, K + |ROLES|) uniforms: the first K drive the within-role gammas,
    the next |ROLES| the between-role one. `a = inf` on a knob makes that level
    deterministic, so all-infinite `alphas` returns `weights` broadcast R times,
    which is exactly U1."""
    w = np.asarray(weights, dtype="float64")
    K = len(w)
    R = u.shape[0]
    roles = np.asarray(roles)
    present = [r for r in range(N_ROLE) if (roles == r).any()]
    a_b = float(alphas.get("a_between", np.inf))

    Pf = np.array([w[roles == r].sum() for r in present], dtype="float64")
    if len(present) > 1 and np.isfinite(a_b):
        gb = _gamma_ppf(np.maximum(a_b * Pf, 1e-6)[None, :], u[:, K:K + len(present)])
        Sb = gb / np.maximum(gb.sum(axis=1, keepdims=True), EPS)
    else:
        Sb = np.repeat(Pf[None, :], R, axis=0)

    S = np.zeros((R, K), dtype="float64")
    for j, r in enumerate(present):
        msk = roles == r
        ww = w[msk]
        tot = ww.sum()
        if tot <= 0:
            continue
        pw = ww / tot
        a_w = float(alphas.get(f"a_{ROLES[r]}", np.inf))
        if msk.sum() > 1 and np.isfinite(a_w):
            gw = _gamma_ppf(np.maximum(a_w * pw, 1e-6)[None, :],
                            u[:, np.flatnonzero(msk)])
            Sw = gw / np.maximum(gw.sum(axis=1, keepdims=True), EPS)
        else:
            Sw = np.repeat(pw[None, :], R, axis=0)
        S[:, msk] = Sw * Sb[:, [j]]
    tot = S.sum(axis=1, keepdims=True)
    return np.where(tot > 0, S / np.maximum(tot, EPS), 1.0 / K)


def realize_batch(ps: ProfileSet, alphas: dict, n_draw: int,
                  seed: int = 0) -> np.ndarray:
    """(n_draw, T) share realisations for EVERY profile at once -- THE TRAINER
    PATH. Same generative form as `realize_shares`, same mean preservation, one
    vectorised gamma call instead of one per profile.

    The draws come from `numpy`'s gamma sampler rather than the counter-based
    inverse CDF, because a season-wide 200-draw Monte Carlo is 26 million gamma
    variates and the exact path is two orders of magnitude slower. The two are
    asserted to agree on mean and SD in `tests/test_usage.py`, the same
    batched-twin discipline `cfb-props-sim/tests/test_usage_alloc.py` uses."""
    sizes = np.diff(ps.offsets)
    T = len(ps.weights)
    w = ps.weights
    roles = ps.roles.astype("int64")
    prof_of = np.repeat(np.arange(ps.n_profiles), sizes)
    rng = np.random.default_rng(int(seed))

    # ---- role masses per (profile, role) ---------------------------------
    pr_role = prof_of * N_ROLE + roles
    role_mass = np.bincount(pr_role, weights=w, minlength=ps.n_profiles * N_ROLE)
    mass_of = role_mass[pr_role]

    a_b = float(alphas.get("a_between", np.inf))
    a_w = np.array([float(alphas.get(f"a_{r}", np.inf)) for r in ROLES])
    role_present = role_mass.reshape(-1, N_ROLE) > 0
    n_present = role_present.sum(axis=1)

    # ---- within-role draw -------------------------------------------------
    pw = np.where(mass_of > 0, w / np.maximum(mass_of, EPS), 0.0)
    role_size = np.bincount(pr_role, minlength=ps.n_profiles * N_ROLE)[pr_role]
    a_elem = a_w[roles] * pw
    dispersed = np.isfinite(a_w[roles]) & (role_size > 1) & (pw > 0)
    Sw = np.repeat(pw[None, :], n_draw, axis=0)
    if dispersed.any():
        g = rng.standard_gamma(np.maximum(a_elem[dispersed], 1e-6),
                               size=(n_draw, int(dispersed.sum())))
        tmp = np.zeros((n_draw, T))
        tmp[:, dispersed] = g
        denom = np.zeros((n_draw, ps.n_profiles * N_ROLE))
        np.add.at(denom, (slice(None), pr_role), tmp)
        d_of = denom[:, pr_role]
        Sw = np.where(dispersed[None, :] & (d_of > 0), tmp / np.maximum(d_of, EPS), Sw)

    # ---- between-role draw ------------------------------------------------
    if np.isfinite(a_b):
        shape = np.maximum(a_b * role_mass, 1e-9).reshape(-1, N_ROLE)
        gb = rng.standard_gamma(np.where(role_present, shape, 1e-9),
                                size=(n_draw, ps.n_profiles, N_ROLE))
        gb = np.where(role_present[None, :, :], gb, 0.0)
        tot = gb.sum(axis=2, keepdims=True)
        Sb = np.where(tot > 0, gb / np.maximum(tot, EPS), 0.0)
        single = (n_present <= 1)
        if single.any():
            Sb[:, single, :] = role_mass.reshape(-1, N_ROLE)[None, single, :]
        Sb_of = Sb.reshape(n_draw, -1)[:, pr_role]
    else:
        Sb_of = np.repeat(mass_of[None, :], n_draw, axis=0)

    S = Sw * Sb_of
    tot = np.zeros((n_draw, ps.n_profiles))
    np.add.at(tot, (slice(None), prof_of), S)
    t_of = tot[:, prof_of]
    return np.where(t_of > 0, S / np.maximum(t_of, EPS),
                    np.repeat((1.0 / np.diff(ps.offsets))[prof_of][None, :], n_draw, axis=0))


def single_level_alphas(a: float) -> dict:
    """U2's knobs: one concentration, no between-role level. `a_between = inf`
    makes the between level deterministic, so the draw is a single flat
    Dirichlet(a * w) over the whole profile -- the CFB single-level design."""
    return {"a_between": float("inf"), **{f"a_{r}": float(a) for r in ROLES}}


def hier_alphas(a_between: float, a_handler: float, a_wing: float,
                a_big: float) -> dict:
    return {"a_between": float(a_between), "a_handler": float(a_handler),
            "a_wing": float(a_wing), "a_big": float(a_big)}


NO_DISPERSION: dict = {"a_between": float("inf"), **{f"a_{r}": float("inf") for r in ROLES}}


def _conditional(S_rows: np.ndarray, gslot: np.ndarray) -> np.ndarray:
    """(R, E, 5) conditional probabilities from (R, T) shares and an (E, 5)
    index of which flat slots the five on the floor occupy."""
    take = S_rows[:, gslot]
    return take / np.maximum(take.sum(axis=2, keepdims=True), EPS)


def marginal_probs(ps: ProfileSet, alphas: dict, n_draw: int = 200, seed: int = 0,
                   chunk: int = 20000) -> np.ndarray:
    """(n, 5) MARGINAL predictive of a Dirichlet arm: the Monte-Carlo average
    over the game's share realisations of `S_i / sum_{j in five} S_j`.

    This is the honest PREGAME predictive of U2/U3. It is not equal to U1's
    normalised weights because the allocation conditions on a SUBSET of the
    profile; the module docstring explains why, and the difference is the price
    the arm pays in sharpness for its dispersion."""
    S = realize_batch(ps, alphas, n_draw, seed)
    n = len(ps.gslot)
    out = np.empty((n, N_ALT), dtype="float64")
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        out[lo:hi] = _conditional(S, ps.gslot[lo:hi]).mean(axis=0)
    return normalise(out)


def sequential_probs(ps: ProfileSet, y: np.ndarray, alphas: dict, n_draw: int = 200,
                     seed: int = 0) -> np.ndarray:
    """(n, 5) POLYA-URN predictive: the same Dirichlet conditioned on the game's
    own earlier events of this class, by sequential importance weighting with
    the prior as the proposal (exact up to Monte Carlo, no resampling).

    NOT a pregame quantity and never a decision input -- it is the diagnostic
    that says how much within-game usage concentration is real."""
    S = realize_batch(ps, alphas, n_draw, seed)
    y = np.asarray(y)
    codes = ps.codes
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.concatenate(
        [[True], codes[order][1:] != codes[order][:-1], [True]]))
    out = np.empty((len(codes), N_ALT), dtype="float64")
    for b in range(len(bounds) - 1):
        idx = order[bounds[b]:bounds[b + 1]]
        p = _conditional(S, ps.gslot[idx])            # (R, E, 5)
        w = np.ones(n_draw)
        for t in range(len(idx)):
            out[idx[t]] = (w[:, None] * p[:, t, :]).sum(axis=0) / max(w.sum(), EPS)
            w = w * np.maximum(p[:, t, y[idx[t]]], EPS)
            s = w.sum()
            w = np.ones(n_draw) if (s <= 0 or not np.isfinite(s)) else w / s * n_draw
    return normalise(out)


# ===========================================================================
# 6. U4: conditional logit
# ===========================================================================
#: The alternative-varying design of the conditional logit. State variables are
#: alternative-INVARIANT and cancel in a conditional logit, so they enter only
#: as interactions with an alternative's own share and role -- which is the only
#: way the pre-registered state features can be identified at all.
CL_FEATURES: tuple[str, ...] = (
    "log_share", "log_prior_share", "log_exposure", "log_minutes",
    "is_G", "is_F", "is_C", "is_transfer",
    "log_share_x_scorediff", "log_share_x_sec", "log_share_x_chance",
    "is_C_x_scorediff", "is_C_x_sec",
)


def cl_design(d: pd.DataFrame, event_class: str, prior_kind: str,
              m: float) -> tuple[np.ndarray, list[str]]:
    """(n, 5, F) conditional-logit design, centred within the choice set."""
    q = normalise(shrunk_rate(d, event_class, prior_kind, m))
    log_share = np.log(np.maximum(q, 1e-9))
    log_prev = np.log(np.maximum(normalise(_block(d, f"prev_rate_{event_class}")), 1e-9))
    pos = _block(d, "position_code")
    is_c = (pos == POSITION_LEVELS.index("C")).astype("float64")
    sd = (d["score_diff"].to_numpy(dtype="float64") / 10.0)[:, None]
    sec = (d["sec_remaining"].to_numpy(dtype="float64") / 600.0)[:, None]
    ch = (d["chance_number"].to_numpy(dtype="float64") - 1.0)[:, None]
    cols = {
        "log_share": log_share,
        "log_prior_share": log_prev,
        "log_exposure": np.log1p(_block(d, "exposure_asof")),
        "log_minutes": np.log1p(_block(d, "minutes_asof")),
        "is_G": (pos == POSITION_LEVELS.index("G")).astype("float64"),
        "is_F": (pos == POSITION_LEVELS.index("F")).astype("float64"),
        "is_C": is_c,
        "is_transfer": _block(d, "is_transfer"),
        "log_share_x_scorediff": log_share * sd,
        "log_share_x_sec": log_share * sec,
        "log_share_x_chance": log_share * ch,
        "is_C_x_scorediff": is_c * sd,
        "is_C_x_sec": is_c * sec,
    }
    names = list(CL_FEATURES)
    X = np.stack([cols[c] for c in names], axis=2)
    return X - X.mean(axis=1, keepdims=True), names


#: Features that READ the prior-season block. On a fold whose training slice has
#: no prior-season history at all (F1 train = 2024, L13) these are not merely
#: constant -- `build_usage_design` falls their value back to the POSITION prior,
#: so the column still varies across the five and a naive fit happily estimates
#: a coefficient for a variable that MEANS something different in the test fold.
#: Measured cost of not catching this: the F1 conditional logit on FGA_3 scores
#: 3.74 against U1's 1.46. `unidentified_features` is what catches it.
PRIOR_SEASON_FEATURES: tuple[str, ...] = ("log_prior_share", "prior_rate")


def unidentified_features(tr: pd.DataFrame) -> list[str]:
    """Feature names whose meaning is not identified by THIS training fold.

    Only one case exists today: the prior-season block on a fold with no
    prior-season on-floor history. The trainer drops what this returns and
    records the drop (the `cbb_sim.models.clock` precedent), rather than letting
    a meaning-shifted column ship."""
    return [] if prior_season_available(tr) else list(PRIOR_SEASON_FEATURES)


def drop_constant_features(X: np.ndarray, names: list[str]
                           ) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove columns with no within-choice-set variation in THIS fold.

    A conditional logit cannot identify a column that is identical across the
    five (it cancels), and a column that is identically zero in the training
    fold -- which is what the prior-season block is on F1 train (L13) -- has an
    unidentified coefficient. The drop is returned so the trainer can record it
    rather than let it pass silently (the `cbb_sim.models.clock` precedent)."""
    scale = np.abs(X).max(axis=(0, 1))
    keep = scale > 1e-10
    return (X[:, :, keep],
            [n for n, k in zip(names, keep, strict=False) if k],
            [n for n, k in zip(names, keep, strict=False) if not k])


class CondLogitArm:
    """Conditional (multinomial) logit over the five alternatives with shared
    coefficients, L2-penalised. Fitted by L-BFGS on the exact gradient."""

    def __init__(self, l2: float = 1.0, max_iter: int = 400):
        self.l2, self.max_iter = float(l2), int(max_iter)

    @staticmethod
    def _nll(beta, X, y, l2):
        eta = X @ beta
        eta = eta - eta.max(axis=1, keepdims=True)
        e = np.exp(eta)
        p = e / e.sum(axis=1, keepdims=True)
        n = len(y)
        ll = -np.log(np.clip(p[np.arange(n), y], EPS, 1.0)).mean()
        obs = np.zeros_like(p)
        obs[np.arange(n), y] = 1.0
        grad = np.einsum("nkf,nk->f", X, (p - obs)) / n
        return ll + l2 * float(beta @ beta) / (2 * n), grad + l2 * beta / n

    def fit(self, X: np.ndarray, y: np.ndarray) -> CondLogitArm:
        from scipy.optimize import minimize
        X = np.ascontiguousarray(X, dtype="float64")
        res = minimize(self._nll, np.zeros(X.shape[2]), args=(X, np.asarray(y), self.l2),
                       jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.beta_ = res.x
        self.converged_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        eta = np.asarray(X, dtype="float64") @ self.beta_
        eta = eta - eta.max(axis=1, keepdims=True)
        e = np.exp(eta)
        return e / e.sum(axis=1, keepdims=True)


# ===========================================================================
# 7. U5: LightGBM over the five
# ===========================================================================
#: The tree arm sees the same alternative features PLUS the raw state variables.
#: A tree can interact a raw state variable with an alternative feature, so --
#: unlike in the logit -- the state does not have to be pre-interacted to be
#: identified.
LGBM_ALT_FEATURES: tuple[str, ...] = (
    "share", "rate", "prior_rate", "exposure_asof", "minutes_asof",
    "minutes_per_game_asof", "games_asof", "position_code", "is_transfer",
    "usage_rank",
)
LGBM_STATE_FEATURES: tuple[str, ...] = ("score_diff", "sec_remaining", "period",
                                        "chance_number")
LGBM_FEATURES: tuple[str, ...] = (*LGBM_ALT_FEATURES, *LGBM_STATE_FEATURES)

LGBM_PARAM_GRID: tuple[dict, ...] = (
    dict(num_leaves=15, learning_rate=0.08, n_estimators=300, min_child_samples=200),
    dict(num_leaves=31, learning_rate=0.06, n_estimators=400, min_child_samples=400),
    dict(num_leaves=63, learning_rate=0.05, n_estimators=500, min_child_samples=800),
    dict(num_leaves=31, learning_rate=0.03, n_estimators=800, min_child_samples=400),
)


def long_frame(d: pd.DataFrame, event_class: str, prior_kind: str,
               m: float) -> tuple[np.ndarray, np.ndarray]:
    """(5n, F) long design for the tree arm and its (5n,) chosen flag. Row order
    is event-major, slot-minor, so `reshape(n, 5)` recovers the choice sets."""
    r = shrunk_rate(d, event_class, prior_kind, m)
    q = normalise(r)
    rank = np.argsort(np.argsort(-q, axis=1, kind="stable"), axis=1).astype("float64")
    blocks = {
        "share": q, "rate": r,
        "prior_rate": _block(d, f"prev_rate_{event_class}"),
        "exposure_asof": _block(d, "exposure_asof"),
        "minutes_asof": _block(d, "minutes_asof"),
        "minutes_per_game_asof": _block(d, "minutes_per_game_asof"),
        "games_asof": _block(d, "games_asof"),
        "position_code": _block(d, "position_code"),
        "is_transfer": _block(d, "is_transfer"),
        "usage_rank": rank,
    }
    n = len(d)
    X = np.empty((n * N_ALT, len(LGBM_FEATURES)), dtype="float32")
    for j, name in enumerate(LGBM_ALT_FEATURES):
        X[:, j] = blocks[name].reshape(-1)
    off = len(LGBM_ALT_FEATURES)
    for j, name in enumerate(LGBM_STATE_FEATURES):
        X[:, off + j] = np.repeat(d[name].to_numpy(dtype="float64"), N_ALT)
    chosen = np.zeros(n * N_ALT, dtype="int8")
    chosen[np.arange(n) * N_ALT + d["y"].to_numpy()] = 1
    return X, chosen


def _group_softmax_objective(y_true: np.ndarray, y_pred: np.ndarray):
    """LightGBM objective: the CONDITIONAL likelihood of the choice among five.

    Rows arrive event-major and slot-minor, so `reshape(-1, 5)` is the choice
    set. The gradient of the grouped softmax log likelihood is `p - y` and its
    (diagonal) Hessian `p (1 - p)` -- the multiclass form, applied within each
    group of five rather than across a fixed class vocabulary.

    This is what U5 has to optimise to be the arm the pre-registration names. A
    plain binary objective softmaxed afterwards optimises a DIFFERENT likelihood
    and comes out systematically over-sharpened: measured on F1 TOV, top decile
    predicted 30.13% against a real 27.89% and bottom decile 11.67% against
    13.16%, a 2.24 pp worst gap that failed the calibration gate while the log
    loss looked good. The miss was an objective mismatch, not a fact about
    trees, and it is fixed here rather than papered over with a temperature --
    a fitted temperature on a model's own output is exactly the shape
    `docs/SIM_GUARDRAILS.md` section 5 bans."""
    z = np.asarray(y_pred, dtype="float64").reshape(-1, N_ALT)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    grad = p - np.asarray(y_true, dtype="float64").reshape(-1, N_ALT)
    hess = np.maximum(p * (1.0 - p), 1e-6)
    return grad.reshape(-1), hess.reshape(-1)


class LgbmChoiceArm:
    """LightGBM scoring each alternative, softmaxed within the choice set.

    The pre-registration calls U5 a "ranker". A lambdarank score has an
    arbitrary scale, so softmaxing one would be badly calibrated by
    construction. This arm instead fits the grouped-softmax likelihood directly
    (`_group_softmax_objective`), which is the conditional-logit link with a
    tree score function, and offsets it by `log q` so that at zero trees it IS
    U1 and what the tree fits is a multiplicative correction to the as-of
    share. The implementation is recorded in `model.md` section 6.

    Row bagging is off: `subsample` would sample ROWS, and a row here is one
    alternative of a choice set, so bagging would split choice sets across the
    in-bag/out-of-bag boundary. Column sampling is unaffected and is kept."""

    BASE = dict(colsample_bytree=0.9, reg_lambda=1.0, verbose=-1,
                boost_from_average=False)

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.params = dict(params or {})
        self.seed = int(seed)

    def fit(self, X: np.ndarray, chosen: np.ndarray,
            init: np.ndarray | None = None) -> LgbmChoiceArm:
        import lightgbm as lgb
        self.clf_ = lgb.LGBMRegressor(random_state=self.seed,
                                      objective=_group_softmax_objective,
                                      **{**self.BASE, **self.params})
        kw = {} if init is None else {"init_score": np.asarray(init, dtype="float64")}
        self.clf_.fit(X, np.asarray(chosen, dtype="float64"), **kw)
        return self

    def predict_proba(self, X: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
        raw = np.asarray(self.clf_.predict(X, raw_score=True), dtype="float64")
        if init is not None:
            raw = raw + np.asarray(init, dtype="float64")
        raw = raw.reshape(-1, N_ALT)
        raw = raw - raw.max(axis=1, keepdims=True)
        e = np.exp(raw)
        return e / e.sum(axis=1, keepdims=True)


def lgbm_init_score(d: pd.DataFrame, event_class: str, prior_kind: str,
                    m: float) -> np.ndarray:
    """(5n,) offset `log(U1 share)` for the tree arm.

    With this offset the arm's raw score is `log q_i + f(x_i)`, so its softmax is
    `q_i e^{f_i} / sum_j q_j e^{f_j}`: at zero trees it is EXACTLY U1, and what
    the tree fits is a multiplicative correction to the as-of share. Without the
    offset the tree has to relearn the share from scratch and the comparison
    stops being "does the tree add anything to U1" and becomes "can a tree
    reproduce a normalisation", which is not the pre-registered question."""
    q = normalise(shrunk_rate(d, event_class, prior_kind, m))
    return np.log(np.maximum(q, 1e-9)).reshape(-1)


# ===========================================================================
# 8. Folds
# ===========================================================================
def fold_slices(design: pd.DataFrame, fold: str = SELECTION_FOLD
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"{fold} train seasons")
    assert_not_sealed(spec["test"], context=f"{fold} test seasons")
    tr = design[design["season"].isin(spec["train"])]
    te = design[design["season"].isin(spec["test"])]
    assert_not_sealed(tr, context=f"{fold} train slice")
    assert_not_sealed(te, context=f"{fold} test slice")
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def walkforward_slices(design: pd.DataFrame, season: int = WF_SEASON,
                       split_date: str = WF_SPLIT_DATE
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The within-season robustness fold of the pre-registration."""
    assert_not_sealed([season], context="within-season walk-forward")
    d = design[design["season"] == season]
    assert_not_sealed(d, context="walk-forward slice")
    cut = pd.Timestamp(split_date)
    dt = pd.to_datetime(d["game_date"])
    return d[dt < cut].reset_index(drop=True), d[dt >= cut].reset_index(drop=True)


# ===========================================================================
# 9. Metrics
# ===========================================================================
def _alt_long(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray):
    """Flatten the choice sets to (event x alternative) rows: the axis the
    pre-registration defines the calibration and responsiveness checks on (the
    PLAYER's as-of rate, not the event)."""
    y = te["y"].to_numpy()
    obs = np.zeros_like(p, dtype="int64")
    obs[np.arange(len(y)), y] = 1
    return obs.reshape(-1), p.reshape(-1), np.asarray(driver).reshape(-1)


def share_calibration(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray,
                      n_bins: int = 10) -> dict:
    """Predicted share vs actual share inside deciles of the player's own as-of
    rate, in percentage points."""
    obs, pred, dr = _alt_long(te, p, driver)
    edges = np.quantile(dr, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, dr, side="right") - 1, 0, n_bins - 1)
    rows, worst = [], 0.0
    for b in range(n_bins):
        msk = idx == b
        if not msk.sum():
            continue
        pp = float(pred[msk].mean() * 100)
        aa = float(obs[msk].mean() * 100)
        rows.append({"bin": b + 1, "n": int(msk.sum()),
                     "driver_mean": round(float(dr[msk].mean()), 6),
                     "pred_pct": round(pp, 4), "actual_pct": round(aa, 4),
                     "gap_pp": round(pp - aa, 4)})
        worst = max(worst, abs(pp - aa))
    return {"bins": rows, "max_abs_gap_pp": round(worst, 4),
            "pass": bool(worst <= CALIB_GATE_PP)}


def share_responsiveness(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray,
                         n_q: int = 5) -> dict:
    obs, pred, dr = _alt_long(te, p, driver)
    out = PM.quintile_responsiveness(dr, obs, np.column_stack([1 - pred, pred]), 1,
                                     n_q=n_q)
    out["pass"] = bool(out["pred_monotone_steps"] >= RESP_MIN_STEPS)
    return out


def top_k_accuracy(te: pd.DataFrame, p: np.ndarray, k: int) -> float:
    y = te["y"].to_numpy()
    order = np.argsort(-p, axis=1, kind="stable")
    return float((order[:, :k] == y[:, None]).any(axis=1).mean())


def score_arm(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray) -> dict:
    y = te["y"].to_numpy()
    onehot = np.eye(N_ALT)[y]
    cal = share_calibration(te, p, driver)
    resp = share_responsiveness(te, p, driver)
    return {
        "n": int(len(te)),
        "log_loss": round(PM.log_loss(y, p), 6),
        "brier": round(float(((p - onehot) ** 2).sum(axis=1).mean()), 6),
        "top1": round(top_k_accuracy(te, p, 1), 6),
        "top3": round(top_k_accuracy(te, p, 3), 6),
        "calibration": cal,
        "calib_pass": cal["pass"],
        "calib_worst_gap_pp": cal["max_abs_gap_pp"],
        "responsiveness": resp,
        "resp_pass": resp["pass"],
        "resp_steps": int(resp["pred_monotone_steps"]),
    }


def bootstrap_se(te: pd.DataFrame, p: np.ndarray, n_rep: int = 200,
                 seed: int = 12345) -> float:
    """Game-level block bootstrap SE of the 5-way log loss -- the noise floor
    for the non-tree arms (`prob_metrics.block_bootstrap_se`)."""
    return PM.block_bootstrap_se(te["game_id"].to_numpy(), te["y"].to_numpy(), p,
                                 n_rep=n_rep, seed=seed)


# ===========================================================================
# 10. The game-level "too narrow / too short" pair
# ===========================================================================
@dataclass
class AllocCells:
    """The (team-game, player) cell index the game-level checks accumulate into.

    Every player who was ON THE FLOOR for at least one of a team-game's events
    gets a cell, so a player the arm never credits contributes an explicit ZERO
    to his own count SD instead of dropping out of it -- which is the difference
    between measuring "too narrow" and measuring nothing."""
    cell_of: np.ndarray          # (n, 5) cell index per alternative
    cell_tg: np.ndarray          # (n_cells,) team-game of each cell
    cell_player: np.ndarray      # (n_cells,) player code of each cell
    cell_player_id: np.ndarray   # (n_cells,) CBBD id of each cell's player
    n_tg: int
    n_players: int
    player_games: np.ndarray     # (n_players,) team-games he was on the floor in
    player_id: np.ndarray        # (n_players,) CBBD id of each player code


def alloc_cells(d: pd.DataFrame) -> AllocCells:
    gid = d["game_id"].to_numpy(dtype="int64")
    tid = d["team_id"].to_numpy(dtype="int64")
    tg, _ = pd.factorize(pd.MultiIndex.from_arrays([gid, tid]), sort=False)
    alts = d[[f"alt_{k}" for k in range(1, N_ALT + 1)]].to_numpy(dtype="int64")
    flat_tg = np.repeat(tg, N_ALT)
    flat_pl = alts.reshape(-1)
    cell, uniq = pd.factorize(pd.MultiIndex.from_arrays([flat_tg, flat_pl]), sort=False)
    cell_tg = np.asarray(uniq.get_level_values(0), dtype="int64")
    cell_pl_id = np.asarray(uniq.get_level_values(1), dtype="int64")
    pcode, puniq = pd.factorize(cell_pl_id, sort=False)
    n_players = int(pcode.max()) + 1 if len(pcode) else 0
    return AllocCells(cell_of=cell.reshape(-1, N_ALT), cell_tg=cell_tg,
                      cell_player=pcode, cell_player_id=cell_pl_id,
                      n_tg=int(tg.max()) + 1 if len(tg) else 0,
                      n_players=n_players,
                      player_games=np.bincount(pcode, minlength=n_players),
                      player_id=np.asarray(puniq, dtype="int64"))


#: Room for a team-game's events inside one game's key space. A team-game has
#: at most a few dozen events of one class; 4096 is two orders of magnitude of
#: headroom and keeps `game_id * EVENT_KEY_STRIDE + ordinal` injective in
#: uint64 for every ESPN game id.
EVENT_KEY_STRIDE = 4096


def event_stream_keys(d: pd.DataFrame, seed: int) -> np.ndarray:
    """One counter-based key per EVENT, off the (seed, game_id, family) stream.

    Every event needs its own stream position: keying only on the game would
    hand every event of a game the SAME uniform, which collapses a team-game's
    whole allocation onto one player (measured before the fix: 3.42 players with
    a three-point attempt per team-game against a real 6.69). The event's
    ordinal within its game is folded into the game id, which keeps the
    per-game independence the RNG rule exists to give while making the draws
    within a game independent of each other."""
    gid = d["game_id"].to_numpy(dtype="int64")
    ordinal = d.groupby("game_id", sort=False).cumcount().to_numpy()
    if ordinal.max(initial=0) >= EVENT_KEY_STRIDE:
        raise ValueError("more events in one game than EVENT_KEY_STRIDE allows")
    return RNG.stream_keys(seed, (gid.astype("uint64") * np.uint64(EVENT_KEY_STRIDE)
                                  + ordinal.astype("uint64")), "usage_alloc")


def _cell_counts(cells: AllocCells, pick: np.ndarray) -> np.ndarray:
    n = len(pick)
    return np.bincount(cells.cell_of[np.arange(n), pick],
                       minlength=len(cells.cell_tg)).astype("float64")


def _tg_stats(cells: AllocCells, cnt: np.ndarray) -> tuple[float, float, float]:
    """(players with >=1 event per team-game, top-1 share, top-3 share)."""
    order = np.lexsort((-cnt, cells.cell_tg))
    tg_sorted = cells.cell_tg[order]
    starts = np.flatnonzero(np.concatenate([[True], tg_sorted[1:] != tg_sorted[:-1]]))
    pos = np.arange(len(order)) - np.repeat(starts, np.diff(np.append(starts, len(order))))
    c_sorted = cnt[order]
    tot = np.bincount(tg_sorted, weights=c_sorted, minlength=cells.n_tg)
    tot = np.maximum(tot, 1.0)
    gt0 = np.bincount(cells.cell_tg, weights=(cnt > 0).astype("float64"),
                      minlength=cells.n_tg)
    top1 = np.bincount(tg_sorted, weights=np.where(pos == 0, c_sorted, 0.0),
                       minlength=cells.n_tg) / tot
    top3 = np.bincount(tg_sorted, weights=np.where(pos < 3, c_sorted, 0.0),
                       minlength=cells.n_tg) / tot
    live = np.bincount(cells.cell_tg, minlength=cells.n_tg) > 0
    return float(gt0[live].mean()), float(top1[live].mean()), float(top3[live].mean())


def _player_sd(cells: AllocCells, cnt: np.ndarray, min_games: int) -> np.ndarray:
    """Per player, the SD across his team-games of his per-game credited count.
    CFB's "too narrow" statistic, computed on the same definition."""
    s1 = np.bincount(cells.cell_player, weights=cnt, minlength=cells.n_players)
    s2 = np.bincount(cells.cell_player, weights=cnt ** 2, minlength=cells.n_players)
    ng = cells.player_games.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 / ng - (s1 / ng) ** 2) * np.where(ng > 1, ng / (ng - 1), np.nan)
    sd = np.sqrt(np.maximum(var, 0.0))
    sd[cells.player_games < min_games] = np.nan
    return sd


def profile_player_roles(ps: ProfileSet) -> pd.Series:
    """CBBD player id -> his modal role over the profiles he appears in. Used to
    report the "too narrow" statistic PER ROLE, which is the moment each of U3's
    within-role concentrations is fitted against."""
    t = pd.DataFrame({"p": ps.players, "role": ps.roles})
    return t.groupby("p")["role"].agg(lambda x: int(x.value_counts().index[0]))


def game_level_check(d: pd.DataFrame, p: np.ndarray | None = None,
                     ps: ProfileSet | None = None, alphas: dict | None = None,
                     n_draw: int = 40, seed: int = 0, min_games: int = 8,
                     player_role: pd.Series | None = None) -> dict:
    """The CFB "too narrow / too short" pair plus the top-1/top-3 share check.

    Re-allocates the ACTUAL event sequence with the ACTUAL fives `n_draw` times,
    which is the pre-registration's isolation clause: the number of events per
    team-game and the five on the floor at each of them come from reality, so
    what is measured is allocation alone -- not rotation and not the event
    model.

    Point arms pass `p`. A Dirichlet arm passes `ps` and `alphas` instead, so
    that each DRAW re-realises the game's share vector and the per-game usage
    variance it buys actually appears in the output."""
    cells = alloc_cells(d)
    n = len(d)
    y = d["y"].to_numpy()
    keys = event_stream_keys(d, seed)

    act = _cell_counts(cells, y)
    gt0_a, t1_a, t3_a = _tg_stats(cells, act)
    sd_a = _player_sd(cells, act, min_games)

    S = realize_batch(ps, alphas, n_draw, seed) if ps is not None else None
    sd_acc = np.zeros(cells.n_players)
    sd_n = np.zeros(cells.n_players)
    gt0_s, t1_s, t3_s = [], [], []
    for r in range(n_draw):
        if S is not None:
            pr = np.empty((n, N_ALT))
            for lo in range(0, n, 20000):
                hi = min(lo + 20000, n)
                pr[lo:hi] = _conditional(S[r:r + 1], ps.gslot[lo:hi])[0]
            pr = normalise(pr)
        else:
            pr = p
        u = RNG.uniforms(keys, r)
        cum = np.cumsum(pr, axis=1)
        cum = cum / cum[:, [-1]]
        pick = np.clip((u[:, None] > cum).sum(axis=1), 0, N_ALT - 1)
        cnt = _cell_counts(cells, pick)
        g0, a1, a3 = _tg_stats(cells, cnt)
        gt0_s.append(g0)
        t1_s.append(a1)
        t3_s.append(a3)
        sd = _player_sd(cells, cnt, min_games)
        ok = np.isfinite(sd)
        sd_acc[ok] += sd[ok]
        sd_n[ok] += 1

    with np.errstate(invalid="ignore"):
        sd_sim = np.where(sd_n > 0, sd_acc / np.maximum(sd_n, 1), np.nan)
    both = np.isfinite(sd_sim) & np.isfinite(sd_a)
    sd_ratio = float(sd_sim[both].mean() / max(sd_a[both].mean(), EPS)) if both.any() else np.nan
    gt0_sim = float(np.mean(gt0_s))
    t1_sim, t3_sim = float(np.mean(t1_s)), float(np.mean(t3_s))
    by_role = {}
    if player_role is not None and both.any():
        roles = player_role.reindex(cells.player_id).to_numpy(dtype="float64")
        for r, name in enumerate(ROLES):
            msk = both & (roles == r)
            if msk.sum() >= 25:
                by_role[name] = {
                    "n_players": int(msk.sum()),
                    "sd_sim": round(float(sd_sim[msk].mean()), 5),
                    "sd_actual": round(float(sd_a[msk].mean()), 5),
                    "sd_ratio": round(float(sd_sim[msk].mean()
                                            / max(sd_a[msk].mean(), EPS)), 5)}
    return {
        "sd_ratio_by_role": by_role,
        "n_team_games": cells.n_tg, "n_draw": n_draw,
        "n_players_graded": int(both.sum()),
        "sd_sim": round(float(sd_sim[both].mean()), 5) if both.any() else None,
        "sd_actual": round(float(sd_a[both].mean()), 5) if both.any() else None,
        "sd_ratio": round(sd_ratio, 5),
        "sd_pass": bool(SD_RATIO_BAND[0] <= sd_ratio <= SD_RATIO_BAND[1]),
        "players_gt0_sim": round(gt0_sim, 5),
        "players_gt0_actual": round(gt0_a, 5),
        "players_gt0_delta": round(gt0_sim - gt0_a, 5),
        "gt0_pass": bool(abs(gt0_sim - gt0_a) <= GT0_TOL),
        "top1_sim_pct": round(t1_sim * 100, 4), "top1_actual_pct": round(t1_a * 100, 4),
        "top1_gap_pp": round((t1_sim - t1_a) * 100, 4),
        "top3_sim_pct": round(t3_sim * 100, 4), "top3_actual_pct": round(t3_a * 100, 4),
        "top3_gap_pp": round((t3_sim - t3_a) * 100, 4),
        "top_pass": bool(abs(t1_sim - t1_a) * 100 <= TOPSHARE_TOL_PP
                         and abs(t3_sim - t3_a) * 100 <= TOPSHARE_TOL_PP),
    }


# ===========================================================================
# 11. The sampler
# ===========================================================================
@dataclass
class UsageState:
    """One game's realised usage, plus its position in the RNG stream.

    `rates[event_class]` maps a CBBD player id to his realised relative rate for
    that class in THIS game. The realisation is drawn once per game by
    `new_game_state` off the (seed, game_id, 'usage') stream, so usage
    persistence across substitutions is a property of the state and not of the
    per-event draw."""
    game_id: int
    seed: int
    rates: dict[str, dict[int, float]]
    counter: int = 0
    key: np.ndarray | None = None

    def next_uniform(self) -> float:
        u = float(RNG.uniforms(self.key, self.counter)[0])
        self.counter += 1
        return u


def new_game_state(game_id: int, seed: int,
                   profiles: dict[str, Profile] | None = None,
                   rates: dict[str, dict[int, float]] | None = None,
                   alphas: dict | None = None,
                   counter_start: int = 0) -> UsageState:
    """Open one game's usage stream.

    `profiles` (class -> `Profile`) draws the hierarchical realisation through
    the exact inverse-CDF path; passing `rates` directly instead is the
    deterministic U1 path. `alphas=None` with a profile means "no dispersion",
    i.e. the realised rates are the as-of weights."""
    key = RNG.stream_keys(seed, np.array([int(game_id)], dtype="uint64"), "usage")
    realised: dict[str, dict[int, float]] = {k: dict(v) for k, v in (rates or {}).items()}
    ctr = int(counter_start)
    for cls in sorted(profiles or {}):
        pr = profiles[cls]
        K = len(pr.weights)
        if alphas is None:
            w = pr.weights
        else:
            u = np.stack([RNG.uniforms(key, ctr + j) for j in range(K + N_ROLE)], axis=1)
            ctr += K + N_ROLE
            w = realize_shares(pr.weights, pr.roles, alphas, u)[0]
        realised[cls] = {int(pl): float(x) for pl, x in zip(pr.players, w, strict=False)}
    return UsageState(game_id=int(game_id), seed=int(seed), rates=realised,
                      counter=ctr, key=key)


def draw_player(five, event_class: str, state: UsageState) -> int:
    """Credit one `event_class` event to one of `five` (CBBD player ids).

    The probability of player i is his realised rate over the realised rates of
    the five on the floor. A lineup in which nobody carries any rate for the
    class falls back to the uniform over the five, which is the only honest
    answer when the model has no information -- never a silent pick of the first
    id. One uniform of the (seed, game_id, 'usage') stream is consumed per call,
    so a game's draws are reproducible and independent of every other game in
    the run."""
    ids = [int(x) for x in five]
    if len(ids) != N_ALT:
        raise ValueError(f"expected {N_ALT} players on the floor, got {len(ids)}")
    table = state.rates.get(event_class, {})
    w = np.array([max(float(table.get(i, 0.0)), 0.0) for i in ids])
    tot = w.sum()
    w = np.full(N_ALT, 1.0 / N_ALT) if (tot <= 0 or not np.isfinite(tot)) else w / tot
    u = state.next_uniform()
    return ids[int(np.clip((u > np.cumsum(w)).sum(), 0, N_ALT - 1))]


__all__ = [
    "ALPHA_GRID", "ARMS", "ARM_ORDER", "CL_FEATURES", "DIRECT_CLASSES",
    "DIRICHLET_ARMS", "EVENT_CLASSES", "FOLDS", "GT0_TOL", "LGBM_FEATURES",
    "LGBM_PARAM_GRID", "NO_DISPERSION", "N_ALT", "POSITION_LEVELS",
    "PRIOR_KINDS", "RESP_MIN_STEPS", "ROLES", "SD_RATIO_BAND", "SELECTION_FOLD",
    "SHRINK_GRID", "TOPSHARE_TOL_PP", "TREE_ARM", "WF_SPLIT_DATE", "AllocCells",
    "CondLogitArm", "LgbmChoiceArm", "Profile", "ProfileSet", "UsageState",
    "alloc_cells", "assign_roles", "event_stream_keys", "bootstrap_se", "build_player_asof",
    "build_profiles", "build_usage_design", "build_usage_events", "cl_design",
    "coverage_report", "draw_player", "drop_constant_features", "fit_shrinkage",
    "fold_slices", "game_level_check", "hier_alphas", "profile_player_roles", "load_minutes",
    "lgbm_init_score", "long_frame", "marginal_probs", "new_game_state", "normalise",
    "prior_season_available", "PRIOR_SEASON_FEATURES", "unidentified_features", "realize_batch", "realize_shares", "score_arm",
    "sequential_probs", "share_calibration", "share_responsiveness",
    "shrunk_rate", "single_level_alphas", "top_k_accuracy", "u1_probs",
    "usable_events", "walkforward_slices",
]
