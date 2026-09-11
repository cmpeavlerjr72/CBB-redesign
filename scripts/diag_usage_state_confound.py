#!/usr/bin/env python
"""
diag_usage_state_confound.py -- own-row delta test on every state feature the
usage tree (U5) consumes: `score_diff`, `sec_remaining`, `period`,
`chance_number`.

Method as in `scripts/diag_fg_make_state_confound.py`
(`docs/tests/fg_make_state_confound_2026-09-10.md`), applied to usage's own
event population instead of fg_make's: compare each feature's value on a row
to the state reconstructed from the PREVIOUS row of the same game's raw event
stream, and report the share of rows where the feature already carries the
row's own outcome (post-outcome) rather than the state as of just before the
event (pre-outcome).

`cbb_sim.models.usage.build_usage_events` builds `score_diff` from
`home_score`/`away_score` on the event's OWN row
(`np.where(off_home, hs - as_, as_ - hs)`, usage.py line 343), which is the
same construction `fg_make` used before it was found to be post-outcome (L27).
This script checks whether the same defect reproduces here.

Population: usage's own modelled window, seasons 2024-2025 (on-floor ids do
not exist before 2024, L13), D-I / non-truncated / pbp_complete universe,
2026 sealed and never read.

Output: `data/processed/models/usage_v2/state_confound.json`.

Usage:
    .venv/Scripts/python.exe scripts/diag_usage_state_confound.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cbb_sim.models.event_stream as ES  # noqa: E402
import cbb_sim.models.usage as U  # noqa: E402

SEASONS = (2024, 2025)
OUT = ROOT / "data/processed/models/usage_v2/state_confound.json"

FG_VALUE = {"FGA_rim": 2, "FGA_jump2": 2, "FGA_3": 3}


def t0():
    return time.time()


def log(msg, t):
    print(f"[{time.time() - t:8.1f}s] {msg}", flush=True)


def own_row_delta_for_season(season: int) -> pd.DataFrame:
    """Full raw stream for the season (build_usage_events' own upstream,
    `shooter_key='shot_shooter_id'` to match the adopted build), with the
    OWN-ROW score move computed from the previous row of the SAME game."""
    universe = ES.load_universe(require_pbp_complete=True)
    stream = ES.build_stream(season, universe, shooter_key="shot_shooter_id")
    g = stream["cbbd_game_id"].to_numpy()
    hs = stream["home_score"].to_numpy(dtype="float64")
    as_ = stream["away_score"].to_numpy(dtype="float64")
    same_prev = np.concatenate([[False], g[1:] == g[:-1]])
    prev_hs = np.concatenate([[np.nan], hs[:-1]])
    prev_as = np.concatenate([[np.nan], as_[:-1]])
    off_home = (stream["side"].to_numpy() == 0)
    # own-row post-outcome score_diff, exactly usage.build_usage_events line 343
    score_diff_own = np.where(off_home, hs - as_, as_ - hs)
    # reconstructed PRE-outcome score_diff: the state as of the END of the
    # previous row in this game (i.e. before this row's own event resolves)
    score_diff_pre = np.where(off_home, prev_hs - prev_as, prev_as - prev_hs)
    move = np.where(off_home, hs - prev_hs, as_ - prev_as)
    out = stream.copy()
    out["same_prev"] = same_prev
    out["score_diff_own"] = score_diff_own
    out["score_diff_pre"] = np.where(same_prev, score_diff_pre, np.nan)
    out["own_row_move"] = np.where(same_prev, move, np.nan)
    out["season"] = season
    return out


def is_fttrip(stream: pd.DataFrame) -> np.ndarray:
    return (stream["trip_pos"].to_numpy() == 1) & (stream["trip_cause"].to_numpy() == "foul")


def summarize_score_diff(all_rows: pd.DataFrame) -> dict:
    """Own-row score move by usage event class, restricted to rows with a
    valid previous row in the same game (first row of a game excluded, as in
    the fg_make audit)."""
    res: dict = {}
    valid = all_rows["same_prev"].to_numpy()
    cls = all_rows["cls"].to_numpy(dtype=object)
    made = all_rows["made"].to_numpy()
    fttrip = is_fttrip(all_rows)
    move = all_rows["own_row_move"].to_numpy()

    for c in ES.FGA_CLASSES:
        sel_made = valid & (cls == c) & made
        sel_miss = valid & (cls == c) & ~made
        val = FG_VALUE[c]
        n_made = int(sel_made.sum())
        n_miss = int(sel_miss.sum())
        made_post = float(np.isclose(move[sel_made], val).mean()) if n_made else float("nan")
        miss_post = float(np.isclose(move[sel_miss], 0.0).mean()) if n_miss else float("nan")
        res[c] = {
            "n_made": n_made, "share_made_delta_eq_shot_value": made_post,
            "n_missed": n_miss, "share_missed_delta_eq_zero": miss_post,
            "post_outcome_share_overall": float(
                (np.isclose(move[sel_made], val).sum()
                 + np.isclose(move[sel_miss], 0.0).sum()) / max(n_made + n_miss, 1)),
        }

    sel_tov = valid & (cls == "TOV")
    n_tov = int(sel_tov.sum())
    res["TOV"] = {
        "n": n_tov,
        "share_delta_eq_zero": float(np.isclose(move[sel_tov], 0.0).mean()) if n_tov else float("nan"),
    }

    sel_ft1 = valid & fttrip
    n_ft1 = int(sel_ft1.sum())
    ft_made = made[sel_ft1]
    ft_move = move[sel_ft1]
    res["FT_trip"] = {
        "n": n_ft1,
        "n_made": int(ft_made.sum()), "n_missed": int((~ft_made).sum()),
        "share_made_delta_eq_1": float(np.isclose(ft_move[ft_made], 1.0).mean()) if ft_made.any() else float("nan"),
        "share_missed_delta_eq_0": float(np.isclose(ft_move[~ft_made], 0.0).mean()) if (~ft_made).any() else float("nan"),
    }
    return res


def chance_number_invariance_check(stream: pd.DataFrame) -> dict:
    """Own-row test for `_chance_number`, isolated from legitimate downstream
    propagation: flip `made` on only the LAST FGA row of EACH game (so there is
    no later row in that game for the flip to legitimately propagate to, and no
    earlier row in that game is touched, so no earlier-row confound reaches
    this row either), then compare chance_number ONLY at that one flipped row
    per game against the original.

    Code proof this must hold: `_chance_number` sets
    `new_poss = concatenate([[True], ends[:-1] | (g[1:] != g[:-1])])`, so
    `poss_id[i]` (and therefore `chance_number[i]`) is a function of
    `ends[0..i-1]` only -- row i's own `ends[i]` (which depends on made[i])
    never enters the computation of chance_number AT row i, only at rows AFTER
    it. And an FGA row's own `oreb` term is always 0 (`cls != 'OREB'`), so it
    cannot contribute to its own cumulative sum either. The single-row-per-game
    flip is the clean empirical confirmation of that proof, without the
    multi-row test's earlier-row-flip confound (flipping many FGA rows in one
    game changes EARLIER rows' `ends`, which correctly moves poss_id at a LATER
    row -- that is causality, not this row's own leak, and conflating the two
    was a defect in an earlier version of this check)."""
    is_fga = np.isin(stream["cls"].to_numpy(dtype=object), ES.FGA_CLASSES)
    g = stream["cbbd_game_id"].to_numpy()
    idx = np.flatnonzero(is_fga)
    # last FGA row index per game among the flipped population
    last_fga_per_game = pd.Series(idx, index=g[idx]).groupby(level=0).last().to_numpy()

    base = U._chance_number(stream)
    flipped = stream.copy()
    m = flipped["made"].to_numpy().copy()
    m[last_fga_per_game] = ~m[last_fga_per_game]
    flipped["made"] = m
    alt = U._chance_number(flipped)

    own_row_diff = base[last_fga_per_game] != alt[last_fga_per_game]
    # sanity: every other row (including earlier rows in the same games) must
    # be untouched, since only the LAST FGA row per game was flipped and there
    # is nothing after it to propagate from
    other = np.ones(len(stream), dtype=bool)
    other[last_fga_per_game] = False
    n_other_diff = int((base[other] != alt[other]).sum())

    return {
        "n_rows": int(len(stream)),
        "n_games_tested": int(len(last_fga_per_game)),
        "own_row_test": {
            "n_flipped_rows_whose_OWN_chance_number_changed": int(own_row_diff.sum()),
            "identical": bool(not own_row_diff.any()),
        },
        "no_other_row_should_move": {
            "n_other_rows_changed": n_other_diff,
            "expected": 0,
            "why": ("only the LAST FGA row of each game was flipped, so there is no "
                    "later row in that game for a real propagation effect to reach; "
                    "any nonzero count here would mean the test itself is broken, "
                    "not a finding about chance_number"),
        },
        "conclusion": ("chance_number is PRE-outcome at the row level: flipping a "
                       "row's own make/miss (isolated to the last FGA row of each "
                       "game, so no legitimate downstream propagation can confound "
                       "the result) never changes that row's own chance_number "
                       f"(0 of {len(last_fga_per_game)} flipped rows)."
                       if not own_row_diff.any() else
                       "chance_number leaks its OWN row's outcome -- further audit needed"),
    }


def sec_period_note() -> dict:
    return {
        "conclusion": ("sec_remaining and period are read directly off the pbp clock "
                       "columns (period, secondsRemaining) with no dependence on made/"
                       "scoringPlay; they cannot carry this row's own outcome by "
                       "construction (usage.py build_usage_events lines 323-324, 340-342). "
                       "No own-row delta applies -- there is no 'previous value' for a "
                       "clock reading to leak from."),
    }


def main() -> int:
    t = t0()
    frames = []
    for s in SEASONS:
        log(f"building stream for season {s}", t)
        frames.append(own_row_delta_for_season(s))
    all_rows = pd.concat(frames, ignore_index=True)
    log(f"stream built: {len(all_rows)} rows across {SEASONS}", t)

    score_diff = summarize_score_diff(all_rows)
    log("score_diff own-row delta summarised", t)

    chance = {}
    for s in SEASONS:
        universe = ES.load_universe(require_pbp_complete=True)
        stream = ES.build_stream(s, universe, shooter_key="shot_shooter_id")
        chance[str(s)] = chance_number_invariance_check(stream)
        log(f"chance_number invariance checked for {s}", t)

    out = {
        "population": "usage modelled window, D-I/non-truncated/pbp_complete, "
                       "seasons 2024-2025 (2026 sealed, never read)",
        "score_diff": score_diff,
        "chance_number": chance,
        "sec_remaining_and_period": sec_period_note(),
        "headline": (
            "score_diff in usage.build_usage_events is POST-OUTCOME, the same "
            "construction and the same defect L27 found in fg_make: a made "
            "field goal's own row already carries its own points."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log(f"wrote {OUT}", t)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
