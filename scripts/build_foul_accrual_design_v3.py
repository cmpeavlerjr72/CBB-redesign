#!/usr/bin/env python
"""
build_foul_accrual_design_v3.py -- v2 PLUS the fouls implied by ROWLESS FT trips (round 8, experiments.md s22).

v3 CHANGE: a non-technical FT trip reached with no immediately preceding foul row, and with no foul by
the other team logged at the same (period, clock), never increments the pbp team-foul counter. v3 logs
each such trip (fouling team = the non-shooting side, possession = the trip possession) and adds
`def_rowless_prior` / `off_rowless_prior`: rowless trips charged to that side in EARLIER possessions of
the same half-segment (period 1; periods 2+ one segment). `def_team_fouls_v3 = def_team_fouls_true +
def_rowless_prior`. Also writes the per-trip log.

v2 DOCSTRING:

build_foul_accrual_design_v2.py -- v1 PLUS the pre-open foul counts (joint foul round, experiments.md s20).

v2 CHANGE (only): `def_pre_open` / `off_pre_open` count the fouls charged to each side
while NO possession was open and attributed to this possession. `_GameMachine._open`
reads `team_fouls` AFTER those fouls, so `def_team_fouls` at open INCLUDES them; the
engine's state is the count BEFORE the possession's own fouls. `def_team_fouls_true =
def_team_fouls - def_pre_open` is the engine-definition state. Everything else is v1.

ORIGINAL v1 DOCSTRING:

Pre-registration: `docs/models/possession_outcome/experiments.md` section 13
(round 6) as amended by section 15 (AMENDMENT A, committed before any fitting).

WHY A REPLAY AND NOT A DIFFERENCE OF `def_team_fouls`. Section 15.1 point 2:
`off_team_fouls`/`def_team_fouls` are read at possession OPEN, and
`_handle_ft_trip -> _ensure` opens the possession AT the foul whenever the
previous possession has already closed. Differencing the stored counts books
most defensive trip fouls into the previous window -- a window in which the
fouling team is the OFFENCE -- and yields a negative silent rate. This script
therefore re-runs `cbb_sim.pbp.possessions._GameMachine` and wraps
`_handle_foul` / `_handle_technical` IN THIS PROCESS ONLY (the
`diag_late_game_tap_v1.py` pattern). `src/cbb_sim/` is not edited.

For every personal foul it records the possession that was OPEN when the foul
happened, the side charged, and whether the foul awarded a free-throw trip.
Fouls with no possession open are attributed to the next possession opened and
counted separately (`n_no_open`).

    .venv/Scripts/python.exe scripts/build_foul_accrual_design_v1.py

Writes `data/processed/models/possession_outcome/round6/foul_accrual_poss_v1.parquet`
(one row per possession, seasons 2022-2025) and `.../build_report_v3.json`.
Season 2026 is SEALED and is not built here.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_DIR = Path("data/processed/models/possession_outcome/round6")
SEASONS = [2022, 2023, 2024, 2025]
_PIN = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _season(season: int) -> tuple[pd.DataFrame, dict]:
    for k in _PIN:
        os.environ[k] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cbb_sim.pbp.events import load_plays
    from cbb_sim.pbp.possessions import _GameMachine, _prepare_events

    t0 = time.time()
    u = pd.read_parquet("data/processed/games_universe.parquet")
    u = u[u["is_d1_game"] & ~u["pbp_truncated"] & (u["season"] == season)]
    meta = {int(r.cbbd_game_id): {"game_id": int(r.game_id), "cbbd_game_id": int(r.cbbd_game_id),
                                  "season": int(r.season), "home_team_id": int(r.home_team_id),
                                  "away_team_id": int(r.away_team_id)}
            for r in u.itertuples()}
    plays = load_plays(season, pbp_dir="data/raw/cbbd/pbp", game_ids=set(meta))
    ev = _prepare_events(plays, with_on_floor=False)

    orig_foul = _GameMachine._handle_foul
    orig_tech = _GameMachine._handle_technical
    orig_trip = _GameMachine._handle_ft_trip

    def patched_trip(self, i, t, prior_fouls, technical):
        rowless = False
        if (not technical) and prior_fouls is None:
            per, sec = int(self.ev["period"][i]), int(self.ev["sec"][i])
            fouler = 1 - int(t)
            rowless = not any((f[2] == fouler and f[4] == per and f[5] == sec)
                              for f in self._foul_log[-12:])
        out = orig_trip(self, i, t, prior_fouls, technical)
        if rowless:
            if self.cur is not None and self.cur.offense_team_id == int(t):
                pi = self.cur.poss_index
            elif self.possessions:
                pi = self.possessions[-1].poss_index
            else:
                pi = None
            self._rowless_log.append((pi, 1 - int(t), int(self.ev["period"][i]),
                                      int(self.ev["sec"][i])))
        return out

    def patched_foul(self, i, t):
        # The possession OPEN at the moment of the foul, if any. If none is
        # open (a dead-ball foul between possessions, and the common case for a
        # bonus foul after a made basket) the foul is attributed to the NEXT
        # possession the machine opens -- `self.poss_index` is the running
        # game-level counter `_open` increments, so that index is `+1`.
        pi = self.cur.poss_index if self.cur is not None else None
        nxt = int(self.poss_index) + 1
        j = i + 1
        leads = bool(j < self.n and self.ev["cls"][j] in ("FT_made", "FT_missed"))
        out = orig_foul(self, i, t)
        self._foul_log.append((pi, nxt, int(t), leads, int(self.ev["period"][i]),
                               int(self.ev["sec"][i])))
        return out

    def patched_tech(self, i, t):
        self._tech_log.append((int(self.ev["period"][i]), int(self.ev["sec"][i]), int(t)))
        return orig_tech(self, i, t)

    _GameMachine._handle_foul = patched_foul
    _GameMachine._handle_technical = patched_tech
    _GameMachine._handle_ft_trip = patched_trip
    rowless_rows: list[dict] = []

    games = ev["game"]
    bounds = np.flatnonzero(np.concatenate([[True], games[1:] != games[:-1], [True]]))
    rows: list[dict] = []
    n_no_open = 0
    n_no_open_unresolved = 0
    n_games = 0
    for b in range(len(bounds) - 1):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        gid = int(games[lo])
        gm = meta.get(gid)
        if gm is None:
            continue
        sub = {"cls": ev["cls"][lo:hi], "team": ev["team"][lo:hi], "sec": ev["sec"][lo:hi],
               "period": ev["period"][lo:hi], "hs": ev["hs"][lo:hi], "as_": ev["as_"][lo:hi],
               "made": ev["made"][lo:hi], "stolen": ev["stolen"][lo:hi], "on_floor": None}
        m = _GameMachine(gm, sub)
        m._foul_log = []
        m._tech_log = []
        m._rowless_log = []
        m.run()
        n_games += 1
        # possession index -> (offence side, defence side)
        sides = {p.poss_index: (p.offense_team_id, p.defense_team_id) for p in m.possessions}
        agg: dict[int, list[int]] = {}
        for pi, nxt, t, leads, per, sec in m._foul_log:
            key = pi if pi is not None else nxt
            if pi is None:
                n_no_open += 1
            s = sides.get(key)
            if s is None:
                n_no_open_unresolved += 1
                continue
            off_side, def_side = s
            a = agg.setdefault(key, [0, 0, 0, 0, 0, 0])   # def_silent, def_trip, off_silent, off_trip, def_pre, off_pre
            if t == def_side:
                a[1 if leads else 0] += 1
                if pi is None:
                    a[4] += 1
            elif t == off_side:
                a[3 if leads else 2] += 1
                if pi is None:
                    a[5] += 1
        s2t = {0: int(gm["home_team_id"]), 1: int(gm["away_team_id"])}
        for pi, fouler, per, sec in m._rowless_log:
            rowless_rows.append({"game_id": gm["game_id"], "season": gm["season"], "poss_index": pi,
                                 "fouling_team_id": s2t[fouler], "period": per, "sec": sec})
        for p in m.possessions:
            a = agg.get(p.poss_index, [0, 0, 0, 0, 0, 0])
            rows.append({
                "game_id": p.game_id, "season": p.season, "period": p.period,
                "poss_index": p.poss_index,
                "offense_team_id": s2t[p.offense_team_id],
                "defense_team_id": s2t[p.defense_team_id],
                "offense_is_home": p.offense_team_id == 0,
                "start_clock": p.start_clock, "start_score_diff": p.start_score_diff,
                "off_team_fouls": p.off_team_fouls, "def_team_fouls": p.def_team_fouls,
                "terminal_event": p.chances[-1].terminal_event,
                "n_chances": len(p.chances),
                "def_silent": a[0], "def_trip": a[1], "off_silent": a[2], "off_trip": a[3],
                "def_pre_open": a[4], "off_pre_open": a[5],
            })
    _GameMachine._handle_foul = orig_foul
    _GameMachine._handle_technical = orig_tech
    _GameMachine._handle_ft_trip = orig_trip
    df = pd.DataFrame(rows)
    rep = {"season": season, "n_games": n_games, "n_possessions": int(len(df)),
           "n_fouls_with_no_open_possession": n_no_open,
           "n_fouls_dropped_unresolved": n_no_open_unresolved,
           "seconds": round(time.time() - t0, 1)}
    rep["n_rowless_trips"] = len(rowless_rows)
    return (df, pd.DataFrame(rowless_rows)), rep


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        out = list(ex.map(_season, SEASONS))
    df = pd.concat([o[0][0] for o in out], ignore_index=True)
    rl = pd.concat([o[0][1] for o in out], ignore_index=True)
    reps = [o[1] for o in out]

    # bonus state at possession OPEN, from the segmenter's own thresholds
    from cbb_sim.pbp.possessions import BONUS_PRIOR_FOULS, DOUBLE_BONUS_PRIOR_FOULS
    df["off_in_bonus"] = (df["def_team_fouls"] >= BONUS_PRIOR_FOULS).astype("int8")
    df["def_in_bonus"] = (df["off_team_fouls"] >= BONUS_PRIOR_FOULS).astype("int8")
    df["off_in_double_bonus"] = (df["def_team_fouls"] >= DOUBLE_BONUS_PRIOR_FOULS).astype("int8")

    u = pd.read_parquet("data/processed/games_universe.parquet",
                        columns=["game_id", "season", "game_date", "neutral_site",
                                 "home_team_id", "season_type"])
    u["game_date"] = pd.to_datetime(u["game_date"])
    df = df.merge(u, on=["game_id", "season"], how="inner")
    oh = df["offense_is_home"].to_numpy()
    nt = df["neutral_site"].to_numpy()
    df["site_home"] = ((~nt) & oh).astype("float32")
    df["site_away"] = ((~nt) & (~oh)).astype("float32")
    first_day = df.groupby("season")["game_date"].transform("min")
    df["days_since_start"] = (df["game_date"] - first_day).dt.days.astype("float32")
    df["season_idx"] = (df["season"] - 2022).astype("float32")
    per = df["period"].to_numpy()
    sc = df["start_clock"].to_numpy()
    df["game_seconds"] = np.where(per == 1, 1200 - sc,
                                  np.where(per == 2, 2400 - sc, 2400 + (per - 2) * 300 - sc))
    df["game_minute"] = (df["game_seconds"] / 60.0).astype("float32")
    df["is_ot"] = (per >= 3).astype("int8")
    # the FIT / PRIMARY mask of 13.5 breakdown 8: regulation, outside the final 2:00
    df["in_fit_window"] = ((per <= 2) & (sc > 120)).astype("int8")

    df["def_team_fouls_true"] = df["def_team_fouls"] - df["def_pre_open"]
    df["off_team_fouls_true"] = df["off_team_fouls"] - df["off_pre_open"]
    df["off_in_bonus_true"] = (df["def_team_fouls_true"] >= BONUS_PRIOR_FOULS).astype("int8")
    # ---- v3: rowless-trip fouls in EARLIER possessions of the same half-segment
    rl = rl.dropna(subset=["poss_index"]).copy()
    rl["poss_index"] = rl["poss_index"].astype("int64")
    rl["seg"] = np.where(rl["period"] <= 1, 1, 2)
    df["seg"] = np.where(df["period"] <= 1, 1, 2)
    for side, col in (("defense_team_id", "def_rowless_prior"), ("offense_team_id", "off_rowless_prior")):
        ev_ = rl.rename(columns={"fouling_team_id": side})[["game_id", "seg", side, "poss_index"]]
        ev_ = ev_.assign(n=1).groupby(["game_id", "seg", side, "poss_index"])["n"].sum().reset_index()
        key = ["game_id", "seg", side]
        q = df[key + ["poss_index"]].reset_index()
        allk = pd.concat([q.assign(n=0, is_q=1), ev_.assign(index=-1, is_q=0)], ignore_index=True)
        # an event at possession index p counts only for possessions AFTER p: queries sort first
        allk = allk.sort_values(key + ["poss_index", "is_q"], ascending=[True, True, True, True, False])
        allk["cum"] = allk.groupby(key)["n"].cumsum()
        got = allk[allk["is_q"] == 1].set_index("index")["cum"]
        df[col] = got.reindex(df.index).fillna(0).astype("int64")
    own = rl.groupby(["game_id", "poss_index"]).size().rename("rowless_own")
    df = df.merge(own.reset_index(), on=["game_id", "poss_index"], how="left")
    df["rowless_own"] = df["rowless_own"].fillna(0).astype("int64")
    df["def_team_fouls_v3"] = df["def_team_fouls_true"] + df["def_rowless_prior"]
    df["off_team_fouls_v3"] = df["off_team_fouls_true"] + df["off_rowless_prior"]
    df["off_in_bonus_v3"] = (df["def_team_fouls_v3"] >= BONUS_PRIOR_FOULS).astype("int8")
    rl.to_parquet(OUT_DIR / "rowless_trips_v3.parquet", index=False)
    df.to_parquet(OUT_DIR / "foul_accrual_poss_v3.parquet", index=False)
    rep = {"built_at": pd.Timestamp.utcnow().isoformat(), "seasons": reps,
           "n_rows": int(len(df)), "seconds_total": round(time.time() - t0, 1),
           "mean_def_silent": float(df["def_silent"].mean()),
           "mean_def_trip": float(df["def_trip"].mean()),
           "mean_off_silent": float(df["off_silent"].mean()),
           "mean_off_trip": float(df["off_trip"].mean()),
           "served_silent_constant": 0.123346}
    (OUT_DIR / "build_report_v3.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
