#!/usr/bin/env python
"""
diag_rotation_sub_hazard.py -- ROUND-4 EVIDENCE: empirical per-player
substitution hazards (out and in) from the 2024 and 2025 on-floor stream.

Why this exists
---------------
Three rounds of the Dirichlet-share/scheduler family (R1, R2) and the
donor-resampling family (R4, R5, R7, R8) cannot produce two structural facts of
a real rotation (L25, `docs/tests/rotation_close_game_audit_2026-09-10.md`
section 7.3):

  * the second half opens at **0.90** starter share in every margin band; no arm
    in three rounds exceeds 0.80;
  * starters are **kept** late in a close game (0.749 in the final 8:00 at
    |margin| <= 5); the best arm reaches 0.728 and the override family's
    reachable maximum over its whole knob grid is 0.699.

Round 4 changes family to per-player discrete-time substitution hazards. This
script is the evidence that precedes the round-4 pre-registration: it measures
the hazards themselves, with confidence intervals, and -- per L25 -- computes
the **reachability** of the two target cells from those hazards BEFORE any arm
is fitted, so a family that cannot reach them is rejected on paper.

What a "hazard" is here
-----------------------
At every possession boundary k (between possession k-1 and possession k) of a
team-game:

  * the **out** risk set is the five on the floor at k-1; the event is "not on
    the floor at k";
  * the **in** risk set is every candidate not on the floor at k-1 and not
    fouled out; the event is "on the floor at k".

That is the same substitution event `rotation.build_hazard_training` derives
(CBBD carries no `Substitution` rows at all in 2024, features.md section 3), at
possession-boundary resolution, which is the resolution the engine acts at.

Descriptive, not predictive
---------------------------
The candidate pool here is every player who appears in the team-game, and the
foul state is that game's own `PersonalFoul` events. Both are contemporaneous
with the game and are correct for a **measurement** of coaching behaviour; the
bake-off's own training path uses the as-of candidate pool and simulated fouls
(`rotation_v4.build_sub_training`), and never these.

Usage
-----
    .venv/Scripts/python.exe scripts/diag_rotation_sub_hazard.py
    .venv/Scripts/python.exe scripts/diag_rotation_sub_hazard.py --games 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.data.player_ids import load_crosswalk  # noqa: E402
from cbb_sim.models import rotation as R  # noqa: E402

OUT_JSON = ROOT / "data" / "processed" / "models" / "rotation" / "sub_hazard_audit_2026-09-10.json"
OUT_MD = ROOT / "docs" / "tests" / "rotation_sub_hazard_audit_2026-09-10.md"

UNDERPOWERED = 300

#: nine time cells, the profile `rotation_close_game_audit` uses plus OT
TIME_CELLS = ["H1 20:00-10:00", "H1 10:00-00:00", "H2 20:00-16:00", "H2 16:00-12:00",
              "H2 12:00-08:00", "H2 08:00-04:00", "H2 04:00-02:00", "H2 02:00-00:00", "OT"]
MARGIN_BANDS = ["|m|<=5", "|m| 6-15", "|m|>15"]
#: `start_reason` levels, which are exactly `engine.state.PREV_END_LEVELS`
REASONS = ["made_FG", "DREB", "TOV", "made_FT", "period_start", "other"]


def time_cell9(period: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """Nine-way time cell. 0-1 first half, 2-7 second half, 8 overtime."""
    p = np.asarray(period)
    c = np.asarray(clock)
    out = np.full(len(p), 8, dtype="int64")
    h1 = p == 1
    out[h1 & (c > 600)] = 0
    out[h1 & (c <= 600)] = 1
    h2 = p == 2
    out[h2 & (c > 960)] = 2
    out[h2 & (c <= 960) & (c > 720)] = 3
    out[h2 & (c <= 720) & (c > 480)] = 4
    out[h2 & (c <= 480) & (c > 240)] = 5
    out[h2 & (c <= 240) & (c > 120)] = 6
    out[h2 & (c <= 120)] = 7
    return out


class Acc:
    """events / trials by integer key, for one hazard table."""

    def __init__(self, size: int):
        self.ev = np.zeros(size, dtype="float64")
        self.n = np.zeros(size, dtype="float64")

    def add(self, key: np.ndarray, y: np.ndarray) -> None:
        if len(key) == 0:
            return
        k = np.asarray(key, dtype="int64")
        self.ev += np.bincount(k, weights=np.asarray(y, dtype="float64"),
                               minlength=len(self.ev))
        self.n += np.bincount(k, minlength=len(self.n))


def wilson(ev: float, n: float, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval; returns (p, lo, hi)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = ev / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(max(0.0, c - h)), float(min(1.0, c + h))


# ---------------------------------------------------------------------------
# season assembly
# ---------------------------------------------------------------------------
def load_side_state(season: int) -> pd.DataFrame:
    """(game_id, poss_index, is_home) -> start_reason code, own team fouls.

    `load_team_possessions` does not carry either, and `rotation.py` is not
    ours to change, so the two columns are re-read from the same parquet and
    merged on the keys it does carry."""
    p = pd.read_parquet(
        ROOT / "data" / "processed" / "possessions" / f"possessions_{season}.parquet",
        columns=["game_id", "poss_index", "offense_is_home", "start_reason",
                 "off_team_fouls", "def_team_fouls", "stolen"])
    rows = []
    for is_home in (True, False):
        off = p["offense_is_home"].to_numpy() == is_home
        rows.append(pd.DataFrame({
            "game_id": p["game_id"].to_numpy(),
            "poss_index": p["poss_index"].to_numpy(),
            "is_home": is_home,
            "reason": p["start_reason"].to_numpy(),
            "stolen": p["stolen"].to_numpy(),
            "own_team_fouls": np.where(off, p["off_team_fouls"].to_numpy(),
                                       p["def_team_fouls"].to_numpy()),
        }))
    return pd.concat(rows, ignore_index=True)


def load_timeouts(season: int) -> set:
    """(cbbd_game_id, period, secondsRemaining) of every timeout row."""
    try:
        df = pd.read_parquet(ROOT / "data" / "raw" / "cbbd" / "pbp" / f"plays_{season}.parquet",
                             columns=["gameId", "id", "playType", "period", "secondsRemaining"])
    except Exception:
        return set()
    df = df.drop_duplicates(subset=["gameId", "id"])
    t = df[df["playType"].isin(["OfficialTVTimeOut", "ShortTimeOut", "RegularTimeOut"])]
    return set(zip(t["gameId"].to_numpy().astype("int64"),
                   t["period"].to_numpy().astype("int64"),
                   t["secondsRemaining"].to_numpy().astype("int64")))


def prior_season_share(season: int, cw: pd.DataFrame) -> dict:
    """CBBD pid -> that player's share of his team's minutes in season-1,
    from hoopR `player_box`, mapped through the crosswalk.

    Returns {} when the prior season's box file does not exist."""
    prev = season - 1
    path = ROOT / "data" / "raw" / "hoopr" / "player_box" / f"player_box_{prev}.parquet"
    if not path.exists():
        return {}
    pb = pd.read_parquet(path, columns=["game_id", "team_id", "athlete_id", "minutes"])
    pb = pb[pb["athlete_id"].notna()]
    pb["minutes"] = pd.to_numeric(pb["minutes"], errors="coerce").fillna(0.0)
    tot = pb.groupby(["team_id", "athlete_id"], as_index=False)["minutes"].sum()
    team_tot = tot.groupby("team_id")["minutes"].transform("sum")
    tot["share"] = np.where(team_tot > 0, tot["minutes"] / team_tot * 5.0, 0.0)
    # espn -> cbbd for THIS season's id space
    m = {}
    sub = cw[cw["season"] == season] if "season" in cw.columns else cw
    for c, e in zip(sub["cbbd_player_id"].to_numpy(), sub["espn_athlete_id"].to_numpy()):
        if pd.notna(e):
            m.setdefault(int(e), int(c))
    out: dict[int, float] = {}
    for e, s in zip(tot["athlete_id"].to_numpy(), tot["share"].to_numpy()):
        c = m.get(int(e))
        if c is not None:
            out[c] = float(out.get(c, 0.0) + s)
    return out


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def sweep(season: int, max_games: int = 0) -> dict:
    t0 = time.time()
    tp = R.load_team_possessions(season)
    side = load_side_state(season)
    tp = tp.merge(side, on=["game_id", "poss_index", "is_home"], how="left")
    tp["reason"] = tp["reason"].fillna("other")
    gids = np.sort(tp["game_id"].unique())
    if max_games:
        gids = gids[:max_games]
        tp = tp[tp["game_id"].isin(set(int(g) for g in gids))]

    gu = pd.read_parquet(ROOT / "data" / "processed" / "games_universe.parquet")
    gu = gu[gu["season"] == season][["game_id", "cbbd_game_id"]]
    cbbd_of = dict(zip(gu["game_id"].to_numpy().astype("int64"),
                       gu["cbbd_game_id"].to_numpy().astype("int64")))
    tos = load_timeouts(season)

    pg = R.player_game_minutes(tp)
    ev = R.player_game_fouls(season).merge(gu, on="cbbd_game_id", how="inner")
    ev = ev.merge(pg[["game_id", "team_id", "pid"]].drop_duplicates(),
                  on=["game_id", "pid"], how="inner")
    feats = R.build_asof_player_features(pg, fouls=ev)
    # as-of share: shrunk-free, the raw as-of minutes share x 5 (so 1.0 is an
    # average rotation slot). Strictly earlier games only, by construction.
    fs = feats[["game_id", "team_id", "pid", "mpg_asof_raw", "team_games_asof"]].copy()
    tt = fs.groupby(["game_id", "team_id"])["mpg_asof_raw"].transform("sum")
    fs["share_asof"] = np.where(tt > 0, fs["mpg_asof_raw"] / tt * 5.0, 0.0)
    share_asof = {(int(g), int(t), int(p)): float(s) for g, t, p, s in
                  zip(fs["game_id"], fs["team_id"], fs["pid"], fs["share_asof"])}
    n_prior = {(int(g), int(t)): int(v) for g, t, v in
               zip(fs["game_id"], fs["team_id"], fs["team_games_asof"])}

    cw = load_crosswalk()
    prior_share = prior_season_share(season, cw)

    evg = {k: g for k, g in ev.groupby(["game_id", "team_id"], sort=False)}

    # ---- accumulators ----------------------------------------------------
    A = {
        # [side(out/in)][starter][time_cell][margin_band]
        "time_margin": Acc(2 * 2 * 9 * 3),
        "reason": Acc(2 * 2 * len(REASONS)),
        "timeout": Acc(2 * 2 * 2),
        "fouls": Acc(2 * 2 * 6),
        "team_fouls": Acc(2 * 2 * 5),
        "run": Acc(2 * 2 * 6),        # stint (out) / rest (in) minutes bucket
        "half_min": Acc(2 * 2 * 6),
        "share_q": Acc(2 * 2 * 5),
        "prior_q": Acc(2 * 2 * 6),    # 5 quintiles + "no prior season"
    }
    # second-half tip: starter share at the FIRST possession of period 2, by band
    tip = Acc(3 * 2)                   # [band][is_starter] -> on-floor at the tip
    # transition into the H2 tip, conditioned on the end of H1
    tip_tr = Acc(2 * 2)                # [is_starter][was_on_at_end_of_H1]
    # opening-tip cell and the close-late cell, for the team-quintile slope
    per_team: dict[tuple, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # occupancy (starter share of on-floor slots) by time cell x band
    occ = Acc(9 * 3 * 2)

    share_edges = None
    shares_all = np.array(sorted(share_asof.values())) if share_asof else np.array([0.0])
    share_edges = np.quantile(shares_all, [0.2, 0.4, 0.6, 0.8])
    pri_all = np.array(sorted(prior_share.values())) if prior_share else np.array([0.0])
    pri_edges = np.quantile(pri_all, [0.2, 0.4, 0.6, 0.8])

    n_tg = 0
    for (gid, tid), g in tp.groupby(["game_id", "team_id"], sort=False):
        gid, tid = int(gid), int(tid)
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        npos = lu.shape[0]
        if npos < 20:
            continue
        dur = g["duration_s"].to_numpy(dtype="float64")
        period = g["period"].to_numpy(dtype="int64")
        clock = g["start_clock"].to_numpy(dtype="int64")
        margin = g["margin"].to_numpy(dtype="int64")
        reason = g["reason"].to_numpy()
        tfouls = np.nan_to_num(g["own_team_fouls"].to_numpy(dtype="float64"))
        poss_idx = g["poss_index"].to_numpy(dtype="int64")
        tc = time_cell9(period, clock)
        mb = R.margin_bucket(margin)

        pids = np.unique(lu)
        nc = len(pids)
        pos = {int(p): i for i, p in enumerate(pids)}
        on = np.zeros((npos, nc), dtype=bool)
        for s in range(5):
            on[np.arange(npos), np.searchsorted(pids, lu[:, s])] = True

        starters = np.zeros(nc, dtype=bool)
        starters[np.searchsorted(pids, np.unique(lu[0]))] = True

        fm = R.actual_foul_matrix(evg.get((gid, tid)), period, clock, pos).astype("int64")
        if fm.shape[0] < npos:
            fm = np.vstack([fm, np.repeat(fm[-1:], npos - fm.shape[0], axis=0)])

        # run length in the current on/off state, in seconds, at the START of k
        cum = np.concatenate([[0.0], np.cumsum(dur)])
        chg = np.empty_like(on)
        chg[0] = True
        chg[1:] = on[1:] != on[:-1]
        idx = np.arange(npos)[:, None]
        run_start = np.maximum.accumulate(np.where(chg, idx, -1), axis=0)
        # the run length AT THE START of possession k must be built from changes
        # up to k-1 only: `chg[k]` IS the label at boundary k, so using it here
        # would make the feature a copy of the answer.
        run_prev = np.vstack([np.zeros((1, nc), dtype=run_start.dtype), run_start[:-1]])
        run_sec = cum[np.arange(npos)][:, None] - cum[np.clip(run_prev, 0, None)]

        # minutes played so far in the CURRENT half, at the start of k
        half = np.where(period == 1, 0, 1)
        on_sec = on * dur[:, None]
        cs = np.cumsum(on_sec, axis=0) - on_sec          # exclusive prefix
        h2_start = int(np.searchsorted(half, 1))
        base = cs[h2_start] if h2_start < npos else cs[-1]
        half_sec = np.where((half == 1)[:, None], cs - base[None, :], cs)

        gshare = np.array([share_asof.get((gid, tid, int(p)), 0.0) for p in pids])
        gpri = np.array([prior_share.get(int(p), -1.0) for p in pids])

        to_flag = np.zeros(npos, dtype=bool)
        cg = cbbd_of.get(gid)
        if cg is not None and tos:
            for k in range(npos):
                if (cg, int(period[k]), int(clock[k])) in tos:
                    to_flag[k] = True

        # ---- occupancy (the cells the gate reads) ------------------------
        st_slot = on[:, starters].sum(axis=1)
        occ.add(tc * 6 + mb * 2 + 0, st_slot)
        occ.add(tc * 6 + mb * 2 + 1, np.full(npos, 5.0))

        # ---- team-level cells for the slope check ------------------------
        rec = per_team[(gid, tid)]
        close_late = (tc >= 5) & (tc <= 7) & (mb == 0)
        rec[0] += float(st_slot[close_late].sum())
        rec[1] += float(5 * close_late.sum())
        open_close = (tc == 0) & (mb == 0)
        rec[2] += float(st_slot[open_close].sum())
        rec[3] += float(5 * open_close.sum())
        rec[4] = float(np.mean(gshare[starters])) if starters.any() else np.nan

        # ---- the second-half tip -----------------------------------------
        if h2_start < npos and h2_start > 0:
            k = h2_start
            b = int(mb[k])
            tip.add(np.array([b * 2 + 0]), np.array([float(on[k, starters].sum())]))
            tip.add(np.array([b * 2 + 1]), np.array([5.0]))
            was = on[k - 1]
            for is_st in (0, 1):
                sel = starters if is_st else ~starters
                if not sel.any():
                    continue
                for w in (0, 1):
                    m = sel & (was == bool(w))
                    if m.any():
                        tip_tr.add(np.full(int(m.sum()), is_st * 2 + w),
                                   on[k][m].astype(float))

        # ---- the hazard risk sets ---------------------------------------
        rb_run = np.clip((run_sec / 60.0), 0, None)
        for k in range(1, npos):
            prev_on = on[k - 1]
            fouled_out = fm[k] >= R.FOUL_OUT
            out_idx = np.flatnonzero(prev_on)
            in_idx = np.flatnonzero(~prev_on & ~fouled_out)
            if len(out_idx) != 5 or len(in_idx) == 0:
                continue
            y_out = (~on[k][out_idx]).astype("float64")
            y_in = on[k][in_idx].astype("float64")
            rcode = REASONS.index(reason[k]) if reason[k] in REASONS else 5
            tfb = int(min(4, tfouls[k] // 3))
            for sidec, ii, yy in ((0, out_idx, y_out), (1, in_idx, y_in)):
                st = starters[ii].astype("int64")
                base_k = sidec * 2 + st
                A["time_margin"].add(base_k * 27 + int(tc[k]) * 3 + int(mb[k]), yy)
                A["reason"].add(base_k * len(REASONS) + rcode, yy)
                A["timeout"].add(base_k * 2 + int(to_flag[k]), yy)
                A["fouls"].add(base_k * 6 + np.clip(fm[k][ii], 0, 5), yy)
                A["team_fouls"].add(base_k * 5 + tfb, yy)
                A["run"].add(base_k * 6 + np.clip((rb_run[k][ii] // 1.0).astype("int64"), 0, 5), yy)
                A["half_min"].add(base_k * 6 + np.clip((half_sec[k][ii] / 300.0).astype("int64"),
                                                       0, 5), yy)
                A["share_q"].add(base_k * 5 + np.searchsorted(share_edges, gshare[ii]), yy)
                pq = np.where(gpri[ii] < 0, 5, np.searchsorted(pri_edges, np.maximum(gpri[ii], 0)))
                A["prior_q"].add(base_k * 6 + pq, yy)
        n_tg += 1

    out = {
        "season": season,
        "team_games": n_tg,
        "games": int(len(gids)),
        "seconds": round(time.time() - t0, 1),
        "share_edges": share_edges.tolist(),
        "prior_edges": pri_edges.tolist(),
        "prior_coverage": float(np.mean([1.0 if p in prior_share else 0.0
                                         for p in list(share_asof.keys())[:0]] or [0.0])),
        "acc": {k: {"ev": v.ev.tolist(), "n": v.n.tolist()} for k, v in A.items()},
        "tip": {"ev": tip.ev.tolist(), "n": tip.n.tolist()},
        "tip_tr": {"ev": tip_tr.ev.tolist(), "n": tip_tr.n.tolist()},
        "occ": {"ev": occ.ev.tolist(), "n": occ.n.tolist()},
        "per_team": {f"{g}_{t}": v for (g, t), v in per_team.items()},
        "timeout_rows": int(len(tos)),
    }
    return out


# ---------------------------------------------------------------------------
# L25 reachability probe -- BEFORE any arm is fitted
# ---------------------------------------------------------------------------
def cell_hazards(season: int, max_games: int = 0):
    """Empirical [out/in][is_starter][time_cell][margin_band][period_boundary]
    hazards on `season`.

    This is the SATURATED cell form of the round-4 family -- the most any
    per-player hazard model conditioned on those cells could know -- so the
    occupancy it produces under the five-on-the-floor constraint is the
    family's reachable frontier for those cells (L25)."""
    tp = R.load_team_possessions(season)
    side = load_side_state(season)
    tp = tp.merge(side, on=["game_id", "poss_index", "is_home"], how="left")
    tp["reason"] = tp["reason"].fillna("other")
    if max_games:
        gids = np.sort(tp["game_id"].unique())[:max_games]
        tp = tp[tp["game_id"].isin(set(int(g) for g in gids))]
    ev = np.zeros((2, 2, 9, 3, 2))
    nn = np.zeros((2, 2, 9, 3, 2))
    for (gid, tid), g in tp.groupby(["game_id", "team_id"], sort=False):
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        npos = lu.shape[0]
        if npos < 20:
            continue
        period = g["period"].to_numpy(dtype="int64")
        clock = g["start_clock"].to_numpy(dtype="int64")
        tc = time_cell9(period, clock)
        mb = R.margin_bucket(g["margin"].to_numpy(dtype="int64"))
        pb = (g["reason"].to_numpy() == "period_start").astype("int64")
        pids = np.unique(lu)
        on = np.zeros((npos, len(pids)), dtype=bool)
        for sl in range(5):
            on[np.arange(npos), np.searchsorted(pids, lu[:, sl])] = True
        st = np.zeros(len(pids), dtype="int64")
        st[np.searchsorted(pids, np.unique(lu[0]))] = 1
        for k in range(1, npos):
            prev = on[k - 1]
            oi = np.flatnonzero(prev)
            ii = np.flatnonzero(~prev)
            if len(oi) != 5 or len(ii) == 0:
                continue
            t, b, q = int(tc[k]), int(mb[k]), int(pb[k])
            for sflag in (0, 1):
                m = oi[st[oi] == sflag]
                if len(m):
                    ev[0, sflag, t, b, q] += float((~on[k][m]).sum())
                    nn[0, sflag, t, b, q] += len(m)
                m = ii[st[ii] == sflag]
                if len(m):
                    ev[1, sflag, t, b, q] += float(on[k][m].sum())
                    nn[1, sflag, t, b, q] += len(m)
    return ev, nn


def reach_probe(train_season: int, test_season: int, n_games: int, seed: int = 7,
                train_games: int = 0) -> dict:
    """Run the saturated cell-hazard sampler on real game scripts, with the REAL
    starting fives and REAL participant pools, under the five-on-the-floor
    constraint, and read the two round-4 target cells plus the opening-tip cell.

    This is NOT a bake-off arm: it has no as-of anything, it is fitted on 2024
    and probed on 2025, and it is graded on nothing. It answers one question --
    CAN this family reach the cells? -- before the round-4 pre-registration is
    written, which is what L25 requires."""
    ev, nn = cell_hazards(train_season, train_games)
    h = np.where(nn > 30, ev / np.maximum(nn, 1), np.nan)
    for a in range(2):
        for sflag in (0, 1):
            for t in range(9):
                tot_n = nn[a, sflag, t].sum()
                fb = (ev[a, sflag, t].sum() / tot_n) if tot_n > 0 else 0.05
                cell = h[a, sflag, t]
                cell[np.isnan(cell)] = fb

    tp = R.load_team_possessions(test_season)
    side = load_side_state(test_season)
    tp = tp.merge(side, on=["game_id", "poss_index", "is_home"], how="left")
    tp["reason"] = tp["reason"].fillna("other")
    gids = np.sort(tp["game_id"].unique())
    rs = np.random.RandomState(2025)
    gids = gids[rs.choice(len(gids), min(n_games, len(gids)), replace=False)]
    tp = tp[tp["game_id"].isin(set(int(g) for g in gids))]
    rng = np.random.default_rng(seed)

    def blank():
        return {"h2_tip": np.zeros((3, 2)), "close_late": np.zeros(2),
                "open_close": np.zeros(2), "late": np.zeros((3, 2)),
                "profile": np.zeros((9, 3, 2))}

    cells, act = blank(), blank()
    for (gid, tid), g in tp.groupby(["game_id", "team_id"], sort=False):
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        npos = lu.shape[0]
        if npos < 20:
            continue
        period = g["period"].to_numpy(dtype="int64")
        clock = g["start_clock"].to_numpy(dtype="int64")
        tc = time_cell9(period, clock)
        mb = R.margin_bucket(g["margin"].to_numpy(dtype="int64"))
        pb = (g["reason"].to_numpy() == "period_start").astype("int64")
        pids = np.unique(lu)
        nc = len(pids)
        if nc < 6:
            continue
        st = np.zeros(nc, dtype=bool)
        st[np.searchsorted(pids, np.unique(lu[0]))] = True
        stf = st.astype("int64")
        onmask = st.copy()
        sim_share = np.zeros(npos)
        for k in range(npos):
            if k > 0:
                oi = np.flatnonzero(onmask)
                bi = np.flatnonzero(~onmask)
                if len(oi) == 5 and len(bi):
                    po = h[0, stf[oi], tc[k], mb[k], pb[k]]
                    exits = rng.random(5) < po
                    ne = int(min(exits.sum(), len(bi)))
                    if ne:
                        pi = np.clip(h[1, stf[bi], tc[k], mb[k], pb[k]], 1e-6, 1 - 1e-6)
                        w = pi / (1.0 - pi)
                        key = -np.log(np.clip(rng.random(len(bi)), 1e-12, 1.0)) / w
                        pick = np.argsort(key)[:ne]
                        onmask[oi[exits][:ne]] = False
                        onmask[bi[pick]] = True
            sim_share[k] = float(onmask[st].sum())
        a_share = np.array([float(np.isin(pids[st], lu[k]).sum()) for k in range(npos)])

        for arr, sh in ((cells, sim_share), (act, a_share)):
            h2 = np.flatnonzero(period == 2)
            if len(h2):
                k = int(h2[0])
                arr["h2_tip"][mb[k], 0] += sh[k]
                arr["h2_tip"][mb[k], 1] += 5.0
            cl = np.isin(tc, (5, 6, 7)) & (mb == 0)
            arr["close_late"][0] += sh[cl].sum()
            arr["close_late"][1] += 5.0 * cl.sum()
            oc = (tc == 0) & (mb == 0)
            arr["open_close"][0] += sh[oc].sum()
            arr["open_close"][1] += 5.0 * oc.sum()
            for b in range(3):
                m = np.isin(tc, (5, 6, 7)) & (mb == b)
                arr["late"][b, 0] += sh[m].sum()
                arr["late"][b, 1] += 5.0 * m.sum()
            for t in range(9):
                for b in range(3):
                    m = (tc == t) & (mb == b)
                    if m.any():
                        arr["profile"][t, b, 0] += sh[m].sum()
                        arr["profile"][t, b, 1] += 5.0 * m.sum()

    out = {"train_season": train_season, "test_season": test_season,
           "n_games": int(len(gids)),
           "hazards": {"ev": ev.tolist(), "n": nn.tolist()}}
    for tag, d in (("sim", cells), ("actual", act)):
        out[tag] = {k: (v[..., 0] / np.maximum(v[..., 1], 1)).tolist()
                    for k, v in d.items()}
        out[tag + "_n"] = {k: v[..., 1].tolist() for k, v in d.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--seasons", default="2024,2025")
    ap.add_argument("--mode", default="sweep", choices=["sweep", "reach"])
    ap.add_argument("--reach-games", type=int, default=400)
    a = ap.parse_args()
    if a.mode == "reach":
        out = reach_probe(2024, 2025, a.reach_games, train_games=a.games)
        pth = OUT_JSON.with_name("sub_hazard_reachability_2026-09-10.json")
        pth.write_text(json.dumps(out), encoding="utf-8")
        print("sim h2_tip     ", np.round(out["sim"]["h2_tip"], 4).tolist())
        print("act h2_tip     ", np.round(out["actual"]["h2_tip"], 4).tolist())
        print("sim close_late ", round(out["sim"]["close_late"], 4),
              "  act", round(out["actual"]["close_late"], 4))
        print("sim open_close ", round(out["sim"]["open_close"], 4),
              "  act", round(out["actual"]["open_close"], 4))
        print("sim late bands ", np.round(out["sim"]["late"], 4).tolist())
        print("act late bands ", np.round(out["actual"]["late"], 4).tolist())
        print("wrote", pth)
        return 0
    res = {}
    for s in [int(x) for x in a.seasons.split(",")]:
        print(f"[sweep] season {s}", flush=True)
        res[str(s)] = sweep(s, a.games)
        print(f"  done: {res[str(s)]['team_games']} team-games "
              f"in {res[str(s)]['seconds']}s", flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
