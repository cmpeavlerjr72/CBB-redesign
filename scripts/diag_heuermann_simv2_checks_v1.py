#!/usr/bin/env python
"""
diag_heuermann_simv2_checks_v1.py -- does OUR engine/data have the exposures Heuermann found in his sim_v2?

Diagnostic only (2026-11-08, Opus worker). Changes no engine code, adopts nothing, fits nothing to sim output.
Reply to `rfq-collab/CBB Model/Heuermann/study/sim_v2/PROGRESS_2026-11-07.md` s2/s4.3 and the 2026-11-07 INTRO
handoff s2 items 3, 4, 6. Report: docs/tests/heuermann_simv2_engine_checks_2026-11-08.md.

Seasons: 2022-2025 labels (2021-22 .. 2024-25). 2026 (2025-26) is SEALED and never loaded.

  real   One instrumented replay of `cbb_sim.pbp.possessions._GameMachine` (v4 switch set: tech lookahead + the
         three phantom fixes) over CBBD plays 2022-2025, wrapped IN THIS PROCESS ONLY (the
         build_foul_accrual_design_v2.py pattern). Per possession: chance-1 class/duration, counts, stolen flag,
         team-credited-turnover flag (no participant on the TOV row; the feed has no shot-clock-violation type),
         the labelled defence foul count and an ALT count that also charges a foul for every free-throw trip with
         no foul row. Per trip: row-less flag. Per technical: rows and duplicate-PersonalFoul rows.
  sim    The served-v3 closed-loop control trajectories already on disk (late-game round 7, 2026-10-13):
         results/trajectories/eg_ctrl_o{0,1000,2000,3000,4000}_s25 = 500 verified 2024-25 games x 125 seeds.
  dup    Duplicate / phantom-censored-record counts on the tables the served clock is fitted on.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/diag_heuermann_simv2_checks_v1.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.pbp import events as EV  # noqa: E402
from cbb_sim.pbp import possessions as PZ  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
PBP_DIR = ROOT / "data/raw/cbbd/pbp"
UNIVERSE = ROOT / "data/processed/games_universe.parquet"
IDS500 = ROOT / "data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet"
TRAJ = [ROOT / f"results/trajectories/eg_ctrl_o{o}_s25/trajectory.parquet" for o in (0, 1000, 2000, 3000, 4000)]
SCRATCH = ROOT / "results/diag_heuermann_simv2"
OUT_CSV = Path(r"C:\Users\devuser\rfq-collab\CBB Model\Handoffs - Communication\2026-11-08_Peavler_REPLY_cbb-sim-v2")
FGA = ("FGA_rim", "FGA_jump2", "FGA_3")
LEAD_BINS = [(1, 3), (4, 6), (7, 12)]
WIN = (31, 60)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# real: instrumented replay
# ----------------------------------------------------------------------------------------------------------------
class TapMachine(PZ._GameMachine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trip_log: list[tuple] = []        # (period, side, rowless, technical, n_ft, sec)
        self.tech_rows = 0
        self.tech_dup_pf = 0
        self.extra: dict[tuple[int, int], int] = {}
        self.alt_def: dict[int, int] = {}
        self.team_tov: dict[int, bool] = {}

    def _handle_ft_trip(self, i, t, prior_fouls, technical):
        k, made = self._collect_trip(i, t)
        if made:
            rowless = prior_fouls is None and not technical
            self.trip_log.append((self.period, int(t), rowless, bool(technical), len(made), int(self.ev["sec"][i])))
            if rowless:
                key = (self.period, self._other(int(t)))
                self.extra[key] = self.extra.get(key, 0) + 1
        return super()._handle_ft_trip(i, t, prior_fouls, technical)

    def _handle_technical(self, i, t):
        self.tech_rows += 1
        return super()._handle_technical(i, t)

    def _handle_foul(self, i, t):
        j = i + 1
        if j < self.n and self.ev["cls"][j] in ("FT_made", "FT_missed") and self.pending_tech_ft_idx == j:
            self.tech_dup_pf += 1
        return super()._handle_foul(i, t)

    def _open(self, team, i):
        super()._open(team, i)
        opp = self._other(team)
        self.alt_def[self.poss_index] = self.cur.def_team_fouls + self.extra.get((self.period, opp), 0)

    def _close(self, terminal, i, reason_next="other"):
        if self.cur is not None and terminal == "TOV" and i < self.n:
            # on_floor column 0 was overwritten with "TOV row carries a participant" (see replay_season)
            self.team_tov[self.cur.poss_index] = not bool(self.ev["on_floor"][i][0] > 0)
        super()._close(terminal, i, reason_next)


def replay_season(season: int, u: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    us = u[u["season"] == season]
    meta = {int(r.cbbd_game_id): {"game_id": int(r.game_id), "cbbd_game_id": int(r.cbbd_game_id),
                                  "season": int(r.season), "home_team_id": int(r.home_team_id),
                                  "away_team_id": int(r.away_team_id)} for r in us.itertuples()}
    plays = EV.load_plays(season, pbp_dir=PBP_DIR, game_ids=set(meta),
                          columns=EV.PLAY_COLUMNS + ("participant_1_id",))
    plays["home_on_1"] = pd.to_numeric(plays["participant_1_id"], errors="coerce").notna().astype("float64")
    for c in ["home_on_2", "home_on_3", "home_on_4", "home_on_5"] + [f"away_on_{k}" for k in range(1, 6)]:
        plays[c] = 0.0
    ev = PZ._prepare_events(plays, with_on_floor=True)
    games = ev["game"]
    bounds = np.flatnonzero(np.concatenate([[True], games[1:] != games[:-1], [True]]))
    prow, trow = [], []
    diag = {"games": 0, "tech_rows": 0, "tech_dup_pf": 0, "end_period_rows": 0}
    sw = PZ.VERSION_EVENT_FIXES["v4"]
    for b in range(len(bounds) - 1):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        gm = meta.get(int(games[lo]))
        if gm is None:
            continue
        sub = {k: ev[k][lo:hi] for k in ("cls", "team", "sec", "period", "hs", "as_", "made", "stolen")}
        sub["on_floor"] = ev["on_floor"][lo:hi]
        m = TapMachine(gm, sub, tech_lookahead=True, andone_made_next="made_FT", **sw)
        m.run()
        diag["games"] += 1
        diag["tech_rows"] += m.tech_rows
        diag["tech_dup_pf"] += m.tech_dup_pf
        diag["end_period_rows"] += int(np.isin(sub["cls"], ["end_period", "end_game"]).sum())
        for per, side, rl, tech, nft, sec in m.trip_log:
            trow.append((gm["game_id"], season, per, side, rl, tech, nft, sec))
        for p in m.possessions:
            ch = p.chances
            c1 = ch[0]
            last = ch[-1]
            fga = sum(c.fga_rim + c.fga_jump2 + c.fga_3 for c in ch)
            prow.append((gm["game_id"], season, p.period, p.poss_index, p.offense_team_id, p.start_clock,
                         last.end_clock, max(0, p.start_clock - last.end_clock), p.start_score_diff, len(ch),
                         c1.terminal_event, max(0, c1.start_clock - c1.end_clock) if c1.end_clock is not None else -1,
                         last.terminal_event, fga, sum(c.fta for c in ch), any(c.stolen for c in ch),
                         bool(m.team_tov.get(p.poss_index, False)), p.def_team_fouls,
                         m.alt_def.get(p.poss_index, p.def_team_fouls)))
    P = pd.DataFrame(prow, columns=["game_id", "season", "period", "poss_index", "off_side", "start_clock",
                                    "end_clock", "duration_s", "start_score_diff", "n_chances", "c1_terminal",
                                    "c1_duration", "terminal", "fga", "fta", "stolen", "team_tov",
                                    "def_fouls_lab", "def_fouls_alt"])
    T = pd.DataFrame(trow, columns=["game_id", "season", "period", "ft_side", "rowless", "technical", "n_ft", "sec"])
    return P, T, diag


def build_real(force: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fp, ft, fd = SCRATCH / "real_poss.parquet", SCRATCH / "real_trips.parquet", SCRATCH / "real_diag.json"
    if fp.exists() and ft.exists() and fd.exists() and not force:
        return pd.read_parquet(fp), pd.read_parquet(ft), json.loads(fd.read_text())
    u = pd.read_parquet(UNIVERSE)
    u = u[u["is_d1_game"] & ~u["pbp_truncated"] & u["cbbd_game_id"].notna()].copy()
    u["cbbd_game_id"] = u["cbbd_game_id"].astype("int64")
    Ps, Ts, D = [], [], {}
    for s in SEASONS:
        assert s != 2026
        t0 = time.time()
        P, T, d = replay_season(s, u)
        d["poss"] = int(len(P))
        D[str(s)] = d
        Ps.append(P)
        Ts.append(T)
        log(f"real {s}: {d['games']} games, {len(P):,} poss, {len(T):,} trips ({time.time() - t0:.0f}s)")
    P, T = pd.concat(Ps, ignore_index=True), pd.concat(Ts, ignore_index=True)
    P.to_parquet(fp, index=False)
    T.to_parquet(ft, index=False)
    fd.write_text(json.dumps(D, indent=1))
    return P, T, D


# ----------------------------------------------------------------------------------------------------------------
# sim
# ----------------------------------------------------------------------------------------------------------------
def load_sim() -> pd.DataFrame:
    cols = ["game_id", "seed", "poss_idx", "period", "clock_start", "clock_end", "off_side", "duration_s",
            "end_type", "n_chances", "fga2", "fga3", "fta", "tov", "points", "margin", "off_in_bonus"]
    fr = []
    for i, f in enumerate(TRAJ):
        t = pd.read_parquet(f, columns=cols)
        t["seed"] = t["seed"].astype(np.int32)
        fr.append(t)
    S = pd.concat(fr, ignore_index=True)
    off_m_post = np.where(S["off_side"].to_numpy() == 0, S["margin"].to_numpy(), -S["margin"].to_numpy())
    S["start_score_diff"] = off_m_post - S["points"].to_numpy()
    S["fga"] = S["fga2"] + S["fga3"]
    return S


# ----------------------------------------------------------------------------------------------------------------
# A: late-game cells
# ----------------------------------------------------------------------------------------------------------------
def _cell_stats(d: pd.DataFrame, side: str) -> dict:
    n = len(d)
    if n == 0:
        return {"n": 0}
    fga = d["fga"].to_numpy() > 0
    fta = d["fta"].to_numpy() > 0
    tov = (d["tov"].to_numpy() > 0) if side == "sim" else (d["terminal"].to_numpy() == "TOV")
    foul_ns = ~fga & fta
    tov_ns = ~fga & ~fta & tov
    other = ~fga & ~fta & ~tov
    dur = d["duration_s"].to_numpy().astype(float)
    out = {"n": n, "p_fga": fga.mean(), "p_foul_noshot": foul_ns.mean(), "p_tov_noshot": tov_ns.mean(),
           "p_other": other.mean(),
           "foul_before_shot_given_foul_or_fga": foul_ns.sum() / max(foul_ns.sum() + fga.sum(), 1),
           "mean_dur_all": dur.mean(), "n_tov": int(tov_ns.sum()),
           "mean_dur_tov": dur[tov_ns].mean() if tov_ns.any() else np.nan,
           "share_tov_dur_ge30": (dur[tov_ns] >= 30).mean() if tov_ns.any() else np.nan}
    if side == "real":
        st = d["stolen"].to_numpy()
        dead = tov_ns & ~st
        out.update({"n_dead_tov": int(dead.sum()),
                    "mean_dur_dead_tov": dur[dead].mean() if dead.any() else np.nan,
                    "share_dead_tov_dur_ge30": (dur[dead] >= 30).mean() if dead.any() else np.nan,
                    "share_dead_tov_team_credited": d["team_tov"].to_numpy()[dead].mean() if dead.any() else np.nan,
                    "p_foul_noshot_bonus_class": (foul_ns & (d["c1_terminal"].to_numpy() == "FT_trip_bonus")).mean(),
                    "p_c1_fga": np.isin(d["c1_terminal"].to_numpy(), FGA).mean()})
    else:
        out.update({"n_games_x_seeds": int(d[["game_id", "seed"]].drop_duplicates().shape[0])})
    return out


def lategame(P: pd.DataFrame, S: pd.DataFrame, ids500: set) -> pd.DataFrame:
    rows = []
    reg = P[(P["period"] == 2) & P["start_clock"].between(*WIN)]
    sreg = S[(S["period"] == 2) & S["clock_start"].between(*WIN)]
    for lo, hi in LEAD_BINS + [(-6, -4)]:
        role = f"leader_{lo}-{hi}" if lo > 0 else f"trailer_{-hi}-{-lo}"
        r_all = reg[reg["start_score_diff"].between(lo, hi)]
        r_500 = r_all[r_all["game_id"].isin(ids500)]
        r_25 = r_all[r_all["season"] == 2025]
        s_c = sreg[sreg["start_score_diff"].between(lo, hi)]
        for src, d, side in (("real_2022-2025_all", r_all, "real"), ("real_2025_all", r_25, "real"),
                             ("real_2025_same500", r_500, "real"),
                             ("sim_servedv3_500x125", s_c, "sim")):
            st = _cell_stats(d, side)
            st.update({"cell": f"P2 {WIN[0]}-{WIN[1]}s {role}", "source": src})
            rows.append(st)
    out = pd.DataFrame(rows)
    lead = ["cell", "source", "n"]
    return out[lead + [c for c in out.columns if c not in lead]]


# ----------------------------------------------------------------------------------------------------------------
# B: feed artefacts
# ----------------------------------------------------------------------------------------------------------------
def rowless(T: pd.DataFrame, P: pd.DataFrame, D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for s, g in T.groupby("season"):
        nt = g[~g["technical"]]
        two = nt[nt["n_ft"] == 2]
        ngames = D[str(s)]["games"]
        rows.append({"season": int(s), "games": ngames, "trips_nontech": len(nt), "two_shot_trips": len(two),
                     "two_shot_rowless": int(two["rowless"].sum()),
                     "two_shot_rowless_rate": two["rowless"].mean(),
                     "all_trips_rowless": int(nt["rowless"].sum()), "all_trips_rowless_rate": nt["rowless"].mean(),
                     "rowless_trips_per_game": nt["rowless"].sum() / ngames,
                     "technical_trips": int(g["technical"].sum()),
                     "technical_rows_per_game": D[str(s)]["tech_rows"] / ngames,
                     "technical_dup_personalfoul_rows_per_game": D[str(s)]["tech_dup_pf"] / ngames})
    rl = pd.DataFrame(rows)
    # bonus-label effect: offence-in-bonus label vs the ALT count (row-less trips charged as fouls)
    reg = P[P["period"] <= 2]
    lab = reg["def_fouls_lab"] >= 6
    alt = reg["def_fouls_alt"] >= 6
    labd = reg["def_fouls_lab"] >= 9
    altd = reg["def_fouls_alt"] >= 9
    eff = []
    for s, idx in reg.groupby("season").groups.items():
        eff.append({"season": int(s), "reg_possessions": len(idx),
                    "share_poss_bonus_label_differs": float((lab[idx] != alt[idx]).mean()),
                    "share_poss_dbl_bonus_label_differs": float((labd[idx] != altd[idx]).mean()),
                    "share_poss_def_count_differs": float((reg.loc[idx, "def_fouls_lab"]
                                                          != reg.loc[idx, "def_fouls_alt"]).mean())})
    return rl, pd.DataFrame(eff)


HOOPR_INERT = {"Substitution", "OfficialTVTimeOut", "ShortTimeOut", "RegularTimeOut", "Dead Ball Rebound",
               "Block Shot", "Jumpball"}


def hoopr_rowless() -> pd.DataFrame:
    """Second source: hoopR (ESPN) pbp. A trip = maximal run of MadeFreeThrow rows by one team over the non-inert
    rows (an Offensive Rebound between two FTs of the same team is administrative and skipped). Row-less = the
    non-inert row before the trip is neither PersonalFoul nor Technical Foul."""
    rows = []
    for s in SEASONS:
        h = pd.read_parquet(ROOT / f"data/raw/hoopr/pbp/play_by_play_{s}.parquet",
                            columns=["game_id", "sequence_number", "type_text", "team_id", "period_number"])
        h = h.drop_duplicates(["game_id", "sequence_number"]).sort_values(["game_id", "sequence_number"], kind="stable")
        h = h[~h["type_text"].isin(HOOPR_INERT)].reset_index(drop=True)
        g, ty, tm = h["game_id"].to_numpy(), h["type_text"].to_numpy(), h["team_id"].to_numpy()
        ft = ty == "MadeFreeThrow"
        prv_ft = np.r_[False, ft[:-1] & (g[1:] == g[:-1])]
        nxt_ft = np.r_[ft[1:] & (g[1:] == g[:-1]), False]
        same_prev = np.r_[False, tm[1:] == tm[:-1]]
        same_next = np.r_[tm[:-1] == tm[1:], False]
        admin = (ty == "Offensive Rebound") & prv_ft & nxt_ft & same_prev & same_next
        h = h[~admin].reset_index(drop=True)
        g, ty, tm = h["game_id"].to_numpy(), h["type_text"].to_numpy(), h["team_id"].to_numpy()
        ft = ty == "MadeFreeThrow"
        cont = np.r_[False, ft[1:] & ft[:-1] & (g[1:] == g[:-1]) & (tm[1:] == tm[:-1])]
        start = ft & ~cont
        trip_id = np.cumsum(start)
        tl = pd.Series(trip_id[ft]).value_counts().sort_index()
        st_idx = np.flatnonzero(start)
        prev_ty = np.where(st_idx > 0, ty[np.maximum(st_idx - 1, 0)], "")
        prev_same_game = np.where(st_idx > 0, g[np.maximum(st_idx - 1, 0)] == g[st_idx], False)
        has_row = prev_same_game & np.isin(prev_ty, ["PersonalFoul", "Technical Foul"])
        tech = prev_same_game & (prev_ty == "Technical Foul")
        n_ft = tl.to_numpy()
        two = (n_ft == 2) & ~tech
        ng = int(pd.unique(g).size)
        rows.append({"season": s, "games": ng, "trips_nontech": int((~tech).sum()), "two_shot_trips": int(two.sum()),
                     "two_shot_rowless": int((two & ~has_row).sum()),
                     "two_shot_rowless_rate": float((two & ~has_row).sum() / max(two.sum(), 1)),
                     "all_trips_rowless_rate": float((~has_row & ~tech).sum() / max((~tech).sum(), 1)),
                     "rowless_trips_per_game": float((~has_row & ~tech).sum() / ng),
                     "technical_rows_per_game": float((ty == "Technical Foul").sum() / ng)})
    return pd.DataFrame(rows)


def bonus_onset(P: pd.DataFrame, S: pd.DataFrame, ids500: set) -> pd.DataFrame:
    """Per (game, half, offence side): seconds remaining at the first possession the offence starts in the bonus."""
    rows = []
    r = P[(P["period"] <= 2) & (P["season"] == 2025) & P["game_id"].isin(ids500)]
    s = S[S["period"] <= 2]
    for src, d, key, flag, clk in (
            ("real_2025_same500_label", r, ["game_id", "period", "off_side"], r["def_fouls_lab"] >= 6, "start_clock"),
            ("real_2025_same500_alt_rowless_counted", r, ["game_id", "period", "off_side"], r["def_fouls_alt"] >= 6,
             "start_clock"),
            ("sim_servedv3_500x125", s, ["game_id", "seed", "period", "off_side"], s["off_in_bonus"] == 1,
             "clock_start")):
        d = d.assign(_b=flag.to_numpy())
        halves = d.groupby(key).size().rename("n").reset_index()
        on = d[d["_b"]].groupby(key)[clk].max().rename("onset").reset_index()
        h = halves.merge(on, on=key, how="left")
        for per in (1, 2):
            hp = h[h["period"] == per]
            rows.append({"source": src, "half": per, "team_halves": len(hp),
                         "share_reaching_bonus": float(hp["onset"].notna().mean()),
                         "mean_onset_sec_remaining": float(hp["onset"].mean()),
                         "median_onset_sec_remaining": float(hp["onset"].median())})
    return pd.DataFrame(rows)


def period_end(P: pd.DataFrame, S: pd.DataFrame, ids500: set) -> pd.DataFrame:
    rows = []
    r = P[P["period"] <= 2]
    for src, d in (("real_2022-2025_all", r), ("real_2025_same500", r[r["game_id"].isin(ids500)])):
        ng = d["game_id"].nunique()
        last = d.sort_values(["game_id", "period", "poss_index"]).groupby(["game_id", "period"]).tail(1)
        rows.append({"source": src, "games": ng, "poss_per_game": len(d) / ng,
                     "end_period_noevent_per_game": (d["terminal"] == "end_period").sum() / ng,
                     "unknown_terminal_per_game": (d["terminal"] == "unknown").sum() / ng,
                     "zero_duration_poss_per_game": (d["duration_s"] == 0).sum() / ng,
                     "poss_starting_le2s_per_game": (d["start_clock"] <= 2).sum() / ng,
                     "poss_starting_le5s_per_game": (d["start_clock"] <= 5).sum() / ng,
                     "periods_last_poss_ends_gt0s_per_game": (last["end_clock"] > 0).sum() / ng,
                     "mean_unlogged_tail_s_when_gt0": float(last.loc[last["end_clock"] > 0, "end_clock"].mean()),
                     "horn_ending_poss_per_game": (d["end_clock"] <= 0).sum() / ng})
    s = S[S["period"] <= 2]
    ng = s[["game_id", "seed"]].drop_duplicates().shape[0]
    last = s.sort_values(["game_id", "seed", "poss_idx"]).groupby(["game_id", "seed", "period"]).tail(1)
    noev = (s["fga"] == 0) & (s["fta"] == 0) & (s["tov"] == 0)
    rows.append({"source": "sim_servedv3_500x125", "games": ng, "poss_per_game": len(s) / ng,
                 "end_period_noevent_per_game": noev.sum() / ng, "unknown_terminal_per_game": 0.0,
                 "zero_duration_poss_per_game": (s["duration_s"] == 0).sum() / ng,
                 "poss_starting_le2s_per_game": (s["clock_start"] <= 2).sum() / ng,
                 "poss_starting_le5s_per_game": (s["clock_start"] <= 5).sum() / ng,
                 "periods_last_poss_ends_gt0s_per_game": (last["clock_end"] > 0).sum() / ng,
                 "mean_unlogged_tail_s_when_gt0": float(last.loc[last["clock_end"] > 0, "clock_end"].mean())
                 if (last["clock_end"] > 0).any() else np.nan,
                 "horn_ending_poss_per_game": (s["clock_end"] <= 0).sum() / ng})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# C: phantom / duplicate censored records on the served clock's training tables
# ----------------------------------------------------------------------------------------------------------------
def dup_checks(P: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(table, check, n_rows, n_bad, note=""):
        rows.append({"table": table, "check": check, "rows": int(n_rows), "count": int(n_bad), "note": note})

    for v in ("v2", "v4"):
        for s in SEASONS:
            p = pd.read_parquet(ROOT / f"data/processed/possessions_{v}/possessions_{s}.parquet",
                                columns=["game_id", "period", "poss_index", "end_clock", "terminal_event"])
            c = pd.read_parquet(ROOT / f"data/processed/possessions_{v}/chances_{s}.parquet",
                                columns=["game_id", "period", "poss_index", "chance_number", "terminal_event", "end_clock"])
            t = f"possessions_{v}/{s}"
            add(t, "duplicate (game_id, period, poss_index)", len(p), p.duplicated(["game_id", "period", "poss_index"]).sum())
            add(t, "duplicate (game_id, poss_index)", len(p), p.duplicated(["game_id", "poss_index"]).sum())
            ep = p[p["terminal_event"] == "end_period"].groupby(["game_id", "period"]).size()
            add(t, "(game, period) with >1 end_period-censored possession", len(ep), (ep > 1).sum())
            hn = p[p["end_clock"] <= 0].groupby(["game_id", "period"]).size()
            add(t, "(game, period) with >1 horn-ending (end_clock<=0) possession", len(hn), (hn > 1).sum(),
                "legit when a 0-s trip/putback follows a 0-s close; inspected below")
            add(f"chances_{v}/{s}", "duplicate (game_id, period, poss_index, chance_number)", len(c),
                c.duplicated(["game_id", "period", "poss_index", "chance_number"]).sum())
            cc = c[c["terminal_event"] == "end_period"].groupby(["game_id", "poss_index"]).size()
            add(f"chances_{v}/{s}", "possessions with >1 end_period chance row", len(cc), (cc > 1).sum())
            nonlast = c.merge(c.groupby(["game_id", "poss_index"])["chance_number"].max().rename("mx").reset_index(),
                              on=["game_id", "poss_index"])
            add(f"chances_{v}/{s}", "end_period chance row that is NOT the possession's last chance",
                len(nonlast), ((nonlast["terminal_event"] == "end_period") & (nonlast["chance_number"] < nonlast["mx"])).sum())
    # the served clock's own design + censoring side tables
    dz = ROOT / "data/processed/models/clock/r6_L2"
    d = pd.read_parquet(dz / "design_v2.parquet", columns=["season", "game_id", "period", "poss_index", "censored",
                                                           "terminal_event", "duration_s", "seconds_remaining"])
    d = d[d["season"].isin(SEASONS)]
    add("clock r6_L2/design_v2", "duplicate (game_id, period, poss_index)", len(d),
        d.duplicated(["game_id", "period", "poss_index"]).sum())
    add("clock r6_L2/design_v2", "censored rows", len(d), d["censored"].sum(), "rows=all design rows 2022-2025")
    ce = d[d["censored"]].groupby(["game_id", "period"]).size()
    add("clock r6_L2/design_v2", "(game, period) with >1 censored row", len(ce), (ce > 1).sum())
    side = pd.concat([pd.read_parquet(dz / f"clock_censoring/censoring_v1_{s}.parquet") for s in SEASONS],
                     ignore_index=True)
    add("clock r6_L2/clock_censoring", "duplicate (game_id, period, poss_index)", len(side),
        side.duplicated(["game_id", "period", "poss_index"]).sum())
    add("clock r6_L2/clock_censoring", "censored_horn rows", len(side), side["censored_horn"].sum())
    ch = side[side["censored_horn"]].groupby(["game_id", "period"]).size()
    add("clock r6_L2/clock_censoring", "(game, period) with >1 censored_horn row", len(ch), (ch > 1).sum(),
        "a censored_horn row is a possession that consumed the clock; >1 only if two possessions both end at 0 s")
    m = d.merge(side[["game_id", "period", "poss_index", "censored_horn"]], on=["game_id", "period", "poss_index"],
                how="left", validate="1:1")
    add("clock r6_L2/design_v2 x censoring", "design rows with no censoring row", len(m), m["censored_horn"].isna().sum())
    # replay itself (one record per possession by construction)
    add("this replay (v4 switches)", "duplicate (game_id, poss_index)", len(P), P.duplicated(["game_id", "poss_index"]).sum())
    ep = P[P["terminal"] == "end_period"].groupby(["game_id", "period"]).size()
    add("this replay (v4 switches)", "(game, period) with >1 end_period possession", len(ep), (ep > 1).sum())
    return pd.DataFrame(rows)


def horn_multi_examples(P: pd.DataFrame) -> pd.DataFrame:
    h = P[P["end_clock"] <= 0]
    k = h.groupby(["game_id", "period"]).size()
    multi = k[k > 1].index
    hm = h.set_index(["game_id", "period"]).loc[multi].reset_index()
    return hm.groupby("terminal").agg(n=("poss_index", "size"), mean_dur=("duration_s", "mean")).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-real", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    P, T, D = build_real(args.force_real)
    ids500 = set(pd.read_parquet(IDS500)["game_id"].astype("int64"))
    log(f"real ready: {len(P):,} poss; loading sim trajectories")
    S = load_sim()
    log(f"sim: {len(S):,} possessions, {S[['game_id', 'seed']].drop_duplicates().shape[0]:,} game-seeds")
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    out = {}
    out["engine_lategame_cells"] = lategame(P, S, ids500)
    rl, eff = rowless(T, P, D)
    out["engine_rowless_ft_trips"] = rl
    out["engine_rowless_bonus_label_effect"] = eff
    out["engine_rowless_ft_trips_hoopr"] = hoopr_rowless()
    out["engine_bonus_onset"] = bonus_onset(P, S, ids500)
    out["engine_period_end_rows"] = period_end(P, S, ids500)
    out["engine_fitwalker_duplicates"] = dup_checks(P)
    out["engine_horn_multi_by_terminal"] = horn_multi_examples(P)
    for k, v in out.items():
        f = OUT_CSV / f"{k}.csv"
        v.to_csv(f, index=False, float_format="%.4f")
        (SCRATCH / f"{k}.csv").write_text(f.read_text())
        log(f"wrote {f.name} ({f.stat().st_size:,} B)")
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(v.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
