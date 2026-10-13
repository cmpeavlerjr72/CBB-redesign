"""grade_endgame_v1.py -- the ONE blind grader of late-game round 7 (endgame margin/clock/foul),
`docs/models/late_game/experiments.md` section 15 (PROMOTED 2026-10-09; PM rulings 2026-10-13 Q1-Q3).

Written and COMMITTED BEFORE any arm of the round was fitted. Every arm of a block goes through the same code path;
arm identity is hidden behind coded labels (seeded shuffle) until the coded table has been written to disk, then the
key is written and the decision rule (which needs the simplicity ranks) is applied.

=====================================================================================================================
MODE `offline`  (s15.A.3-4; fold 2 selects, fold 1 confirms; 2025-26 sealed)
=====================================================================================================================
Inputs: `results/endgame/offline/<fold>/<block>/<arm>.npz`, one file per arm, written by the trainers
(`train_endgame_{clock,event,foul}_v1.py`). Keys `game_id, period, poss_index` must cover the block's test-season
window rows exactly (`build_endgame_window_v1`). Arrays: C `pmf` (n, 91) first-chance duration law and `pmf_end`
(n, 91) the same law conditional on the realised chance-1 end class (K2's re-draw law); E `prob` (n, 6) in
`possession_outcome.CLASSES` order; F `prob` (n, 4) for min(K, 3).

Block C (clock), rows = window rows with possession start <= 240 s; SELECTION on period 2, period 1 REPORTED.
  primary   `dev`  held-out first-chance duration deviance, -2 x mean log-lik (log p(y) uncensored, log P(T >= y)
            censored, clipped at 1e-12), as clock sections 32/37.
  co-prim.  `cal`  mean over the POWERED role-band x time-bucket cells (n >= 60 test rows) of
            |mean model E[min(T, R)] - mean actual seconds| (R = seconds left at the start). Underpowered cells are
            listed and excluded. Both `dev` and `cal` must beat the control by more than one floor ("PLUS").
  slopes    (a) `slope_lead`, `slope_trail`: OLS slope of the 4 model band means on the 4 actual band means of
            seconds per possession (bands 0-3/4-7/8-12/13+), leaders and trailers separately, target 1.
            HOLDS if |slope - 1| is not worse than the control's by more than its floor.
            (b) `slope_tempo`: the same across offence TEMPO quintiles (as-of `off_tempo_rel`, the pregame value
            relative to the league mean), all window rows. HOLDS if not below the control's by more than its floor.
            (s15.0) `slope_team_lead`, `slope_team_trail`: across offence PRIOR-STRENGTH quintiles (as-of
            `off_rating_off_c`, relative to its snapshot's league mean), leaders / trailers. HOLD as (b).
  reported  `dev_end` (conditional law on the realised end class), per-cell table, period-1 lines.
Block E (event), rows = window FIRST chances (`population == 'first'`); selection period 2.
  primary   `mll` six-class log loss (clip 1e-12).
  slopes    (c) `slope_tov_lead`: OLS slope of model mean P(TOV) on actual TOV rate across the 4 leader bands.
            HOLDS as (a). Reported: team prior-quintile TOV slope, TOV / FT / 3PA calibration by role-band.
Block F (foul state), rows = section 29's window (period 2, start <= 120 s, regulation).
  primary   `mll` multinomial log loss of min(K, 3), every arm's probabilities floored at 1e-4 and renormalised
            (section 29.2). Reported: binary (K >= 1) log loss, E[K] by cell.
  slope     (d) E[K] by defence foul count must FALL from 4-5 to 6-8 (and 9+), as in the data: HOLDS if the arm's
            E[K | 6-8] < E[K | 4-5].

Floors (s15.A.4): offline fits are deterministic (reseed floor 0, proved by the `*s1` noise-floor arm, which is a
spec-identical refit under seed 1 and is NEVER selectable); the binding floor of every line is 2 x the paired
game-block bootstrap SE of (arm - control): 200 draws over test-season games, numpy seed 12345.

Decision (s15.A.4 and section 29.2, applied literally):
  C, E  an arm WINS if on F2 it beats the control by more than one floor on the primary (C: `dev` AND `cal`), every
        mandatory slope HOLDS, and F1 keeps the sign of the primary delta(s). Among winners the simplest (lowest
        rank) whose primary is within one floor of the best winner's wins.
  F     the simplest arm that beats LF0 by more than one floor on F2 with the same sign on F1; LF2 over LF1 only if
        LF2 beats LF1 by more than one floor; (d) must hold.

=====================================================================================================================
MODE `loop`  (closed loop, s15.A.3-4 and the mandatory gates MD1-MD6 of s15.1)
=====================================================================================================================
Runs: `scripts/run_engine_window_v2.py` on the 500-game verified stride sample
(`data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet`), served stack v3 + the arm's default-off
flags, paired seeds (the arm and the control share seeds 0..S-1), trajectories for every seed
(`results/trajectories/<tag>/trajectory.parquet`), games in `results/endgame/loop/<tag>/games.parquet`.
Control floor draws = four control runs at disjoint seed offsets (1000, 2000, 3000, 4000).

Real side: verified finals (`CBB_TRUTH=verified_v1`) for game-level lines; the possession-table path
(`diag_c4_lib_v1.std_from_real`, possessions_v4otc) on the sample games whose pbp final equals the verified final for
window lines (like-for-like with the engine definitions).

Floor per line = max(2 x SD of the four control draws, 2 x paired game-bootstrap SE of (arm - control)); the
bootstrap is 200 draws over sample games, numpy seed 12345.

Seed-count check (s15.0): a line's floor is RESOLVED at S seeds when its floor < |control - target|, i.e. a move
that closes the gap can register. If any block primary or MD line is unresolved at 25 seeds, every closed-loop run
is repeated at 50 seeds and the 50-seed read is the one used (relabelled `s50`).

Block primaries (toward the actual by more than one floor):
  C  `md4_h2_last4_poss` (H2 possessions ENDING with <= 240 s left, both teams, per game; MD4) and co-primary
     `md5_pts13_last2` (H2 points per game in possessions starting <= 120 s, games with |home margin| >= 13 at the
     2:00 reference; MD5). Both are MD gates.
  F  `tie_rate` (regulation tie = OT rate, MD3). Reported with floors: P0/P1, leader bonus occupancy (offence
     leading 1-3 / 4-6, possessions starting <= 120 s in H2), trailing-team fouls per leader possession
     (trailing 1-3 / 4-6 / 7+; the engine definition on both sides: defence team-foul counter change).
  E  leader TOV rate in the 13+ state at the 4:00 and 2:00 references (D3 (b) definition); the 8-12 lines (already
     on target) are move-away vetoes.
Hard vetoes (any regression beyond its floor vs the control fails the arm; "regression" = |arm - target| exceeds
|control - target| by more than the floor): G1 possession mean and SD; first-half and second-half points share;
window possessions by k (round-3/4 reading: k = |home margin| at the first H2 possession starting <= 120 s, 0-6;
window = those possessions with |margin| <= 6; read against the actual unless the control already exceeds it by more
than a floor, then against the control); G5 margin and total SD ratios (target 1); G9 margin bias (target 0) and
MC-corrected slope (target 1); first-half buzzer test (H1 PPP of possessions starting <= 10 s must not move away from
the actual; bit-identical first halves pass trivially); D1 home margin on non-neutral games (sim - real) and the
eFG home-minus-away gap; D2 game-possession slopes on `env` and `gap`, total slope on `env`, lead-change slope on
|spread| (read as move-away vetoes against the real slope); OT / tie rate must not move away from the actual.
G9 total bias is REPORTED as priced exposure by half (H1, H2+OT), not a disqualifier.
Mandatory gates (s15.1), every candidate must pass all six:
  MD1  2 Cov(margin minutes 0-36, last-4:00 increment): sim within-game across seeds vs realised (real - sim mean
       products, MC-corrected by 1/S). The gap must SHRINK by more than its floor; a move away is a veto, no move a
       fail.
  MD2  sim/realised final-margin SD ratio by |spread| quintile (ESPN BET close, quintile edges over every fold-2
       game with a close): must not RISE beyond its floor in any quintile and must FALL in Q1 beyond its floor.
  MD3  OT rate toward the actual (verified finals of the sample) beyond the floor, and never below 3.5%.
  MD4  H2 minutes 36-40 possession count (M-end, both teams) toward the real count beyond the floor.
  MD5  last-2-minute points in the 13+-at-2:00 state toward real beyond the floor.
  MD6  overall margin SD ratio (sim own SD / MC-corrected realised residual SD) must not rise beyond its floor;
       Var(minutes 0-36) must not rise beyond its floor; Var(last 4:00) reported.
Decision (s15.A.4): a block winner or composition is a CANDIDATE if its block primary moves toward the actual by more
than one floor, every hard veto passes and MD1-MD6 pass; otherwise the clear best is the arm that passes every veto
and beats every other veto-passing arm by more than one floor on the primary; otherwise NO ARM ADOPTED.

    .venv/Scripts/python.exe scripts/grade_endgame_v1.py offline --block C
    .venv/Scripts/python.exe scripts/grade_endgame_v1.py loop --control eg_ctrl_o0_s25 \
        --draws eg_ctrl_o1000_s25 eg_ctrl_o2000_s25 eg_ctrl_o3000_s25 eg_ctrl_o4000_s25 --arms eg_LF2_o0_s25 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("CBB_TRUTH", "verified_v1")

import build_endgame_window_v1 as W  # noqa: E402

OFF_DIR = ROOT / "results/endgame/offline"
LOOP_DIR = ROOT / "results/endgame/loop"
TRAJ_DIR = ROOT / "results/trajectories"
GRADE_DIR = ROOT / "results/endgame/grade"
SAMPLE = ROOT / "data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet"
N_BOOT, BOOT_SEED, CODE_SEED = 200, 12345, 20261013
EPS_C, EPS_E, EPS_F = 1e-12, 1e-12, 1e-4
MIN_N = 60
GRID = np.arange(91, dtype=np.float64)

#: simplicity ranks (s15.A.2). Noise-floor arms are never selectable.
RANK = {"GT0": 0, "GT2": 1, "GT1": 2, "EV0": 0, "EV1": 1, "LF0": 0, "LF1": 1, "LF2": 2}
CONTROL = {"C": "GT0", "E": "EV0", "F": "LF0"}
NOISE = {"C": "GT0s1", "E": "EV0s1", "F": "LF0s1"}


# =====================================================================================================================
# shared helpers
# =====================================================================================================================
def boot_weights(n_games: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.empty((n_boot, n_games))
    for b in range(n_boot):
        w[b] = np.bincount(rng.integers(0, n_games, n_games), minlength=n_games)
    return w


def gsum(gi: np.ndarray, key: np.ndarray, vals: np.ndarray, G: int, K: int) -> np.ndarray:
    """(G, K) per-game x per-key sums."""
    out = np.zeros(G * K)
    np.add.at(out, gi * K + key, vals)
    return out.reshape(G, K)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.var(x[ok]) == 0:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def quintile(v: np.ndarray) -> np.ndarray:
    e = np.nanquantile(v, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(e, v, side="right").astype(np.int64)


def code_arms(block: str, arms: list[str]) -> dict[str, str]:
    """arm -> blind code. The control keeps the label CTRL (every delta is read against it)."""
    rest = sorted(a for a in arms if a != CONTROL.get(block))
    perm = np.random.default_rng(CODE_SEED + ord(block[0])).permutation(len(rest))
    out = {CONTROL[block]: "CTRL"} if CONTROL.get(block) in arms else {}
    for j, i in enumerate(perm):
        out[rest[i]] = f"{block}-{chr(65 + j)}"
    return out


def fmt(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


# =====================================================================================================================
# offline
# =====================================================================================================================
def truth(block: str, fold: str) -> pd.DataFrame:
    if block == "C":
        _, te = W.split(W.clock_window(), fold)
        t = pd.DataFrame({"game_id": te["game_id"].to_numpy(np.int64), "period": te["period"].to_numpy(np.int64),
                          "poss_index": te["poss_index"].to_numpy(np.int64),
                          "y": np.clip(te["duration_s"].to_numpy(np.int64), 0, 90),
                          "cen": te["censored"].to_numpy(bool), "R": te["seconds_remaining"].to_numpy(float),
                          "m": te["score_diff"].to_numpy(np.int64), "end1": te["end1_code"].to_numpy(np.int64),
                          "tempo": te["off_tempo_rel"].to_numpy(float),
                          "strength": te["off_rating_off_c"].to_numpy(float)})
        t["sec"] = t["R"]
    elif block == "E":
        _, te = W.split(W.event_window(), fold)
        t = pd.DataFrame({"game_id": te["game_id"].to_numpy(np.int64), "period": te["period"].to_numpy(np.int64),
                          "poss_index": te["poss_index"].to_numpy(np.int64), "y": te["y"].to_numpy(np.int64),
                          "m": te["start_score_diff"].to_numpy(np.int64), "sec": te["start_clock"].to_numpy(float),
                          "strength": te["off_rating_off_c"].to_numpy(float)})
    else:
        _, te = W.split(W.foul_window(), fold)
        t = pd.DataFrame({"game_id": te["game_id"].to_numpy(np.int64), "period": te["period"].to_numpy(np.int64),
                          "poss_index": te["poss_index"].to_numpy(np.int64),
                          "k": np.clip(te["def_silent"].to_numpy(np.int64), 0, 3),
                          "m": te["start_score_diff"].to_numpy(np.int64), "sec": te["start_clock"].to_numpy(float),
                          "dfl": te["def_team_fouls_true"].to_numpy(np.int64)})
    t["rb"] = W.roleband(t["m"].to_numpy())
    t["tb"] = W.tbucket(t["sec"].to_numpy())
    return t


def load_pred(block: str, fold: str, arm: str, t: pd.DataFrame) -> dict:
    z = np.load(OFF_DIR / fold / block / f"{arm}.npz", allow_pickle=False)
    k = pd.DataFrame({"game_id": z["game_id"].astype(np.int64), "period": z["period"].astype(np.int64),
                      "poss_index": z["poss_index"].astype(np.int64), "_r": np.arange(len(z["game_id"]))})
    j = t[["game_id", "period", "poss_index"]].merge(k, on=["game_id", "period", "poss_index"], how="left",
                                                     validate="1:1")
    if j["_r"].isna().any() or len(k) != len(t):
        raise SystemExit(f"{fold}/{block}/{arm}: prediction rows do not cover the test window exactly")
    r = j["_r"].to_numpy(np.int64)
    out = {kk: np.asarray(z[kk])[r] for kk in z.files if kk not in ("game_id", "period", "poss_index", "meta")}
    out["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return out


def offline_rows(block: str, t: pd.DataFrame, P: dict) -> dict:
    """Per-row quantities the lines are built from."""
    if block == "C":
        p = np.asarray(P["pmf"], float)
        y, cen, R = t["y"].to_numpy(), t["cen"].to_numpy(), t["R"].to_numpy()
        ii = np.arange(len(y))
        pu = p[ii, y]
        S = 1.0 - np.cumsum(p, axis=1)[ii, y] + pu
        ll = np.log(np.clip(np.where(cen, S, pu), EPS_C, None))
        e = (p * np.minimum(GRID[None, :], R[:, None])).sum(axis=1)
        out = {"ll": ll, "e": e}
        if "pmf_end" in P:
            pe = np.asarray(P["pmf_end"], float)
            pue = pe[ii, y]
            Se = 1.0 - np.cumsum(pe, axis=1)[ii, y] + pue
            out["ll_end"] = np.log(np.clip(np.where(cen, Se, pue), EPS_C, None))
        return out
    if block == "E":
        p = np.asarray(P["prob"], float)
        y = t["y"].to_numpy()
        return {"ll": np.log(np.clip(p[np.arange(len(y)), y], EPS_E, None)), "p": p}
    p = np.clip(np.asarray(P["prob"], float), EPS_F, None)
    p = p / p.sum(1, keepdims=True)
    k = t["k"].to_numpy()
    pb = np.where(k >= 1, 1 - p[:, 0], p[:, 0])
    return {"ll": np.log(p[np.arange(len(k)), k]), "llb": np.log(np.clip(pb, EPS_F, 1)),
            "ek": (p * np.arange(4)).sum(1), "p": p}


class OfflineLines:
    """Line values from per-game sums; `w` (G,) bootstrap multiplicities (ones = the full sample)."""

    def __init__(self, block: str, t: pd.DataFrame, period: int | None):
        self.block = block
        sel = np.ones(len(t), bool) if period is None else (t["period"].to_numpy() == period)
        self.sel = sel
        tt = t[sel].reset_index(drop=True)
        self.t = tt
        g, self.gi = np.unique(tt["game_id"].to_numpy(), return_inverse=True)
        self.G = len(g)
        one = np.ones(len(tt))
        self.n_g = np.bincount(self.gi, minlength=self.G).astype(float)
        if block == "C":
            y = tt["y"].to_numpy(float)
            rb, tb = tt["rb"].to_numpy(), tt["tb"].to_numpy()
            self.cell = rb * W.N_TB + tb
            self.cell_n = gsum(self.gi, self.cell, one, self.G, 45)
            self.cell_y = gsum(self.gi, self.cell, y, self.G, 45)
            self.powered = self.cell_n.sum(0) >= MIN_N
            self.rb_n = gsum(self.gi, rb, one, self.G, 9)
            self.rb_y = gsum(self.gi, rb, y, self.G, 9)
            self.tq = quintile(tt["tempo"].to_numpy())
            self.tq_n = gsum(self.gi, self.tq, one, self.G, 5)
            self.tq_y = gsum(self.gi, self.tq, y, self.G, 5)
            role = W.role(tt["m"].to_numpy())
            sq = quintile(tt["strength"].to_numpy())
            self.sk = np.where(role > 0, sq, np.where(role < 0, 5 + sq, 10))
            self.sk_n = gsum(self.gi, self.sk, one, self.G, 11)
            self.sk_y = gsum(self.gi, self.sk, y, self.G, 11)
        elif block == "E":
            y = tt["y"].to_numpy()
            rb = tt["rb"].to_numpy()
            self.rb = rb
            self.rb_n = gsum(self.gi, rb, one, self.G, 9)
            self.rb_tov = gsum(self.gi, rb, (y == 0).astype(float), self.G, 9)
            role = W.role(tt["m"].to_numpy())
            sq = quintile(tt["strength"].to_numpy())
            self.sk = np.where(role > 0, sq, np.where(role < 0, 5 + sq, 10))
            self.sk_n = gsum(self.gi, self.sk, one, self.G, 11)
            self.sk_tov = gsum(self.gi, self.sk, (y == 0).astype(float), self.G, 11)
        else:
            k = tt["k"].to_numpy(float)
            fi = np.searchsorted([3, 5, 8], tt["dfl"].to_numpy(), side="left").clip(0, 3)
            self.fi = fi
            self.f_n = gsum(self.gi, fi, one, self.G, 4)
            self.f_k = gsum(self.gi, fi, k, self.G, 4)

    def arm_sums(self, rows: dict) -> dict:
        s = {"ll": np.bincount(self.gi, weights=rows["ll"][self.sel], minlength=self.G)}
        if self.block == "C":
            e = rows["e"][self.sel]
            s["cell_e"] = gsum(self.gi, self.cell, e, self.G, 45)
            s["rb_e"] = gsum(self.gi, self.t["rb"].to_numpy(), e, self.G, 9)
            s["tq_e"] = gsum(self.gi, self.tq, e, self.G, 5)
            s["sk_e"] = gsum(self.gi, self.sk, e, self.G, 11)
            if "ll_end" in rows:
                s["ll_end"] = np.bincount(self.gi, weights=rows["ll_end"][self.sel], minlength=self.G)
        elif self.block == "E":
            ptov = rows["p"][self.sel][:, 0]
            s["rb_ptov"] = gsum(self.gi, self.rb, ptov, self.G, 9)
            s["sk_ptov"] = gsum(self.gi, self.sk, ptov, self.G, 11)
        else:
            s["llb"] = np.bincount(self.gi, weights=rows["llb"][self.sel], minlength=self.G)
            s["f_ek"] = gsum(self.gi, self.fi, rows["ek"][self.sel], self.G, 4)
            s["ek"] = np.bincount(self.gi, weights=rows["ek"][self.sel], minlength=self.G)
        return s

    def lines(self, s: dict, w: np.ndarray) -> dict:
        n = w @ self.n_g
        out = {}
        if self.block == "C":
            out["dev"] = -2.0 * (w @ s["ll"]) / n
            if "ll_end" in s:
                out["dev_end"] = -2.0 * (w @ s["ll_end"]) / n
            cn, cy, ce = w @ self.cell_n, w @ self.cell_y, w @ s["cell_e"]
            with np.errstate(invalid="ignore", divide="ignore"):
                gap = np.abs(ce / cn - cy / cn)
            out["cal"] = float(np.nanmean(gap[self.powered]))
            rn, ry, re_ = w @ self.rb_n, w @ self.rb_y, w @ s["rb_e"]
            with np.errstate(invalid="ignore", divide="ignore"):
                am, mm = ry / rn, re_ / rn
            out["slope_lead"] = ols_slope(am[0:4], mm[0:4])
            out["slope_trail"] = ols_slope(am[4:8], mm[4:8])
            tn, ty, te = w @ self.tq_n, w @ self.tq_y, w @ s["tq_e"]
            out["slope_tempo"] = ols_slope(ty / tn, te / tn)
            kn, ky, ke = w @ self.sk_n, w @ self.sk_y, w @ s["sk_e"]
            out["slope_team_lead"] = ols_slope(ky[0:5] / kn[0:5], ke[0:5] / kn[0:5])
            out["slope_team_trail"] = ols_slope(ky[5:10] / kn[5:10], ke[5:10] / kn[5:10])
        elif self.block == "E":
            out["mll"] = -(w @ s["ll"]) / n
            rn, rt, rp = w @ self.rb_n, w @ self.rb_tov, w @ s["rb_ptov"]
            out["slope_tov_lead"] = ols_slope(rt[0:4] / rn[0:4], rp[0:4] / rn[0:4])
            kn, kt, kp = w @ self.sk_n, w @ self.sk_tov, w @ s["sk_ptov"]
            out["slope_team_tov_lead"] = ols_slope(kt[0:5] / kn[0:5], kp[0:5] / kn[0:5])
        else:
            out["mll"] = -(w @ s["ll"]) / n
            out["bll"] = -(w @ s["llb"]) / n
            out["ek"] = (w @ s["ek"]) / n
            fn, fe = w @ self.f_n, w @ s["f_ek"]
            ek = fe / np.maximum(fn, 1e-12)
            out["ek_f45"], out["ek_f68"], out["ek_f9"] = float(ek[1]), float(ek[2]), float(ek[3])
        return {k: float(v) for k, v in out.items()}

    def actual(self) -> dict:
        w = np.ones(self.G)
        out = {}
        if self.block == "C":
            rn, ry = w @ self.rb_n, w @ self.rb_y
            out["band_mean_actual"] = {W.ROLEBAND_LABELS[j]: float(ry[j] / rn[j]) if rn[j] else None for j in range(9)}
            out["band_n"] = {W.ROLEBAND_LABELS[j]: int(rn[j]) for j in range(9)}
            out["underpowered_cells"] = [f"{W.ROLEBAND_LABELS[c // 5]} | {W.BUCKET_LABELS[c % 5]}"
                                         for c in np.flatnonzero(~self.powered) if self.cell_n.sum(0)[c] > 0]
        elif self.block == "E":
            rn, rt = w @ self.rb_n, w @ self.rb_tov
            out["tov_rate_actual"] = {W.ROLEBAND_LABELS[j]: float(rt[j] / rn[j]) if rn[j] else None for j in range(9)}
            out["band_n"] = {W.ROLEBAND_LABELS[j]: int(rn[j]) for j in range(9)}
        else:
            fn, fk = w @ self.f_n, w @ self.f_k
            out["ek_actual_by_fouls"] = {lab: float(fk[j] / fn[j]) if fn[j] else None
                                         for j, lab in enumerate(("0-3", "4-5", "6-8", "9+"))}
            out["mean_k"] = float(fk.sum() / fn.sum())
        return out


def offline_cell_table(block: str, t: pd.DataFrame, rows: dict, period: int) -> list[dict]:
    """Per role-band x time-bucket cell (C), per role-band (E), per section-29 cell (F): actual vs model."""
    sel = t["period"].to_numpy() == period
    out = []
    if block == "C":
        tt, e = t[sel], rows["e"][sel]
        for rb in range(9):
            for tb in range(5):
                m = (tt["rb"].to_numpy() == rb) & (tt["tb"].to_numpy() == tb)
                if not m.any():
                    continue
                out.append({"cell": f"{W.ROLEBAND_LABELS[rb]} | {W.BUCKET_LABELS[tb]}", "n": int(m.sum()),
                            "actual_s": float(tt["y"].to_numpy()[m].mean()), "model_s": float(e[m].mean()),
                            "underpowered": bool(m.sum() < MIN_N)})
    elif block == "E":
        tt, p = t[sel], rows["p"][sel]
        y = tt["y"].to_numpy()
        for rb in range(9):
            m = tt["rb"].to_numpy() == rb
            if not m.any():
                continue
            out.append({"cell": W.ROLEBAND_LABELS[rb], "n": int(m.sum()),
                        "tov": [float((y[m] == 0).mean()), float(p[m, 0].mean())],
                        "fga3": [float((y[m] == 3).mean()), float(p[m, 3].mean())],
                        "ft_trip": [float((y[m] >= 4).mean()), float(p[m, 4:].sum(1).mean())],
                        "underpowered": bool(m.sum() < MIN_N)})
    return out


def offline(block: str, arms: list[str] | None) -> int:
    folds = ("F2", "F1")
    found = sorted(p.stem for p in (OFF_DIR / "F2" / block).glob("*.npz"))
    arms = arms or found
    ctrl = CONTROL[block]
    if ctrl not in arms:
        raise SystemExit(f"block {block}: control {ctrl} has no prediction file")
    codes = code_arms(block, arms)
    periods = (2, 1) if block in ("C", "E") else (2,)
    res: dict = {"block": block, "codes_written_before_key": True, "folds": {}}
    for fold in folds:
        t = truth(block, fold)
        rows = {a: offline_rows(block, t, load_pred(block, fold, a, t)) for a in arms
                if (OFF_DIR / fold / block / f"{a}.npz").exists()}
        res["folds"][fold] = {"n_rows": int(len(t)), "periods": {}}
        for per in periods:
            L = OfflineLines(block, t, per)
            wb = boot_weights(L.G)
            one = np.ones(L.G)
            sums = {a: L.arm_sums(r) for a, r in rows.items()}
            val = {a: L.lines(s, one) for a, s in sums.items()}
            boot = {a: [L.lines(s, wb[b]) for b in range(N_BOOT)] for a, s in sums.items()}
            comp = {}
            for a in rows:
                if a == ctrl:
                    continue
                d = {}
                for k in val[a]:
                    db = np.array([boot[a][b][k] - boot[ctrl][b][k] for b in range(N_BOOT)])
                    se = float(np.nanstd(db, ddof=1))
                    d[k] = {"arm": val[a][k], "ctrl": val[ctrl][k], "delta": val[a][k] - val[ctrl][k],
                            "floor": 2 * se}
                    if k.startswith("slope_"):
                        da = [abs(boot[a][b][k] - 1) - abs(boot[ctrl][b][k] - 1) for b in range(N_BOOT)]
                        d[k]["d_abs_err"] = abs(val[a][k] - 1) - abs(val[ctrl][k] - 1)
                        d[k]["floor_abs_err"] = 2 * float(np.nanstd(da, ddof=1))
                comp[codes[a]] = d
            if block == "F" and "LF1" in rows and "LF2" in rows:
                db = np.array([boot["LF2"][b]["mll"] - boot["LF1"][b]["mll"] for b in range(N_BOOT)])
                comp["LF2_vs_LF1"] = {"delta": val["LF2"]["mll"] - val["LF1"]["mll"],
                                      "floor": 2 * float(np.std(db, ddof=1))}
            res["folds"][fold]["periods"][str(per)] = {
                "n_rows": int(L.sel.sum()), "n_games": int(L.G), "ctrl": val[ctrl],
                "coded": comp, "actual": L.actual(),
                "cells_ctrl": offline_cell_table(block, t, rows[ctrl], per) if block != "F" else None,
                "cells_by_code": ({codes[a]: offline_cell_table(block, t, rows[a], per) for a in rows if a != ctrl}
                                  if block != "F" else None)}
        if block == "F":
            res["folds"][fold]["ek_by_cell"] = f_cells(t, rows)
    GRADE_DIR.mkdir(parents=True, exist_ok=True)
    coded_path = GRADE_DIR / f"offline_{block}_coded.json"
    coded_path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"coded table written: {coded_path}", flush=True)
    # ---- unblind ----------------------------------------------------------------------------------------------
    key = {c: a for a, c in codes.items()}
    (GRADE_DIR / f"offline_{block}_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    dec = decide_offline(block, res, key)
    res["key"] = key
    res["decision"] = dec
    res["written_at"] = datetime.now(UTC).isoformat()
    (GRADE_DIR / f"offline_{block}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print_offline(block, res, key)
    return 0


def f_cells(t: pd.DataFrame, rows: dict) -> list[dict]:
    rc = np.where(t["m"].to_numpy() > 0, np.where(np.abs(t["m"]) <= 3, 0, np.where(np.abs(t["m"]) <= 6, 1, 2)),
                  np.where(t["m"].to_numpy() == 0, 3, 4))
    ci = np.searchsorted([30, 60, 120], t["sec"].to_numpy(), side="left").clip(0, 2)
    fi = np.searchsorted([3, 5, 8], t["dfl"].to_numpy(), side="left").clip(0, 3)
    out = []
    for ri, rn in enumerate(("def trail 1-3", "def trail 4-6", "def trail 7+", "tied", "def leading")):
        for c, cn in enumerate(("(0,30]", "(30,60]", "(60,120]")):
            for f, fname in enumerate(("0-3", "4-5", "6-8", "9+")):
                m = (rc == ri) & (ci == c) & (fi == f)
                if m.sum() < 20:
                    continue
                out.append({"cell": f"{rn}|{cn}|{fname}", "n": int(m.sum()), "act_K": float(t["k"].to_numpy()[m].mean()),
                            **{a: float(r["ek"][m].mean()) for a, r in rows.items()},
                            "underpowered": bool(m.sum() < 200)})
    return out


def decide_offline(block: str, res: dict, key: dict) -> dict:
    """The literal s15.A.4 / s29.2 rule on the unblinded table."""
    out = {"per_arm": {}}
    f2 = res["folds"]["F2"]["periods"]["2"]["coded"]
    f1 = res["folds"]["F1"]["periods"]["2"]["coded"] if "F1" in res["folds"] else {}
    prim = ("dev", "cal") if block == "C" else ("mll",)
    slopes = {"C": ("slope_lead", "slope_trail", "slope_tempo", "slope_team_lead", "slope_team_trail"),
              "E": ("slope_tov_lead",), "F": ()}[block]
    winners = []
    for code, d in f2.items():
        if code not in key:
            continue
        arm = key[code]
        beats = {p: bool(d[p]["delta"] < -d[p]["floor"]) for p in prim}
        hold = {}
        for s in slopes:
            x = d[s]
            if s in ("slope_lead", "slope_trail", "slope_tov_lead"):
                hold[s] = bool(x["d_abs_err"] <= x["floor_abs_err"])
            else:
                hold[s] = bool(x["delta"] >= -x["floor"])
        if block == "F":
            hold["d_ek_falls_4-5_to_6-8"] = bool(d["ek_f68"]["arm"] < d["ek_f45"]["arm"])
        f1sign = {p: (bool(np.sign(f1[code][p]["delta"]) == np.sign(d[p]["delta"])) if code in f1 else None)
                  for p in prim}
        noise = arm == NOISE[block]
        win = (not noise) and all(beats.values()) and all(hold.values()) and all(v is not False for v in f1sign.values())
        out["per_arm"][arm] = {"code": code, "beats_ctrl_by_floor": beats, "slopes_hold": hold,
                               "f1_keeps_sign": f1sign, "noise_floor_arm": noise, "wins": bool(win),
                               "primary_floors": {p: (-d[p]["delta"] / d[p]["floor"]) if d[p]["floor"] else None
                                                  for p in prim}}
        if noise:
            out["noise_floor"] = {p: {"delta": d[p]["delta"], "floor": d[p]["floor"]} for p in prim}
        if win:
            winners.append(arm)
    if block == "F":
        rule = "s29.2: simplest arm beating LF0 by > 1 floor on F2, same sign on F1; LF2 over LF1 only if LF2 beats LF1 by > 1 floor"
        sel = None
        if "LF1" in winners:
            sel = "LF1"
            v = res["folds"]["F2"]["periods"]["2"]["coded"].get("LF2_vs_LF1")
            if "LF2" in winners and v and v["delta"] < -v["floor"]:
                sel = "LF2"
        elif "LF2" in winners:
            sel = "LF2"
        out.update({"winners": winners, "selected": sel, "rule": rule})
        return out
    sel = None
    if winners:
        p0 = prim[0]
        best = min(winners, key=lambda a: f2[out["per_arm"][a]["code"]][p0]["arm"])
        bd = f2[out["per_arm"][best]["code"]]
        ok = [a for a in winners if all(f2[out["per_arm"][a]["code"]][p]["arm"] - bd[p]["arm"] <= bd[p]["floor"]
                                        for p in prim)]
        sel = min(ok, key=lambda a: RANK[a])
    out.update({"winners": winners, "selected": sel,
                "rule": "s15.A.4: beats control by > 1 floor on the F2 primary, all mandatory slopes hold, F1 keeps the "
                        "sign; simplest winner whose primary is within one floor of the best"})
    return out


def print_offline(block: str, res: dict, key: dict) -> None:
    for fold, fr in res["folds"].items():
        for per, pr in fr["periods"].items():
            print(f"== block {block} {fold} period {per}: {pr['n_rows']:,} rows, {pr['n_games']:,} games; control "
                  + " ".join(f"{k}={fmt(v)}" for k, v in pr["ctrl"].items()))
            for code, d in sorted(pr["coded"].items()):
                arm = key.get(code, code)
                if not isinstance(next(iter(d.values()), None), dict):
                    print(f"   {code:10s} {arm:6s} " + json.dumps(d))
                    continue
                print(f"   {code:10s} {arm:6s} " + " ".join(
                    f"{k}={fmt(x['arm'])} (d {x['delta']:+.4f} fl {x['floor']:.4f})" for k, x in d.items()))
    print("decision:", json.dumps(res["decision"], indent=1, default=float))


# =====================================================================================================================
# closed loop
# =====================================================================================================================
def _real_side(ids: np.ndarray) -> dict:
    import diag_c4_lib_v1 as L

    from cbb_sim.eval import reference as REF
    r = L.std_from_real(ids)
    rgs = L.game_seed(r).set_index("game_id")
    act = REF.load_actual_games(2025).merge(REF.load_actual_possessions(2025), on="game_id", how="left")
    act = act.set_index("game_id").reindex(ids)
    fin = pd.read_parquet(ROOT / "data/processed/truth/game_finals_v2.parquet").set_index("game_id").reindex(ids)
    ok = (rgs["final_margin"].reindex(ids).to_numpy() == (fin["home_score"] - fin["away_score"]).to_numpy())
    return {"std": r, "gs": rgs, "act": act, "fin": fin, "pbp_ok": pd.Series(ok, index=ids)}


def _pregame(ids: np.ndarray) -> pd.DataFrame:
    """Pregame ratings exactly as the D2 diagnostic used them: the fold-2 all-games extract (rating sign fitted on all
    5,710 games), restricted to the sample. Rebuilt with the same function if the extract is missing."""
    p = ROOT / "results/diag_c4/f2all50_pregame.parquet"
    if p.exists():
        pg = pd.read_parquet(p).set_index("game_id").reindex(ids)
    else:
        import diag_trajectory_vs_pbp_v1 as TV
        pg = TV.pregame_tables(set(int(g) for g in ids), 2025).set_index("game_id").reindex(ids)
        u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet", columns=["game_id", "neutral_site"])
        pg = pg.join(u.set_index("game_id"), rsuffix="_u")
    ln = pd.read_parquet(ROOT / "data/processed/lines/lines_close_v2_verified.parquet")
    ln = ln[(ln["season"] == 2025) & (ln["provider"] == "ESPN BET")]
    edges = np.nanquantile(ln["close_spread_home"].abs().to_numpy(), [0.2, 0.4, 0.6, 0.8])
    sp = ln.drop_duplicates("game_id").set_index("game_id")["close_spread_home"].reindex(ids)
    pg["abs_spread"] = sp.abs().to_numpy()
    pg["spread_q"] = np.where(sp.isna(), -1, np.searchsorted(edges, sp.abs().to_numpy(), side="right"))
    return pg


def per_game_frame(d: pd.DataFrame, n_seeds: int, ids: np.ndarray) -> pd.DataFrame:
    """Per game: sums over seeds of every quantity a line is built from (standardised possession frame `d`)."""
    import diag_c4_lib_v1 as L
    d = d.sort_values(["game_id", "seed", "k"], kind="stable").reset_index(drop=True)
    key = d["game_id"].to_numpy(np.int64) * 1000 + d["seed"].to_numpy(np.int64)
    gs = L.game_seed(d)
    gsk = gs["game_id"].to_numpy(np.int64) * 1000 + gs["seed"].to_numpy(np.int64)
    gs.index = gsk
    per, cs, ce = d["period"].to_numpy(), d["cs"].to_numpy(), d["ce"].to_numpy()
    pts, pre, post = d["pts"].to_numpy(float), d["pre"].to_numpy(), d["post"].to_numpy()
    reg = per <= 2
    f = pd.DataFrame(index=gsk)
    f["game_id"] = gs["game_id"].to_numpy()
    f["final_margin"] = gs["final_margin"].to_numpy(float)
    f["reg_margin"] = gs["reg_margin"].to_numpy(float)
    f["ot"] = gs["ot"].to_numpy(float)
    f["lead_changes"] = gs["lead_changes"].to_numpy(float)
    f["total"] = gs["total_pts"].to_numpy(float)
    s = pd.Series

    def by(mask, vals):
        return s(np.where(mask, vals, 0.0)).groupby(key).sum().reindex(gsk).fillna(0).to_numpy()
    f["h1_pts"] = by(per == 1, pts)
    f["h2_pts"] = by(per == 2, pts)
    f["poss_reg"] = by(reg, np.ones(len(d)))
    f["md4_poss"] = by((per == 2) & (ce <= 240), np.ones(len(d)))
    f["h1buz_n"] = by((per == 1) & (cs <= 10), np.ones(len(d)))
    f["h1buz_pts"] = by((per == 1) & (cs <= 10), pts)
    # margin at minute 36 = post of the last H2 possession starting with > 240 s left (else regulation margin)
    m36 = s(np.where((per == 2) & (cs > 240), post, np.nan)).groupby(key).last().reindex(gsk)
    f["m36"] = np.where(m36.isna(), f["reg_margin"], m36.to_numpy())
    # 2:00 reference: home margin before the first H2 possession starting <= 120 s
    first120 = s(np.where((per == 2) & (cs <= 120), pre, np.nan)).groupby(key).first().reindex(gsk).to_numpy()
    f["m120"] = first120
    a120 = np.abs(first120)
    f["st13"] = (a120 >= 13).astype(float)
    f["pts_last2"] = by((per == 2) & (cs <= 120), pts)
    # window possessions (round 3): H2, start <= 120, |pre| <= 6
    f["win_poss"] = by((per == 2) & (cs <= 120) & (np.abs(pre) <= 6), np.ones(len(d)))
    # eFG and home margin channels
    oh = d["off_side"].to_numpy() == 0
    for side, msk in (("h", oh), ("a", ~oh)):
        f[f"fgm2_{side}"] = by(msk, d["fgm2"].to_numpy(float))
        f[f"fgm3_{side}"] = by(msk, d["fgm3"].to_numpy(float))
        f[f"fga_{side}"] = by(msk, (d["fga2"] + d["fga3"]).to_numpy(float))
        f[f"poss_reg_{side}"] = by(msk & reg, np.ones(len(d)))
        f[f"pts_reg_{side}"] = by(msk & reg, pts)
    # late frame lines (bonus, fouls, TOV by reference state)
    lf = L.late_frame(d, within=300)
    lk = lf["game_id"].to_numpy(np.int64) * 1000 + lf["seed"].to_numpy(np.int64)
    lper, lcs, moff = lf["period"].to_numpy(), lf["cs"].to_numpy(), lf["moff"].to_numpy()
    w2 = (lper == 2) & (lcs <= 120)

    def lby(mask, vals):
        return s(np.where(mask, vals, 0.0)).groupby(lk).sum().reindex(gsk).fillna(0).to_numpy()
    bonus = lf["bonus"].to_numpy(float)
    dfoul = lf["dfoul"].to_numpy(float)
    okf = np.isfinite(dfoul)
    for lab, lo, hi in (("13", 1, 3), ("46", 4, 6), ("7p", 7, 999)):
        m = w2 & (moff >= lo) & (moff <= hi)
        f[f"lead{lab}_n"] = lby(m, np.ones(len(lf)))
        f[f"lead{lab}_bonus"] = lby(m, bonus)
        f[f"lead{lab}_fn"] = lby(m & okf, np.ones(len(lf)))
        f[f"lead{lab}_fouls"] = lby(m & okf, np.nan_to_num(dfoul))
    # leader TOV rate by reference state (D3 (b)): ref = home margin before the first possession starting <= W
    for W_, tag in ((240, "r4"), (120, "r2")):
        inw = (lper == 2) & (lcs <= W_)
        ref = s(np.where(inw, lf["pre"].to_numpy(), np.nan)).groupby(lk).transform("first").to_numpy()
        sg = np.where(lf["off_side"].to_numpy() == 0, 1, -1)
        rm = ref * sg
        for lab, lo, hi in (("812", 8, 12), ("13p", 13, 999)):
            m = inw & (rm >= lo) & (rm <= hi)
            f[f"ltov_{tag}_{lab}_n"] = lby(m, np.ones(len(lf)))
            f[f"ltov_{tag}_{lab}_t"] = lby(m, lf["tov"].to_numpy(float))
            f[f"lsec_{tag}_{lab}_d"] = lby(m, lf["dur"].to_numpy(float))
    # down 1-3 with < 30 s (D4 c), reported
    m = (lper == 2) & (lcs < 30) & (moff <= -1) & (moff >= -3)
    f["d13_n"] = lby(m, np.ones(len(lf)))
    f["d13_pts"] = lby(m, lf["pts"].to_numpy(float))
    f["d13_3pa"] = lby(m, (lf["cls"].to_numpy() == "3PA").astype(float))
    f["d13_2pa"] = lby(m, (lf["cls"].to_numpy() == "2PA").astype(float))
    g = f.groupby("game_id")
    out = g.sum(numeric_only=True)
    out["n_seeds"] = g.size()
    for c in ("final_margin", "reg_margin", "m36", "total"):
        out[f"{c}_sq"] = g[c].apply(lambda v: float((v.to_numpy() ** 2).sum()))
    out["m36_x_late"] = g.apply(lambda x: float((x["m36"] * (x["reg_margin"] - x["m36"])).sum()))
    out["late_sq"] = g.apply(lambda x: float(((x["reg_margin"] - x["m36"]) ** 2).sum()))
    out["poss_sum"] = out["poss_reg"]
    out["poss_sq"] = g["poss_reg"].apply(lambda v: float((v.to_numpy() ** 2).sum()))
    return out.reindex(ids)


def run_frame(tag: str, ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import diag_c4_lib_v1 as L
    t = pd.read_parquet(TRAJ_DIR / tag / "trajectory.parquet")
    t = t[t["game_id"].isin(set(ids.tolist()))]
    d = L.std_from_sim(t)
    del t
    S = int(d["seed"].nunique())
    pg = per_game_frame(d, S, ids)
    gm = pd.read_parquet(LOOP_DIR / tag / "games.parquet")
    gm = gm[gm["game_id"].isin(set(ids.tolist()))]
    meta = json.loads((LOOP_DIR / tag / "run_meta.json").read_text(encoding="utf-8"))
    fm = (gm["home_pts"] - gm["away_pts"]).abs()
    gcol = pd.DataFrame({"p1_cnt": ((gm["n_periods"] <= 2) & (fm == 1)).astype(float).groupby(gm["game_id"]).sum(),
                         "gm_poss_sum": gm.groupby("game_id")["possessions"].sum(),
                         "gm_poss_sq": (gm["possessions"] ** 2).groupby(gm["game_id"]).sum()}).reindex(ids)
    pg = pg.join(gcol)
    if (gm.groupby("game_id").size().reindex(ids) != pg["n_seeds"]).any():
        raise SystemExit(f"{tag}: games.parquet and trajectory disagree on game-seeds")
    return pg, gm, {"n_seeds": S, "flags": {k: v for k, v in meta.get("adapter_flags", {}).items()
                                              if k.startswith("ENGINE_")}, "seeds": meta.get("seeds")}


class LoopLines:
    """Every closed-loop line from per-game sums; `w` (G,) bootstrap multiplicities."""

    def __init__(self, ids, real, pg_info):
        self.ids = ids
        self.real = real
        self.pg = pg_info
        rd = real["std"]
        rpg = per_game_frame(rd.assign(seed=0), 1, ids)
        self.r = rpg
        self.ok = real["pbp_ok"].reindex(ids).fillna(False).to_numpy() & rpg["n_seeds"].notna().to_numpy()
        fin = real["fin"]
        self.y = (fin["home_score"] - fin["away_score"]).to_numpy(float)
        self.ytot = (fin["home_score"] + fin["away_score"]).to_numpy(float)
        self.yot = (fin["n_periods"] > 2).to_numpy(float)
        act = real["act"]
        self.apos = act["game_poss"].to_numpy(float) if "game_poss" in act else None
        self.neutral = pg_info["neutral_site"].fillna(False).to_numpy(bool)
        self.sq = pg_info["spread_q"].to_numpy()
        self.absp = pg_info["abs_spread"].to_numpy(float)
        s = int(pg_info["_rating_sign"].dropna().iloc[0]) if "_rating_sign" in pg_info else 1
        self.env = (pg_info["home_off_c"] + pg_info["away_off_c"] - s * (pg_info["home_def_c"] + pg_info["away_def_c"])).to_numpy(float)
        self.gapr = (pg_info["home_net"] - pg_info["away_net"]).abs().to_numpy(float)

    def lines(self, p: pd.DataFrame, gm: pd.DataFrame, w: np.ndarray) -> dict:
        r, ok = self.r, self.ok
        S = p["n_seeds"].to_numpy(float)
        Sc = float(np.nanmax(S))
        out = {}

        def wm(x, mask=None):
            m = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
            return float((w[m] * x[m]).sum() / max(w[m].sum(), 1e-12))

        def ratio(num, den, mask=None):
            m = np.ones(len(w), bool) if mask is None else mask
            return float((w[m] * num[m]).sum() / max((w[m] * den[m]).sum(), 1e-12))
        mu = p["final_margin"].to_numpy() / S
        s2 = (p["final_margin_sq"].to_numpy() - S * mu ** 2) / (S - 1)
        # ---- MD3 / tie rate / P1
        out["tie_rate"] = ratio(p["ot"].to_numpy(), S)
        out["act_tie_rate"] = wm(self.yot)
        # P1 per game-seed: not OT and |final margin| == 1 needs seed-level data -> from gm
        out["P1"] = ratio(p["p1_cnt"].to_numpy(), S)
        out["P0_over_P1"] = out["tie_rate"] / max(out["P1"], 1e-12)
        # ---- G1
        if self.apos is not None:
            ps, pq = p["gm_poss_sum"].to_numpy(), p["gm_poss_sq"].to_numpy()
            n = (w * S).sum()
            m_ = (w * ps).sum() / n
            out["G1_poss_mean"] = float(m_)
            out["G1_poss_sd"] = float(np.sqrt(max((w * pq).sum() / n - m_ ** 2, 0)))
            out["act_G1_poss_mean"] = wm(self.apos)
            am = self.apos
            mm = np.isfinite(am)
            out["act_G1_poss_sd"] = float(np.sqrt((w[mm] * (am[mm] - out["act_G1_poss_mean"]) ** 2).sum() / w[mm].sum()))
        # ---- G5 / G9 / MD6
        res = self.y - mu
        sd_own = np.sqrt(np.maximum(s2, 0))
        out["G5_margin_sd_ratio"] = wm(sd_own) / np.sqrt(wm((res - wm(res)) ** 2))
        tmu = p["total"].to_numpy() / S
        ts2 = (p["total_sq"].to_numpy() - S * tmu ** 2) / (S - 1)
        tres = self.ytot - tmu
        out["G5_total_sd_ratio"] = wm(np.sqrt(np.maximum(ts2, 0))) / np.sqrt(wm((tres - wm(tres)) ** 2))
        out["G9_margin_bias"] = wm(mu - self.y)
        out["G9_total_bias"] = wm(tmu - self.ytot)
        h1 = p["h1_pts"].to_numpy() / S
        rh1, rh2 = r["h1_pts"].to_numpy(), r["h2_pts"].to_numpy()
        out["G9_total_bias_H1"] = wm(h1 - rh1, ok)
        out["G9_total_bias_H2OT"] = wm((tmu - h1) - (self.ytot - rh1), ok)
        var_mc = wm(s2) / Sc
        xm = mu - wm(mu)
        var_pred = wm(xm ** 2)
        slope = (w * xm * (self.y - wm(self.y))).sum() / max((w * xm ** 2).sum(), 1e-12)
        out["G9_margin_slope"] = float(slope * var_pred / max(var_pred - var_mc, 1e-9))
        out["MD6_margin_sd_ratio"] = float(np.sqrt(wm(s2)) / np.sqrt(max(wm(res ** 2) - wm(s2) / Sc, 1e-9)))
        # half shares (per game-seed mean of share ~ per game mean of seed-summed share; use totals ratio)
        out["half1_share"] = ratio(p["h1_pts"].to_numpy(), p["total"].to_numpy())
        out["half2_share"] = ratio(p["h2_pts"].to_numpy(), p["total"].to_numpy())
        out["act_half1_share"] = ratio(rh1, self.ytot, ok)
        out["act_half2_share"] = ratio(rh2, self.ytot, ok)
        # ---- MD1 (2 Cov(m36, last4)) and MD6 Var(0-36), Var(last 4)
        m36mu = p["m36"].to_numpy() / S
        regmu = p["reg_margin"].to_numpy() / S
        latemu = regmu - m36mu
        v36 = (p["m36_sq"].to_numpy() - S * m36mu ** 2) / (S - 1)
        vl = (p["late_sq"].to_numpy() - S * latemu ** 2) / (S - 1)
        c_ = (p["m36_x_late"].to_numpy() - S * m36mu * latemu) / (S - 1)
        ra = r["m36"].to_numpy() - m36mu
        rl = (r["reg_margin"].to_numpy() - r["m36"].to_numpy()) - latemu
        out["MD1_cov_sim"] = 2 * wm(c_, ok)
        out["MD1_cov_real"] = 2 * (wm(ra * rl, ok) - wm(c_, ok) / Sc)
        out["MD1_gap"] = out["MD1_cov_sim"] - out["MD1_cov_real"]
        out["MD6_var036_sim"] = wm(v36, ok)
        out["MD6_var036_real"] = wm(ra ** 2, ok) - wm(v36, ok) / Sc
        out["MD6_varlast4_sim"] = wm(vl, ok)
        out["MD6_varlast4_real"] = wm(rl ** 2, ok) - wm(vl, ok) / Sc
        # ---- MD2 by |spread| quintile
        for q in range(5):
            m = self.sq == q
            if (w[m] > 0).sum() < 3:
                out[f"MD2_q{q + 1}"] = float("nan")
                continue
            out[f"MD2_q{q + 1}"] = float(np.sqrt(wm(s2, m)) / np.sqrt(max(wm(res ** 2, m) - wm(s2, m) / Sc, 1e-9)))
        # ---- MD4 / MD5
        out["MD4_h2_last4_poss"] = wm(p["md4_poss"].to_numpy() / S, ok)
        out["act_MD4_h2_last4_poss"] = wm(r["md4_poss"].to_numpy(), ok)
        out["MD5_pts13_last2"] = ratio(p["pts13_last2"].to_numpy(), p["st13"].to_numpy(), ok)
        out["act_MD5_pts13_last2"] = ratio(r["pts13_last2"].to_numpy(), r["st13"].to_numpy(), ok)
        out["share_st13"] = ratio(p["st13"].to_numpy(), S, ok)
        out["act_share_st13"] = wm(r["st13"].to_numpy(), ok)
        # ---- F lines
        for lab in ("13", "46", "7p"):
            out[f"lead{lab}_bonus"] = ratio(p[f"lead{lab}_bonus"].to_numpy(), p[f"lead{lab}_n"].to_numpy(), ok)
            out[f"act_lead{lab}_bonus"] = ratio(r[f"lead{lab}_bonus"].to_numpy(), r[f"lead{lab}_n"].to_numpy(), ok)
            out[f"trail_fouls_{lab}"] = ratio(p[f"lead{lab}_fouls"].to_numpy(), p[f"lead{lab}_fn"].to_numpy(), ok)
            # no like-for-like real value: the possession table's open-time team-foul counters miss fouls that
            # fall on the possession-change second (real counter change 0.06 per leader possession vs 0.65 pbp
            # PersonalFoul events); the real reference is D4's pbp count (an upper bound), reported in the doc.
        # ---- E lines (leader TOV) and leader seconds per possession
        for tg in ("r4", "r2"):
            for lab in ("812", "13p"):
                out[f"ltov_{tg}_{lab}"] = ratio(p[f"ltov_{tg}_{lab}_t"].to_numpy(), p[f"ltov_{tg}_{lab}_n"].to_numpy(), ok)
                out[f"act_ltov_{tg}_{lab}"] = ratio(r[f"ltov_{tg}_{lab}_t"].to_numpy(), r[f"ltov_{tg}_{lab}_n"].to_numpy(), ok)
                out[f"lsec_{tg}_{lab}"] = ratio(p[f"lsec_{tg}_{lab}_d"].to_numpy(), p[f"ltov_{tg}_{lab}_n"].to_numpy(), ok)
                out[f"act_lsec_{tg}_{lab}"] = ratio(r[f"lsec_{tg}_{lab}_d"].to_numpy(), r[f"ltov_{tg}_{lab}_n"].to_numpy(), ok)
        # ---- window possessions by k (round 3 reading)
        for j, v in self._wink(p, w, ok).items():
            out[j] = v
        # ---- H1 buzzer test
        out["h1buz_ppp"] = ratio(p["h1buz_pts"].to_numpy(), p["h1buz_n"].to_numpy(), ok)
        out["act_h1buz_ppp"] = ratio(r["h1buz_pts"].to_numpy(), r["h1buz_n"].to_numpy(), ok)
        # ---- D1 home margin (non-neutral) and eFG home-minus-away gap
        nn = ~self.neutral
        out["D1_home_margin_bias"] = wm(mu - self.y, nn)
        efg = lambda x, sd: (x[f"fgm2_{sd}"].to_numpy() + 1.5 * x[f"fgm3_{sd}"].to_numpy())  # noqa: E731
        out["D1_efg_gap"] = ratio(efg(p, "h"), p["fga_h"].to_numpy(), ok) - ratio(efg(p, "a"), p["fga_a"].to_numpy(), ok)
        out["act_D1_efg_gap"] = ratio(efg(r, "h"), r["fga_h"].to_numpy(), ok) - ratio(efg(r, "a"), r["fga_a"].to_numpy(), ok)
        # ---- D2 slopes (weighted OLS over games)
        okr = ok & np.isfinite(self.env) & np.isfinite(self.gapr)
        sp = (p["poss_reg_h"].to_numpy() + p["poss_reg_a"].to_numpy()) / S
        rp = r["poss_reg_h"].to_numpy() + r["poss_reg_a"].to_numpy()
        out["D2_poss_on_env"], out["act_D2_poss_on_env"] = self._wslope(self.env, sp, w, okr), self._wslope(self.env, rp, w, okr)
        out["D2_poss_on_gap"], out["act_D2_poss_on_gap"] = self._wslope(self.gapr, sp, w, okr), self._wslope(self.gapr, rp, w, okr)
        out["D2_total_on_env"], out["act_D2_total_on_env"] = (self._wslope(self.env, tmu, w, okr & np.isfinite(self.ytot)),
                                                              self._wslope(self.env, self.ytot, w, okr & np.isfinite(self.ytot)))
        oks = ok & np.isfinite(self.absp)
        out["D2_leadchg_on_spread"] = self._wslope(self.absp, p["lead_changes"].to_numpy() / S, w, oks)
        out["act_D2_leadchg_on_spread"] = self._wslope(self.absp, r["lead_changes"].to_numpy(), w, oks)
        # ---- D4 (c) reported
        out["d13_ppp"] = ratio(p["d13_pts"].to_numpy(), p["d13_n"].to_numpy(), ok)
        out["act_d13_ppp"] = ratio(r["d13_pts"].to_numpy(), r["d13_n"].to_numpy(), ok)
        out["d13_3pa"] = ratio(p["d13_3pa"].to_numpy(), p["d13_n"].to_numpy(), ok)
        out["act_d13_3pa"] = ratio(r["d13_3pa"].to_numpy(), r["d13_n"].to_numpy(), ok)
        return out

    @staticmethod
    def _wslope(x, y, w, m):
        m = m & np.isfinite(x) & np.isfinite(y) & (w > 0)
        ww, xx, yy = w[m], x[m], y[m]
        xm, ym = (ww * xx).sum() / ww.sum(), (ww * yy).sum() / ww.sum()
        return float((ww * (xx - xm) * (yy - ym)).sum() / max((ww * (xx - xm) ** 2).sum(), 1e-12))

    def _wink(self, p, w, ok) -> dict:
        out = {}
        for j in range(7):
            a, b = p.get(f"wk{j}_w"), p.get(f"wk{j}_n")
            ra, rb = self.r.get(f"wk{j}_w"), self.r.get(f"wk{j}_n")
            if a is None:
                continue
            out[f"win_k{j}"] = float((w * a.to_numpy()).sum() / max((w * b.to_numpy()).sum(), 1e-12))
            out[f"act_win_k{j}"] = float((w * ok * ra.to_numpy()).sum() / max((w * ok * rb.to_numpy()).sum(), 1e-12))
            out[f"act_n_k{j}"] = float((w * ok * rb.to_numpy()).sum())
        return out


def add_seed_level(pg: pd.DataFrame, d_std: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Lines that need the seed-level state at 2:00: points in the 13+ state, window possessions by k."""
    d = d_std
    key = d["game_id"].to_numpy(np.int64) * 1000 + d["seed"].to_numpy(np.int64)
    per, cs, pre = d["period"].to_numpy(), d["cs"].to_numpy(), d["pre"].to_numpy()
    m2 = (per == 2) & (cs <= 120)
    s = pd.Series
    ref = s(np.where(m2, pre, np.nan)).groupby(key).transform("first").to_numpy()
    k = np.abs(ref)
    pts = d["pts"].to_numpy(float)
    gid = d["game_id"].to_numpy(np.int64)
    out = pd.DataFrame(index=ids)
    out["pts13_last2"] = s(np.where(m2 & (k >= 13), pts, 0.0)).groupby(gid).sum().reindex(ids).fillna(0).to_numpy()
    winm = m2 & (np.abs(pre) <= 6)
    first = np.r_[True, key[1:] != key[:-1]]
    gkey = pd.DataFrame({"key": key, "gid": gid, "k": k, "win": winm.astype(float)})
    perk = gkey.groupby("key").agg(gid=("gid", "first"), k=("k", "first"), win=("win", "sum"))
    for j in range(7):
        sel = perk["k"] == j
        out[f"wk{j}_n"] = sel.groupby(perk["gid"]).sum().reindex(ids).fillna(0).to_numpy()
        out[f"wk{j}_w"] = perk["win"].where(sel, 0).groupby(perk["gid"]).sum().reindex(ids).fillna(0).to_numpy()
    del first
    return pg.join(out)


#: (line, target key or constant, kind). kind: "toward" = primary/gate must move toward the target;
#: "veto" = must not move away by more than the floor.
VETO_LINES = [("G1_poss_mean", "act_G1_poss_mean"), ("G1_poss_sd", "act_G1_poss_sd"),
              ("half1_share", "act_half1_share"), ("half2_share", "act_half2_share"),
              ("G5_margin_sd_ratio", 1.0), ("G5_total_sd_ratio", 1.0), ("G9_margin_bias", 0.0),
              ("G9_margin_slope", 1.0), ("h1buz_ppp", "act_h1buz_ppp"), ("D1_home_margin_bias", 0.0),
              ("D1_efg_gap", "act_D1_efg_gap"), ("D2_poss_on_env", "act_D2_poss_on_env"),
              ("D2_poss_on_gap", "act_D2_poss_on_gap"), ("D2_total_on_env", "act_D2_total_on_env"),
              ("D2_leadchg_on_spread", "act_D2_leadchg_on_spread"), ("tie_rate", "act_tie_rate"),
              ("ltov_r4_812", "act_ltov_r4_812"), ("ltov_r2_812", "act_ltov_r2_812")]
PRIMARY = {"C": [("MD4_h2_last4_poss", "act_MD4_h2_last4_poss"), ("MD5_pts13_last2", "act_MD5_pts13_last2")],
           "F": [("tie_rate", "act_tie_rate")],
           # s15.A.3 quotes the E targets for the 13+ state only (19.0 / 21.1% real vs 22.1 / 25.4% served at the
           # 4:00 / 2:00 references); the 8-12 leader TOV rate is already on target (D3: 17.9 vs 17.9, 16.4 vs 17.4),
           # so it cannot move toward the actual by a floor and is read as a move-away veto instead (clarified
           # 2026-10-13 BEFORE any closed-loop arm ran; control-only smoke showed the 8-12 gap ~0 on the sample).
           "E": [("ltov_r4_13p", "act_ltov_r4_13p"), ("ltov_r2_13p", "act_ltov_r2_13p")]}
REPORTED = ["P1", "P0_over_P1", "lead13_bonus", "lead46_bonus", "lead7p_bonus", "trail_fouls_13", "trail_fouls_46",
            "trail_fouls_7p", "lsec_r4_812", "lsec_r4_13p", "lsec_r2_812", "lsec_r2_13p", "share_st13",
            "G9_total_bias", "G9_total_bias_H1", "G9_total_bias_H2OT", "d13_ppp", "d13_3pa",
            "MD6_varlast4_sim", "MD1_cov_sim", "MD1_cov_real"]


def loop(a) -> int:
    ids = np.sort(pd.read_parquet(SAMPLE)["game_id"].astype("int64").to_numpy())
    import diag_c4_lib_v1 as L
    real = _real_side(ids)
    pgi = _pregame(ids)
    LL = LoopLines(ids, real, pgi)
    LL.r = add_seed_level(LL.r, real["std"].assign(seed=0), ids)
    tags = [a.control, *a.draws, *a.arms]
    frames, info = {}, {}
    for t in tags:
        pg, gm, inf = run_frame(t, ids)
        tr = pd.read_parquet(TRAJ_DIR / t / "trajectory.parquet")
        tr = tr[tr["game_id"].isin(set(ids.tolist()))]
        pg = add_seed_level(pg, L.std_from_sim(tr), ids)
        del tr
        frames[t] = (pg, gm)
        info[t] = inf
        print(f"loaded {t}: {inf['n_seeds']} seeds, flags {inf['flags']}", flush=True)
    G = len(ids)
    one = np.ones(G)
    val = {t: LL.lines(frames[t][0], frames[t][1], one) for t in tags}
    wb = boot_weights(G)
    names = [k for k in val[a.control] if not k.startswith("act_")]
    draw_sd = {k: float(np.nanstd([val[t][k] for t in a.draws], ddof=1)) for k in names}
    codes = {t: f"L-{chr(65 + i)}" for i, t in enumerate(np.random.default_rng(CODE_SEED).permutation(a.arms))}
    bctrl = [LL.lines(frames[a.control][0], frames[a.control][1], wb[b]) for b in range(N_BOOT)]
    coded = {}
    for t in a.arms:
        barm = [LL.lines(frames[t][0], frames[t][1], wb[b]) for b in range(N_BOOT)]
        rows = {}
        for k in names:
            se = float(np.nanstd([barm[b][k] - bctrl[b][k] for b in range(N_BOOT)], ddof=1))
            rows[k] = {"arm": val[t][k], "ctrl": val[a.control][k], "delta": val[t][k] - val[a.control][k],
                       "draw_sd": draw_sd[k], "boot_se": se, "floor": max(2 * draw_sd[k], 2 * se)}
        coded[codes[t]] = rows
    act = {k: v for k, v in val[a.control].items() if k.startswith("act_")}
    res = {"spec": "docs/models/late_game/experiments.md s15 (closed loop)", "control": a.control, "draws": a.draws,
           "n_games": int(G), "n_pbp_ok": int(LL.ok.sum()), "seeds": {t: info[t]["n_seeds"] for t in tags},
           "control_lines": val[a.control], "draw_lines": {t: val[t] for t in a.draws}, "draw_sd": draw_sd,
           "actual": act, "coded": coded}
    GRADE_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.with_name(out.stem + "_coded.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("coded closed-loop table written", flush=True)
    key = {c: t for t, c in codes.items()}
    res["key"] = key
    res["flags"] = {t: info[t]["flags"] for t in tags}
    blocks = sorted({"C", "F"} | {b for b in a.block_of.values()}) if a.block_of else ["C", "F"]
    res["seed_check"] = seed_check(val[a.control], act, draw_sd, blocks)
    res["verdicts"] = {key[c]: judge(rows, act, val[a.control], a.block_of.get(key[c], "F") if a.block_of else "F")
                       for c, rows in coded.items()}
    res["written_at"] = datetime.now(UTC).isoformat()
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for t, v in res["verdicts"].items():
        print(t, json.dumps({k: v[k] for k in ("candidate", "primary_moves", "vetoes_failing", "md_failing")}))
    print("seed check:", json.dumps(res["seed_check"]))
    return 0


def _target(tgt, act: dict, ctrl: dict):
    if isinstance(tgt, str):
        return act.get(tgt, ctrl.get(tgt))
    return tgt


def seed_check(ctrl: dict, act: dict, draw_sd: dict, blocks=("C", "F")) -> dict:
    """s15.0: a line's floor is resolved when 2 x draw SD < |control - target|. Only lines that must MOVE toward a
    target decide (the primaries of the blocks in the loop, MD1, MD2 Q1, MD3-MD5); "must not rise" lines (MD2 Q2-Q5,
    MD6, vetoes) have no gap to resolve and are read on their floors as they stand."""
    lines = {k: t for b in blocks for k, t in PRIMARY[b]}
    lines.update({"MD1_gap": 0.0, "tie_rate": "act_tie_rate", "MD4_h2_last4_poss": "act_MD4_h2_last4_poss",
                  "MD5_pts13_last2": "act_MD5_pts13_last2", "MD2_q1": 1.0})
    out = {}
    for k, t in lines.items():
        tv = _target(t, act, ctrl)
        gap = abs(ctrl[k] - tv)
        out[k] = {"floor_draws": 2 * draw_sd[k], "gap": gap, "resolved": bool(2 * draw_sd[k] < gap)}
    out["all_resolved"] = all(v["resolved"] for v in out.values() if isinstance(v, dict))
    return out


def judge(rows: dict, act: dict, ctrl: dict, block: str) -> dict:
    def toward(k, tgt):
        tv = _target(tgt, act, ctrl)
        r = rows[k]
        e0, e1 = abs(r["ctrl"] - tv), abs(r["arm"] - tv)
        return {"target": tv, "ctrl": r["ctrl"], "arm": r["arm"], "floor": r["floor"],
                "improve": e0 - e1, "floors": (e0 - e1) / r["floor"] if r["floor"] else None,
                "toward_by_floor": bool(e0 - e1 > r["floor"]), "away_by_floor": bool(e1 - e0 > r["floor"])}
    v = {"block": block}
    prim = {k: toward(k, t) for k, t in PRIMARY[block]}
    v["primary"] = prim
    v["primary_moves"] = all(x["toward_by_floor"] for x in prim.values())
    vet = {k: toward(k, t) for k, t in VETO_LINES}
    vet = {k: {**x, "pass": not x["away_by_floor"]} for k, x in vet.items()}
    wk = {}
    for j in range(7):
        k = f"win_k{j}"
        if k not in rows:
            continue
        r, av = rows[k], act.get(f"act_{k}")
        base_excess = r["ctrl"] - av > r["floor"]
        ref = r["ctrl"] if base_excess else av
        n_act = act.get(f"act_n_k{j}", 0.0)
        under = bool(n_act < MIN_N)
        wk[k] = {"arm": r["arm"], "ctrl": r["ctrl"], "actual": av, "floor": r["floor"], "n_actual_games": n_act,
                 "read_against": "control" if base_excess else "actual",
                 "underpowered": under,
                 "pass": bool(under or r["arm"] - ref <= r["floor"])}
    vet["window_poss_by_k"] = {"pass": all(x["pass"] for x in wk.values()), "by_k": wk}
    v["vetoes"] = vet
    v["vetoes_failing"] = [k for k, x in vet.items() if not x["pass"]]
    md = {}
    g = rows["MD1_gap"]
    md["MD1"] = {"ctrl_gap": g["ctrl"], "arm_gap": g["arm"], "floor": g["floor"],
                 "pass": bool(abs(g["ctrl"]) - abs(g["arm"]) > g["floor"]),
                 "veto_away": bool(abs(g["arm"]) - abs(g["ctrl"]) > g["floor"])}
    q = {f"q{i}": {"ctrl": rows[f"MD2_q{i}"]["ctrl"], "arm": rows[f"MD2_q{i}"]["arm"], "floor": rows[f"MD2_q{i}"]["floor"],
                   "rises": bool(rows[f"MD2_q{i}"]["delta"] > rows[f"MD2_q{i}"]["floor"])} for i in range(1, 6)}
    md["MD2"] = {"by_quintile": q, "q1_falls": bool(rows["MD2_q1"]["delta"] < -rows["MD2_q1"]["floor"]),
                 "pass": bool(not any(x["rises"] for x in q.values()) and rows["MD2_q1"]["delta"] < -rows["MD2_q1"]["floor"])}
    t3 = toward("tie_rate", "act_tie_rate")
    md["MD3"] = {**t3, "pass": bool(t3["toward_by_floor"] and rows["tie_rate"]["arm"] >= 0.035)}
    md["MD4"] = {**toward("MD4_h2_last4_poss", "act_MD4_h2_last4_poss")}
    md["MD4"]["pass"] = md["MD4"]["toward_by_floor"]
    md["MD5"] = {**toward("MD5_pts13_last2", "act_MD5_pts13_last2")}
    md["MD5"]["pass"] = md["MD5"]["toward_by_floor"]
    r6, r6v = rows["MD6_margin_sd_ratio"], rows["MD6_var036_sim"]
    md["MD6"] = {"ratio": {"ctrl": r6["ctrl"], "arm": r6["arm"], "floor": r6["floor"]},
                 "var036": {"ctrl": r6v["ctrl"], "arm": r6v["arm"], "floor": r6v["floor"]},
                 "varlast4": {"ctrl": rows["MD6_varlast4_sim"]["ctrl"], "arm": rows["MD6_varlast4_sim"]["arm"]},
                 "pass": bool(r6["delta"] <= r6["floor"] and r6v["delta"] <= r6v["floor"])}
    v["md"] = md
    v["md_failing"] = [k for k, x in md.items() if not x["pass"]]
    v["reported"] = {k: rows[k] for k in REPORTED if k in rows}
    v["candidate"] = bool(v["primary_moves"] and not v["vetoes_failing"] and not v["md_failing"])
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    o = sub.add_parser("offline")
    o.add_argument("--block", choices=["C", "E", "F"], required=True)
    o.add_argument("--arms", nargs="*", default=None)
    lp = sub.add_parser("loop")
    lp.add_argument("--control", required=True)
    lp.add_argument("--draws", nargs=4, required=True)
    lp.add_argument("--arms", nargs="+", required=True)
    lp.add_argument("--block-of", nargs="*", default=[], help="tag=BLOCK (C, F, E); compositions use F")
    lp.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "offline":
        return offline(a.block, a.arms)
    a.block_of = dict(kv.split("=", 1) for kv in a.block_of)
    return loop(a)


if __name__ == "__main__":
    raise SystemExit(main())
