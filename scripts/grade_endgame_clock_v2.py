"""grade_endgame_clock_v2.py -- the ONE blind grader of the combined endgame/clock follow-up round,
`docs/models/clock/experiments.md` section 46 (PRE-REGISTERED 2026-10-14 with PM rulings 1-10). Blocks A and B
(this file's offline modes); the loop / full modes serve every block (A, B now; C later).

Opus engine worker, 2026-10-14. Written and COMMITTED BEFORE any section-46 arm was fitted (no CA*/FP* fit output
existed at commit time). Adopts nothing. Line definitions are IMPORTED unchanged from `grade_endgame_v1.py` (window
rows, deviance, E[min(T, R)], bootstrap, MD1-MD6, closed-loop per-game frames) and `grade_clock_env_loop_v1.py`
(the section-45 veto list incl. the dispersion gates, round-8 lines, env quintiles, team table); only the lines
section 46 adds are new here.

=====================================================================================================================
MODE `offline --block A`  (s46 2.A, 3, 4.1; fold 2 selects, fold 1 confirms; period 2 selects, period 1 REPORTED)
=====================================================================================================================
Rows: the late-game window (`build_endgame_window_v1.clock_window`: regulation, start <= 240 s), test season of the
fold. Inputs `results/endgame_clock_v2/offline/<fold>/A/<arm>.npz` (keys game_id, period, poss_index; `pmf` (n, 91)
first-chance duration law, `pmf_end` (n, 91) the same law conditional on the realised chance-1 end class).
Cells: 9 margin buckets (leader 1-3/4-7/8-12/13+, trailer the same, tied) x 5 time buckets = 45 (`endgame.roleband`,
`endgame.tbucket`).
Lines (period 2 selects):
  dev       first-chance duration deviance on ALL window rows (v1 `dev`: -2 mean log-lik, censored rows P(T >= y),
            clip 1e-12). REPORTED (s46 3: "(i) reported").
  dev_pos   the same on rows where the CONTROL CA0 assigns positive mass to the observed outcome (control
            p(y) > 1e-12, or P(T >= y) > 1e-12 if censored); the excluded share is written. PRIMARY (dual deviance
            gate on (ii)).
  cal       mean over POWERED cells (>= 200 test rows; s46 3 "< 200 rows (cells): UNDERPOWERED") of
            |mean model E[min(T, R)] - mean actual first-chance seconds|. PRIMARY. `cal45` = the same over every
            populated cell (reported). Every cell is in the per-cell table with its underpowered label; the (30,60]
            bucket's nine cells are printed explicitly.
  slope_tempo       model vs actual mean seconds across offence TEMPO quintiles (v1 (b)). HARD GATE: not below the
                    control's by more than its floor (the 10-13 finding: a pooled law kills it).
  slope_zoff        the same across quintiles of z_off = off_c(offence) + def_c(defence) (clock s41.B (b)). Gate as
                    tempo.
  slope_defz        the same across quintiles of the defence's def_c (s41.B (d)). Gate as tempo.
  slope_team_lead / slope_team_trail   across offence PRIOR-STRENGTH quintiles, leaders / trailers (v1, s15.0).
                    Gate as tempo.
  slope_lead / slope_trail  model band means on actual band means (v1 (a)), target 1. Gate: |slope - 1| not worse
                    than the control's by more than its floor.
  REPORTED: dev_end (conditional law), pergame_abs_err (mean over games of |sum model - sum actual| / rows),
            start-type table (prev_end), team right-sign share (offence teams with >= 200 test rows), period 1.
Noise floor (s46 3): floor = max(|CA0s1 - CA0| (reseed difference), 2 x paired game-block bootstrap SE (200 draws,
numpy seed 12345)), per line and per arm.
Shrink grid (ruling 7): mode `select-m` reads fold-1 files `<arm>_m{20,50,100}.npz` for CA1 and CA2 and picks, per
arm, the m with the lowest fold-1 period-2 `dev_pos`; among the m whose `dev_pos` is within one fold-1 floor of the
best, the LARGEST m (strongest shrink toward the served law) is taken. Frozen to
`results/endgame_clock_v2/grade/m_selection.json` before any fold-2 fit of CA1/CA2. Fold 2 never sees the grid.
Decision (s46 4.1, literal): an arm WINS block A if on F2 period 2 it beats CA0 by more than one floor on BOTH
`dev_pos` and `cal`, every gate above HOLDS, and F1 keeps the sign of both deltas. CA0s1 is never selectable. Among
winners the simplest (rank CA3 < CA2 < CA1) whose `dev_pos` and `cal` are each within one floor of the best winner's
wins. No winner: block A REFUTED. (`dev` is reported; a win on `dev` that does not survive on `dev_pos` does not win.)

=====================================================================================================================
MODE `offline --block B`  (s46 2.B, rulings 3, 8, 9)
=====================================================================================================================
Rows FPS: window FIRST chances (`event_window`, possession_outcome round-2 design), regulation, start_clock <= 30 s,
offence start margin in [-6, 0]. Cells: trailing 1-3 (PRIMARY, the only primary cell, ruling 9), trailing 4-6 and
tied (REPORTED). Period 2 selects, period 1 REPORTED (ruling 8).
Inputs `results/endgame_clock_v2/offline/<fold>/B/<arm>.npz`: `prob` (n, 6) in `possession_outcome.CLASSES` order,
`pmake` (n,) the served/arm fg_make probability of the first-chance shot (NaN when the chance did not end in a FGA
or the shot has no fg_make design row; the NaN mask must be identical for every arm).
Lines (trailing 1-3, period 2):
  mll        six-class log loss (clip 1e-12). PRIMARY.
  make_dev   binary make deviance -2 mean log-lik of the realised make (clip 1e-12). PRIMARY.
  REPORTED: share_3pa, share_2pa (rim + jump2), share_tov, share_ft (model vs actual), make rate model vs actual,
            expected shot points per FPS possession (sum over shot classes of P(class) x P(make) x value, on rows with
            a make prediction), the same lines in trailing 4-6 and tied, period 1.
  slope_team_3pa / slope_team_make  model vs actual across offence PRIOR-STRENGTH quintiles: gate "not below the
            control's by more than its floor"; every FPS slope is labelled UNDERPOWERED (< 3,000 rows), the gate
            value is still computed and printed.
Decision (literal s46 4.1): an arm WINS block B if on F2 it beats FP0 by more than one floor on BOTH `mll` AND
`make_dev`, the slope gates hold, and F1 keeps both signs. FP0s1 never selectable. Simplest winner (FP1 < FP2)
within one floor of the best on both. FP3 = FP2's laws on LF2 (ruling 3): offline it is FP2 by construction (LF2
changes the foul-accrual law only, not the class or make law); it is a composition read in the loop alongside FP2.
REPORTED-ONLY reading (selects nothing; for the PM): "beats FP0 by more than one floor on every primary line its law
changes and is not worse beyond floor on the other" (FP1 changes `mll` only).

=====================================================================================================================
MODES `loop` (500 x S screen) and `full` (5,710 x 200 box read)   (s46 3-4, rulings 1, 2)
=====================================================================================================================
Runs: `run_engine_window_v2.py`, served stack v3, fold 2 / 2025, paired seeds, trajectories on every seed.
Lines: `grade_clock_env_loop_v1.Lines` (every `grade_endgame_v1.LoopLines` line incl. MD1-MD6, plus the round-8
lines, H1 env slope, env quintiles, dispersion replicas) and the dispersion summaries of
`diag_margin_dispersion_v2.py` on each run's own extract. Floor = max(2 x SD of the four control draws, 2 x paired
game-bootstrap SE) per line (200 draws, seed 12345).
Hard vetoes (the section-45 list, every line; "regression" = |arm - target| - |control - target| > floor):
  G1 possession mean AND SD (5.629 vs 5.474 on the full set, printed in every table header), half shares H1/H2, G5
  margin/total SD ratios, G9 margin bias/slope, tempo elasticity ratio, team pace slope, OREB/FGM/TOV/FTA within-game
  slopes, OT/tie rate, D1 home margin and eFG gap, margin SD ratio overall and by |spread| Q1-Q5 (band
  max(0.009, floor); literal 0.009 printed = R3), pace SD ratio overall in [0.95, 1.05] and not away in Q1-Q5,
  regulation total SD ratio (band max(0.020, floor); [0.974, 1.014] printed), overall regulation margin SD rise,
  R2 H1 env magnitude, R4 env-quintile possession gap.
  For blocks A and B, R2 and R4 are NO-REGRESSION vetoes (neither block targets env): R2 fails if the H1 possession
  slope on env moves away from the actual by more than its floor or overshoots the actual by more than the control's
  shortfall; R4 fails if more than two env quintiles' possession gap (real - sim) is further from 0 than the
  control's by more than the cell floor (round-10 ruling). Round 10's H1 "must move toward" reading was E1's
  (an env arm's) and is not applied to A/B. Block C (later) reads R2 as a primary-type line per s46 2.C.
  G9 total bias: REPORTED by channel (Decision 11).
MD1-MD6 (late_game s15.1, `grade_endgame_v1.judge` md block unchanged): every one must pass.
Primaries: A = MD4 (H2 possessions ending in the last 4:00) AND MD5 (13+ last-2-min points) toward actual beyond
floor; B = MD3 (tie rate toward actual beyond floor, never below 3.5%) AND FPS PPP (`d13_ppp`, trailing 1-3 with
< 30 s) toward actual beyond floor.
Screen verdict (`loop`, s46 4.2): "proceed to full size" iff both primaries move, no hard veto fails and MD1-MD6 pass;
otherwise "stop". It NEVER says candidate. Seed-count check: the primaries, MD1, MD2 Q1, MD3-MD5 resolved
(2 x draw SD < |control - target|); if a deciding line is unresolved the screen is re-run at 50 seeds and relabelled.
Full verdict (`full`, s46 4.3): CANDIDATE iff the same conditions hold at 5,710 x 200 against four 200-seed draws
(200 vs 200); else NOT A CANDIDATE.

    set CBB_TRUTH=verified_v1
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py select-m
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py offline --block A
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py offline --block B
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py offline --block C   # s46 2.C C-pre (added 2026-10-14
        # before any block-C fit; definitions in the `offline --block C` comment block below)
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py loop --block A --control <tag> --draws <4 tags> \
        --arm <tag> --loop-dir results/endgame_clock_v2/loop --disp-dir results/endgame_clock_v2/dispersion --out <json>
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py full --block A --control <tag> --draws <4 tags> \
        --arm <tag> --full-dir <dir> --disp-dir <dir> --out <json>
    # s49 (added 2026-10-14 BEFORE any s49 fit; definitions in the SECTION 49 comment block above `main`):
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py s49-select-m
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py s49-offline
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py loop --spec s49 --block A ...   (screen: proceed / stop)
    .venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py full --spec s49 --block A ...   (CANDIDATE / NOT A CANDIDATE)
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
import grade_endgame_v1 as EG  # noqa: E402

OUT = ROOT / "results/endgame_clock_v2"
OFF_DIR = OUT / "offline"
GRADE_DIR = OUT / "grade"
N_BOOT, BOOT_SEED, CODE_SEED = EG.N_BOOT, EG.BOOT_SEED, 20261014
EPS = 1e-12
MIN_CELL = 200            # s46 3: cells < 200 rows UNDERPOWERED
MIN_SLOPE = 3000          # s46 3: slopes on < 3,000 rows UNDERPOWERED
GRID = np.arange(91, dtype=np.float64)
M_GRID = (20, 50, 100)

CONTROL = {"A": "CA0", "B": "FP0", "C": "E0"}
NOISE = {"A": "CA0s1", "B": "FP0s1"}
RANK = {"CA0": 0, "CA3": 1, "CA2": 2, "CA1": 3, "FP0": 0, "FP1": 1, "FP2": 2, "FP3": 3}
PRIM = {"A": ("dev_pos", "cal"), "B": ("mll", "make_dev")}
GATES_NOT_BELOW = {"A": ("slope_tempo", "slope_zoff", "slope_defz", "slope_team_lead", "slope_team_trail"),
                   "B": ("slope_team_3pa", "slope_team_make")}
GATES_ABS1 = {"A": ("slope_lead", "slope_trail"), "B": ()}
FPS_CELLS = ("trail 1-3", "trail 4-6", "tied")
SHOT_IDX = {1: 2.0, 2: 2.0, 3: 3.0}        # possession_outcome class index -> shot value (rim, jump2, three)


# =====================================================================================================================
# helpers
# =====================================================================================================================
def quintile(v: np.ndarray) -> np.ndarray:
    return EG.quintile(v)


def code_arms(block: str, arms: list[str]) -> dict[str, str]:
    rest = sorted(a for a in arms if a != CONTROL[block])
    perm = np.random.default_rng(CODE_SEED + ord(block[0])).permutation(len(rest))
    out = {CONTROL[block]: "CTRL"} if CONTROL[block] in arms else {}
    for j, i in enumerate(perm):
        out[rest[i]] = f"{block}-{chr(65 + j)}"
    return out


def fps_cell(m: np.ndarray) -> np.ndarray:
    """0 trailing 1-3, 1 trailing 4-6, 2 tied, -1 outside FPS."""
    m = np.asarray(m)
    return np.select([(m <= -1) & (m >= -3), (m <= -4) & (m >= -6), m == 0], [0, 1, 2], -1).astype(np.int64)


# =====================================================================================================================
# truth frames
# =====================================================================================================================
def truth_A(fold: str) -> pd.DataFrame:
    _, te = W.split(W.clock_window(), fold)
    t = pd.DataFrame({"game_id": te["game_id"].to_numpy(np.int64), "period": te["period"].to_numpy(np.int64),
                      "poss_index": te["poss_index"].to_numpy(np.int64),
                      "y": np.clip(te["duration_s"].to_numpy(np.int64), 0, 90),
                      "cen": te["censored"].to_numpy(bool), "R": te["seconds_remaining"].to_numpy(float),
                      "m": te["score_diff"].to_numpy(np.int64), "end1": te["end1_code"].to_numpy(np.int64),
                      "tempo": te["off_tempo_rel"].to_numpy(float),
                      "strength": te["off_rating_off_c"].to_numpy(float),
                      "zoff": (te["off_rating_off_c"] + te["def_rating_def_c"]).to_numpy(float),
                      "defz": te["def_rating_def_c"].to_numpy(float),
                      "team": te["offense_team_id"].to_numpy(np.int64),
                      "start": te["prev_end"].astype(str).to_numpy()})
    t["rb"] = W.roleband(t["m"].to_numpy())
    t["tb"] = W.tbucket(t["R"].to_numpy())
    return t


def fps_rows(d: pd.DataFrame) -> pd.DataFrame:
    """FPS rows of an event-window frame (shared with the trainer so both see the same rows)."""
    return d[(d["period"] <= 2) & (d["start_clock"] <= 30) & (d["start_score_diff"] <= 0)
             & (d["start_score_diff"] >= -6)].reset_index(drop=True)


def truth_B(fold: str) -> pd.DataFrame:
    _, te = W.split(W.event_window(), fold)
    te = fps_rows(te)
    y = te["y"].to_numpy(np.int64)
    made = np.full(len(te), np.nan)
    for cls, col in ((1, "fgm_rim"), (2, "fgm_jump2"), (3, "fgm_3")):
        r = y == cls
        made[r] = (te[col].to_numpy()[r] > 0).astype(float)
    return pd.DataFrame({"game_id": te["game_id"].to_numpy(np.int64), "period": te["period"].to_numpy(np.int64),
                         "poss_index": te["poss_index"].to_numpy(np.int64), "y": y, "made": made,
                         "m": te["start_score_diff"].to_numpy(np.int64), "cell": fps_cell(te["start_score_diff"]),
                         "strength": te["off_rating_off_c"].to_numpy(float)})


def load_pred(block: str, fold: str, arm: str, t: pd.DataFrame) -> dict:
    z = np.load(OFF_DIR / fold / block / f"{arm}.npz", allow_pickle=False)
    k = pd.DataFrame({"game_id": z["game_id"].astype(np.int64), "period": z["period"].astype(np.int64),
                      "poss_index": z["poss_index"].astype(np.int64), "_r": np.arange(len(z["game_id"]))})
    j = t[["game_id", "period", "poss_index"]].merge(k, on=["game_id", "period", "poss_index"], how="left",
                                                     validate="1:1")
    if j["_r"].isna().any() or len(k) != len(t):
        raise SystemExit(f"{fold}/{block}/{arm}: prediction rows do not cover the test rows exactly")
    r = j["_r"].to_numpy(np.int64)
    out = {kk: np.asarray(z[kk])[r] for kk in z.files if kk not in ("game_id", "period", "poss_index", "meta")}
    out["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return out


# =====================================================================================================================
# per-row quantities
# =====================================================================================================================
def rows_A(t: pd.DataFrame, P: dict) -> dict:
    p = np.asarray(P["pmf"], float)
    y, cen, R = t["y"].to_numpy(), t["cen"].to_numpy(), t["R"].to_numpy()
    ii = np.arange(len(y))
    pu = p[ii, y]
    S = 1.0 - np.cumsum(p, axis=1)[ii, y] + pu
    raw = np.where(cen, S, pu)
    out = {"ll": np.log(np.clip(raw, EPS, None)), "raw": raw,
           "e": (p * np.minimum(GRID[None, :], R[:, None])).sum(axis=1)}
    if "pmf_end" in P:
        pe = np.asarray(P["pmf_end"], float)
        pue = pe[ii, y]
        Se = 1.0 - np.cumsum(pe, axis=1)[ii, y] + pue
        out["ll_end"] = np.log(np.clip(np.where(cen, Se, pue), EPS, None))
    return out


def rows_B(t: pd.DataFrame, P: dict) -> dict:
    p = np.asarray(P["prob"], float)
    y = t["y"].to_numpy()
    pm = np.asarray(P["pmake"], float)
    mk = t["made"].to_numpy()
    ok = np.isfinite(pm) & np.isfinite(mk)
    llm = np.where(ok, np.log(np.clip(np.where(mk == 1, pm, 1 - pm), EPS, None)), 0.0)
    # expected shot points per possession on rows with a make prediction for EVERY shot class: the arm's own make
    # probability is only known for the realised class, so the reported line uses P(class) x value x pmake on the
    # realised shot rows (sum over rows / rows), i.e. predicted points from shots given the realised shot class mix.
    return {"ll": np.log(np.clip(p[np.arange(len(y)), y], EPS, None)), "p": p, "pmake": pm, "ok_make": ok,
            "llm": llm}


# =====================================================================================================================
# offline line machinery
# =====================================================================================================================
class LinesA:
    def __init__(self, t: pd.DataFrame, period: int, pos_mask: np.ndarray):
        sel = t["period"].to_numpy() == period
        self.sel = sel
        tt = t[sel].reset_index(drop=True)
        self.t = tt
        g, self.gi = np.unique(tt["game_id"].to_numpy(), return_inverse=True)
        self.G = len(g)
        one = np.ones(len(tt))
        gs = EG.gsum
        self.n_g = np.bincount(self.gi, minlength=self.G).astype(float)
        self.pos = pos_mask[sel]
        self.npos_g = np.bincount(self.gi, weights=self.pos.astype(float), minlength=self.G)
        y = tt["y"].to_numpy(float)
        self.y_g = np.bincount(self.gi, weights=y, minlength=self.G)
        rb, tb = tt["rb"].to_numpy(), tt["tb"].to_numpy()
        self.rb = rb
        self.cell = rb * W.N_TB + tb
        self.cell_n = gs(self.gi, self.cell, one, self.G, 45)
        self.cell_y = gs(self.gi, self.cell, y, self.G, 45)
        tot = self.cell_n.sum(0)
        self.powered = tot >= MIN_CELL
        self.populated = tot > 0
        self.rb_n = gs(self.gi, rb, one, self.G, 9)
        self.rb_y = gs(self.gi, rb, y, self.G, 9)
        role = W.role(tt["m"].to_numpy())
        self.q = {}
        for nm, col in (("tempo", "tempo"), ("zoff", "zoff"), ("defz", "defz")):
            q = quintile(tt[col].to_numpy())
            self.q[nm] = (q, gs(self.gi, q, one, self.G, 5), gs(self.gi, q, y, self.G, 5))
        sq = quintile(tt["strength"].to_numpy())
        self.sk = np.where(role > 0, sq, np.where(role < 0, 5 + sq, 10))
        self.sk_n = gs(self.gi, self.sk, one, self.G, 11)
        self.sk_y = gs(self.gi, self.sk, y, self.G, 11)

    def sums(self, r: dict) -> dict:
        gs = EG.gsum
        e = r["e"][self.sel]
        ll = r["ll"][self.sel]
        s = {"ll": np.bincount(self.gi, weights=ll, minlength=self.G),
             "llpos": np.bincount(self.gi, weights=np.where(self.pos, ll, 0.0), minlength=self.G),
             "e_g": np.bincount(self.gi, weights=e, minlength=self.G),
             "cell_e": gs(self.gi, self.cell, e, self.G, 45), "rb_e": gs(self.gi, self.rb, e, self.G, 9),
             "sk_e": gs(self.gi, self.sk, e, self.G, 11)}
        for nm, (q, _, _) in self.q.items():
            s[f"q_{nm}"] = gs(self.gi, q, e, self.G, 5)
        if "ll_end" in r:
            s["ll_end"] = np.bincount(self.gi, weights=r["ll_end"][self.sel], minlength=self.G)
        return s

    def lines(self, s: dict, w: np.ndarray) -> dict:
        n = w @ self.n_g
        out = {"dev": -2.0 * (w @ s["ll"]) / n, "dev_pos": -2.0 * (w @ s["llpos"]) / (w @ self.npos_g)}
        if "ll_end" in s:
            out["dev_end"] = -2.0 * (w @ s["ll_end"]) / n
        cn, cy, ce = w @ self.cell_n, w @ self.cell_y, w @ s["cell_e"]
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.abs(ce / cn - cy / cn)
        out["cal"] = float(np.nanmean(gap[self.powered]))
        out["cal45"] = float(np.nanmean(gap[self.populated]))
        rn, ry, re_ = w @ self.rb_n, w @ self.rb_y, w @ s["rb_e"]
        with np.errstate(invalid="ignore", divide="ignore"):
            am, mm = ry / rn, re_ / rn
        out["slope_lead"] = EG.ols_slope(am[0:4], mm[0:4])
        out["slope_trail"] = EG.ols_slope(am[4:8], mm[4:8])
        for nm, (_, qn, qy) in self.q.items():
            a, b, c = w @ qn, w @ qy, w @ s[f"q_{nm}"]
            out[f"slope_{nm}"] = EG.ols_slope(b / a, c / a)
        kn, ky, ke = w @ self.sk_n, w @ self.sk_y, w @ s["sk_e"]
        out["slope_team_lead"] = EG.ols_slope(ky[0:5] / kn[0:5], ke[0:5] / kn[0:5])
        out["slope_team_trail"] = EG.ols_slope(ky[5:10] / kn[5:10], ke[5:10] / kn[5:10])
        m = w > 0
        out["pergame_abs_err"] = float((w[m] * np.abs(s["e_g"][m] - self.y_g[m]) / self.n_g[m]).sum() / w[m].sum())
        return {k: float(v) for k, v in out.items()}


class LinesB:
    def __init__(self, t: pd.DataFrame, period: int, cell: int):
        sel = (t["period"].to_numpy() == period) & (t["cell"].to_numpy() == cell)
        self.sel = sel
        tt = t[sel].reset_index(drop=True)
        self.t = tt
        g, self.gi = np.unique(tt["game_id"].to_numpy(), return_inverse=True)
        self.G = len(g)
        self.n_g = np.bincount(self.gi, minlength=self.G).astype(float)
        y = tt["y"].to_numpy()
        self.y = y
        mk = tt["made"].to_numpy()
        self.sq = quintile(tt["strength"].to_numpy())
        gs = EG.gsum
        one = np.ones(len(tt))
        self.sq_n = gs(self.gi, self.sq, one, self.G, 5)
        self.sq_3 = gs(self.gi, self.sq, (y == 3).astype(float), self.G, 5)
        self.act = {"share_3pa": np.bincount(self.gi, weights=(y == 3).astype(float), minlength=self.G),
                    "share_2pa": np.bincount(self.gi, weights=np.isin(y, (1, 2)).astype(float), minlength=self.G),
                    "share_tov": np.bincount(self.gi, weights=(y == 0).astype(float), minlength=self.G),
                    "share_ft": np.bincount(self.gi, weights=(y >= 4).astype(float), minlength=self.G)}
        self.mk = mk

    def sums(self, r: dict) -> dict:
        p = r["p"][self.sel]
        ok = r["ok_make"][self.sel]
        pm = np.where(ok, r["pmake"][self.sel], 0.0)
        mk = np.where(ok, self.mk, 0.0)
        gs = EG.gsum
        val = np.array([SHOT_IDX.get(int(c), 0.0) for c in self.y])
        s = {"ll": np.bincount(self.gi, weights=r["ll"][self.sel], minlength=self.G),
             "llm": np.bincount(self.gi, weights=r["llm"][self.sel], minlength=self.G),
             "nm": np.bincount(self.gi, weights=ok.astype(float), minlength=self.G),
             "pm": np.bincount(self.gi, weights=pm, minlength=self.G),
             "am": np.bincount(self.gi, weights=mk, minlength=self.G),
             "pts_pred": np.bincount(self.gi, weights=np.where(ok, val * pm, 0.0), minlength=self.G),
             "pts_act": np.bincount(self.gi, weights=np.where(ok, val * mk, 0.0), minlength=self.G),
             "share_3pa": np.bincount(self.gi, weights=p[:, 3], minlength=self.G),
             "share_2pa": np.bincount(self.gi, weights=p[:, 1] + p[:, 2], minlength=self.G),
             "share_tov": np.bincount(self.gi, weights=p[:, 0], minlength=self.G),
             "share_ft": np.bincount(self.gi, weights=p[:, 4] + p[:, 5], minlength=self.G),
             "sq_3": gs(self.gi, self.sq, p[:, 3], self.G, 5),
             "sq_nm": gs(self.gi, self.sq, ok.astype(float), self.G, 5),
             "sq_pm": gs(self.gi, self.sq, pm, self.G, 5),
             "sq_am": gs(self.gi, self.sq, mk, self.G, 5)}
        return s

    def lines(self, s: dict, w: np.ndarray) -> dict:
        n = w @ self.n_g
        nm = w @ s["nm"]
        out = {"mll": -(w @ s["ll"]) / n, "make_dev": -2.0 * (w @ s["llm"]) / max(nm, 1e-12),
               "make_rate": (w @ s["pm"]) / max(nm, 1e-12), "act_make_rate": (w @ s["am"]) / max(nm, 1e-12),
               "shot_pts_per_shot": (w @ s["pts_pred"]) / max(nm, 1e-12),
               "act_shot_pts_per_shot": (w @ s["pts_act"]) / max(nm, 1e-12)}
        for k in ("share_3pa", "share_2pa", "share_tov", "share_ft"):
            out[k] = (w @ s[k]) / n
            out[f"act_{k}"] = (w @ self.act[k]) / n
        qn = w @ self.sq_n
        out["slope_team_3pa"] = EG.ols_slope((w @ self.sq_3) / qn, (w @ s["sq_3"]) / qn)
        qm = w @ s["sq_nm"]
        out["slope_team_make"] = EG.ols_slope((w @ s["sq_am"]) / qm, (w @ s["sq_pm"]) / qm)
        return {k: float(v) for k, v in out.items()}


def cell_table_A(t: pd.DataFrame, rows: dict, period: int) -> list[dict]:
    sel = t["period"].to_numpy() == period
    tt = t[sel]
    out = []
    for rb in range(9):
        for tb in range(5):
            m = (tt["rb"].to_numpy() == rb) & (tt["tb"].to_numpy() == tb)
            if not m.any():
                continue
            out.append({"cell": f"{W.ROLEBAND_LABELS[rb]} | {W.BUCKET_LABELS[tb]}", "n": int(m.sum()),
                        "actual_s": float(tt["y"].to_numpy()[m].mean()),
                        **{a: float(r["e"][sel][m].mean()) for a, r in rows.items()},
                        "underpowered": bool(m.sum() < MIN_CELL)})
    return out


def start_table_A(t: pd.DataFrame, rows: dict, period: int) -> list[dict]:
    sel = t["period"].to_numpy() == period
    tt = t[sel]
    out = []
    for st in sorted(tt["start"].unique()):
        m = tt["start"].to_numpy() == st
        out.append({"start": st, "n": int(m.sum()), "actual_s": float(tt["y"].to_numpy()[m].mean()),
                    **{a: float(r["e"][sel][m].mean()) for a, r in rows.items()},
                    "underpowered": bool(m.sum() < MIN_CELL)})
    return out


def team_sign_A(t: pd.DataFrame, rows: dict, ctrl: str, period: int) -> dict:
    sel = t["period"].to_numpy() == period
    tt = t[sel]
    df = pd.DataFrame({"team": tt["team"].to_numpy(), "y": tt["y"].to_numpy(float),
                       **{a: r["e"][sel] for a, r in rows.items()}})
    g = df.groupby("team")
    n = g.size()
    mu = g.mean()[n >= MIN_CELL]
    out = {"n_teams": int(len(mu))}
    for a in rows:
        if a == ctrl:
            continue
        sgn = np.sign(mu[a] - mu[ctrl]) == np.sign(mu["y"] - mu[ctrl])
        out[a] = {"right_sign_share": float(sgn.mean()),
                  "abs_err_down_share": float(((mu[a] - mu["y"]).abs() < (mu[ctrl] - mu["y"]).abs()).mean())}
    return out


def compare(val: dict, boot: dict, ctrl: str, noise: str | None, arms: list[str]) -> dict:
    """Per arm and line: value, delta vs control, floor = max(reseed difference, 2 x bootstrap SE)."""
    comp = {}
    for a in arms:
        if a == ctrl:
            continue
        d = {}
        for k in val[a]:
            db = np.array([boot[a][b][k] - boot[ctrl][b][k] for b in range(N_BOOT)])
            se = float(np.nanstd(db, ddof=1))
            rs = abs(val[noise][k] - val[ctrl][k]) if (noise in val and k in val[noise]) else 0.0
            fl = float(max(rs, 2 * se))
            d[k] = {"arm": val[a][k], "ctrl": val[ctrl][k], "delta": val[a][k] - val[ctrl][k], "floor": fl,
                    "boot_se": se, "reseed_diff": rs}
            if k.startswith("slope_"):
                da = [abs(boot[a][b][k] - 1) - abs(boot[ctrl][b][k] - 1) for b in range(N_BOOT)]
                d[k]["d_abs_err"] = abs(val[a][k] - 1) - abs(val[ctrl][k] - 1)
                d[k]["floor_abs_err"] = float(max(abs(abs(val[noise][k] - 1) - abs(val[ctrl][k] - 1))
                                                  if noise in val else 0.0, 2 * float(np.nanstd(da, ddof=1))))
        comp[a] = d
    return comp


def offline_fold(block: str, fold: str, arms: list[str], periods=(2, 1)) -> dict:
    ctrl, noise = CONTROL[block], NOISE[block]
    if block == "A":
        t = truth_A(fold)
        rows = {a: rows_A(t, load_pred("A", fold, a, t)) for a in arms}
        pos = rows[ctrl]["raw"] > EPS
        res = {"n_rows": int(len(t)), "periods": {}}
        for per in periods:
            L = LinesA(t, per, pos)
            wb = EG.boot_weights(L.G)
            one = np.ones(L.G)
            sums = {a: L.sums(r) for a, r in rows.items()}
            val = {a: L.lines(s, one) for a, s in sums.items()}
            boot = {a: [L.lines(s, wb[b]) for b in range(N_BOOT)] for a, s in sums.items()}
            res["periods"][str(per)] = {
                "n_rows": int(L.sel.sum()), "n_games": int(L.G), "ctrl": val[ctrl],
                "excluded_zero_mass_rows": int((~L.pos).sum()), "excluded_share": float((~L.pos).mean()),
                "excluded_dev_contribution": float(-2.0 * rows[ctrl]["ll"][L.sel][~L.pos].sum() / L.sel.sum()),
                "compare": compare(val, boot, ctrl, noise, arms),
                "cells": cell_table_A(t, rows, per), "start_types": start_table_A(t, rows, per),
                "team_sign": team_sign_A(t, rows, ctrl, per),
                "n_cells_powered": int(L.powered.sum()), "n_cells_populated": int(L.populated.sum())}
        return res
    t = truth_B(fold)
    preds = {a: load_pred("B", fold, a, t) for a in arms}
    masks = {a: np.isfinite(np.asarray(p["pmake"], float)) for a, p in preds.items()}
    for a, mk in masks.items():
        if not np.array_equal(mk, masks[ctrl]):
            raise SystemExit(f"{fold}/B/{a}: pmake NaN mask differs from the control's")
    rows = {a: rows_B(t, p) for a, p in preds.items()}
    res = {"n_rows": int(len(t)), "make_rows_matched": int(masks[ctrl].sum()),
           "fga_rows": int(np.isfinite(t["made"].to_numpy()).sum()), "periods": {}}
    for per in periods:
        res["periods"][str(per)] = {}
        for ci, cname in enumerate(FPS_CELLS):
            L = LinesB(t, per, ci)
            if L.G < 2:
                continue
            wb = EG.boot_weights(L.G)
            one = np.ones(L.G)
            sums = {a: L.sums(r) for a, r in rows.items()}
            val = {a: L.lines(s, one) for a, s in sums.items()}
            boot = {a: [L.lines(s, wb[b]) for b in range(N_BOOT)] for a, s in sums.items()}
            res["periods"][str(per)][cname] = {
                "n_rows": int(L.sel.sum()), "n_games": int(L.G),
                "n_make_rows": int(rows[ctrl]["ok_make"][L.sel].sum()),
                "underpowered_slopes": bool(L.sel.sum() < MIN_SLOPE),
                "underpowered_cell": bool(L.sel.sum() < MIN_CELL), "ctrl": val[ctrl],
                "compare": compare(val, boot, ctrl, noise, arms)}
    return res


def decide(block: str, res: dict) -> dict:
    """The literal s46 4.1 rule on the unblinded table."""
    def f2c(a):
        return (res["folds"]["F2"]["periods"]["2"]["compare"] if block == "A"
                else res["folds"]["F2"]["periods"]["2"]["trail 1-3"]["compare"]).get(a)

    def f1c(a):
        f = res["folds"].get("F1")
        if not f:
            return None
        return (f["periods"]["2"]["compare"] if block == "A" else f["periods"]["2"]["trail 1-3"]["compare"]).get(a)
    out = {"per_arm": {}}
    winners = []
    for a in res["arms"]:
        if a == CONTROL[block]:
            continue
        d = f2c(a)
        beats = {p: bool(d[p]["delta"] < -d[p]["floor"]) for p in PRIM[block]}
        hold = {s: bool(d[s]["delta"] >= -d[s]["floor"]) for s in GATES_NOT_BELOW[block]}
        hold.update({s: bool(d[s]["d_abs_err"] <= d[s]["floor_abs_err"]) for s in GATES_ABS1[block]})
        c1 = f1c(a)
        f1sign = {p: (bool(np.sign(c1[p]["delta"]) == np.sign(d[p]["delta"])) if c1 else None) for p in PRIM[block]}
        noise = a == NOISE[block]
        win = (not noise) and all(beats.values()) and all(hold.values()) and all(v is not False for v in f1sign.values())
        rec = {"beats_ctrl_by_floor": beats, "gates_hold": hold, "f1_keeps_sign": f1sign, "noise_floor_arm": noise,
               "primary_floors": {p: (-d[p]["delta"] / d[p]["floor"]) if d[p]["floor"] else None for p in PRIM[block]},
               "wins": bool(win)}
        if block == "A":
            rec["dual_deviance"] = {"dev_delta": d["dev"]["delta"], "dev_floor": d["dev"]["floor"],
                                    "dev_pos_delta": d["dev_pos"]["delta"], "dev_pos_floor": d["dev_pos"]["floor"],
                                    "full_set_win_survives_on_positive_mass": bool(d["dev_pos"]["delta"]
                                                                                    < -d["dev_pos"]["floor"])}
        if block == "B":
            changed = ("mll",) if a == "FP1" else ("mll", "make_dev")
            other = tuple(p for p in PRIM["B"] if p not in changed)
            rec["reported_only_changed_law_reading"] = bool(
                not noise and all(d[p]["delta"] < -d[p]["floor"] for p in changed)
                and all(d[p]["delta"] <= d[p]["floor"] for p in other) and all(hold.values())
                and all(f1sign[p] is not False for p in changed))
        out["per_arm"][a] = rec
        if win:
            winners.append(a)
    sel = None
    if winners:
        best = min(winners, key=lambda a: f2c(a)[PRIM[block][0]]["arm"])
        bd = f2c(best)
        ok = [a for a in winners if all(f2c(a)[p]["arm"] - bd[p]["arm"] <= bd[p]["floor"] for p in PRIM[block])]
        sel = min(ok, key=lambda a: RANK[a])
    out.update({"winners": winners, "selected": sel,
                "rule": "s46 4.1: beats control by > 1 floor on every block primary on F2 period 2, every "
                        "responsiveness gate holds, dual-deviance gate (A), F1 keeps the sign; simplest winner "
                        "within one floor of the best; no winner = block REFUTED"})
    return out


def offline(block: str, arms: list[str] | None) -> int:
    found = sorted(p.stem for p in (OFF_DIR / "F2" / block).glob("*.npz"))
    arms = arms or [a for a in found if "_m" not in a]
    if CONTROL[block] not in arms:
        raise SystemExit(f"block {block}: control has no prediction file")
    codes = code_arms(block, arms)
    res: dict = {"block": block, "spec": "docs/models/clock/experiments.md s46", "folds": {}}
    for fold in ("F2", "F1"):
        have = [a for a in arms if (OFF_DIR / fold / block / f"{a}.npz").exists()]
        res["folds"][fold] = offline_fold(block, fold, have)
    coded = recode(res, codes)
    GRADE_DIR.mkdir(parents=True, exist_ok=True)
    (GRADE_DIR / f"offline_{block}_coded.json").write_text(json.dumps(coded, indent=1, default=float), encoding="utf-8")
    print(f"coded table written: {GRADE_DIR / f'offline_{block}_coded.json'}", flush=True)
    key = {c: a for a, c in codes.items()}
    (GRADE_DIR / f"offline_{block}_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    res["arms"] = arms
    res["key"] = key
    res["decision"] = decide(block, res)
    res["written_at"] = datetime.now(UTC).isoformat()
    (GRADE_DIR / f"offline_{block}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["decision"], indent=1, default=float))
    return 0


def recode(obj, codes: dict):
    """Replace arm names used as dict keys by their codes (deep)."""
    if isinstance(obj, dict):
        return {codes.get(k, k): recode(v, codes) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recode(v, codes) for v in obj]
    return obj


def select_m() -> int:
    """Ruling 7: m in {20, 50, 100} per shrunk arm, on fold 1 only."""
    out = {"rule": "lowest F1 period-2 dev_pos; among m within one F1 floor of the best, the largest m", "arms": {}}
    for arm in ("CA1", "CA2"):
        names = [f"{arm}_m{m}" for m in M_GRID]
        have = [n for n in names if (OFF_DIR / "F1" / "A" / f"{n}.npz").exists()]
        if len(have) != len(M_GRID):
            raise SystemExit(f"{arm}: fold-1 grid incomplete ({have})")
        r = offline_fold("A", "F1", ["CA0", "CA0s1", *have], periods=(2,))
        c = r["periods"]["2"]["compare"]
        best = min(have, key=lambda n: c[n]["dev_pos"]["arm"])
        ok = [n for n in have if c[n]["dev_pos"]["arm"] - c[best]["dev_pos"]["arm"] <= c[best]["dev_pos"]["floor"]]
        chosen = max(ok, key=lambda n: int(n.split("_m")[1]))
        out["arms"][arm] = {"m": int(chosen.split("_m")[1]), "best_by_dev_pos": best, "within_floor": ok,
                            "f1_lines": {n: {k: c[n][k] for k in ("dev_pos", "dev", "cal", "slope_tempo")} for n in have}}
        print(arm, "->", out["arms"][arm]["m"], {n: round(c[n]["dev_pos"]["arm"], 5) for n in have}, flush=True)
    out["written_at"] = datetime.now(UTC).isoformat()
    GRADE_DIR.mkdir(parents=True, exist_ok=True)
    p = GRADE_DIR / "m_selection.json"
    if p.exists() and os.environ.get("EGC_OVERWRITE") != "1":
        raise SystemExit(f"{p} exists (frozen); EGC_OVERWRITE=1 only for an own rerun before any F2 fit")
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


# =====================================================================================================================
# MODE `offline --block C`  (s46 2.C, 2.D, 4.1 and rulings 5, 6, 10; added 2026-10-14 BEFORE any block-C fit existed)
# =====================================================================================================================
# Rows, scoring and every round-9 line are IMPORTED UNCHANGED from `grade_clock_env_v1.py` (load_rows, score_arm, Agg,
# segment_lines, team_sign_share): regulation possessions of clock-complete test-season games, each arm's IMPLIED
# possession time given the realised chances, game-level implied possessions pm = 1200 n / sum t (actual pa = 1200 n /
# sum of actual durations). Arms: E0 = served K2 (control), E0s1 = K2 refit seed 1, ED/EG/EM/EH =
# `clock/s46C_<arm>/<fold>`, <arm>s1 = the same under --seed 1 (noise-floor arms, never selectable).
# Lines s46 adds (game-level; per-game sufficient statistics so the same 200 game-block bootstrap weights apply):
#   ratio_env         round 9 / round 10 primary in offline form: slope(pm on env) / slope(pa on env). C-pre (i).
#   ratio_env_h1      the same on first-half rows only. C-pre (iv).
#   xq51              offence-team quintile (team mean of x_off = off_c(offence) + def_c(defence) over its test rows)
#                     of game possessions per team-game, Q5 mean - Q1 mean (round-10 loop team-table definition, the
#                     game-level count assigned to each offence's team-game). C-pre (ii): model <= actual + floor,
#                     actual = the offline actual on the same games (binding); the loop's literal +0.35 is printed.
#   envq{k}_gap       game env quintile k (edges over the fold's test games): mean(pa - pm). C-pre (iii): no quintile
#                     with |arm gap| - |control gap| > its floor.
#   g1_sd             SD over games of pm (actual: of pa). Offline form of the G1 possession SD co-primary; the
#                     expectation SD lacks the within-game sampling term, so the loop translation
#                     sqrt(5.629^2 + var_arm - var_ctrl) is REPORTED beside it (never a selector offline).
#   ratio_tot         offline form of the round-10 total-on-env primary: implied total = 2 pm x the game's realised
#                     points per possession (the clock does not change PPP); slope ratio on env. REPORTED.
#   pergame_abs       mean over games |pm - pa| (multi-level, per game).
# C-pre (s46 2.C as tightened by ruling 6; fold 2; hard gate; an arm failing any item is OUT, no refit to pass):
#   (i) 0.7 <= ratio_env <= 1.15; (ii) xq51 <= actual xq51 + floor; (iii) no env quintile further from 0 than the
#   control's beyond its floor; (iv) 0.7 <= ratio_env_h1 <= 1.15; (v) deviance on the control's positive-mass rows not
#   worse than the control's beyond its floor. Fold 1 confirms the sign of the env move (reported).
# Offline vetoes REPORTED with C-pre (round-9 41.A s5 vetoes and responsiveness holds): |count gap| worse by > 0.10;
#   any start type |gap| worse by > 0.20 s; any Nov-Mar month worse by > 0.25 s; offence-tempo quintile slope and game
#   tempo elasticity not below the control beyond floor. The section-45 hard-veto list is closed-loop only.
# Floor per line = max(|<arm>s1 - <arm>| reseed difference, 2 x SD of the paired game-block bootstrap of arm - E0
#   (200 draws, numpy seed 12345)). Blind: tables under coded labels, key after.
C_ARMS = ("EG", "ED", "EM", "EH")          # s46 rank (fewest degrees of freedom first)
C_LINES = ("deviance", "deviance_pos", "ratio_env", "ratio_env_h1", "xq51", "g1_sd", "ratio_tot", "pergame_abs",
           "slope_oq", "elast_ratio", "slope_zq", "slope_dq", "ratio_gap", "count_gap",
           *[f"envq{k}_gap" for k in range(1, 6)])
G1_LOOP_CTRL = 5.629


def _c_manifest(arm: str, fold: str) -> Path:
    ckd = ROOT / "data/processed/models/clock"
    if arm in ("E0", "E0s1"):
        return ckd / "r8_K2" / (fold if arm == "E0" else f"{fold}_seed1") / "manifest.json"
    base, s1 = (arm[:-2], True) if arm.endswith("s1") else (arm, False)
    return ckd / f"s46C_{base}" / (f"{fold}_seed1" if s1 else fold) / "manifest.json"


class LinesC:
    """Game-level C lines over G9 rows `d` (columns t_<arm>), indexed in G9.Agg's factorize order."""

    def __init__(self, d: pd.DataFrame, arms: list[str]):
        gi, games = pd.factorize(d["game_id"])
        self.G = len(games)
        bc = lambda v: np.bincount(gi, weights=v, minlength=self.G)  # noqa: E731
        h1 = d["period"].to_numpy() <= 1
        self.n, self.A = bc(np.ones(len(d))), bc(d["dur_poss"].to_numpy(np.float64))
        self.n1, self.A1 = bc(h1.astype(float)), bc(np.where(h1, d["dur_poss"].to_numpy(np.float64), 0.0))
        self.T = {a: bc(d[f"t_{a}"].to_numpy()) for a in arms}
        self.T1 = {a: bc(np.where(h1, d[f"t_{a}"].to_numpy(), 0.0)) for a in arms}
        first = d.groupby(gi).first()
        self.env = first["env"].to_numpy(np.float64)
        self.ppp = bc(d["points"].to_numpy(np.float64)) / self.n
        self.envq = quintile(self.env)
        tg = d.groupby(["game_id", "offense_team_id"], sort=False).agg(x=("z_raw", "mean")).reset_index()
        tm = tg.groupby("offense_team_id")["x"].mean()
        tq = pd.Series(quintile(tm.to_numpy()), index=tm.index)
        self.tg_g = pd.Index(games).get_indexer(tg["game_id"])
        self.tg_q = tg["offense_team_id"].map(tq).to_numpy(np.int64)
        self.envq_n = np.bincount(self.envq, minlength=5)
        self.tq_teams = np.bincount(tq.to_numpy(), minlength=5)
        self.tq_tg = np.bincount(self.tg_q, minlength=5)

    @staticmethod
    def _slope(y, x, w):
        m = w > 0
        xm = np.average(x[m], weights=w[m])
        ym = np.average(y[m], weights=w[m])
        return float(np.sum(w[m] * (x[m] - xm) * (y[m] - ym)) / np.sum(w[m] * (x[m] - xm) ** 2))

    def lines(self, a: str, w: np.ndarray) -> dict:
        m = w > 0
        pm, pa = 1200 * self.n / self.T[a], 1200 * self.n / self.A
        with np.errstate(invalid="ignore", divide="ignore"):
            pm1, pa1 = 1200 * self.n1 / self.T1[a], 1200 * self.n1 / self.A1
        ok1 = m & np.isfinite(pm1) & np.isfinite(pa1) & (self.n1 > 0)
        w1 = np.where(ok1, w, 0.0)
        pm1, pa1 = np.nan_to_num(pm1, posinf=0.0), np.nan_to_num(pa1, posinf=0.0)
        out = {"ratio_env": self._slope(pm, self.env, w) / self._slope(pa, self.env, w),
               "slope_env_model": self._slope(pm, self.env, w), "slope_env_act": self._slope(pa, self.env, w),
               "ratio_env_h1": self._slope(pm1, self.env, w1) / self._slope(pa1, self.env, w1),
               "ratio_tot": self._slope(2 * pm * self.ppp, self.env, w) / self._slope(2 * pa * self.ppp, self.env, w),
               "pergame_abs": float(np.average(np.abs(pm - pa)[m], weights=w[m]))}
        mu = np.average(pm[m], weights=w[m])
        mua = np.average(pa[m], weights=w[m])
        out["g1_var"] = float(np.average((pm[m] - mu) ** 2, weights=w[m]))
        out["g1_sd"] = float(np.sqrt(out["g1_var"]))
        out["act_g1_sd"] = float(np.sqrt(np.average((pa[m] - mua) ** 2, weights=w[m])))
        for k in range(5):
            s = m & (self.envq == k)
            out[f"envq{k + 1}_gap"] = float(np.average((pa - pm)[s], weights=w[s]))
            out[f"envq{k + 1}_pm"] = float(np.average(pm[s], weights=w[s]))
            out[f"envq{k + 1}_pa"] = float(np.average(pa[s], weights=w[s]))
        wt = w[self.tg_g]
        den = np.bincount(self.tg_q, weights=wt, minlength=5)
        qm = np.bincount(self.tg_q, weights=wt * pm[self.tg_g], minlength=5) / den
        qa = np.bincount(self.tg_q, weights=wt * pa[self.tg_g], minlength=5) / den
        out["xq51"], out["act_xq51"] = float(qm[4] - qm[0]), float(qa[4] - qa[0])
        for k in range(5):
            out[f"xq{k + 1}_pm"], out[f"xq{k + 1}_pa"] = float(qm[k]), float(qa[k])
        return out


def offline_C_fold(fold: str, arms: list[str], rows: pd.DataFrame) -> dict:
    import grade_clock_env_v1 as G9
    G8 = G9.G8
    d = rows
    arms = [a for a in arms if _c_manifest(a, fold).exists()]
    if "E0" not in arms:
        raise SystemExit("E0 (served K2) schedule missing")
    for a in arms:
        r = G9.score_arm(G8.load_schedule(_c_manifest(a, fold)), d)
        for k, v in r.items():
            d[f"{k}_{a}"] = v
        print(f"{fold} scored {a}", flush=True)
    pos = d["prp_E0"].to_numpy() > G9.CLIP
    AG = G9.Agg(d, arms, pos)
    LC = LinesC(d, arms)
    one = np.ones(AG.G)

    def all_lines(a, w):
        return {**AG.lines(a, w), **LC.lines(a, w)}
    val = {a: all_lines(a, one) for a in arms}
    for a in arms:
        st = d.groupby("prev_end").agg(m=(f"t_{a}", "mean"), act=("dur_poss", "mean"), n=("dur_poss", "size"))
        val[a]["gap_by_start"] = (st["m"] - st["act"]).to_dict()
        val[a]["n_by_start"] = st["n"].astype(int).to_dict()
        mo = d.groupby("month").agg(m=(f"t_{a}", "mean"), act=("dur_poss", "mean"), n=("dur_poss", "size"))
        val[a]["gap_by_month"] = {str(k): float(v) for k, v in (mo["m"] - mo["act"]).items()}
    rng = np.random.default_rng(BOOT_SEED)
    keys = [k for k in C_LINES if k in val["E0"]]
    boots = {a: {k: [] for k in keys} for a in arms}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, AG.G, AG.G), minlength=AG.G).astype(float)
        for a in arms:
            L = all_lines(a, w)
            for k in keys:
                boots[a][k].append(L[k])
    e0 = val["E0"]
    for a in arms:
        if a == "E0":
            continue
        V = {}
        for k in keys:
            V[f"d_{k}"] = val[a][k] - e0[k]
            V[f"boot2se_{k}"] = 2 * float(np.std(np.array(boots[a][k]) - np.array(boots["E0"][k])))
            dabs = np.abs(np.array(boots[a][k])) - np.abs(np.array(boots["E0"][k]))
            V[f"boot2se_abs_{k}"] = 2 * float(np.std(dabs))
        val[a]["vs_E0"] = V
    # reseed term and binding floors
    for a in arms:
        if a == "E0":
            continue
        s1 = None if a.endswith("s1") else f"{a}s1"
        V = val[a]["vs_E0"]
        for k in keys:
            rs = abs(val[s1][k] - val[a][k]) if s1 in val else 0.0
            V[f"reseed_{k}"] = rs
            V[f"floor_{k}"] = max(rs, V[f"boot2se_{k}"])
            V[f"floor_abs_{k}"] = max(rs, V[f"boot2se_abs_{k}"])
    fr = {"n_rows": int(len(d)), "n_games": int(AG.G), "e0_zero_mass_rows": int((~pos).sum()),
          "envq_games": LC.envq_n.tolist(), "xq_teams": LC.tq_teams.tolist(), "xq_team_games": LC.tq_tg.tolist(),
          "arms": val}
    fr["segments"] = G9.segment_lines(d, arms)
    fr["team_sign_share"] = G9.team_sign_share(d, arms)
    rp = {}
    for a in arms:
        if a.endswith("s1") and a[:-2] in arms:
            rp[f"{a[:-2]}_vs_{a}_max_abs_pmf_diff"] = G9.pmf_diff(G8.load_schedule(_c_manifest(a[:-2], fold)),
                                                                  G8.load_schedule(_c_manifest(a, fold)), d)
    fr["reseed_proof"] = rp
    return fr


def c_pre(F: dict) -> dict:
    f2, f1 = F["F2"], F.get("F1")
    out = {"rule": ("s46 2.C C-pre as tightened by ruling 6 (fold 2): (i) 0.7 <= ratio_env <= 1.15; (ii) xq51 <= actual "
                    "xq51 + floor; (iii) no env quintile with |arm gap| - |control gap| > floor; (iv) 0.7 <= "
                    "ratio_env_h1 <= 1.15; (v) deviance_pos not worse than control beyond floor. Any failure = OUT."),
           "arms": {}}
    e0 = f2["arms"]["E0"]
    for a in C_ARMS:
        if a not in f2["arms"]:
            continue
        L, V = f2["arms"][a], f2["arms"][a]["vs_E0"]
        c = {"i_ratio_env_in_0.7_1.15": bool(0.7 <= L["ratio_env"] <= 1.15),
             "ii_xq51_le_actual_plus_floor": bool(L["xq51"] <= L["act_xq51"] + V["floor_xq51"]),
             "iii_envq_not_further": bool(all(abs(L[f"envq{k}_gap"]) - abs(e0[f"envq{k}_gap"]) <= V[f"floor_abs_envq{k}_gap"]
                                              for k in range(1, 6))),
             "iv_ratio_env_h1_in_0.7_1.15": bool(0.7 <= L["ratio_env_h1"] <= 1.15),
             "v_dev_pos_not_worse": bool(V["d_deviance_pos"] <= V["floor_deviance_pos"])}
        veto = {"count_gap_worse_gt_0.10": bool(abs(L["count_gap"]) - abs(e0["count_gap"]) > 0.10),
                "start_type_worse_gt_0.20s": bool(max(abs(L["gap_by_start"][k]) - abs(e0["gap_by_start"][k])
                                                      for k in e0["gap_by_start"]) > 0.20),
                "month_worse_gt_0.25s": bool(max(abs(L["gap_by_month"][k]) - abs(e0["gap_by_month"][k])
                                                 for k in ("11", "12", "1", "2", "3") if k in e0["gap_by_month"]) > 0.25),
                "tempo_q_slope_below_ctrl": bool(V["d_slope_oq"] < -V["floor_slope_oq"]),
                "elasticity_below_ctrl": bool(V["d_elast_ratio"] < -V["floor_elast_ratio"])}
        g1 = {"offline_sd": L["g1_sd"], "ctrl_sd": e0["g1_sd"], "act_sd": L["act_g1_sd"], "d": V["d_g1_sd"],
              "floor": V["floor_g1_sd"],
              "loop_translation": float(np.sqrt(G1_LOOP_CTRL ** 2 + L["g1_var"] - e0["g1_var"]))}
        f1c = None
        if f1 is not None and a in f1["arms"]:
            L1, E1 = f1["arms"][a], f1["arms"]["E0"]
            f1c = {"ratio_env": L1["ratio_env"], "ctrl_ratio_env": E1["ratio_env"],
                   "env_move_same_sign_as_F2": bool(np.sign(L1["ratio_env"] - E1["ratio_env"]) ==
                                                    np.sign(L["ratio_env"] - e0["ratio_env"])),
                   "env_toward_1": bool(abs(L1["ratio_env"] - 1) < abs(E1["ratio_env"] - 1))}
        out["arms"][a] = {"checks": c, "C_pre": "PASS" if all(c.values()) else "FAIL",
                          "failed": [k for k, v in c.items() if not v], "offline_vetoes_reported": veto,
                          "g1_sd": g1, "F1": f1c}
    out["survivors"] = [a for a, r in out["arms"].items() if r["C_pre"] == "PASS"]
    return out


def offline_C(arms: list[str] | None) -> int:
    import grade_clock_env_v1 as G9
    arms = arms or ["E0", "E0s1", *C_ARMS, *[f"{a}s1" for a in C_ARMS]]
    rows = G9.load_rows()
    F = {}
    for fold in ("F2", "F1"):
        d = rows[fold]
        season = 2025 if fold == "F2" else 2024
        p4 = pd.read_parquet(G9.TK.P4 / f"possessions_{season}.parquet", columns=["game_id", "period", "poss_index", "points"])
        p4 = p4.rename(columns={"period": "_p", "poss_index": "_i"})
        p4["_p"], p4["_i"] = p4["_p"].astype("int64"), p4["_i"].astype("int64")
        d["_p"], d["_i"] = d["period"].astype("int64"), d["poss_index"].astype("int64")
        d = d.merge(p4, on=["game_id", "_p", "_i"], how="left", validate="1:1").drop(columns=["_p", "_i"])
        d["points"] = d["points"].fillna(0.0)
        F[fold] = offline_C_fold(fold, arms, d)
    dec = c_pre(F)
    codes = code_arms("C", [a for a in F["F2"]["arms"]])
    GRADE_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# s46 block C offline grade (coded arms; key below the tables)", ""]
    for fold in ("F2", "F1"):
        fr = F[fold]
        md += [f"## {fold}: {fr['n_rows']:,} rows, {fr['n_games']:,} games; control zero-mass rows {fr['e0_zero_mass_rows']}",
               "", "| arm | dev_pos (d; floor) | ratio_env (d; floor) | ratio_env_h1 | xq51 (act) | env Q1..Q5 gap | "
               "g1_sd (act) | ratio_tot | slope_oq | elast | pergame_abs |", "|---|---|---|---:|---|---|---|---:|---:|---:|---:|"]
        for a in sorted(fr["arms"], key=lambda x: codes.get(x, x)):
            L = fr["arms"][a]
            V = L.get("vs_E0", {})
            dp = f" ({V['d_deviance_pos']:+.4f}; {V['floor_deviance_pos']:.4f})" if V else ""
            de = f" ({V['d_ratio_env']:+.3f}; {V['floor_ratio_env']:.3f})" if V else ""
            gq = " / ".join(f"{L[f'envq{k}_gap']:+.2f}" for k in range(1, 6))
            md.append(f"| {codes.get(a, a)} | {L['deviance_pos']:.4f}{dp} | {L['ratio_env']:.3f}{de} | {L['ratio_env_h1']:.3f} | "
                      f"{L['xq51']:+.3f} ({L['act_xq51']:+.3f}) | {gq} | {L['g1_sd']:.3f} ({L['act_g1_sd']:.3f}) | "
                      f"{L['ratio_tot']:.3f} | {L['slope_oq']:.3f} | {L['elast_ratio']:.3f} | {L['pergame_abs']:.3f} |")
        md.append("")
    md += ["## Key (written after the coded tables)", "",
           *[f"- {v} = {k}" for k, v in sorted(codes.items(), key=lambda x: x[1])], "",
           "## C-pre (literal)", "", f"Rule: {dec['rule']}", "", f"Survivors: {dec['survivors']}", ""]
    for a, r in dec["arms"].items():
        md.append(f"- {a}: C-pre {r['C_pre']}; failed {r['failed']}; vetoes {r['offline_vetoes_reported']}")
    rep = {"spec": "docs/models/clock/experiments.md s46 block C", "codes": codes, "folds": F, "c_pre": dec,
           "written_at": datetime.now(UTC).isoformat()}
    (GRADE_DIR / "offline_C.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    (GRADE_DIR / "offline_C_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("C-pre survivors:", dec["survivors"], flush=True)
    for a, r in dec["arms"].items():
        print(a, r["C_pre"], r["failed"], flush=True)
    return 0


# =====================================================================================================================
# closed loop / full size
# =====================================================================================================================
PRIMARY_LOOP = {"A": [("MD4_h2_last4_poss", "act_MD4_h2_last4_poss"), ("MD5_pts13_last2", "act_MD5_pts13_last2")],
                "B": [("tie_rate", "act_tie_rate"), ("d13_ppp", "act_d13_ppp")]}


def r2_veto(rows: dict, act: dict) -> dict:
    """R2 as a no-regression veto for blocks A and B (module docstring)."""
    import grade_clock_env_loop_v1 as GL
    h = GL.toward(rows["H1_poss_on_env"], act["act_H1_poss_on_env"])
    tv, c, a = h["target"], h["ctrl"], h["arm"]
    shortfall = abs(tv - c)
    overshoot = (a - tv) if (tv - c) > 0 else (tv - a)
    return {**h, "overshoot_beyond_actual": overshoot, "ctrl_shortfall": shortfall,
            "pass": bool((not h["away_by_floor"]) and not (overshoot > shortfall))}


def r4_veto(rows: dict) -> dict:
    cells, n_bad = [], 0
    for k in range(1, 6):
        r = rows[f"envq{k}_poss_gap"]
        worse = bool(abs(r["arm"]) - abs(r["ctrl"]) > r["floor"])
        n_bad += worse
        cells.append({"q": k, "ctrl": r["ctrl"], "arm": r["arm"], "floor": r["floor"], "further_beyond_floor": worse})
    return {"cells": cells, "n_further": n_bad, "pass": bool(n_bad <= 2)}


def judge_loop(block: str, rows: dict, act: dict, ctrl_lines: dict) -> dict:
    import grade_clock_env_loop_v1 as GL
    g = GL.judge(rows, act)
    vet = {k: v for k, v in g["vetoes"].items() if k != "H1_poss_on_env"}
    vet["R2_H1_env"] = r2_veto(rows, act)
    vet["R4_envq_poss_gap"] = r4_veto(rows)
    e = EG.judge(rows, act, ctrl_lines, "C")
    md = e["md"]
    prim = {k: GL.toward(rows[k], act[t]) for k, t in PRIMARY_LOOP[block]}
    moves = all(x["toward_by_floor"] for x in prim.values())
    if block == "B":
        moves = moves and bool(rows["tie_rate"]["arm"] >= 0.035)
    v = {"block": block, "primary": prim, "primary_moves": bool(moves), "vetoes": vet,
         "vetoes_failing": [k for k, x in vet.items() if not x["pass"]], "md": md,
         "md_failing": [k for k, x in md.items() if not x["pass"]],
         "reported": {**g["reported"], "G1_poss_sd_header": rows["G1_poss_sd"],
                      "bonus": {k: rows[k] for k in ("lead13_bonus", "lead46_bonus", "lead7p_bonus") if k in rows},
                      "d13_3pa": rows.get("d13_3pa")}}
    v["passes"] = bool(v["primary_moves"] and not v["vetoes_failing"] and not v["md_failing"])
    return v


def seed_check_loop(block: str, ctrl: dict, act: dict, draw_sd: dict) -> dict:
    lines = dict(PRIMARY_LOOP[block])
    lines.update({"MD1_gap": 0.0, "MD2_q1": 1.0, "tie_rate": "act_tie_rate", "MD4_h2_last4_poss": "act_MD4_h2_last4_poss",
                  "MD5_pts13_last2": "act_MD5_pts13_last2"})
    out = {}
    for k, t in lines.items():
        tv = act[t] if isinstance(t, str) else t
        gap = abs(ctrl[k] - tv)
        out[k] = {"floor_draws": 2 * draw_sd[k], "gap": gap, "resolved": bool(2 * draw_sd[k] < gap)}
    out["all_resolved"] = all(v["resolved"] for v in out.values() if isinstance(v, dict))
    return out


def _finish(a, LN, val: dict, frames: dict, info: dict, lines_fn, mode: str) -> int:
    import grade_clock_env_loop_v1 as GL
    tags = [a.control, *a.draws, a.arm]
    if info[a.arm].get("seeds") != info[a.control].get("seeds"):
        raise SystemExit("arm and control seeds differ (pairing broken)")
    act = {k: v for k, v in val[a.control].items() if k.startswith("act_")}
    names = [k for k in val[a.control] if not k.startswith("act_")]
    draw_sd = {k: float(np.nanstd([val[t][k] for t in a.draws], ddof=1)) for k in names}
    G = len(LN.ids)
    wb = EG.boot_weights(G)
    diffs = {k: [] for k in names}
    for b in range(N_BOOT):
        la, lc = lines_fn(a.arm, wb[b]), lines_fn(a.control, wb[b])
        for k in names:
            src = GL.DISP_REP.get(k, k)
            if src in la:
                diffs[k].append(la[src] - lc[src])
    rows = {}
    for k in names:
        se = float(np.nanstd(diffs[k], ddof=1)) if diffs[k] else float("nan")
        rows[k] = {"arm": val[a.arm][k], "ctrl": val[a.control][k], "delta": val[a.arm][k] - val[a.control][k],
                   "draw_sd": draw_sd[k], "boot_se": se, "floor": float(np.nanmax([2 * draw_sd[k], 2 * se]))}
    code = "L-" + chr(65 + int(np.random.default_rng(CODE_SEED).integers(0, 26)))
    res = {"spec": "docs/models/clock/experiments.md s46", "mode": mode, "block": a.block, "control": a.control,
           "draws": a.draws, "arm_code": code, "n_games": int(G), "n_pbp_ok": int(LN.LL.ok.sum()),
           "info": {t: info[t] for t in tags}, "actual": act, "rows": rows, "draw_lines": {t: val[t] for t in a.draws}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_name(out.stem + "_coded.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    res["arm"] = a.arm
    res["seed_check"] = seed_check_loop(a.block, val[a.control], act, draw_sd)
    if getattr(a, "spec", "s46") == "s49":
        v = s49_judge(rows, act, val[a.control], mode)
        res["spec"] = "docs/models/clock/experiments.md s49"
    else:
        v = judge_loop(a.block, rows, act, val[a.control])
        if mode == "loop":
            v["screen"] = "proceed to full size" if v["passes"] else "stop"
        else:
            v["full_verdict"] = "CANDIDATE" if v["passes"] else "NOT A CANDIDATE"
    res["verdict"] = v
    res["quintiles_env"] = [{"q": k, **{f"poss_gap_{x}": rows[f"envq{k}_poss_gap"][x] for x in ("ctrl", "arm", "floor")}}
                            for k in range(1, 6)]
    res["written_at"] = datetime.now(UTC).isoformat()
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(mode, a.arm, v.get("screen", v.get("full_verdict")), "| primary_moves", v.get("primary_moves"),
          "| vetoes failing", v["vetoes_failing"], "| md failing", v["md_failing"], flush=True)
    for k, x in v.get("primary", v.get("full_primaries", v.get("proceed_lines", {}))).items():
        print(f"  {k}: act {x['target']:.4f} ctrl {x['ctrl']:.4f} arm {x['arm']:.4f} floor {x['floor']:.4f} "
              f"floors {x['floors']}", flush=True)
    print("seed check:", json.dumps(res["seed_check"]), flush=True)
    return 0


def loop(a) -> int:
    import diag_c4_lib_v1 as L
    import grade_clock_env_loop_v1 as GL
    import grade_clock_r8_loop_v1 as R8
    ld = ROOT / a.loop_dir
    EG.LOOP_DIR = ld
    GL.LOOP_DIR = ld
    GL.DISP_DIR = ROOT / a.disp_dir
    R8.EV = ld
    ids = np.sort(pd.read_parquet(EG.SAMPLE)["game_id"].astype("int64").to_numpy())
    real = EG._real_side(ids)
    pgi = EG._pregame(ids)
    LN = GL.Lines(ids, real, pgi)
    frames, info = {}, {}
    for t in [a.control, *a.draws, a.arm]:
        pg, gm, inf = EG.run_frame(t, ids)
        tr = pd.read_parquet(EG.TRAJ_DIR / t / "trajectory.parquet")
        tr = tr[tr["game_id"].isin(set(ids.tolist()))]
        pg = EG.add_seed_level(pg, L.std_from_sim(tr), ids)
        sx = GL.sim_extras(tr, ids)
        del tr
        g8 = R8.per_game(R8.load_run(t))
        frames[t] = (pg, gm, sx, g8)
        info[t] = inf
        if inf["n_seeds"] != a.seeds:
            raise SystemExit(f"{t}: {inf['n_seeds']} seeds, expected {a.seeds}")
        print(f"loaded {t}: {inf['n_seeds']} seeds, flags {inf['flags']}", flush=True)
    one = np.ones(len(ids))
    val = {t: {**LN.lines(*frames[t], one), **GL.disp_values(t)} for t in frames}
    return _finish(a, LN, val, frames, info, lambda t, w: LN.lines(*frames[t], w), "loop")


def full(a) -> int:
    """5,710 x 200 read: every trajectory line on all 200 seeds of every run (200 vs 200), chunked by game."""
    import diag_c4_lib_v1 as L
    import grade_clock_env_loop_v1 as GL
    import grade_clock_r8_loop_v1 as R8
    import pyarrow.dataset as ds
    fd = ROOT / a.full_dir
    GL.DISP_DIR = ROOT / a.disp_dir
    R8.EV = fd
    tags = [a.control, *a.draws, a.arm]
    ids = np.sort(pd.read_parquet(fd / a.control / "games.parquet", columns=["game_id"])["game_id"].unique().astype("int64"))
    real = EG._real_side(ids)
    pgi = EG._pregame(ids)
    LN = GL.Lines(ids, real, pgi)
    frames, info = {}, {}
    for t in tags:
        gm = pd.read_parquet(fd / t / "games.parquet")
        if gm["seed"].nunique() != a.seeds or gm["game_id"].nunique() != len(ids):
            raise SystemExit(f"{t}: {gm['seed'].nunique()} seeds / {gm['game_id'].nunique()} games")
        dset = ds.dataset(ROOT / "results/trajectories" / t / "trajectory.parquet", format="parquet")
        pgs, sxs = [], []
        for part in np.array_split(ids, 6):
            tr = dset.to_table(filter=ds.field("game_id").isin(part.tolist())).to_pandas()
            d = L.std_from_sim(tr)
            pgc = EG.per_game_frame(d, a.seeds, part)
            pgs.append(EG.add_seed_level(pgc, d, part))
            sxs.append(GL.sim_extras(tr, part))
            del tr, d
        pg = pd.concat(pgs).reindex(ids)
        sx = pd.concat(sxs).reindex(ids)
        fm = (gm["home_pts"] - gm["away_pts"]).abs()
        gcol = pd.DataFrame({"p1_cnt": ((gm["n_periods"] <= 2) & (fm == 1)).astype(float).groupby(gm["game_id"]).sum(),
                             "gm_poss_sum": gm.groupby("game_id")["possessions"].sum(),
                             "gm_poss_sq": (gm["possessions"] ** 2).groupby(gm["game_id"]).sum()}).reindex(ids)
        pg = pg.join(gcol)
        if (gm.groupby("game_id").size().reindex(ids) != pg["n_seeds"]).any():
            raise SystemExit(f"{t}: games.parquet and trajectory disagree on game-seeds")
        g8 = R8.per_game(R8.load_run(t))
        meta = json.loads((fd / t / "run_meta.json").read_text(encoding="utf-8"))
        frames[t] = (pg, gm, sx, g8)
        info[t] = {"flags": {k: v for k, v in meta.get("adapter_flags", {}).items() if k.startswith("ENGINE_")},
                   "seeds": meta.get("seeds") if t in (a.control, a.arm) else None,
                   "engine_commit": meta.get("engine_commit"), "runtime_s": meta.get("runtime_s")}
        print(f"loaded {t}: commit {info[t]['engine_commit']}", flush=True)
    one = np.ones(len(ids))
    val = {t: {**LN.lines(*frames[t], one), **GL.disp_values(t)} for t in tags}
    return _finish(a, LN, val, frames, info, lambda t, w: LN.lines(*frames[t], w), "full")


# =====================================================================================================================
# SECTION 49 extension (clock experiments.md s49, PRE-REGISTERED by PM Fable 5.1 2026-10-14 ~12:00 ET).
# Added 2026-10-14 by the Opus engine worker and COMMITTED BEFORE ANY s49 FIT OUTPUT EXISTED (no file under
# results/endgame_clock_v2/offline_s49 at commit time). Arms CA0 (served K2, control), CA0s1 (K2 seed-1 refit, noise
# floor, never selectable), CA1 (ratio cell law, m re-chosen on fold 1), CA1E (within-cell env scale + CA1 ratio on
# the de-scaled baseline). Inputs `results/endgame_clock_v2/offline_s49/<fold>/A/<arm>.npz` (same keys as s46 A).
#
# Primaries (unchanged s46 block A lines, F2 period 2): P1 = `dev_pos`, P2 = `cal` (powered cells; `cal45` reported).
# ACTUAL-REFERENCED responsiveness lines (each has an actual value: the same statistic on the realised seconds y,
# the model's on E[min(T, R)]; period-2 window rows; per-game sufficient statistics so the game-block bootstrap applies):
#   tempo / zoff / defz   OLS slope of quintile mean seconds on quintile mean feature (s per unit), quintiles of the
#                         offence tempo / z_off = off_c + def_c / defence def_c over all period-2 rows.
#   team_trail / team_lead  the same over offence prior-strength quintiles (`off_rating_off_c`, quintiles over all
#                         period-2 rows), trailer rows / leader rows (s per rating point; the diagnostic's line (a)).
#   band_lead             leader bands 1-3/4-7/8-12/13+: OLS slope of band mean seconds on band mean margin (s/pt).
#   band_trail            the same over the four trailer bands on band mean |margin|. s49's "trailer strength slope"
#                         in the first list is read as THIS line (the s46 `slope_trail` band gate it replaces; the
#                         team-prior trailer line is listed separately and the within-cell trailer-strength line is
#                         in the within-cell group), so the seven s46 gates map one-to-one onto s49 lines.
#   wc_zoff / wc_defz / wc_trail  within-cell Q5 - Q1 (the diagnostic's part (c)): total Q5 - Q1 of mean seconds
#                         minus between = sum_c (share(Q5,c) - share(Q1,c)) x actual cell mean (pooled over the five
#                         quintiles; the same weights for every source). Cells = 45 (margin bucket x time bucket).
#                         z_off and defence-z over all period-2 rows; trailer strength over trailer rows.
# Gate (every line; s49): |L_arm - L_act| - |L_ctrl - L_act| <= floor, floor = sqrt(paired^2 + SE(L_act)^2),
#   paired = max(|L_CA0s1 - L_CA0| (reseed), 2 x SD of the paired bootstrap of L_arm - L_ctrl), SE(L_act) = SD of the
#   bootstrap of L_act (200 game-block draws, numpy seed 12345, the same weights for every source).
# G1 possession SD (registered line, loop translation): per test game, implied regulation possessions
#   N_g - n_w + n_w x A_w / E_w (N_g regulation possessions of the game from possessions_v4, n_w its window rows of
#   both halves, A_w their realised first-chance seconds, E_w the model's expected seconds), so only the window law
#   moves it; translation T = sqrt(5.629^2 + var_arm - var_ctrl) (5.629 = the full-set control loop SD, 5.474 the
#   full-set actual). Gate: |T_arm - 5.474| - |5.629 - 5.474| <= max(|T_CA0s1 - 5.629|, 2 x SD of the bootstrap T_arm).
# REPORTED: per-cell trailer and leader strength slopes (row-level OLS of seconds on `off_rating_off_c` within the
#   cell) and within-cell Q5 - Q1 (actual / control / each arm), the two early trailer-13+ cells and the nine (30,60]
#   cells flagged `explicit`; cells < 3,000 rows UNDERPOWERED for slopes, < 200 rows UNDERPOWERED as cells; the s46
#   lines (dev, dev_end, cal45, pergame_abs_err), start-type table, team right-sign share, fold 1 every line.
# Decision (s49, literal): an arm WINS if on F2 it beats CA0 by more than one floor on P1 AND P2, F1 keeps the sign of
#   both deltas, and no gate fails (the ten lines + G1 SD). If CA1 and CA1E both win, CA1E is preferred ONLY if it
#   beats CA1 on P1 by more than the floor (max(P1 reseed diff, 2 x SD of the paired bootstrap CA1E - CA1)) AND is
#   closer to actual than CA1 on at least two of wc_zoff / wc_defz / wc_trail beyond floor (the gate form with CA1 as
#   the reference); otherwise CA1 (ties to the simpler model). No winner: "offline refuted".
# m selection (`s49-select-m`, fold 1 only, the ruling-7 rule unchanged): per arm in {CA1, CA1E}, lowest F1 period-2
#   dev_pos over m in {20, 50, 100}; among m within one F1 floor of the best, the largest m. Frozen to
#   `results/endgame_clock_v2/grade/s49_m_selection.json` before any fold-2 fit.
# Screen (`loop --spec s49`, 500 x 25, four control draws): "proceed" iff at least one of {MD5 13+ last-2-min
#   points, tie rate, leader seconds at 2:00 13+ (`lsec_r2_13p`)} is closer to actual beyond floor, no s45 veto fails
#   (the s46 list incl. R2/R4 no-regression) and MD1-MD6 do not regress (|arm - target| - |ctrl - target| <= floor;
#   targets MD1 gap 0, MD2 q1-q5 1, MD3 actual tie rate, MD4/MD5 actual, MD6 ratio 1); else "stop". Seed check: if no
#   proceed line moves beyond floor but one moves toward actual inside its floor, the verdict reads
#   "stop (rerun at 50 seeds)" and the screen is re-run at 50 seeds.
# Full (`full --spec s49`, 5,710 x 200): CANDIDATE iff MD5 and `lsec_r2_13p` are each closer to actual beyond floor
#   and not past it (an arm on the far side of actual by more than the floor fails), MD4 (H2 36-40 possessions) and tie
#   rate do not regress beyond floor, every s45 veto passes (hard) and MD1-MD6 do not regress; else NOT A CANDIDATE.
# =====================================================================================================================
S49_DIR = OUT / "offline_s49"
S49_ARMS = ("CA0", "CA0s1", "CA1", "CA1E")
S49_SLOPES = ("tempo", "zoff", "defz", "team_trail", "team_lead", "band_lead", "band_trail")
S49_WC = ("wc_zoff", "wc_defz", "wc_trail")
S49_LINES = S49_SLOPES + S49_WC
G1_LOOP_ACT = 5.474
P4_DIR = ROOT / "data/processed/possessions_v4"


def load_pred_s49(fold: str, arm: str, t: pd.DataFrame) -> dict:
    z = np.load(S49_DIR / fold / "A" / f"{arm}.npz", allow_pickle=False)
    k = pd.DataFrame({"game_id": z["game_id"].astype(np.int64), "period": z["period"].astype(np.int64),
                      "poss_index": z["poss_index"].astype(np.int64), "_r": np.arange(len(z["game_id"]))})
    j = t[["game_id", "period", "poss_index"]].merge(k, on=["game_id", "period", "poss_index"], how="left",
                                                     validate="1:1")
    if j["_r"].isna().any() or len(k) != len(t):
        raise SystemExit(f"s49 {fold}/{arm}: prediction rows do not cover the test rows exactly")
    r = j["_r"].to_numpy(np.int64)
    out = {kk: np.asarray(z[kk])[r] for kk in z.files if kk not in ("game_id", "period", "poss_index", "meta")}
    out["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return out


class LinesS49:
    """Actual-referenced s49 lines on one period's window rows. `lines(s, w, sa)`: s = `sums(v)` for any source's row
    values v (the realised seconds y or a model's E[min(T, R)]), sa = `sums(y)` (the actual cell means)."""

    def __init__(self, t: pd.DataFrame, period: int):
        sel = t["period"].to_numpy() == period
        self.sel = sel
        tt = t[sel].reset_index(drop=True)
        self.t = tt
        _, self.gi = np.unique(tt["game_id"].to_numpy(), return_inverse=True)
        self.G = int(self.gi.max()) + 1
        one = np.ones(len(tt))
        gs = EG.gsum
        role = W.role(tt["m"].to_numpy())
        rb, tb = tt["rb"].to_numpy(), tt["tb"].to_numpy()
        self.cell = rb * W.N_TB + tb
        am = np.abs(tt["m"].to_numpy()).astype(float)
        st = tt["strength"].to_numpy(float)
        sq = quintile(st)
        qz, qd = quintile(tt["zoff"].to_numpy()), quintile(tt["defz"].to_numpy())
        self.groups = {
            "tempo": (quintile(tt["tempo"].to_numpy()), 5, tt["tempo"].to_numpy(float), 5),
            "zoff": (qz, 5, tt["zoff"].to_numpy(float), 5),
            "defz": (qd, 5, tt["defz"].to_numpy(float), 5),
            "team_trail": (np.where(role < 0, sq, 5), 6, st, 5),
            "team_lead": (np.where(role > 0, sq, 5), 6, st, 5),
            "band_lead": (np.where(rb < 4, rb, 4), 5, am, 4),
            "band_trail": (np.where((rb >= 4) & (rb < 8), rb - 4, 4), 5, am, 4)}
        self.gn, self.gx = {}, {}
        for nm, (key, K, x, _) in self.groups.items():
            self.gn[nm] = gs(self.gi, key, one, self.G, K)
            self.gx[nm] = gs(self.gi, key, x, self.G, K)
        self.wc = {"wc_zoff": qz * 45 + self.cell, "wc_defz": qd * 45 + self.cell,
                   "wc_trail": np.where(role < 0, sq * 45 + self.cell, 225)}
        self.wn = {k: gs(self.gi, v, one, self.G, 226) for k, v in self.wc.items()}
        self.st = st
        self.cn = gs(self.gi, self.cell, one, self.G, 45)
        self.cx = gs(self.gi, self.cell, st, self.G, 45)
        self.cxx = gs(self.gi, self.cell, st * st, self.G, 45)
        self.cq_key = sq * 45 + self.cell
        self.cqn = gs(self.gi, self.cq_key, one, self.G, 225)

    def sums(self, v_full: np.ndarray) -> dict:
        v = np.asarray(v_full, float)[self.sel]
        gs = EG.gsum
        s = {nm: gs(self.gi, key, v, self.G, K) for nm, (key, K, _, _) in self.groups.items()}
        for k, key in self.wc.items():
            s[k] = gs(self.gi, key, v, self.G, 226)
        s["cv"] = gs(self.gi, self.cell, v, self.G, 45)
        s["cxv"] = gs(self.gi, self.cell, self.st * v, self.G, 45)
        s["cqv"] = gs(self.gi, self.cq_key, v, self.G, 225)
        return s

    def lines(self, s: dict, w: np.ndarray, sa: dict) -> dict:
        out = {}
        for nm, (_, _, _, k) in self.groups.items():
            n = w @ self.gn[nm]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[nm] = EG.ols_slope(((w @ self.gx[nm]) / n)[:k], ((w @ s[nm]) / n)[:k])
        for k in self.wc:
            n = (w @ self.wn[k])[:225].reshape(5, 45)
            ca = (w @ sa[k])[:225].reshape(5, 45).sum(0) / np.maximum(n.sum(0), 1e-12)
            sh = n / np.maximum(n.sum(1, keepdims=True), 1e-12)
            m = (w @ s[k])[:225].reshape(5, 45)
            qm = m.sum(1) / np.maximum(n.sum(1), 1e-12)
            out[k] = float((qm[4] - qm[0]) - ((sh[4] - sh[0]) * ca).sum())
        return {k: float(v) for k, v in out.items()}

    def cell_lines(self, s: dict) -> dict:
        """Per-cell strength slope (row OLS of seconds on strength) and within-cell strength Q5 - Q1, (45,) each."""
        n, x, xx = self.cn.sum(0), self.cx.sum(0), self.cxx.sum(0)
        v, xv = s["cv"].sum(0), s["cxv"].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (xv - x * v / n) / (xx - x * x / n)
            qn = self.cqn.sum(0).reshape(5, 45)
            qv = s["cqv"].sum(0).reshape(5, 45) / qn
        return {"slope": slope, "q5_q1": qv[4] - qv[0], "n": n, "n_q1": qn[0], "n_q5": qn[4]}


def g1_frame(fold: str, t: pd.DataFrame) -> tuple:
    """(game index per window row over both halves, N_g regulation possessions, n_w, A_w) for the G1 SD line."""
    season = int(W.split(W.clock_window(), fold)[1]["season"].iloc[0])
    p4 = pd.read_parquet(P4_DIR / f"possessions_{season}.parquet", columns=["game_id", "period"])
    N = p4[p4["period"] <= 2].groupby("game_id").size()
    g, gi = np.unique(t["game_id"].to_numpy(), return_inverse=True)
    Ng = N.reindex(g).to_numpy(float)
    if np.isnan(Ng).any():
        raise SystemExit(f"G1 frame {fold}: {int(np.isnan(Ng).sum())} games without a possessions_v4 count")
    nw = np.bincount(gi, minlength=len(g)).astype(float)
    Aw = np.bincount(gi, weights=t["y"].to_numpy(float), minlength=len(g))
    return gi, Ng, nw, Aw


def g1_var(gi, Ng, nw, Aw, e: np.ndarray, w: np.ndarray) -> float:
    Ew = np.bincount(gi, weights=e, minlength=len(Ng))
    x = Ng - nw + nw * Aw / Ew
    mu = np.average(x, weights=w)
    return float(np.average((x - mu) ** 2, weights=w))


def s49_gate(La, Lc, Lact, Ls1, Ba, Bc, Bact) -> dict:
    """The s49 actual-referenced gate for one line (module comment)."""
    paired = max(abs(Ls1 - Lc), 2 * float(np.nanstd(np.asarray(Ba) - np.asarray(Bc), ddof=1)))
    se_act = float(np.nanstd(np.asarray(Bact), ddof=1))
    floor = float(np.sqrt(paired ** 2 + se_act ** 2))
    excess = abs(La - Lact) - abs(Lc - Lact)
    return {"actual": Lact, "ctrl": Lc, "arm": La, "arm_minus_act": La - Lact, "ctrl_minus_act": Lc - Lact,
            "excess": excess, "paired_floor": paired, "se_actual": se_act, "floor": floor,
            "pass": bool(excess <= floor)}


def s49_fold(fold: str, arms: list[str], periods=(2, 1)) -> dict:
    t = truth_A(fold)
    rows = {a: rows_A(t, load_pred_s49(fold, a, t)) for a in arms}
    pos = rows["CA0"]["raw"] > EPS
    gi1, Ng, nw, Aw = g1_frame(fold, t)
    wg = EG.boot_weights(len(Ng))
    v1 = {a: g1_var(gi1, Ng, nw, Aw, rows[a]["e"], np.ones(len(Ng))) for a in arms}
    vb = {a: np.array([g1_var(gi1, Ng, nw, Aw, rows[a]["e"], wg[b]) for b in range(N_BOOT)]) for a in arms}
    g1 = {"act_offline_sd": float(np.std(Ng)), "ctrl_offline_sd": float(np.sqrt(v1["CA0"])), "n_games": int(len(Ng))}
    for a in arms:
        if a == "CA0":
            continue
        T = float(np.sqrt(G1_LOOP_CTRL ** 2 + v1[a] - v1["CA0"]))
        Tb = np.sqrt(np.maximum(G1_LOOP_CTRL ** 2 + vb[a] - vb["CA0"], 0.0))
        Ts1 = float(np.sqrt(G1_LOOP_CTRL ** 2 + v1["CA0s1"] - v1["CA0"])) if "CA0s1" in v1 else G1_LOOP_CTRL
        fl = max(abs(Ts1 - G1_LOOP_CTRL), 2 * float(np.std(Tb, ddof=1)))
        ex = abs(T - G1_LOOP_ACT) - abs(G1_LOOP_CTRL - G1_LOOP_ACT)
        g1[a] = {"offline_sd": float(np.sqrt(v1[a])), "loop_translation": T, "ctrl_loop": G1_LOOP_CTRL,
                 "act_loop": G1_LOOP_ACT, "excess": ex, "floor": fl, "pass": bool(ex <= fl)}
    res = {"n_rows": int(len(t)), "g1_sd": g1, "periods": {}}
    for per in periods:
        LA = LinesA(t, per, pos)
        wb = EG.boot_weights(LA.G)
        one = np.ones(LA.G)
        sums = {a: LA.sums(r) for a, r in rows.items()}
        val = {a: LA.lines(s, one) for a, s in sums.items()}
        boot = {a: [LA.lines(s, wb[b]) for b in range(N_BOOT)] for a, s in sums.items()}
        comp = compare(val, boot, "CA0", "CA0s1", arms)
        L = LinesS49(t, per)
        if L.G != LA.G:
            raise SystemExit("game index mismatch")
        sa = L.sums(t["y"].to_numpy(float))
        ss = {a: L.sums(r["e"]) for a, r in rows.items()}
        lv = {"act": L.lines(sa, one, sa), **{a: L.lines(s, one, sa) for a, s in ss.items()}}
        lb = {"act": [L.lines(sa, wb[b], sa) for b in range(N_BOOT)],
              **{a: [L.lines(s, wb[b], sa) for b in range(N_BOOT)] for a, s in ss.items()}}
        gates = {}
        for a in arms:
            if a == "CA0":
                continue
            gates[a] = {k: s49_gate(lv[a][k], lv["CA0"][k], lv["act"][k], lv["CA0s1"][k], [x[k] for x in lb[a]],
                                    [x[k] for x in lb["CA0"]], [x[k] for x in lb["act"]]) for k in S49_LINES}
        cl = {"act": L.cell_lines(sa), **{a: L.cell_lines(s) for a, s in ss.items()}}
        cells = []
        for c in range(45):
            n = int(cl["act"]["n"][c])
            if n == 0:
                continue
            rb, tb = divmod(c, W.N_TB)
            r = {"cell": f"{W.ROLEBAND_LABELS[rb]} | {W.BUCKET_LABELS[tb]}", "n": n,
                 "n_q1": int(cl["act"]["n_q1"][c]), "n_q5": int(cl["act"]["n_q5"][c]),
                 "slope_power": "ok" if n >= MIN_SLOPE else "UNDERPOWERED",
                 "cell_power": "ok" if n >= MIN_CELL else "UNDERPOWERED",
                 "explicit": bool((rb == 7 and tb in (0, 1)) or tb == 3)}
            for s in cl:
                r[f"slope_{s}"] = float(cl[s]["slope"][c])
                r[f"q5_q1_{s}"] = float(cl[s]["q5_q1"][c])
            cells.append(r)
        res["periods"][str(per)] = {
            "n_rows": int(LA.sel.sum()), "n_games": int(LA.G), "excluded_zero_mass_rows": int((~LA.pos).sum()),
            "compare": comp, "s49_lines": lv, "s49_gates": gates, "cells_s49": cells,
            "cells": cell_table_A(t, rows, per), "start_types": start_table_A(t, rows, per),
            "team_sign": team_sign_A(t, rows, "CA0", per)}
        if per == 2 and "CA1" in arms and "CA1E" in arms:
            d = np.array([x["dev_pos"] for x in boot["CA1E"]]) - np.array([x["dev_pos"] for x in boot["CA1"]])
            rs = abs(val["CA0s1"]["dev_pos"] - val["CA0"]["dev_pos"]) if "CA0s1" in val else 0.0
            wc = {k: s49_gate(lv["CA1E"][k], lv["CA1"][k], lv["act"][k], lv["CA0s1"][k], [x[k] for x in lb["CA1E"]],
                              [x[k] for x in lb["CA1"]], [x[k] for x in lb["act"]]) for k in S49_WC}
            res["periods"][str(per)]["ca1e_vs_ca1"] = {
                "dev_pos_delta": val["CA1E"]["dev_pos"] - val["CA1"]["dev_pos"],
                "dev_pos_floor": float(max(rs, 2 * float(np.std(d, ddof=1)))),
                "wc_closer_beyond_floor": {k: bool(-x["excess"] > x["floor"]) for k, x in wc.items()}, "wc": wc}
    return res


def s49_decide(F: dict, arms: list[str]) -> dict:
    p2 = F["F2"]["periods"]["2"]
    p1 = F["F1"]["periods"]["2"] if "F1" in F else None
    out = {"per_arm": {}}
    winners = []
    for a in arms:
        if a in ("CA0", "CA0s1"):
            continue
        d = p2["compare"][a]
        beats = {p: bool(d[p]["delta"] < -d[p]["floor"]) for p in ("dev_pos", "cal")}
        f1 = {p: (bool(np.sign(p1["compare"][a][p]["delta"]) == np.sign(d[p]["delta"])) if p1 else None)
              for p in ("dev_pos", "cal")}
        gates = {k: p2["s49_gates"][a][k]["pass"] for k in S49_LINES}
        gates["g1_sd"] = F["F2"]["g1_sd"][a]["pass"]
        win = all(beats.values()) and all(v is not False for v in f1.values()) and all(gates.values())
        out["per_arm"][a] = {"beats_ctrl_by_floor": beats, "f1_keeps_sign": f1, "gates_pass": gates,
                             "gates_failing": [k for k, v in gates.items() if not v],
                             "primary_floors": {p: -d[p]["delta"] / d[p]["floor"] if d[p]["floor"] else None
                                                for p in ("dev_pos", "cal")}, "wins": bool(win)}
        if win:
            winners.append(a)
    sel = None
    if len(winners) == 1:
        sel = winners[0]
    elif set(winners) == {"CA1", "CA1E"}:
        c = p2["ca1e_vs_ca1"]
        pref = (c["dev_pos_delta"] < -c["dev_pos_floor"]) and sum(c["wc_closer_beyond_floor"].values()) >= 2
        sel = "CA1E" if pref else "CA1"
        out["ca1e_preference"] = {**c, "ca1e_preferred": bool(pref)}
    out.update({"winners": winners, "selected": sel, "verdict": f"offline winner {sel}" if sel else "offline refuted",
                "rule": "s49: beats CA0 by > 1 floor on dev_pos AND cal (F2 period 2), F1 keeps both signs, no "
                        "actual-referenced gate fails (ten lines + G1 SD loop translation); both win -> CA1E only if "
                        "it beats CA1 on P1 beyond floor AND is closer to actual on >= 2 within-cell lines beyond "
                        "floor, else CA1"})
    return out


def s49_offline(arms: list[str] | None) -> int:
    arms = arms or [a for a in S49_ARMS if (S49_DIR / "F2" / "A" / f"{a}.npz").exists()]
    if "CA0" not in arms or "CA0s1" not in arms:
        raise SystemExit("s49: CA0 and CA0s1 prediction files are required")
    codes = code_arms("A", arms)
    F = {}
    for fold in ("F2", "F1"):
        have = [a for a in arms if (S49_DIR / fold / "A" / f"{a}.npz").exists()]
        F[fold] = s49_fold(fold, have)
        print(f"s49 graded {fold}", flush=True)
    GRADE_DIR.mkdir(parents=True, exist_ok=True)
    (GRADE_DIR / "s49_offline_coded.json").write_text(json.dumps(recode(F, codes), indent=1, default=float),
                                                      encoding="utf-8")
    key = {c: a for a, c in codes.items()}
    (GRADE_DIR / "s49_offline_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    dec = s49_decide(F, arms)
    rep = {"spec": "docs/models/clock/experiments.md s49", "arms": arms, "key": key, "folds": F, "decision": dec,
           "written_at": datetime.now(UTC).isoformat()}
    (GRADE_DIR / "s49_offline.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(dec, indent=1, default=float))
    return 0


def s49_select_m() -> int:
    out = {"rule": "s49 (ruling-7 rule unchanged): lowest F1 period-2 dev_pos; among m within one F1 floor of the "
                   "best, the largest m", "arms": {}}
    t = truth_A("F1")
    base = {n: rows_A(t, load_pred_s49("F1", n, t)) for n in ("CA0", "CA0s1")}
    for arm in ("CA1", "CA1E"):
        names = [f"{arm}_m{m}" for m in M_GRID]
        have = [n for n in names if (S49_DIR / "F1" / "A" / f"{n}.npz").exists()]
        if len(have) != len(M_GRID):
            raise SystemExit(f"{arm}: fold-1 grid incomplete ({have})")
        rows = {**base, **{n: rows_A(t, load_pred_s49("F1", n, t)) for n in have}}
        L = LinesA(t, 2, rows["CA0"]["raw"] > EPS)
        wb = EG.boot_weights(L.G)
        sums = {n: L.sums(r) for n, r in rows.items()}
        val = {n: L.lines(s, np.ones(L.G)) for n, s in sums.items()}
        boot = {n: [L.lines(s, wb[b]) for b in range(N_BOOT)] for n, s in sums.items()}
        c = compare(val, boot, "CA0", "CA0s1", list(rows))
        best = min(have, key=lambda n: c[n]["dev_pos"]["arm"])
        ok = [n for n in have if c[n]["dev_pos"]["arm"] - c[best]["dev_pos"]["arm"] <= c[best]["dev_pos"]["floor"]]
        chosen = max(ok, key=lambda n: int(n.split("_m")[1]))
        out["arms"][arm] = {"m": int(chosen.split("_m")[1]), "best_by_dev_pos": best, "within_floor": ok,
                            "f1_dev_pos": {n: c[n]["dev_pos"] for n in have}}
        print(arm, "->", out["arms"][arm]["m"], {n: round(c[n]["dev_pos"]["arm"], 5) for n in have}, flush=True)
    out["written_at"] = datetime.now(UTC).isoformat()
    p = GRADE_DIR / "s49_m_selection.json"
    if p.exists() and os.environ.get("EGC_OVERWRITE") != "1":
        raise SystemExit(f"{p} exists (frozen)")
    GRADE_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


S49_PROCEED = (("MD5_pts13_last2", "act_MD5_pts13_last2"), ("tie_rate", "act_tie_rate"),
               ("lsec_r2_13p", "act_lsec_r2_13p"))


def s49_md_noreg(rows: dict, act: dict) -> dict:
    import grade_clock_env_loop_v1 as GL
    tg = {"MD1_gap": 0.0, **{f"MD2_q{i}": 1.0 for i in range(1, 6)}, "tie_rate": act["act_tie_rate"],
          "MD4_h2_last4_poss": act["act_MD4_h2_last4_poss"], "MD5_pts13_last2": act["act_MD5_pts13_last2"],
          "MD6_margin_sd_ratio": 1.0}
    out = {}
    for k, tv in tg.items():
        x = GL.toward(rows[k], tv)
        out[k] = {**x, "pass": not x["away_by_floor"]}
    return out


def s49_judge(rows: dict, act: dict, ctrl_lines: dict, mode: str) -> dict:
    import grade_clock_env_loop_v1 as GL
    base = judge_loop("A", rows, act, ctrl_lines)
    vet = base["vetoes"]
    md = s49_md_noreg(rows, act)
    v = {"spec": "s49", "vetoes": vet, "vetoes_failing": [k for k, x in vet.items() if not x["pass"]],
         "md_noreg": md, "md_failing": [k for k, x in md.items() if not x["pass"]], "reported": base["reported"],
         "s46_md_block_reported": base["md"]}
    pl = {k: GL.toward(rows[k], act[t]) for k, t in S49_PROCEED}
    v["proceed_lines"] = pl
    clean = not v["vetoes_failing"] and not v["md_failing"]
    if mode == "loop":
        moves = any(x["toward_by_floor"] for x in pl.values())
        inside = any((0 < x["improve"] <= x["floor"]) for x in pl.values())
        v["seed_check_s49"] = {"rerun_at_50": bool((not moves) and inside)}
        v["screen"] = ("proceed" if (moves and clean) else
                       ("stop (rerun at 50 seeds)" if v["seed_check_s49"]["rerun_at_50"] and clean else "stop"))
        v["passes"] = v["screen"] == "proceed"
        return v
    prim = {}
    for k, t in (("MD5_pts13_last2", "act_MD5_pts13_last2"), ("lsec_r2_13p", "act_lsec_r2_13p")):
        x = GL.toward(rows[k], act[t])
        past = bool(np.sign(x["arm"] - x["target"]) != np.sign(x["ctrl"] - x["target"])
                    and abs(x["arm"] - x["target"]) > x["floor"])
        prim[k] = {**x, "past_actual_beyond_floor": past, "pass": bool(x["toward_by_floor"] and not past)}
    nr = {}
    for k, t in (("MD4_h2_last4_poss", "act_MD4_h2_last4_poss"), ("tie_rate", "act_tie_rate")):
        x = GL.toward(rows[k], act[t])
        nr[k] = {**x, "pass": not x["away_by_floor"]}
    v.update({"full_primaries": prim, "must_not_regress": nr})
    ok = all(x["pass"] for x in prim.values()) and all(x["pass"] for x in nr.values()) and clean
    v["full_verdict"] = "CANDIDATE" if ok else "NOT A CANDIDATE"
    v["passes"] = bool(ok)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    o = sub.add_parser("offline")
    o.add_argument("--block", choices=["A", "B", "C"], required=True)
    o.add_argument("--arms", nargs="*", default=None)
    sub.add_parser("select-m")
    s9 = sub.add_parser("s49-offline")
    s9.add_argument("--arms", nargs="*", default=None)
    sub.add_parser("s49-select-m")
    for nm in ("loop", "full"):
        p = sub.add_parser(nm)
        p.add_argument("--block", choices=["A", "B"], required=True)
        p.add_argument("--control", required=True)
        p.add_argument("--draws", nargs=4, required=True)
        p.add_argument("--arm", required=True)
        p.add_argument("--seeds", type=int, default=25 if nm == "loop" else 200)
        p.add_argument("--disp-dir", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--spec", choices=["s46", "s49"], default="s46")
        if nm == "loop":
            p.add_argument("--loop-dir", default="results/endgame_clock_v2/loop")
        else:
            p.add_argument("--full-dir", required=True)
    a = ap.parse_args()
    if a.mode == "offline":
        return offline_C(a.arms) if a.block == "C" else offline(a.block, a.arms)
    if a.mode == "select-m":
        return select_m()
    if a.mode == "s49-offline":
        return s49_offline(a.arms)
    if a.mode == "s49-select-m":
        return s49_select_m()
    if os.environ["CBB_TRUTH"] != "verified_v1":
        raise SystemExit("set CBB_TRUTH=verified_v1")
    return loop(a) if a.mode == "loop" else full(a)


if __name__ == "__main__":
    raise SystemExit(main())
