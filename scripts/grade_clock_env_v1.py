"""grade_clock_env_v1.py -- clock round 9 OFFLINE grade (experiments.md section 41). ONE blind grader over
every arm of the round (E0 = served K2, E0s1 = K2 refitted under seed 1, E1, E2, E3; optional reference K2M,
never selectable), both folds. Written and committed BEFORE any round-9 arm was fitted (section 41.0 item 5).

Truth: test-season possessions of the L2 v4 design (clock-complete games, regulation, horn censoring), joined to
chances_v4 (first-chance d1, chance-1 end class, chance count) and possessions_v4 (whole-possession duration).
Per possession, each arm's IMPLIED time given the realised chances (section 38's definition, the law the clock
consumes):  t = min(E[min(T1 | end1, R)] + sum of continuation means, R).

Lines (per arm, per fold; per-game sufficient statistics so every line is bootstrapped by game weights):
  deviance        PRIMARY. -2 mean log-lik of the realised POSSESSION duration under the arm's possession law
                  given the realised chance count and end class: conditional d1 pmf convolved with the
                  continuation pmfs (n-1 continuations; mass past 90 s at 90); horn-censored rows score P(T >= y).
  deviance_pos    the same on the rows where the CONTROL E0 assigns positive mass (prob > 1e-12, the clip),
                  with excluded_share = share of rows E0 assigns zero mass (section 41.0 item 2; a gate).
  d1_dev_cond / d1_dev_marg   first-chance deviance of the conditional / marginal law (reported).
  ratio_env / ratio_gap       game level: slope of implied possessions (1200 x n / sum t) on env (resp. gap),
                  divided by the actual slope (1200 x n / sum of actual durations) on the same games. env = both
                  offences' off_c + both defences' def_c; gap = |net(home) - net(away)|, net = off_c - def_c.
                  The joint (env, gap) regression and a team-game x_off line are reported.
  slope_oq        (a) model vs actual quintile means of implied duration, offence-team season-mean tempo quintile.
  slope_zq        (b) quintiles of the row's z_off (off_c(offence) + def_c(defence)).
  slope_gapq      (c) quintiles of the game's gap.
  slope_dq        (d) quintiles of the defence's def_c (the defence's part of z_off).
  slope_gq        game-prior (tempo_prior_game) quintile slope of implied counts; elast_ratio = round-8 game-level
                  tempo elasticity ratio (log implied count on log rel_home + log rel_away, month FE).
  count_gap       implied possessions per team-game, model - actual; gap_by_start / gap_by_month (seconds).
Segments (reported): month, site (offence home/away/neutral), start type, conference tier, half, offence-team
strength quintile, plus the share of teams whose arm-minus-E0 mean implied duration has the sign of
actual-minus-E0. Cells < 3,000 rows (slopes) or < 150 games (ratios, elasticities) are UNDERPOWERED.
Floors: 2 x SD of the paired game-block bootstrap (200 draws, seed 12345) of each (arm - E0) line.
Reseed proof: max abs pmf difference E0 vs E0s1 and each arm vs its --seed 1 refit if present.

BLIND: arms are scored and tabulated under coded labels (a fixed hash of the arm name); the key and the decision
are written after the coded tables. Decision: section 41.A section 5, applied literally (`decide`), plus a
REPORTED-ONLY relative reading of the two absolute ranges (see experiments.md 41.B).

    .venv/Scripts/python.exe scripts/grade_clock_env_v1.py [--arms E0,E0s1,E1,E2,E3] [--folds F2,F1]
Writes results/clock_env/offline_grade.json, results/clock_env/offline_tables.md.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.models import clock as CK  # noqa: E402
from cbb_sim.models import clock_r7  # noqa: E402,F401  (unpickling)
from cbb_sim.models import clock_r8  # noqa: E402
from cbb_sim.models import clock_env  # noqa: E402,F401  (unpickling)
import grade_clock_r8_offline_v1 as G8  # noqa: E402
import train_clock_r8_chance_v1 as TK  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
OUT = ROOT / "results/clock_env"
GRID = np.arange(CK.DURATION_CAP + 1, dtype=np.float64)
NG = CK.DURATION_CAP + 1
NBOOT = 200
BOOT_SEED = 12345
SALT = "clock-env-r9-2026-10-13"
CANDIDATES = ("E1", "E2", "E3")          # simplicity order
NOISE = ("E0s1",)
REFS = ("K2M",)
POWER = {"ACC", "Big Ten", "Big 12", "SEC", "Big East"}
CLIP = 1e-12


def manifest_of(arm: str, fold: str) -> Path:
    if arm == "E0":
        return CKD / "r8_K2" / fold / "manifest.json"
    if arm == "E0s1":
        return CKD / "r8_K2" / f"{fold}_seed1" / "manifest.json"
    if arm == "K2M":
        return CKD / "r8_K2M" / fold / "manifest.json"
    return CKD / f"r9_env_{arm}" / fold / "manifest.json"


def code_of(arm: str) -> str:
    return "X" + hashlib.sha256((SALT + arm).encode()).hexdigest()[:4].upper()


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- data
def load_rows() -> dict[str, pd.DataFrame]:
    univ = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    design = TK.T.load_design()
    design = design[design["season"].isin([2024, 2025])]
    first, _ = TK.chance_tables([2024, 2025])
    design = TK.build_d1_design(design, first)
    tiers = []
    for s in (2024, 2025):
        cg = pd.read_parquet(ROOT / f"data/raw/cbbd/games_{s}.parquet", columns=["id", "homeConference", "awayConference"])
        tiers.append(cg.rename(columns={"id": "cbbd_game_id"}))
    cg = pd.concat(tiers, ignore_index=True)
    cg["npow"] = cg["homeConference"].isin(POWER).astype(int) + cg["awayConference"].isin(POWER).astype(int)
    u2 = univ[["game_id", "cbbd_game_id"]].merge(cg[["cbbd_game_id", "npow"]], on="cbbd_game_id", how="left")
    tier_of = u2.drop_duplicates("game_id").set_index("game_id")["npow"]
    out = {}
    for fold, season in (("F2", 2025), ("F1", 2024)):
        d = design[(design["season"] == season) & (design["period"] <= 2.0) & design["game_id"].isin(cc)].copy()
        p4 = pd.read_parquet(TK.P4 / f"possessions_{season}.parquet", columns=["game_id", "period", "poss_index", "duration_s"])
        p4 = p4.rename(columns={"period": "_p", "poss_index": "_i", "duration_s": "dur_poss"})
        p4["_p"] = p4["_p"].astype("int64"); p4["_i"] = p4["_i"].astype("int64")
        d["_p"] = d["period"].astype("int64"); d["_i"] = d["poss_index"].astype("int64")
        d = d.merge(p4, on=["game_id", "_p", "_i"], how="left", validate="1:1").drop(columns=["_p", "_i"])
        d = d[d["dur_poss"].notna()].reset_index(drop=True)
        oo, od, do, dd = (d[c].to_numpy(np.float64) for c in clock_env.RATING_COLS)
        d["z_raw"] = oo + dd
        d["dd_raw"] = dd
        # game-level env / gap from the home view (identical from either offence)
        home_off = d["offense_is_home"].to_numpy().astype(bool)
        d["env"] = oo + od + do + dd
        net_o, net_d = oo - od, do - dd
        d["gap"] = np.abs(net_o - net_d)
        d["net_off"] = net_o
        d["month"] = pd.to_datetime(d["game_date"]).dt.month
        d["site"] = np.where(d["site_home"] + d["site_away"] == 0, "neutral",
                             np.where(d["site_home"] > 0, "off_home", "off_away"))
        d["tier"] = d["game_id"].map(tier_of).map({2: "both_power", 1: "one_power", 0: "neither_power"}).fillna("unknown")
        d["half"] = np.where(d["period"].to_numpy() <= 1, "H1", "H2")
        d["oq"] = d["offense_team_id"].map(pd.qcut(d.groupby("offense_team_id")["off_tempo_rel"].mean(), 5, labels=False))
        d["zq"] = pd.qcut(d["z_raw"].rank(method="first"), 5, labels=False).astype(int)
        d["dq"] = pd.qcut(d["dd_raw"].rank(method="first"), 5, labels=False).astype(int)
        gg = d.groupby("game_id").agg(gap=("gap", "first"), tp=("tempo_prior_game", "first"))
        d["gapq"] = d["game_id"].map(pd.qcut(gg["gap"].rank(method="first"), 5, labels=False))
        d["gq"] = d["game_id"].map(pd.qcut(gg["tp"].rank(method="first"), 5, labels=False))
        tq = d.groupby("offense_team_id")["net_off"].mean()
        d["teamq"] = d["offense_team_id"].map(pd.qcut(tq.rank(method="first"), 5, labels=False))
        d["R"] = d["seconds_remaining"].to_numpy(np.float64)
        d["y_poss"] = np.clip(d["dur_poss"].to_numpy(np.int64), 0, CK.DURATION_CAP)
        _ = home_off
        out[fold] = d
        log(f"{fold}: {len(d):,} rows, {d['game_id'].nunique():,} games")
    return out


# --------------------------------------------------------------------------- scoring
def cont_matrices(cont_pmf: np.ndarray, n_max: int) -> dict[int, np.ndarray]:
    """M_n (NG x NG): row j = d1 value, column y = possession duration (mass past the cap at the cap)."""
    c2, c3 = cont_pmf[0], cont_pmf[1]
    mats = {1: np.eye(NG)}
    ker = c2.copy()
    for n in range(2, n_max + 1):
        if n >= 3:
            ker = np.convolve(ker, c3)
        M = np.zeros((NG, NG))
        L = len(ker)
        for j in range(NG):
            m = min(L, CK.DURATION_CAP - j)
            if m > 0:
                M[j, j:j + m] = ker[:m]
            M[j, CK.DURATION_CAP] += ker[max(m, 0):].sum()
        mats[n] = M
    return mats


def score_arm(sched, d: pd.DataFrame) -> dict[str, np.ndarray]:
    cuts = np.array([np.datetime64(r[0], "ns") for r in sched])
    seg = np.searchsorted(cuts, pd.to_datetime(d["game_date"]).to_numpy(), side="right") - 1
    if (seg < 0).any():
        raise AssertionError("test rows dated before the first refit")
    n = len(d)
    res = {k: np.empty(n) for k in ("e_c", "e_m", "ll1c", "ll1m", "llp", "prp", "cont")}
    y1 = np.clip(d["duration_s"].to_numpy().astype(np.int64), 0, CK.DURATION_CAP)
    c1 = d["censored"].to_numpy(dtype=bool)
    yp = d["y_poss"].to_numpy()
    cp = d["censored_poss"].to_numpy(dtype=bool)
    nch = np.maximum(d["n_ch"].to_numpy().astype(np.int64), 1)
    n_max = int(nch.max())
    for k in np.unique(seg):
        rows = np.flatnonzero(seg == k)
        arm = sched[int(k)][1]
        mc = arm.mean_cont()
        res["cont"][rows] = np.where(nch[rows] >= 2, mc[0], 0.0) + np.maximum(nch[rows] - 2, 0) * mc[1]
        mats = cont_matrices(arm.cont_pmf, n_max)
        for lo in range(0, len(rows), G8.CHUNK):
            r = rows[lo:lo + G8.CHUNK]
            blk = d.iloc[r].reset_index(drop=True)
            Rr = blk["R"].to_numpy()[:, None]
            ii = np.arange(len(r))
            for fn, ek, lk in ((arm.pmf, "e_m", "ll1m"), (arm.pmf_end, "e_c", "ll1c")):
                p = np.asarray(fn(blk), dtype=np.float64)
                res[ek][r] = (p * np.minimum(GRID[None, :], Rr)).sum(axis=1)
                pu = p[ii, y1[r]]
                S = 1.0 - np.cumsum(p, axis=1)[ii, y1[r]] + pu
                res[lk][r] = np.log(np.clip(np.where(c1[r], S, pu), CLIP, None))
                if ek == "e_c":
                    pc = p
            pp = np.empty_like(pc)
            for nn in np.unique(nch[r]):
                sel = nch[r] == nn
                pp[sel] = pc[sel] @ mats[int(nn)]
            pu = pp[ii, yp[r]]
            S = 1.0 - np.cumsum(pp, axis=1)[ii, yp[r]] + pu
            pr = np.where(cp[r], S, pu)
            res["prp"][r] = pr
            res["llp"][r] = np.log(np.clip(pr, CLIP, None))
    res["t"] = np.minimum(res["e_c"] + res["cont"], d["R"].to_numpy())
    return res


def pmf_diff(sched_a, sched_b, d: pd.DataFrame, n: int = 30000) -> float:
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(len(d), size=min(n, len(d)), replace=False))
    blk = d.iloc[idx].reset_index(drop=True)
    cuts = np.array([np.datetime64(r[0], "ns") for r in sched_a])
    seg = np.searchsorted(cuts, pd.to_datetime(blk["game_date"]).to_numpy(), side="right") - 1
    mx = 0.0
    for k in np.unique(seg):
        rr = np.flatnonzero(seg == k)
        b = blk.iloc[rr].reset_index(drop=True)
        for fn in ("pmf", "pmf_end"):
            pa = getattr(sched_a[int(k)][1], fn)(b)
            pb = getattr(sched_b[int(k)][1], fn)(b)
            mx = max(mx, float(np.abs(np.asarray(pa) - np.asarray(pb)).max()))
    return mx


# --------------------------------------------------------------------------- per-game aggregation and lines
def wls(y, X, w):
    A = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return b[1:]


class Agg:
    def __init__(self, d: pd.DataFrame, arms: list[str], pos_mask: np.ndarray):
        gi, games = pd.factorize(d["game_id"])
        self.G = len(games)
        self.gi = gi
        bc = lambda v: np.bincount(gi, weights=v, minlength=self.G)  # noqa: E731
        self.n = bc(np.ones(len(d)))
        self.A = bc(d["dur_poss"].to_numpy(np.float64))
        self.npos = bc(pos_mask.astype(float))
        first = d.groupby(gi).first()
        self.env = first["env"].to_numpy(np.float64)
        self.gap = first["gap"].to_numpy(np.float64)
        self.X = (np.log(first["off_tempo_rel"].clip(0.5, 2.0)) + np.log(first["def_tempo_rel"].clip(0.5, 2.0))).to_numpy()
        mo = pd.to_datetime(first["game_date"]).dt.month.astype(int)
        self.M = pd.get_dummies(mo, drop_first=True).to_numpy(dtype=float)
        self.gq = first["gq"].to_numpy().astype(int)
        self.T, self.LL, self.LLpos, self.L1c, self.L1m = {}, {}, {}, {}, {}
        for a in arms:
            self.T[a] = bc(d[f"t_{a}"].to_numpy())
            self.LL[a] = bc(d[f"llp_{a}"].to_numpy())
            self.LLpos[a] = bc(np.where(pos_mask, d[f"llp_{a}"].to_numpy(), 0.0))
            self.L1c[a] = bc(d[f"ll1c_{a}"].to_numpy())
            self.L1m[a] = bc(d[f"ll1m_{a}"].to_numpy())
        self.q = {}
        for q in ("oq", "zq", "gapq", "dq"):
            qq = d[q].to_numpy().astype(int)
            N = np.zeros((self.G, 5)); Aq = np.zeros((self.G, 5))
            np.add.at(N, (gi, qq), 1.0)
            np.add.at(Aq, (gi, qq), d["dur_poss"].to_numpy(np.float64))
            Tq = {}
            for a in arms:
                E = np.zeros((self.G, 5))
                np.add.at(E, (gi, qq), d[f"t_{a}"].to_numpy())
                Tq[a] = E
            self.q[q] = (N, Aq, Tq)

    def lines(self, a: str, w: np.ndarray) -> dict:
        n = w @ self.n
        out = {"deviance": float(-2 * (w @ self.LL[a]) / n),
               "deviance_pos": float(-2 * (w @ self.LLpos[a]) / (w @ self.npos)),
               "excluded_share": float(1 - (w @ self.npos) / n),
               "d1_dev_cond": float(-2 * (w @ self.L1c[a]) / n),
               "d1_dev_marg": float(-2 * (w @ self.L1m[a]) / n)}
        out["count_gap"] = float(1200 * n / (w @ self.T[a]) - 1200 * n / (w @ self.A))
        keep = w > 0
        ww = w[keep]
        pm = (1200 * self.n / self.T[a])[keep]
        pa = (1200 * self.n / self.A)[keep]
        for nm, x in (("env", self.env[keep]), ("gap", self.gap[keep])):
            sm = wls(pm, x[:, None], ww)[0]
            sa = wls(pa, x[:, None], ww)[0]
            out[f"slope_{nm}_model"], out[f"slope_{nm}_act"], out[f"ratio_{nm}"] = float(sm), float(sa), float(sm / sa)
        bj = wls(pm, np.column_stack([self.env[keep], self.gap[keep]]), ww)
        ba = wls(pa, np.column_stack([self.env[keep], self.gap[keep]]), ww)
        out["joint_env_ratio"], out["joint_gap_ratio"] = float(bj[0] / ba[0]), float(bj[1] / ba[1])
        out["game_cal_slope"] = float(wls(pa, pm[:, None], ww)[0])
        Z = np.column_stack([self.X[keep], self.M[keep]])
        el = wls(np.log(pm), Z, ww)[0]
        ea = wls(np.log(pa), Z, ww)[0]
        out["elast_ratio"] = float(el / ea)
        for q in ("oq", "zq", "gapq", "dq"):
            N, Aq, Tq = self.q[q]
            nn = w @ N
            out[f"slope_{q}"] = float(np.polyfit((w @ Aq) / nn, (w @ Tq[a]) / nn, 1)[0])
        gm = pd.DataFrame({"m": pm, "a": pa, "w": ww, "q": self.gq[keep]})
        t = gm.groupby("q").apply(lambda g: pd.Series({"m": np.average(g["m"], weights=g["w"]),
                                                      "a": np.average(g["a"], weights=g["w"])}), include_groups=False)
        out["slope_gq"] = float(np.polyfit(t["a"], t["m"], 1)[0])
        return out


BOOT_KEYS = ["deviance", "deviance_pos", "d1_dev_cond", "ratio_env", "ratio_gap", "slope_env_model", "slope_gap_model",
             "slope_oq", "slope_zq",
             "slope_gapq", "slope_dq", "slope_gq", "elast_ratio", "count_gap", "game_cal_slope"]
ABS1_KEYS = ["ratio_env", "ratio_gap", "slope_zq", "slope_gapq", "slope_dq", "elast_ratio"]   # |x - 1| lines


def segment_lines(d: pd.DataFrame, arms: list[str]) -> dict:
    segs = {}
    for by in ("month", "site", "prev_end", "tier", "half", "teamq"):
        for k, g in d.groupby(by):
            ng = g["game_id"].nunique()
            s = {"n_rows": int(len(g)), "n_games": int(ng), "actual_mean_s": float(g["dur_poss"].mean())}
            pg = g.groupby("game_id").agg(n=("dur_poss", "size"), A=("dur_poss", "sum"), env=("env", "first"),
                                          gap=("gap", "first"), **{f"T_{a}": (f"t_{a}", "sum") for a in arms})
            pg = pg[(pg["A"] > 0) & np.all([pg[f"T_{a}"] > 0 for a in arms], axis=0)]
            pa = 1200 * pg["n"].to_numpy() / pg["A"].to_numpy()
            for a in arms:
                r = {"gap_s": float(g[f"t_{a}"].mean() - g["dur_poss"].mean()),
                     "count_gap": float(1200 / g[f"t_{a}"].mean() - 1200 / g["dur_poss"].mean())}
                if len(pg) >= 150:
                    pm = 1200 * pg["n"].to_numpy() / pg[f"T_{a}"].to_numpy()
                    w = np.ones(len(pg))
                    for nm in ("env", "gap"):
                        x = pg[nm].to_numpy()[:, None]
                        r[f"ratio_{nm}"] = float(wls(pm, x, w)[0] / wls(pa, x, w)[0])
                else:
                    r["ratio_env"] = r["ratio_gap"] = "UNDERPOWERED"
                if len(g) >= 3000:
                    t = g.groupby("zq").agg(m=(f"t_{a}", "mean"), act=("dur_poss", "mean"))
                    r["slope_zq"] = float(np.polyfit(t["act"], t["m"], 1)[0]) if len(t) > 1 else None
                else:
                    r["slope_zq"] = "UNDERPOWERED"
                s[a] = r
            segs[f"{by}={k}"] = s
    return segs


def team_sign_share(d: pd.DataFrame, arms: list[str], min_rows: int = 300) -> dict:
    g = d.groupby("offense_team_id").agg(n=("dur_poss", "size"), act=("dur_poss", "mean"),
                                         **{f"m_{a}": (f"t_{a}", "mean") for a in arms})
    g = g[g["n"] >= min_rows]
    out = {"n_teams": int(len(g)), "min_rows": min_rows}
    for a in arms:
        if a == "E0":
            continue
        da = g[f"m_{a}"] - g["m_E0"]
        dr = g["act"] - g["m_E0"]
        moved = da.abs() > 1e-9
        out[a] = {"share_right_sign": float((np.sign(da[moved]) == np.sign(dr[moved])).mean()) if moved.any() else None,
                  "n_moved": int(moved.sum()),
                  "share_abs_err_down": float(((g[f"m_{a}"] - g["act"]).abs() < (g["m_E0"] - g["act"]).abs()).mean())}
    return out


def teamgame_xoff(d: pd.DataFrame, arms: list[str]) -> dict:
    tg = d.groupby(["game_id", "offense_team_id"]).agg(n=("dur_poss", "size"), A=("dur_poss", "sum"),
                                                       x=("z_raw", "first"), **{f"T_{a}": (f"t_{a}", "sum") for a in arms})
    pa = 1200 * tg["n"].to_numpy() / tg["A"].to_numpy()
    x = tg["x"].to_numpy()[:, None]
    w = np.ones(len(tg))
    sa = wls(pa, x, w)[0]
    return {"actual_slope": float(sa), **{a: float(wls(1200 * tg["n"].to_numpy() / tg[f"T_{a}"].to_numpy(), x, w)[0] / sa)
                                          for a in arms}}


# --------------------------------------------------------------------------- decision (section 41.A section 5)
def decide(F: dict, arms: list[str]) -> dict:
    """Literal rule. F[fold]['arms'][arm] carries lines and 'vs_E0' deltas/floors."""
    out = {"rule": ("41.A s5: an arm WINS if on fold 2 (a) deviance beats E0 by more than its floor [41.0 item 2: and "
                    "the win survives on E0's positive-mass subset beyond that line's floor], (b) the env and gap "
                    "possession-slope ratios each move toward 1 by more than their floors and land in [0.7, 1.3], "
                    "(c) the quintile slopes (a)-(d) hold [(a) offence tempo in [0.85, 1.15] and not below E0 by more "
                    "than its floor; (b)-(d) target 1, not further from 1 than E0 by more than the floor; game-level "
                    "tempo elasticity ratio within [0.9, 1.1]], (d) fold 1 does not reverse the sign of (a) or (b). "
                    "Vetoes: |implied possessions per team-game gap| worse than E0's by > 0.10; any start type's "
                    "|mean gap| worse by > 0.20 s; any of Nov-Mar worse by > 0.25 s. Among winners the simplest "
                    "(E1 < E2 < E3) whose deviance and slope ratios are each within one floor of the best winner's. "
                    "If none wins the round is REFUTED."), "arms": {}}
    f2, f1 = F.get("F2"), F.get("F1")
    winners, rel_winners = [], []
    for a in arms:
        if a not in CANDIDATES or a not in f2["arms"]:
            continue
        L, V = f2["arms"][a], f2["arms"][a]["vs_E0"]
        c = {}
        c["a_dev"] = V["d_deviance"] < -V["d_deviance_floor"]
        c["a_dev_pos_gate"] = V["d_deviance_pos"] < -V["d_deviance_pos_floor"]
        for nm in ("env", "gap"):
            c[f"b_{nm}_toward1"] = V[f"d_abs1_ratio_{nm}"] < -V[f"d_abs1_ratio_{nm}_floor"]
            c[f"b_{nm}_in_0.7_1.3"] = 0.7 <= L[f"ratio_{nm}"] <= 1.3
        c["c_oq_in_0.85_1.15"] = 0.85 <= L["slope_oq"] <= 1.15
        c["c_oq_not_below_E0"] = V["d_slope_oq"] >= -V["d_slope_oq_floor"]
        for q in ("zq", "gapq", "dq"):
            c[f"c_{q}_not_away"] = V[f"d_abs1_slope_{q}"] <= V[f"d_abs1_slope_{q}_floor"]
        c["c_elast_in_0.9_1.1"] = 0.9 <= L["elast_ratio"] <= 1.1
        if f1 is not None and a in f1["arms"]:
            V1 = f1["arms"][a]["vs_E0"]
            c["d_F1_dev_sign"] = V1["d_deviance"] < 0
            c["d_F1_env_sign"] = V1["d_abs1_ratio_env"] < 0
            c["d_F1_gap_sign"] = V1["d_abs1_ratio_gap"] < 0
        else:
            c["d_F1_dev_sign"] = c["d_F1_env_sign"] = c["d_F1_gap_sign"] = False
        veto = {"v_count_gap": V["d_abs_count_gap"] > 0.10,
                "v_start": V["max_start_gap_worsening_s"] > 0.20,
                "v_month": V["max_month_gap_worsening_s_NovMar"] > 0.25}
        wins = all(c.values()) and not any(veto.values())
        rel = {k: v for k, v in c.items() if k not in ("c_oq_in_0.85_1.15", "c_elast_in_0.9_1.1")}
        rel["c_elast_not_away"] = V["d_abs1_elast_ratio"] <= V["d_abs1_elast_ratio_floor"]
        rel_wins = all(rel.values()) and not any(veto.values())
        sd_rep = {nm: {"d_slope_model": V[f"d_slope_{nm}_model"], "floor": V[f"d_slope_{nm}_model_floor"],
                       "toward_actual_beyond_floor": bool(np.sign(V[f"d_slope_{nm}_model"]) ==
                                                           np.sign(L[f"slope_{nm}_act"] - f2["arms"]["E0"][f"slope_{nm}_model"])
                                                           and abs(V[f"d_slope_{nm}_model"]) > V[f"d_slope_{nm}_model_floor"])}
                  for nm in ("env", "gap")}
        out["arms"][a] = {"checks": c, "vetoes": veto, "WINS": bool(wins), "slope_difference_reading_REPORTED": sd_rep,
                          "relative_reading_checks": rel, "relative_reading_WINS": bool(rel_wins)}
        if wins:
            winners.append(a)
        if rel_wins:
            rel_winners.append(a)

    def pick(ws):
        if not ws:
            return None
        best_dev = min(f2["arms"][a]["deviance"] for a in ws)
        best = {nm: min(abs(f2["arms"][a][f"ratio_{nm}"] - 1) for a in ws) for nm in ("env", "gap")}
        for a in CANDIDATES:
            if a not in ws:
                continue
            V = f2["arms"][a]["vs_E0"]
            ok = (f2["arms"][a]["deviance"] - best_dev <= V["d_deviance_floor"]
                  and all(abs(f2["arms"][a][f"ratio_{nm}"] - 1) - best[nm] <= V[f"d_abs1_ratio_{nm}_floor"]
                          for nm in ("env", "gap")))
            if ok:
                return a
        return ws[0]
    out["winners"], out["selected"] = winners, pick(winners)
    out["outcome"] = (f"WINNER {out['selected']}" if out["selected"] else "REFUTED (no arm wins by the literal rule)")
    out["relative_reading_winners"], out["relative_reading_selected"] = rel_winners, pick(rel_winners)
    for a in arms:
        if a in NOISE:
            out["arms"][a] = {"WINS": False, "note": "noise-floor arm, never selectable"}
        if a in REFS:
            out["arms"][a] = {"WINS": False, "note": "reference only, never selectable"}
    return out


# --------------------------------------------------------------------------- main
def fmt(v, nd=4):
    return v if isinstance(v, str) else ("nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="E0,E0s1,E1,E2,E3,K2M")
    ap.add_argument("--folds", default="F2,F1")
    ap.add_argument("--tag", default="offline")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    rep = {"floor_rule": f"2 x SD of the paired game-block bootstrap ({NBOOT} draws, seed {BOOT_SEED}) of arm - E0",
           "codes": {}, "folds": {}, "reseed_proof": {}}
    md = ["# Clock round 9 offline grade (coded arms; key below the tables)", ""]
    for fold in a.folds.split(","):
        d = rows[fold]
        arms = [x for x in a.arms.split(",") if manifest_of(x, fold).exists()]
        if "E0" not in arms:
            raise FileNotFoundError("E0 (served K2) schedule missing")
        scheds = {}
        for arm in arms:
            scheds[arm] = G8.load_schedule(manifest_of(arm, fold))
            r = score_arm(scheds[arm], d)
            for k, v in r.items():
                d[f"{k}_{arm}"] = v
            log(f"{fold} {code_of(arm)} scored")
        pos = d["prp_E0"].to_numpy() > CLIP
        A = Agg(d, arms, pos)
        w1 = np.ones(A.G)
        fr = {"n_rows": int(len(d)), "n_games": int(A.G), "e0_zero_mass_rows": int((~pos).sum()), "arms": {}}
        for arm in arms:
            L = A.lines(arm, w1)
            st = d.groupby("prev_end").agg(m=(f"t_{arm}", "mean"), act=("dur_poss", "mean"), n=("dur_poss", "size"))
            L["gap_by_start"] = (st["m"] - st["act"]).round(4).to_dict()
            L["n_by_start"] = st["n"].astype(int).to_dict()
            mo = d.groupby("month").agg(m=(f"t_{arm}", "mean"), act=("dur_poss", "mean"), n=("dur_poss", "size"))
            L["gap_by_month"] = {str(k): round(v, 4) for k, v in (mo["m"] - mo["act"]).items()}
            L["n_by_month"] = {str(k): int(v) for k, v in mo["n"].items()}
            fr["arms"][arm] = L
        rng = np.random.default_rng(BOOT_SEED)
        boots = {arm: {k: [] for k in BOOT_KEYS} for arm in arms}
        for _ in range(NBOOT):
            w = np.bincount(rng.integers(0, A.G, A.G), minlength=A.G).astype(float)
            for arm in arms:
                Lb = A.lines(arm, w)
                for k in BOOT_KEYS:
                    boots[arm][k].append(Lb[k])
        e0 = fr["arms"]["E0"]
        for arm in arms:
            if arm == "E0":
                continue
            L = fr["arms"][arm]
            V = {}
            for k in BOOT_KEYS:
                diff = np.array(boots[arm][k]) - np.array(boots["E0"][k])
                V[f"d_{k}"] = L[k] - e0[k]
                V[f"d_{k}_floor"] = 2 * float(np.std(diff))
            for k in ABS1_KEYS:
                diff = np.abs(np.array(boots[arm][k]) - 1) - np.abs(np.array(boots["E0"][k]) - 1)
                V[f"d_abs1_{k}"] = abs(L[k] - 1) - abs(e0[k] - 1)
                V[f"d_abs1_{k}_floor"] = 2 * float(np.std(diff))
            V["d_abs_count_gap"] = abs(L["count_gap"]) - abs(e0["count_gap"])
            V["max_start_gap_worsening_s"] = float(max(abs(L["gap_by_start"][k]) - abs(e0["gap_by_start"][k])
                                                       for k in e0["gap_by_start"]))
            V["max_month_gap_worsening_s_NovMar"] = float(max(abs(L["gap_by_month"][k]) - abs(e0["gap_by_month"][k])
                                                              for k in ("11", "12", "1", "2", "3") if k in e0["gap_by_month"]))
            L["vs_E0"] = V
        fr["segments"] = segment_lines(d, arms)
        fr["team_sign_share"] = team_sign_share(d, arms)
        fr["teamgame_xoff_ratio"] = teamgame_xoff(d, arms)
        # reseed proofs
        rp = {}
        if "E0s1" in scheds:
            rp["E0_vs_E0s1_max_abs_pmf_diff"] = pmf_diff(scheds["E0"], scheds["E0s1"], d)
        for arm in CANDIDATES:
            p1 = CKD / f"r9_env_{arm}" / f"{fold}_seed1" / "manifest.json"
            if arm in scheds and p1.exists():
                rp[f"{arm}_vs_seed1_max_abs_pmf_diff"] = pmf_diff(scheds[arm], G8.load_schedule(p1), d)
        rep["reseed_proof"][fold] = rp
        rep["folds"][fold] = fr
        # coded table
        md += [f"## {fold} ({fr['n_rows']:,} rows, {fr['n_games']:,} games; E0 zero-mass rows {fr['e0_zero_mass_rows']})", "",
               "| arm | deviance (d vs E0; floor) | deviance on E0>0 rows (d; floor) | excl. share | d1 dev cond | "
               "ratio env (d|r-1|; floor) | ratio gap (d|r-1|; floor) | slope oq (d; floor) | slope zq | slope gapq | slope dq | "
               "slope gq | elast ratio | count gap (d|gap|) |",
               "|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
        for arm in sorted(arms, key=code_of):
            L = fr["arms"][arm]
            V = L.get("vs_E0")
            if V is None:
                md.append(f"| {code_of(arm)} | {fmt(L['deviance'])} | {fmt(L['deviance_pos'])} | {fmt(L['excluded_share'], 5)} | "
                          f"{fmt(L['d1_dev_cond'])} | {fmt(L['ratio_env'], 3)} | {fmt(L['ratio_gap'], 3)} | {fmt(L['slope_oq'], 3)} | "
                          f"{fmt(L['slope_zq'], 3)} | {fmt(L['slope_gapq'], 3)} | {fmt(L['slope_dq'], 3)} | {fmt(L['slope_gq'], 3)} | "
                          f"{fmt(L['elast_ratio'], 3)} | {fmt(L['count_gap'], 3)} |")
            else:
                md.append(f"| {code_of(arm)} | {fmt(L['deviance'])} ({V['d_deviance']:+.4f}; {V['d_deviance_floor']:.4f}) | "
                          f"{fmt(L['deviance_pos'])} ({V['d_deviance_pos']:+.4f}; {V['d_deviance_pos_floor']:.4f}) | "
                          f"{fmt(L['excluded_share'], 5)} | {fmt(L['d1_dev_cond'])} | "
                          f"{fmt(L['ratio_env'], 3)} ({V['d_abs1_ratio_env']:+.3f}; {V['d_abs1_ratio_env_floor']:.3f}) | "
                          f"{fmt(L['ratio_gap'], 3)} ({V['d_abs1_ratio_gap']:+.3f}; {V['d_abs1_ratio_gap_floor']:.3f}) | "
                          f"{fmt(L['slope_oq'], 3)} ({V['d_slope_oq']:+.3f}; {V['d_slope_oq_floor']:.3f}) | "
                          f"{fmt(L['slope_zq'], 3)} | {fmt(L['slope_gapq'], 3)} | {fmt(L['slope_dq'], 3)} | {fmt(L['slope_gq'], 3)} | "
                          f"{fmt(L['elast_ratio'], 3)} | {fmt(L['count_gap'], 3)} ({V['d_abs_count_gap']:+.3f}) |")
        md.append("")
        log(f"{fold} graded")
    rep["codes"] = {code_of(x): x for x in a.arms.split(",")}
    if "F2" in rep["folds"]:
        rep["decision"] = decide(rep["folds"], list(rep["folds"]["F2"]["arms"]))
    md += ["## Key (written after the coded tables)", "", *[f"- {k} = {v}" for k, v in sorted(rep["codes"].items())], ""]
    if "decision" in rep:
        md += ["## Decision (literal rule)", "", f"Rule: {rep['decision']['rule']}", "",
               f"**{rep['decision']['outcome']}**; winners {rep['decision']['winners']}; relative reading (REPORTED ONLY, "
               f"not the rule): winners {rep['decision']['relative_reading_winners']}, selected "
               f"{rep['decision']['relative_reading_selected']}", ""]
        for arm, r in rep["decision"]["arms"].items():
            md.append(f"- {arm}: {json.dumps(r, default=str)}")
    (OUT / f"{a.tag}_grade.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    (OUT / f"{a.tag}_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote {OUT / (a.tag + '_grade.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
