#!/usr/bin/env python
"""
train_clock_v3c_site.py -- Lane G (overnight 2026-09-30): clock site-term
bake-off. Pre-registration: `docs/models/clock/experiments.md` section 30
(committed e5dd38c BEFORE this ran).

    .venv/Scripts/python.exe scripts/train_clock_v3c_site.py --arms C0,C1,C2 --folds F2,F1

Arms (S1 monthly, both folds):
    C0  served base `empirical_km3_srfloor | P3`
    C1  C0 grid + offence site (neutral/home/away) as the LAST cell dimension
    C2  C0 pmf, durations rescaled by exp(beta * site_signed); beta per refit
        by censored log-likelihood on the refit's training rows

Writes predictions in the shared grading schema (y = realised duration of
uncensored possessions, p = predicted mean of the pmf truncated below the
row's seconds remaining) and CRPS_trunc per arm to results/home_site/clock/.
Sibling of train_clock_v3c_s1.py (not edited); `clock.py`/`clock_v3.py` are
read only -- the extra feature set is registered in THIS process's dicts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

NJ = int(os.environ.get("CBB_NJOBS", "2"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(NJ)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import clock as ck  # noqa: E402
from cbb_sim.models import clock_v3 as c3  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

DESIGN_V2 = Path("data/processed/models/clock/design_v2.parquet")
ALL_SEASONS = [2022, 2023, 2024, 2025]
OUT = Path("results/home_site/clock")
BETA_GRID = np.round(np.arange(-0.05, 0.05 + 1e-9, 0.0025), 4)
BETA_SUBSAMPLE = 300_000
GRID = c3.GRID
N_GRID = c3.N_GRID

# --- C1: register the site dimension in THIS process only -------------------
ck.FEATURE_SETS["P3S_dummy"] = list(c3.FEATURES_P3_DUMMY) + ["site_home", "site_away"]
ck.EMPIRICAL_DIM_SIZES["site_code"] = 3
ck.EMPIRICAL_DIMS["P3S_dummy"] = tuple(ck.EMPIRICAL_DIMS["P3_dummy"]) + ("site_code",)


def site_code(df: pd.DataFrame) -> np.ndarray:
    return (df["site_home"].to_numpy(dtype="int64") * 1 + df["site_away"].to_numpy(dtype="int64") * 2)


class EmpiricalArmV3PS(c3.EmpiricalArmV3P):
    def _codes(self, df):
        codes = super()._codes(df)
        codes["site_code"] = site_code(df)
        return codes


def fit_empirical_ps(tr: pd.DataFrame) -> c3.StateWrapArm:
    """`clock_v3._fit_empirical_p` verbatim, with the site-aware arm class and
    the P3S dims (site last). sr floor as served."""
    fs = "P3S_dummy"
    tr = c3.add_p_state(tr.copy())
    dims = ck.EMPIRICAL_DIMS[fs]
    sizes = tuple(ck.EMPIRICAL_DIM_SIZES[d] for d in dims)
    tempo = tr["tempo_prior_game"].to_numpy(dtype="float64")
    arm = EmpiricalArmV3PS(
        feature_set_name=fs, dims=dims, sizes=sizes,
        level_pmfs=[], level_counts=[], level_events=[],
        tempo_edges=(float(np.quantile(tempo, 1 / 3)), float(np.quantile(tempo, 2 / 3))),
        season_map={s: i for i, s in enumerate(sorted(int(x) for x in tr["season"].unique()))},
        sr_floor_bucket=c3.SR_FLOOR_BUCKET, features=ck.feature_set(fs),
        name="empirical_km3_srfloor_P3S")
    codes = arm._codes(tr)
    y = tr["duration_s"].to_numpy(dtype="int64")
    cen = tr["censored"].to_numpy(dtype=bool)
    pooled = np.zeros((1, N_GRID), dtype="float64")
    np.add.at(pooled, (np.zeros(int((~cen).sum()), dtype="int64"), y[~cen]), 1.0)
    pooled = ck._normalise(pooled)
    for lv in range(len(dims) + 1):
        n_cells = 1 if lv == 0 else int(np.prod(sizes[:lv]))
        keys = arm._keys(codes, lv)
        if lv == 0:
            parent_pmf, parent_of = pooled, np.zeros(1, dtype="int64")
        else:
            parent_pmf = arm.level_pmfs[lv - 1]
            parent_of = ((np.arange(n_cells, dtype="int64") // sizes[lv - 1])
                         if lv > 1 else np.zeros(n_cells, dtype="int64"))
        pmf, counts, events = c3.kaplan_meier_pmf_v3(keys, y, cen, n_cells, parent_pmf, parent_of)
        arm.level_pmfs.append(ck._normalise(pmf))
        arm.level_counts.append(counts)
        arm.level_events.append(events)
    arm.level_counts[0] = np.maximum(arm.level_counts[0], arm.min_cell)
    arm.level_events[0] = np.maximum(arm.level_events[0], arm.min_events)
    return c3.StateWrapArm(arm, "P3")


def scale_matrix(f: float) -> np.ndarray:
    """(N_GRID, N_GRID) mass map T -> round(T * f), clipped to the grid."""
    M = np.zeros((N_GRID, N_GRID))
    tgt = np.clip(np.rint(GRID * f).astype(int), 0, N_GRID - 1)
    M[np.arange(N_GRID), tgt] = 1.0
    return M


@dataclass
class ScaledArm:
    inner: object
    beta: float

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        p = self.inner.pmf(df)
        s = (df["site_home"].to_numpy(dtype="float64") - df["site_away"].to_numpy(dtype="float64"))
        for sv in (1.0, -1.0):
            r = np.flatnonzero(s == sv)
            if len(r):
                p[r] = p[r] @ scale_matrix(float(np.exp(self.beta * sv)))
        return p


def fit_beta(inner, rows: pd.DataFrame, seed: int) -> tuple[float, dict]:
    s = (rows["site_home"].to_numpy(dtype="float64") - rows["site_away"].to_numpy(dtype="float64"))
    nn = np.flatnonzero(s != 0)
    rng = np.random.default_rng(seed)
    if len(nn) > BETA_SUBSAMPLE:
        nn = np.sort(rng.choice(nn, BETA_SUBSAMPLE, replace=False))
    sub = rows.iloc[nn]
    ss = s[nn]
    y = sub["duration_s"].to_numpy(dtype="int64")
    cen = sub["censored"].to_numpy(dtype=bool)
    r = sub["seconds_remaining"].to_numpy(dtype="float64")
    p0 = inner.pmf(sub)
    ll = {}
    for b in BETA_GRID:
        p = p0.copy()
        for sv in (1.0, -1.0):
            m = ss == sv
            p[m] = p0[m] @ scale_matrix(float(np.exp(b * sv)))
        ll[float(b)] = float(c3.censored_loglik_rows(p, y, cen, r).mean())
    best = max(ll, key=ll.get)
    return best, ll


def trunc_mean(arm, df: pd.DataFrame, chunk: int = 100_000) -> np.ndarray:
    out = np.empty(len(df))
    r = df["seconds_remaining"].to_numpy(dtype="float64")
    for a in range(0, len(df), chunk):
        b = min(a + chunk, len(df))
        tp, _ = c3.truncate_pmf(arm.pmf(df.iloc[a:b]), r[a:b])
        out[a:b] = tp @ GRID.astype("float64")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="C0,C1,C2")
    ap.add_argument("--folds", default="F2,F1")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(ALL_SEASONS, context="clock site round")
    cols = list(dict.fromkeys([*c3.s1_fit_columns("empirical_km3"),
                               *ck.feature_set("P3_dummy"), "site_home", "site_away",
                               "game_id", "offense_team_id", "defense_team_id", "censored_horn",
                               "censored_old"]))
    design = pd.read_parquet(DESIGN_V2)
    design, cdiag = c3.attach_horn_censoring(design, ALL_SEASONS)
    c3.set_flag_inplace(design, "horn")
    design = design[[c for c in cols if c in design.columns]].copy()
    print(f"[{time.time() - t0:5.0f}s] design {design.shape}", flush=True)
    for fold in a.folds.split(","):
        tr, te = ck.fold_slices(design, fold)
        te = te.reset_index(drop=True)
        te_dates = pd.to_datetime(te["game_date"])
        tr_dates = pd.to_datetime(tr["game_date"])
        cuts = PO.month_boundaries(te_dates)
        for arm_name in a.arms.split(","):
            t1 = time.time()
            p = np.zeros(len(te))
            crps_rows = np.full(len(te), np.nan)
            betas = []
            for k, cut in enumerate(cuts):
                nxt = cuts[k + 1] if k + 1 < len(cuts) else None
                seg = ((te_dates >= cut) if nxt is None
                       else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
                if not seg.any():
                    continue
                before = (te_dates < cut).to_numpy()
                rows = tr if not before.any() else pd.concat([tr, te.loc[before]], ignore_index=True)
                if arm_name == "C1":
                    arm = fit_empirical_ps(rows)
                else:
                    arm = c3.fit_arm_v3b("empirical_km3_srfloor", "P3", rows, seed=20260910)
                    if arm_name == "C2":
                        b, ll = fit_beta(arm, rows, seed=20260930 + k)
                        betas.append({"refit": str(pd.Timestamp(cut).date()), "beta": b,
                                      "ll_at_0": ll[0.0], "ll_at_beta": ll[b]})
                        arm = ScaledArm(arm, b)
                ts = te.loc[seg]
                p[seg] = trunc_mean(arm, ts)
                sc = c3.score_arm_v3(arm, ts)
                crps_rows[np.flatnonzero(seg)] = sc["_crps_trunc_rows"]
                del rows
            unc = ~te["censored"].to_numpy(dtype=bool)
            out = pd.DataFrame({
                "game_id": te["game_id"].to_numpy(), "off_id": te["offense_team_id"].to_numpy(),
                "def_id": te["defense_team_id"].to_numpy(),
                "site": (te["site_home"] - te["site_away"]).to_numpy().astype("int8"),
                "y": te["duration_s"].to_numpy(dtype="float64"), "p": p, "w": 1.0,
                "game_date": te_dates.to_numpy(), "driver": te["tempo_prior_game"].to_numpy(),
                "driver_defined": True, "crps_trunc": crps_rows})[unc]
            out.to_parquet(OUT / f"preds_{arm_name}_{fold}_s0.parquet", index=False)
            meta = {"arm": arm_name, "fold": fold, "n_test_uncensored": int(unc.sum()),
                    "crps_trunc": float(np.nanmean(crps_rows[unc])), "betas": betas,
                    "pred_mean": float(p[unc].mean()), "act_mean": float(out["y"].mean()),
                    "fit_s": round(time.time() - t1, 1)}
            (OUT / f"meta_{arm_name}_{fold}_s0.json").write_text(json.dumps(meta, indent=1, default=float))
            print(f"[{time.time() - t0:5.0f}s] {arm_name} {fold}: crps_trunc {meta['crps_trunc']:.6f} "
                  f"pred {meta['pred_mean']:.3f} act {meta['act_mean']:.3f} betas "
                  f"{[x['beta'] for x in betas]} ({meta['fit_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
