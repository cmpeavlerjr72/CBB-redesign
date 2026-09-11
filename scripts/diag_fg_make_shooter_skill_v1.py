#!/usr/bin/env python
"""
diag_fg_make_shooter_skill_v1.py -- how much GENUINE shooter skill is there per
shot class, once the shooter is keyed on `shot_shooter_id`?

    .venv/Scripts/python.exe scripts/diag_fg_make_shooter_skill_v1.py

Writes `data/processed/models/fg_make/shooter_skill_v1.json`; the prose is
`docs/tests/fg_make_shooter_skill_2026-09-10.md`.

WHY THIS RUNS BEFORE ROUND 4 IS PRE-REGISTERED
----------------------------------------------
Round 3 (L29) found the `FGA_3` shooter-quintile span collapses from 35.9 pp to
2.8 pp once the shooter label is corrected. Two readings of that number are
possible and they imply opposite round-4 designs:

  * 2.8 pp IS the truth -- college three-point shooting is nearly all team shot
    quality and noise, and the honest shooter block is small or absent;
  * 2.8 pp is UNDER-SHRINKAGE -- the raw as-of rate on a handful of attempts is
    mostly binomial noise, a tree fed the raw rate learns to distrust it, and
    the predicted span collapses even though the true between-shooter SD is
    much larger.

These are separable WITHOUT fitting any model, which is what this script does.
Four measurements per class, all on 2022-2025 (2026 sealed) and all on the
CORRECTED label:

 1. **Method-of-moments between-shooter SD.** Per (season, shooter) attempt and
    make counts; the beta-binomial moment estimator splits the observed spread
    of shooter rates into binomial noise and true between-shooter variance.
    This is the quantity "is 2.8 pp the truth" is really asking about.
 2. **Reliability of an as-of rate** at 25 / 50 / 100 / 200 prior attempts:
    `n / (n + m)` with `m = p(1-p) / sigma^2` from (1). `m` is also exactly the
    shrinkage strength a B1-style arm should use, so it is a fitted quantity
    this round can hand to the trainer rather than a grid guess.
 3. **Prior-season to current-season correlation** of the same shooter's rate,
    raw and disattenuated for the binomial noise in both seasons. A skill that
    is real persists across a summer; one that is shot-quality context does not
    persist when the player's role or team changes.
 4. **Slope of the realised make rate on the as-of rate, BY ATTEMPT-COUNT
    BUCKET.** If the raw as-of rate is unbiased, the slope is 1 in every
    bucket; if it is noisy, the slope is well below 1 at low counts and climbs.
    That slope profile IS the under-shrinkage test, measured directly on
    outcomes with no model in between.

Nothing here fits or selects anything. It is evidence for the round-4
pre-registration and is committed before that round runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import fg_make as FG  # noqa: E402

FG_DIR = Path("data/processed/models/fg_make")
DESIGN = FG_DIR / "design_v2_shotshooter.parquet"
EVENTS = FG_DIR / "events_v2_shotshooter.parquet"
OUT = FG_DIR / "shooter_skill_v1.json"
SEASONS = [2022, 2023, 2024, 2025]

#: Where the reliability question is actually asked. A college season is
#: 30-35 games, so 200 three-point attempts is a high-volume starter's WHOLE
#: season and 25 is about three weeks of one.
RELIABILITY_AT = (25, 50, 100, 200)
#: Attempt-count buckets for the slope profile. The first is the "no history"
#: cell the model sees on opening night and every new player.
ATT_BUCKETS = ((0, 0), (1, 24), (25, 49), (50, 99), (100, 199), (200, 10**9))
MIN_CELL = 500          # below this a cell is labelled UNDERPOWERED, never read


# ---------------------------------------------------------------------------
# 1. method-of-moments between-shooter SD
# ---------------------------------------------------------------------------
def mom_between_sd(n: np.ndarray, k: np.ndarray) -> dict:
    """Beta-binomial moment estimator of the between-cluster SD.

        S          = sum_i n_i (p_i - pbar)^2
        E[S]       = (I - 1) pbar(1 - pbar) + (N - sum n_i^2 / N) sigma^2
        sigma^2hat = [S - (I - 1) pbar(1 - pbar)] / (N - sum n_i^2 / N)

    `n_i` attempts and `k_i` makes per cluster (here: one shooter-season). The
    first term of E[S] is the binomial noise a set of I identical shooters
    would produce, so subtracting it is what separates skill from sampling.
    A negative estimate means the observed spread is SMALLER than binomial
    noise alone; it is reported as measured (clipped only where an SD is
    printed) rather than hidden."""
    n = np.asarray(n, dtype="float64")
    k = np.asarray(k, dtype="float64")
    keep = n > 0
    n, k = n[keep], k[keep]
    I, N = len(n), n.sum()
    if I < 2 or N <= 0:
        return {"n_clusters": int(I), "n_trials": float(N), "sigma2": None}
    p = k / n
    pbar = k.sum() / N
    S = float((n * (p - pbar) ** 2).sum())
    denom = float(N - (n ** 2).sum() / N)
    binom = (I - 1) * pbar * (1.0 - pbar)
    sigma2 = (S - binom) / denom if denom > 0 else None
    out = {
        "n_clusters": int(I), "n_trials": float(N), "mean_rate": round(float(pbar), 6),
        "S": round(S, 4), "expected_binomial_S": round(float(binom), 4),
        "denom": round(denom, 4),
        "sigma2": None if sigma2 is None else round(float(sigma2), 8),
    }
    if sigma2 is not None:
        out["between_sd_pp"] = round(float(np.sqrt(max(sigma2, 0.0))) * 100, 4)
        out["skill_share_of_observed_spread"] = (
            round(float(max(S - binom, 0.0) / S), 4) if S > 0 else None)
        out["m_shrinkage_attempts"] = (
            round(float(pbar * (1 - pbar) / sigma2), 2) if sigma2 > 0 else None)
        # +/-4 sigma is the realised span a 5-quintile bucketing of a NORMAL
        # skill distribution would show between the top and bottom quintile
        # means (E[top 20%] - E[bottom 20%] = 2 * 1.3998 * sigma).
        out["implied_quintile_span_pp"] = (
            round(2 * 1.3998 * float(np.sqrt(max(sigma2, 0.0))) * 100, 3))
    return out


def reliability(m: float | None, at=RELIABILITY_AT) -> dict:
    if m is None or not np.isfinite(m) or m <= 0:
        return {str(n): None for n in at}
    return {str(n): round(float(n / (n + m)), 4) for n in at}


# ---------------------------------------------------------------------------
# 2. per-shooter-season counts, from the design itself
# ---------------------------------------------------------------------------
def shooter_season_counts(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["season", "shooter_id"], as_index=False).agg(
        att=("y", "size"), mk=("y", "sum"))
    g["rate"] = g["mk"] / g["att"]
    return g


def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(slope, intercept, SE of slope) of y on x, no library."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n = len(x)
    if n < 30:
        return float("nan"), float("nan"), float("nan")
    xb, yb = x.mean(), y.mean()
    sxx = float(((x - xb) ** 2).sum())
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    b = float(((x - xb) * (y - yb)).sum() / sxx)
    a = float(yb - b * xb)
    resid = y - (a + b * x)
    s2 = float((resid ** 2).sum() / max(n - 2, 1))
    return b, a, float(np.sqrt(s2 / sxx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="fg_make shooter-skill evidence")

    if not DESIGN.exists():
        raise FileNotFoundError(
            f"{DESIGN} missing; run scripts/build_engine_inputs_shotshooter.py "
            "(it caches the shot_shooter_id design)")
    d = pd.read_parquet(DESIGN)
    d = d[d["season"].isin(SEASONS)]
    print(f"design {d.shape} (shot_shooter_id) in {time.time() - t0:.1f}s", flush=True)

    report: dict = {
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "design": str(DESIGN), "shooter_key": "shot_shooter_id",
        "seasons": SEASONS, "n_rows": int(len(d)), "by_class": {},
    }

    for cls_name in FG.SHOT_CLASSES:
        sl = d[d["shot_class"] == cls_name]
        cls_out: dict = {"n_attempts": int(len(sl))}

        # ---- 1. between-shooter SD, pooled and by season -------------------
        sc = shooter_season_counts(sl)
        cls_out["mom_pooled"] = mom_between_sd(sc["att"].to_numpy(), sc["mk"].to_numpy())
        cls_out["mom_by_season"] = {}
        for s in SEASONS:
            ss = sc[sc["season"] == s]
            cls_out["mom_by_season"][str(s)] = mom_between_sd(
                ss["att"].to_numpy(), ss["mk"].to_numpy())
        # the same estimator restricted to shooters with real volume, so a
        # reader can see whether the pooled number is driven by the long tail
        # of 1-5 attempt players
        cls_out["mom_min_att"] = {}
        for lo in (10, 25, 50):
            ss = sc[sc["att"] >= lo]
            cls_out["mom_min_att"][str(lo)] = mom_between_sd(
                ss["att"].to_numpy(), ss["mk"].to_numpy())

        m = cls_out["mom_pooled"].get("m_shrinkage_attempts")
        cls_out["reliability_of_asof_rate"] = reliability(m)
        cls_out["reliability_note"] = (
            "reliability(n) = n / (n + m), m = p(1-p)/sigma^2 from mom_pooled")

        # ---- 3. prior season -> current season ----------------------------
        cur = sc.rename(columns={"att": "att_cur", "mk": "mk_cur", "rate": "rate_cur"})
        prv = sc.copy()
        prv["season"] = prv["season"] + 1
        prv = prv.rename(columns={"att": "att_prev", "mk": "mk_prev", "rate": "rate_prev"})
        j = cur.merge(prv, on=["season", "shooter_id"], how="inner")
        ps: dict = {}
        for lo in (1, 25, 50, 100):
            sub = j[(j["att_prev"] >= lo) & (j["att_cur"] >= lo)]
            if len(sub) < 30:
                ps[str(lo)] = {"n": int(len(sub)), "verdict": "UNDERPOWERED"}
                continue
            r = float(np.corrcoef(sub["rate_prev"], sub["rate_cur"])[0, 1])
            # disattenuate: divide by sqrt(reliability_prev * reliability_cur),
            # each reliability computed at that subset's OWN mean attempt count
            if m and m > 0:
                rel_p = float(sub["att_prev"].mean() / (sub["att_prev"].mean() + m))
                rel_c = float(sub["att_cur"].mean() / (sub["att_cur"].mean() + m))
                r_dis = r / np.sqrt(rel_p * rel_c)
            else:
                rel_p = rel_c = r_dis = float("nan")
            b, _, se = ols_slope(sub["rate_prev"].to_numpy(), sub["rate_cur"].to_numpy())
            ps[str(lo)] = {
                "n": int(len(sub)),
                "mean_att_prev": round(float(sub["att_prev"].mean()), 1),
                "mean_att_cur": round(float(sub["att_cur"].mean()), 1),
                "corr_raw": round(r, 4),
                "reliability_prev": round(rel_p, 4), "reliability_cur": round(rel_c, 4),
                "corr_disattenuated": round(float(r_dis), 4),
                "ols_slope_cur_on_prev": round(b, 4), "slope_se": round(se, 4),
                "verdict": "ok" if len(sub) >= MIN_CELL else "UNDERPOWERED",
            }
        cls_out["prior_to_current"] = ps

        # ---- 4. slope of realised on as-of, by attempt-count bucket --------
        att = sl["shooter_att_c"].to_numpy(dtype="float64")
        # the model's own feature: as-of rate MINUS the league as-of rate
        x = sl["shooter_make_c"].to_numpy(dtype="float64")
        lg = sl["lg_make_asof"].to_numpy(dtype="float64")
        y = sl["y"].to_numpy(dtype="float64") - lg      # realised, same centring
        buckets = {}
        for lo, hi in ATT_BUCKETS:
            msk = (att >= lo) & (att <= hi)
            n = int(msk.sum())
            lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
            if n < 30:
                buckets[lab] = {"n": n, "verdict": "UNDERPOWERED"}
                continue
            b, a_, se = ols_slope(x[msk], y[msk])
            # Decision-8-style quintile read inside the bucket
            sub_x, sub_y = x[msk], y[msk] + lg[msk]
            q = np.full(n, -1)
            if np.ptp(sub_x) > 0:
                order = np.argsort(sub_x, kind="stable")
                q[order] = (np.arange(n) * 5) // n
            spans = []
            for k in range(5):
                mm = q == k
                spans.append(float(sub_y[mm].mean()) if mm.sum() else float("nan"))
            buckets[lab] = {
                "n": n,
                "share_of_class_pct": round(100.0 * n / len(sl), 3),
                "mean_asof_dev_pp": round(float(sub_x.mean()) * 100, 3),
                "sd_asof_dev_pp": round(float(sub_x.std()) * 100, 3),
                "realised_rate": round(float(sub_y.mean()), 5),
                "slope_realised_on_asof": round(b, 4), "slope_se": round(se, 4),
                "slope_t": round(b / se, 2) if np.isfinite(se) and se > 0 else None,
                "quintile_realised_rate": [round(v, 5) for v in spans],
                "quintile_span_pp": round((spans[4] - spans[0]) * 100, 3),
                "verdict": "ok" if n >= MIN_CELL else "UNDERPOWERED",
            }
        cls_out["slope_by_attempt_bucket"] = buckets

        # the same slope on ALL rows where the feature is defined, which is the
        # population Decision 8 reads
        defined = att > 0
        b, _, se = ols_slope(x[defined], y[defined])
        cls_out["slope_all_defined"] = {
            "n": int(defined.sum()), "slope": round(b, 4), "se": round(se, 4)}

        report["by_class"][cls_name] = cls_out
        mp = cls_out["mom_pooled"]
        print(f"{cls_name}: between-shooter SD {mp.get('between_sd_pp')} pp, "
              f"m={mp.get('m_shrinkage_attempts')}, implied quintile span "
              f"{mp.get('implied_quintile_span_pp')} pp, "
              f"slope(all defined) {cls_out['slope_all_defined']['slope']}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {a.out} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
