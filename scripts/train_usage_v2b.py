#!/usr/bin/env python
"""
train_usage_v2b.py -- L4 SHOT ALLOCATION (usage), ROUND 2b: the S1 training
scheme confirmation.

Round 2 (`scripts/train_usage_v2.py`) fixed the shooter label and re-decided the
arm per event class under the ROUND-1 scheme -- one static fit on the training
seasons -- because holding the scheme fixed is what isolates the label fix.
L21 then makes S1 the standing default scheme for every sub-model:

    S1  in-season walk-forward. Refit at each month boundary of the test season
        on all prior seasons PLUS the test season to date, strictly before the
        refit date. Each test event is scored by the most recent refit at or
        before its own game date, so no event is ever in its own fit and every
        test event is still scored -- which is what keeps S1's log loss
        comparable with the static arm's on the identical test set.

This trainer asks the one question L21 leaves open for this model: does the
round-2 winner get BETTER or WORSE under S1? It compares, per class, the
round-2 F1 winner (plus the runner-up when that runner-up sits inside the
round-2 floor) refit under S1 against the SAME arm fit statically, on the same
fold, with the same features, the same parameter grids and the same seeds.
Nothing else moves.

WHAT AN S1 REFIT REFITS. Everything the arm's fit chooses: the shrinkage prior
and strength (`usage.fit_shrinkage`), the ridge penalty's design, and the tree.
The LightGBM hyper-parameters are NOT re-searched per month -- they are carried
from the round-2 F1 search, which saw the training season only, because
re-searching them monthly would let the test season pick its own capacity.

The monthly artifacts are persisted so the engine can select the fit that was
current for a game's own month; naming is documented in
`docs/models/usage/experiments.md` section 9 and in `write_artifacts` below.

Usage:
    .venv/Scripts/python.exe scripts/train_usage_v2b.py
    .venv/Scripts/python.exe scripts/train_usage_v2b.py --classes FGA_3 --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import train_usage_v1 as V1  # noqa: E402
import train_usage_v2 as V2  # noqa: E402

from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import usage as U  # noqa: E402

SCHEMES: tuple[str, ...] = ("S0", "S1")
ROUND2_DIR = ROOT / "data/processed/models/usage_v2"
DEFAULT_OUT = ROOT / "data/processed/models/usage_s1"


# ---------------------------------------------------------------------------
# One arm, one scheme
# ---------------------------------------------------------------------------
def _fit_one(arm: str, tr: pd.DataFrame, te_seg: pd.DataFrame, cls: str,
             seed: int, lgbm_params: dict, args) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit `arm` on `tr`, predict `te_seg`. Returns (probs, driver, meta).

    The driver is the arm's own shrunk as-of rate under the shrinkage THIS fit
    chose, so the calibration and responsiveness axes move with the fit rather
    than being frozen at the static fit's choice."""
    shrink = U.fit_shrinkage(tr, cls)
    pk, m = shrink["best"]["prior"], shrink["best"]["m"]
    driver = U.shrunk_rate(te_seg, cls, pk, m)
    meta = {"prior_kind": pk, "shrink_m": m, "n_train": int(len(tr)),
            "train_log_loss": shrink["best"]["train_log_loss"]}

    if arm == "proportional":
        return U.u1_probs(te_seg, cls, pk, m), driver, meta

    if arm in U.DIRICHLET_ARMS:
        # At every fitted concentration in round 1 and round 2 these collapse
        # onto U1 ("no dispersion", experiments.md R4), and their MARGINAL
        # predictive is what the log loss scores. Re-fitting the concentration
        # inside a monthly loop would cost hours to reproduce U1's numbers, so
        # the scheme comparison runs them at U1's marginal and says so.
        return U.u1_probs(te_seg, cls, pk, m), driver, {**meta, "note":
                                                        "scored at the U1 marginal (fitted alphas are no-dispersion)"}

    if arm == "cond_logit":
        Xtr, Xte, names, unid, const = V1.cl_matrices(tr, te_seg, cls, pk, m)
        model = U.CondLogitArm(l2=args.l2).fit(Xtr, tr["y"].to_numpy())
        meta.update({"l2": args.l2, "n_features": len(names),
                     "dropped_unidentified": unid, "dropped_constant": const,
                     "coefficients": {n: round(float(b), 5)
                                      for n, b in zip(names, model.beta_, strict=False)},
                     "converged": bool(model.converged_)})
        return model.predict_proba(Xte), driver, meta

    if arm == U.TREE_ARM:
        Ltr, ctr, Lte, kept, dropped, itr, ite = V1.lgbm_matrices(tr, te_seg, cls, pk, m)
        model = U.LgbmChoiceArm(lgbm_params, seed=seed).fit(Ltr, ctr, init=itr)
        meta.update({"params": lgbm_params, "n_features": len(kept),
                     "dropped_unidentified": dropped})
        return model.predict_proba(Lte, init=ite), driver, {**meta, "_model": model,
                                                            "_features": kept}

    raise KeyError(f"unknown arm {arm!r}")


def fit_predict_scheme(arm: str, scheme: str, tr: pd.DataFrame, te: pd.DataFrame,
                       cls: str, seed: int, lgbm_params: dict, args,
                       keep_models: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """`(probs, driver, meta)` for one arm under one scheme on the whole test fold.

    S1's month partition is `possession_outcome.month_boundaries`, the same
    function the possession-outcome round-2 S1 arms use, so the two models'
    schedules cannot drift apart."""
    if scheme == "S0":
        p, driver, meta = _fit_one(arm, tr, te, cls, seed, lgbm_params, args)
        model = meta.pop("_model", None)
        feats = meta.pop("_features", None)
        out = {"scheme": "S0", "n_fits": 1, "segments": [
            {"refit_date": None, "n_scored": int(len(te)), **meta}]}
        if keep_models:
            out["_models"] = [{"refit_date": None, "model": model, "features": feats,
                               "prior_kind": meta["prior_kind"], "shrink_m": meta["shrink_m"]}]
        return p, driver, out

    if scheme != "S1":
        raise KeyError(f"unknown scheme {scheme!r}; known: {SCHEMES}")

    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    cuts = PO.month_boundaries(te_dates)
    p = np.zeros((len(te), U.N_ALT), dtype="float64")
    driver = np.zeros((len(te), U.N_ALT), dtype="float64")
    scored = np.zeros(len(te), dtype=bool)
    segments, models = [], []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None
               else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        # prior seasons in full, plus the test season STRICTLY BEFORE the cut
        fit_rows = tr if not before.any() else pd.concat(
            [tr, te.loc[before]], ignore_index=True)
        pp, dd, meta = _fit_one(arm, fit_rows, te.loc[seg], cls, seed, lgbm_params, args)
        model = meta.pop("_model", None)
        feats = meta.pop("_features", None)
        p[seg] = pp
        driver[seg] = dd
        scored |= seg
        max_train = (tr_dates.max() if not before.any()
                     else max(tr_dates.max(), te_dates[before].max()))
        segments.append({"refit_date": str(cut.date()), "n_scored": int(seg.sum()),
                         "n_train_from_test_season": int(before.sum()),
                         "max_train_date": str(pd.Timestamp(max_train).date()),
                         **meta})
        if keep_models:
            models.append({"refit_date": str(cut.date()), "model": model,
                           "features": feats, "prior_kind": meta["prior_kind"],
                           "shrink_m": meta["shrink_m"]})
    if not scored.all():
        raise AssertionError("S1 left test rows unscored; the month partition is not a cover")
    out = {"scheme": "S1", "n_fits": len(segments), "segments": segments,
           "earliest_train_date": str(tr_dates.min().date()) if len(tr_dates) else None}
    if keep_models:
        out["_models"] = models
    return p, driver, out


# ---------------------------------------------------------------------------
# Artifacts the engine can select by month
# ---------------------------------------------------------------------------
def write_artifacts(out_dir: Path, cls: str, arm: str, meta: dict) -> list[dict]:
    """Persist every monthly fit and return the manifest rows.

    NAMING (documented in `experiments.md` section 9 so the engine can rely on
    it): `usage_s1/{event_class}/{arm}_{YYYY-MM-DD}.{json|joblib}`, where the
    date is the REFIT date -- the first day of the month the fit becomes
    current. The engine selects, for a game, the artifact with the LATEST refit
    date at or before the game's own date; a game earlier than the first refit
    date uses the training-fold fit, which is the row with `refit_date = null`.
    Every row also carries the shrinkage the fit chose, because the U1 path the
    engine currently runs needs nothing else."""
    d = out_dir / cls
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg in meta.get("_models", []):
        stamp = seg["refit_date"] or "static"
        row = {"event_class": cls, "arm": arm, "refit_date": seg["refit_date"],
               "prior_kind": seg["prior_kind"], "shrink_m": seg["shrink_m"]}
        if seg["model"] is not None:
            import joblib
            path = d / f"{arm}_{stamp}.joblib"
            joblib.dump({"arm": arm, "event_class": cls,
                         "refit_date": seg["refit_date"], "model": seg["model"],
                         "features": seg["features"],
                         "prior_kind": seg["prior_kind"],
                         "shrink_m": seg["shrink_m"]}, path, compress=3)
            row["artifact"] = str(path.relative_to(ROOT)).replace("\\", "/")
        else:
            path = d / f"{arm}_{stamp}.json"
            path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            row["artifact"] = str(path.relative_to(ROOT)).replace("\\", "/")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
def score(te: pd.DataFrame, p: np.ndarray, driver: np.ndarray, boot_reps: int) -> dict:
    sc = U.score_arm(te, p, driver)
    sc["bootstrap_se"] = round(U.bootstrap_se(te, p, n_rep=boot_reps), 6)
    sc["team_segment"] = V2.team_segment(te, p)
    return sc


def which_arms(res_f1: dict) -> list[str]:
    """The round-2 F1 winner, plus the runner-up when it is inside the floor.

    "Inside the floor" is the same test the round-2 decision rule used to
    declare a tie, so this is not a new threshold -- it is the same one, read
    off the same numbers."""
    dec = res_f1["decision"]
    win = dec.get("winner")
    if win is None:
        return []
    floor = float(dec.get("floor", 0.0))
    lls = {a: res_f1["arms"][a]["log_loss"] for a in dec.get("eligible", [])}
    base = lls[win]
    close = [a for a, v in sorted(lls.items(), key=lambda kv: kv[1])
             if a != win and v - base <= floor]
    return [win, *close]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--round2", default=str(ROUND2_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--classes", nargs="*", default=list(U.EVENT_CLASSES))
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--l2", type=float, default=None,
                    help="ridge penalty for cond_logit; default = the round-2 F1 value")
    ap.add_argument("--seeds", type=int, default=3,
                    help="spec-identical retrains for the tree's noise floor; each "
                         "replays the whole monthly schedule")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.boot_reps, args.seeds = 50, 2

    r2_dir = Path(args.round2)
    out_dir = Path(args.out)
    # a relative --out is relative to the REPO, not to the shell's cwd, so the
    # manifest's repo-relative paths resolve wherever the trainer is launched
    if not r2_dir.is_absolute():
        r2_dir = ROOT / r2_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads((r2_dir / "results_v2.json").read_text())
    by_key = {(r["fold"], r["event_class"]): r for r in results}
    meta2 = json.loads((r2_dir / f"build_report_{args.version}_shotshooter.json").read_text())
    events = pd.read_parquet(r2_dir / f"events_{args.version}_shotshooter.parquet")
    asof = pd.read_parquet(r2_dir / f"asof_{args.version}_shotshooter.parquet")

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    manifest: list[dict] = []
    out: list[dict] = []
    for cls in args.classes:
        r2 = by_key.get(("F1", cls))
        if r2 is None:
            log(f"{cls}: no round-2 F1 result; skipped")
            continue
        arms = which_arms(r2)
        if not arms:
            log(f"{cls}: round 2 adopted nothing; nothing to confirm")
            continue
        lgbm_params = r2["arms"][U.TREE_ARM]["params"]
        l2 = args.l2 if args.l2 is not None else r2["arms"]["cond_logit"]["l2"]
        args.l2 = l2
        design = U.build_usage_design(events, asof, cls)
        tr, te = U.fold_slices(design)
        log(f"[{cls}] arms {arms}; train {len(tr):,} test {len(te):,}; "
            f"round-2 floor {r2['decision'].get('floor'):.6f}")
        row = {"event_class": cls, "fold": U.SELECTION_FOLD, "arms": {},
               "round2_winner": r2["decision"]["winner"],
               "round2_floor": r2["decision"].get("floor"),
               "n_train": int(len(tr)), "n_test": int(len(te))}
        for arm in arms:
            row["arms"][arm] = {}
            for scheme in SCHEMES:
                p, driver, m = fit_predict_scheme(
                    arm, scheme, tr, te, cls, args.seed, lgbm_params, args,
                    keep_models=(scheme == "S1"))
                sc = score(te, p, driver, args.boot_reps)
                if scheme == "S1":
                    manifest.extend(write_artifacts(out_dir, cls, arm, m))
                m.pop("_models", None)
                sc["scheme_meta"] = m
                if arm == U.TREE_ARM:
                    seed_lls = []
                    for sd in range(args.seeds):
                        pp, dd, _ = fit_predict_scheme(
                            arm, scheme, tr, te, cls, args.seed + sd, lgbm_params, args)
                        seed_lls.append(PM.log_loss(te["y"].to_numpy(), pp))
                    sc["seed_log_losses"] = [round(x, 6) for x in seed_lls]
                    sc["seed_sd"] = (round(float(np.std(seed_lls, ddof=1)), 6)
                                     if len(seed_lls) > 1 else None)
                row["arms"][arm][scheme] = sc
                log(f"  {arm:<14} {scheme}: ll {sc['log_loss']:.6f} "
                    f"calib {sc['calib_worst_gap_pp']:.3f}pp "
                    f"slope {sc['resp_slope_ratio']} steps {sc['resp_steps']}/4 "
                    f"boot {sc['bootstrap_se']:.6f} fits {m['n_fits']}")
            row["arms"][arm]["decision"] = decide_scheme(row["arms"][arm])
            log(f"  -> {arm}: {row['arms'][arm]['decision']['scheme']} "
                f"-- {row['arms'][arm]['decision']['reason']}")
        out.append(row)
        del design
        (out_dir / "results_v2b.json").write_text(
            json.dumps(out, indent=2, default=float), encoding="utf-8")
        (out_dir / "s1_manifest.json").write_text(
            json.dumps({"naming": ("usage_s1/{event_class}/{arm}_{refit_date|static}."
                                   "{json|joblib}; the engine takes the LATEST refit_date "
                                   "at or before a game's own date"),
                        "shooter_key": meta2["shooter_key"],
                        "possessions_version": meta2["possessions_version"],
                        "rows": manifest}, indent=2), encoding="utf-8")
        (out_dir / "train_log_v2b.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nwrote {out_dir}/results_v2b.json and s1_manifest.json "
        f"({time.time() - t0:.1f}s)")
    return 0


def decide_scheme(arm_block: dict) -> dict:
    """The pre-registered rule: adopt S1 unless it regresses calibration or log
    loss BEYOND THE FLOOR.

    The floor is the larger of the two schemes' own block-bootstrap SEs, so a
    move inside the noise is not a regression in either direction. L21 predicts
    a calibration gain and no log-loss gain, so a log loss that is merely flat
    is NOT a reason to reject."""
    s0, s1 = arm_block["S0"], arm_block["S1"]
    floor = max(s0["bootstrap_se"], s1["bootstrap_se"])
    d_ll = s1["log_loss"] - s0["log_loss"]
    d_cal = s1["calib_worst_gap_pp"] - s0["calib_worst_gap_pp"]
    ll_regress = d_ll > floor
    cal_regress = (not s1["calib_pass"]) and s0["calib_pass"]
    resp_regress = (not s1["resp_pass"]) and s0["resp_pass"]
    adopt = not (ll_regress or cal_regress or resp_regress)
    bits = [f"log loss {d_ll:+.6f} ({d_ll / floor:+.2f} floors, floor {floor:.6f})",
            f"worst calibration gap {d_cal:+.3f} pp "
            f"({s0['calib_worst_gap_pp']:.3f} -> {s1['calib_worst_gap_pp']:.3f})",
            f"slope ratio {s0['resp_slope_ratio']} -> {s1['resp_slope_ratio']}"]
    reason = "; ".join(bits)
    if ll_regress:
        reason += "; S1 REGRESSES log loss beyond the floor"
    if cal_regress:
        reason += "; S1 FAILS the calibration gate where the static fit passes"
    if resp_regress:
        reason += "; S1 FAILS the responsiveness gate where the static fit passes"
    return {"scheme": "S1" if adopt else "S0", "adopt_s1": bool(adopt),
            "floor": floor, "delta_log_loss": round(d_ll, 6),
            "delta_log_loss_floors": round(d_ll / floor, 3),
            "delta_calib_pp": round(d_cal, 4), "reason": reason}


if __name__ == "__main__":
    raise SystemExit(main())
