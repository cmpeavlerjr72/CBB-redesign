"""grade_fg_make_pae_v1.py -- the ONE grading script for the fg_make PAE round (experiments.md section 31).

Every arm is scored by the same code, blind: the arms found on disk are shuffled under a fixed RNG into codes
A1..An, every metric is computed per code, and the code -> label map is applied only when the final tables are
written. The test rows are reconstructed from the design exactly as the trainer slices them (`FG.fold_slices`,
`FG.class_slice`) and every arm's prediction file is asserted row-aligned to them (game_id, shooter_id, y).

Metrics per (fold, class, arm):
  * log loss (primary), Brier; `FG.score` battery: decile calibration (worst gap, level, shape), Decision 8 on the
    two served drivers (shooter_make_c, def_allow_c);
  * PAE responsiveness: predicted vs realised make rate by quintile of the shooting team's pae_off_c (rows with
    pae_off_n > 0) and of the defending team's pae_def_c (pae_def_n > 0), slope ratio + steps, Decision-8 verdict
    (low_span_exempt) plus the "slope positive" clause;
  * paired game-block bootstrap SE (200 reps) of LL(arm) - LL(R), overall and per segment (site, month,
    located / unlocated);
  * P2 drift check: log loss gap located vs unlocated rows for R and P2, and the P2 - R gain on each.
Decision per class by the registered rule (section 31.6), read on F2; F1 reported.

    .venv/Scripts/python.exe scripts/grade_fg_make_pae_v1.py --runs data/processed/models/fg_make/round_pae \
        --out results/fg_make_pae_v1/grade_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
PAE_COLS = ROOT / "data/processed/models/fg_make/pae_v1/design_pae_cols_v1.parquet"
BUILD_REPORT = ROOT / "data/processed/models/fg_make/pae_v1/build_report_v1.json"
ARMS = {"R": ("R", 0), "R_seed": ("R", 1), "P1": ("P1", 0), "P2": ("P2", 0)}
N_BOOT, BOOT_SEED, POWERED = 200, 12345, 5000
CALIB_MIN_FLOOR_PP = 0.10
MONTHS = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar+Apr", 4: "Mar+Apr"}
DCOLS = ["game_id", "season", "game_date", "shot_class", "y", "shooter_id", "offense_is_home", "neutral_site",
         "chance_number", "shooter_make_c", "shooter_att_c", "def_allow_c", "def_att_prior", "made"]


def ll_rows(y, p):
    return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1.0))


def paired_boot(games, d, n_rep=N_BOOT, seed=BOOT_SEED):
    """Game-block bootstrap SE of mean(d) (d = per-row log-loss difference)."""
    g = pd.DataFrame({"g": games, "d": d}).groupby("g")["d"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    reps = np.empty(n_rep)
    for r in range(n_rep):
        k = rng.integers(0, len(s), len(s))
        reps[r] = s[k].sum() / n[k].sum()
    return float(reps.std(ddof=1))


def pae_resp(te, p):
    out = {}
    for feat, gate in (("pae_off_c", "pae_off_n"), ("pae_def_c", "pae_def_n")):
        m = te[gate].to_numpy() > 0
        r = PM.quintile_responsiveness(FG._rank(te.loc[m, feat].to_numpy()), te["y"].to_numpy()[m],
                                       np.column_stack([1 - p[m], p[m]]), 1, n_q=5)
        r["n_defined"] = int(m.sum())
        d8 = FG.decision8_verdict({feat: r}, reading="low_span_exempt")["by_driver"][feat]
        r["d8"] = d8
        r["slope_positive"] = bool(r["slope_ratio"] is not None and r["slope_ratio"] > 0)
        r["pass"] = bool(d8["pass"] and r["slope_positive"])
        out[feat] = r
    return out


def load_test(fold):
    des = pd.read_parquet(DESIGN, columns=DCOLS)
    cols = pd.read_parquet(PAE_COLS)
    for k in ("game_id", "shooter_id", "shot_class", "made"):
        assert (cols[k].to_numpy() == des[k].to_numpy()).all(), k
    for c in ("pae_off_c", "pae_def_c", "pae_off_n", "pae_def_n", "cell_p", "located"):
        des[c] = cols[c].to_numpy()
    des["site"] = np.where(des["neutral_site"].astype(bool), "neutral",
                           np.where(des["offense_is_home"].astype(bool), "home", "away"))
    des["month"] = pd.to_datetime(des["game_date"]).dt.month.map(MONTHS)
    _, te = FG.fold_slices(des, fold)
    return {c: FG.class_slice(te, c) for c in FG.SHOT_CLASSES}


def grade_fold(runs: Path, fold: str, rng) -> dict:
    tes = load_test(fold)
    found = {}
    for lab, (arm, seed) in ARMS.items():
        f = runs / f"{arm}_s{seed}_{fold}" / f"preds_{fold}_s{seed}.parquet"
        if f.exists():
            found[lab] = f
    labels = list(found)
    codes = [f"A{i + 1}" for i in range(len(labels))]
    perm = rng.permutation(len(labels))
    code_of = {labels[i]: codes[perm[i]] for i in range(len(labels))}
    preds = {}
    for lab, f in found.items():
        pr = pd.read_parquet(f)
        preds[code_of[lab]] = pr
    res = {"codes": {}, "fold": fold}
    for c in FG.SHOT_CLASSES:
        te = tes[c]
        y = te["y"].to_numpy()
        per = {}
        for code, pr in preds.items():
            pc = pr[pr["shot_class"] == c]
            assert len(pc) == len(te), (code, c, len(pc), len(te))
            for k in ("game_id", "shooter_id", "y"):
                assert (pc[k].to_numpy() == te[k].to_numpy()).all(), (code, c, k)
            p = pc["p"].to_numpy(dtype="float64")
            sc = FG.score(te, np.column_stack([1 - p, p]))
            cal = sc["calibration"]["MAKE"]
            per[code] = {"p": p, "ll_rows": ll_rows(y, p), "log_loss": sc["log_loss"], "brier": sc["brier"],
                         "pred_make": sc["pred_make_rate"], "actual_make": sc["actual_make_rate"],
                         "calib_pass": sc["calib_pass"], "calib_worst_gap_pp": sc["calib_worst_gap_pp"],
                         "calib_level_pp": cal["level_shift_pp"],
                         "calib_shape_pp": cal["max_abs_gap_pp_after_level_shift"],
                         "d8_pass": sc["resp_pass_decision8"],
                         "d8_failed": sc["resp_failed_drivers_decision8"],
                         "d8_slopes": {k: v["slope_ratio"] for k, v in sc["responsiveness"].items()},
                         "pae_resp": pae_resp(te, p)}
        res["codes"][c] = per
        res.setdefault("_te", {})[c] = te
    res["code_of"] = code_of
    return res


def compare(res: dict, ref_code: str, arm_code: str, c: str) -> dict:
    te = res["_te"][c]
    a, r = res["codes"][c][arm_code], res["codes"][c][ref_code]
    d = a["ll_rows"] - r["ll_rows"]
    g = te["game_id"].to_numpy()
    out = {"gain": float(-(d.mean())), "boot_se": paired_boot(g, d), "segments": {}}
    for seg in ("site", "month", "located"):
        for v in sorted(te[seg].astype(str).unique()):
            m = (te[seg].astype(str) == v).to_numpy()
            out["segments"][f"{seg}={v}"] = {"n": int(m.sum()), "gain": float(-(d[m].mean())),
                                             "boot_se": paired_boot(g[m], d[m]),
                                             "powered": bool(m.sum() >= POWERED)}
    loc = te["located"].to_numpy().astype(bool)
    out["ll_located_vs_unlocated"] = {"arm": [float(a["ll_rows"][loc].mean()), float(a["ll_rows"][~loc].mean()) if (~loc).any() else None],
                                      "ref": [float(r["ll_rows"][loc].mean()), float(r["ll_rows"][~loc].mean()) if (~loc).any() else None]}
    return out


def decide(F: dict) -> dict:
    """Section 31.6, per class, on F2."""
    co = F["code_of"]
    R, RS = co["R"], co["R_seed"]
    lk = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))["leak_test"]
    leak_ok = True
    leak_detail = {}
    for col in ("pae_off_c", "pae_def_c"):
        rows = lk[col]
        bad = [r for r in rows if r["verdict"] == "LEAK"]
        leak_detail[col] = {str(r["season"]): (round(r["corr_asjoined"], 4), r["verdict"]) for r in rows}
        leak_ok &= not bad
    out = {"leak_ok": leak_ok, "leak": leak_detail, "by_class": {}}
    for c in FG.SHOT_CLASSES:
        per = F["codes"][c]
        seed_diff = abs(per[R]["log_loss"] - per[RS]["log_loss"])
        gap_floor = max(abs(per[R]["calib_worst_gap_pp"] - per[RS]["calib_worst_gap_pp"]), CALIB_MIN_FLOOR_PP)
        cls = {"seed_diff": seed_diff, "calib_gap_floor_pp": gap_floor, "arms": {}}
        for lab in ("P1", "P2"):
            if lab not in co:
                continue
            A = co[lab]
            cmp_ = compare(F, R, A, c)
            floor = max(seed_diff, cmp_["boot_se"])
            seg_regress = [k for k, s in cmp_["segments"].items()
                           if s["powered"] and -s["gain"] > 2 * s["boot_se"]]
            a = per[A]
            checks = {
                "leak": leak_ok,
                "gain_gt_floor": cmp_["gain"] > floor,
                "calib_gate": a["calib_pass"],
                "calib_not_worse": a["calib_worst_gap_pp"] <= per[R]["calib_worst_gap_pp"] + gap_floor,
                "pae_off_resp": a["pae_resp"]["pae_off_c"]["pass"],
                "pae_def_resp": a["pae_resp"]["pae_def_c"]["pass"],
                "no_segment_regression": not seg_regress,
                "d8_not_flipped": a["d8_pass"] or not per[R]["d8_pass"],
            }
            if lab == "P2":
                un = cmp_["segments"].get("located=False")
                lo = cmp_["segments"].get("located=True")
                checks["p1_candidate"] = cls["arms"].get("P1", {}).get("verdict") == "CANDIDATE (offline)"
                checks["unlocated_not_negative_beyond_floor"] = (un is None) or (un["gain"] >= -floor)
                checks["located_gain_gt_floor"] = (lo is not None) and lo["gain"] > floor
            verdict = ("REFUSED (leak)" if not leak_ok else
                       "CANDIDATE (offline)" if all(checks.values()) else "NOT CANDIDATE")
            cls["arms"][lab] = {"gain": cmp_["gain"], "boot_se": cmp_["boot_se"], "floor": floor,
                                "gain_in_floors": cmp_["gain"] / floor if floor > 0 else None,
                                "checks": checks, "segment_regressions": seg_regress,
                                "segments": cmp_["segments"], "ll_loc_unloc": cmp_["ll_located_vs_unlocated"],
                                "verdict": verdict}
        if "P1" in cls["arms"] and "P2" in cls["arms"]:
            p1, p2 = cls["arms"]["P1"], cls["arms"]["P2"]
            if p2["verdict"].startswith("CANDIDATE") and p1["verdict"].startswith("CANDIDATE"):
                cmp12 = compare(F, co["P1"], co["P2"], c)
                fl = max(seed_diff, cmp12["boot_se"])
                cls["p2_vs_p1"] = {"gain": cmp12["gain"], "floor": fl,
                                   "preferred": "P2" if cmp12["gain"] > fl else "P1 (tie goes to the simpler arm)"}
        out["by_class"][c] = cls
    return out


def strip(F):
    out = {"fold": F["fold"], "code_of": F["code_of"], "by_class": {}}
    lab_of = {v: k for k, v in F["code_of"].items()}
    for c, per in F["codes"].items():
        out["by_class"][c] = {lab_of[code]: {k: v for k, v in d.items() if k not in ("p", "ll_rows")}
                              for code, d in per.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "data/processed/models/fg_make/round_pae")
    ap.add_argument("--out", type=Path, default=ROOT / "results/fg_make_pae_v1/grade_v1.json")
    a = ap.parse_args()
    rng = np.random.default_rng(20261025)
    report = {"created_at": pd.Timestamp.now("UTC").isoformat(), "folds": {}}
    for fold in ("F1", "F2"):
        F = grade_fold(a.runs, fold, rng)
        if "R" not in F["code_of"] or "R_seed" not in F["code_of"]:
            print(f"{fold}: R / R_seed missing; skipped")
            continue
        dec = decide(F)
        # reveal labels only here
        report["folds"][fold] = {"metrics": strip(F), "decision": dec}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for fold, fr in report["folds"].items():
        print(f"== {fold} (decision rule is read on F2 only)")
        for c, cls in fr["decision"]["by_class"].items():
            m = fr["metrics"]["by_class"][c]
            print(f"  {c}: R ll {m['R']['log_loss']:.6f}  R_seed diff {cls['seed_diff']:.2e}")
            for lab, d in cls["arms"].items():
                failed = [k for k, v in d["checks"].items() if not v]
                print(f"    {lab}: gain {d['gain']:+.2e} floor {d['floor']:.2e} ({d['gain_in_floors']:+.2f} fl) "
                      f"-> {d['verdict']}  failed={failed}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
