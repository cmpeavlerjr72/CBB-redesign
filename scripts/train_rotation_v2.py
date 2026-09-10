#!/usr/bin/env python
"""
train_rotation_v2.py -- ROUND 2 of the pre-registered L4 rotation bake-off
(`docs/models/rotation/experiments.md` section 3), F1: train 2024, test 2025.

Round 1 (`train_rotation_v1.py`) found no eligible arm. Round 2 keeps R2 as the
incumbent reference and adds two arms that separate "who plays together" from
"when the coach deviates":

  R5 `hybrid`             R4's own-recent-games stint-sequence resampling (k
                          fitted on 2024 from {3, 5, 8, all-season}) under two
                          fitted, state-conditioned override hazards: a
                          foul-trouble exit/re-entry pair on
                          (fouls x period x seconds remaining x is_starter) and
                          a margin pair on (|margin| x seconds remaining x
                          is_starter) plus the close-and-late re-entry term.
                          Every coefficient is fitted; no hand-set thresholds.
  R6 `stint_hazard_v2`    R3 respecified: fouls, |margin| and seconds remaining
                          entered as explicit interactions with `is_starter` and
                          `period`, because round 1's flat entry could not
                          represent foul trouble at all (a per-possession
                          constant cancels in the entry softmax).

Everything else -- data, folds, metrics, tolerances, noise floor, decision rules
-- is unchanged and is imported from `train_rotation_v1`, so the two rounds are
graded by literally the same code. The test set is the full set of 2025 games
with complete on-floor data, or a random subset of at least 1,500 games whose
seed is recorded in the results JSON.

Usage:
    .venv/Scripts/python.exe scripts/train_rotation_v2.py --append-experiments
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.data.player_ids import load_crosswalk  # noqa: E402
from cbb_sim.models import rotation as R  # noqa: E402

import train_rotation_v1 as V1  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
EXPERIMENTS = ROOT / "docs" / "models" / "rotation" / "experiments.md"
TRAIN_SEASON, TEST_SEASON = 2024, 2025
TEST_SUBSET_SEED = 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_rotation_v2")

DONOR_K_GRID = (3, 5, 8, 0)      # 0 == every earlier game of the season


# ---------------------------------------------------------------------------
# Round-2 fits
# ---------------------------------------------------------------------------
def fit_donor_k(tp_full, tp_sub, feats_sub, fit, targets, keys, seed=7):
    """Pick R4/R5's donor depth by how well the resampled sequence reproduces the
    training season's lineup concentration (top-1 and top-3 five-man lineup share
    and distinct lineups per team-game).

    The donor bank is built from the WHOLE training season and only the scored
    team-games are subsampled: building it from a subsample leaves most teams
    with no earlier game in it at all, R4 falls back to "the as-of top five, all
    game", and the fit sees a substitution rate of 0.02 instead of 0.17."""
    scripts = R.build_scripts(tp_sub)
    priors = R.build_priors(feats_sub, fit)
    use = [k for k in keys if k in priors and k in scripts]
    scores, best, best_err = {}, DONOR_K_GRID[0], np.inf
    for k_don in DONOR_K_GRID:
        donors = R.build_donor_bank(tp_full, k_donors=k_don)
        arm = R.R4StintResample(fit, donors)
        t1, t3, nlu = [], [], []
        for key in use:
            lu, _ = arm.simulate(priors[key], scripts[key], R.game_stream(seed, key[0]))
            ks = [tuple(sorted(int(x) for x in r)) for r in lu]
            c: dict = {}
            for x in ks:
                c[x] = c.get(x, 0) + 1
            v = sorted(c.values(), reverse=True)
            n = float(sum(v))
            t1.append(v[0] / n)
            t3.append(sum(v[:3]) / n)
            nlu.append(len(v))
        got = {"lu_top1": float(np.mean(t1)), "lu_top3": float(np.mean(t3)),
               "n_lineups": float(np.mean(nlu))}
        err = sum(((got[m] - targets[m]) / targets[m]) ** 2 for m in got)
        scores[str(k_don)] = {**got, "err": err}
        if err < best_err:
            best, best_err = k_don, err
    return int(best), scores


STATE_CELLS_FIT = ["late_starter_share_b0", "late_starter_share_b1",
                   "late_starter_share_b2", "foul_trouble_share"]


def train_state_targets(train: dict, keys, rotation_sets, asof_starters):
    """The training season's own state-dependence cells, computed through the
    grader's code path so the fit and the gate measure the same thing."""
    recs, _, _ = V1.actual_records(train, keys, rotation_sets, asof_starters)
    return R.aggregate_stats(recs)


def fit_override_scale(fit, donors, feats, scripts, keys, target_rate,
                       state_targets, rotation_sets, seed=7,
                       grid=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                       bases=(0.005, 0.02, 0.06)):
    """The single fitted knob on R5's overrides, fitted on the TRAINING season
    against the four state-dependence cells the overrides exist to produce.

    Fitting it against the substitution rate instead -- the first thing tried --
    selects the overrides straight out: the donor sequence already substitutes
    slightly more often than the real one (0.176 vs 0.152 on train), so every
    positive scale moves that number the wrong way and the grid returns 0.00,
    i.e. plain R4 with a different donor depth. The substitution rate is the
    donor's job; the override's job is state dependence, so that is what it is
    fitted against. The rate at the chosen scale is reported alongside."""
    priors = R.build_priors(feats, fit)
    use = [k for k in keys if k in priors and k in scripts]
    scores, best, best_err = {}, (grid[0], bases[0]), np.inf
    for p0 in bases:
        for sc in grid:
            f = R.RotationFit(**{**fit.__dict__,
                                 "notes": {**fit.notes, "r5_override_scale": float(sc),
                                           "r5_block_base": float(p0)}})
            arm = R.R5Hybrid(f, donors)
            recs, chg, tot = [], 0, 0
            for key in use:
                prior, script = priors[key], scripts[key]
                lu, fh = arm.simulate(prior, script, R.game_stream(seed, key[0]))
                starters = set(int(x) for x in prior.pids[prior.starters()[:5]])
                recs.append(R.team_game_stats(lu, script.dur, script.time_bucket,
                                              script.margin_bucket, starters, fh,
                                              prior.pids, rotation_sets.get(key, set())))
                ks = [tuple(sorted(int(x) for x in r)) for r in lu]
                for i in range(1, len(ks)):
                    tot += 1
                    chg += ks[i] != ks[i - 1]
            got = R.aggregate_stats(recs)
            err = 0.0
            for c in STATE_CELLS_FIT:
                a, b = got.get(c, np.nan), state_targets.get(c, np.nan)
                if np.isfinite(a) and np.isfinite(b):
                    err += (a - b) ** 2
            scores[f"scale={sc},p0={p0}"] = {
                **{c: float(got.get(c, np.nan)) for c in STATE_CELLS_FIT},
                "change_rate": chg / max(tot, 1), "err": float(err)}
            if err < best_err:
                best, best_err = (sc, p0), err
    return best, scores


def fit_round2(train: dict, fit: R.RotationFit, args) -> R.RotationFit:
    tp, feats = train["tp"], train["feats"]
    notes = dict(fit.notes)
    tr_games = sorted(tp["game_id"].unique())
    rs = np.random.RandomState(11)
    hz_games = rs.choice(tr_games, size=min(args.hazard_games, len(tr_games)), replace=False)

    # --- R6: respecified hazards -------------------------------------------
    Xe, ye, Xn, yn = R.build_hazard_training(tp, feats, fit, list(hz_games),
                                             fouls=train["fouls"], design=R._r6_design)
    ex, en = R.fit_hazard_models(Xe, ye, Xn, yn)
    notes["r6_exit"], notes["r6_enter"] = ex, en
    notes["r6_features"] = R.R6_FEATURES
    log.info("R6 hazards: exit %d rows (base %.4f), enter %d rows (base %.4f)",
             ex["n"], ex["base_rate"], en["n"], en["base_rate"])

    # --- R5: override hazards ----------------------------------------------
    Xe, ye, Xn, yn = R.build_hazard_training(tp, feats, fit, list(hz_games),
                                             fouls=train["fouls"], design=R._override_design)
    oex, oen = R.fit_hazard_models(Xe, ye, Xn, yn)
    notes["r5_exit"], notes["r5_enter"] = oex, oen
    notes["r5_features"] = R.OVERRIDE_FEATURES
    log.info("R5 override hazards: exit base %.4f, enter base %.4f",
             oex["base_rate"], oen["base_rate"])

    fit.notes = notes
    targets = notes["scheduler_targets_train"]
    sched_ids = list(hz_games)[: args.donor_games]
    tp_s = tp[tp["game_id"].isin(sched_ids)]
    feats_s = feats[feats["game_id"].isin(sched_ids)]
    scripts_s = R.build_scripts(tp_s)
    keys = list(scripts_s)[: args.donor_keys]

    k_don, k_scores = fit_donor_k(tp, tp_s, feats_s, fit, targets, keys)
    notes["r5_donor_k"] = k_don
    notes["r5_donor_k_grid"] = k_scores
    log.info("donor depth k fitted: %s (0 = all earlier games) -- grid %s",
             k_don, {kk: round(vv["err"], 5) for kk, vv in k_scores.items()})

    donors_tr = R.build_donor_bank(tp, k_donors=k_don)
    priors_s = R.build_priors(feats_s, fit)
    rot_s = {(int(a), int(b)): set(int(x) for x in g.loc[g["mpg_asof_raw"] >= 10.0, "pid"])
             for (a, b), g in feats_s.groupby(["game_id", "team_id"], sort=False)}
    asof_s = {k: set(int(x) for x in v.pids[v.starters()[:5]]) for k, v in priors_s.items()}
    use_keys = [k for k in keys if k in priors_s]
    st_targets = train_state_targets(train, use_keys, rot_s, asof_s)
    notes["r5_state_targets_train"] = {c: float(st_targets[c]) for c in STATE_CELLS_FIT}
    (scale, p0), sc_scores = fit_override_scale(fit, donors_tr, feats_s, scripts_s,
                                                use_keys, targets["change_rate"],
                                                st_targets, rot_s)
    notes["r5_override_scale"] = scale
    notes["r5_block_base"] = p0
    notes["r5_override_scale_grid"] = sc_scores
    key = f"scale={scale},p0={p0}"
    log.info("R5 override fitted: scale=%.2f p0=%.3f -> train state cells %s "
             "(targets %s), change rate %.4f vs train %.4f", scale, p0,
             {c[-2:]: round(sc_scores[key][c], 4) for c in STATE_CELLS_FIT},
             {c[-2:]: round(st_targets[c], 4) for c in STATE_CELLS_FIT},
             sc_scores[key]["change_rate"], targets["change_rate"])
    fit.notes = notes
    return fit


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-games", type=int, default=1600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise-seeds", type=int, default=50)
    ap.add_argument("--noise-games", type=int, default=150)
    ap.add_argument("--tilt-games", type=int, default=1500)
    ap.add_argument("--hazard-games", type=int, default=400)
    ap.add_argument("--theta-keys", type=int, default=150)
    ap.add_argument("--theta-games", type=int, default=250)
    ap.add_argument("--donor-games", type=int, default=250)
    ap.add_argument("--donor-keys", type=int, default=150)
    ap.add_argument("--min-prior-games", type=int, default=3)
    ap.add_argument("--append-experiments", action="store_true")
    ap.add_argument("--tag", type=str, default="F1-round2")
    args = ap.parse_args()

    t0 = time.time()
    assert_not_sealed([TRAIN_SEASON, TEST_SEASON], context="rotation bake-off F1 round 2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = V1.load_season(TRAIN_SEASON)
    test = V1.load_season(TEST_SEASON)

    cw = load_crosswalk()
    av = R.attach_hoopr_availability(test["feats"], TEST_SEASON, cw, pg=test["pg"])
    ok = av["hoopr_join_ok"]
    cw_cov = {
        "asof_rows": int(len(av)),
        "crosswalk_hit": float(av["espn_id"].notna().mean()),
        "hoopr_box_join": float(ok.mean()),
        "dnp_agreement": float((av.loc[ok, "did_not_play"].astype(bool)
                                == (av.loc[ok, "onfloor_minutes"] <= 0)).mean()),
    }

    fit, notes = V1.fit_all(train, args)
    fit = fit_round2(train, fit, args)
    fit.to_json(OUT_DIR / "rotation_fit_round2.json")

    priors_te = R.build_priors(test["feats"], fit, min_prior_games=args.min_prior_games)
    scripts_te = R.build_scripts(test["tp"])
    both: dict[int, list] = {}
    for (gid, tid) in scripts_te:
        if (gid, tid) in priors_te:
            both.setdefault(gid, []).append((gid, tid))
    eligible = sorted([g for g, v in both.items() if len(v) == 2])
    if len(eligible) > args.test_games:
        rs = np.random.RandomState(TEST_SUBSET_SEED)
        sel = sorted(rs.choice(eligible, size=args.test_games, replace=False))
        subset_note = (f"random subset of {args.test_games} of {len(eligible)} eligible "
                       f"games, numpy RandomState seed {TEST_SUBSET_SEED}")
    else:
        sel = eligible
        subset_note = f"all {len(eligible)} eligible games"
    log.info("test universe: %s", subset_note)

    keys_by_game = {}
    for g in sel:
        home = [k for k in both[g] if scripts_te[k].is_home]
        away = [k for k in both[g] if not scripts_te[k].is_home]
        keys_by_game[g] = home + away
    keys = [k for g in sel for k in keys_by_game[g]]

    rotation_sets = {}
    fs = test["feats"]
    kset = set(keys)
    fs = fs[[(int(g), int(t)) in kset for g, t in zip(fs["game_id"], fs["team_id"])]]
    for key, g in fs.groupby(["game_id", "team_id"], sort=False):
        rotation_sets[(int(key[0]), int(key[1]))] = set(
            int(p) for p in g.loc[g["mpg_asof_raw"] >= 10.0, "pid"])

    asof_starters = {k: set(int(x) for x in v.pids[v.starters()[:5]])
                     for k, v in priors_te.items()}
    log.info("computing actual metrics on %d team-games", len(keys))
    a_recs, a_rot, (a_diag, a_overlap) = V1.actual_records(test, keys, rotation_sets,
                                                           asof_starters)
    a_row = V1.build_row("ACTUAL", a_recs, a_rot, {}, np.array([]))
    act_minutes = V1.pooled_and_within_sd(a_rot)[3]
    a_row["ks_minutes_d"], a_row["ks_minutes_p"] = np.nan, np.nan
    diag_row = R.aggregate_stats(a_diag) if a_diag else {}
    diag_row["starter_overlap_of5"] = float(np.mean(a_overlap)) if a_overlap else float("nan")

    donors_full = R.build_donor_bank(test["tp"], k_donors=fit.notes["r5_donor_k"])
    arms = {
        "R2_hier_dirichlet": R.R2HierDirichlet(fit),
        "R5_hybrid": R.R5Hybrid(fit, donors_full),
        "R6_stint_hazard_v2": R.R6StintHazard(fit),
    }
    seeds = list(range(args.seeds))
    rows, verdicts, lineup_top1 = {}, {}, {}
    for name, arm in arms.items():
        ts = time.time()
        recs, rot, lus = V1.sim_records(arm, priors_te, scripts_te, keys_by_game,
                                        rotation_sets, seeds)
        row = V1.build_row(name, recs, rot, a_row, act_minutes)
        row["change_rate"] = R.sim_change_rate(lus)
        rows[name] = row
        verdicts[name] = V1.verdict(row, a_row)
        lineup_top1[name] = np.array([r["lu_top1"] for r in recs])
        log.info("%-20s %.1fs  nonzero %.2f top5 %.4f top8 %.4f lu1 %.4f min %.2f "
                 "sd %.2f/%.2f late b0/b1/b2 %.4f/%.4f/%.4f ft %.4f",
                 name, time.time() - ts, row["n_nonzero_mean"], row["top5_share"],
                 row["top8_share"], row["lu_top1"], row["minutes_mean"],
                 row["minutes_sd_pooled"], row["minutes_sd_within"],
                 row["late_starter_share_b0"], row["late_starter_share_b1"],
                 row["late_starter_share_b2"], row["foul_trouble_share"])

    a_top1 = np.array([r["lu_top1"] for r in a_recs])
    for name in rows:
        d, p = R.ks_2samp(lineup_top1[name], a_top1)
        rows[name]["ks_lineup_top1_d"] = d
        rows[name]["ks_lineup_top1_p"] = p

    log.info("noise floor: %d seeds x %d games", args.noise_seeds, args.noise_games)
    nsel = sel[: args.noise_games]
    nkeys = {g: keys_by_game[g] for g in nsel}
    noise = {}
    for name, arm in arms.items():
        per_seed = []
        for sd in range(args.noise_seeds):
            recs, rot, _ = V1.sim_records(arm, priors_te, scripts_te, nkeys,
                                          rotation_sets, [1000 + sd])
            per_seed.append(V1.build_row(name, recs, rot, a_row, act_minutes))
        noise[name] = {k: float(np.std([r[k] for r in per_seed], ddof=1))
                       for k in ["minutes_mean", "minutes_sd_pooled", "minutes_sd_within",
                                 "top5_share", "top8_share", "n_nonzero_mean", "lu_top1",
                                 "lu_top3", "lu_top5", "late_starter_share_b0",
                                 "late_starter_share_b1", "late_starter_share_b2",
                                 "foul_trouble_share"]}
        noise[name]["n_seeds"] = args.noise_seeds
        noise[name]["n_games"] = len(nsel)

    decision = decide2(rows, verdicts)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fold": args.tag, "round": 2,
        "train_season": TRAIN_SEASON, "test_season": TEST_SEASON,
        "config": vars(args), "test_universe": subset_note,
        "test_subset_seed": TEST_SUBSET_SEED,
        "crosswalk_coverage": cw_cov,
        "n_eligible_games": len(eligible), "n_simulated_games": len(sel),
        "actual": a_row, "actual_asof_starters": diag_row,
        "arms": rows, "verdicts": verdicts, "noise": noise,
        "decision": decision,
        "fit": {k: v for k, v in fit.__dict__.items() if k not in ("tilt", "notes")},
        "fit_notes": {k: v for k, v in fit.notes.items() if k != "scheduler_grid"},
    }
    # A dev smoke run must not be able to overwrite a graded artifact: the tag
    # goes in the filename unless it is the graded one. (It did, once.)
    stem = ("rotation_F1_round2" if args.tag.startswith("F1-round2")
            else f"rotation_round2_{args.tag}")
    (OUT_DIR / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2, default=V1._json_default))
    pd.DataFrame([a_row] + [rows[n] for n in rows]).to_csv(
        OUT_DIR / f"{stem}_table.csv", index=False)

    md = render_round2(payload)
    if args.append_experiments:
        with EXPERIMENTS.open("a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)
    log.info("done in %.1f min", (time.time() - t0) / 60)


SIMPLICITY = {"R2_hier_dirichlet": 1, "R5_hybrid": 2, "R6_stint_hazard_v2": 3}


def decide2(rows: dict, verdicts: dict) -> dict:
    scored = []
    for name, v in verdicts.items():
        g8 = sum(1 for c in V1.G8_CELLS if v.get(c))
        state = sum(1 for c in V1.STATE_CELLS if v.get(c))
        eligible = all(v.get(c) for c in V1.STATE_CELLS)   # round 2: ALL state cells
        scored.append({
            "arm": name, "g8_pass": g8, "g8_cells": len(V1.G8_CELLS),
            "state_pass": state, "state_cells": len(V1.STATE_CELLS),
            "total_pass": g8 + state, "eligible": bool(eligible),
            "ks_lineup_top1_d": rows[name]["ks_lineup_top1_d"],
            "simplicity_rank": SIMPLICITY[name],
        })
    scored.sort(key=lambda r: (-r["total_pass"], r["ks_lineup_top1_d"], r["simplicity_rank"]))
    winners = [r for r in scored if r["eligible"]]
    return {"ranking": scored,
            "winner": winners[0]["arm"] if winners else None,
            "diagnosis": None if winners else
            "no arm has every state-dependence cell inside +/- 3 pp"}


def render_round2(p: dict) -> str:
    a, rows, v, noise = p["actual"], p["arms"], p["verdicts"], p["noise"]
    dg = p.get("actual_asof_starters") or {}
    names = list(rows)
    L, W = [], None
    W = L.append
    W(f"\n## 4. Round-2 results (train {p['train_season']}, test {p['test_season']}) -- "
      f"run {p['generated_at'][:19]}Z\n")
    W(f"Test universe: {p['test_universe']} (subset seed {p['test_subset_seed']}), "
      f"{p['config']['seeds']} seeds per arm; noise floor "
      f"{p['config']['noise_seeds']} seeds x {p['config']['noise_games']} games. "
      f"Rotation player-games per arm {rows[names[0]]['n_rot_player_games']:,} "
      f"(actual {a['n_rot_player_games']:,}); every cell is far above the "
      f"n < 300 UNDERPOWERED threshold.\n")

    W("### 4.1 G8 cells\n")
    W("| cell | tol | ACTUAL | " + " | ".join(names) + " |")
    W("|---|---|---:|" + "---:|" * len(names))

    def line(label, key, tol, vkey, fmt="{:.4f}"):
        W(f"| {label} | {tol} | " + fmt.format(a[key]) + " | " +
          " | ".join(fmt.format(rows[n][key]) + "  " + ("PASS" if v[n].get(vkey) else "FAIL")
                     for n in names) + " |")

    line("minutes mean (rotation players)", "minutes_mean", "+/- 2.0", "minutes_mean", "{:.2f}")
    W("| minutes SD ratio, pooled | 0.9-1.1 | 1.000 | " + " | ".join(
        f"{v[n]['_sd_ratio_pooled_val']:.3f}  " +
        ("PASS" if v[n]['sd_ratio_pooled'] else "FAIL") for n in names) + " |")
    W("| minutes SD ratio, within-player | 0.9-1.1 | 1.000 | " + " | ".join(
        f"{v[n]['_sd_ratio_within_val']:.3f}  " +
        ("PASS" if v[n]['sd_ratio_within'] else "FAIL") for n in names) + " |")
    line("top-5 share of team minutes", "top5_share", "+/- 2 pp", "top5_share")
    line("top-8 share of team minutes", "top8_share", "+/- 2 pp", "top8_share")
    line("players with > 0 minutes", "n_nonzero_mean", "+/- 1.0", "n_nonzero", "{:.2f}")

    W("\n### 4.2 State-dependence cells (an arm missing ANY of these is ineligible)\n")
    W("| cell | ACTUAL | " + " | ".join(names) + " |")
    W("|---|---:|" + "---:|" * len(names))
    lab = {0: r"\|margin\| <= 5", 1: r"\|margin\| 6-15", 2: r"\|margin\| > 15"}
    for b in range(R.N_MARGIN_BUCKETS):
        k = f"late_starter_share_b{b}"
        W(f"| final 8:00 starters' share, {lab[b]} (n={a[f'late_slots_b{b}']:,.0f} poss) | "
          f"{a[k]:.4f} | " + " | ".join(
              f"{rows[n][k]:.4f} ({(rows[n][k]-a[k])*100:+.1f} pp)  " +
              ("PASS" if v[n][k] else "FAIL") for n in names) + " |")
    k = "foul_trouble_share"
    W(f"| starters on floor while carrying >= 4 fouls (n={a['foul_trouble_opps']:,.0f}) | "
      f"{a[k]:.4f} | " + " | ".join(
          f"{rows[n][k]:.4f} ({(rows[n][k]-a[k])*100:+.1f} pp)  " +
          ("PASS" if v[n][k] else "FAIL") for n in names) + " |")
    W(f"| (diagnostic) at exactly 4 fouls (n={a['foul4_opps']:,.0f}) | {a['foul4_share']:.4f} | "
      + " | ".join(f"{rows[n]['foul4_share']:.4f}" for n in names) + " |")

    if dg:
        W(f"\nThe rows above count, on each side, the five that side actually started. "
          f"The model's as-of starter set overlaps the real starting five on "
          f"**{dg['starter_overlap_of5']:.2f} of 5**; re-grading the ACTUAL on-floor "
          f"sequence with the MODEL's starter set separates \"wrong five\" from "
          f"\"wrong rotation\":\n")
        W("| cell | ACTUAL (own starters) | ACTUAL (as-of starter set) | " +
          " | ".join(names) + " |")
        W("|---|---:|---:|" + "---:|" * len(names))
        for b in range(R.N_MARGIN_BUCKETS):
            k = f"late_starter_share_b{b}"
            W(f"| final 8:00, {lab[b]} | {a[k]:.4f} | {dg[k]:.4f} | " +
              " | ".join(f"{rows[n][k]:.4f}" for n in names) + " |")
        W(f"| >= 4 fouls | {a['foul_trouble_share']:.4f} | {dg['foul_trouble_share']:.4f} | "
          + " | ".join(f"{rows[n]['foul_trouble_share']:.4f}" for n in names) + " |")

    W("\n### 4.3 Lineup concentration\n")
    W("| metric | ACTUAL | " + " | ".join(names) + " |")
    W("|---|---:|" + "---:|" * len(names))
    for label, key, fmt in (("top-1 lineup share of possessions", "lu_top1", "{:.4f}"),
                            ("top-3 lineup share", "lu_top3", "{:.4f}"),
                            ("top-5 lineup share", "lu_top5", "{:.4f}"),
                            ("distinct lineups per team-game", "n_lineups_mean", "{:.2f}"),
                            ("K-S of per-player minutes (D)", "ks_minutes_d", "{:.4f}")):
        av = a.get(key, float("nan"))
        W(f"| {label} | " + (fmt.format(av) if np.isfinite(av) else "--") + " | " +
          " | ".join(fmt.format(rows[n][key]) for n in names) + " |")
    W("| K-S of the top-1 lineup share distribution (D) | -- | " +
      " | ".join(f"{rows[n]['ks_lineup_top1_d']:.4f}" for n in names) + " |")
    W("| substitution rate at a possession boundary | " +
      f"{p['fit_notes']['observed_change_rate_train']:.4f} (train) | " +
      " | ".join(f"{rows[n]['change_rate']:.4f}" for n in names) + " |")

    W(f"\n### 4.4 Noise floor ({p['config']['noise_seeds']} seeds x "
      f"{p['config']['noise_games']} games)\n")
    W("| metric | " + " | ".join(names) + " |")
    W("|---|" + "---:|" * len(names))
    for k in ["minutes_mean", "top5_share", "n_nonzero_mean", "lu_top1",
              "late_starter_share_b0", "late_starter_share_b1",
              "late_starter_share_b2", "foul_trouble_share"]:
        W(f"| {k} | " + " | ".join(f"{noise[n][k]:.5f}" for n in names) + " |")

    W("\n### 4.5 R5's fitted donor depth and override coefficients\n")
    fn = p["fit_notes"]
    W(f"Donor depth **k = {fn['r5_donor_k']}** "
      f"({'every earlier game of the season' if fn['r5_donor_k'] == 0 else 'last %d games' % fn['r5_donor_k']}), "
      f"chosen on the training season by lineup-concentration error:\n")
    W("| k | top-1 lineup share | top-3 | distinct lineups | rel. sq. error |")
    W("|---|---:|---:|---:|---:|")
    for k, sc in fn["r5_donor_k_grid"].items():
        W(f"| {k if k != '0' else 'all'} | {sc['lu_top1']:.4f} | {sc['lu_top3']:.4f} | "
          f"{sc['n_lineups']:.2f} | {sc['err']:.5f} |")
    W(f"\n`override_scale` = **{fn['r5_override_scale']:.2f}**, `p0` (neutral-state block "
      f"floor) = **{fn['r5_block_base']:.3f}**, fitted on the TRAINING "
      f"season against the four state-dependence cells the overrides exist to produce. "
      f"Train targets: "
      + ", ".join(f"{c.replace('late_starter_share_b', 'late band ')}"
                  f"{'>= 4 fouls' if c == 'foul_trouble_share' else ''} {v:.4f}"
                  for c, v in fn["r5_state_targets_train"].items()) + ".\n")
    W("| scale, p0 | late b0 | late b1 | late b2 | >= 4 fouls | sub rate | sq. err |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for k, sc in fn["r5_override_scale_grid"].items():
        W(f"| {k} | {sc['late_starter_share_b0']:.4f} | {sc['late_starter_share_b1']:.4f} | "
          f"{sc['late_starter_share_b2']:.4f} | {sc['foul_trouble_share']:.4f} | "
          f"{sc['change_rate']:.4f} | {sc['err']:.6f} |")
    W("")
    W("Fitted override coefficients (logit scale; positive = more likely to be benched / "
      "to come on):\n")
    W("| feature | exit (bench-him) | enter (bring-him-back) |")
    W("|---|---:|---:|")
    for i, f in enumerate(fn["r5_features"]):
        W(f"| `{f}` | {fn['r5_exit']['coef'][i]:+.4f} | {fn['r5_enter']['coef'][i]:+.4f} |")
    W(f"| _intercept_ | {fn['r5_exit']['intercept']:+.4f} | "
      f"{fn['r5_enter']['intercept']:+.4f} |")

    W("\n### 4.6 Decision\n")
    W("| arm | G8 cells | state cells | total | eligible (all 4 state cells) | "
      "lineup K-S D | simplicity |")
    W("|---|---:|---:|---:|---|---:|---:|")
    for r in p["decision"]["ranking"]:
        W(f"| {r['arm']} | {r['g8_pass']}/{r['g8_cells']} | {r['state_pass']}/"
          f"{r['state_cells']} | {r['total_pass']} | {'yes' if r['eligible'] else 'NO'} | "
          f"{r['ks_lineup_top1_d']:.4f} | {r['simplicity_rank']} |")
    if p["decision"]["winner"]:
        W(f"\n**Winner: `{p['decision']['winner']}`** by the pre-registered rule.\n")
    else:
        W(f"\n**No arm adopted.** {p['decision']['diagnosis']}. Cell-by-cell misses:\n")
        W("| arm | cell | sim | actual | miss |")
        W("|---|---|---:|---:|---:|")
        for n in names:
            for c in V1.STATE_CELLS:
                if not v[n].get(c):
                    W(f"| {n} | {c} | {rows[n][c]:.4f} | {a[c]:.4f} | "
                      f"{(rows[n][c]-a[c])*100:+.1f} pp |")
        W("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
