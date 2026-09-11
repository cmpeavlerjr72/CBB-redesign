#!/usr/bin/env python
"""
train_usage_v2.py -- L4 SHOT ALLOCATION (usage), ROUND 2: the shooter-label
data fix.

Round 1 (`scripts/train_usage_v1.py`, `docs/models/usage/experiments.md`
sections 1-7) keyed the field-goal shooter on CBBD `participant_1_id`. That
column is the ASSISTER on 48.98% of assisted made field goals -- CBBD's
`participants` array is not ordered shooter-first -- which mislabels 4.9-14.1%
of rows per field-goal class, at a rate that differs by a factor of ~3 BETWEEN
the classes. Evidence: `docs/tests/shooter_key_audit_2026-09-10.md`.

THIS TRAINER CHANGES EXACTLY ONE THING: the shooter label. It calls
`usage.build_usage_events(shooter_key="shot_shooter_id")` and drops the
0.04-0.27% of field-goal rows with no shooter id (never imputed, never fallen
back to `participant_1_id`, reported per season and per team). Everything else
-- arms, features, folds, metrics, gates, decision rule, seeds, draw counts --
is imported from `train_usage_v1` and is byte-identical, so the two rounds
differ by the label and by nothing else. That is what the pre-registration in
`experiments.md` section 8 promises and it is enforced by import rather than by
copy-paste.

Artifacts go to a VERSIONED SIBLING directory (`data/processed/models/usage_v2`
by default). Nothing under `data/processed/models/usage` is written or moved:
the engine worker reads `usage/asof_v2.parquet` and `usage/usage_params_v1.json`
concurrently and must not be disturbed mid-run.

Usage:
    .venv/Scripts/python.exe scripts/train_usage_v2.py
    .venv/Scripts/python.exe scripts/train_usage_v2.py --classes FGA_3 --quick
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

# Four other workers share this machine (CLAUDE.md, "Worker discipline"). The
# caps are set before numpy/lightgbm import so they actually bind.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import train_usage_v1 as V1  # noqa: E402

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import usage as U  # noqa: E402

SEASONS = V1.SEASONS
SHOOTER_KEY = "shot_shooter_id"
ROUND1_RESULTS = ROOT / "data/processed/models/usage/results_v1.json"
#: Minimum test events for a team to enter the per-team segment table. Below it
#: the per-team log loss is noise and the team is counted as underpowered.
TEAM_MIN_EVENTS = 200


# ---------------------------------------------------------------------------
# Data (the ONLY thing that differs from round 1)
# ---------------------------------------------------------------------------
def build_tables(version: str, out_dir: Path, rebuild: bool, log):
    """`train_usage_v1.build_tables` with `shooter_key` threaded through and a
    drop report added. Kept here rather than parameterised in v1 so that v1
    stays exactly the script that produced round 1."""
    ev_path = out_dir / f"events_{version}_shotshooter.parquet"
    asof_path = out_dir / f"asof_{version}_shotshooter.parquet"
    meta_path = out_dir / f"build_report_{version}_shotshooter.json"
    if ev_path.exists() and asof_path.exists() and meta_path.exists() and not rebuild:
        return (pd.read_parquet(ev_path), pd.read_parquet(asof_path),
                json.loads(meta_path.read_text()))

    universe = ES.load_universe(require_pbp_complete=True)
    rim = ES.rim_override_for_version(version)
    per_season, coverage, drops, relabel = {}, {}, {}, {}
    for s in SEASONS:
        raw = U.build_usage_events(s, universe, rim_override_max_ft=rim,
                                   shooter_key=SHOOTER_KEY)
        coverage[str(s)] = U.coverage_report(raw)
        drops[str(s)] = U.shooter_drop_report(raw)
        relabel[str(s)] = {"n_fga_relabelled": raw.attrs["n_fga_relabelled"],
                           "n_fga_no_shooter": raw.attrs["n_fga_no_shooter"]}
        log(f"  {s}: {raw.attrs['n_fga_relabelled']:,} FGA rows relabelled, "
            f"{raw.attrs['n_fga_no_shooter']:,} with no shooter id")
        per_season[s] = U.usable_events(raw)
    minutes = U.load_minutes([SEASONS[0] - 1, *SEASONS])
    asof = U.build_player_asof(per_season, minutes=minutes)
    events = pd.concat([per_season[s] for s in SEASONS], ignore_index=True)
    meta = {
        "possessions_version": version,
        "shooter_key": SHOOTER_KEY,
        "rim_override_max_ft": rim,
        "coverage": coverage,
        "shooter_drops": drops,
        "relabelled": relabel,
        "minutes_join_pct": asof.attrs.get("minutes_join_pct"),
        "n_events": int(len(events)),
        "n_player_games": int(len(asof)),
        "position_known_pct": round(
            float((asof["position_group"] != "UNK").mean() * 100), 4),
        "prior_season_pct": {str(s): round(float(
            asof.loc[asof.season == s, "has_prior_season"].mean() * 100), 4)
            for s in SEASONS},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(ev_path, index=False)
    asof.to_parquet(asof_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return events, asof, meta


# ---------------------------------------------------------------------------
# Segments the round-2 pre-registration adds on top of round 1's battery
# ---------------------------------------------------------------------------
def team_segment(te: pd.DataFrame, p: np.ndarray) -> dict:
    """Per-OFFENSIVE-TEAM log loss distribution on the test fold.

    Reported so a headline that improves on average cannot hide a set of teams
    it makes worse (`CLAUDE.md`, multi-level evidence). Teams under
    `TEAM_MIN_EVENTS` are excluded and counted, never presented as signal."""
    y = te["y"].to_numpy()
    ll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    d = pd.DataFrame({"t": te["team_id"].to_numpy(), "ll": ll})
    g = d.groupby("t")["ll"].agg(["size", "mean"])
    big = g[g["size"] >= TEAM_MIN_EVENTS]
    if not len(big):
        return {"teams": 0, "underpowered_teams": int(len(g))}
    q = big["mean"]
    return {"teams": int(len(big)), "underpowered_teams": int(len(g) - len(big)),
            "min": round(float(q.min()), 6), "p10": round(float(q.quantile(0.10)), 6),
            "median": round(float(q.median()), 6),
            "p90": round(float(q.quantile(0.90)), 6),
            "max": round(float(q.max()), 6), "sd": round(float(q.std(ddof=1)), 6)}


def run_fold_with_segments(cls, tr, te, fold, args, log, lgbm_params=None):
    """`V1.run_fold`, plus the per-team segment.

    `V1.run_fold` does not return its per-arm probability matrices, so the
    segment is computed by re-deriving them the same way `V1.run_fold` scores
    them -- which would duplicate the fit. Instead the arm probabilities are
    captured by monkey-patching `V1.score_arm`'s caller: `U.score_arm` is called
    once per arm with exactly the matrix we need, so it is wrapped for the
    duration of the call. No numbers change; the wrapper only records."""
    captured: list[np.ndarray] = []
    real = U.score_arm

    def spy(te_, p, driver):
        captured.append(np.array(p, copy=True))
        return real(te_, p, driver)

    U.score_arm = spy
    try:
        out = V1.run_fold(cls, tr, te, fold, args, log, lgbm_params=lgbm_params)
    finally:
        U.score_arm = real
    for arm, p in zip(U.ARMS, captured, strict=False):
        out["arms"][arm]["team_segment"] = team_segment(te, p)
    return out


# ---------------------------------------------------------------------------
# Report -- appended to experiments.md as section 8
# ---------------------------------------------------------------------------
def fmt(x, nd=4):
    return V1.fmt(x, nd)


def load_round1() -> dict:
    if not ROUND1_RESULTS.exists():
        return {}
    r1 = json.loads(ROUND1_RESULTS.read_text())
    return {(r["fold"], r["event_class"]): r for r in r1}


def write_report(meta: dict, results: list[dict], path: Path, args) -> None:
    r1 = load_round1()
    L: list[str] = []
    A = L.append
    A("")
    A(f"### 8.4 Run configuration (run {time.strftime('%Y-%m-%d')} by "
      f"`scripts/train_usage_v2.py`, after the section 8.1-8.3 pre-registration "
      f"was committed)\n")
    A("| item | value |")
    A("|---|---|")
    A("| trainer | `scripts/train_usage_v2.py` |")
    A(f"| shooter key | `{meta['shooter_key']}` (round 1: `participant_1_id`) |")
    A(f"| possessions version | `{meta['possessions_version']}` "
      f"(rim override {meta['rim_override_max_ft']} ft) |")
    A("| universe | D-I, non-truncated, `pbp_complete` (identical to round 1) |")
    A(f"| credited events (modelled) | {meta['n_events']:,} "
      f"(round 1: 1,634,792) |")
    A(f"| player-games with as-of inputs | {meta['n_player_games']:,} |")
    A(f"| roster position known | {meta['position_known_pct']}% |")
    A(f"| hoopR minutes joined through the player crosswalk | {meta['minutes_join_pct']}% |")
    A("| player-games with a prior season of on-floor history | "
      + ", ".join(f"{k}: {v}%" for k, v in meta["prior_season_pct"].items()) + " |")
    A(f"| Monte-Carlo draws (marginal / Polya / game-level / alpha fit) | "
      f"{args.mc_draws} / {args.seq_draws} / {args.sim_draws} / {args.alpha_draws} |")
    A(f"| seed | {args.seed} |")
    A("")

    A("### 8.5 What the label fix moved, and what it dropped\n")
    A("| season | FGA rows relabelled | FGA rows with no `shot_shooter_id` |")
    A("|---|---:|---:|")
    for s, v in meta["relabelled"].items():
        A(f"| {s} | {v['n_fga_relabelled']:,} | {v['n_fga_no_shooter']:,} |")
    A("")
    A("Drop rate by season and class (rows with no credited player id; NEVER "
      "imputed), with the per-team distribution over teams with >= 50 rows of "
      "the class:\n")
    A("| season | class | rows | dropped | drop % | per-team min / median / p95 / max % | teams > 1% | underpowered teams |")
    A("|---|---|---:|---:|---:|---|---:|---:|")
    for s, rep in meta["shooter_drops"].items():
        for c in U.EVENT_CLASSES:
            bc = rep["by_class"].get(c)
            if not bc:
                continue
            bt = rep["by_team"].get(c, {})
            span = ("--" if not bt.get("teams") else
                    f"{bt['min_pct']} / {bt['median_pct']} / {bt['p95_pct']} / {bt['max_pct']}")
            A(f"| {s} | {c} | {bc['n']:,} | {bc['dropped_no_player_id']:,} | "
              f"{bc['drop_pct']} | {span} | {bt.get('teams_over_1pct', '--')} | "
              f"{bt.get('underpowered_teams', '--')} |")
    A("")
    A("### 8.6 Event coverage (reported, not silently filtered)\n")
    A("| season | class | events | five resolved | credited id present | modelled |")
    A("|---|---|---:|---:|---:|---:|")
    for s, cov in meta["coverage"].items():
        for c, r in cov.items():
            A(f"| {s} | {c} | {r['n']:,} | {r['five_resolved_pct']}% | "
              f"{r['player_id_present_pct']}% | {r['modelled_pct']}% |")
    A("")

    # ---- F1 -------------------------------------------------------------
    A("### 8.7 F1 results (train 2024, test 2025) -- the selection fold\n")
    for res in [r for r in results if r["fold"] == "F1"]:
        A(f"#### {res['event_class']}\n")
        A(f"Train {res['n_train']:,} events, test {res['n_test']:,}. "
          f"Fitted shrinkage: prior `{res['prior_kind']}`, m = {res['shrink_m']:g} "
          f"pseudo on-floor events. Uniform-over-five log loss = 1.609438.\n")
        A("| arm | log loss | Brier | top-1 | top-3 | calib worst (pp) | resp steps | "
          "slope ratio | boot SE | SD ratio | players >=1 (sim / real) | "
          "top-1 (sim / real) | top-3 (sim / real) | eligible |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
        for arm in U.ARMS:
            r = res["arms"][arm]
            gl = r["game_level"]
            ok = res["decision"]["checks"][arm]
            A(f"| {arm} | {r['log_loss']:.6f} | {r['brier']:.4f} | {r['top1']:.4f} | "
              f"{r['top3']:.4f} | {r['calib_worst_gap_pp']:.3f} | {r['resp_steps']}/4 | "
              f"{fmt(r.get('resp_slope_ratio'), 4)} | {r['bootstrap_se']:.6f} | "
              f"{fmt(gl['sd_ratio'], 4)} | "
              f"{fmt(gl['players_gt0_sim'], 3)} / {fmt(gl['players_gt0_actual'], 3)} | "
              f"{fmt(gl['top1_sim_pct'], 2)}% / {fmt(gl['top1_actual_pct'], 2)}% | "
              f"{fmt(gl['top3_sim_pct'], 2)}% / {fmt(gl['top3_actual_pct'], 2)}% | "
              f"{'yes' if all(ok.values()) else 'NO (' + ', '.join(k for k, v in ok.items() if not v) + ')'} |")
        A("")
        A(f"**Decision: {res['decision']['winner'] or 'NO WINNER'}.** "
          f"{res['decision']['reason']}\n")
        prev = r1.get(("F1", res["event_class"]))
        if prev:
            A(f"Round 1 on this fold decided **{prev['decision']['winner'] or 'NO WINNER'}**; "
              f"round 2 decides **{res['decision']['winner'] or 'NO WINNER'}**. "
              "Per-arm log loss, round 1 -> round 2 (the LABELS differ between the "
              "rounds, so the LEVELS are not a like-for-like comparison and only the "
              "ordering and the gate verdicts are):\n")
            A("| arm | round 1 | round 2 | delta |")
            A("|---|---:|---:|---:|")
            for arm in U.ARMS:
                a, b = prev["arms"][arm]["log_loss"], res["arms"][arm]["log_loss"]
                A(f"| {arm} | {a:.6f} | {b:.6f} | {b - a:+.6f} |")
            A("")
        A("Per-team log loss (teams with >= "
          f"{TEAM_MIN_EVENTS} test events of this class):\n")
        A("| arm | teams | min | p10 | median | p90 | max | SD | underpowered |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm in U.ARMS:
            t = res["arms"][arm].get("team_segment", {})
            if not t.get("teams"):
                A(f"| {arm} | 0 | -- | -- | -- | -- | -- | -- | "
                  f"{t.get('underpowered_teams', '--')} |")
                continue
            A(f"| {arm} | {t['teams']} | {t['min']:.4f} | {t['p10']:.4f} | "
              f"{t['median']:.4f} | {t['p90']:.4f} | {t['max']:.4f} | {t['sd']:.4f} | "
              f"{t['underpowered_teams']} |")
        A("")
        A("Responsiveness by player as-of-rate quintile (the matchup-specific "
          "slope check; predicted vs actual credited share, %):\n")
        A("| arm | Q1 pred / act | Q2 | Q3 | Q4 | Q5 | span pred / act (pp) | "
          "slope ratio | steps | verdict |")
        A("|---|---|---|---|---|---|---|---:|---:|---|")
        for arm in U.ARMS:
            rp = res["arms"][arm]["responsiveness"]
            cells = " | ".join(
                f"{100 * p:.2f} / {100 * a:.2f}"
                for p, a in zip(rp["pred_share"], rp["actual_share"], strict=False))
            A(f"| {arm} | {cells} | {100 * rp['span_pred']:.2f} / "
              f"{100 * rp['span_actual']:.2f} | {fmt(rp['slope_ratio'], 4)} | "
              f"{rp['pred_monotone_steps']}/{rp['steps_required']} | "
              f"{'PASS' if rp['pass'] else 'FAIL'} |")
        A("")
        lg = res["arms"]["lgbm"]
        A(f"Noise floor. Block-bootstrap SE (the floor used by the decision rule) "
          f"is the per-arm `boot SE` column above; the largest over the eligible arms "
          f"is {res['decision'].get('floor', float('nan')):.6f}. "
          f"Spec-identical LightGBM retrains under seeds {list(V1.LGBM_SEEDS)}: "
          f"{lg['seed_log_losses']}, SD {lg['seed_sd']}.\n")
        A("Transfer subset (2025 credited players whose modal team changed):\n")
        A("| arm | transfers | continuing | no prior season |")
        A("|---|---:|---:|---:|")
        for arm in U.ARMS:
            t = res["arms"][arm]["transfer"]
            A(f"| {arm} | {fmt(t['transfer_log_loss'], 6)} | "
              f"{fmt(t['continuing_log_loss'], 6)} | "
              f"{fmt(t['no_prior_season_log_loss'], 6)} |")
        A(f"\n(n transfers = {res['arms']['proportional']['transfer']['transfer_n']:,})\n")

    # ---- WF -------------------------------------------------------------
    wf = [r for r in results if r["fold"] == "WF2025"]
    if wf:
        A(f"### 8.8 Robustness fold: within-2025 walk-forward "
          f"(train before {U.WF_SPLIT_DATE}, test after)\n")
        A("| class | arm | log loss | boot SE | calib (pp) | resp | slope ratio | "
          "SD ratio | players >=1 delta | eligible |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for res in wf:
            for arm in U.ARMS:
                r = res["arms"][arm]
                gl = r["game_level"]
                ok = res["decision"]["checks"][arm]
                A(f"| {res['event_class']} | {arm} | {r['log_loss']:.6f} | "
                  f"{r['bootstrap_se']:.6f} | {r['calib_worst_gap_pp']:.3f} | "
                  f"{r['resp_steps']}/4 | {fmt(r.get('resp_slope_ratio'), 4)} | "
                  f"{fmt(gl['sd_ratio'], 4)} | {fmt(gl['players_gt0_delta'], 3)} | "
                  f"{'yes' if all(ok.values()) else 'no'} |")
        A("")
        A("| class | winner | reason |")
        A("|---|---|---|")
        for res in wf:
            A(f"| {res['event_class']} | {res['decision']['winner'] or 'NO WINNER'} | "
              f"{res['decision']['reason']} |")
        A("")

    # ---- the decision table --------------------------------------------
    A("### 8.9 Round 2 decision, against round 1\n")
    A("| class | round 1 winner (F1) | round 2 winner (F1) | changed? | "
      "round 1 winner (WF) | round 2 winner (WF) | floor (F1) |")
    A("|---|---|---|---|---|---|---:|")
    for res in [r for r in results if r["fold"] == "F1"]:
        c = res["event_class"]
        p1 = r1.get(("F1", c), {}).get("decision", {}).get("winner")
        p2 = r1.get(("WF2025", c), {}).get("decision", {}).get("winner")
        w2 = res["decision"]["winner"]
        wwf = next((x["decision"]["winner"] for x in wf if x["event_class"] == c), None)
        A(f"| {c} | {p1 or '--'} | {w2 or 'NO WINNER'} | "
          f"{'**YES**' if p1 != w2 else 'no'} | {p2 or '--'} | {wwf or '--'} | "
          f"{res['decision'].get('floor', float('nan')):.6f} |")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_results(out_dir: Path, report: Path, meta: dict, results: list[dict],
                  args) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_v2.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    params = {
        "possessions_version": args.version,
        "shooter_key": SHOOTER_KEY,
        "seed": args.seed,
        "per_class": {r["event_class"]: {
            "prior_kind": r["prior_kind"], "shrink_m": r["shrink_m"],
            "winner": r["decision"]["winner"],
            "dirichlet_alphas": r["arms"]["dirichlet"]["alphas"],
            "hier_alphas": r["arms"]["hier_dirichlet"]["alphas"],
            "cond_logit_l2": r["arms"]["cond_logit"]["l2"],
            "inner_split": V1.split_of(r),
            "cond_logit_features": r["arms"]["cond_logit"]["features"],
            "cond_logit_coefficients": r["arms"]["cond_logit"]["coefficients"],
            "lgbm_params": r["arms"]["lgbm"]["params"],
        } for r in results if r["fold"] == "F1"},
    }
    (out_dir / "usage_params_v2.json").write_text(json.dumps(params, indent=2),
                                                  encoding="utf-8")
    write_report(meta, results, report, args)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2",
                    help="possessions build to read the rim override from")
    ap.add_argument("--out", default="data/processed/models/usage_v2")
    ap.add_argument("--report", default=None)
    ap.add_argument("--classes", nargs="*", default=list(U.EVENT_CLASSES))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--mc-draws", type=int, default=200)
    ap.add_argument("--seq-draws", type=int, default=100)
    ap.add_argument("--sim-draws", type=int, default=40)
    ap.add_argument("--alpha-draws", type=int, default=15)
    ap.add_argument("--alpha-max-tg", type=int, default=3000)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--inner-split", default=V1.INNER_SPLIT_DATE)
    ap.add_argument("--skip-wf", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.mc_draws, args.seq_draws, args.sim_draws = 40, 30, 10
        args.alpha_draws, args.alpha_max_tg, args.boot_reps = 6, 800, 50

    out_dir = ROOT / args.out
    report = Path(args.report) if args.report else out_dir / "report_v2.md"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    log(f"shooter_key = {SHOOTER_KEY} (round 1 used participant_1_id)")
    events, asof, meta = build_tables(args.version, out_dir, args.rebuild, log)
    log(f"events {len(events):,} asof {len(asof):,} ({time.time() - t0:.1f}s)")

    results = []
    for cls in args.classes:
        design = U.build_usage_design(events, asof, cls)
        tr, te = U.fold_slices(design)
        f1 = run_fold_with_segments(cls, tr, te, "F1", args, log)
        results.append(f1)
        if not args.skip_wf:
            wtr, wte = U.walkforward_slices(design)
            results.append(run_fold_with_segments(
                cls, wtr, wte, "WF2025", args, log,
                lgbm_params=f1["arms"]["lgbm"]["params"]))
        del design
        write_results(out_dir, report, meta, results, args)   # checkpoint
        (out_dir / "train_log_v2.txt").write_text("\n".join(log_lines),
                                                  encoding="utf-8")

    write_results(out_dir, report, meta, results, args)
    (out_dir / "train_log_v2.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nwrote {report} and {out_dir}/usage_params_v2.json "
        f"({time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
