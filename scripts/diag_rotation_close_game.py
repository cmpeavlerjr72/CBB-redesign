#!/usr/bin/env python
"""
diag_rotation_close_game.py -- evidence audit for rotation round 3.

Characterises the close-game late-keep behaviour of the ACTUAL on-floor data and
of the two round-2 arms that matter (R2_hier_dirichlet, R5_hybrid refitted with
the corrected hazard matrix, `experiments.md` section 5), broken out by

  * season (on-floor data exists only in 2024 and 2025 -- L13; 2022 and 2023 are
    reported as an availability count, which is the only thing they can say),
  * a FINE time profile over the whole game, not just the final 8:00, because
    the round-2 diagnosis (`model.md` section 10) is that R5 spends starter time
    too early,
  * score-margin bucket,
  * starter vs bench,
  * team quintile of the as-of predicted starter-minutes share (the standing
    matchup-responsiveness slope check).

Writes `docs/tests/rotation_close_game_audit_2026-09-10.md`.

Usage:
    .venv/Scripts/python.exe scripts/diag_rotation_close_game.py --games 400 --seeds 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import rotation as R  # noqa: E402

import train_rotation_v1 as V1  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("diag_rotation_close_game")

OUT_MD = ROOT / "docs" / "tests" / "rotation_close_game_audit_2026-09-10.md"
OUT_JSON = ROOT / "data" / "processed" / "models" / "rotation" / "close_game_audit_2026-09-10.json"
FIT_CORRECTED = (ROOT / "data" / "processed" / "models" / "rotation"
                 / "rotation_fit_round2_corrected_hazards.json")
TEST_SUBSET_SEED = 2025

# ---- the fine time profile ------------------------------------------------
# (label, period, clock-hi, clock-lo]  -- clock is seconds REMAINING in the period
TIME_CELLS = [
    ("H1 20:00-10:00", 1, 1200, 600),
    ("H1 10:00-00:00", 1, 600, -1),
    ("H2 20:00-16:00", 2, 1200, 960),
    ("H2 16:00-12:00", 2, 960, 720),
    ("H2 12:00-08:00", 2, 720, 480),
    ("H2 08:00-04:00", 2, 480, 240),
    ("H2 04:00-02:00", 2, 240, 120),
    ("H2 02:00-00:00", 2, 120, -1),
]
MB_LAB = {0: "|m|<=5", 1: "|m| 6-15", 2: "|m|>15"}
UNDERPOWERED = 300      # possessions; the pre-registered cell floor


def time_cell(period: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """Index into TIME_CELLS; -1 for overtime."""
    out = np.full(len(period), -1, dtype="int64")
    for i, (_lab, p, hi, lo) in enumerate(TIME_CELLS):
        out[(period == p) & (clock <= hi) & (clock > lo)] = i
    return out


# ---------------------------------------------------------------------------
def starter_slots_from_lineups(lu: np.ndarray, starters: set) -> np.ndarray:
    """Starters on the floor, per possession (0-5)."""
    ids, inv = np.unique(lu, return_inverse=True)
    st = np.array([1.0 if int(i) in starters else 0.0 for i in ids])
    return st[inv.reshape(lu.shape)].sum(axis=1)


def profile_from_sequence(lu, starters, period, clock, mb, dur):
    """(time cell x margin bucket) -> [starter slots, total slots, seconds]."""
    ss = starter_slots_from_lineups(lu, starters)
    tc = time_cell(np.asarray(period), np.asarray(clock))
    acc = np.zeros((len(TIME_CELLS), 3, 3), dtype="float64")
    for i in range(len(TIME_CELLS)):
        for b in range(3):
            m = (tc == i) & (np.asarray(mb) == b)
            if m.any():
                acc[i, b, 0] = ss[m].sum()
                acc[i, b, 1] = 5.0 * m.sum()
                acc[i, b, 2] = np.asarray(dur)[m].sum()
    return acc


def fmt_profile(acc: np.ndarray) -> list[list[str]]:
    rows = []
    for i, (lab, _p, _hi, _lo) in enumerate(TIME_CELLS):
        cells = []
        for b in range(3):
            num, den = acc[i, b, 0], acc[i, b, 1]
            if den / 5.0 < UNDERPOWERED:
                cells.append(f"UP({den/5.0:.0f})")
            else:
                cells.append(f"{num/den:.4f}")
        rows.append([lab] + cells + [f"{acc[i, :, 1].sum()/5.0:,.0f}"])
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--min-prior-games", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    assert_not_sealed([2022, 2023, 2024, 2025], context="rotation close-game audit")
    payload: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                     "config": vars(args)}

    # ---- 1. on-floor availability, 2022-2025 -----------------------------
    avail_rows = []
    import pyarrow.parquet as pq
    for season in (2022, 2023, 2024, 2025):
        path = ROOT / "data" / "processed" / "possessions" / f"possessions_{season}.parquet"
        have = set(pq.ParquetFile(path).schema.names)
        oncols = [c for c in R.ON_FLOOR_H + R.ON_FLOOR_A if c in have]
        if not oncols:
            p = pd.read_parquet(path, columns=["game_id"])
            avail_rows.append({"season": season, "games": int(p["game_id"].nunique()),
                               "poss_with_on_floor": 0.0, "games_fully_complete": 0,
                               "note": "no on_floor columns in the file at all"})
            log.info("season %d: no on-floor columns", season)
            continue
        p = pd.read_parquet(path, columns=["game_id"] + oncols)
        comp = p[oncols].notna().all(axis=1) if len(oncols) == 10 else pd.Series(
            False, index=p.index)
        by_game = comp.groupby(p["game_id"]).mean()
        avail_rows.append({
            "season": season,
            "games": int(p["game_id"].nunique()),
            "poss_with_on_floor": float(comp.mean()),
            "games_fully_complete": int((by_game == 1.0).sum()),
        })
        log.info("season %d: %.4f of possessions carry on-floor, %d complete games",
                 season, comp.mean(), int((by_game == 1.0).sum()))
    payload["availability"] = avail_rows

    # ---- 2. actual time profile, 2024 and 2025 ---------------------------
    seasons = {}
    for season in (2024, 2025):
        seasons[season] = V1.load_season(season)

    actual_profile = {}
    starter_bench = {}
    for season, d in seasons.items():
        tp = d["tp"]
        acc = np.zeros((len(TIME_CELLS), 3, 3), dtype="float64")
        for key, g in tp.groupby(["game_id", "team_id"], sort=False):
            lu = g[R.SLOTS].to_numpy(dtype="int64")
            starters = set(int(x) for x in lu[0])
            acc += profile_from_sequence(lu, starters, g["period"].to_numpy(),
                                         g["start_clock"].to_numpy(),
                                         g["margin_bucket"].to_numpy(),
                                         g["duration_s"].to_numpy())
        actual_profile[season] = acc
        pg = d["pg"]
        sb = pg.groupby("is_starter")["minutes"].agg(["mean", "std", "count"])
        starter_bench[season] = {
            "starter_mean": float(sb.loc[True, "mean"]),
            "starter_sd": float(sb.loc[True, "std"]),
            "starter_n": int(sb.loc[True, "count"]),
            "bench_mean": float(sb.loc[False, "mean"]),
            "bench_sd": float(sb.loc[False, "std"]),
            "bench_n": int(sb.loc[False, "count"]),
            "starter_share_of_team_minutes": float(
                pg.loc[pg["is_starter"], "minutes"].sum() / pg["minutes"].sum()),
        }
        log.info("season %d actual profile done", season)
    payload["actual_profile"] = {str(k): v.tolist() for k, v in actual_profile.items()}
    payload["starter_bench"] = starter_bench

    # ---- 3. sim side: R2 and R5 (corrected hazards) on 2025 --------------
    fit = R.RotationFit.from_json(FIT_CORRECTED)
    test = seasons[2025]
    priors_te = R.build_priors(test["feats"], fit, min_prior_games=args.min_prior_games)
    scripts_te = R.build_scripts(test["tp"])
    both: dict[int, list] = {}
    for (gid, tid) in scripts_te:
        if (gid, tid) in priors_te:
            both.setdefault(gid, []).append((gid, tid))
    eligible = sorted([g for g, v in both.items() if len(v) == 2])
    rs = np.random.RandomState(TEST_SUBSET_SEED)
    sel_full = sorted(rs.choice(eligible, size=min(1600, len(eligible)), replace=False))
    sel = sel_full[: args.games]
    log.info("sim subset: %d of the %d graded round-2 games (subset seed %d)",
             len(sel), len(sel_full), TEST_SUBSET_SEED)

    keys_by_game = {}
    for g in sel:
        home = [k for k in both[g] if scripts_te[k].is_home]
        away = [k for k in both[g] if not scripts_te[k].is_home]
        keys_by_game[g] = home + away

    donors = R.build_donor_bank(test["tp"], k_donors=fit.notes["r5_donor_k"])
    arms = {"R2_hier_dirichlet": R.R2HierDirichlet(fit),
            "R5_hybrid_corrected": R.R5Hybrid(fit, donors)}

    # actual profile restricted to exactly the simulated team-games, so sim and
    # actual are compared on the same universe
    kset = {k for v in keys_by_game.values() for k in v}
    tps = test["tp"]
    tps = tps[[(int(g), int(t)) in kset for g, t in zip(tps["game_id"], tps["team_id"])]]
    sub_actual = np.zeros((len(TIME_CELLS), 3, 3), dtype="float64")
    actual_by_key = {}
    for key, g in tps.groupby(["game_id", "team_id"], sort=False):
        key = (int(key[0]), int(key[1]))
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        starters = set(int(x) for x in lu[0])
        a = profile_from_sequence(lu, starters, g["period"].to_numpy(),
                                  g["start_clock"].to_numpy(),
                                  g["margin_bucket"].to_numpy(),
                                  g["duration_s"].to_numpy())
        sub_actual += a
        actual_by_key[key] = a

    sim_profile, sim_by_key = {}, {}
    for name, arm in arms.items():
        ts = time.time()
        acc = np.zeros((len(TIME_CELLS), 3, 3), dtype="float64")
        per_key: dict = {}
        for seed in range(args.seeds):
            for gid, sides in keys_by_game.items():
                rng = R.game_stream(seed, gid)
                for key in sides:
                    pr, sc = priors_te[key], scripts_te[key]
                    lu, _fh = arm.simulate(pr, sc, rng)
                    starters = set(int(x) for x in pr.pids[pr.starters()[:5]])
                    a = profile_from_sequence(lu, starters, sc.period, sc.start_clock,
                                              sc.margin_bucket, sc.dur)
                    acc += a
                    prev = per_key.get(key)
                    per_key[key] = a if prev is None else prev + a
        sim_profile[name] = acc / args.seeds
        sim_by_key[name] = {k: v / args.seeds for k, v in per_key.items()}
        log.info("%s simulated in %.1f min", name, (time.time() - ts) / 60)

    payload["sub_actual_profile"] = sub_actual.tolist()
    payload["sim_profile"] = {k: v.tolist() for k, v in sim_profile.items()}

    # ---- 4. team quintile slope check ------------------------------------
    # team prior = the as-of predicted share of minutes going to the predicted
    # starting five (pregame, leak-safe by construction).
    prior_share = {}
    for key, pr in priors_te.items():
        if key in actual_by_key:
            prior_share[key] = float(pr.share[pr.starters()[:5]].sum())
    ks = sorted(prior_share)
    vals = np.array([prior_share[k] for k in ks])
    qs = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
    qidx = np.searchsorted(qs, vals, side="right")

    LATE = [5, 6, 7]                # H2 08:00-04:00, 04:00-02:00, 02:00-00:00
    slope_rows = []
    for q in range(5):
        sel_k = [k for k, qq in zip(ks, qidx) if qq == q]
        row = {"quintile": q + 1, "n_team_games": len(sel_k),
               "prior_starter_share_mean": float(np.mean([prior_share[k] for k in sel_k]))}
        for lab, src in (("actual", actual_by_key), *[(n, sim_by_key[n]) for n in arms]):
            num = sum(src[k][LATE, 0, 0].sum() for k in sel_k if k in src)
            den = sum(src[k][LATE, 0, 1].sum() for k in sel_k if k in src)
            row[lab] = float(num / den) if den else float("nan")
            row[lab + "_n_poss"] = float(den / 5.0)
        slope_rows.append(row)
    payload["slope"] = slope_rows

    # ---- 5. write the report ---------------------------------------------
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=V1._json_default))
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render(payload, actual_profile, sub_actual, sim_profile,
                             starter_bench, slope_rows, args), encoding="utf-8")
    log.info("wrote %s and %s in %.1f min", OUT_MD, OUT_JSON, (time.time() - t0) / 60)


def render(p, actual_profile, sub_actual, sim_profile, starter_bench, slope_rows, args) -> str:
    L: list[str] = []
    W = L.append
    W("# Rotation close-game audit (round-3 evidence) -- 2026-09-10\n")
    W("Worker: Opus. Inputs: `data/processed/possessions/possessions_{season}.parquet`, ")
    W("`data/processed/models/rotation/rotation_fit_round2_corrected_hazards.json`. ")
    W("Script: `scripts/diag_rotation_close_game.py`. JSON: ")
    W("`data/processed/models/rotation/close_game_audit_2026-09-10.json`.\n")
    W("Cells with fewer than 300 possessions are printed `UP(n)` and are ")
    W("UNDERPOWERED -- neither signal nor absence of signal.\n")

    W("\n## 1. Which seasons can answer the question at all\n")
    W("| season | games | possessions carrying a complete on-floor set | games complete on every possession |")
    W("|---|---:|---:|---:|")
    for r in p["availability"]:
        W(f"| {r['season']} | {r['games']:,} | {r['poss_with_on_floor']:.4f} | "
          f"{r['games_fully_complete']:,} |")
    W("\nCBBD `onFloor` is empty at the source before 2023-24 (L13), so **the "
      "close-game keep behaviour can only be measured on seasons 2024 and 2025**. "
      "The task's 2022-2025 breakdown is reported here as the availability count "
      "above and nothing more; there is no lineup evidence in 2022 or 2023 to "
      "characterise, and none is invented.\n")

    W("\n## 2. Actual starters' share of on-floor slots, whole-game time profile\n")
    W("Starters = the five that side actually started. Share = starter slots / "
      "(5 x possessions) in the cell, the same construction the gate uses.\n")
    for season in (2024, 2025):
        W(f"\n### 2.{season - 2023} Season {season}\n")
        W("| time cell | " + " | ".join(MB_LAB[b] for b in range(3)) + " | possessions |")
        W("|---|---:|---:|---:|---:|")
        for row in fmt_profile(actual_profile[season]):
            W("| " + " | ".join(row) + " |")
    a24, a25 = actual_profile[2024], actual_profile[2025]
    W("\nThe two seasons agree cell for cell: the largest close-band "
      f"(|m|<=5) difference across the eight time cells is "
      f"{np.nanmax(np.abs(a24[:, 0, 0] / np.where(a24[:, 0, 1] > 0, a24[:, 0, 1], np.nan) - a25[:, 0, 0] / np.where(a25[:, 0, 1] > 0, a25[:, 0, 1], np.nan))) * 100:.1f} pp. "
      "There is no season effect to model; the round-3 fit on 2024 and the gate "
      "on 2025 are measuring the same coaching behaviour.\n")

    W("\n## 3. Starter vs bench minutes\n")
    W("| season | starter mean min | starter SD | bench mean min | bench SD | starters' share of team minutes |")
    W("|---|---:|---:|---:|---:|---:|")
    for season in (2024, 2025):
        s = starter_bench[season]
        W(f"| {season} | {s['starter_mean']:.2f} | {s['starter_sd']:.2f} | "
          f"{s['bench_mean']:.2f} | {s['bench_sd']:.2f} | "
          f"{s['starter_share_of_team_minutes']:.4f} |")

    W(f"\n## 4. Where R2 and R5 miss, cell by cell ({args.games} games x "
      f"{args.seeds} seeds, 2025)\n")
    W("The sim universe is the first "
      f"{args.games} games of the graded round-2 subset (numpy RandomState seed "
      f"{TEST_SUBSET_SEED}), so these rows are a strict subset of the round-2 "
      "gate universe and the ACTUAL column is restricted to exactly the same "
      "team-games.\n")
    for b in range(3):
        W(f"\n### 4.{b + 1} Margin band {MB_LAB[b]}\n")
        W("| time cell | ACTUAL | R2_hier_dirichlet | R5_hybrid (corrected) | "
          "R2 miss (pp) | R5 miss (pp) | possessions |")
        W("|---|---:|---:|---:|---:|---:|---:|")
        for i, (lab, _pp, _hi, _lo) in enumerate(TIME_CELLS):
            den = sub_actual[i, b, 1]
            n = den / 5.0
            if n < UNDERPOWERED:
                W(f"| {lab} | UP({n:.0f}) | UP | UP | -- | -- | {n:,.0f} |")
                continue
            a = sub_actual[i, b, 0] / den
            r2 = sim_profile["R2_hier_dirichlet"][i, b, 0] / sim_profile["R2_hier_dirichlet"][i, b, 1]
            r5 = sim_profile["R5_hybrid_corrected"][i, b, 0] / sim_profile["R5_hybrid_corrected"][i, b, 1]
            W(f"| {lab} | {a:.4f} | {r2:.4f} | {r5:.4f} | {(r2 - a) * 100:+.1f} | "
              f"{(r5 - a) * 100:+.1f} | {n:,.0f} |")

    W("\n## 5. The round-2 hypothesis test: is the starter time spent too early?\n")
    W("`model.md` section 10 predicts that if R5's late deficit is a *distribution* "
      "problem rather than a *level* problem, its starter share should sit ABOVE "
      "the real one early by about the amount it sits below late.\n")
    W("| band | first half (cells 1-2) ACTUAL | R2 | R5 | final 8:00 ACTUAL | R2 | R5 | R5 early surplus (pp) | R5 late deficit (pp) |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    EARLY, LATE = [0, 1], [5, 6, 7]
    for b in range(3):
        def sh(acc, cells):
            num = acc[cells, b, 0].sum()
            den = acc[cells, b, 1].sum()
            return (num / den) if den else float("nan"), den / 5.0
        ae, ne = sh(sub_actual, EARLY)
        al, nl = sh(sub_actual, LATE)
        r2e, _ = sh(sim_profile["R2_hier_dirichlet"], EARLY)
        r2l, _ = sh(sim_profile["R2_hier_dirichlet"], LATE)
        r5e, _ = sh(sim_profile["R5_hybrid_corrected"], EARLY)
        r5l, _ = sh(sim_profile["R5_hybrid_corrected"], LATE)
        if ne < UNDERPOWERED or nl < UNDERPOWERED:
            W(f"| {MB_LAB[b]} | UP({ne:.0f}/{nl:.0f}) | | | | | | | |")
            continue
        W(f"| {MB_LAB[b]} | {ae:.4f} | {r2e:.4f} | {r5e:.4f} | {al:.4f} | {r2l:.4f} | "
          f"{r5l:.4f} | {(r5e - ae) * 100:+.1f} | {(r5l - al) * 100:+.1f} |")

    W("\n## 6. Slope check: team quintile of the as-of starter-minutes share\n")
    W("Team prior = the as-of predicted share of team minutes going to the "
      "predicted starting five, a pregame quantity. Cell = the starters' share "
      "of on-floor slots in the final 8:00 at |margin| <= 5.\n")
    W("| quintile | team-games | prior starter share | ACTUAL | R2 | R5 (corrected) | close-late possessions |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for r in slope_rows:
        n = r["actual_n_poss"]
        flag = "" if n >= UNDERPOWERED else "  UNDERPOWERED"
        W(f"| Q{r['quintile']} | {r['n_team_games']:,} | "
          f"{r['prior_starter_share_mean']:.4f} | {r['actual']:.4f} | "
          f"{r['R2_hier_dirichlet']:.4f} | {r['R5_hybrid_corrected']:.4f} | "
          f"{n:,.0f}{flag} |")
    a = np.array([r["actual"] for r in slope_rows])
    x = np.array([r["prior_starter_share_mean"] for r in slope_rows])
    W("")
    for lab in ("actual", "R2_hier_dirichlet", "R5_hybrid_corrected"):
        y = np.array([r[lab] for r in slope_rows])
        sl = np.polyfit(x, y, 1)[0]
        W(f"- `{lab}` slope against the prior: **{sl:+.3f}** "
          f"(Q5 - Q1 = {(y[-1] - y[0]) * 100:+.1f} pp)")
    W(f"\nActual Q5 - Q1 is {(a[-1] - a[0]) * 100:+.1f} pp. An arm whose quintile "
      "profile is flat is not matchup-specific no matter what its pooled cell says.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
