"""
diag_clock_v1.py -- the DIAGNOSIS the L5 pre-registration demands when no arm
passes the emergent G1 ("report the failure and the diagnosis; adopt nothing").

Run AFTER `scripts/train_clock_v1.py`; it reuses that run's cached
`design.parquet` and refits the same arms with the same seed, so every number
here is comparable to the bake-off table row by row.

Three questions, each with its own table:

  D1  WHERE does the conditional mean go wrong?
      Actual mean duration by seconds-remaining band vs each arm's predicted
      mean on the same rows. The emergent possession count is 2400 / (mean
      duration realised over the chain), so a mean that is wrong only in the
      last 45 seconds of a period still moves the count.

  D2  WHY is the per-game count SD too narrow?
      Decomposes the actual per-game possession count into the part a
      state-conditional i.i.d. duration model can produce and the part it
      cannot. Compares the sim's own count SD against the actual, and against
      the SD of the model's own per-game mean prediction (the between-game
      spread the FEATURES carry).

  D3  RESPONSIVENESS (`CLAUDE.md` standing rule: matchup-specific, not
      league-average). Emergent possessions per game, sim vs actual, bucketed
      by the pregame tempo-prior quintile. A model that sat flat at the league
      mean would fail this even with a perfect overall mean.

It also persists the two arms the next iteration will want to diff against --
the lowest-CRPS arm and the best-calibrated arm -- as
`reference_not_adopted_{arm}.pkl` with `adopted: False` in the payload, the
same convention `possession_outcome` uses for a bake-off that adopted nothing.
Nothing here is wired into the engine.

Artifacts -> `data/processed/models/clock/`
    diagnosis_by_clock_band_F2.csv
    diagnosis_count_variance_F2.json
    diagnosis_responsiveness_F2.csv
    reference_not_adopted_{arm}.pkl
Appends a "diagnosis" section to `docs/models/clock/experiments.md`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.models import clock as ck

OUT_DIR = Path("data/processed/models/clock")
DOC = Path("docs/models/clock/experiments.md")
BASE_SEED = 20260910

#: Fine bands through the end-of-period region, coarse beyond it. The point of
#: the table is that every arm smooths a curve the data bends sharply.
BANDS: tuple[tuple[int, int], ...] = (
    (0, 10), (10, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45),
    (45, 50), (50, 60), (60, 90), (90, 1201),
)


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def band_labels(sr: np.ndarray) -> np.ndarray:
    out = np.empty(len(sr), dtype=object)
    for lo, hi in BANDS:
        out[(sr >= lo) & (sr < hi)] = f"{lo}-{hi - 1}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--feature-set", default="C_plus_score")
    args = ap.parse_args()

    design = pd.read_parquet(OUT_DIR / "design.parquet")
    tr, te = ck.fold_slices(design, ck.SELECTION_FOLD)
    reg = te[te["period"] <= 2.0]
    _log(f"F2 train {len(tr):,} / test {len(te):,} rows; regulation test rows {len(reg):,}")

    arms = {}
    for name in ck.ARMS:
        t0 = time.time()
        arms[name] = ck.fit_arm(name, tr, args.feature_set, seed=args.seed)
        _log(f"refit {name} / {args.feature_set} in {time.time() - t0:.1f}s")

    # ---------------- D1: conditional mean by clock band -------------------
    ok = ~reg["censored"].to_numpy(dtype=bool)
    sr = reg["seconds_remaining"].to_numpy(dtype="float64")
    band = band_labels(sr)
    y = reg["duration_s"].to_numpy(dtype="float64")
    ran_out = y >= sr - 0.5

    rows = []
    preds = {}
    for name, arm in arms.items():
        m = np.empty(len(reg))
        for start in range(0, len(reg), 100_000):
            stop = min(start + 100_000, len(reg))
            m[start:stop] = arm.pmf(reg.iloc[start:stop]) @ ck.GRID.astype("float64")
        preds[name] = m

    for lo, hi in BANDS:
        lab = f"{lo}-{hi - 1}"
        sel = (band == lab) & ok
        n = int(sel.sum())
        if n == 0:
            continue
        r = {"band_seconds_remaining": lab, "n": n,
             "actual_mean": round(float(y[sel].mean()), 3),
             "actual_median": float(np.median(y[sel])),
             "pct_clock_ran_out": round(100.0 * float(ran_out[(band == lab)].mean()), 2),
             "pct_censored": round(100.0 * float(reg["censored"].to_numpy()[(band == lab)].mean()), 2)}
        for name in arms:
            r[f"{name}_mean"] = round(float(preds[name][sel].mean()), 3)
            r[f"{name}_gap"] = round(float(preds[name][sel].mean() - y[sel].mean()), 3)
        rows.append(r)
    d1 = pd.DataFrame(rows)
    d1.to_csv(OUT_DIR / "diagnosis_by_clock_band_F2.csv", index=False)
    _log("D1 written")

    # ---------------- D2: where the count variance goes --------------------
    per_game_actual = reg.groupby("game_id").size() / 2.0
    chains = {name: ck.chain_halves(arm, te, seed=args.seed) for name, arm in arms.items()}
    d2: dict = {
        "actual_mean": round(float(per_game_actual.mean()), 4),
        "actual_sd": round(float(per_game_actual.std(ddof=1)), 4),
        "n_games": int(len(per_game_actual)),
        "arms": {},
    }
    # The between-game spread the FEATURES carry: each game's model-implied
    # mean duration, turned into a count. Anything the actual SD has beyond
    # this has to come from somewhere the model cannot see.
    gm = pd.DataFrame({"game_id": reg["game_id"].to_numpy()})
    for name in arms:
        gm[name] = preds[name]
    game_mean = gm.groupby("game_id").mean()
    for name in arms:
        implied = 2400.0 / (2.0 * game_mean[name].to_numpy())
        pg = chains[name].per_game
        d2["arms"][name] = {
            "sim_mean": round(float(pg["sim_poss"].mean()), 4),
            "sim_sd": round(float(pg["sim_poss"].std(ddof=1)), 4),
            "sd_delta": round(float(pg["sim_poss"].std(ddof=1) - per_game_actual.std(ddof=1)), 4),
            "feature_implied_count_sd": round(float(implied.std(ddof=1)), 4),
            "corr_sim_actual": round(float(np.corrcoef(pg["sim_poss"], pg["actual_poss"])[0, 1]), 4),
        }
    # A pure renewal reference: how much count SD you get from i.i.d. draws
    # alone, with NO between-game variation at all.
    mu = float(y[ok].mean())
    sd = float(y[ok].std(ddof=1))
    d2["iid_renewal_count_sd_reference"] = round(float(np.sqrt(2400.0 / mu) * sd / mu / 2.0), 4)
    d2["note"] = ("iid_renewal_count_sd_reference is sqrt(N) * sd(D) / mean(D) / 2 for "
                  "N = 2400/mean(D) draws per game, i.e. the per-team count SD produced by "
                  "identically-distributed possessions with no game-to-game pace variation at all.")
    (OUT_DIR / "diagnosis_count_variance_F2.json").write_text(json.dumps(d2, indent=2))
    _log("D2 written")

    # ---------------- D3: responsiveness -----------------------------------
    tempo = te.groupby("game_id")["tempo_prior_game"].first()
    q = pd.qcut(tempo, 5, labels=[1, 2, 3, 4, 5])
    rows = []
    for name in arms:
        pg = chains[name].per_game.set_index("game_id")
        j = pd.DataFrame({"tempo_q": q.reindex(pg.index), "sim": pg["sim_poss"],
                          "actual": pg["actual_poss"]}).dropna()
        for qq, g in j.groupby("tempo_q", observed=True):
            rows.append({"arm": name, "tempo_quintile": int(qq), "n": int(len(g)),
                         "sim_mean": round(float(g["sim"].mean()), 3),
                         "actual_mean": round(float(g["actual"].mean()), 3),
                         "delta": round(float(g["sim"].mean() - g["actual"].mean()), 3)})
    d3 = pd.DataFrame(rows)
    spans = []
    for name, g in d3.groupby("arm"):
        g = g.sort_values("tempo_quintile")
        sp_s = float(g["sim_mean"].iloc[-1] - g["sim_mean"].iloc[0])
        sp_a = float(g["actual_mean"].iloc[-1] - g["actual_mean"].iloc[0])
        steps = int(np.sum(np.sign(np.diff(g["sim_mean"])) == np.sign(np.diff(g["actual_mean"]))))
        spans.append({"arm": name, "span_sim": round(sp_s, 3), "span_actual": round(sp_a, 3),
                      "slope_ratio": round(sp_s / sp_a, 4) if sp_a else None,
                      "steps_agreeing": steps, "n_steps": 4})
    d3 = d3.merge(pd.DataFrame(spans), on="arm")
    d3.to_csv(OUT_DIR / "diagnosis_responsiveness_F2.csv", index=False)
    _log("D3 written")

    # ---------------- reference artifacts, explicitly NOT adopted ----------
    grid = pd.read_csv(OUT_DIR / "grid_results.csv")
    f2 = grid[(grid["fold"] == ck.SELECTION_FOLD) & (grid["feature_set"] == args.feature_set)]
    keep = {str(f2.loc[f2["crps"].idxmin(), "arm"]),
            str(f2.loc[f2["pit_leak_failures"].idxmin(), "arm"])}
    for name in sorted(keep):
        payload = {
            "adopted": False,
            "why": "the L5 bake-off adopted nothing; see docs/models/clock/model.md section 4",
            "arm": name, "feature_set": args.feature_set, "fold": ck.SELECTION_FOLD,
            "seed": args.seed, "features": ck.arm_features(arms[name]), "model": arms[name],
        }
        with open(OUT_DIR / f"reference_not_adopted_{name}.pkl", "wb") as fh:
            pickle.dump(payload, fh)
        _log(f"persisted reference_not_adopted_{name}.pkl (adopted=False)")

    write_section(d1, d2, d3, args)
    _log("done")


def _md(df: pd.DataFrame, cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_section(d1, d2, d3, args) -> None:
    from datetime import UTC, datetime
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    lines: list[str] = ["", f"## 3. Run R2 -- diagnosis of the R1 failure ({stamp})", "",
                        f"`scripts/diag_clock_v1.py`, seed {args.seed}, feature set "
                        f"`{args.feature_set}` for every arm (the bake-off's own refit, same seed, "
                        "so these rows are comparable to section 2 row by row).", ""]
    a = lines.append
    a("### 3.1 D1 -- the conditional mean of duration by seconds remaining")
    a("")
    cols = ["band_seconds_remaining", "n", "actual_mean", "pct_clock_ran_out"] + \
           [f"{k}_mean" for k in ck.ARMS]
    a(_md(d1, cols))
    a("")
    a("Read: the actual conditional mean bends sharply inside the last 45 seconds of a period "
      "(17.9 s with more than 90 s left, down to about 3 s with fewer than 10). Every arm smooths "
      "that bend -- the empirical arm because its finest pre-registered seconds-remaining bucket is "
      "0-34, the parametric and hazard arms because `seconds_remaining` enters them linearly. The "
      "emergent possession count is 2400 divided by the mean duration REALISED over the chain, so a "
      "conditional mean that is only wrong in the last 45 seconds of each half still moves the "
      "count, and it moves the end-of-half check directly.")
    a("")
    a("### 3.2 D2 -- where the per-game count variance goes")
    a("")
    a("```json")
    a(json.dumps(d2, indent=2))
    a("```")
    a("")
    a("Read: `iid_renewal_count_sd_reference` is the per-team count SD you get from "
      "identically-distributed possessions with NO game-to-game pace variation. "
      "`feature_implied_count_sd` is the extra between-game spread this model's own features carry. "
      "The two together are what any state-conditional i.i.d. duration model can produce, and the "
      "gap to `actual_sd` is the part of real pace dispersion that is neither in the features nor in "
      "possession-level noise -- the same shape as L10 one layer down: independent draws understate "
      "dispersion.")
    a("")
    a("### 3.3 D3 -- responsiveness (CLAUDE.md standing rule)")
    a("")
    a(_md(d3[["arm", "tempo_quintile", "n", "sim_mean", "actual_mean", "delta", "span_sim",
              "span_actual", "slope_ratio", "steps_agreeing"]],
          ["arm", "tempo_quintile", "n", "sim_mean", "actual_mean", "delta", "span_sim",
           "span_actual", "slope_ratio", "steps_agreeing"]))
    a("")
    a("Read: the emergent count must SLOPE with the pregame tempo prior, not sit flat at the league "
      "mean. `slope_ratio` near 1 with 4 of 4 agreeing steps says the model is matchup-specific; a "
      "ratio near 0 would say it is a league average wearing a feature vector.")
    a("")
    with DOC.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
