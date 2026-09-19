#!/usr/bin/env python
"""
train_foul_bonus_cond_v1.py -- Block T of possession-outcome round 6: the
CONDITIONAL bonus-trip rate given the offence is already in the bonus.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 13
Block T, as redefined by section 15.3 (AMENDMENT A, committed before fitting):
the object is scored as its own binary -- `P(FT_trip_bonus | off_in_bonus)` --
not as the six-class log loss, and it is fitted INDEPENDENTLY of Block F so
neither absorbs the other's error.

    .venv/Scripts/python.exe scripts/train_foul_bonus_cond_v1.py --seed 0

Rows: the served round-2 design (`.../round2/design.parquet`, first_chance
style source, possessions v2 -- the design the SERVED `round2_s1` arm was fitted
on), restricted to `in_bonus == 1`, ANTI-JOINED against the verified technical
free-throw trips (section 15.4) so that mis-tagged technical attempts cannot be
counted as bonus trips. Both populations are pooled with an `is_cont` flag, a
declared cost deviation from the served two-population fit.

Arms (13.1 Block T): T0 reference = the served `C_plus_state` bundle refitted as
a binary; T2 adds the raw team-foul COUNTS (the served model sees only the
binary `in_bonus` and cannot tell 7 fouls from 11); T1 adds the as-of
league-centred defence foul rate and offence drawn-foul rate; T3 = T1 + T2.
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

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

ROUND_DIR = Path("data/processed/models/possession_outcome/round6")
DESIGN = Path("data/processed/models/possession_outcome/round2/design.parquet")
TECH = Path("data/processed/models/free_throw/technical_target_verified_trips_v1.parquet")
FOLDS = PO.FOLDS
TECH_CLOCK_TOL = 25          # seconds; fixed before the run, not tuned


def anti_join_technicals(ch: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop FT-trip chances that the verified technical table says are really
    technical free throws (section 15.4). Matched on (season, game_id, period)
    with the technical's clock inside the chance's own clock window widened by
    `TECH_CLOCK_TOL` seconds."""
    if not TECH.exists():
        return ch, {"status": "TECHNICAL TABLE MISSING -- NOT anti-joined"}
    t = pd.read_parquet(TECH)
    trips = ch["terminal_event"].isin(["FT_trip_bonus", "FT_trip_shooting"])
    cand = ch[trips][["season", "game_id", "period", "start_clock", "end_clock"]].copy()
    cand["row"] = cand.index
    m = cand.merge(t[["season", "game_id", "period", "clock", "n_attempts_verified"]],
                   on=["season", "game_id", "period"], how="inner")
    hit = m[(m["clock"] <= m["start_clock"] + TECH_CLOCK_TOL)
            & (m["clock"] >= m["end_clock"] - TECH_CLOCK_TOL)]
    drop = set(hit["row"].tolist())
    rep = {"n_verified_technicals": int(len(t)),
           "n_trip_chances": int(trips.sum()),
           "n_chances_dropped": len(drop),
           "n_bonus_trips_before": int((ch["terminal_event"] == "FT_trip_bonus").sum()),
           "clock_tolerance_s": TECH_CLOCK_TOL}
    out = ch.drop(index=list(drop))
    rep["n_bonus_trips_after"] = int((out["terminal_event"] == "FT_trip_bonus").sum())
    rep["bonus_trip_rate_before"] = round(float(
        (ch["terminal_event"] == "FT_trip_bonus").mean()), 6)
    rep["bonus_trip_rate_after"] = round(float(
        (out["terminal_event"] == "FT_trip_bonus").mean()), 6)
    return out, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ch = pd.read_parquet(DESIGN)
    ch_clean, tech_rep = anti_join_technicals(ch)
    print(json.dumps(tech_rep, indent=1))

    # the accrual design carries the open-time foul COUNTS and the as-of rates
    acc = pd.read_parquet(ROUND_DIR / "foul_accrual_poss_v1.parquet",
                          columns=["game_id", "season", "poss_index",
                                   "def_team_fouls", "off_team_fouls"])
    ch_clean = ch_clean.merge(acc, on=["game_id", "season", "poss_index"], how="left")

    from train_foul_accrual_v1 import build_features  # noqa: PLC0415
    des = pd.read_parquet(ROUND_DIR / "foul_accrual_poss_v1.parquet")
    des = build_features(des)
    keys = des[["season", "game_id", "poss_index", "def_foul_c", "off_drawn_c",
                "is_conf_game"]]
    ch_clean = ch_clean.merge(keys, on=["season", "game_id", "poss_index"], how="left")
    for c in ("def_foul_c", "off_drawn_c", "def_team_fouls", "off_team_fouls"):
        ch_clean[c] = ch_clean[c].astype("float32").fillna(0.0)

    d = ch_clean[ch_clean["in_bonus"] == 1].copy()
    d["y_bonus"] = (d["terminal_event"] == "FT_trip_bonus").astype("int8")
    d["is_cont"] = (d["population"] == "cont").astype("float32")
    d["in_fit_window"] = ((d["period"] <= 2) & (d["start_clock"] > 120)).astype("int8")
    per = d["period"].to_numpy()
    sc = d["start_clock"].to_numpy()
    d["game_minute"] = np.where(per == 1, (1200 - sc) / 60.0,
                                np.where(per == 2, (2400 - sc) / 60.0,
                                         (2400 + (per - 2) * 300 - sc) / 60.0))
    d["days_since_start"] = d["days_since_start"].astype("float32")

    base = PO.feature_set("C_plus_state", "cont") + ["is_cont"]
    T0 = [c for c in base if c in d.columns]
    T2 = T0 + ["def_team_fouls", "off_team_fouls"]
    T1 = T0 + ["def_foul_c", "off_drawn_c"]
    T3 = T0 + ["def_team_fouls", "off_team_fouls", "def_foul_c", "off_drawn_c"]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    for fold, spec in FOLDS.items():
        assert_not_sealed(spec["train"], context=f"{fold} train")
        assert_not_sealed(spec["test"], context=f"{fold} test")
        tr = d[d["season"].isin(spec["train"]) & (d["in_fit_window"] == 1)]
        te = d[d["season"].isin(spec["test"])].copy()
        out = te[["game_id", "season", "period", "poss_index", "offense_team_id",
                  "defense_team_id", "in_fit_window", "game_minute", "site_home",
                  "site_away", "is_conf_game", "days_since_start", "def_team_fouls",
                  "y_bonus"]].copy()
        out["sec_rem"] = te["start_clock"].to_numpy()
        out["off_in_bonus"] = 1
        for name, feats in (("T0", T0), ("T1", T1), ("T2", T2), ("T3", T3)):
            sc_ = StandardScaler()
            X = sc_.fit_transform(tr[feats].to_numpy(dtype="float64"))
            m = LogisticRegression(C=1.0, max_iter=300, random_state=a.seed)
            m.fit(X, tr["y_bonus"].to_numpy())
            out[name] = m.predict_proba(sc_.transform(te[feats].to_numpy(dtype="float64")))[:, 1]
        out.to_parquet(ROUND_DIR / f"preds_y_bonus_{fold}_seed{a.seed}.parquet", index=False)
        print(f"  {fold}: n_train {len(tr):,} n_test {len(te):,} "
              f"base {tr['y_bonus'].mean():.6f} test {te['y_bonus'].mean():.6f}")

    # cleaned TRUTH curves the PM asked for
    ch_clean["game_minute"] = np.where(
        ch_clean["period"] == 1, (1200 - ch_clean["start_clock"]) / 60.0,
        np.where(ch_clean["period"] == 2, (2400 - ch_clean["start_clock"]) / 60.0,
                 (2400 + (ch_clean["period"] - 2) * 300 - ch_clean["start_clock"]) / 60.0))
    ch["game_minute"] = np.where(
        ch["period"] == 1, (1200 - ch["start_clock"]) / 60.0,
        np.where(ch["period"] == 2, (2400 - ch["start_clock"]) / 60.0,
                 (2400 + (ch["period"] - 2) * 300 - ch["start_clock"]) / 60.0))
    edges = [0, 5, 10, 15, 20, 25, 30, 35, 38, 45]
    rows = []
    for label, frame in (("raw", ch), ("cleaned", ch_clean)):
        f = frame[(frame["season"] == 2025) & (frame["period"] <= 2)].copy()
        f["b"] = pd.cut(f["game_minute"], edges, right=False)
        g = f.groupby("b", observed=True).apply(
            lambda x: pd.Series({
                "n": len(x),
                "p_in_bonus": float(x["in_bonus"].mean()),
                "p_trip_given_bonus": float(
                    (x.loc[x["in_bonus"] == 1, "terminal_event"] == "FT_trip_bonus").mean())
                if (x["in_bonus"] == 1).any() else float("nan"),
            }), include_groups=False)
        for k, r in g.iterrows():
            rows.append({"which": label, "bucket": str(k), "n": int(r["n"]),
                         "p_in_bonus": round(r["p_in_bonus"], 5),
                         "p_trip_given_bonus": round(r["p_trip_given_bonus"], 5)})
    tech_rep["truth_curves_2025"] = rows
    tech_rep["seconds"] = round(time.time() - t0, 1)
    (ROUND_DIR / "technical_antijoin_report.json").write_text(json.dumps(tech_rep, indent=1))
    print(json.dumps({k: v for k, v in tech_rep.items() if k != "truth_curves_2025"}, indent=1))


if __name__ == "__main__":
    main()
