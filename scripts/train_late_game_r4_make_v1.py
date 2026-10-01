"""train_late_game_r4_make_v1.py -- late-game ROUND 4 (a2): the buzzer make law (experiments.md 9.2).

First-chance field-goal attempts in periods 1-2 with <= 1 s left AT THE SHOT (chance end clock in the data;
the engine computes possession-start clock minus the fed `chance_elapsed_s`). One grader, both folds.

    MK0  make rate by class x period over ALL first-chance shots (no buzzer term)
    MK1  class x period, fitted on the <= 1 s rows (shrunk to MK0's cell, m = 50)
    MK2  class x period x {0 s, 1 s} (shrunk to MK1's cell, m = 50)

Metric: log loss on the test fold's <= 1 s rows; floor = game-block bootstrap SE of the paired delta.
Also reported: realised <= 1 s make by offence prior-season make-rate quintile (responsiveness, data side).

    .venv/Scripts/python.exe scripts/train_late_game_r4_make_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from grade_late_game_v1 import block_se                              # noqa: E402

CH = ROOT / "data/processed/possessions_v4/chances_{}.parquet"
OUT = ROOT / "data/processed/models/late_game/round4"
FOLDS = {"F1": ([2022, 2023], 2024), "F2": ([2022, 2023, 2024], 2025)}
CLS = {"FGA_rim": ("fga_rim", "fgm_rim", 0), "FGA_jump2": ("fga_jump2", "fgm_jump2", 1),
       "FGA_3": ("fga_3", "fgm_3", 2)}
M = 50.0


def load(seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        c = pd.read_parquet(str(CH).format(s))
        c = c[(c["period"] <= 2) & (c["chance_number"] == 1) & c["terminal_event"].isin(list(CLS))]
        for ev, (a, m, k) in CLS.items():
            x = c[c["terminal_event"] == ev]
            rows.append(pd.DataFrame({"game_id": x["game_id"].to_numpy(), "season": s,
                                      "off": x["offense_team_id"].to_numpy(),
                                      "per": x["period"].to_numpy() - 1, "cls": k,
                                      "tl": x["end_clock"].to_numpy(),
                                      "y": (x[m] > 0).astype(float).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def fit(tr: pd.DataFrame) -> dict:
    k0 = tr["cls"] * 2 + tr["per"]
    n0 = np.bincount(k0, minlength=6).astype(float)
    r0 = np.bincount(k0, weights=tr["y"], minlength=6) / np.maximum(n0, 1)
    b = tr[tr["tl"] <= 1]
    k1 = b["cls"] * 2 + b["per"]
    n1 = np.bincount(k1, minlength=6).astype(float)
    e1 = np.bincount(k1, weights=b["y"], minlength=6)
    r1 = (e1 + M * r0) / (n1 + M)
    k2 = k1 * 2 + b["tl"].clip(0, 1)
    n2 = np.bincount(k2, minlength=12).astype(float)
    e2 = np.bincount(k2, weights=b["y"], minlength=12)
    r2 = (e2 + M * np.repeat(r1, 2)) / (n2 + M)
    return {"MK0": r0.tolist(), "MK1": r1.tolist(), "MK2": r2.tolist(),
            "n_buzzer_train": int(len(b)), "index": "cls*2+per (cls 0 rim,1 jump2,2 three; per 0 H1,1 H2); MK2: *2+tl"}


def predict(lut: dict, arm: str, d: pd.DataFrame) -> np.ndarray:
    k = (d["cls"] * 2 + d["per"]).to_numpy()
    if arm == "MK2":
        k = k * 2 + d["tl"].clip(0, 1).to_numpy()
    return np.asarray(lut[arm])[k]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for fold, (trs, tes) in FOLDS.items():
        tr, te = load(trs), load([tes])
        lut = fit(tr)
        b = te[te["tl"] <= 1].reset_index(drop=True)
        y, g = b["y"].to_numpy(), b["game_id"].to_numpy()
        ll = {}
        for arm in ("MK0", "MK1", "MK2"):
            p = np.clip(predict(lut, arm, b), 1e-6, 1 - 1e-6)
            ll[arm] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        rows = {}
        for arm in ll:
            d = ll[arm] - ll["MK0"]
            se = block_se(d, g) if arm != "MK0" else None
            rows[arm] = {"logloss": float(ll[arm].mean()), "d": float(d.mean()), "se": se,
                         "floors_vs_MK0": float(-d.mean() / se) if se else None}
        d = ll["MK2"] - ll["MK1"]
        se = block_se(d, g)
        rows["MK2_vs_MK1"] = {"d": float(d.mean()), "se": se, "floors": float(-d.mean() / se)}
        cal = []
        for c in range(3):
            for pr in range(2):
                m = (b["cls"] == c) & (b["per"] == pr)
                cal.append({"cls": c, "per": pr + 1, "n": int(m.sum()), "actual": float(y[m].mean()),
                            "MK0": float(predict(lut, "MK0", b[m]).mean()), "MK1": float(predict(lut, "MK1", b[m]).mean())})
        # responsiveness, data side: offence prior-season make rate (all first-chance shots) quintile
        prior = tr[tr["season"] == max(trs)].groupby("off")["y"].mean()
        b["q"] = pd.qcut(b["off"].map(prior).rank(method="first"), 5, labels=False)
        resp = b.groupby("q")["y"].agg(["size", "mean"]).reset_index().to_dict("records")
        rep[fold] = {"n_train_buzzer": lut["n_buzzer_train"], "n_test_buzzer": int(len(b)), "arms": rows,
                     "calibration": cal, "responsiveness_offence_prior_quintile": resp}
        if fold == "F2":
            (OUT / "make_F2.json").write_text(json.dumps(lut), encoding="utf-8")
    (OUT / "make_grade.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for f, r in rep.items():
        print(f"== {f}: buzzer train {r['n_train_buzzer']:,} test {r['n_test_buzzer']:,}")
        for k, v in r["arms"].items():
            print(f"   {k:10s} {v}")
        for c in r["calibration"]:
            print(f"   {c}")
        print("   resp:", r["responsiveness_offence_prior_quintile"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
