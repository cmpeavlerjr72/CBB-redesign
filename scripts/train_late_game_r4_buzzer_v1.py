"""train_late_game_r4_buzzer_v1.py -- late-game ROUND 4 (a1): the end-of-period NO-SHOT law (experiments.md 9).

The possession_outcome model drops `end_period` chances ("the clock model's job", possession_outcome.py) and
the engine never produces one: every possession, however little time is left, ends in a TOV / shot / FT
trip. This fits P(possession ends at the horn with no terminal event | possession-start state) on
`possessions_v4`, periods 1-2, possessions starting at <= 35 s, and grades every arm with ONE code path on
both folds. Deterministic cell laws: the reseed floor is zero; the floor is the game-block bootstrap SE of
the paired per-row log-loss delta.

Arms (hierarchical cell rates, each level shrunk to its parent with m = 50 pseudo-rows):
    BZ0  period                                   (no time term: the "is there a time effect" baseline)
    BZ1  period x start bucket
    BZ2  period x start bucket x role3            (role = sign of the offence's score_diff)
    BZ3  period x start bucket x role3 x start type (made / DREB / TOV / other)

    .venv/Scripts/python.exe scripts/train_late_game_r4_buzzer_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from grade_late_game_v1 import block_se                              # noqa: E402

POSS = ROOT / "data/processed/possessions_v4/possessions_{}.parquet"
OUT = ROOT / "data/processed/models/late_game/round4"
FOLDS = {"F1": ([2022, 2023], 2024), "F2": ([2022, 2023, 2024], 2025)}
EDGES = np.array([1, 3, 6, 10, 15, 20, 35])          # start-clock buckets (0,1] (1,3] ... (20,35]
M = 50.0
ARMS = {"BZ0": ("per",), "BZ1": ("per", "sb"), "BZ2": ("per", "sb", "role"),
        "BZ3": ("per", "sb", "role", "st")}
SIZES = {"per": 2, "sb": len(EDGES), "role": 3, "st": 4}
ST = {"made_FG": 0, "made_FT": 0, "DREB": 1, "TOV": 2}


def load(seasons) -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(str(POSS).format(s), columns=[
        "game_id", "season", "period", "start_clock", "start_score_diff", "start_reason",
        "terminal_event"]) for s in seasons])
    d = d[(d["period"] <= 2) & (d["start_clock"] <= 35) & (d["terminal_event"] != "unknown")]
    return d.reset_index(drop=True)


def codes(d: pd.DataFrame) -> dict:
    return {"per": (d["period"].to_numpy() - 1).astype(int),
            "sb": np.searchsorted(EDGES, d["start_clock"].to_numpy(), side="left").clip(0, len(EDGES) - 1),
            "role": (np.sign(d["start_score_diff"].to_numpy()) + 1).astype(int),
            "st": d["start_reason"].map(ST).fillna(3).astype(int).to_numpy()}


def key(c: dict, dims) -> np.ndarray:
    k = np.zeros(len(c["per"]), dtype=np.int64)
    for dm in dims:
        k = k * SIZES[dm] + c[dm]
    return k


def fit(tr: pd.DataFrame, dims) -> dict:
    """Hierarchical shrunk rates: level j cell -> (events + M * parent) / (n + M)."""
    c = codes(tr)
    y = (tr["terminal_event"].to_numpy() == "end_period").astype(float)
    base = y.mean()
    levels = []
    parent = np.full(1, base)
    for j in range(1, len(dims) + 1):
        kk = key(c, dims[:j])
        n_cells = int(np.prod([SIZES[x] for x in dims[:j]]))
        n = np.bincount(kk, minlength=n_cells).astype(float)
        e = np.bincount(kk, weights=y, minlength=n_cells)
        par = parent[np.arange(n_cells) // SIZES[dims[j - 1]]] if j > 1 else np.full(n_cells, base)
        rate = (e + M * par) / (n + M)
        levels.append(rate)
        parent = rate
    return {"dims": list(dims), "rate": levels[-1].tolist(), "base": float(base),
            "n_train": int(len(tr)), "edges": EDGES.tolist(), "m": M}


def predict(lut: dict, d: pd.DataFrame) -> np.ndarray:
    return np.asarray(lut["rate"])[key(codes(d), lut["dims"])]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for fold, (trs, tes) in FOLDS.items():
        tr, te = load(trs), load([tes])
        y = (te["terminal_event"].to_numpy() == "end_period").astype(float)
        g = te["game_id"].to_numpy()
        ll = {}
        luts = {}
        for arm, dims in ARMS.items():
            lut = fit(tr, dims)
            p = np.clip(predict(lut, te), 1e-6, 1 - 1e-6)
            ll[arm] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            luts[arm] = lut
        rows = {}
        for arm in ARMS:
            d = ll[arm] - ll["BZ0"]
            se = block_se(d, g) if arm != "BZ0" else None
            rows[arm] = {"logloss": float(ll[arm].mean()), "d_vs_BZ0": float(d.mean()), "se": se,
                         "floors_vs_BZ0": float(-d.mean() / se) if se else None}
        for a, b in (("BZ2", "BZ1"), ("BZ3", "BZ2")):
            d = ll[a] - ll[b]
            se = block_se(d, g)
            rows[f"{a}_vs_{b}"] = {"d": float(d.mean()), "se": se, "floors": float(-d.mean() / se)}
        # calibration by period x start bucket (and role in P2) for the reported table
        cal = []
        c = codes(te)
        for pr in (0, 1):
            for sb in range(len(EDGES)):
                m = (c["per"] == pr) & (c["sb"] == sb)
                cal.append({"period": pr + 1, "start_le": int(EDGES[sb]), "n": int(m.sum()),
                            "actual": float(y[m].mean()) if m.any() else None,
                            **{a: float(predict(luts[a], te[m]).mean()) if m.any() else None for a in ARMS}})
        report[fold] = {"n_train": len(tr), "n_test": len(te), "base_rate_test": float(y.mean()),
                        "arms": rows, "calibration": cal}
        if fold == "F2":
            for arm, lut in luts.items():
                (OUT / f"buzzer_{arm}_F2.json").write_text(json.dumps(lut), encoding="utf-8")
    (OUT / "buzzer_grade.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for fold, r in report.items():
        print(f"== {fold}: train {r['n_train']:,} test {r['n_test']:,} base {r['base_rate_test']:.4f}")
        for k, v in r["arms"].items():
            print(f"   {k:12s} {v}")
        for x in r["calibration"]:
            print(f"   P{x['period']} start<={x['start_le']:2d} n={x['n']:6d} act {x['actual']:.3f} "
                  + " ".join(f"{a} {x[a]:.3f}" for a in ARMS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
