"""build_chance_time_lut_v1.py -- chance_time round 1 tables (lane I, 2026-09-30; chance_time/experiments.md s1.2).

Training seasons only (F2: 2022-2024, F1: 2022-2023). Writes data/processed/models/chance_time/<fold>/lut_v1.npz:
  cont_q[b, k, :]  1001 quantiles of fg_make design `chance_elapsed_s` for chance bucket b (0: chance 2, 1: chance 3+)
                   and shot class k (0 rim, 1 jump2, 2 three)
  r_q[g, d, :]     1001 quantiles of (chance-1 duration / possession duration), start group g (0: DREB/TOV,
                   1: other), possession-duration bin d (BIN_EDGES); thin cells (< MIN_N) use the group's pooled row
  bin_edges        lower edges of the duration bins
Usage: build_chance_time_lut_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

FOLDS = {"F2": [2022, 2023, 2024], "F1": [2022, 2023]}
QS = np.linspace(0.0, 1.0, 1001)
BIN_EDGES = np.array(list(range(0, 31)) + [31, 36, 41, 51, 61], dtype=np.int64)
MIN_N = 200
CLASSES = ("FGA_rim", "FGA_jump2", "FGA_3")


def dur_bin(d: np.ndarray) -> np.ndarray:
    return np.searchsorted(BIN_EDGES, np.asarray(d), side="right") - 1


def ratio_frame(seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        p = pd.read_parquet(ROOT / f"data/processed/possessions_v2/possessions_{s}.parquet",
                            columns=["game_id", "period", "poss_index", "duration_s", "start_reason"])
        c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{s}.parquet",
                            columns=["game_id", "period", "poss_index", "chance_number", "duration_s"])
        c1 = c[c["chance_number"] == 1].rename(columns={"duration_s": "c1"})[
            ["game_id", "period", "poss_index", "c1"]]
        m = p.merge(c1, on=["game_id", "period", "poss_index"], how="left")
        m["c1"] = m["c1"].fillna(m["duration_s"])
        D = m["duration_s"].to_numpy(np.float64)
        r = np.where(D > 0, np.clip(m["c1"].to_numpy(np.float64) / np.maximum(D, 1e-9), 0.0, 1.0), 1.0)
        out.append(pd.DataFrame({"season": s, "D": D, "r": r,
                                 "g": np.where(m["start_reason"].isin(["DREB", "TOV"]), 0, 1)}))
    return pd.concat(out, ignore_index=True)


def build(fold: str) -> dict:
    seasons = FOLDS[fold]
    assert_not_sealed(seasons, context="chance_time lut")
    d = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet",
                        columns=["season", "shot_class", "chance_number", "chance_elapsed_s"])
    d = d[d["season"].isin(seasons)]
    cont_q = np.zeros((2, 3, len(QS)))
    cont_n = np.zeros((2, 3), dtype=np.int64)
    for b, sel in enumerate((d["chance_number"] == 2, d["chance_number"] >= 3)):
        for k, c in enumerate(CLASSES):
            v = d.loc[sel & (d["shot_class"] == c), "chance_elapsed_s"].to_numpy(np.float64)
            cont_n[b, k] = len(v)
            cont_q[b, k] = np.quantile(v, QS)
    rf = ratio_frame(seasons)
    nb = len(BIN_EDGES)
    r_q = np.zeros((2, nb, len(QS)))
    r_n = np.zeros((2, nb), dtype=np.int64)
    bins = dur_bin(rf["D"].to_numpy())
    for g in (0, 1):
        pooled = np.quantile(rf.loc[rf["g"] == g, "r"].to_numpy(), QS)
        for b in range(nb):
            v = rf.loc[(rf["g"] == g).to_numpy() & (bins == b), "r"].to_numpy()
            r_n[g, b] = len(v)
            r_q[g, b] = np.quantile(v, QS) if len(v) >= MIN_N else pooled
    out_dir = ROOT / "data/processed/models/chance_time" / fold
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "lut_v1.npz", cont_q=cont_q, cont_n=cont_n, r_q=r_q, r_n=r_n, bin_edges=BIN_EDGES)
    meta = {"fold": fold, "train_seasons": seasons, "builder": "scripts/build_chance_time_lut_v1.py",
            "n_quantiles": len(QS), "min_n": MIN_N, "cont_n": cont_n.tolist(), "r_n": r_n.tolist(),
            "cont_median": cont_q[:, :, 500].tolist(),
            "share_r_eq_1": {str(g): float((rf.loc[rf["g"] == g, "r"] >= 0.999).mean()) for g in (0, 1)},
            "created_at": pd.Timestamp.utcnow().isoformat()}
    (out_dir / "lut_v1.json").write_text(json.dumps(meta, indent=1))
    print(fold, json.dumps({k: meta[k] for k in ("cont_n", "cont_median", "share_r_eq_1")}), flush=True)
    return meta


if __name__ == "__main__":
    for f in ("F2", "F1"):
        build(f)
