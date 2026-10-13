"""train_endgame_foul_v1.py -- block F (foul state) of late-game round 7 = possession_outcome experiments.md section 29
EXACTLY as registered (window, cells, hierarchy, m), promoted by late_game/experiments.md section 15.

Window: period 2, possession start <= 120 s, regulation. Target K = the DEFENCE's non-trip fouls in the possession
(`def_silent`, engine-definition table `round6/foul_accrual_poss_v2.parquet`), clipped at 3.
  LF0    served accrual LUT as a Bernoulli (P(K=0) = 1-p, P(K=1) = p). F2: `round7/lut_acc_A2_F2.npz` (the served
         R9ao3 table); F1: the fold-1 sibling `fold1_v1/loop/lut_acc_A2_F1.npz` (honest for 2023-24; section 29's
         lane-L script read the F2 table on both folds, which F1 cannot use: its training seasons include 2024).
  LF1    window refit, binary P(K >= 1) by cell.
  LF2    window refit, count P(K = 0..3) by cell.
  LF0s1  the F block's winner-class law (`--winner`) refitted under seed 1: noise-floor arm, never selectable.
Cells: defence role (trailing by 1-3 / 4-6 / 7+, tied, leading) x start clock ((0,30], (30,60], (60,120]) x defence
team fouls (0-3, 4-5, 6-8, 9+); hierarchy clock -> clock x fouls -> full cell, m = 50. Trained on 2022-23 (F1) /
2022-24 (F2). The fit is section 29's `train_late_foul_v1.fit` (imported, unchanged).

Writes test predictions `results/endgame/offline/<fold>/F/<arm>.npz` and engine artifacts
`data/processed/models/late_game/endgame_v1/<fold>/late_foul_<LF1|LF2>.json` (the `late_foul.py` format).

    .venv/Scripts/python.exe scripts/train_endgame_foul_v1.py --fold F2 --arm LF2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_endgame_window_v1 as W  # noqa: E402
import train_late_foul_v1 as S29  # noqa: E402

LUT = {"F2": ROOT / "data/processed/models/possession_outcome/round7/lut_acc_A2_F2.npz",
       "F1": ROOT / "data/processed/models/fold1_v1/loop/lut_acc_A2_F1.npz"}
ART = ROOT / "data/processed/models/late_game/endgame_v1"
OFF = ROOT / "results/endgame/offline"


def lut_p(d, fold):
    t = np.load(LUT[fold])
    ci = np.searchsorted(t["clock_cuts"], d["start_clock"].to_numpy(), side="right")
    mi = np.searchsorted(t["margin_cuts"], d["start_score_diff"].to_numpy(), side="right")
    m = int(t["max_fouls"])
    site = np.where(d["neutral_site"], 0, np.where(d["offense_is_home"], 1, 2))
    return t["lut"][1, ci, mi, np.clip(d["def_team_fouls_true"].to_numpy(), 0, m),
                    np.clip(d["off_team_fouls_true"].to_numpy(), 0, m), site]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--arm", choices=["LF0", "LF1", "LF2", "LF0s1"], required=True)
    ap.add_argument("--winner", choices=["LF1", "LF2"], default=None, help="LF0s1 only")
    a = ap.parse_args()
    seed = 1 if a.arm == "LF0s1" else 0
    np.random.seed(seed)
    t0 = time.time()
    tr, te = W.split(W.foul_window(), a.fold)
    meta = {"arm": a.arm, "fold": a.fold, "seed": seed, "created_at": datetime.now(UTC).isoformat(),
            "train_seasons": sorted(int(s) for s in tr["season"].unique()), "test_season": int(te["season"].iloc[0]),
            "spec": "possession_outcome/experiments.md s29 via late_game/experiments.md s15"}
    if a.arm == "LF0":
        p0 = lut_p(te, a.fold)
        P = np.column_stack([1 - p0, p0, np.zeros_like(p0), np.zeros_like(p0)])
        meta["lut"] = str(LUT[a.fold].relative_to(ROOT))
    else:
        law_arm = a.arm if a.arm != "LF0s1" else a.winner
        if law_arm is None:
            raise SystemExit("LF0s1 needs --winner")
        lut = S29.fit(tr, law_arm == "LF1")
        P = S29.probs(lut, te)
        d = ART / (a.fold if seed == 0 else f"{a.fold}_seed{seed}")
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"late_foul_{law_arm}.json"
        if f.exists() and seed == 0 and os.environ.get("EG_OVERWRITE") != "1":
            raise SystemExit(f"{f} exists; refusing to overwrite")
        f.write_text(json.dumps({
            "arm": law_arm, "probs": lut.tolist(), "clock_edges": S29.CLOCK_EDGES.tolist(),
            "foul_edges": S29.FOUL_EDGES.tolist(),
            "roles": ["def_trail_1_3", "def_trail_4_6", "def_trail_7p", "tied", "def_leading"],
            "window": "period 2, start <= 120 s, regulation", "train_seasons": meta["train_seasons"], "m": S29.M,
            "fold": a.fold, "seed": seed}), encoding="utf-8")
        meta["artifact"] = str(f.relative_to(ROOT)).replace("\\", "/")
    out = OFF / a.fold / "F"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.arm}.npz", game_id=te["game_id"].to_numpy(np.int64),
                        period=te["period"].to_numpy(np.int64), poss_index=te["poss_index"].to_numpy(np.int64),
                        prob=np.asarray(P, dtype=np.float64), meta=np.array(json.dumps(meta)))
    print(f"[{time.strftime('%H:%M:%S')}] {a.fold} {a.arm}: train {len(tr):,} test {len(te):,} -> {out / a.arm}.npz "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
