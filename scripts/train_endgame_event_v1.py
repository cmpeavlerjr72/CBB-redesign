"""train_endgame_event_v1.py -- block E (event) of late-game round 7, late_game/experiments.md section 15.

Arms (law code in `cbb_sim.models.endgame`, shared with the engine):
  EV0    the served possession_outcome FIRST-chance model (round2_s1 LightGBM, S1 monthly schedule:
         F2 `models/engine/event_round2_s1_F2_2025`, F1 `models/fold1_v1/engine_scratch/event_round2_s1_F1_2024`).
         Control. No fit.
  EV1    class probabilities refit on window first chances by role-band x time bucket, shrunk to EV0 with m = 50
         pseudo-rows (observed/expected form, EV0's row-level effects incl. site carried), training seasons only.
  EV0s1  the EV1 law refitted under seed 1 (s15.0: the noise-floor arm, never selectable).
Training rows: window first chances (periods 1-2, start <= 240 s) of the fold's TRAINING seasons; EV0 on those rows is
the fold's FIRST S1 refit (trained on exactly those seasons; in-sample, stated). Test rows: the test season's window
first chances scored by the S1 schedule (`max_train_date < game_date` asserted per row). Features are the round-2
design's own columns (`C_plus_state`, the columns the served model was fitted on). 2025-26 is never read.

    .venv/Scripts/python.exe scripts/train_endgame_event_v1.py --fold F2 --arm EV1
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_endgame_window_v1 as W  # noqa: E402

from cbb_sim.models import endgame as EG  # noqa: E402

EV_DIR = {"F2": ROOT / "data/processed/models/engine/event_round2_s1_F2_2025",
          "F1": ROOT / "data/processed/models/fold1_v1/engine_scratch/event_round2_s1_F1_2024"}
ART = ROOT / "data/processed/models/late_game/endgame_v1"
OFF = ROOT / "results/endgame/offline"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def served(df: pd.DataFrame, fold: str, first_only: bool) -> np.ndarray:
    idx = json.loads((EV_DIR[fold] / "index.json").read_text(encoding="utf-8"))
    info = idx["populations"]["first"]
    feats = info["features"]
    segs = info["segments"]
    X = np.ascontiguousarray(df[feats].to_numpy(dtype=np.float32))
    if first_only:
        seg = np.zeros(len(df), dtype=np.int64)
    else:
        gd = pd.to_datetime(df["game_date"]).to_numpy()
        cuts = np.array([np.datetime64(s["refit_date"], "ns") for s in segs])
        seg = np.searchsorted(cuts, gd, side="right") - 1
        mtd = np.array([np.datetime64(s["max_train_date"], "ns") for s in segs])[seg]
        if (seg < 0).any() or not (mtd < gd).all():
            raise SystemExit("created_at < tipoff violated for a test row")
    out = np.empty((len(df), 6))
    for k in np.unique(seg):
        r = np.flatnonzero(seg == k)
        w = joblib.load(EV_DIR[fold] / Path(segs[int(k)]["file"]).name)
        if w.get("anchor"):
            raise SystemExit("an anchored event artifact is not the served law")
        try:
            w["model"].clf_.set_params(n_jobs=4)
        except Exception:                                            # noqa: BLE001
            pass
        out[r] = w["model"].predict_proba(X[r])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--arm", choices=["EV0", "EV1", "EV0s1"], required=True)
    a = ap.parse_args()
    seed = 1 if a.arm == "EV0s1" else 0
    np.random.seed(seed)
    t0 = time.time()
    tr, te = W.split(W.event_window(), a.fold)
    log(f"{a.fold} {a.arm} (seed {seed}): train {len(tr):,}, test {len(te):,}")
    p0t = served(te, a.fold, first_only=False)
    meta = {"arm": a.arm, "fold": a.fold, "seed": seed, "created_at": datetime.now(UTC).isoformat(),
            "train_seasons": sorted(int(s) for s in tr["season"].unique()), "test_season": int(te["season"].iloc[0]),
            "served": str(EV_DIR[a.fold].relative_to(ROOT)), "spec": "late_game/experiments.md s15"}
    if a.arm == "EV0":
        p = p0t
    else:
        p0 = served(tr, a.fold, first_only=True)
        law = EG.fit_ev1(tr["period"].to_numpy(), tr["start_clock"].to_numpy(), tr["start_score_diff"].to_numpy(),
                         tr["y"].to_numpy(), p0)
        meta["law"] = law.info
        p = law.apply(p0t, te["period"].to_numpy(), te["start_clock"].to_numpy(), te["start_score_diff"].to_numpy())
        d = ART / (a.fold if seed == 0 else f"{a.fold}_seed{seed}")
        d.mkdir(parents=True, exist_ok=True)
        f = d / "ev1_law.npz"
        if f.exists() and os.environ.get("EG_OVERWRITE") != "1":
            raise SystemExit(f"{f} exists; refusing to overwrite (EG_OVERWRITE=1 for an own rerun)")
        law.save(f)
        meta["artifact"] = str(f.relative_to(ROOT)).replace("\\", "/")
        log("EV1 ratio range " + f"{law.r.min():.3f}-{law.r.max():.3f}; leader 13+ TOV ratio by bucket (P2): "
            + " ".join(f"{x:.3f}" for x in law.r[1, 3, :, 0]))
    out = OFF / a.fold / "E"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.arm}.npz", game_id=te["game_id"].to_numpy(np.int64),
                        period=te["period"].to_numpy(np.int64), poss_index=te["poss_index"].to_numpy(np.int64),
                        prob=p, meta=np.array(json.dumps(meta)))
    log(f"wrote {out / a.arm}.npz ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
