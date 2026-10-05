"""build_engine_tovlevel_offsets_v1.py -- per-game `po_first` offsets for PO section 30 arms T1 / T2. PO worker, 2026-10-05.

Versioned sibling of `build_engine_anchor_offsets_v1.py` (not edited). Writes an .npz in the
`ENGINE_SEASON_ANCHOR` contract (`src/cbb_sim/engine/season_anchor_serving.py`): `game_ids (G,)`, `po_first (G, 6)`
with ONLY column 0 (TOV) non-zero, = logit L(s, game date) - logit Lbar, from the SAME `TovLevel` the trainer used
(rebuilt from the design; its n0 / Lbar / priors are asserted equal to the artifact's anchor mark). Levels use only
`first` rows dated strictly before the game's date. Never overwrites.

    .venv/Scripts/python.exe scripts/build_engine_tovlevel_offsets_v1.py --arm T1 --fold F2 \
        --artifact-dir data/processed/models/possession_outcome/tovlevel/T1_F2_s0/event_round2_s1_F2_2025 \
        --input-dir data/processed/models/engine_v3 --out data/processed/models/possession_outcome/tovlevel/T1_F2_s0/offsets_F2_2025.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import train_possession_outcome_s1_tovlevel_v1 as TL  # noqa: E402
from build_engine_anchor_offsets_v1 import FOLD_TEST, PO_DESIGN, engine_games  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["T1", "T2"], required=True)
    ap.add_argument("--fold", required=True, choices=sorted(FOLD_TEST))
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--design", type=Path, default=PO_DESIGN)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    season = FOLD_TEST[a.fold]
    assert_not_sealed(season)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists: never overwrite")
    mark = json.loads((a.artifact_dir / "index.json").read_text(encoding="utf-8"))["populations"]["first"]["anchor"]
    if mark["arm"] != a.arm:
        raise SystemExit(f"artifact arm {mark['arm']} != --arm {a.arm}")
    d = pd.read_parquet(a.design, columns=["season", "population", "y", "game_date"])
    lv, _ = TL.build_level(d, a.fold, season, a.arm)
    assert str(lv.n0) == str(mark["n0"]), (lv.n0, mark["n0"])
    assert abs(lv.Lbar - mark["Lbar"]) < 1e-12, (lv.Lbar, mark["Lbar"])
    for s, v in mark["prior_by_season"].items():
        assert abs(lv.prior(int(s)) - v) < 1e-12, (s, v)
    games = engine_games(a.input_dir, a.fold, season)
    off = np.zeros((len(games), len(PO.CLASSES)))
    off[:, TL.TOV] = lv.offset(np.full(len(games), season), games["date"].to_numpy())
    lev = TL.expit(TL.logit(lv.Lbar) + off[:, TL.TOV])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, game_ids=games["game_id"].to_numpy(), po_first=off)
    u = pd.DataFrame({"date": games["date"].dt.date.astype(str), "L": lev}).drop_duplicates("date").sort_values("date")
    meta = {"builder": "build_engine_tovlevel_offsets_v1", "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "fold": a.fold, "season": season, "families": ["po"], "arm": a.arm, "n0": str(lv.n0), "Lbar": lv.Lbar,
            "input_dir": str(a.input_dir), "artifact_dir": str(a.artifact_dir), "n_games": int(len(games)),
            "level_by_date": dict(zip(u["date"], u["L"].astype(float)))}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"wrote {a.out}: {len(games)} games, TOV offset range [{off[:, 0].min():+.4f}, {off[:, 0].max():+.4f}], "
          f"level range [{lev.min():.4f}, {lev.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
