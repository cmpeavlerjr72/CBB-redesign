"""train_rebound_v3_par_anchor_artifacts_v1.py -- anchored rebound (season-drift anchor O) PLUS engine artifacts.
Lane M, 2026-09-30. Versioned wrapper; nothing existing is edited.

Same arguments as `train_rebound_v3_par_anchor_v1.py` (lane C: `--anchor O` + every argument of lane J's
`train_rebound_v3_par_v1.py`). It combines:
  * lane C's `install_anchor_O()` (arm `O`: init_score on the OREB raw score, `_a5_off` = anchor O values), and
  * lane G's `train_rebound_v3_par_artifacts_v1` worker and writer (the fitted model is kept and written as
    `<out-dir>/[<stem>/]artifacts/<cell>/manifest.json + seg_<refit>.joblib`, the layout `ReboundAdapter` reads),
which G's wrapper refuses for offset arms because the engine had no offset feed. The engine now has one
(`ENGINE_SEASON_ANCHOR`, `src/cbb_sim/engine/season_anchor_serving.py`), so each joblib here carries
    "anchor": {"arm": "O", "kind": "binary", "cols": [OREB], ...}
and the engine refuses to serve it unless the per-game offsets (`scripts/build_engine_anchor_offsets_v1.py`,
families rb) are fed. The worker also keeps, per cut, the scored rows' design index, game_id and `_a5_off`, so the
engine's predictions can be checked against this model's own offline predictions (lane M proof c).

    python scripts/train_rebound_v3_par_anchor_artifacts_v1.py --anchor O --stage 2 --folds F2 --arms O \
        --team-rate-table data/processed/team_rate_features_E3_v4.parquet --n-jobs 24 \
        --out-dir data/processed/models/rebound/round3_par_TO_art_v1
Only stage 2 (S1_weekly) writes artifacts.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import train_rebound_v3_par_v1 as J  # noqa: E402  (pins threads on import)
import train_rebound_v3_par_anchor_v1 as CA  # noqa: E402
import train_rebound_v3_par_artifacts_v1 as GA  # noqa: E402
import train_rebound_v3_round3 as R3  # noqa: E402

import json  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from cbb_sim.models import rebound as RB  # noqa: E402

ANCHOR_MARK = {"arm": "O", "kind": "binary", "cols": [RB.CLASS_INDEX["OREB"]],
               "target": "OREB share of live misses",
               "definition": "cbb_sim.season_anchor.anchor_O; offset = logit(L_asof) - logit(Lbar_train)",
               "serve": "ENGINE_SEASON_ANCHOR=<anchor_offsets .npz with rb_oreb>",
               "writer": "train_rebound_v3_par_anchor_artifacts_v1"}


def _worker_anchor_art(spec, feats, seed, pool, sub, feed_sub, refit, n_scored, params_override):
    """Lane G's artifact worker, plus the scored rows' identity and offsets."""
    r = GA._worker_art(spec, feats, seed, pool, sub, feed_sub, refit, n_scored, params_override)
    r["rows"] = {"index": sub.index.to_numpy(), "game_id": sub["game_id"].to_numpy(),
                 "a5_off": sub["_a5_off"].to_numpy(dtype="float64")}
    return r


def mark_artifacts(dirs: list[str]) -> None:
    for d in dirs:
        d = Path(d)
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for e in man["artifacts"]:
            p = d / e["path"]
            w = joblib.load(p)
            w["anchor"] = dict(ANCHOR_MARK)
            w["note"] = w.get("note", "") + "; anchored (season-drift O): serve only with ENGINE_SEASON_ANCHOR"
            joblib.dump(w, p)
        man["anchor"] = dict(ANCHOR_MARK)
        (d / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")


def main() -> int:
    import argparse
    argv = sys.argv[1:]
    if "--anchor" not in argv or argv[argv.index("--anchor") + 1] != "O":
        raise SystemExit("--anchor O is required (this wrapper writes ANCHORED artifacts)")
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--stage", type=int, default=1); ap.add_argument("--arms", default="")
    ap.add_argument("--folds", default="F2"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=""); ap.add_argument("--team-rate-table", default="")
    ap.add_argument("--mode", default="run")
    a, _ = ap.parse_known_args(argv)
    if a.arms != "O":
        raise SystemExit("--arms O only: the artifact writer's feature list is arm O's")
    J._worker = _worker_anchor_art
    rc = CA.main()                      # parses --anchor, installs O, runs J.main
    if rc in (0, None) and a.mode == "run" and a.stage == 2:
        stem = Path(a.team_rate_table).stem if a.team_rate_table else ""
        w = GA.write_artifacts(Path(a.out_dir).resolve(), stem, a.stage, a.folds, a.arms, a.seed,
                               lambda arm: R3.features_for(R3.arm_spec(arm)))
        mark_artifacts(w)
        print("anchored artifact dirs:", *w, sep="\n  ")
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
