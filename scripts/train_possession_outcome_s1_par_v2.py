"""possession_outcome round-2 S1 retrain, parallel -- v2: team-rate table AND feature tables together.

Versioned sibling of `scripts/train_possession_outcome_s1_par_v1.py` (lane J, NOT edited). v1 refuses
`--team-rate-table` with `--feature-table` (docs/ops/aws_launch_chain.md section 18 item 4; the box ran
arm Tfs on a hand-combined table instead). The full retrain needs both at once (E3 features for F_T AND
the corrected in_bonus overlay), so v2:

  1. reads `--design`;
  2. applies `--team-rate-table` through `cbb_sim.team_rate_adapter.apply(frame, path, 'possession_outcome',
     fold, missing)` exactly as v1 does (optional; absent = served expanding-mean features, variant F_R);
  3. applies every `--feature-table PATH:KEYS[:COLS]` in order through `train_par_common_v1.overlay_columns`
     exactly as v1 does (keys comma-separated; COLS comma-separated, default every shared column);
  4. writes the combined design to `<out-root>/design_combined.parquet` and hands it to v1's `run()` as a
     plain `--design` (no table arguments), so fitting, the memo shim, the UNMODIFIED
     `build_engine_event_round2.build()` artifact step and the report are v1's code, byte for byte.

With neither a table nor a feature table, v2 is v1 (it passes `--design` through untouched). Nothing is
chosen here: arms and feature sets come from the round-2 `verdict.json`, as in v1.

    python scripts/train_possession_outcome_s1_par_v2.py --fold F2 --season 2025 --design <design.parquet> \
        --out-root <root>/po --n-jobs 3 \
        [--team-rate-table data/processed/team_rate_features_E3_v4.parquet] \
        --feature-table <root>/foul_state/in_bonus_overlay.parquet:game_id,poss_index,chance_number:in_bonus
Resume: rerun the same command (v1's per-fit checkpoints under <out-root>/cuts/).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
import train_par_common_v1 as C  # noqa: E402

C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

import train_possession_outcome_s1_par_v1 as V1  # noqa: E402


def combine(a) -> tuple[Path, dict]:
    out_root = Path(a.out_root)
    C.assert_fresh_out_dir(out_root)
    C.stamp_owner(out_root, "train_possession_outcome_s1_par_v2", vars(a))
    dst = out_root / "design_combined.parquet"
    rep_p = out_root / "combine_report.json"
    if dst.exists() and rep_p.exists():                     # resume: the combined design is final once written
        return dst, json.loads(rep_p.read_text(encoding="utf-8"))
    design = pd.read_parquet(a.design)
    rep: dict = {"design": str(a.design), "design_sha256": C.sha256_file(Path(a.design)),
                 "team_rate": None, "feature_tables": []}
    if a.team_rate_table:
        from cbb_sim.team_rate_adapter import apply as team_rate_apply
        design = team_rate_apply(design, a.team_rate_table, "possession_outcome", fold=a.fold,
                                 missing=a.team_rate_missing)
        rep["team_rate"] = {"table": str(a.team_rate_table),
                            "table_sha256": C.sha256_file(Path(a.team_rate_table)),
                            "missing": a.team_rate_missing}
    for spec in a.feature_table:
        parts = spec.split(":")
        # a Windows drive letter ("C:\...") is part of the path, not a separator
        if len(parts) >= 2 and len(parts[0]) == 1 and parts[1].startswith(("\\", "/")):
            parts = [parts[0] + ":" + parts[1], *parts[2:]]
        path, keys = Path(parts[0]), parts[1].split(",")
        cols = parts[2].split(",") if len(parts) > 2 and parts[2] else None
        design, r = C.overlay_columns(design, path, keys=keys, cols=cols)
        rep["feature_tables"].append(r)
    tmp = dst.with_suffix(".parquet.tmp")
    design.to_parquet(tmp, index=False)
    tmp.replace(dst)
    rep_p.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return dst, rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2", choices=sorted(V1.B.FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--design", default=str(V1.B.R2_DESIGN))
    ap.add_argument("--verdict", default=str(V1.B.R2_VERDICT))
    ap.add_argument("--engine-dir", default=str(V1.B.ENGINE_DIR))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--team-rate-table", default="")
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise")
    ap.add_argument("--feature-table", action="append", default=[],
                    help="PATH:KEYS[:COLS], repeatable; applied after the team-rate table")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--stop-at", default="")
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--sample-games", type=int, default=0)
    ap.add_argument("--test-n-estimators", type=int, default=0)
    ap.add_argument("--max-cuts", type=int, default=0)
    a = ap.parse_args()
    if a.team_rate_table or a.feature_table:
        design, _ = combine(a)
    else:
        design = Path(a.design)
    ns = argparse.Namespace(
        mode="run", fold=a.fold, season=a.season, seed=a.seed, design=str(design), verdict=a.verdict,
        engine_dir=a.engine_dir, out_root=a.out_root, team_rate_table="", team_rate_missing=a.team_rate_missing,
        feature_table="", overlay_keys="game_id,offense_team_id", overlay_cols="", n_jobs=a.n_jobs,
        stop_at=a.stop_at, no_artifacts=a.no_artifacts, sample_games=a.sample_games,
        test_n_estimators=a.test_n_estimators, max_cuts=a.max_cuts, scratch="")
    r = V1.run(ns)
    return 0 if r.get("complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
