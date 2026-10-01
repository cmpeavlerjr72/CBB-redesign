"""train_fg_make_v4_par_fold_v1.py -- the served fg_make round-4 B1 S1 trainer (par_v2 = par_v1 + adapter v2) for ANY
fold (lane D, 2026-10-01; fold-1 confirmation, docs/models/engine/experiments.md section 2).

par_v1 hard-codes fold 2 in three places: the row split (`FG.fold_slices(design, "F2")`), the joblib payload label
(`"fold": "F2"`) and the manifest (`"fold": "F2", "season": 2025`). This wrapper changes exactly those, nothing else:
  * par_v1's module global `FG` is replaced by a proxy whose `fold_slices` ignores the literal and splits on --fold
    (every other attribute is the real module; `R4M.fit_m`, which selects m on its own fixed split, is untouched);
  * par_v1's `joblib.dump` and manifest writes are relabelled to --fold / the fold's test season.
With --fold F2 it is par_v2 (identity: the same command as the chain's fg stage).

    .venv/Scripts/python.exe scripts/train_fg_make_v4_par_fold_v1.py --fold F1 --mode run --arms B1 --no-floor \
        --n-jobs 3 --out-dir data/processed/models/fold1_v1/fg
CAVEAT carried from the served spec: the shrinkage m is selected by `R4M.fit_m` on the fold-1 TEST season
(`f1_test_log_loss`), so a fold-1 read inherits a hyperparameter chosen on 2023-24 (common to both arms of a paired read).
"""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FOLD_TEST = {"F1": 2024, "F2": 2025}


def _take(argv, flag, default):
    if flag in argv:
        i = argv.index(flag)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


FOLD = _take(sys.argv, "--fold", "F2")
if FOLD not in FOLD_TEST:
    raise SystemExit(f"--fold {FOLD}: one of {sorted(FOLD_TEST)}")

import cbb_sim.team_rate_adapter as TRA  # noqa: E402
import cbb_sim.team_rate_adapter_v2 as TRA2  # noqa: E402

TRA.apply = TRA2.apply                       # = par_v2

import joblib as _joblib  # noqa: E402

import train_fg_make_v4_par_v1 as P  # noqa: E402

_real_FG = P.FG
_proxy = types.ModuleType("fg_make_fold_proxy")
_proxy.__dict__.update(_real_FG.__dict__)
_proxy.fold_slices = lambda design, fold: _real_FG.fold_slices(design, FOLD)
P.FG = _proxy


def _dump(obj, path, *a, **k):
    if isinstance(obj, dict) and "fold" in obj:
        obj = {**obj, "fold": FOLD}
    return _joblib.dump(obj, path, *a, **k)


_jl = types.ModuleType("joblib_fold_proxy")
_jl.__dict__.update(_joblib.__dict__)
_jl.dump = _dump
P.joblib = _jl


def _fix_manifests(out_dir: Path) -> None:
    for m in out_dir.glob("*/manifest_*.json"):
        obj = json.loads(m.read_text(encoding="utf-8"))
        obj["fold"], obj["season"] = FOLD, FOLD_TEST[FOLD]
        m.write_text(json.dumps(obj, indent=2), encoding="utf-8")


if __name__ == "__main__":
    out = Path(_take(list(sys.argv), "--out-dir", "."))
    rc = P.main()
    if rc == 0:
        _fix_manifests(out if out.is_absolute() else Path.cwd() / out)
    raise SystemExit(rc)
