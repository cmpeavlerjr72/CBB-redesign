"""digest_engine_run.py -- parity digest for an engine_v0 results directory.

The question this answers, and the only one: does a run of `run_engine.py` on
one box (e.g. an AWS sweep box) produce EXACTLY the same simulation as the
same run on the home Windows box? RNG is seeded on (seed, game_id, family)
(CLAUDE.md), so for a fixed set of (game_id, seed) pairs and a fixed set of
adapter artifacts the answer must be bit-identical -- this is a hard equality
check, not a tolerance.

Unlike cfb-props-sim's `parity_digest.py` (which calls the sweep's pure
`run_one(game_id, seed)` in-process), the CBB engine's unit of work is the
whole multiprocess `run_engine.py` invocation, and its contract is the output
it already writes: `results/engine_v0/<tag>/{games.parquet,players.parquet,
run_meta.json}`. So this script digests THOSE FILES rather than re-simulating
anything -- it can be pointed at any completed results directory, on any box,
including one already fetched from HF.

    # on the home (reference) box, after a smoke run:
    .venv/Scripts/python.exe scripts/digest_engine_run.py --emit \
        --results results/engine_v0/<tag> --out docs/ops/parity_reference_windows.json

    # on the cloud box, after the SAME smoke run (same --tag, same --seeds,
    # same --max-games, same adapter flags):
    python scripts/digest_engine_run.py --compare docs/ops/parity_reference_windows.json \
        --results results/engine_v0/<tag>

`--compare` exits 0 ONLY on a byte-identical sha256 of the canonicalized row
data. On any mismatch it prints every differing (game_id, seed) row/column
with both values and exits 1. It does not apply a tolerance and does not
decide a small difference is fine -- per CLAUDE.md ("no hand tuning on engine
output") and the cfb precedent, tolerance is a PM adjudication, never a
script's.

WHAT IS HASHED
--------------
`games.parquet`: every row (one per game_id x seed), all columns, floats
rounded to 6 dp, sorted by (game_id, seed).

`players.parquet` (if present -- `--no-players` runs write none): every row
(one per game_id x seed x athlete_id), all columns, floats rounded to 6 dp,
sorted by (game_id, seed, athlete_id).

`run_meta.json`'s `adapter_flags` and `engine_rules_from_data` (the sub-model
selection and rule-era config that determines what the run computed) are
included in the hash. `created_at`, `runtime_s`, `possessions_per_second`,
`workers` and other pure-performance/provenance fields are NOT hashed --
recorded in metadata only, so a box being faster or slower cannot fail parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# run_meta fields that describe WHAT was computed (hashed) vs HOW FAST / WHEN
# (metadata only, never hashed -- a faster box must not fail the gate).
META_HASHED_KEYS = ("fold", "season", "n_games", "adapter_flags", "engine_rules_from_data")
DIGEST_VERSION = 3

# Every ENGINE_* switch pulled out of run_meta.json's `adapter_flags` into its
# own top-level `flags` block (digest_version 2). `adapter_flags` already
# carries these (and was already hashed via META_HASHED_KEYS), so this does
# not change WHAT is checked -- it exists so a config mismatch (e.g. a box
# that ran with the wrong ENGINE_FG_MAKE) prints as one obvious `flags.*`
# line instead of being buried in a full adapter_flags blob diff.


def _round(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 6)
    return x


def _rows(df: pd.DataFrame, sort_cols: list[str]) -> list[dict]:
    df = df.sort_values(sort_cols).reset_index(drop=True)
    cols = list(df.columns)
    out = []
    for rec in df.itertuples(index=False, name=None):
        out.append({c: _round(v) for c, v in zip(cols, rec)})
    return out


def _posix(obj):
    """Normalize embedded OS-native path strings to POSIX before hashing.

    `digest_version` 3 (2026-09-11, AWS launch chain section 12/13): the first
    two cross-platform parity runs (v2 and v3 references) each reported the
    SAME cosmetic mismatch -- `meta.adapter_flags` differs as a whole because
    a `sources`/`manifest` provenance string was built with `str(Path(...))`,
    which is OS-native (`data\\processed\\...` on Windows,
    `data/processed/...` on Linux). Zero simulated value ever differed on
    either occasion. Per this script's own design (`created_at`, `runtime_s`,
    `workers` etc. are provenance/performance and already excluded from the
    hash), a path SEPARATOR is exactly that kind of provenance detail, not a
    computed value -- so it is normalized here rather than left to keep
    failing the gate on every future box. This only rewrites backslashes to
    forward slashes inside string leaves of the HASHED metadata block (never
    touches `games`/`players` row data, and never touches any adapter default
    or simulated output)."""
    if isinstance(obj, str):
        return obj.replace("\\", "/")
    if isinstance(obj, dict):
        return {k: _posix(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_posix(v) for v in obj]
    return obj


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _versions() -> dict:
    import joblib
    import lightgbm
    import numpy
    import pandas as _pd
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "lightgbm": lightgbm.__version__, "pandas": _pd.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
            "scipy": scipy.__version__}


def build_digest(results_dir: Path) -> dict:
    games_p = results_dir / "games.parquet"
    players_p = results_dir / "players.parquet"
    meta_p = results_dir / "run_meta.json"
    if not games_p.exists():
        raise FileNotFoundError(f"{games_p} missing -- not a completed engine_v0 results dir")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    games = pd.read_parquet(games_p)
    game_rows = _rows(games, ["game_id", "seed"])

    if players_p.exists():
        players = pd.read_parquet(players_p)
        player_rows = _rows(players, ["game_id", "seed", "athlete_id"])
    else:
        player_rows = []

    hashed_meta = _posix({k: meta.get(k) for k in META_HASHED_KEYS if k in meta})
    adapter_flags = meta.get("adapter_flags", {}) or {}
    flags = {k: v for k, v in adapter_flags.items() if k.startswith("ENGINE_")}

    return {
        "digest_version": DIGEST_VERSION,
        "n_game_rows": len(game_rows),
        "n_player_rows": len(player_rows),
        "games": game_rows,
        "players": player_rows,
        "meta": hashed_meta,
        "flags": flags,
    }


def canonical(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha_of(doc: dict) -> str:
    return hashlib.sha256(canonical(doc).encode("utf-8")).hexdigest()


def _flatten(doc: dict) -> dict:
    out = {}
    for row in doc["games"]:
        key = f"games.game{row['game_id']}.seed{row['seed']}"
        for c, v in row.items():
            out[f"{key}.{c}"] = v
    for row in doc["players"]:
        key = f"players.game{row['game_id']}.seed{row['seed']}.athlete{row['athlete_id']}"
        for c, v in row.items():
            out[f"{key}.{c}"] = v
    out["n_game_rows"] = doc["n_game_rows"]
    out["n_player_rows"] = doc["n_player_rows"]
    for k, v in doc["meta"].items():
        out[f"meta.{k}"] = v
    for k, v in doc.get("flags", {}).items():
        out[f"flags.{k}"] = v
    return out


def diff_docs(ref: dict, cur: dict, limit: int = 200) -> list[str]:
    a, b = _flatten(ref), _flatten(cur)
    lines = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<MISSING>"), b.get(k, "<MISSING>")
        if va != vb:
            lines.append(f"  {k}: reference={va!r}  this_box={vb!r}")
            if len(lines) >= limit:
                lines.append(f"  ... (truncated at {limit} differing fields)")
                break
    return lines


def cmd_emit(results_dir: Path, out_path: Path) -> int:
    doc = build_digest(results_dir)
    sha = sha_of(doc)
    ref = {"sha256": sha,
           "metadata": {"platform": platform.platform(), "machine": platform.machine(),
                        "versions": _versions(), "git_sha": _git_sha(),
                        "results_dir": str(results_dir)},
           "digest": doc}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[digest] {doc['n_game_rows']} game rows, {doc['n_player_rows']} player rows")
    print(f"[digest] flags: {doc['flags']}")
    print(f"[digest] wrote {out_path}")
    print(f"[digest] SHA256 {sha}")
    return 0


def cmd_compare(results_dir: Path, ref_path: Path) -> int:
    ref_doc = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_sha, ref_frame = ref_doc["sha256"], ref_doc["digest"]
    cur_frame = build_digest(results_dir)
    cur_sha = sha_of(cur_frame)
    rmeta = ref_doc.get("metadata", {})
    print(f"[digest] reference : {ref_sha}")
    print(f"[digest]   built on: {rmeta.get('platform')} git {str(rmeta.get('git_sha'))[:10]}")
    print(f"[digest]   versions: {rmeta.get('versions')}")
    print(f"[digest]   flags   : {ref_frame.get('flags')}")
    print(f"[digest] this box  : {cur_sha}")
    print(f"[digest]   platform: {platform.platform()} git {_git_sha()[:10]}")
    print(f"[digest]   versions: {_versions()}")
    print(f"[digest]   flags   : {cur_frame.get('flags')}")
    if ref_frame.get("flags") != cur_frame.get("flags"):
        print("[digest] WARNING: flags differ between reference and this box -- "
              "this is not the same engine config, any sha match/mismatch below "
              "is not a meaningful parity result.")
    if ref_sha == cur_sha:
        print("[digest] PASS -- bit-identical digest. Sweep may proceed.")
        return 0
    print("[digest] FAIL -- digest mismatch. DO NOT TRUST A LARGE SWEEP FROM THIS BOX.")
    if ref_frame.get("digest_version") != cur_frame.get("digest_version"):
        print(f"[digest] digest_version differs "
              f"({ref_frame.get('digest_version')} vs {cur_frame.get('digest_version')}) "
              "-- the harness itself changed; re-emit the reference on Windows.")
    lines = diff_docs(ref_frame, cur_frame)
    print(f"[digest] {len(lines)} differing field(s):")
    for ln in lines:
        print(ln)
    print("[digest] Tolerance is NOT this script's call. Take the field list "
          "above back to the PM.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True,
                    help="path to a results/engine_v0/<tag> directory")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--emit", action="store_true", help="write the reference digest")
    g.add_argument("--compare", metavar="REFERENCE.json",
                   help="diff this results dir's digest against a reference")
    ap.add_argument("--out", default="docs/ops/parity_reference_windows.json",
                    help="--emit output path")
    args = ap.parse_args()
    results_dir = Path(args.results)
    if args.emit:
        return cmd_emit(results_dir, Path(args.out))
    return cmd_compare(results_dir, Path(args.compare))


if __name__ == "__main__":
    sys.exit(main())
