"""run_po4b_closed_loop_sample_v1.py -- versioned wrapper: `run_po4b_closed_loop.py` with a game-sample FILE. Lane G.

    .venv/Scripts/python.exe scripts/run_po4b_closed_loop_sample_v1.py --sample-file \
        data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet  <every run_po4b_closed_loop.py argument>

`run_po4b_closed_loop.py` (not edited) selects its 500 games with `subset_rows(games)` (sorted game_id, every 11th).
This wrapper replaces that one function with a lookup of the file's `game_id` column in `inp.games` (rows returned in the
file's sorted order; a game id absent from the inputs is a hard error) and sets the expected size to the file's length,
then runs the original `main()` unchanged (`--input-dir`, `--arm`, `--seeds`, `--seed-offset`, `--tag`, ... all native).
Only the PARENT process uses `subset_rows`, so this holds under spawn and fork. Files (under data/processed/truth/):
  stride500_verified_v1_F2_2025.parquet          same rule on the verified universe (10 games in common with po4b_R_s25)
  stride500_verified_minswap_v1_F2_2025.parquet  existing sample, the one unplayed game swapped (499 in common)
  stride500_current_rule_reproduced_F2_2025.parquet  the current sample as ids
Grade with CBB_TRUTH=verified_v1.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_po4b_closed_loop as R  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if "--sample-file" not in argv:  # default sample is now the verified stride sample inside run_po4b_closed_loop.py
        sys.argv = [sys.argv[0], *argv]
        return R.main()
    i = argv.index("--sample-file"); path = Path(argv[i + 1]); del argv[i:i + 2]
    ids = np.sort(pd.read_parquet(path)["game_id"].to_numpy().astype("int64"))
    if len(ids) != len(set(ids.tolist())):
        raise SystemExit("sample file has duplicate game ids")

    def subset_rows(games: pd.DataFrame) -> np.ndarray:
        pos = {int(g): k for k, g in enumerate(games["game_id"].to_numpy())}
        miss = [int(g) for g in ids if int(g) not in pos]
        if miss:
            raise SystemExit(f"{len(miss)} sample game ids are not in the engine inputs, e.g. {miss[:5]}")
        return np.array([pos[int(g)] for g in ids], dtype=np.int64)

    R.subset_rows = subset_rows
    R.SUBSET_N = len(ids)
    R.SAMPLE_FILE = str(path)
    sys.argv = [sys.argv[0], *argv]
    rc = R.main()
    # the original run_meta does not know the sample file or the default-off engine switches; stamp them
    import json, os
    tag = argv[argv.index("--tag") + 1]
    rd = Path(argv[argv.index("--results-dir") + 1]) if "--results-dir" in argv else R.DEFAULT_RESULTS
    mp = rd / tag / "run_meta.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["sample_file"] = str(path); m["sample_n"] = int(len(ids))
        for k in ("ENGINE_SHOT_BLOCK", "ENGINE_FOUL_JOINT", "ENGINE_TEAM_RATE_DRAW", "ENGINE_FOUL_ACCRUAL", "CBB_TRUTH"):
            m[f"env_{k}"] = os.environ.get(k)
        mp.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
