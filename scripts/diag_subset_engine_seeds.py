"""diag_subset_engine_seeds.py -- write a seed SUBSET of an existing engine
results directory to a NEW directory, so a larger run can be graded on exactly
the seeds another run has without re-simulating it and without touching a byte
of the original.

WHY THIS EXISTS. A paired comparison of two engine configurations must be read
on the SAME seed values, because RNG is keyed on (seed, game_id, family)
(CLAUDE.md) and paired arms share aligned streams. The 50-seed
`F2_2025_s200_rewire1` run and a fresh 20-seed run therefore pair only on seeds
0-19. Re-running the 50-seed job to get them is both wasteful and, since it
straddles two working trees, not reproducible. Subsetting its written output is
exact.

It is also how the seed-offset NOISE FLOOR is produced without any new
simulation: two disjoint seed windows of the SAME run are two spec-identical
draws, which is precisely what CLAUDE.md's floor asks for.

`--out` must not exist. Nothing is ever written over.

    .venv/Scripts/python.exe scripts/diag_subset_engine_seeds.py \
        --results results/engine_v0/F2_2025_s200_rewire1 \
        --seeds 0:20 --out results/engine_v0/F2_2025_s200_rewire1_seeds00_19
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--seeds", required=True, help="lo:hi, half-open (0:20 = seeds 0..19)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.seeds.split(":"))
    src, out = Path(args.results), Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to write over an existing directory: {out}")
    out.mkdir(parents=True)

    g = pd.read_parquet(src / "games.parquet")
    keep = g[(g["seed"] >= lo) & (g["seed"] < hi)].reset_index(drop=True)
    n_games = int(keep["game_id"].nunique())
    per = keep.groupby("seed")["game_id"].nunique()
    full = sorted(int(s) for s in per[per == n_games].index)
    if len(full) != hi - lo:
        print(f"  NOTE: {hi-lo} seeds requested, {len(full)} are complete over "
              f"all {n_games} games; keeping only the complete ones")
    keep = keep[keep["seed"].isin(full)].reset_index(drop=True)
    keep.to_parquet(out / "games.parquet", index=False)

    pp = src / "players.parquet"
    n_p = 0
    if pp.exists():
        p = pd.read_parquet(pp)
        p = p[p["seed"].isin(full)].reset_index(drop=True)
        p.to_parquet(out / "players.parquet", index=False)
        n_p = len(p)

    meta = json.loads((src / "run_meta.json").read_text(encoding="utf-8"))
    meta["engine_tag"] = f"engine_v0/{out.name}"
    meta["seeds"] = full
    meta["n_seeds"] = len(full)
    meta["n_rows"] = int(len(keep))
    meta["subset_of"] = str(src)
    meta["subset_seed_window"] = [lo, hi]
    meta["subset_note"] = ("seed subset of an existing run; no re-simulation. "
                           "Every row is byte-identical to the parent run's.")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                       encoding="utf-8")
    print(f"wrote {out}: {len(keep):,} game rows, {n_p:,} player rows, "
          f"seeds {full[0]}..{full[-1]} ({len(full)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
