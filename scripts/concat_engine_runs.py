"""
concat_engine_runs.py -- merge chunked `run_engine.py` outputs into one
gradeable results directory.

Why this exists: `scripts/run_engine.py` writes its `games.parquet` /
`players.parquet` exactly once, at process end -- it has no incremental
checkpoint, unlike cfb-props-sim's sweep runner. `scripts/run_aws_sweep.sh`
works around that by running `run_engine.py` once per seed chunk, each with
its own `--seed-offset` and its own tag suffix
(`<tag>_off<offset>_n<n>/`), so a spot kill costs at most one chunk. But
`scripts/eval_gates.py`, `grade_market_games.py`, `grade_market_props.py` and
`diag_engine_multilevel.py` all read ONE `results/<tag>/` directory through
`cbb_sim.eval.contract.load_engine_results`. This script is the missing step
between "several chunks landed" and "one gradeable run": it validates the
chunks are actually comparable, concatenates them, and writes a merged
directory that satisfies the SAME contract a single `run_engine.py`
invocation would have produced.

    .venv/Scripts/python.exe scripts/concat_engine_runs.py \
        --tag F2_2025_s200_r2event --results-dir results/engine_v0

    finds results/engine_v0/F2_2025_s200_r2event_off*_n*/, validates them,
    and writes results/engine_v0/F2_2025_s200_r2event/{games.parquet,
    players.parquet,run_meta.json}.

WHAT IS REFUSED (raises, writes nothing)
-----------------------------------------
* Zero matching chunk directories.
* A chunk directory that itself fails `cbb_sim.eval.contract` validation
  (missing required columns/keys) -- reuses that module rather than
  reimplementing it, so "gradeable" means the same thing here as everywhere
  else in the repo.
* Chunks whose `fold`, `season`, `n_games`, `adapter_flags` or
  `engine_rules_from_data` differ -- mixing configs would silently produce a
  run that looks like one coherent sim but isn't.
* **Overlapping seeds** across chunks (the PM's explicit requirement): each
  chunk's run_meta `seeds` list is checked pairwise; any seed appearing in
  more than one chunk stops the merge and names the offending chunks and
  seed ids.
* Duplicate (game_id, seed) rows in the concatenated games frame (a second
  sanity check independent of the seed-list check above).
* An existing output directory, unless `--overwrite` is passed (CLAUDE.md:
  "never overwrite a data file another worker may be reading").
* Inconsistent `players.parquet` presence across chunks (some have it, some
  don't), unless `--allow-partial-players` is passed -- in which case
  `players.parquet` is DROPPED from the merged output entirely (a partial
  player table would misrepresent coverage; "not found" is the contract's
  own legitimate state for "no player data", a partial table is not).

MERGED run_meta.json
---------------------
Same required keys as a single run (`engine_tag`, `created_at`, `seeds`,
`fold`, `backtest`, `sealed_touched`) plus the informational fields
`run_engine.py` itself writes (`n_games`, `n_rows`, `possessions_simulated`,
`adapter_flags`, `engine_rules_from_data`, `diagnostics` summed per key,
etc.), PLUS two fields no single run has, both requested by the PM:

    "chunks": [{"tag", "dir", "seed_offset", "n_seeds", "seeds",
                "created_at", "runtime_s", "possessions_simulated"}, ...]
    "merged_from": {"tool": "scripts/concat_engine_runs.py", "merged_at",
                     "source_results_dir", "n_chunks"}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval.contract import (  # noqa: E402
    ContractError,
    load_games,
    load_players,
    load_run_meta,
)

CHUNK_RE = re.compile(r"^(?P<tag>.+)_off(?P<offset>\d+)_n(?P<n>\d+)$")

#: run_meta keys that MUST be identical across every chunk being merged --
#: anything else differing means the chunks are not the same experiment.
CONSISTENT_META_KEYS = ("fold", "season", "n_games", "backtest", "sealed_touched",
                        "adapter_flags", "engine_rules_from_data", "ast_is_placeholder")


class ConcatError(RuntimeError):
    """A chunk set cannot be merged; nothing was written."""


def find_chunks(results_root: Path, tag: str) -> list[Path]:
    dirs = sorted(p for p in results_root.glob(f"{tag}_off*_n*") if p.is_dir())
    parsed = []
    for d in dirs:
        m = CHUNK_RE.match(d.name)
        if not m or m.group("tag") != tag:
            continue
        parsed.append(d)
    return parsed


def _chunk_offset_n(d: Path) -> tuple[int, int]:
    m = CHUNK_RE.match(d.name)
    return int(m.group("offset")), int(m.group("n"))


def load_chunk(d: Path) -> dict:
    """Load + contract-validate one chunk. Raises ContractError (via the
    shared contract module) on a malformed chunk -- the same failure mode a
    malformed single run would hit in eval_gates.py."""
    meta = load_run_meta(d).frame
    games = load_games(d).frame
    players_res = load_players(d)
    offset, n = _chunk_offset_n(d)
    return {
        "dir": d, "meta": meta, "games": games, "players": players_res.frame,
        "has_players": players_res.frame is not None,
        "seed_offset": offset, "n_declared": n,
    }


def validate_consistency(chunks: list[dict]) -> None:
    first = chunks[0]["meta"]
    for c in chunks[1:]:
        for k in CONSISTENT_META_KEYS:
            a, b = first.get(k), c["meta"].get(k)
            if a != b:
                raise ConcatError(
                    f"chunk {c['dir'].name} has {k}={b!r} but {chunks[0]['dir'].name} "
                    f"has {k}={a!r}; refusing to merge chunks from different configs")


def validate_no_overlapping_seeds(chunks: list[dict]) -> list[int]:
    seen: dict[int, str] = {}
    all_seeds: list[int] = []
    for c in chunks:
        seeds = c["meta"].get("seeds", [])
        for s in seeds:
            s = int(s)
            if s in seen:
                raise ConcatError(
                    f"seed {s} appears in both {seen[s]} and {c['dir'].name}; "
                    "refusing to merge chunks with overlapping seeds")
            seen[s] = c["dir"].name
            all_seeds.append(s)
    return sorted(all_seeds)


def merge(chunks: list[dict], out_tag: str,
          all_seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    games = pd.concat([c["games"] for c in chunks], ignore_index=True)
    dup = games.duplicated(subset=["game_id", "seed"]).sum()
    if dup:
        # Should be unreachable once validate_no_overlapping_seeds (called
        # BEFORE this, in main()) has passed -- kept as an independent
        # assertion in case a chunk's games.parquet disagrees with its own
        # run_meta 'seeds' list.
        raise ConcatError(
            f"{dup} duplicate (game_id, seed) row(s) in the concatenated games frame -- "
            "chunks are not disjoint even though their declared seed lists did not overlap")

    have_players = [c["has_players"] for c in chunks]
    if all(have_players):
        players = pd.concat([c["players"] for c in chunks], ignore_index=True)
    elif not any(have_players):
        players = None
    else:
        missing = [c["dir"].name for c in chunks if not c["has_players"]]
        raise ConcatError(
            f"players.parquet present in some chunks but not {missing}; pass "
            "--allow-partial-players to merge anyway (players.parquet will be DROPPED "
            "from the output, not partially included)")

    first_meta = chunks[0]["meta"]
    diagnostics: dict = {}
    for c in chunks:
        for k, v in (c["meta"].get("diagnostics") or {}).items():
            diagnostics[k] = diagnostics.get(k, 0) + v

    total_poss = sum(int(c["meta"].get("possessions_simulated", 0)) for c in chunks)
    total_runtime = sum(float(c["meta"].get("runtime_s", 0.0)) for c in chunks)
    any_partial = any(bool(c["meta"].get("partial", False)) for c in chunks)
    total_dropped = sum(int(c["meta"].get("seeds_dropped_incomplete", 0)) for c in chunks)
    total_n_requested = sum(int(c["meta"].get("n_seeds_requested", 0)) for c in chunks)

    meta = {
        "engine_tag": f"engine_v0/{out_tag}",
        "created_at": datetime.now(UTC).isoformat(),
        "seeds": all_seeds,
        "n_seeds": len(all_seeds),
        "n_seeds_requested": total_n_requested,
        "fold": first_meta["fold"],
        "backtest": first_meta["backtest"],
        "sealed_touched": first_meta["sealed_touched"],
        "partial": any_partial,
        "seeds_dropped_incomplete": total_dropped,
        "season": first_meta.get("season"),
        "n_games": first_meta.get("n_games"),
        "n_rows": int(len(games)),
        "possessions_simulated": total_poss,
        "runtime_s": round(total_runtime, 1),
        "possessions_per_second": round(total_poss / max(total_runtime, 1e-9), 1),
        "workers": [c["meta"].get("workers") for c in chunks],
        "ast_is_placeholder": first_meta.get("ast_is_placeholder", True),
        "adapter_flags": first_meta.get("adapter_flags"),
        "engine_rules_from_data": first_meta.get("engine_rules_from_data"),
        "inputs_meta": first_meta.get("inputs_meta"),
        "diagnostics": diagnostics,
        "chunks": [
            {
                "tag": c["dir"].name,
                "dir": str(c["dir"]),
                "seed_offset": c["seed_offset"],
                "n_seeds": len(c["meta"].get("seeds", [])),
                "seeds": [int(s) for s in c["meta"].get("seeds", [])],
                "created_at": c["meta"].get("created_at"),
                "runtime_s": c["meta"].get("runtime_s"),
                "possessions_simulated": c["meta"].get("possessions_simulated"),
            }
            for c in chunks
        ],
        "merged_from": {
            "tool": "scripts/concat_engine_runs.py",
            "merged_at": datetime.now(UTC).isoformat(),
            "n_chunks": len(chunks),
        },
    }
    return games, players, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True,
                    help="base tag; chunks are <tag>_off<N>_n<K>/ under --results-dir")
    ap.add_argument("--results-dir", default="results/engine_v0")
    ap.add_argument("--out-tag", default=None,
                    help="output directory name under --results-dir (default: --tag)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting an existing output directory")
    ap.add_argument("--allow-partial-players", action="store_true",
                    help="merge even if players.parquet exists in only some chunks "
                         "(it is then DROPPED from the output, never partially included)")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    out_tag = args.out_tag or args.tag
    out_dir = results_root / out_tag

    chunk_dirs = find_chunks(results_root, args.tag)
    if not chunk_dirs:
        print(f"[concat] no chunks found matching {results_root}/{args.tag}_off*_n* ", file=sys.stderr)
        return 2
    if out_dir.exists() and not args.overwrite:
        print(f"[concat] FATAL: {out_dir} already exists; pass --overwrite to replace it "
              "(never overwritten silently -- CLAUDE.md worker discipline).", file=sys.stderr)
        return 2

    print(f"[concat] {len(chunk_dirs)} chunk(s) for tag '{args.tag}':")
    for d in chunk_dirs:
        print(f"  {d}")

    try:
        chunks = [load_chunk(d) for d in chunk_dirs]
        validate_consistency(chunks)
        all_seeds = validate_no_overlapping_seeds(chunks)  # named PM requirement: refuse overlap
        if args.allow_partial_players:
            # Bypass the strict all-or-nothing check inside merge() by
            # pre-clearing player frames when not universal, per the
            # documented "drop entirely, never partial" rule.
            have = [c["has_players"] for c in chunks]
            if any(have) and not all(have):
                print("[concat] --allow-partial-players: players.parquet not present in "
                      "every chunk; dropping it from the merged output.", file=sys.stderr)
                for c in chunks:
                    c["has_players"] = False
                    c["players"] = None
        games, players, meta = merge(chunks, out_tag, all_seeds)
    except (ContractError, ConcatError) as e:
        print(f"[concat] FATAL: {e}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out_dir / "games.parquet", index=False)
    if players is not None:
        players.to_parquet(out_dir / "players.parquet", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"[concat] wrote {out_dir}: {len(games):,} game rows "
          f"({'no players.parquet' if players is None else f'{len(players):,} player rows'}), "
          f"{meta['n_seeds']} distinct seeds, partial={meta['partial']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
