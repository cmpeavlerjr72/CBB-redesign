"""tests/test_hf_sync_paths.py

Pure-function tests of `scripts/hf_sync_data.py`'s local<->remote path
mapping, for each of the four `BULK_DIRS` keys (`raw`, `results`,
`engine_inputs`, `model_artifacts`). No network: these test the path
arithmetic only, never `push()`/`pull()` themselves (which call HF).

Why this file exists: the 2026-09-11 AWS launch (docs/ops/aws_launch_chain.md
section 12) found `pull()` landing `engine_inputs`/`model_artifacts` files one
directory too deep -- `push()` uploads each bulk dir with `path_in_repo=d`
(e.g. `engine_inputs/arrays_F2_2025.npz`), but `pull()` passed
`local_dir=root` straight to `snapshot_download`, which mirrors the file's
*full* repo path (prefix included) under `local_dir`, landing it at
`root/engine_inputs/arrays_F2_2025.npz` instead of `root/arrays_F2_2025.npz`.
`raw`/`results` share the identical `_root_for`/`path_in_repo` shape and were
suspected (not yet confirmed at the time) to have the same defect.

The fix factors the mapping into two pure, inverse functions:
  - `_local_to_remote(d, rel_path)`: local rel path -> HF repo path
    (already used by `local_files()` to compute what a push would upload).
  - `_remote_to_local_rel(d, remote_path)`: HF repo path -> local rel path
    (now used by the fixed `pull()` to place downloaded files correctly).

These tests prove the two are exact inverses -- i.e. push-layout equals
pull-layout -- for all four keys, both at the repo root and under an
arbitrary `--dest-root` override (the same override `pull()` uses for a
scratch-directory verification pull).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hf_sync_data as HF  # noqa: E402

ALL_KEYS = list(HF.BULK_DIRS)
assert ALL_KEYS == ["raw", "results", "engine_inputs", "model_artifacts"]

# A handful of representative relative paths: a top-level file, a nested
# file, and a deeply nested one (mirrors real shapes, e.g.
# `event_round2_s1_F2_2025/cont_2024-11-01.joblib` under `engine_inputs`).
REL_PATHS = [
    "arrays_F2_2025.npz",
    "names_F2_2025.json",
    "event_round2_s1_F2_2025/cont_2024-11-01.joblib",
    "fg_make/round2b/S_C_s1/winner_FGA_rim.joblib",
]


# ---------------------------------------------------------------------------
# 1. _local_to_remote / _remote_to_local_rel are exact inverses, per key
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", ALL_KEYS)
@pytest.mark.parametrize("rel", REL_PATHS)
def test_round_trip_local_to_remote_to_local(d: str, rel: str):
    remote = HF._local_to_remote(d, rel)
    assert remote == f"{d}/{rel}"
    back = HF._remote_to_local_rel(d, remote)
    assert back == rel


@pytest.mark.parametrize("d", ALL_KEYS)
@pytest.mark.parametrize("rel", REL_PATHS)
def test_round_trip_remote_to_local_to_remote(d: str, rel: str):
    remote_in = f"{d}/{rel}"
    local_rel = HF._remote_to_local_rel(d, remote_in)
    remote_out = HF._local_to_remote(d, local_rel)
    assert remote_out == remote_in


# ---------------------------------------------------------------------------
# 2. _remote_to_local_rel refuses a path that doesn't carry the expected
#    prefix -- guards against a key mixup silently producing a wrong path
#    instead of failing loudly.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", ALL_KEYS)
def test_remote_to_local_rejects_foreign_prefix(d: str):
    other = next(k for k in ALL_KEYS if k != d)
    with pytest.raises(ValueError):
        HF._remote_to_local_rel(d, f"{other}/some_file.bin")


@pytest.mark.parametrize("d", ALL_KEYS)
def test_remote_to_local_rejects_bare_filename(d: str):
    # A path with no "<d>/" prefix at all (e.g. a stray root-level repo file)
    # must raise, not silently return something wrong.
    with pytest.raises(ValueError):
        HF._remote_to_local_rel(d, "some_file.bin")


# ---------------------------------------------------------------------------
# 3. _root_for: the four keys resolve to distinct, correctly-shaped local
#    roots, and an override root reproduces the exact same relative layout
#    as the default (repo) root -- this is what lets a scratch `--dest-root`
#    pull stand in for a real pull in the round-trip check below.
# ---------------------------------------------------------------------------
def test_root_for_default_matches_known_layout():
    assert HF._root_for("raw") == HF.DATA_DIR / "raw"
    assert HF._root_for("results") == HF.ROOT / "results"
    assert HF._root_for("engine_inputs") == HF.DATA_DIR / "processed" / "models" / "engine"
    assert HF._root_for("model_artifacts") == HF.DATA_DIR / "processed" / "models"


def test_root_for_dest_root_override_preserves_relative_layout(tmp_path):
    for d in ALL_KEYS:
        default_rel = HF._root_for(d).relative_to(HF.ROOT)
        override_root = HF._root_for(d, dest_root=tmp_path)
        assert override_root == tmp_path / default_rel


def test_root_for_keys_are_pairwise_distinct(tmp_path):
    roots = {d: HF._root_for(d, dest_root=tmp_path) for d in ALL_KEYS}
    assert len(set(roots.values())) == len(ALL_KEYS)


# ---------------------------------------------------------------------------
# 4. End-to-end path arithmetic (still no network): given files written
#    under `_root_for(d, dest_root)`, `_local_to_remote` maps them to the
#    paths a push would use, and `_remote_to_local_rel` maps those back to
#    exactly where they started -- for every key, under a scratch root.
#    This is the same mapping the fixed `pull()` applies after
#    `snapshot_download` stages files under `<staging>/<d>/...`.
# ---------------------------------------------------------------------------
def test_full_round_trip_under_scratch_dest_root(tmp_path):
    for d in ALL_KEYS:
        root = HF._root_for(d, dest_root=tmp_path)
        root.mkdir(parents=True, exist_ok=True)
        for rel in REL_PATHS:
            local_path = root / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(rel.encode("utf-8"))

        for rel in REL_PATHS:
            local_path = root / rel
            rel_posix = local_path.relative_to(root).as_posix()
            remote = HF._local_to_remote(d, rel_posix)
            assert remote == f"{d}/{rel_posix}"
            recovered_rel = HF._remote_to_local_rel(d, remote)
            recovered_path = root / recovered_rel
            assert recovered_path == local_path
            assert recovered_path.read_bytes() == rel.encode("utf-8")
