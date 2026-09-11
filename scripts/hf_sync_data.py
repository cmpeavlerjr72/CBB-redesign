"""Sync the gitignored bulk data dirs to/from a private HuggingFace dataset.

Source/target: https://huggingface.co/datasets/mvpeav/cbb-sim-data (PRIVATE)
Covers: data/raw, results, engine_inputs

`raw` and `results` are gitignored wholesale (rebuildable bulk, and
results/raw pulls can exceed GitHub's size limits), so a `git clone` does not
bring them. This script is the offline/second-device path: `pull` on the new
machine instead of re-running the download / pull_*.py chain.

`engine_inputs` maps to `data/processed/models/engine/` -- unlike `raw` and
`results`, most of that directory IS git-tracked (the engine's per-fold
arrays/joblibs), but `event_round2_s1_*/` under it (the S1 possession-outcome
artifacts, ~45 MB) is gitignored and has no other sync path (PM decision
2026-09-10, docs/ops/aws_launch_chain.md section 4). Syncing the whole ~67 MB
directory rather than carving out just the gitignored subdirectory is a
deliberate simplification -- the duplication of already-tracked files on HF
costs little and keeps `_root_for` a plain one-directory-in, one-prefix-out
mapping like `raw` and `results`.

`model_artifacts` maps to the whole `data/processed/models/` tree, but
-- unlike `engine_inputs` -- only the gitignored subset of it is ever synced
(`_gitignored_files`, backed by `git check-ignore --stdin`): most of
`data/processed/models/` is git-tracked (lookup tables, fitted sub-models --
the unrecoverable state this repo protects), and duplicating all of that
onto HF would be wasteful and confusing. This is the generic, growing-set
sync path for round-N / S1 bake-off scratch (e.g.
`data/processed/models/fg_make/round2{,b}/`) that is gitignored per
`.gitignore` but has no dedicated bulk key of its own the way `engine_inputs`
does. Push computes the ignored-file list fresh each run and passes it as
`allow_patterns`, so a file that becomes tracked later automatically drops
out of what gets pushed. Pull is unfiltered (`model_artifacts/**`) because
only ignored files were ever pushed under that prefix, so there is nothing
tracked to accidentally pull back.

Everything else the sim needs -- other data/processed models+lookups and
data/reference -- IS tracked in git and arrives with the clone. Do not add
them here.

Both directions retry with exponential backoff and resume where they left
off (upload_large_folder / snapshot_download keep local resume state), so a
DNS blip is survivable rather than fatal.

Run: .venv/Scripts/python.exe scripts/hf_sync_data.py pull
     .venv/Scripts/python.exe scripts/hf_sync_data.py pull --dirs raw
     .venv/Scripts/python.exe scripts/hf_sync_data.py push
     .venv/Scripts/python.exe scripts/hf_sync_data.py status
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPO_ID = "mvpeav/cbb-sim-data"
BULK_DIRS = ["raw", "results", "engine_inputs", "model_artifacts"]

# Path each bulk key resolves to, relative to whatever root it's rooted at
# (the repo by default; see `_root_for`'s `dest_root` param). This is also
# the HF repo prefix each key pushes under (`push()` uses `path_in_repo=d`),
# which is why `_local_to_remote`/`_remote_to_local_rel` below key off the
# same strings.
_REL_ROOT = {
    "results": Path("results"),
    "engine_inputs": Path("data") / "processed" / "models" / "engine",
    "model_artifacts": Path("data") / "processed" / "models",
}

# Single wave -- unlike CFB there is no multi-wave priority split here yet.
PUSH_WAVES = [
    ("wave1-all", ["raw/**", "results/**", "engine_inputs/**", "model_artifacts/**"]),
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def resolve_token() -> str:
    """HF_TOKEN from the environment, else from .env. Never printed."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        env_file = ROOT / ".env"
        if env_file.exists():
            env = dict(re.findall(r"^\s*([A-Z_]+)\s*=\s*(.+?)\s*$", env_file.read_text(), re.M))
            tok = env.get("HF_TOKEN", "").strip()
    if not tok:
        sys.exit(
            "No HF_TOKEN. Set it in the environment or in .env (see .env.example).\n"
            "The token is a secret -- never committed, never uploaded."
        )
    return tok


def with_retry(label: str, fn, max_attempts: int = 0, base_sleep: int = 30) -> bool:
    """Run fn(), retrying on any exception. max_attempts=0 means forever.

    Backoff caps at 10 min so an overnight DNS outage is ridden out rather
    than burning through the retries in the first two minutes.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            fn()
            log(f"{label}: OK (attempt {attempt})")
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if max_attempts and attempt >= max_attempts:
                log(f"{label}: GAVE UP after {attempt} attempts -- {type(e).__name__}: {e}")
                return False
            sleep = min(base_sleep * (2 ** min(attempt - 1, 5)), 600)
            log(f"{label}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:200]}) "
                f"-- retrying in {sleep}s")
            time.sleep(sleep)


def _root_for(d: str, dest_root: Path = ROOT) -> Path:
    """Local directory a bulk key resolves to, rooted at `dest_root`.

    `dest_root` defaults to the repo (`ROOT`) for normal push/pull/status
    use. `pull()` accepts an override (`--dest-root` on the CLI) so a
    verification pull can land in a scratch directory instead of the repo --
    see docs/ops/aws_launch_chain.md section 12.
    """
    rel = _REL_ROOT.get(d, Path("data") / d)
    return dest_root / rel


# Real-repo-rooted paths, used by the git-ignore-based `model_artifacts`
# scan below (which shells out to `git check-ignore` against ROOT and so is
# only meaningful for the real repo, never a `--dest-root` scratch pull).
RESULTS_DIR = _root_for("results")
ENGINE_INPUTS_DIR = _root_for("engine_inputs")
MODEL_ARTIFACTS_DIR = _root_for("model_artifacts")


def _local_to_remote(d: str, rel_path: str) -> str:
    """Map a POSIX path relative to `_root_for(d)` to its path in the HF
    repo. `push()` uploads each bulk dir with `path_in_repo=d`
    (`upload_folder(folder_path=root, path_in_repo=d, ...)`), so a file at
    `root/<rel_path>` lives on HF at `<d>/<rel_path>`."""
    rel_path = rel_path.strip("/")
    return f"{d}/{rel_path}" if rel_path else d


def _remote_to_local_rel(d: str, remote_path: str) -> str:
    """Inverse of `_local_to_remote`: strip the `<d>/` prefix HF preserves
    on disk, returning a path relative to `_root_for(d)`.

    This is the mapping `pull()` used to skip: it passed `local_dir=root`
    straight to `snapshot_download`, which mirrors each file's *full* repo
    path (including the `<d>/` prefix) under `local_dir`, landing files at
    `root/<d>/<rel_path>` instead of `root/<rel_path>` -- one directory too
    deep (docs/ops/aws_launch_chain.md section 12)."""
    prefix = f"{d}/"
    if not remote_path.startswith(prefix):
        raise ValueError(f"{remote_path!r} does not start with expected prefix {prefix!r}")
    return remote_path[len(prefix):]


def _gitignored_files(root: Path) -> list[Path]:
    """Files under root that git considers ignored (not the tracked ones).

    Backs the `model_artifacts` bulk dir: `data/processed/models/` is mostly
    git-tracked, and only the gitignored scratch under it (round-N / S1
    bake-off artifacts, per `.gitignore`) should ever leave for HF. Uses
    `git check-ignore --stdin`, which echoes back only the lines that match
    an ignore rule, so no other filtering is needed.
    """
    all_files = [p for p in root.rglob("*") if p.is_file()]
    if not all_files:
        return []
    rels = [p.relative_to(ROOT).as_posix() for p in all_files]
    # Binary stdin/stdout on purpose: subprocess's text mode translates "\n"
    # to "\r\n" on Windows when writing to the child's stdin, which corrupts
    # each path with a trailing \r -- git then C-style-quotes the "unusual"
    # path back, breaking a plain splitlines() parse. Encode/decode by hand
    # instead of using text=True.
    payload = ("\n".join(rels) + "\n").encode("utf-8")
    proc = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input=payload, capture_output=True, cwd=ROOT,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    ignored = set(proc.stdout.decode("utf-8").splitlines())
    return [ROOT / r for r in ignored]


def _model_artifacts_files(root: Path) -> list[Path]:
    """Gitignored files under data/processed/models/ for the `model_artifacts`
    bulk dir -- excluding the engine/ subtree, which is `engine_inputs`'s own
    dedicated key. Without this exclusion, `event_round2_s1_*/` (gitignored)
    would get swept up here too and pushed a second time under a different
    HF prefix, duplicating the same ~45MB for no reason."""
    return [p for p in _gitignored_files(root) if ENGINE_INPUTS_DIR not in p.parents]


def push(dirs: list[str], token: str, max_attempts: int) -> None:
    from huggingface_hub import HfApi

    os.environ["HF_TOKEN"] = token  # upload_large_folder reads it from the env
    waves = [(name, pats) for name, pats in PUSH_WAVES
             if any(p.split("/")[0] in dirs for p in pats)]
    log(f"push -> {REPO_ID} (private); {len(waves)} wave(s); dirs={dirs}")
    try:
        outstanding = missing_set(token, dirs)
        log(f"outstanding before push: {len(outstanding)} files")
    except Exception as e:
        outstanding = None
        log(f"could not compute outstanding ({type(e).__name__}) -- running all waves")
    if outstanding is not None and not outstanding:
        log("already fully mirrored; nothing to do")
        status(token)
        return

    # The bulk dirs live at different real paths (data/raw, results,
    # data/processed/models/engine, data/processed/models), so push each as
    # its own folder_path with a matching path_in_repo prefix (_root_for owns
    # that mapping).
    failed = []
    for d in dirs:
        root = _root_for(d)
        if not root.is_dir():
            log(f"--- {d}: no local dir, skipping")
            continue
        upload_kwargs = dict(
            repo_id=REPO_ID, folder_path=str(root), repo_type="dataset",
            path_in_repo=d, commit_message=f"sync {d}",
        )
        if d == "model_artifacts":
            # Only the gitignored subset of data/processed/models/ -- the
            # rest is git-tracked already and must not be duplicated onto HF.
            patterns = [p.relative_to(root).as_posix() for p in _model_artifacts_files(root)]
            if not patterns:
                log(f"--- {d}: no gitignored files under {root}, skipping upload")
                continue
            upload_kwargs["allow_patterns"] = patterns
            log(f"--- {d}: uploading {len(patterns)} gitignored file(s) from {root}")
        else:
            log(f"--- {d}: uploading from {root}")
        ok = with_retry(
            f"push-{d}",
            # huggingface-hub >= 1.x dropped path_in_repo from
            # upload_large_folder; upload_folder still supports it and is
            # fine at this size (single commit, ~1-2 GB).
            lambda kwargs=upload_kwargs: HfApi(token=token).upload_folder(**kwargs),
            max_attempts=max_attempts,
        )
        if not ok:
            failed.append(d)
    log(f"push finished; failed dirs: {failed or 'none'}")
    status(token)


def pull(dirs: list[str], token: str, max_attempts: int, dest_root: Path = ROOT) -> None:
    """Download each bulk dir from HF into `_root_for(d, dest_root)`.

    `snapshot_download(local_dir=...)` mirrors each file's full repo path
    (including the `<d>/` prefix `push()` uploads under) beneath `local_dir`.
    Passing `local_dir=root` directly therefore used to land files at
    `root/<d>/<rel_path>` -- one directory too deep (confirmed on the
    2026-09-11 AWS launch for `engine_inputs`/`model_artifacts`; `raw`/
    `results` share the same `_root_for`/`path_in_repo` shape and were
    reproduced as having the identical bug). Fixed by downloading into a
    scratch staging dir first, then moving each file from
    `staging/<d>/<rel_path>` to `root/<rel_path>` via `_remote_to_local_rel`
    -- the exact inverse of the `_local_to_remote` mapping `local_files()`
    uses to compute what a push would put on the remote, so push-layout and
    pull-layout are now provably the same mapping applied in both
    directions. See docs/ops/aws_launch_chain.md section 12.
    """
    from huggingface_hub import snapshot_download

    log(f"pull <- {REPO_ID}; dirs={dirs}; dest_root={dest_root}")
    for d in dirs:
        root = _root_for(d, dest_root)
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".hf_pull_staging_{d}_", dir=str(root)))
        moved = 0
        try:
            ok = with_retry(
                f"pull-{d}",
                lambda d=d, staging=staging: snapshot_download(
                    repo_id=REPO_ID, repo_type="dataset", local_dir=str(staging),
                    allow_patterns=[f"{d}/**"], token=token, max_workers=8,
                ),
                max_attempts=max_attempts,
            )
            if not ok:
                log(f"pull-{d}: download did not complete; moving whatever was staged")
            staged_root = staging / d
            if staged_root.is_dir():
                for f in staged_root.rglob("*"):
                    if not f.is_file():
                        continue
                    rel = _remote_to_local_rel(d, f.relative_to(staging).as_posix())
                    dest = root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                    moved += 1
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        n = sum(1 for f in root.rglob("*") if f.is_file()) if root.is_dir() else 0
        log(f"  {d}: {n} files ({moved} moved this run)")


def remote_files(token: str) -> set:
    from huggingface_hub import HfApi
    files = HfApi(token=token).list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    return {f for f in files if not f.startswith(".")}


def local_files(dirs: list[str]) -> set:
    out = set()
    for d in dirs:
        root = _root_for(d)
        if not root.is_dir():
            continue
        if d == "model_artifacts":
            # Only the gitignored subset counts as "belongs on HF" here --
            # the rest of data/processed/models/ is tracked and arrives via
            # git clone, not this sync path.
            files = _model_artifacts_files(root)
        else:
            files = [p for p in root.rglob("*") if p.is_file()]
        out |= {_local_to_remote(d, p.relative_to(root).as_posix()) for p in files}
    return out


def missing_set(token: str, dirs: list[str]) -> set:
    """Local files not present on the remote. Set difference, NOT a count
    compare -- counts can match while contents differ."""
    return local_files(dirs) - remote_files(token)


def status(token: str, show: int = 15) -> None:
    remote, missing = remote_files(token), missing_set(token, BULK_DIRS)
    log(f"remote {REPO_ID}: {len(remote)} files")
    for d in BULK_DIRS:
        loc = len(local_files([d]))
        miss = len([m for m in missing if m.startswith(d + "/")])
        flag = "  <-- INCOMPLETE" if miss else ""
        log(f"  {d:11s} local={loc:5d}  missing_on_remote={miss:5d}{flag}")
    for m in sorted(missing)[:show]:
        log(f"      missing: {m}")
    if len(missing) > show:
        log(f"      ... and {len(missing) - show} more")
    log(f"  TOTAL still to upload: {len(missing)} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "pull", "status"])
    ap.add_argument("--dirs", nargs="+", default=BULK_DIRS, choices=BULK_DIRS,
                    help="subset of the bulk dirs (default: all three)")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="retries per wave; 0 = forever (default, for overnight runs)")
    ap.add_argument("--dest-root", type=Path, default=None,
                    help="pull only: local root the bulk dirs resolve under "
                         "(default: the repo). Lets a verification pull land in a "
                         "scratch directory instead of the repo, e.g. to check the "
                         "local<->remote path mapping without touching real data.")
    args = ap.parse_args()
    token = resolve_token()
    if args.action == "status":
        status(token)
    elif args.action == "push":
        push(args.dirs, token, args.max_attempts)
    else:
        dest_root = args.dest_root.resolve() if args.dest_root else ROOT
        pull(args.dirs, token, args.max_attempts, dest_root=dest_root)


if __name__ == "__main__":
    main()
