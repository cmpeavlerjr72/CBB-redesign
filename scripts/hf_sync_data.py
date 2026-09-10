"""Sync the gitignored bulk data dirs to/from a private HuggingFace dataset.

Source/target: https://huggingface.co/datasets/mvpeav/cbb-sim-data (PRIVATE)
Covers: data/raw, results

These two dirs are in .gitignore (rebuildable bulk, and results/raw pulls can
exceed GitHub's size limits), so a `git clone` does not bring them. This
script is the offline/second-device path: `pull` on the new machine instead
of re-running the download / pull_*.py chain.

Everything else the sim needs -- data/processed models+lookups and
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
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REPO_ID = "mvpeav/cbb-sim-data"
BULK_DIRS = ["raw", "results"]

# Single wave -- unlike CFB there is no multi-wave priority split here yet.
PUSH_WAVES = [
    ("wave1-all", ["raw/**", "results/**"]),
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


def _root_for(d: str) -> Path:
    return RESULTS_DIR if d == "results" else DATA_DIR / d


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

    # Both bulk dirs live at the repo root (data/raw, results), not both
    # under data/, so push each dir as its own folder_path with a matching
    # path_in_repo prefix.
    failed = []
    for d in dirs:
        root = _root_for(d)
        if not root.is_dir():
            log(f"--- {d}: no local dir, skipping")
            continue
        log(f"--- {d}: uploading from {root}")
        ok = with_retry(
            f"push-{d}",
            # huggingface-hub >= 1.x dropped path_in_repo from
            # upload_large_folder; upload_folder still supports it and is
            # fine at this size (single commit, ~1-2 GB).
            lambda d=d, root=root: HfApi(token=token).upload_folder(
                repo_id=REPO_ID, folder_path=str(root), repo_type="dataset",
                path_in_repo=d, commit_message=f"sync {d}",
            ),
            max_attempts=max_attempts,
        )
        if not ok:
            failed.append(d)
    log(f"push finished; failed dirs: {failed or 'none'}")
    status(token)


def pull(dirs: list[str], token: str, max_attempts: int) -> None:
    from huggingface_hub import snapshot_download

    log(f"pull <- {REPO_ID}; dirs={dirs}")
    for d in dirs:
        root = _root_for(d)
        root.mkdir(parents=True, exist_ok=True)
        with_retry(
            f"pull-{d}",
            lambda d=d, root=root: snapshot_download(
                repo_id=REPO_ID, repo_type="dataset", local_dir=str(root),
                allow_patterns=[f"{d}/**"], token=token, max_workers=8,
            ),
            max_attempts=max_attempts,
        )
        n = sum(1 for f in root.rglob("*") if f.is_file()) if root.is_dir() else 0
        log(f"  {d}: {n} files")


def remote_files(token: str) -> set:
    from huggingface_hub import HfApi
    files = HfApi(token=token).list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    return {f for f in files if not f.startswith(".")}


def local_files(dirs: list[str]) -> set:
    out = set()
    for d in dirs:
        root = _root_for(d)
        if root.is_dir():
            out |= {f"{d}/{p.relative_to(root).as_posix()}"
                    for p in root.rglob("*") if p.is_file()}
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
                    help="subset of the bulk dirs (default: both)")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="retries per wave; 0 = forever (default, for overnight runs)")
    args = ap.parse_args()
    token = resolve_token()
    if args.action == "status":
        status(token)
    elif args.action == "push":
        push(args.dirs, token, args.max_attempts)
    else:
        pull(args.dirs, token, args.max_attempts)


if __name__ == "__main__":
    main()
