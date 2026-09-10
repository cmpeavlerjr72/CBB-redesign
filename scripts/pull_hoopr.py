#!/usr/bin/env python
"""
pull_hoopr.py -- download hoopR men's college basketball parquet files
from the sportsdataverse/hoopR-mbb-data GitHub repo (CC BY 4.0).

Source layout (files committed on `main`, not GitHub releases):
    https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/
        mbb/<dir>/parquet/<stem>_<season>.parquet

hoopR "season" is the *ending* year of the season (2021-22 season -> 2022).

Usage:
    python scripts/pull_hoopr.py
    python scripts/pull_hoopr.py --seasons 2024 2025 2026 --datasets pbp schedules
    python scripts/pull_hoopr.py --out data/raw/hoopr --max-retries 6 --workers 4

Writes each file to  <out>/<dataset>/<stem>_<season>.parquet
and maintains         <out>/manifest.json  (url, bytes, sha256, downloaded_at per file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

import requests

BASE_URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/{dirname}/parquet/{stem}_{season}.parquet"

# dataset CLI key -> (github directory name, filename stem prefix)
DATASET_MAP = {
    "pbp": ("pbp", "play_by_play"),
    "player_box": ("player_box", "player_box"),
    "team_box": ("team_box", "team_box"),
    "schedules": ("schedules", "mbb_schedule"),
    "shots": ("shots", "shots"),
    "rosters": ("rosters", "rosters"),
    "game_rosters": ("game_rosters", "game_rosters"),
    "player_core": ("player_core", "player_core"),
}

# hoopR-mbb-data only carries these two datasets for these seasons (verified
# against the repo tree on 2026-09-10).
ROSTER_LIMITED_DATASETS = {"rosters", "game_rosters"}
ROSTER_LIMITED_SEASONS = {2025, 2026}

DEFAULT_SEASONS = [2022, 2023, 2024, 2025, 2026]
DEFAULT_DATASETS = list(DATASET_MAP.keys())

CHUNK_SIZE = 1 << 20  # 1 MiB
USER_AGENT = "cbb-clean-sheet-pull-hoopr/1.0 (+github.com/sportsdataverse/hoopR-mbb-data consumer)"

log = logging.getLogger("pull_hoopr")


@dataclass
class Target:
    dataset: str
    season: int
    dirname: str
    stem: str
    url: str
    dest: Path
    optional: bool = False  # True => a 404 is not an error (e.g. "next season" schedule)


@dataclass
class Result:
    target: Target
    status: str  # skip | downloaded | not_found | failed
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    error: Optional[str] = None


def build_targets(seasons: list[int], datasets: list[str], out_dir: Path) -> list[Target]:
    targets: list[Target] = []
    for ds in datasets:
        if ds not in DATASET_MAP:
            log.warning("unknown dataset %r, skipping", ds)
            continue
        dirname, stem = DATASET_MAP[ds]

        season_list = seasons
        if ds in ROSTER_LIMITED_DATASETS:
            season_list = [s for s in seasons if s in ROSTER_LIMITED_SEASONS]
            skipped = sorted(set(seasons) - ROSTER_LIMITED_SEASONS)
            if skipped:
                log.info(
                    "%s only exists for seasons %s upstream; skipping requested seasons %s",
                    ds, sorted(ROSTER_LIMITED_SEASONS), skipped,
                )

        for season in season_list:
            fname = f"{stem}_{season}.parquet"
            targets.append(Target(
                dataset=ds, season=season, dirname=dirname, stem=stem,
                url=BASE_URL.format(dirname=dirname, stem=stem, season=season),
                dest=out_dir / ds / fname,
            ))

        if ds == "schedules" and season_list:
            next_season = max(season_list) + 1
            if next_season not in season_list:
                fname = f"{stem}_{next_season}.parquet"
                targets.append(Target(
                    dataset=ds, season=next_season, dirname=dirname, stem=stem,
                    url=BASE_URL.format(dirname=dirname, stem=stem, season=next_season),
                    dest=out_dir / ds / fname,
                    optional=True,
                ))
    return targets


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def head_content_length(session: requests.Session, url: str, timeout: int) -> tuple[Optional[int], int]:
    """Returns (content_length_or_None, http_status)."""
    r = session.head(url, timeout=timeout, allow_redirects=True)
    if r.status_code == 404:
        return None, 404
    r.raise_for_status()
    cl = r.headers.get("Content-Length")
    return (int(cl) if cl is not None else None), r.status_code


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(session: requests.Session, target: Target, max_retries: int, timeout: int,
                  backoff_base: float) -> Result:
    target.dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.dest.with_suffix(target.dest.suffix + ".part")

    try:
        expected_size, status = head_content_length(session, target.url, timeout)
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else None
        if status_code == 404 and target.optional:
            return Result(target, "not_found")
        return Result(target, "failed", error=f"HEAD failed: {e}")
    except requests.RequestException as e:
        expected_size, status = None, None  # fall through and try GET anyway
        status = None
    else:
        if status == 404:
            if target.optional:
                log.info("%s season %s not present upstream (404), skipping (optional)",
                          target.dataset, target.season)
                return Result(target, "not_found")
            return Result(target, "failed", error="404 Not Found")

    if target.dest.exists():
        existing_size = target.dest.stat().st_size
        if expected_size is not None and existing_size == expected_size:
            digest = sha256_of(target.dest)
            log.info("SKIP  %-14s %s  (%s bytes, already complete)", target.dataset, target.dest.name, existing_size)
            return Result(target, "skip", bytes=existing_size, sha256=digest)
        else:
            log.info("existing %s size %s != expected %s, re-downloading",
                      target.dest.name, existing_size, expected_size)
            target.dest.unlink()

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resume_pos = part_path.stat().st_size if part_path.exists() else 0
            headers = {}
            mode = "wb"
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"
                mode = "ab"

            with session.get(target.url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    if target.optional:
                        return Result(target, "not_found")
                    raise IOError("404 Not Found")
                if resume_pos > 0 and r.status_code == 200:
                    # server doesn't support Range -> restart clean
                    resume_pos = 0
                    mode = "wb"
                r.raise_for_status()
                with open(part_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            final_size = part_path.stat().st_size
            if expected_size is not None and final_size != expected_size:
                raise IOError(f"size mismatch after download: got {final_size}, expected {expected_size}")

            part_path.rename(target.dest)
            digest = sha256_of(target.dest)
            log.info("OK    %-14s %s  (%s bytes, attempt %d)", target.dataset, target.dest.name, final_size, attempt)
            return Result(target, "downloaded", bytes=final_size, sha256=digest)

        except Exception as e:  # noqa: BLE001 - retry on anything transient
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = backoff_base * (2 ** (attempt - 1))
            log.warning("retry %d/%d for %s after error: %s (sleeping %.1fs)",
                        attempt, max_retries, target.dest.name, e, sleep_s)
            time.sleep(sleep_s)

    log.error("FAILED %s after %d attempts: %s", target.dest.name, max_retries, last_err)
    return Result(target, "failed", error=str(last_err))


def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("manifest.json unreadable, starting fresh")
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS,
                     help=f"hoopR season(s), ending year (default: {DEFAULT_SEASONS})")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list(DATASET_MAP.keys()),
                     help=f"datasets to pull (default: all -> {DEFAULT_DATASETS})")
    ap.add_argument("--out", type=Path, default=Path("data/raw/hoopr"), help="output root directory")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=90, help="per-request timeout (seconds)")
    ap.add_argument("--backoff-base", type=float, default=2.0, help="base seconds for exponential backoff")
    ap.add_argument("--workers", type=int, default=4, help="concurrent downloads")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest_lock = Lock()

    targets = build_targets(sorted(set(args.seasons)), args.datasets, out_dir)
    log.info("Planned %d file(s) across datasets=%s seasons=%s", len(targets), args.datasets, sorted(set(args.seasons)))

    session = make_session()
    counts = {"skip": 0, "downloaded": 0, "not_found": 0, "failed": 0}
    failed_targets: list[Target] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, session, t, args.max_retries, args.timeout, args.backoff_base): t
            for t in targets
        }
        for fut in as_completed(futures):
            res = fut.result()
            counts[res.status] += 1
            if res.status in ("downloaded", "skip"):
                rel_key = str(res.target.dest.relative_to(out_dir)).replace("\\", "/")
                with manifest_lock:
                    manifest[rel_key] = {
                        "url": res.target.url,
                        "dataset": res.target.dataset,
                        "season": res.target.season,
                        "path": rel_key,
                        "bytes": res.bytes,
                        "sha256": res.sha256,
                        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "status": res.status,
                    }
                    save_manifest(manifest_path, manifest)
            elif res.status == "failed":
                failed_targets.append(res.target)

    log.info(
        "Done. downloaded=%d skipped=%d not_found(optional)=%d failed=%d",
        counts["downloaded"], counts["skip"], counts["not_found"], counts["failed"],
    )
    if failed_targets:
        log.error("Failed files:")
        for t in failed_targets:
            log.error("  %s season=%s -> %s", t.dataset, t.season, t.url)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
