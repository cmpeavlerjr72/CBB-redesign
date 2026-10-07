"""Preflight: does this checkout hold every input the box jobs read? (lane J 2026-09-30)

    .venv/Scripts/python.exe scripts/ops_box_inputs_check_v1.py            # local presence + size
    .venv/Scripts/python.exe scripts/ops_box_inputs_check_v1.py --remote   # also: on HF or git-tracked

Run it LOCALLY before the launch (with --remote, needs HF token in .env) and again ON THE BOX after
the pulls (no --remote needed). Exit code 1 if anything is missing. The list is the union of the
empirical read trace (scripts/ops_trace_reads_v1.py on the par_v1 identity runs) and the code paths
the identity runs do not reach (conference flags, shot_block feed, leak test skipped, engine v3).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM_TABLES = ["data/processed/team_rate_features_E3_v2.parquet",
               "data/processed/team_rate_features_E3opp_v2.parquet"]
JOBS = {
    "possession_outcome S1 (R2/T/Topp)": [
        "data/processed/models/possession_outcome/round2/design.parquet",
        "data/processed/models/possession_outcome/round2/verdict.json",
        "data/processed/models/engine/games_F2_2025.parquet", *TEAM_TABLES],
    "fg_make round4 B1 (R2/T)": [
        "data/processed/models/fg_make/design_v2_shotshooter.parquet",
        "data/processed/models/fg_make/events_v2_shotshooter.parquet",
        "data/processed/models/fg_make/design_v4_extra_v2.parquet",
        "data/processed/models/fg_make/lgbm_ladder_v2.json", *TEAM_TABLES[:1]],
    "rebound S1_weekly stage 2 (+ --feeds)": [
        "data/processed/models/rebound/round3/design_round3.parquet",
        "data/processed/games_universe.parquet",
        "data/raw/hoopr/schedules/mbb_schedule_2022.parquet",
        "data/raw/hoopr/schedules/mbb_schedule_2023.parquet",
        "data/raw/hoopr/schedules/mbb_schedule_2024.parquet",
        "data/raw/hoopr/schedules/mbb_schedule_2025.parquet",
        "data/processed/models/fg_make/events_v2_shotshooter.parquet", *TEAM_TABLES[:1]],
    "engine 200-seed loops (v2 default + v3 sibling)": [
        "data/processed/models/engine/games_F2_2025_v2.parquet",
        "data/processed/models/engine/arrays_F2_2025_v2.npz",
        "data/processed/models/engine/event_round2_s1_F2_2025/index.json",
        "data/processed/models/engine_v3/games_F2_2025.parquet",
        "data/processed/models/engine_v3/arrays_F2_2025.npz",
        "data/processed/models/engine_v3/names_F2_2025.json",
        "docs/ops/parity_reference_windows_v6.json",
        # served v3 (2026-10-07): v10 = default (clock K2), v9 = SERVED_V2
        "docs/ops/parity_reference_windows_v9.json", "docs/ops/parity_reference_windows_v10.json",
        "data/processed/models/clock/r8_K2/F2/manifest.json",
        "data/processed/models/clock/r8_K2/v5b_bakeoff/v5b_bakeoff_report.json"],
}


def tracked(rel: str) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                          capture_output=True).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true")
    a = ap.parse_args()
    remote = set()
    if a.remote:
        sys.path.insert(0, str(ROOT / "scripts"))
        import hf_sync_data as H
        remote = H.remote_files(H.resolve_token())
    bad = 0
    for job, files in JOBS.items():
        print(f"\n== {job}")
        for rel in files:
            p = ROOT / rel
            here = p.is_file()
            sz = f"{p.stat().st_size / 1e6:8.1f} MB" if here else "   MISSING "
            note = ""
            if a.remote:
                if tracked(rel):
                    note = "git"
                else:
                    cands = [f"model_artifacts/{rel[len('data/processed/models/'):]}"
                             if rel.startswith("data/processed/models/") else "",
                             f"raw/{rel[len('data/raw/'):]}" if rel.startswith("data/raw/") else "",
                             f"engine_inputs_v3/{rel[len('data/processed/models/engine_v3/'):]}"
                             if rel.startswith("data/processed/models/engine_v3/") else "",
                             f"team_rate_tables/{Path(rel).name}" if "team_rate_features_" in rel else "",
                             f"engine_inputs/{rel[len('data/processed/models/engine/'):]}"
                             if rel.startswith("data/processed/models/engine/") else ""]
                    note = "HF" if any(c and c in remote for c in cands) else "NOT ON HF, NOT IN GIT"
                    if note.startswith("NOT"):
                        bad += 1
            if not here:
                bad += 1
            print(f"  {sz}  {rel}  {note}")
    print(f"\n{'ALL PRESENT' if not bad else str(bad) + ' PROBLEM(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
