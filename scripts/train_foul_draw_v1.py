"""train_foul_draw_v1.py -- shot-location round 1, Read A foul channel (experiments.md 1.3): and-one on MADE located
FGA_jump2 / FGA_3 attempts, the served fg_make feature set reused as the base, arms R / A1 / A2.

PARTIAL BY CONSTRUCTION: the feed logs no field-goal attempt for a missed shot on which a shooting foul was called,
so the and-one rate on made attempts is the only observable location-conditioned foul quantity. It is never described
as "foul rate by location".

This is `train_fg_make_v6_loc_v1.py` run with `--target and_one` (same wrapper, same served trainer, same population
filter discipline); every other option is passed through unchanged.

    .venv/Scripts/python.exe scripts/train_foul_draw_v1.py --arm A1 --fold F2 --seed 0 --arms B1 --no-floor \
        --no-leak --n-jobs 6 --out-dir data/processed/models/shot_location/round1/andone/A1_s0_F2
"""
import runpy
import sys
from pathlib import Path

if any(x.startswith("--target") for x in sys.argv[1:]):
    raise SystemExit("--target is fixed to and_one by this script")
sys.argv = [sys.argv[0], "--target", "and_one", *sys.argv[1:]]
runpy.run_path(str(Path(__file__).resolve().parent / "train_fg_make_v6_loc_v1.py"), run_name="__main__")
