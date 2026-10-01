"""
exp_aggregation_swap_v1.py -- paired closed-loop channel swaps for the G9 aggregation
over-spread decomposition (docs/models/aggregation/experiments.md section 1).

One arm = the served engine on a tagged v3 input set (S0 = defaults, S1 = E3 v4 + Stage B
T artifacts) with ONE input family replaced IN MEMORY by its league-average value (the
centred `_c` features -> 0, tempo -> its neutral value). Nothing on disk is modified; no
engine module is edited; the swap is applied to this process's copy of the arrays before
`Adapters.load` (team_static / slot_static) and to the loaded event adapter's round-2
team block afterwards. RNG streams are keyed on (seed, game_id, family), so every arm of
one (stack, game set, seed set) is paired with every other.

    .venv/Scripts/python.exe scripts/exp_aggregation_swap_v1.py --stack S0 --arm PO \
        --games-file results/aggregation_v1/games_sample_v1.txt --seeds 16 --workers 4

Writes results/aggregation_v1/<stack>_<arm>_s<seeds>_o<offset>/games.parquet + run_meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_PIN = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS")
OUT_ROOT = ROOT / "results" / "aggregation_v1"
STACK_DIRS = {"S0": "data/processed/models/engine_v3_S0_laneA",
              "S1": "data/processed/models/engine_v3_S1_laneA"}

PO_OFF = ["off_3pa_c", "off_rim_c", "off_tov_c", "off_ftr_c"]
PO_DEF = ["opp_def_3pa_c", "opp_def_rim_c", "opp_def_tov_c", "opp_def_ftr_c"]
FG_OFF = ["off_make_c__rim", "off_make_c__jump2", "off_make_c__three"]
FG_DEF = ["def_allow_c__rim", "def_allow_c__jump2", "def_allow_c__three"]
RB_OFF = ["off_oreb_c"]
RB_DEF = ["opp_def_dreb_c"]
# rating columns, per possession row: off_rating_* = the OFFENCE team's ratings,
# def_rating_* = the DEFENCE team's ratings.
RAT_OFFTEAM = ["off_rating_off_c", "off_rating_def_c"]
RAT_DEFTEAM = ["def_rating_off_c", "def_rating_def_c"]
PACE = ["off_tempo_rel", "def_tempo_rel", "tempo_prior_game"]
PLY = ["shooter_shrunk_dev_c__rim", "shooter_shrunk_dev_c__jump2", "shooter_shrunk_dev_c__three"]

#: arm -> (team columns to neutralise, slot columns to neutralise)
ARMS: dict[str, tuple[list[str], list[str]]] = {
    "FULL": ([], []),
    "PO": (PO_OFF + PO_DEF, []),
    "FG": (FG_OFF + FG_DEF, []),
    "RB": (RB_OFF + RB_DEF, []),
    "RAT": (RAT_OFFTEAM + RAT_DEFTEAM, []),
    "RAT_PO": (RAT_OFFTEAM + RAT_DEFTEAM, []),     # event block ONLY (possession_outcome's copy)
    "PACE": (PACE, []),
    "PLY": ([], PLY),
    "TEAM": (PO_OFF + PO_DEF + FG_OFF + FG_DEF + RB_OFF + RB_DEF + RAT_OFFTEAM + RAT_DEFTEAM, []),
    "OFF": (PO_OFF + FG_OFF + RB_OFF + RAT_OFFTEAM, []),
    "DEF": (PO_DEF + FG_DEF + RB_DEF + RAT_DEFTEAM, []),
    "ALL": (PO_OFF + PO_DEF + FG_OFF + FG_DEF + RB_OFF + RB_DEF + RAT_OFFTEAM + RAT_DEFTEAM + PACE, PLY),
}

_W: dict = {}


def neutral_value(col: str, rules: dict) -> float:
    if col in ("off_tempo_rel", "def_tempo_rel"):
        return 1.0
    if col == "tempo_prior_game":
        return float(rules["clock_tempo_fallback"]["tempo_prior_game"])
    return 0.0          # every other swapped column is centred on its snapshot's league mean


#: arms whose swap reaches ONLY the event adapter's round-2 team block
EVENT_ONLY = {"RAT_PO"}


def apply_swap_inputs(inp, arm: str) -> dict:
    tcols, scols = ARMS[arm]
    if arm in EVENT_ONLY:
        return {"team": [], "slot": []}
    done = {"team": [], "slot": []}
    for c in tcols:
        j = inp.team_names[c]
        inp.team_static[:, :, j] = neutral_value(c, inp.rules)
        done["team"].append(c)
    for c in scols:
        j = inp.slot_names[c]
        inp.slot_static[:, :, :, j] = 0.0
        done["slot"].append(c)
    return done


def apply_swap_event(ad, arm: str, team_cols: list[str]) -> list[str]:
    tcols, _ = ARMS[arm]
    pos = {c: i for i, c in enumerate(team_cols)}
    tb = np.array(ad.event.team_block, copy=True)
    hit = []
    for c in tcols:
        if c in pos:
            tb[:, :, pos[c]] = 0.0
            hit.append(c)
    ad.event.team_block = tb
    return hit


def _load(stack: str, arm: str, adir: str, over: dict):
    for k in _PIN:
        os.environ[k] = "1"
    from cbb_sim.engine import adapters as AD
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    for k, v in over.items():
        setattr(AD, k, Path(v))
    inp = EngineInputs.load(ROOT / STACK_DIRS[stack], "F2_2025")
    sw_in = apply_swap_inputs(inp, arm)
    ad = Adapters.load(inp, "F2", 2025)
    idx = json.loads((Path(over["ENGINE_DIR"]) / "event_round2_s1_F2_2025" / "index.json")
                     .read_text(encoding="utf-8"))
    sw_ev = apply_swap_event(ad, arm, idx["team_cols"])
    return inp, ad, {"inputs": sw_in, "event_block": sw_ev}


def _init(stack, arm, adir, over):
    sys.path.insert(0, str(ROOT / "src"))
    _W["inp"], _W["ad"], _W["swap"] = _load(stack, arm, adir, over)


def _run(job):
    from cbb_sim.engine import loop as L
    rows, seeds = job
    gi = np.repeat(np.asarray(rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(rows))
    res = L.simulate_chunk(_W["inp"], _W["ad"], gi, sd, keep_players=False)
    return res.games, res.n_possessions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=sorted(STACK_DIRS), required=True)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--games-file", default=None, help="one game_id per line")
    ap.add_argument("--all-games", action="store_true", help="every row of the stack's inputs (5,710)")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-block", type=int, default=40)
    ap.add_argument("--tag-suffix", default="")
    args = ap.parse_args()
    for k in _PIN:
        os.environ[k] = "1"
    t0 = time.time()
    from run_engine import engine_provenance
    from run_engine_live import prepare_from_overlay
    from cbb_sim.engine.inputs import EngineInputs
    prov = engine_provenance()
    sdir = ROOT / STACK_DIRS[args.stack]
    adir = OUT_ROOT / "_adapter_dirs" / f"{args.stack}_{args.arm}_{args.seed_offset}{args.tag_suffix}"
    over = {k: str(v) for k, v in prepare_from_overlay(sdir / "overlay", "F2", 2025, adir).items()}
    inp = EngineInputs.load(sdir, "F2_2025")
    gid = inp.games["game_id"].to_numpy()
    if args.all_games:
        rows = np.arange(len(gid), dtype=np.int64)
    else:
        want = [int(x) for x in Path(args.games_file).read_text().split()]
        pos = {int(g): i for i, g in enumerate(gid)}
        rows = np.array(sorted(pos[g] for g in want), dtype=np.int64)
    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    jobs = [(rows[i:i + args.games_per_block], seeds) for i in range(0, len(rows), args.games_per_block)]
    tag = f"{args.stack}_{args.arm}_s{args.seeds}_o{args.seed_offset}{args.tag_suffix}"
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now():%H:%M:%S}] {tag}: {len(rows)} games x {len(seeds)} seeds, "
          f"{len(jobs)} blocks, {args.workers} workers", flush=True)
    frames, n_poss, swap = [], 0, None
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.stack, args.arm, str(adir), over)) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            g, npss = f.result()
            frames.append(g)
            n_poss += npss
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} blocks {time.time() - t0:.0f}s", flush=True)
    _, _, swap = _load(args.stack, args.arm, str(adir), over)   # record what was swapped (parent copy)
    games = pd.concat(frames, ignore_index=True).sort_values(["game_id", "seed"]).reset_index(drop=True)
    games.to_parquet(out / "games.parquet", index=False)
    meta = {"tag": tag, "stack": args.stack, "arm": args.arm, "stack_dir": STACK_DIRS[args.stack],
            "swap": swap, "neutral_rule": "centred _c -> 0; tempo_rel -> 1.0; tempo_prior_game -> rules fallback",
            "n_games": int(len(rows)), "seeds": seeds.tolist(), "workers": args.workers,
            "possessions": int(n_poss), "runtime_s": round(time.time() - t0, 1),
            "created_at": datetime.now(UTC).isoformat(), **prov}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"[{datetime.now():%H:%M:%S}] wrote {out} rows={len(games)} {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
