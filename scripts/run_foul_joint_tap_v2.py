"""run_foul_joint_tap_v2.py -- versioned sibling of run_foul_joint_tap_v1.py (foul round 9, lane C,
2026-09-30): adds `--sample-file` (the verified 500-game sample, as
`run_po4b_closed_loop_sample_v1.py`) and records the possession START TYPE
(`prev_end` code at open) per possession. Everything else is v1 unchanged; v1's docstring follows.

run_foul_joint_tap_v1.py -- instrumented paired closed loop for the joint
foul-accrual / FT-trip round (possession_outcome experiments.md section 20).

MEASUREMENT HARNESS. NOTHING IN `src/cbb_sim/` IS CHANGED BY THIS SCRIPT.
The instrumentation is the in-process tap pattern `diag_g4_tap_v1.py`
established: `StreamBook`, `categorical`, `state.new_state` and `_foul_p` are
wrapped IN THIS PROCESS ONLY and every wrapper returns the real value
unchanged, so the simulation is bit-identical to an un-tapped run. That claim
is CHECKED: the reference arm's `games.parquet` is compared column for column
against `results/engine_v0/po4b_R_s25` (same 500 games, same 25 seeds).

Everything else (subset rule, served-stack pins, block layout) is
`run_po4b_closed_loop.py`'s, imported rather than copied.

Per possession it records the OPEN state (period, clock, margin, both team
foul counts, bonus flags), the possession's FT-trip classes drawn, the trip
fouls charged to the defence before the silent draw (and-ones included), the
silent-foul draw, and the FTA / FGA the possession produced. Game level:
`games_v2.parquet` = the engine's games table PLUS `home_team_id`,
`away_team_id`, `neutral`, `game_date` (a versioned sibling; `games.parquet`
is written unchanged for the parity check).

    .venv/Scripts/python.exe scripts/run_foul_joint_tap_v1.py --tag fj_R_s25 \
        [--env ENGINE_FOUL_ACCRUAL=round6_F5e ...] [--workers 6] [--seeds 25]
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

import run_po4b_closed_loop as P  # noqa: E402

_W: dict = {}


class Tap:
    """Per-worker recorder. One instance, re-armed per simulate_chunk call."""

    def __init__(self):
        self.st = None
        self.main_book = None
        self.recs: list[np.ndarray] = []
        self.pending = None
        self.silent_const = None

    def arm(self, st):
        n = len(st.game_index)
        self.st = st
        self.o_def = np.zeros(n, np.int16)
        self.o_off = np.zeros(n, np.int16)
        self.o_fta = np.zeros(n, np.int32)
        self.o_fga = np.zeros(n, np.int32)
        self.o_per = np.zeros(n, np.int8)
        self.o_sec = np.zeros(n, np.int16)
        self.o_sd = np.zeros(n, np.int16)
        self.o_side = np.zeros(n, np.int8)
        self.o_prev = np.zeros(n, np.int8)
        self.n_sh = np.zeros(n, np.int16)
        self.n_bo = np.zeros(n, np.int16)
        self.n_ch = np.zeros(n, np.int16)
        self.ev_rows = None

    def fga(self, rows, side):
        b = self.st.box
        return (b["fga2_rim"][rows, side] + b["fga2_jump"][rows, side]
                + b["fga3"][rows, side]).astype(np.int32)

    def open(self, rows):
        st = self.st
        off = st.off[rows].astype(np.int64)
        self.o_side[rows] = off
        self.o_prev[rows] = st.prev_end[rows]
        self.o_def[rows] = st.team_fouls[rows, 1 - off]
        self.o_off[rows] = st.team_fouls[rows, off]
        self.o_fta[rows] = st.box["fta"][rows, off]
        self.o_fga[rows] = self.fga(rows, off)
        self.o_per[rows] = st.period[rows]
        self.o_sec[rows] = st.seconds_remaining[rows]
        self.o_sd[rows] = st.off_score_diff()[rows]
        self.n_sh[rows] = 0
        self.n_bo[rows] = 0
        self.n_ch[rows] = 0

    def close(self, rows, u):
        st = self.st
        off = self.o_side[rows].astype(np.int64)
        trip_f = st.team_fouls[rows, 1 - off].astype(np.int32) - self.o_def[rows]
        fta = st.box["fta"][rows, off].astype(np.int32) - self.o_fta[rows]
        fga = self.fga(rows, off) - self.o_fga[rows]
        rec = np.column_stack([
            st.seed[rows], st.game_index[rows], off, self.o_per[rows], self.o_sec[rows],
            self.o_sd[rows], self.o_def[rows], self.o_off[rows], st.bonus_prior_fouls[rows],
            st.double_bonus_prior_fouls[rows], self.n_ch[rows], self.n_sh[rows],
            self.n_bo[rows], trip_f, fta, fga, self.o_prev[rows]]).astype(np.int32)
        uu = np.asarray(u, np.float64)
        self.recs.append((rec, uu, None))
        self.pending = len(self.recs) - 1

    def set_p(self, p, p_off=None):
        if self.pending is not None:
            rec, uu, _ = self.recs[self.pending]
            po = np.zeros(len(uu)) if p_off is None else np.asarray(p_off, np.float64)
            self.recs[self.pending] = (rec, uu, (np.asarray(p, np.float64), po))
            self.pending = None


REC_COLS = ["seed", "gidx", "off_side", "period", "sec", "sd", "def_fouls", "off_fouls",
            "bonus_thr", "dbonus_thr", "n_chances", "n_shoot_trip", "n_bonus_trip",
            "trip_fouls", "fta", "fga", "prev_end"]


def _install_tap(silent_const: float) -> Tap:
    from cbb_sim.engine import loop as L
    from cbb_sim.engine import rng as R
    from cbb_sim.engine import state as S
    from cbb_sim.models import possession_outcome as PO

    T = Tap()
    T.silent_const = silent_const
    n_po = len(PO.CLASSES)
    c_sh, c_bo = PO.CLASS_INDEX["FT_trip_shooting"], PO.CLASS_INDEX["FT_trip_bonus"]

    class TapBook(R.StreamBook):
        def __init__(self, seeds, game_ids, families=R.FAMILIES):
            super().__init__(seeds, game_ids, families)
            self._is_main = tuple(families) == tuple(R.FAMILIES)
            if self._is_main:
                T.main_book = self

        def draw(self, family, rows=None):
            u = super().draw(family, rows)
            if self._is_main and T.st is not None and rows is not None:
                if family == "clock":
                    T.open(np.asarray(rows))
                elif family == "event":
                    T.ev_rows = np.asarray(rows)
                elif family == "foul_accrual":
                    T.close(np.asarray(rows), u)
                    if L._load_foul_lut(os.environ.get("ENGINE_FOUL_ACCRUAL", "reference")) is None \
                            and not os.environ.get("ENGINE_FOUL_JOINT"):
                        T.set_p(np.full(len(rows), T.silent_const))
            return u

    real_cat = L.categorical

    def cat_tap(u, probs):
        out = real_cat(u, probs)
        if (len(u) and probs.shape[1] == n_po and T.ev_rows is not None
                and len(T.ev_rows) == len(u)):
            r = T.ev_rows
            T.n_ch[r] += 1
            T.n_sh[r] += (out == c_sh)
            T.n_bo[r] += (out == c_bo)
            T.ev_rows = None
        return out

    real_new = S.new_state

    def new_state_tap(*a, **k):
        st = real_new(*a, **k)
        T.arm(st)
        return st

    real_fp = L._foul_p

    def foul_p_tap(*a, **k):
        p = real_fp(*a, **k)
        T.set_p(p)
        return p

    L.StreamBook = TapBook
    L.categorical = cat_tap
    S.new_state = new_state_tap
    L._foul_p = foul_p_tap
    # a joint-round module, if any, reports its silent probability through the
    # same hook (`foul_joint.TAP_HOOK`)
    try:
        from cbb_sim.engine import foul_joint as FJ
        FJ.TAP_HOOK = T.set_p
    except ImportError:
        pass
    return T


def _init_worker(tag, fold, season, input_dir, flags):
    for k in P._PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)
    sys.path.insert(0, str(ROOT / "src"))
    from cbb_sim.engine import adapters as AD
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    if os.environ.get("R9_ENGINE_DIR"):
        # v2: the docker-mount equivalent for a LOCAL run, in this process only: the adapters'
        # fixed artifact dir is pointed at a scratch dir holding the tag's event overlay plus
        # copies of the served joblibs (`scripts/ops_r9_engine_dir_v1.py`).
        AD.ENGINE_DIR = Path(os.environ["R9_ENGINE_DIR"])
    inp = EngineInputs.load(input_dir, tag)
    _W["inp"] = inp
    _W["ad"] = Adapters.load(inp, fold, season)
    _W["tap"] = _install_tap(float(inp.rules["silent_foul_per_possession"]))


def _run_block(job):
    from cbb_sim.engine import loop as L
    game_rows, seeds, keep_players = job
    inp, ad, T = _W["inp"], _W["ad"], _W["tap"]
    gi = np.repeat(np.asarray(game_rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(game_rows))
    T.recs = []
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=keep_players)
    recs = []
    for rec, uu, p in T.recs:
        if p is None:
            raise RuntimeError("silent-foul probability not captured for a possession block")
        pd_, po_ = p
        sil = (uu < pd_).astype(np.int32)
        offf = ((uu >= pd_) & (uu < pd_ + po_)).astype(np.int32)
        recs.append(np.column_stack([rec, sil, (pd_ * 1e6).astype(np.int32), offf]))
    R = np.vstack(recs)
    T.st = None
    return res.games, res.players, res.diag, res.n_possessions, R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games-per-block", type=int, default=20)
    ap.add_argument("--n-games", type=int, default=P.SUBSET_N)
    ap.add_argument("--no-players", action="store_true")
    ap.add_argument("--results-dir", default="results/engine_v0")
    ap.add_argument("--input-dir", default=str(P.INPUT_DIR))
    ap.add_argument("--sample-file", default=None)
    ap.add_argument("--all-games", action="store_true", help="every game of the input slate (full size)")
    ap.add_argument("--agg-halves", action="store_true",
                    help="write half_agg.parquet (per seed x game x offence side x half sums) "
                         "instead of the per-possession poss_tap.parquet")
    args = ap.parse_args()
    if int(args.season) >= 2026:
        raise SystemExit("season is SEALED")
    drift = P._assert_served_stack(False)
    from run_engine import engine_provenance
    prov = engine_provenance()
    extra = dict(kv.split("=", 1) for kv in args.env)
    flags = dict(P.PINNED_SUBMODELS)
    flags["ENGINE_EVENT"] = "round2_s1"
    flags.update(extra)
    for k, v in flags.items():
        os.environ[k] = v
    t0 = time.time()
    from cbb_sim.engine.inputs import EngineInputs
    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    if args.all_games:
        rows = np.arange(len(inp.games), dtype=np.int64)
    elif args.sample_file:
        ids = np.sort(pd.read_parquet(args.sample_file)["game_id"].to_numpy().astype("int64"))
        pos = {int(g): k for k, g in enumerate(inp.games["game_id"].to_numpy())}
        miss = [int(g) for g in ids if int(g) not in pos]
        if miss:
            raise SystemExit(f"{len(miss)} sample ids not in inputs, e.g. {miss[:5]}")
        rows = np.array([pos[int(g)] for g in ids], dtype=np.int64)[: args.n_games]
    else:
        rows = P.subset_rows(inp.games)[: args.n_games]
    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    jobs = [(rows[g0:g0 + args.games_per_block], seeds, not args.no_players)
            for g0 in range(0, len(rows), args.games_per_block)]
    print(f"foul-joint tap {args.tag}: {len(rows)} games x {len(seeds)} seeds, "
          f"env {extra}, {args.workers} workers, {len(jobs)} blocks", flush=True)
    gf, pf, recs = [], [], []
    diag: dict = {}
    n_poss = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, int(args.season),
                                       args.input_dir, flags)) as ex:
        futs = [ex.submit(_run_block, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            g, p, d, np_, R = fut.result()
            gf.append(g)
            if p is not None and len(p):
                pf.append(p)
            recs.append(R)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
            n_poss += np_
            if i % 5 == 0:
                print(f"  {i}/{len(jobs)} blocks {time.time() - t0:.0f}s", flush=True)
    games = pd.concat(gf, ignore_index=True)
    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)
    meta_cols = inp.games[["game_id", "home_team_id", "away_team_id", "neutral", "game_date"]]
    games.merge(meta_cols, on="game_id", how="left", validate="many_to_one").to_parquet(
        out / "games_v2.parquet", index=False)
    if pf:
        pd.concat(pf, ignore_index=True).to_parquet(out / "players.parquet", index=False)
    R = pd.DataFrame(np.vstack(recs), columns=REC_COLS + ["silent", "p_silent_x1e6", "off_foul"])
    R["game_id"] = inp.games["game_id"].to_numpy()[R["gidx"].to_numpy()]
    if args.agg_halves:
        # v2: compact per (seed, game, offence side, half) sums instead of the per-possession file
        R["half"] = np.where(R["period"] >= 3, 3, R["period"]).astype("int8")
        R["and_one"] = R["trip_fouls"] - R["n_shoot_trip"] - R["n_bonus_trip"]
        R["poss"] = 1
        R["in_bonus"] = (R["def_fouls"] >= R["bonus_thr"]).astype("int32")
        A = R.groupby(["seed", "game_id", "off_side", "half"])[
            ["poss", "in_bonus", "fta", "fga", "n_shoot_trip", "n_bonus_trip", "and_one", "trip_fouls",
             "silent", "off_foul"]].sum().reset_index()
        A.to_parquet(out / "half_agg.parquet", index=False)
    else:
        R.to_parquet(out / "poss_tap.parquet", index=False)
    meta = {"tag": args.tag, "created_at": datetime.now(UTC).isoformat(),
            "round": "possession_outcome experiments.md section 20 (joint foul round)",
            "flags": flags, "extra_env": extra, "seeds": [int(s) for s in seeds],
            "n_games": int(len(rows)), "n_rows": int(len(games)),
            "served_stack_drift": drift, **prov, "possessions_simulated": int(n_poss),
            "runtime_s": round(time.time() - t0, 1), "workers": args.workers,
            "diagnostics": diag, "sample_file": args.sample_file,
            "fold": args.fold, "season": int(args.season), "backtest": True,
            "input_dir": args.input_dir,
            "env_CBB_TRUTH": os.environ.get("CBB_TRUTH")}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}: {len(games)} rows, {len(R)} possessions, {time.time() - t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
