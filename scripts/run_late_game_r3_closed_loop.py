"""run_late_game_r3_closed_loop.py -- late-game ROUND 3: one arm through the SERVED v2 engine, tapped.

Pre-registration: `docs/models/late_game/experiments.md` sections 6 and 7 (amendment 61cb7ef).

    .venv/Scripts/python.exe scripts/run_late_game_r3_closed_loop.py --tag lg3_R9_s25
    .venv/Scripts/python.exe scripts/run_late_game_r3_closed_loop.py --late-game clk_Dt --tag lg3_Dt9_s25
    .venv/Scripts/python.exe scripts/run_late_game_r3_closed_loop.py --foul-joint reference --tag lg3_R0_s25

Differences from round 2's `run_late_game_r2_closed_loop.py` (whose tap this reuses unchanged):

  * NO pins. The base is the plain served default (served stack v2, parity v9). The run REFUSES to start
    if any `ENGINE_*` variable is set in the calling environment; the only switches it sets are
    `ENGINE_LATE_GAME` (the arm) and, for the B0 attribution base, `ENGINE_FOUL_JOINT=reference`.
  * Inputs `data/processed/models/engine_v3` (tag F2_2025), sample = the verified minswap 500
    (`--sample-file`), `CBB_TRUTH=verified_v1`.
  * A read-only POSSESSION LOG for every period-2 possession starting at <= 180 s (section 7.6): start
    state (clock, sides' scores, team fouls, bonus flags, intended and used duration) and the possession's
    own box deltas for the offence (FTA, FTM, FGA by type, TOV, OREB, points), closed at the next clock
    call of the same simulation or, for the last possession, from the final state. It returns every
    adapter value unchanged and draws nothing.
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

import run_late_game_r2_closed_loop as R2                             # noqa: E402

DEFAULT_RESULTS = Path("results/engine_v0")
INPUT_DIR = Path("data/processed/models/engine_v3")
SAMPLE_FILE = Path("data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet")
LOG_SEC = 180
_PIN = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS")
CNT = ("fta", "ftm", "fga3", "fga2_rim", "fga2_jump", "tov", "oreb")
_W: dict = {}


class PossLog:
    """Per-chunk possession log (period 2, start <= LOG_SEC)."""

    def new_chunk(self, n: int):
        self.open = np.zeros(n, dtype=bool)
        self.o = {k: np.zeros(n, dtype=np.int32) for k in
                  ("sec", "off", "hp", "ap", "tf_off", "tf_def", "bon", "dbon", "dur", "used")
                  + tuple(f"c_{c}" for c in CNT)}
        self.rows: list[dict] = []

    def close(self, st, sims: np.ndarray):
        sims = sims[self.open[sims]]
        if not len(sims):
            return
        off = self.o["off"][sims]
        rec = {k: v[sims].copy() for k, v in self.o.items() if not k.startswith("c_")}
        rec["sim"] = sims.copy()
        rec["d_pts_off"] = st.pts[sims, off].astype(np.int32) - np.where(off == 0, rec["hp"], rec["ap"])
        rec["d_pts_def"] = st.pts[sims, 1 - off].astype(np.int32) - np.where(off == 0, rec["ap"], rec["hp"])
        for c in CNT:
            rec[f"d_{c}"] = st.box[c][sims, off].astype(np.int32) - self.o[f"c_{c}"][sims]
        rec["end_period"] = st.period[sims].astype(np.int32)
        self.rows.append(rec)
        self.open[sims] = False

    def open_rows(self, st, sims: np.ndarray, dur: np.ndarray):
        if not len(sims):
            return
        off = st.off[sims].astype(np.int64)
        o = self.o
        o["sec"][sims] = st.seconds_remaining[sims]
        o["off"][sims] = off
        o["hp"][sims] = st.pts[sims, 0]
        o["ap"][sims] = st.pts[sims, 1]
        o["tf_off"][sims] = st.team_fouls[sims, off]
        o["tf_def"][sims] = st.team_fouls[sims, 1 - off]
        o["bon"][sims] = st.in_bonus()[sims]
        o["dbon"][sims] = st.in_double_bonus()[sims]
        o["dur"][sims] = np.rint(dur).astype(np.int32)
        o["used"][sims] = np.minimum(np.rint(dur).astype(np.int32), o["sec"][sims])
        for c in CNT:
            o[f"c_{c}"][sims] = st.box[c][sims, off]
        self.open[sims] = True

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        return pd.DataFrame({k: np.concatenate([r[k] for r in self.rows]) for k in self.rows[0]})


PLOG = PossLog()


class ClockTap3(R2.ClockTap):
    def draw(self, team, x, u, *a):
        dur = super().draw(team, x, u, *a)
        st = R2.TAP.st
        act = np.flatnonzero(st.active)
        if len(act) != len(x):
            return dur
        PLOG.close(st, act)
        sel = (st.period[act] == 2) & (st.seconds_remaining[act] <= LOG_SEC)
        PLOG.open_rows(st, act[sel], np.asarray(dur, float)[sel])
        return dur


def _init_worker(tag, fold, season, input_dir, flags):
    for k in _PIN:
        os.environ[k] = "1"
    for k in [k for k in os.environ if k.startswith("ENGINE_")]:
        os.environ.pop(k)
    for k, v in flags.items():
        os.environ[k] = str(v)
    os.environ["CBB_TRUTH"] = "verified_v1"
    sys.path.insert(0, str(ROOT / "src"))
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    ad = Adapters.load(inp, fold, season)
    ad.clock = ClockTap3(ad.clock)
    ad.event = R2.EventTap(ad.event)
    real_new = L.S.new_state

    def new_state(*a, **k):
        st = real_new(*a, **k)
        R2.TAP.new_chunk(st)
        PLOG.new_chunk(len(st.active))
        return st
    L.S.new_state = new_state
    _W.update(inp=inp, ad=ad)


def _run_block(job):
    from cbb_sim.engine import loop as L
    game_rows, seeds = job
    inp, ad = _W["inp"], _W["ad"]
    gi = np.repeat(np.asarray(game_rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(game_rows))
    R2.TAP.clk[:] = 0
    R2.TAP.ev[:] = 0
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=False)
    st = R2.TAP.st
    PLOG.close(st, np.flatnonzero(PLOG.open))
    gids = inp.games["game_id"].to_numpy()[gi]
    T = R2.TAP
    t = pd.DataFrame({"game_id": gids.astype("int64"), "seed": sd.astype("int32"),
                      "h1_home_pts": T.h1_pts[:, 0], "h1_away_pts": T.h1_pts[:, 1],
                      "h1_poss": T.h1_poss, "home_margin_120": T.m120,
                      "fta_at_120": T.fta120, "fta_reg_end_if_ot": T.fta_reg,
                      "window_poss": T.win_poss})
    g = res.games.merge(t, on=["game_id", "seed"], how="left", validate="one_to_one")
    pl = PLOG.frame()
    if len(pl):
        s = pl.pop("sim").to_numpy()
        pl.insert(0, "seed", sd[s].astype("int32"))
        pl.insert(0, "game_id", gids[s].astype("int64"))
    return g, res.diag, res.n_possessions, T.clk.copy(), T.ev.copy(), T.n_calls_mismatch, pl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--late-game", default="off")
    ap.add_argument("--foul-joint", default=None, help="B0 attribution base only: 'reference'")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--games-per-block", type=int, default=25)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--max-games", type=int, default=None, help="timing slice only")
    ap.add_argument("--sample-file", default=str(SAMPLE_FILE))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    args = ap.parse_args()
    if int(args.season) >= 2026:
        raise SystemExit(f"season {args.season} is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context="late-game round 3")
    stray = sorted(k for k in os.environ if k.startswith("ENGINE_"))
    if stray:
        raise SystemExit(f"refusing: ENGINE_* set in the calling env {stray}; the base is the plain default")
    if args.foul_joint not in (None, "reference"):
        raise SystemExit("--foul-joint takes only 'reference' (the B0 attribution base)")
    from run_engine import engine_provenance
    prov = engine_provenance()
    t0 = time.time()
    flags: dict = {}
    if args.late_game != "off":
        flags["ENGINE_LATE_GAME"] = args.late_game
    if args.foul_joint:
        flags["ENGINE_FOUL_JOINT"] = args.foul_joint
    for k in _PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = v
    os.environ["CBB_TRUTH"] = "verified_v1"
    from cbb_sim.engine.inputs import EngineInputs
    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    ids = np.sort(pd.read_parquet(args.sample_file)["game_id"].to_numpy().astype("int64"))
    pos = {int(g): k for k, g in enumerate(inp.games["game_id"].to_numpy())}
    rows = np.array([pos[int(g)] for g in ids], dtype=np.int64)
    if args.max_games:
        rows = rows[:args.max_games]
    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    jobs = [(rows[g0:g0 + args.games_per_block], seeds[s0:s0 + args.seeds_per_block])
            for s0 in range(0, len(seeds), args.seeds_per_block)
            for g0 in range(0, len(rows), args.games_per_block)]
    print(f"lg3 {args.late_game} foul={args.foul_joint or 'served'}: {len(rows)} games x {len(seeds)} "
          f"seeds, {len(jobs)} blocks, {args.workers} workers, commit {prov.get('engine_commit')}",
          flush=True)
    gframes, pframes, diag, n_poss = [], [], {}, 0
    clk = np.zeros((3, 4, 3))
    ev = np.zeros((3, 4, 4))
    mism = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, int(args.season),
                                       args.input_dir, flags)) as ex:
        futs = [ex.submit(_run_block, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            g, d, np_, c, e, mm, pl = fut.result()
            gframes.append(g)
            pframes.append(pl)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + v
            n_poss += np_
            clk += c
            ev += e
            mism = max(mism, mm)
            if i % 5 == 0:
                print(f"  {i}/{len(jobs)} blocks, {n_poss:,} poss, {time.time() - t0:.0f}s", flush=True)
    games = pd.concat(gframes, ignore_index=True).sort_values(["game_id", "seed"]).reset_index(drop=True)
    if games.groupby("seed")["game_id"].nunique().min() != len(rows):
        raise SystemExit("incomplete seeds")
    out = Path(args.results_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    tapcols = ["h1_home_pts", "h1_away_pts", "h1_poss", "home_margin_120", "fta_at_120",
               "fta_reg_end_if_ot", "window_poss"]
    games.drop(columns=tapcols).to_parquet(out / "games.parquet", index=False)
    games[["game_id", "seed"] + tapcols].to_parquet(out / "tap_sims.parquet", index=False)
    plog = pd.concat(pframes, ignore_index=True).sort_values(
        ["game_id", "seed", "sec"], ascending=[True, True, False]).reset_index(drop=True)
    plog.to_parquet(out / "tap_poss.parquet", index=False)
    np.savez(out / "tap_window.npz", clock=clk, event=ev)
    from cbb_sim.engine.adapters import Adapters
    ad = Adapters.load(inp, args.fold, int(args.season))
    el = time.time() - t0
    meta = {
        "engine_tag": f"engine_v0/{args.tag}",
        "round": "late_game round 3 (docs/models/late_game/experiments.md sections 6-7)",
        "created_at": datetime.now(UTC).isoformat(), "arm": args.late_game,
        "base": "B0 (v2 + ENGINE_FOUL_JOINT=reference)" if args.foul_joint else "B9 (served v2)",
        "sample_file": str(args.sample_file), "seeds": [int(s) for s in seeds],
        "n_seeds": int(len(seeds)), "seed_offset": int(args.seed_offset), "fold": args.fold,
        "season": int(args.season), "backtest": True, "sealed_touched": False,
        "partial": bool(args.max_games), "n_games": int(len(rows)), "n_rows": int(len(games)),
        "keep_players": False, "input_dir": str(args.input_dir),
        "game_ids": [int(x) for x in inp.games["game_id"].to_numpy()[rows]],
        **prov, "possessions_simulated": int(n_poss), "runtime_s": round(el, 1),
        "workers": int(args.workers), "tap_row_mismatch_calls": int(mism),
        "n_logged_possessions": int(len(plog)), "adapter_flags": ad.flags, "diagnostics": diag,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({len(games):,} rows, {len(plog):,} logged poss, {el:.0f}s, "
          f"tap mismatches {mism})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
