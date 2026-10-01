"""run_late_game_r4_closed_loop.py -- late-game ROUND 4 tapped closed loop (experiments.md section 9).

Round 3's runner (`run_late_game_r3_closed_loop.py`) with three read-only additions:

  * the possession log covers BOTH halves (period 1 and 2 possessions starting at <= 180 s) and carries the
    offence's made-shot counts by type (`d_fgm3`, `d_fgm2_rim`, `d_fgm2_jump`) and the period;
  * a SHOT log for every field-goal attempt whose possession started at <= 35 s in either half: period,
    possession-start clock, chance number, fed `chance_elapsed_s`, shot class, the probability fg_make
    returned (after the arm, if any), and the offence's live score_diff;
  * generic switches: `--env KEY=VAL` for the round-4 flags only (`ENGINE_LATE_GAME`, `ENGINE_LG_BUZZER`,
    `ENGINE_LG_MAKE`, `ENGINE_FOUL_JOINT=reference`). Nothing else may be set.

    .venv/Scripts/python.exe scripts/run_late_game_r4_closed_loop.py --tag lg4_R9_s25
    .venv/Scripts/python.exe scripts/run_late_game_r4_closed_loop.py --env ENGINE_LATE_GAME=clk_Dt --tag lg4_Dt_s25
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
import run_late_game_r3_closed_loop as R3                             # noqa: E402

ALLOWED_ENV = {"ENGINE_LATE_GAME", "ENGINE_LG_BUZZER", "ENGINE_LG_MAKE", "ENGINE_FOUL_JOINT", "ENGINE_LATE_FOUL"}
SHOT_SEC = 35
R3.CNT = ("fta", "ftm", "fga3", "fga2_rim", "fga2_jump", "tov", "oreb", "fgm3", "fgm2_rim", "fgm2_jump")
_W: dict = {}


class PossLog4(R3.PossLog):
    def new_chunk(self, n):
        super().new_chunk(n)
        self.o["per"] = np.zeros(n, dtype=np.int32)

    def open_rows(self, st, sims, dur):
        super().open_rows(st, sims, dur)
        if len(sims):
            self.o["per"][sims] = st.period[sims]


PLOG = PossLog4()
R3.PLOG = PLOG


class ShotLog:
    def new_chunk(self):
        self.rows: list[dict] = []

    def frame(self):
        if not self.rows:
            return pd.DataFrame()
        return pd.DataFrame({k: np.concatenate([r[k] for r in self.rows]) for k in self.rows[0]})


SLOG = ShotLog()


class ClockTap4(R2.ClockTap):
    def draw(self, team, x, u, *a):
        dur = super().draw(team, x, u, *a)
        st = R2.TAP.st
        act = np.flatnonzero(st.active)
        if len(act) != len(x):
            return dur
        PLOG.close(st, act)
        sel = (st.period[act] <= 2) & (st.seconds_remaining[act] <= R3.LOG_SEC)
        PLOG.open_rows(st, act[sel], np.asarray(dur, float)[sel])
        return dur


class FgTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def predict(self, shot_class, team, slot, xs, gidx=None, *a, **k):
        from cbb_sim.engine.adapters import STATE_INDEX as I
        p = self._r.predict(shot_class, team, slot, xs, gidx, *a, **k)
        sec = xs[:, I["seconds_remaining"]]
        per = xs[:, I["period"]]
        m = (sec <= SHOT_SEC) & (per <= 2)
        if m.any():
            SLOG.rows.append({
                "per": per[m].astype(np.int8), "sec": sec[m].astype(np.int16),
                "chance": xs[m, I["chance_number"]].astype(np.int8),
                "elapsed": xs[m, I["chance_elapsed_s"]].astype(np.float32),
                "score_diff": xs[m, I["score_diff"]].astype(np.int16),
                "cls": np.full(int(m.sum()), {"FGA_rim": 0, "FGA_jump2": 1, "FGA_3": 2}.get(shot_class, 9), np.int8),
                "p": np.asarray(p, float)[m].astype(np.float32)})
        return p


def _init_worker(tag, fold, season, input_dir, flags):
    for k in R3._PIN:
        os.environ[k] = "1"
    for k in [k for k in os.environ if k.startswith("ENGINE_")]:
        os.environ.pop(k)
    for k, v in flags.items():
        os.environ[k] = str(v)
    os.environ["CBB_TRUTH"] = "verified_v1"
    if os.environ.get("LG_LOG_SEC"):
        R3.LOG_SEC = int(os.environ["LG_LOG_SEC"])
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    ad = Adapters.load(inp, fold, season)
    ad.clock = ClockTap4(ad.clock)
    ad.event = R2.EventTap(ad.event)
    ad.fg = FgTap(ad.fg)
    real_new = L.S.new_state

    def new_state(*a, **k):
        st = real_new(*a, **k)
        R2.TAP.new_chunk(st)
        PLOG.new_chunk(len(st.active))
        SLOG.new_chunk()
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
    return g, res.diag, res.n_possessions, T.clk.copy(), T.ev.copy(), T.n_calls_mismatch, pl, SLOG.frame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL, round-4 flags only")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--games-per-block", type=int, default=25)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--log-sec", type=int, default=None, help="possession-log window (default 180 s)")
    ap.add_argument("--sample-file", default=str(R3.SAMPLE_FILE))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", default=str(R3.DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(R3.INPUT_DIR))
    args = ap.parse_args()
    if args.log_sec:
        R3.LOG_SEC = int(args.log_sec)
        os.environ["LG_LOG_SEC"] = str(args.log_sec)
    if int(args.season) >= 2026:
        raise SystemExit(f"season {args.season} is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context="late-game round 4")
    stray = sorted(k for k in os.environ if k.startswith("ENGINE_"))
    if stray:
        raise SystemExit(f"refusing: ENGINE_* set in the calling env {stray}")
    flags = {}
    for kv in args.env:
        k, v = kv.split("=", 1)
        if k not in ALLOWED_ENV or (k == "ENGINE_FOUL_JOINT" and v != "reference"):
            raise SystemExit(f"--env {kv}: not a round-4 switch")
        flags[k] = v
    from run_engine import engine_provenance
    prov = engine_provenance()
    t0 = time.time()
    for k in R3._PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = v
    os.environ["CBB_TRUTH"] = "verified_v1"
    from cbb_sim.engine.inputs import EngineInputs
    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    if args.sample_file == "all":
        ids = np.sort(inp.games["game_id"].to_numpy().astype("int64"))
    else:
        ids = np.sort(pd.read_parquet(args.sample_file)["game_id"].to_numpy().astype("int64"))
    pos = {int(g): k for k, g in enumerate(inp.games["game_id"].to_numpy())}
    rows = np.array([pos[int(g)] for g in ids], dtype=np.int64)
    if args.max_games:
        rows = rows[:args.max_games]
    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    jobs = [(rows[g0:g0 + args.games_per_block], seeds[s0:s0 + args.seeds_per_block])
            for s0 in range(0, len(seeds), args.seeds_per_block)
            for g0 in range(0, len(rows), args.games_per_block)]
    print(f"lg4 {flags or 'served'}: {len(rows)} games x {len(seeds)} seeds, {len(jobs)} blocks, "
          f"{args.workers} workers, commit {prov.get('engine_commit')}", flush=True)
    gframes, pframes, sframes, diag, n_poss = [], [], [], {}, 0
    clk = np.zeros((3, 4, 3))
    ev = np.zeros((3, 4, 4))
    mism = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, int(args.season), args.input_dir, flags)) as ex:
        futs = [ex.submit(_run_block, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            g, d, np_, c, e, mm, pl, sl = fut.result()
            gframes.append(g)
            pframes.append(pl)
            sframes.append(sl)
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
        ["game_id", "seed", "per", "sec"], ascending=[True, True, True, False]).reset_index(drop=True)
    plog.to_parquet(out / "tap_poss.parquet", index=False)
    pd.concat(sframes, ignore_index=True).to_parquet(out / "tap_shots.parquet", index=False)
    np.savez(out / "tap_window.npz", clock=clk, event=ev)
    from cbb_sim.engine.adapters import Adapters
    ad = Adapters.load(inp, args.fold, int(args.season))
    el = time.time() - t0
    meta = {"engine_tag": f"engine_v0/{args.tag}",
            "round": "late_game round 4 (docs/models/late_game/experiments.md section 9)",
            "created_at": datetime.now(UTC).isoformat(), "arm": flags.get("ENGINE_LATE_GAME", "off"),
            "flags": flags, "sample_file": str(args.sample_file), "seeds": [int(s) for s in seeds],
            "n_seeds": int(len(seeds)), "seed_offset": int(args.seed_offset), "fold": args.fold,
            "season": int(args.season), "backtest": True, "sealed_touched": False,
            "partial": bool(args.max_games), "n_games": int(len(rows)), "n_rows": int(len(games)),
            "input_dir": str(args.input_dir), **prov, "possessions_simulated": int(n_poss),
            "runtime_s": round(el, 1), "workers": int(args.workers), "tap_row_mismatch_calls": int(mism),
            "adapter_flags": ad.flags, "diagnostics": diag}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({len(games):,} rows, {len(plog):,} logged poss, {el:.0f}s, mism {mism})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
