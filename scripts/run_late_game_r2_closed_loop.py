"""run_late_game_r2_closed_loop.py -- late-game ROUND 2: one arm through the engine, tapped.

Pre-registration: `docs/models/late_game/experiments.md` section 4 (commit 1c7aa68).

    .venv/Scripts/python.exe scripts/run_late_game_r2_closed_loop.py \
        --late-game clk_C2 --seeds 25 --tag lg2_W_C2_s25

`--late-game off` (default) is the served reference. The pinned stack, the
served-stack assertion, the 500-game stride subset and the seed layout are
`run_po4b_closed_loop.py`'s, imported rather than restated, so every arm pairs
with `po4b_R_s25` game by game and seed by seed.

THE TAP (read-only; it returns every adapter value unchanged). The engine writes
no possession file, so the round-2 lines that are not whole-game box numbers are
accumulated live, in each worker:

  * `loop.S.new_state` is wrapped to keep a handle on the chunk's GameState;
    the clock proxy then maps its rows to simulations through
    `np.flatnonzero(st.active)`, which is exactly the `act` the loop passed
    (asserted per call).
  * per simulation: first-half points (both sides) and first-half possession
    count (the first-half veto); home margin, both sides' FTA at the first H2
    possession starting at <= 120 s; FTA at the start of OT (end of
    regulation); window possessions.
  * window clock draws by (role, clock bucket): n, intended and consumed sums.
  * window FIRST chances by (live role, clock bucket): n, sum P(bonus FT),
    sum P(FGA_3), sum P(any FGA) -- read off the served probabilities, and the
    role is the OFFENCE's own `score_diff` sign on every row.
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

import run_po4b_closed_loop as PO4B                                  # noqa: E402

DEFAULT_RESULTS = Path("results/engine_v0")
INPUT_DIR = Path("data/processed/models/engine")
_PIN = PO4B._PIN
BUCKETS = np.array([0.0, 10.0, 30.0, 60.0, 120.0])     # (0,10] (10,30] (30,60] (60,120]
ENRICHED_SRC = Path("results/engine_v0/F2_2025_s200_v5b_A/games.parquet")

_W: dict = {}


def _bucket(sec: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(BUCKETS, sec, side="left") - 1, 0, 3)


class _Tap:
    def __init__(self):
        self.st = None
        self.clk = np.zeros((3, 4, 3))          # role x bucket x (n, intended, consumed)
        self.ev = np.zeros((3, 4, 4))           # role x bucket x (n, p_bonus, p_3, p_fga)
        self.n_calls_mismatch = 0

    def new_chunk(self, st):
        n = len(st.active)
        self.st = st
        self.h1_pts = np.full((n, 2), -1, dtype=np.int64)
        self.h1_poss = np.zeros(n, dtype=np.int64)
        self.m120 = np.full(n, np.nan)
        self.fta120 = np.full(n, np.nan)
        self.fta_reg = np.full(n, np.nan)
        self.win_poss = np.zeros(n, dtype=np.int64)


TAP = _Tap()


class ClockTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def draw(self, team, x, u, *a):
        from cbb_sim.engine.adapters import STATE_INDEX as I
        from cbb_sim.models import late_game as LGM
        dur = self._r.draw(team, x, u, *a)
        st = TAP.st
        act = np.flatnonzero(st.active)
        if len(act) != len(x):
            TAP.n_calls_mismatch += 1
            return dur
        per = x[:, I["period"]]
        sec = x[:, I["seconds_remaining"]]
        sd = x[:, I["score_diff"]]
        p1 = per == 1
        np.add.at(TAP.h1_poss, act[p1], 1)
        h2 = (per == 2) & (TAP.h1_pts[act, 0] < 0)
        if h2.any():
            TAP.h1_pts[act[h2]] = st.pts[act[h2]]
        m = (per == 2) & (sec <= 120) & np.isnan(TAP.m120[act])
        if m.any():
            r = act[m]
            TAP.m120[r] = st.pts[r, 0] - st.pts[r, 1]
            TAP.fta120[r] = st.box["fta"][r, 0] + st.box["fta"][r, 1]
        o = (per >= 3) & np.isnan(TAP.fta_reg[act])
        if o.any():
            r = act[o]
            TAP.fta_reg[r] = st.box["fta"][r, 0] + st.box["fta"][r, 1]
        w = LGM.in_window(per, sec, sd)
        if w.any():
            np.add.at(TAP.win_poss, act[w], 1)
            role = (np.sign(sd[w]).astype(np.int64) + 1)
            b = _bucket(sec[w])
            d = np.asarray(dur, float)[w]
            np.add.at(TAP.clk, (role, b, 0), 1.0)
            np.add.at(TAP.clk, (role, b, 1), d)
            np.add.at(TAP.clk, (role, b, 2), np.minimum(d, sec[w]))
        return dur


class EventTap:
    def __init__(self, real):
        self._r = real

    def __getattr__(self, k):
        return getattr(self._r, k)

    def predict(self, team, state, is_first, gidx=None, off=None):
        from cbb_sim.engine.adapters import STATE_INDEX as I
        from cbb_sim.models import late_game as LGM
        from cbb_sim.models import possession_outcome as PO
        p = self._r.predict(team, state, is_first, gidx, off)
        per = state[:, I["period"]]
        sec = state[:, I["seconds_remaining"]]
        sd = state[:, I["score_diff"]]
        w = np.asarray(is_first, bool) & LGM.in_window(per, sec, sd)
        if w.any():
            role = (np.sign(sd[w]).astype(np.int64) + 1)
            b = _bucket(sec[w])
            pw = np.asarray(p, float)[w]
            ci = PO.CLASS_INDEX
            fga = pw[:, ci["FGA_3"]] + pw[:, ci["FGA_rim"]] + pw[:, ci["FGA_jump2"]]
            np.add.at(TAP.ev, (role, b, 0), 1.0)
            np.add.at(TAP.ev, (role, b, 1), pw[:, ci["FT_trip_bonus"]])
            np.add.at(TAP.ev, (role, b, 2), pw[:, ci["FGA_3"]])
            np.add.at(TAP.ev, (role, b, 3), fga)
        return p


def _init_worker(tag, fold, season, input_dir, flags):
    for k in _PIN:
        os.environ[k] = "1"
    for k, v in flags.items():
        os.environ[k] = str(v)
    if "ENGINE_LATE_GAME" not in flags:
        os.environ.pop("ENGINE_LATE_GAME", None)
    os.environ.pop("ENGINE_CLOCK_SEGMENT", None)
    sys.path.insert(0, str(ROOT / "src"))
    from cbb_sim.engine import loop as L
    from cbb_sim.engine.adapters import Adapters
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(input_dir, tag)
    ad = Adapters.load(inp, fold, season)
    ad.clock = ClockTap(ad.clock)
    ad.event = EventTap(ad.event)
    real_new = L.S.new_state

    def new_state(*a, **k):
        st = real_new(*a, **k)
        TAP.new_chunk(st)
        return st
    L.S.new_state = new_state
    _W.update(inp=inp, ad=ad)


def _run_block(job):
    from cbb_sim.engine import loop as L
    game_rows, seeds = job
    inp, ad = _W["inp"], _W["ad"]
    gi = np.repeat(np.asarray(game_rows, dtype=np.int64), len(seeds))
    sd = np.tile(np.asarray(seeds, dtype=np.int64), len(game_rows))
    TAP.clk[:] = 0
    TAP.ev[:] = 0
    res = L.simulate_chunk(inp, ad, gi, sd, keep_players=False)
    gids = inp.games["game_id"].to_numpy()[gi]
    t = pd.DataFrame({"game_id": gids.astype("int64"), "seed": sd.astype("int32"),
                      "h1_home_pts": TAP.h1_pts[:, 0], "h1_away_pts": TAP.h1_pts[:, 1],
                      "h1_poss": TAP.h1_poss, "home_margin_120": TAP.m120,
                      "fta_at_120": TAP.fta120, "fta_reg_end_if_ot": TAP.fta_reg,
                      "window_poss": TAP.win_poss})
    g = res.games.merge(t, on=["game_id", "seed"], how="left", validate="one_to_one")
    return g, res.diag, res.n_possessions, TAP.clk.copy(), TAP.ev.copy(), TAP.n_calls_mismatch


def sample_rows(games: pd.DataFrame, sample: str) -> np.ndarray:
    if sample == "stride":
        return PO4B.subset_rows(games)
    if sample == "enriched":
        s = pd.read_parquet(ENRICHED_SRC, columns=["game_id", "home_pts", "away_pts"])
        mm = (s["home_pts"] - s["away_pts"]).groupby(s["game_id"]).mean().abs()
        pick = set(mm.sort_values(kind="stable").index[:PO4B.SUBSET_N].tolist())
        rows = np.flatnonzero(games["game_id"].isin(pick).to_numpy())
        return rows[np.argsort(games["game_id"].to_numpy()[rows], kind="stable")]
    raise ValueError(sample)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--late-game", default="off")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-block", type=int, default=25)
    ap.add_argument("--seeds-per-block", type=int, default=25)
    ap.add_argument("--sample", default="stride", choices=("stride", "enriched"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    args = ap.parse_args()
    if int(args.season) >= 2026:
        raise SystemExit(f"season {args.season} is SEALED")
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([int(args.season)], context="late-game round 2")
    drift = PO4B._assert_served_stack(False)
    from run_engine import engine_provenance
    prov = engine_provenance()
    t0 = time.time()
    flags = dict(PO4B.PINNED_SUBMODELS)
    flags["ENGINE_EVENT"] = "round2_s1"
    if args.late_game != "off":
        flags["ENGINE_LATE_GAME"] = args.late_game
    for k in _PIN:
        os.environ.setdefault(k, "1")
    for k, v in flags.items():
        os.environ[k] = v
    if args.late_game == "off":
        os.environ.pop("ENGINE_LATE_GAME", None)
    from cbb_sim.engine.inputs import EngineInputs
    in_tag = f"{args.fold}_{args.season}"
    inp = EngineInputs.load(args.input_dir, in_tag)
    rows = sample_rows(inp.games, args.sample)
    if len(rows) != PO4B.SUBSET_N:
        raise SystemExit(f"sample produced {len(rows)} games")
    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds, dtype=np.int64)
    jobs = [(rows[g0:g0 + args.games_per_block], seeds[s0:s0 + args.seeds_per_block])
            for s0 in range(0, len(seeds), args.seeds_per_block)
            for g0 in range(0, len(rows), args.games_per_block)]
    print(f"lg2 {args.late_game} [{args.sample}]: {len(rows)} games x {len(seeds)} seeds, "
          f"{len(jobs)} blocks, {args.workers} workers, commit {prov.get('engine_commit')}", flush=True)
    gframes, diag, n_poss = [], {}, 0
    clk = np.zeros((3, 4, 3))
    ev = np.zeros((3, 4, 4))
    mism = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(in_tag, args.fold, int(args.season),
                                       args.input_dir, flags)) as ex:
        futs = [ex.submit(_run_block, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            g, d, np_, c, e, mm = fut.result()
            gframes.append(g)
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
    np.savez(out / "tap_window.npz", clock=clk, event=ev)
    from cbb_sim.engine.adapters import Adapters
    ad = Adapters.load(inp, args.fold, int(args.season))
    el = time.time() - t0
    meta = {
        "engine_tag": f"engine_v0/{args.tag}",
        "round": "late_game round 2 (docs/models/late_game/experiments.md section 4)",
        "created_at": datetime.now(UTC).isoformat(), "arm": args.late_game,
        "sample": args.sample, "seeds": [int(s) for s in seeds], "n_seeds": int(len(seeds)),
        "seed_offset": int(args.seed_offset), "fold": args.fold, "season": int(args.season),
        "backtest": True, "sealed_touched": False, "partial": False,
        "n_games": int(len(rows)), "n_rows": int(len(games)), "keep_players": False,
        "game_ids": [int(x) for x in inp.games["game_id"].to_numpy()[rows]],
        "served_stack_drift": drift, **prov, "possessions_simulated": int(n_poss),
        "runtime_s": round(el, 1), "workers": int(args.workers),
        "tap_row_mismatch_calls": int(mism),
        "adapter_flags": ad.flags, "diagnostics": diag,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({len(games):,} rows, {el:.0f}s, tap mismatches {mism})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
