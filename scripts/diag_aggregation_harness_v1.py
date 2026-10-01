"""
diag_aggregation_harness_v1.py -- Part C of docs/models/aggregation/experiments.md section 1:
the deterministic expected-points harness (no Monte Carlo) for the cascade aggregation
over-spread decomposition.

For one STACK (a tagged v3 input dir with its overlay of artifacts) and every swap ARM of
`exp_aggregation_swap_v1.py` (same column lists, same neutral values), predict per
(game, offence side), through the engine's own adapter classes (read-only):

  possession_outcome first chance at two reference states -> P(TOV), P(trip), P(FGA_k)
  fg_make per class, weighted over slots by rot_share x usage   -> p_k
  rebound P(OREB)/(P(OREB)+P(DREB)) over a fixed miss mix        -> p_oreb
  free_throw over slots                                          -> f

PPP = [sum_k P(FGA_k) v_k p_k + 2 f P(trip)] / [1 - p_oreb * sum_k P(FGA_k) (1 - p_k)]
X_h(game) = P_ref * (PPP_home_offence - PPP_away_offence)

Writes results/aggregation_v1/harness_<stack>.parquet (one row per game x side x arm).

    .venv/Scripts/python.exe scripts/diag_aggregation_harness_v1.py --stack S0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp_aggregation_swap_v1 as SW  # noqa: E402

OUT = ROOT / "results" / "aggregation_v1"
SHOT = ("rim", "jump2", "three")
FGCLS = {"rim": "FGA_rim", "jump2": "FGA_jump2", "three": "FGA_3"}
V = {"rim": 2.0, "jump2": 2.0, "three": 3.0}
USAGE_IDX = {"rim": 0, "jump2": 1, "three": 2, "TOV": 3, "FT": 4}
HARNESS_ARMS = ["FULL", "PO", "FG", "RB", "RAT", "RAT_PO", "PLY", "TEAM", "OFF", "DEF", "ALL"]
EXTRA_STACKS = {"R": "data/processed/models/engine_v3_R_laneA", "R2": "data/processed/models/engine_v3_R2_laneA",
                "T_PO": "data/processed/models/engine_v3_TPO_laneA"}


def swapped_arrays(inp, team_block, team_cols, arm):
    tcols, scols = SW.ARMS[arm]
    ts = inp.team_static.astype(np.float64).copy()
    ss = inp.slot_static
    tb = team_block.astype(np.float64).copy()
    pos = {c: i for i, c in enumerate(team_cols)}
    for c in tcols:
        if c in pos:
            tb[:, :, pos[c]] = 0.0
        if arm not in SW.EVENT_ONLY:
            ts[:, :, inp.team_names[c]] = SW.neutral_value(c, inp.rules)
    if scols:
        ss = ss.copy()
        for c in scols:
            ss[:, :, :, inp.slot_names[c]] = 0.0
    return tb, ts, ss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--arms", default=",".join(HARNESS_ARMS))
    ap.add_argument("--fg-first-refit", action="store_true",
                    help="addendum B: score every game's fg_make by its FIRST refit (trained before the season)")
    args = ap.parse_args()
    t0 = time.time()
    from run_engine_live import prepare_from_overlay
    from cbb_sim.engine import adapters as AD
    from cbb_sim.engine.inputs import EngineInputs
    sdir = ROOT / (SW.STACK_DIRS.get(args.stack) or EXTRA_STACKS.get(args.stack)
                   or f"data/processed/models/engine_v3_{args.stack}_laneA")
    over = prepare_from_overlay(sdir / "overlay", "F2", 2025, OUT / "_adapter_dirs" / f"harness_{args.stack}")
    for k, v in over.items():
        setattr(AD, k, Path(v))
    inp = EngineInputs.load(sdir, "F2_2025")
    G = inp.n_games
    ev = AD.EventAdapter.load(inp, "round2_s1", "F2", 2025)
    fg = AD.FgMakeAdapter.load(inp, "F2", "decision8", "round4_B1")
    ft = AD.FreeThrowAdapter.load(inp, "F2", "s1_conf_aligned")
    rb = AD.ReboundAdapter.load(inp, "F2", "s1_weekly")
    idx = json.loads((AD.ENGINE_DIR / "event_round2_s1_F2_2025" / "index.json").read_text(encoding="utf-8"))
    team_cols = idx["team_cols"]
    gidx = np.repeat(np.arange(G), 2)
    sidx = np.tile([0, 1], G)
    S = inp.n_slots
    SI = AD.STATE_INDEX
    C = list(AD.PO.CLASSES)

    def state(n, **kw):
        st = np.zeros((n, len(AD.STATE_COLS)))
        for k, v in kw.items():
            st[:, SI[k]] = v
        return st

    seg_po = ev.manifests["first"].segments(gidx)
    seg_fg = {k: fg.manifests[FGCLS[k]].segments(gidx) for k in SHOT}
    if args.fg_first_refit:
        seg_fg = {k: np.zeros_like(v) for k, v in seg_fg.items()}
    seg_ft = ft.manifest.segments(gidx)
    seg_rb = rb.manifest.segments(gidx)

    def by_seg(models, seg, fn):
        out = None
        for k in np.unique(seg):
            r = np.flatnonzero(seg == k)
            v = fn(models[k], r)
            if out is None:
                out = np.full((len(seg),) + v.shape[1:], np.nan)
            out[r] = v
        return out

    share, usage, valid = inp.rot_share, inp.usage_rate, inp.roster_valid
    # free_throw: no team features and no swapped slot column -> identical in every arm
    num = np.zeros(len(gidx)); den = np.zeros(len(gidx))
    st_ft = state(len(gidx), period=2, seconds_remaining=600, score_diff=0, in_bonus=1)
    ts0 = inp.team_static[gidx, sidx]
    for s_ in range(S):
        w = share[gidx, sidx, s_] * usage[gidx, sidx, s_, USAGE_IDX["FT"]] * valid[gidx, sidx, s_]
        use = w > 1e-4
        if not use.any():
            continue
        sl = inp.slot_static[gidx, sidx, s_]
        p = by_seg(ft.models_by_seg, seg_ft[use],
                   lambda m, r: m.predict_proba(np.ascontiguousarray(
                       AD._assemble(ft.plan, np.nan_to_num(ts0[use][r]), sl[use][r], st_ft[use][r]),
                       dtype=np.float32))[:, AD.FT.CLASS_INDEX["MAKE"]])
        num[use] += w[use] * p; den[use] += w[use]
    p_ft = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    p_ft = np.where(np.isfinite(p_ft), p_ft, np.nanmean(p_ft))

    frames = []
    for arm in args.arms.split(","):
        ta = time.time()
        tb_all, ts_all, ss_all = swapped_arrays(inp, ev.team_block, team_cols, arm)
        tb = tb_all[gidx, sidx]
        ts = ts_all[gidx, sidx]
        rec = {"game_idx": gidx, "side": sidx, "arm": np.full(len(gidx), arm)}
        acc = 0
        for kw in ({"period": 1, "seconds_remaining": 900, "score_diff": 0, "in_bonus": 0, "is_transition": 0},
                   {"period": 2, "seconds_remaining": 600, "score_diff": 0, "in_bonus": 1, "is_transition": 0}):
            st = state(len(gidx), **kw)
            p = by_seg(ev.models_first, seg_po,
                       lambda m, r: m.predict_proba(np.ascontiguousarray(
                           AD._assemble(ev.plan_first, tb[r], None, st[r]), dtype=np.float32)))
            acc = acc + p / 2
        rec["p_tov"] = acc[:, C.index("TOV")]
        rec["p_trip"] = acc[:, C.index("FT_trip_shooting")] + acc[:, C.index("FT_trip_bonus")]
        for k, cname in (("rim", "FGA_rim"), ("jump2", "FGA_jump2"), ("three", "FGA_3")):
            rec[f"p_fga_{k}"] = acc[:, C.index(cname)]
        orb = np.zeros(len(gidx))
        for mt, w in (("miss_rim", 0.35), ("miss_jump2", 0.35), ("miss_three", 0.30)):
            st = state(len(gidx), period=1, seconds_remaining=600, score_diff=0, in_bonus=0, **{mt: 1})
            pr = by_seg(rb.models_by_seg, seg_rb,
                        lambda m, r: m.predict_proba(np.ascontiguousarray(
                            AD._assemble(rb.plan, ts[r], None, st[r]), dtype=np.float32)))
            orb = orb + w * pr[:, 0] / (pr[:, 0] + pr[:, 1])
        rec["p_oreb"] = orb
        st = state(len(gidx), period=1, seconds_remaining=600, in_bonus=0, chance_number=1,
                   chance_elapsed_s=12, is_transition_f=0)
        for k in SHOT:
            num = np.zeros(len(gidx)); den = np.zeros(len(gidx))
            for s_ in range(S):
                w = share[gidx, sidx, s_] * usage[gidx, sidx, s_, USAGE_IDX[k]] * valid[gidx, sidx, s_]
                use = w > 1e-4
                if not use.any():
                    continue
                sl = ss_all[gidx, sidx, s_]
                p = by_seg(fg.models_by_seg[FGCLS[k]], seg_fg[k][use],
                           lambda m, r: m.predict_proba(np.ascontiguousarray(
                               AD._assemble(fg.plans[FGCLS[k]], ts[use][r], sl[use][r], st[use][r]),
                               dtype=np.float32))[:, AD.FG.CLASS_INDEX["MAKE"]])
                num[use] += w[use] * p; den[use] += w[use]
            v = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            rec[f"p_make_{k}"] = np.where(np.isfinite(v), v, np.nanmean(v))
        rec["p_ft"] = p_ft
        df = pd.DataFrame(rec)
        pts = sum(df[f"p_fga_{k}"] * V[k] * df[f"p_make_{k}"] for k in SHOT) + 2.0 * df["p_ft"] * df["p_trip"]
        miss = sum(df[f"p_fga_{k}"] * (1 - df[f"p_make_{k}"]) for k in SHOT)
        df["ppp"] = pts / (1 - df["p_oreb"] * miss)
        frames.append(df)
        print(f"[{time.strftime('%H:%M:%S')}] {args.stack} {arm}: {time.time() - ta:.0f}s "
              f"mean ppp {df['ppp'].mean():.4f}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["game_id"] = inp.games["game_id"].to_numpy()[out["game_idx"].to_numpy()]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"harness_{args.stack}{'_fgfirst' if args.fg_first_refit else ''}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path} {out.shape} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
