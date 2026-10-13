"""grade_clock_env_full_v1.py -- clock round 10 (docs/models/clock/experiments.md section 43) FULL-SIZE paired read of the
candidate E1 against served v3: fold 2 / 2025, all 5,710 games x 200 paired seeds, four control floor draws at seed offsets
1000-4000 (200 seeds each). Written and COMMITTED BEFORE the full-size E1 run completed. Adopts nothing; the PM decides.

Same lines, same decision code and same tables as the 500 x 25 read (`grade_clock_env_loop_v1`, imported: `Lines`, `judge`,
`seed_check`, `team_table`, `real_start_types`, `disp_values`), with two differences forced by size and stated here:
  * trajectory-based lines (every `grade_endgame_v1.LoopLines` line incl. the two primaries, the first-half env slope, the
    dispersion gates via `diag_margin_dispersion_v2 --seeds 50` on `diag_c4_extract_chunked_v1` extracts, start types, team
    table, time of decision) are read on the FIRST 50 SEEDS of each run, the 5,710 x 50 definition the D2 real/control numbers
    (+0.152 / +0.011, 0.600 / 0.408) and the dispersion baselines (1.047 ...) were measured on; trajectories are read in
    game chunks;
  * game-level lines from games.parquet use all 200 seeds: the round-8 lines (tempo elasticity ratio, team pace slope,
    OREB / FGM / TOV / FTA within-game possession slopes, count calibration), plus 200-seed cross-checks of the primaries
    (`D2_poss_on_env_200` on the engine's game possession count incl. OT, `D2_total_on_env_200`) and G5 / G9 / OT
    (reported).
Floors: max(2 x SD of the four control draws, 2 x paired game-bootstrap SE), 200 bootstrap draws over the 5,710 games,
numpy seed 12345. Cells under 150 games are UNDERPOWERED.

AMENDMENT (2026-10-13 ~14:35 EDT, committed BEFORE the full-size E1 run started and before any full-size arm output existed):
the four control floor draws are 5,710 x 50 seeds each (offsets 1000-4000, trajectories for all 50), not 5,710 x 200.
Measured local throughput is ~120 game-seeds/s (2.6 h per 200-seed run), so six 200-seed runs would take ~15 h against the
brief's ~1 h per run. Section 43 fixes "four control draws" but not their size. Every trajectory-based line (both primaries,
the first-half env slope, the dispersion gates, the multi-level tables) is read on 50 seeds in the arm and the control, so
50-seed draws are exactly matched to those lines. For the lines read on all 200 seeds (keys `r8_*` and `*_200`) the draws'
SD is scaled by sqrt(50 / 200) (Monte Carlo SE scaling) before the floor is formed; the unscaled draw SD is written too.

    set CBB_TRUTH=verified_v1
    .venv/Scripts/python.exe scripts/grade_clock_env_full_v1.py
writes results/clock_env10/grade_full_s200.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

os.environ.setdefault("CBB_TRUTH", "verified_v1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import diag_c4_lib_v1 as L  # noqa: E402
import grade_clock_env_loop_v1 as GL  # noqa: E402
import grade_clock_r8_loop_v1 as R8  # noqa: E402
import grade_endgame_v1 as EG  # noqa: E402

FULL = ROOT / "results/clock_env10/full"
GL.DISP_DIR = ROOT / "results/clock_env10/full_dispersion"
R8.EV = FULL
S_TRAJ, S_ALL = 50, 200
CONTROL = "clk10full_ctrl_o0_s200"
ARM = "clk10full_E1_o0_s200"
DRAWS = [f"clk10full_ctrl_o{o}_s50" for o in (1000, 2000, 3000, 4000)]
S_DRAW = 50
OFFSET = {CONTROL: 0, ARM: 0, **{d: int(d.split("_o")[1].split("_")[0]) for d in DRAWS}}
CHUNKS = 6


def load_tag(tag: str, ids: np.ndarray) -> tuple:
    o = OFFSET[tag]
    gm = pd.read_parquet(FULL / tag / "games.parquet")
    if gm["seed"].nunique() != (S_DRAW if tag in DRAWS else S_ALL) or gm["game_id"].nunique() != len(ids):
        raise SystemExit(f"{tag}: {gm['seed'].nunique()} seeds / {gm['game_id'].nunique()} games")
    gmt = gm[gm["seed"] < o + S_TRAJ]
    dset = ds.dataset(ROOT / "results/trajectories" / tag / "trajectory.parquet", format="parquet")
    pgs, sxs, stgs = [], [], []
    for part in np.array_split(ids, CHUNKS):
        t = dset.to_table(filter=ds.field("game_id").isin(part.tolist()) & (ds.field("seed") < o + S_TRAJ)).to_pandas()
        d = L.std_from_sim(t)
        pgc = EG.per_game_frame(d, S_TRAJ, part)
        pgc = EG.add_seed_level(pgc, d, part)
        pgs.append(pgc)
        sxs.append(GL.sim_extras(t, part))
        stgs.append(L.team_game(d))
        del t, d
    pg = pd.concat(pgs).reindex(ids)
    sx = pd.concat(sxs).reindex(ids)
    stg = pd.concat(stgs, ignore_index=True)
    fm = (gmt["home_pts"] - gmt["away_pts"]).abs()
    gcol = pd.DataFrame({"p1_cnt": ((gmt["n_periods"] <= 2) & (fm == 1)).astype(float).groupby(gmt["game_id"]).sum(),
                         "gm_poss_sum": gmt.groupby("game_id")["possessions"].sum(),
                         "gm_poss_sq": (gmt["possessions"] ** 2).groupby(gmt["game_id"]).sum()}).reindex(ids)
    pg = pg.join(gcol)
    if (gmt.groupby("game_id").size().reindex(ids) != pg["n_seeds"]).any() or int(pg["n_seeds"].min()) != S_TRAJ:
        raise SystemExit(f"{tag}: games.parquet and trajectory disagree on game-seeds")
    g8 = R8.per_game(R8.load_run(tag))
    meta = json.loads((FULL / tag / "run_meta.json").read_text(encoding="utf-8"))
    flags = {k: v for k, v in meta.get("adapter_flags", {}).items() if k.startswith("ENGINE_")}
    return pg, gmt, sx, g8, stg, {"flags": flags, "seeds": meta.get("seeds"), "engine_commit": meta.get("engine_commit"),
                                  "engine_dirty_src": meta.get("engine_dirty_src"), "runtime_s": meta.get("runtime_s")}


def lines200(LN, g8: pd.DataFrame, w: np.ndarray) -> dict:
    LL = LN.LL
    a = g8.reindex(LN.ids)
    poss, tmu = a["poss"].to_numpy(float), a["t_mean"].to_numpy(float)
    okr = LL.ok & np.isfinite(LL.env)
    out = {"D2_poss_on_env_200": LL._wslope(LL.env, poss, w, okr),
           "D2_total_on_env_200": LL._wslope(LL.env, tmu, w, np.isfinite(LL.env) & np.isfinite(LL.ytot)),
           "act_D2_total_on_env_200": LL._wslope(LL.env, LL.ytot, w, np.isfinite(LL.env) & np.isfinite(LL.ytot))}
    wT = pd.Series(w, index=LN.ids).reindex(LN.T.index).fillna(0).to_numpy()
    r8 = R8.lines(g8.loc[LN.T.index], LN.T, LN.tg, wT)
    for k in ("g5_total_sd_ratio", "g5_margin_sd_ratio", "g9_total_bias", "g9_margin_bias", "g9_total_slope_mc",
              "g9_margin_slope_mc", "ot_rate", "poss_sd_pooled"):
        out[f"r8_{k}"] = r8[k]
    return out


def main() -> int:
    tags = [CONTROL, *DRAWS, ARM]
    ids = np.sort(pd.read_parquet(FULL / CONTROL / "games.parquet", columns=["game_id"])["game_id"].unique().astype("int64"))
    real = EG._real_side(ids)
    pgi = EG._pregame(ids)
    LN = GL.Lines(ids, real, pgi)
    frames, info, stgs = {}, {}, {}
    for t in tags:
        pg, gmt, sx, g8, stg, inf = load_tag(t, ids)
        frames[t] = (pg, gmt, sx, g8)
        stgs[t] = stg
        info[t] = inf
        print(f"loaded {t}: flags {inf['flags'].get('ENGINE_CLOCK')}, commit {inf['engine_commit']}", flush=True)
    if info[ARM]["seeds"] != info[CONTROL]["seeds"]:
        raise SystemExit("arm and control seeds differ (pairing broken)")
    G = len(ids)
    one = np.ones(G)
    val = {t: {**LN.lines(*frames[t], one), **lines200(LN, frames[t][3], one), **GL.disp_values(t)} for t in tags}
    act = {k: v for k, v in val[CONTROL].items() if k.startswith("act_")}
    names = [k for k in val[CONTROL] if not k.startswith("act_")]
    draw_sd_raw = {k: float(np.nanstd([val[t][k] for t in DRAWS], ddof=1)) for k in names}
    s200 = lambda k: k.startswith("r8_") or k.endswith("_200")  # noqa: E731
    draw_sd = {k: v * (np.sqrt(S_DRAW / S_ALL) if s200(k) else 1.0) for k, v in draw_sd_raw.items()}
    wb = EG.boot_weights(G)
    diffs = {k: [] for k in names}
    for b in range(EG.N_BOOT):
        la = {**LN.lines(*frames[ARM], wb[b]), **lines200(LN, frames[ARM][3], wb[b])}
        lc = {**LN.lines(*frames[CONTROL], wb[b]), **lines200(LN, frames[CONTROL][3], wb[b])}
        for k in names:
            src = GL.DISP_REP.get(k, k)
            if src in la:
                diffs[k].append(la[src] - lc[src])
        if b % 50 == 0:
            print(f"  bootstrap {b}/{EG.N_BOOT}", flush=True)
    rows = {}
    for k in names:
        se = float(np.nanstd(diffs[k], ddof=1)) if diffs[k] else float("nan")
        rows[k] = {"arm": val[ARM][k], "ctrl": val[CONTROL][k], "delta": val[ARM][k] - val[CONTROL][k],
                   "draw_sd": draw_sd[k], "draw_sd_raw": draw_sd_raw[k], "boot_se": se, "floor": float(np.nanmax([2 * draw_sd[k], 2 * se]))}
    res = {"spec": "docs/models/clock/experiments.md section 43 (full-size read of a candidate)", "control": CONTROL,
           "draws": DRAWS, "arm": ARM, "n_games": int(G), "n_pbp_ok": int(LN.LL.ok.sum()), "n_r8_games": int(len(LN.T)),
           "seeds_traj": S_TRAJ, "seeds_all": S_ALL, "info": info, "actual": act, "rows": rows,
           "draw_lines": {t: val[t] for t in DRAWS}}
    res["seed_check"] = GL.seed_check(val[CONTROL], act, draw_sd)
    res["verdict"] = GL.judge(rows, act)
    x200 = {}
    for k, tk in (("D2_poss_on_env_200", "act_D2_poss_on_env"), ("D2_total_on_env_200", "act_D2_total_on_env_200")):
        x200[k] = GL.toward(rows[k], act[tk])
    res["verdict"]["primary_200_crosscheck"] = x200
    rs = GL.real_start_types(ids)
    st_real = rs.groupby("prev_end")["duration_s"].agg(["mean", "size"])
    st_games = rs.groupby("prev_end")["game_id"].nunique()
    res["start_type"] = {lv: {"real_sec": float(st_real["mean"].get(lv, np.nan)), "real_rows": int(st_real["size"].get(lv, 0)),
                              "games": int(st_games.get(lv, 0)), "ctrl_sec": rows[f"st_{lv}_sec"]["ctrl"],
                              "arm_sec": rows[f"st_{lv}_sec"]["arm"], "floor": rows[f"st_{lv}_sec"]["floor"]}
                         for lv in GL.START_LEVELS}
    res["quintiles"] = {}
    rp = (LN.LL.r["poss_reg_h"] + LN.LL.r["poss_reg_a"]).to_numpy(float)
    for nm, q in (("envq", LN.envq), ("homeq", LN.homeq)):
        tab = []
        for k in range(5):
            m = LN.LL.ok & (q == k)
            mt = (q == k) & np.isfinite(LN.LL.ytot)
            tab.append({"q": k + 1, "n_games_poss": int(m.sum()), "n_games_total": int(mt.sum()),
                        "underpowered": bool(m.sum() < GL.UNDER), "real_poss": float(np.nanmean(rp[m])),
                        "real_total": float(np.nanmean(LN.LL.ytot[mt])),
                        **{f"{c}_{x}": rows[f"{nm}{k + 1}_{c}"][x] for c in ("poss_sim", "poss_gap", "total_sim", "total_gap")
                           for x in ("ctrl", "arm", "floor")}})
        res["quintiles"][nm] = tab
    res["team_table"] = GL.team_table(ids, pgi, real, {"ctrl": stgs[CONTROL], "arm": stgs[ARM]})
    res["written_at"] = datetime.now(UTC).isoformat()
    out = ROOT / "results/clock_env10/grade_full_s200.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    v = res["verdict"]
    print("CANDIDATE (full size)" if v["candidate"] else "NOT A CANDIDATE (full size)", "| primary_moves", v["primary_moves"],
          "| vetoes failing", v["vetoes_failing"])
    for k, x in {**v["primary"], **x200}.items():
        print(f"  {k}: act {x['target']:.4f} ctrl {x['ctrl']:.4f} arm {x['arm']:.4f} floor {x['floor']:.4f} floors {x['floors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
