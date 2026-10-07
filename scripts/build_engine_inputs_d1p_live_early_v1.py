#!/usr/bin/env python
"""
build_engine_inputs_d1p_live_early_v1.py -- LIVE-path day-1 inputs over the first 46 days (days 0-45) of a past season
(2026-10-07, docs/tests/early_gap_live_path_2026-10-07.md). Unlike the bake-off window builder (`cmd_window`, which suppresses the
in-season rotation prior for every window game), this runs `build_live` UNPATCHED with the A3 seed (+ optional R1 fallback), so the
seed fires only on a team-game whose 15 slots are all anonymous (the team's first game), exactly as in 2026-27 serving. Later games
keep their served in-season rotation priors. Assembled on the served v3 base; shot-block LUT shooter/known zeroed for the seeded
sides only (same convention as the bake-off assemble, where window shooter state is held anonymous). Nothing served is written.

    python scripts/build_engine_inputs_d1p_live_early_v1.py --fold F2 [--fallback R1]
Output: data/processed/models/engine_v3_d1p_live_<fold>/ (gitignored model dir; see docs).
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_engine_inputs_anon_window_v1 as AW  # noqa: E402
import build_engine_inputs_day1prior_v1 as DP  # noqa: E402
import build_engine_inputs_live as BL  # noqa: E402

DAYS = 46


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2"); ap.add_argument("--fallback", default="R1")
    a = ap.parse_args()
    fold, c = a.fold, AW.CFG[a.fold]
    tag = f"{fold}_{c['season']}"
    g = pd.read_parquet(c["base"] / f"games_{tag}.parquet"); g["game_date"] = pd.to_datetime(g["game_date"])
    d0 = g["game_date"].min(); s0 = d0.strftime("%Y-%m-%d")
    w = g[g["game_date"] < d0 + pd.Timedelta(days=DAYS)]
    od = ROOT / "data/processed/models" / f"engine_v3_d1p_live_{fold}"; sd = od / "_stage"; sd.mkdir(parents=True, exist_ok=True)
    BL.ENGINE_DIR = c["template"]
    zb = dict(np.load(c["base"] / f"arrays_{tag}.npz"))
    pos = {int(x): i for i, x in enumerate(g["game_id"])}
    arrs = {k: v.copy() for k, v in zb.items()}
    lut = dict(np.load(c["lut"])); assert np.array_equal(lut["game_id"], g["game_id"].to_numpy())
    lut["shooter"] = lut["shooter"].copy(); lut["known"] = lut["known"].copy()
    seeded_tg = 0; seeded_games = set(); same = {}; t0 = time.time(); fb_teams = set()
    for D in sorted(w["game_date"].dt.strftime("%Y-%m-%d").unique()):
        st = f"D1PLIVE_{fold}_{D}"
        ids = w.loc[w["game_date"] == pd.Timestamp(D), "game_id"].tolist()
        if not (sd / f"games_{st}.parquet").exists():
            slate = BL.load_slate_from_universe(D, c["season"], only_ids=ids)
            as_of = pd.to_datetime(slate["tipoff_utc"], utc=True).min() - pd.Timedelta(minutes=30)
            inp, diag = DP.build(slate, as_of, c["season"], fold, "A3", season_start=s0, anon=False, strict_finish=False,
                                 minutes_source="onfloor" if fold == "F2" else "box", fallback=a.fallback or None)
            inp.save(sd, st)
            (sd / f"diag_{st}.json").write_text(json.dumps({k: diag.get(k) for k in ("d1p_team_games", "d1p_slots", "d1p_fallback_teams", "rotation_fallback_team_games")}, default=str))
        gg = pd.read_parquet(sd / f"games_{st}.parquet"); z = np.load(sd / f"arrays_{st}.npz")
        dg = json.loads((sd / f"diag_{st}.json").read_text()); fb_teams |= set(dg.get("d1p_fallback_teams") or [])
        idx = np.array([pos[int(x)] for x in gg["game_id"]])
        for k in zb:
            if k in AW.PLAYER_ARRAYS:
                arrs[k][idx] = z[k]
            else:
                ok = np.array_equal(z[k], zb[k][idx], equal_nan=True) if z[k].dtype.kind == "f" else np.array_equal(z[k], zb[k][idx])
                same[k] = same.get(k, True) and bool(ok)
        diff = (z["roster_cbbd"] != zb["roster_cbbd"][idx])          # (n,2,15)
        side = diff.any(axis=2)
        for r, s in zip(*np.nonzero(side)):
            lut["shooter"][idx[r], s, :] = 0; lut["known"][idx[r], s, :] = 0
            seeded_tg += 1; seeded_games.add(int(gg["game_id"].iloc[r]))
        print(f"[{time.time()-t0:5.0f}s] {D} seeded team-games so far {seeded_tg}", flush=True)
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f"arrays_{tag}.npz", **arrs)
    for f in (f"games_{tag}.parquet", f"event_block_{tag}.npz"):
        shutil.copy2(c["base"] / f, od / f)
    lp = od / f"shot_block_K2_Ocell_d1plive_{tag}.npz"; np.savez_compressed(lp, **lut)
    names = json.loads((c["base"] / f"names_{tag}.json").read_text(encoding="utf-8"))
    rel = lp.relative_to(ROOT).as_posix()
    names.setdefault("meta", {})["shot_block_lut"] = {"K2_Ocell": rel, "K2_Ocell_v3in": rel}
    (od / f"names_{tag}.json").write_text(json.dumps(names, indent=1, default=str), encoding="utf-8")
    w[["game_id", "game_date"]].to_parquet(od / f"window_ids_{fold}.parquet", index=False)
    rep = dict(fold=fold, window_games=len(w), seeded_team_games=seeded_tg, games_with_seeded_side=len(seeded_games),
               fallback_teams_used=sorted(map(int, fb_teams)), non_player_arrays_equal_base=same)
    (od / "assemble_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))

if __name__ == "__main__":
    main()
