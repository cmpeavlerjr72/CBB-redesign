"""train_fg_make_two_stage_v3.py -- lane A day 2026-10-01, round 4 (docs/models/fg_make/experiments.md section 29):
the jumper league-level round. A thin versioned wrapper of train_fg_make_two_stage_v2.py (not edited): it replaces
the design's `lg_make_asof` column (the league level that v2 enters as a logit offset, and that the harness serves
per game) by an alternative AS-OF league level, writes the patched design next to the outputs, and runs v2 on it.

--league asof      v2 unchanged except the opening-day value: the served column back-fills a season's first date
                   from its NEXT date (forward-looking); here opening day takes the previous season's final rate
                   (falls back to the served value only for the first season in the design). [reference]
--league roll28    LJ1: league make rate over the previous 28 days of the same season, strictly before the game
                   date; if that window holds fewer than 1,500 attempts of the type, the season-to-date as-of rate.
--league seasonal  LJ2: logit(season-to-date as-of) + s_k(b), b = 14-day bucket of days since the season's first
                   date, s_k(b) = mean over the fold's TRAINING seasons of logit(realised league rate in that bucket)
                   - logit(as-of rate at the bucket's games), attempt-weighted. Fitted on training seasons only.
All three use only games before tip. Every other argument is v2's.

    .venv/Scripts/python.exe scripts/train_fg_make_two_stage_v3.py --league roll28 --arm TSR --fold F2 --seed 0
"""
from __future__ import annotations

import train_par_common_v1 as C

C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_fg_make_two_stage_v1 as V1  # noqa: E402
import train_fg_make_two_stage_v2 as V2  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

EPS = 1e-6


def lgt(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def daily(design: pd.DataFrame) -> pd.DataFrame:
    d = design[["season", "game_date", "shot_class", "y"]].copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    day = d.groupby(["season", "shot_class", "game_date"], as_index=False).agg(mk=("y", "sum"), att=("y", "size"))
    day = day.sort_values(["season", "shot_class", "game_date"])
    g = day.groupby(["season", "shot_class"])
    day["cmk"] = g["mk"].cumsum() - day["mk"]
    day["catt"] = g["att"].cumsum() - day["att"]
    day["asof"] = np.where(day["catt"] > 0, day["cmk"] / day["catt"].clip(lower=1), np.nan)
    fin = day.groupby(["season", "shot_class"]).agg(fm=("mk", "sum"), fa=("att", "sum")).reset_index()
    fin["prev_final"] = fin["fm"] / fin["fa"]
    fin["season"] = fin["season"] + 1
    day = day.merge(fin[["season", "shot_class", "prev_final"]], on=["season", "shot_class"], how="left")
    first = day.groupby(["season", "shot_class"])["game_date"].transform("min")
    day["dss"] = (day["game_date"] - first).dt.days
    return day


def league_level(design: pd.DataFrame, mode: str, train_seasons: list[int]) -> tuple[np.ndarray, dict]:
    day = daily(design)
    # opening day (and any date with nothing earlier): the previous season's final rate, never a later date
    base = day["asof"].copy()
    nodata = base.isna()
    base[nodata] = day.loc[nodata, "prev_final"]
    info = {"mode": mode, "opening_rows_prev_final": int(nodata.sum())}
    if mode == "asof":
        lvl = base
    elif mode == "roll28":
        out = []
        for (_, _), g in day.groupby(["season", "shot_class"]):
            dts = g["game_date"].to_numpy()
            mk = g["mk"].to_numpy(float); at = g["att"].to_numpy(float)
            cm = np.concatenate([[0], np.cumsum(mk)]); ca = np.concatenate([[0], np.cumsum(at)])
            lo = np.searchsorted(dts, dts - np.timedelta64(28, "D"), side="left")
            hi = np.arange(len(dts))                      # strictly before this date
            wm, wa = cm[hi] - cm[lo], ca[hi] - ca[lo]
            out.append(pd.Series(np.where(wa >= 1500, wm / np.maximum(wa, 1), np.nan), index=g.index))
        r = pd.concat(out).reindex(day.index)
        lvl = r.where(r.notna(), base)
        info["rows_window_fallback"] = int(r.isna().sum())
    elif mode == "seasonal":
        day["b"] = (day["dss"] // 14).clip(upper=11)
        tr = day[day["season"].isin(train_seasons) & base.notna()].copy()
        tr["gap"] = lgt(tr["mk"] / tr["att"]) - lgt(base[tr.index])
        # attempt-weighted mean of the realised-vs-as-of logit gap per (type, bucket); pooled over training seasons
        prof = tr.assign(w=tr["att"]).groupby(["shot_class", "b"]).apply(
            lambda z: np.average(z["gap"], weights=z["w"]), include_groups=False).rename("s").reset_index()
        day = day.merge(prof, on=["shot_class", "b"], how="left")
        day["s"] = day["s"].fillna(0.0)
        lvl = pd.Series(1 / (1 + np.exp(-(lgt(base.to_numpy()) + day["s"].to_numpy()))), index=day.index)
        info["profile"] = {f"{c}|{int(b)}": float(s) for c, b, s in prof.itertuples(index=False)}
    else:
        raise SystemExit(mode)
    day["lvl"] = lvl.to_numpy()
    key = design[["season", "shot_class", "game_date"]].copy()
    key["game_date"] = pd.to_datetime(key["game_date"])
    m = key.merge(day[["season", "shot_class", "game_date", "lvl"]], on=["season", "shot_class", "game_date"],
                  how="left")
    v = m["lvl"].to_numpy(float).copy()
    served = design["lg_make_asof"].to_numpy(float)
    miss = ~np.isfinite(v)
    info["rows_fallback_served"] = int(miss.sum())
    v[miss] = served[miss]
    return v, info


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--league", choices=["asof", "roll28", "seasonal"], required=True)
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/processed/models/fg_make/round_lj")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a, rest = ap.parse_known_args()
    design = pd.read_parquet(V1.DESIGN)
    v, info = league_level(design, a.league, FG.FOLDS[a.fold]["train"])
    design["lg_make_asof"] = v.astype("float32")
    a.out_root.mkdir(parents=True, exist_ok=True)
    dp = a.out_root / f"_design_{a.league}_{a.fold}.parquet"
    if not dp.exists():
        design.to_parquet(dp, index=False)
    V1.DESIGN = dp
    arm_label = f"{a.arm}{ {'asof': 'J0', 'roll28': 'J1', 'seasonal': 'J2'}[a.league] }"
    sys.argv = [sys.argv[0], "--arm", a.arm, "--fold", a.fold, "--seed", str(a.seed),
                "--out-root", str(a.out_root / a.league), *rest]
    rc = V2.main()
    out = a.out_root / a.league / f"{a.arm}_s{a.seed}_{a.fold}"
    (out / "league_level.json").write_text(json.dumps({**info, "label": arm_label}, indent=1, default=float),
                                           encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
