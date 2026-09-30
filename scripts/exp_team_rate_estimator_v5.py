"""exp_team_rate_estimator_v5.py -- key-coverage fix for the Stage B tables (PM item, 2026-09-30).

Root cause: 6 D-I, non-truncated games with pbp have NO hoopR team-box rows at all. The design
frames contain them (from pbp), but the estimator's table, which is built from the box, did not.
They account for 773 possession_outcome, 572 fg_make and 388 rebound design rows, on the offence and
defence keys alike. `results/team_rate_estimator/missing_keys_v2.csv` lists them by cause.

Fix: those games enter the panel as SCHEDULE-ONLY rows, with zero counts. That gives no observation
and no Kalman update, but the game still takes an index step, so process variance q accrues once,
as for any game played. Each such team-game then gets exactly what the estimator says entering that
game: the filtered state from games strictly before it. A team with no history at all gets the
estimator's own prior (c0 = rho c_prev, P0), with no extra rule. The fitted parameters are unchanged:
the six games carry no outcome data, so no fit sees them.

Outputs are new versions; the v1/v2 files stay:
  data/processed/team_rate_features_E3_v3.parquet, ..._E3opp_v3.parquet   (17 rate-sides x {c, v, L})
  data/processed/team_rate_variance_O1a_v2.parquet                        (S3's variance source)
Coverage is asserted against the possession_outcome, fg_make and rebound design frames on both folds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import exp_team_rate_estimator_v4 as X4  # noqa: E402  (sets X3.RATES = RATES_V2)
X3 = X4.X3; X2 = X3.X2
from cbb_sim.live.team_rate_estimator import RATES_V2  # noqa: E402

OUT = Path("results/team_rate_estimator")
DESIGNS = {
    "possession_outcome": ("data/processed/models/possession_outcome/round2/design.parquet", "offense_team_id", "defense_team_id"),
    "fg_make": ("data/processed/models/fg_make/design_v2_shotshooter.parquet", "off_team_id", "def_team_id"),
    "rebound": ("data/processed/models/rebound/round3/design_round3.parquet", "off_team_id", "def_team_id"),
}
_orig_team_games = X2.team_games


def design_games():
    """(season, game_id, team_id, opp_id) of every design row's two teams, all three sub-models."""
    rows = []
    for p, o, d in DESIGNS.values():
        f = pd.read_parquet(p, columns=["season", "game_id", o, d]).drop_duplicates()
        rows.append(f.rename(columns={o: "team_id", d: "opp_id"}))
        rows.append(f.rename(columns={d: "team_id", o: "opp_id"}))
    return pd.concat(rows).drop_duplicates(["season", "game_id", "team_id"])


def team_games_with_schedule():
    tg = _orig_team_games()
    need = design_games()
    have = set(zip(tg["game_id"], tg["team_id"]))
    miss = need[[(g, t) not in have for g, t in zip(need["game_id"], need["team_id"])]].copy()
    u = pd.read_parquet("data/processed/games_universe.parquet", columns=["game_id", "game_date", "season"])
    u = u[u["season"] <= 2025]
    miss = miss.merge(u[["game_id", "game_date"]], on="game_id", how="left")
    assert miss["game_date"].notna().all(), "a design game has no schedule date"
    miss["game_date"] = pd.to_datetime(miss["game_date"])
    cnt = [c for c in tg.columns if c not in ("season", "game_id", "team_id", "opp_id", "game_date")]
    for c in cnt:
        miss[c] = 0.0
    miss["schedule_only"] = True
    tg = tg.assign(schedule_only=False)
    out = pd.concat([tg, miss[tg.columns]], ignore_index=True)
    json.dump({"n_schedule_only_team_games": int(len(miss)), "games": sorted(int(g) for g in miss["game_id"].unique())},
              open(OUT / "schedule_only_v5.json", "w"), indent=1)
    print("schedule-only team-games added:", len(miss), "games:", miss["game_id"].nunique(), flush=True)
    return out.drop(columns="schedule_only")


X2.team_games = team_games_with_schedule      # used by X3.build_all


def coverage_check(table: pd.DataFrame):
    res = {}
    for sub, (p, o, d) in DESIGNS.items():
        f = pd.read_parquet(p, columns=["season", "game_id", o, d])
        for fold, max_season in (("F1", 2024), ("F2", 2025)):
            ff = f[f["season"] <= max_season]
            t = table[table["fold"] == fold]
            keys = set(zip(t["season"], t["game_id"], t["team_id"]))
            n_miss = sum((s, g, x) not in keys for col in (o, d) for s, g, x in zip(ff["season"], ff["game_id"], ff[col]))
            res[f"{sub}|{fold}"] = {"design_rows": len(ff), "missing_keys": int(n_miss)}
    print(json.dumps(res, indent=1))
    assert all(v["missing_keys"] == 0 for v in res.values()), res
    json.dump(res, open(OUT / "coverage_v5.json", "w"), indent=1)


def emit_features(Ps, with_opp: bool):
    label = "E3opp" if with_opp else "E3"
    path = Path(f"data/processed/team_rate_features_{label}_v3.parquet")
    assert not path.exists()
    u = pd.read_parquet("data/processed/games_universe.parquet", columns=["game_id", "tipoff_utc", "season"])
    tip_all = u[u["season"] <= 2025][["game_id", "tipoff_utc"]].assign(tipoff_utc=lambda x: pd.to_datetime(x["tipoff_utc"], utc=True))
    pm = X4.merged_params()
    wide = []
    for fold, fd in X3.FOLDS.items():
        parts = []
        for rate in RATES_V2:
            adj = X3.opp_adj(Ps, pm, fold, rate) if with_opp else {"off": None, "def": None}
            for side in ("off", "def"):
                P = Ps[(rate, side)]
                tip = P.df[["season", "team_id", "j", "game_id"]].merge(tip_all, on="game_id", how="left")
                assert tip["tipoff_utc"].notna().all()
                td = tip.sort_values(["season", "team_id", "j"]).groupby(["season", "team_id"])["tipoff_utc"].diff().dropna()
                assert (td > pd.Timedelta(0)).all(), "as-of unsafe"
                sel = (P.season <= fd["test"])[P.ts]
                c, v = X3.run_filter(P, pm[f"{fold}|{rate}|{side}"], adj[side])
                d = P.df.loc[sel, ["season", "game_id", "team_id", "game_date", "L"]].copy()
                d[f"{rate}_{side}_c"] = c[P.ts[sel], P.j[sel]]
                d[f"{rate}_{side}_v"] = v[P.ts[sel], P.j[sel]]
                parts.append(d.rename(columns={"L": f"{rate}_{side}_L"}).set_index(["season", "game_id", "team_id", "game_date"]))
        w = pd.concat(parts, axis=1).reset_index(); w.insert(0, "fold", fold); wide.append(w)
    out = pd.concat(wide, ignore_index=True)
    assert not out.duplicated(["fold", "game_id", "team_id"]).any() and not out.isna().any().any()
    coverage_check(out)
    # the v2 values must be unchanged except for the teams that played a schedule-only game
    v2 = pd.read_parquet(f"data/processed/team_rate_features_{label}_v2.parquet")
    m = v2.merge(out, on=["fold", "season", "game_id", "team_id"], suffixes=("", "__v3"))
    cols = [c for c in v2.columns if c.endswith(("_c", "_v", "_L"))]
    diff = np.max(np.abs(np.column_stack([m[c] - m[c + "__v3"] for c in cols])), axis=1)
    so = json.load(open(OUT / "schedule_only_v5.json"))
    print(f"rows changed vs v2: {int((diff > 0).sum())} of {len(m)}; new rows: {len(out) - len(v2)}")
    out.to_parquet(path, index=False)
    print("wrote", path, out.shape)


def emit_o1a(Ps):
    path = Path("data/processed/team_rate_variance_O1a_v2.parquet")
    assert not path.exists()
    prm = {k: X3.prm_from_json(v) for k, v in json.load(open(OUT / "params_o1a_v4.json")).items()}
    wide = []
    for fold, fd in X3.FOLDS.items():
        parts = []
        for rate in RATES_V2:
            for side in ("off", "def"):
                P = Ps[(rate, side)]
                sel = (P.season <= fd["test"])[P.ts]
                _, v = X3.run_filter(P, prm[f"{fold}|{rate}|{side}"])
                d = P.df.loc[sel, ["season", "game_id", "team_id", "game_date"]].copy()
                d[f"{rate}_{side}_v_o1a"] = v[P.ts[sel], P.j[sel]]
                d[f"{rate}_{side}_phi_o1a"] = prm[f"{fold}|{rate}|{side}"].phi
                parts.append(d.set_index(["season", "game_id", "team_id", "game_date"]))
        w = pd.concat(parts, axis=1).reset_index(); w.insert(0, "fold", fold); wide.append(w)
    out = pd.concat(wide, ignore_index=True)
    assert not out.isna().any().any()
    coverage_check(out)
    out.to_parquet(path, index=False)
    print("wrote", path, out.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["features", "opp", "o1a"])
    a = ap.parse_args()
    Ps, _, _ = X3.build_all()
    if a.part == "features":
        emit_features(Ps, False)
    elif a.part == "opp":
        emit_features(Ps, True)
    else:
        emit_o1a(Ps)


if __name__ == "__main__":
    main()
