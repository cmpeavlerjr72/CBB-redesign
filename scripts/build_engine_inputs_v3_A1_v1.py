"""build_engine_inputs_v3_A1_v1.py -- input arm A1 of free_throw experiments.md section 13.3 (lane I, 2026-10-01).

A prior for the anonymous (unknown-shooter) slots. In each team-game, the anonymous roster slots (roster_cbbd <= 0), in
slot order, receive the FT shooter block of the team's 2025 CBBD season-roster players who are NOT among the game's named
candidates, ordered by 2024 minutes (hoopR player box; none = 0; ties by cbbd id). The block follows the FT design's
definitions (`FT.build_ft_design`): has_prior_season / prior_season_ft from the completed 2024 season (league-centred on
the as-of league FT% of the game date), shooter_ft_asof / shooter_fta_asof from 2025 attempts strictly before the game
date (0 if none). ONLY the four FT shooter-block slot columns of anonymous slots change; the base `engine_v3` is read,
never written. Output: `data/processed/models/engine_v3_I_A1/` (refuses an existing dir) + `builder_report.json`.

Also writes the offline check (2025 only): for real attempts by off-roster shooters, coverage (shooter among the game's
A1-filled players) and the served model's mean p / log loss with the all-zero block vs the shooter's A1 block.

    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_A1_v1.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402

BASE = Path("data/processed/models/engine_v3")
OUT = Path("data/processed/models/engine_v3_I_A1")
TAG = "F2_2025"
if OUT.exists():
    raise SystemExit(f"{OUT} exists: never overwrite")

z = dict(np.load(BASE / f"arrays_{TAG}.npz"))
names = json.loads((BASE / f"names_{TAG}.json").read_text(encoding="utf-8"))
games = pd.read_parquet(BASE / f"games_{TAG}.parquet").reset_index(drop=True)
sn = names["slot_names"]
ros = z["roster_cbbd"]
G, _, S = ros.shape

# ---- FT history -------------------------------------------------------------
ES.load_universe()
att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
att = att[att["season"].isin([2022, 2023, 2024, 2025])]
d = FT.build_ft_design(att)
lg = d[d["season"] == 2025].groupby("game_date")["lg_ft_asof"].first().sort_index()
prev = d[d["season"] == 2024].groupby("shooter_id")["y"].agg(["sum", "size"])
prev_rate = (prev["sum"] / prev["size"]).to_dict()
cur = d[d["season"] == 2025].groupby(["shooter_id", "game_date"])["y"].agg(["sum", "size"]).reset_index()
cur = cur.sort_values(["shooter_id", "game_date"])
cur["cum_m"] = cur.groupby("shooter_id")["sum"].cumsum() - cur["sum"]       # strictly before that game date
cur["cum_a"] = cur.groupby("shooter_id")["size"].cumsum() - cur["size"]
cur_by = {k: v for k, v in cur.groupby("shooter_id")}


def lg_on(date):
    i = lg.index.searchsorted(date, side="right") - 1
    return float(lg.iloc[max(i, 0)])


def block(pid, date):
    """(shooter_ft_asof, shooter_fta_asof, prior_season_ft, has_prior_season) as FT.build_ft_design defines them."""
    L = lg_on(date)
    a = m = 0.0
    x = cur_by.get(pid)
    if x is not None:
        before = x[x["game_date"] < date]
        same = x[x["game_date"] == date]
        if len(same):
            a, m = float(same["cum_a"].iloc[0]), float(same["cum_m"].iloc[0])
        elif len(before):
            a, m = float(before["cum_a"].iloc[-1] + before["size"].iloc[-1]), float(before["cum_m"].iloc[-1] + before["sum"].iloc[-1])
    own = (m / a - L) if a > 0 else 0.0
    pr = prev_rate.get(pid)
    return own, a, ((pr - L) if pr is not None else 0.0), (1.0 if pr is not None else 0.0)


# ---- rosters and prior-season minutes ----------------------------------------
r = pd.read_parquet("data/raw/cbbd/rosters/roster_2025.parquet", columns=["cbbd_player_id", "team_source_id", "source_id"])
r = r.dropna(subset=["cbbd_player_id", "team_source_id"])
r["cbbd_player_id"] = r["cbbd_player_id"].astype("int64")
r["team_source_id"] = pd.to_numeric(r["team_source_id"], errors="coerce")
pb24 = pd.read_parquet("data/raw/hoopr/player_box/player_box_2024.parquet", columns=["athlete_id", "minutes"])
mins = pb24.groupby("athlete_id")["minutes"].sum()
r["min24"] = pd.to_numeric(r["source_id"], errors="coerce").map(mins).fillna(0.0)
team_roster = {int(t): x.sort_values(["min24", "cbbd_player_id"], ascending=[False, True])["cbbd_player_id"].tolist()
               for t, x in r.groupby("team_source_id")}

sl = z["slot_static"].copy()
cols = [sn["shooter_ft_asof"], sn["shooter_fta_asof"], sn["prior_season_ft"], sn["has_prior_season_ft"]]
assert not np.any(sl[..., cols][ros <= 0]), "anonymous slots were expected to carry an all-zero FT block"
filled = {}
n_anon = n_fill = 0
dates = pd.to_datetime(games["game_date"]).to_numpy()
for g in range(G):
    for side, tcol in ((0, "home_team_id"), (1, "away_team_id")):
        anon = np.flatnonzero(ros[g, side] <= 0)
        n_anon += len(anon)
        if not len(anon):
            continue
        named = set(int(p) for p in ros[g, side] if p > 0)
        pool = [p for p in team_roster.get(int(games[tcol].iloc[g]), []) if p not in named]
        for k, s in enumerate(anon[:len(pool)]):
            pid = pool[k]
            sl[g, side, s, cols] = np.array(block(pid, pd.Timestamp(dates[g])), dtype=np.float32)
            filled[(g, side, pid)] = s
            n_fill += 1

OUT.mkdir(parents=True)
for f in (f"games_{TAG}.parquet", f"names_{TAG}.json", f"event_block_{TAG}.npz"):
    shutil.copy2(BASE / f, OUT / f)
z2 = dict(z)
z2["slot_static"] = sl
np.savez_compressed(OUT / f"arrays_{TAG}.npz", **z2)

# ---- offline check on real off-roster attempts --------------------------------
from cbb_sim.engine.adapters import Adapters  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
inp = EngineInputs.load(str(BASE), TAG)
ad = Adapters.load(inp, "F2", 2025)
pos = {int(gid): i for i, gid in enumerate(games["game_id"].to_numpy())}
dd = d[(d["season"] == 2025) & d["game_id"].isin(pos)].reset_index(drop=True)
gi = dd["game_id"].map(pos).to_numpy().astype(np.int64)
side = np.where(dd["team_id"].to_numpy() == games["home_team_id"].to_numpy()[gi], 0, 1)
sid = dd["shooter_id"].to_numpy().astype(np.int64)
on = (ros[gi, side] == sid[:, None]).any(axis=1)
off = np.flatnonzero(~on)
FEATS = ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season", "season_idx",
         "seconds_remaining", "period", "score_diff", "in_bonus"]
X = np.asarray(FT.design_matrix(dd.iloc[off], tuple(FEATS)), dtype=np.float32)
cov = np.array([(int(gi[i]), int(side[i]), int(sid[i])) in filled for i in off])
Xz, Xa = X.copy(), X.copy()
Xz[:, :4] = 0.0
for j, i in enumerate(off):
    s = filled.get((int(gi[i]), int(side[i]), int(sid[i])))
    if s is not None:
        Xa[j, :4] = sl[gi[i], side[i], s, cols]


def pr(Xm):
    segs = ad.ft.manifest.segments(gi[off])
    p = np.empty(len(Xm))
    for k in np.unique(segs):
        rr = np.flatnonzero(segs == k)
        p[rr] = ad.ft.models_by_seg[k].predict_proba(Xm[rr])[:, 1]
    return p


y = dd["y"].to_numpy()[off].astype(float)


def ll(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


pz, pa, preal = pr(Xz), pr(Xa), pr(X)
rep = {"base": str(BASE), "out": str(OUT), "anonymous_slots": int(n_anon), "anonymous_slots_filled": int(n_fill),
       "changed_columns": ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season_ft"],
       "only_anonymous_slots_changed": bool(np.array_equal(sl[ros > 0], z["slot_static"][ros > 0])),
       "offline_off_roster": {"n": int(len(off)), "coverage": float(cov.mean()), "ft_pct": float(y.mean()),
                              "mean_p_zero_block": float(pz.mean()), "mean_p_A1_block": float(pa.mean()),
                              "mean_p_real_features": float(preal.mean()),
                              "log_loss_zero": ll(pz), "log_loss_A1": ll(pa), "log_loss_real": ll(preal),
                              "covered_mean_p_zero": float(pz[cov].mean()) if cov.any() else None,
                              "covered_mean_p_A1": float(pa[cov].mean()) if cov.any() else None,
                              "covered_ft_pct": float(y[cov].mean()) if cov.any() else None}}
(OUT / "builder_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
print(json.dumps(rep, indent=1, default=float))
