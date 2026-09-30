#!/usr/bin/env python
"""
build_engine_shot_block_lut_v1.py -- engine lookups for the DRAWN block flag
(`docs/models/shot_block/experiments.md` section 5). Sibling inputs: nothing
served is read-modified or overwritten.

For every game of the engine slate `games_F2_2025_v2.parquet` (5,710 games) it
writes `data/processed/models/engine/shot_block_<arm>_F2_2025.npz` with

    game_id       (G,)            the engine slate order (checked at load)
    team          (G, 2, 2)       [side as DEFENCE: def_block_c, side as OFFENCE: off_blocked_c]
    shooter       (G, 2, S)       shooter_blocked_c per roster slot (k = 50)
    known         (G, 2, S)       1 for a real roster player, 0 for an anonymous tail slot
    anchor        (G, 3)          logit(L_asof(type, date)) - logit(Lbar(type)), rim/jump2/three
                                  (zeros for arm K2)
    coef (18,), mu (17,), sd (17,), features (17,)

Model: the F2 fit (train 2022-2024) of `train_shot_block_v2_round2`'s K2 /
K2_Ocell spec, refit here with the same function and data. Every as-of value is
strictly before the GAME DATE (cumulative daily totals, `searchsorted` left),
never the game itself. Anchor day 0 = 2024 end level per shot type.

    .venv/Scripts/python.exe scripts/build_engine_shot_block_lut_v1.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp_season_drift_anchor_v1 as X  # noqa: E402
import train_shot_block_v1 as SB  # noqa: E402
from cbb_sim import season_anchor as SA  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402

ENGINE_DIR = ROOT / "data/processed/models/engine"
TYPES = ("rim", "jump2", "three")
K_SHOOTER = 50.0
TRAIN = [2022, 2023, 2024]
TEST = 2025


def asof_by_date(keys: pd.DataFrame, events: pd.DataFrame, key: str, num: str, den: str) -> tuple:
    """(num_before, den_before) for each (key, date) in `keys`, from events of the
    same season strictly before the date."""
    ev = events.groupby([key, "game_date"], as_index=False)[[num, den]].sum()
    ev = ev.sort_values([key, "game_date"])
    ev["cn"] = ev.groupby(key)[num].cumsum()
    ev["cd"] = ev.groupby(key)[den].cumsum()
    out_n = np.zeros(len(keys))
    out_d = np.zeros(len(keys))
    grp = {k: g for k, g in ev.groupby(key)}
    kk = keys[key].to_numpy()
    dd = keys["game_date"].to_numpy()
    for i, (k, d) in enumerate(zip(kk, dd)):
        g = grp.get(k)
        if g is None:
            continue
        j = np.searchsorted(g["game_date"].to_numpy(), d, side="left") - 1
        if j >= 0:
            out_n[i] = g["cn"].to_numpy()[j]
            out_d[i] = g["cd"].to_numpy()[j]
    return out_n, out_d


def league_before(dates: np.ndarray, events: pd.DataFrame, num: str, den: str) -> np.ndarray:
    day = events.groupby("game_date")[[num, den]].sum().sort_index()
    cn, cd = day[num].cumsum().to_numpy(), day[den].cumsum().to_numpy()
    j = np.searchsorted(day.index.to_numpy(), dates, side="left") - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(j >= 0, cn[np.maximum(j, 0)] / np.maximum(cd[np.maximum(j, 0)], 1e-9), np.nan)


def main() -> int:
    assert_not_sealed(TRAIN + [TEST], context="shot_block engine LUT")
    inp = EngineInputs.load(ENGINE_DIR, "F2_2025")
    g = inp.games.reset_index(drop=True)
    G, S = len(g), inp.n_slots
    gdate = pd.to_datetime(g["game_date"]).dt.normalize().to_numpy()

    full, _ = SB.build_design()
    full["game_date"] = pd.to_datetime(full["game_date"]).dt.normalize()
    ev = full[full["season"] == TEST].copy()
    ev["blk"] = ev["blocked"].astype("float64")
    ev["n"] = 1.0

    # ---- team as-of block rates (league-centred), per game x side ------------
    team = np.zeros((G, 2, 2), dtype="float32")
    lg = league_before(gdate, ev, "blk", "n")
    for side, tcol in ((0, "home_team_id"), (1, "away_team_id")):
        keys = pd.DataFrame({"t": g[tcol].to_numpy(), "game_date": gdate})
        dn, dd = asof_by_date(keys, ev.rename(columns={"def_team_id": "t"}), "t", "blk", "n")
        on, od = asof_by_date(keys, ev.rename(columns={"off_team_id": "t"}), "t", "blk", "n")
        with np.errstate(invalid="ignore", divide="ignore"):
            dv, ov = dn / np.where(dd > 0, dd, np.nan), on / np.where(od > 0, od, np.nan)
        team[:, side, 0] = np.where(np.isnan(dv) | np.isnan(lg), 0.0, dv - lg)
        team[:, side, 1] = np.where(np.isnan(ov) | np.isnan(lg), 0.0, ov - lg)

    # ---- shooter as-of rate per roster slot ---------------------------------
    shooter = np.zeros((G, 2, S), dtype="float32")
    known = (inp.roster_cbbd >= 0).astype("uint8")
    evs = ev[ev["shooter_id"] >= 0]
    rid = inp.roster_cbbd.reshape(-1)
    keys = pd.DataFrame({"t": rid, "game_date": np.repeat(gdate, 2 * S)})
    sn, sd_ = asof_by_date(keys, evs.rename(columns={"shooter_id": "t"}), "t", "blk", "n")
    # the design's shooter rate is centred on the league rate over KNOWN-shooter rows
    lg_s = league_before(gdate, evs, "blk", "n")
    lgr = np.repeat(np.where(np.isnan(lg_s), 0.0, lg_s), 2 * S)
    val = (sn + K_SHOOTER * lgr) / (sd_ + K_SHOOTER) - lgr
    shooter[:] = np.where(rid >= 0, val, 0.0).reshape(G, 2, S)

    # ---- validation against the design's own as-of columns -------------------
    gpos = pd.Series(np.arange(G), index=g["game_id"].to_numpy())
    chk = ev[ev["game_id"].isin(gpos.index)].drop_duplicates(["game_id", "def_team_id"])
    gi = gpos.loc[chk["game_id"]].to_numpy()
    side_def = np.where(chk["def_team_id"].to_numpy() == g["home_team_id"].to_numpy()[gi], 0, 1)
    d_team = np.abs(team[gi, side_def, 0] - chk["def_block_c"].to_numpy())
    val_rep = {"team_def_block_c_match_1e-6": float((d_team < 1e-6).mean()),
               "team_def_block_c_max_abs": float(d_team.max()), "n_team_games": int(len(chk))}

    # ---- models (F2 fit) and anchors -----------------------------------------
    F = X.fold_data("shot_block", "F2")
    df, tr = F["df"], F["tr"]
    Xm = df[SB.KC].to_numpy(dtype="float32")
    meta = {"validation": val_rep, "features": list(SB.KC), "k_shooter": K_SHOOTER,
            "slate": "games_F2_2025 (engine inputs v2)", "n_games": G}
    for arm in ("K2", "K2_Ocell"):
        anchor = np.zeros((G, 3), dtype="float32")
        off = None
        if arm == "K2_Ocell":
            off = np.zeros(len(df))
            for ti, t in enumerate(TYPES):
                m = F["sub"] == t
                a = SA.anchor_O(df.loc[m, "season"].to_numpy(), df.loc[m, "game_date"].to_numpy(),
                                F["num"][m], F["den"][m], TRAIN, "binary")
                off[m] = a.offset()[:, 0]
                # engine slate: as-of level of type t strictly before each game date
                evt = ev[ev["miss_type"] == t]
                L = league_before(gdate, evt, "blk", "n")
                prior = float(np.asarray(a.meta["prior_by_season"][TEST])[0])
                L = np.where(np.isnan(L), prior, L)
                anchor[:, ti] = (SA.link(L, "binary") - SA.link(a.Lbar[0], "binary")).astype("float32")
                meta[f"{arm}_{t}"] = {"Lbar": float(a.Lbar[0]), "prior_2024_end": prior}
        mdl = X.fit_glm(Xm[tr], F["num"][tr, 0], None if off is None else off[tr], None, "binomial")
        out = ENGINE_DIR / f"shot_block_{arm}_F2_2025.npz"
        np.savez_compressed(out, game_id=g["game_id"].to_numpy(), team=team, shooter=shooter,
                            known=known, anchor=anchor, coef=mdl["theta"], mu=mdl["mu"],
                            sd=mdl["sd"], features=np.array(SB.KC))
        meta[arm] = {"file": str(out.relative_to(ROOT)), "converged": mdl["converged"],
                     "coef": [round(float(c), 6) for c in mdl["theta"]]}
        print(arm, "->", out, mdl["converged"], flush=True)
    meta["created_at"] = pd.Timestamp.now("UTC").isoformat()
    (ENGINE_DIR / "shot_block_lut_F2_2025.meta.json").write_text(json.dumps(meta, indent=1, default=str),
                                                                 encoding="utf-8")
    print(json.dumps(val_rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
