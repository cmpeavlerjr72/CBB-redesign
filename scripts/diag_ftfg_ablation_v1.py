"""diag_ftfg_ablation_v1.py -- served v2 vs ENGINE_FT_SCORE=FTn on the same games and seeds (lane B, 2026-10-01).

DIAGNOSTIC (attribution by ablation, Decision 10 style); adopts nothing. Compares the FT x opponent-FG
covariance split (luck / composition / FTA volume, by half and lead state) between the served tap
(`results/g5_channels/ftfg_tap`) and the FTn tap (`results/g5_channels/ftfg_tap_FTn`) on the games the
FTn tap covers, plus FT%, within-team FT x own-FG and the points-level home/away covariance.
Base rates p0 / f0 are the served 200-seed per-game means in both arms (same yardstick).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diag_ftfg_report_v1 as RP  # noqa: E402


def load(tapdir: Path, gids=None):
    trips = pd.concat([pd.read_parquet(p) for p in sorted(tapdir.glob("trips_*.parquet"))], ignore_index=True)
    games = pd.concat([pd.read_parquet(p) for p in sorted(tapdir.glob("games_*.parquet"))], ignore_index=True)
    if gids is not None:
        games = games[games["game_id"].isin(gids)]
        trips = trips[trips["game_id"].isin(gids)]
    return trips, games.sort_values(["game_id", "seed"]).reset_index(drop=True)


def prep(trips, games, b):
    games = games[games["game_id"].isin(b.index)].reset_index(drop=True)
    X = RP.fg_resid(games, b)
    t = trips.copy()
    t["half"] = RP.ctx_half(t["period"].to_numpy(), t["sec"].to_numpy())
    t["lead"] = RP.ctx_lead((t["pts_off"] - t["pts_def"]).to_numpy())
    t["kind"] = np.where(t["oao"], "1and1", np.where(t["n_att"] == 1, "and1", "2or3shot"))
    f0 = np.where(t["side"] == 0, b.loc[t["game_id"], "f0_h"].to_numpy(), b.loc[t["game_id"], "f0_a"].to_numpy())
    t["luck"] = t["ftm"] - t["psum"]
    t["comp"] = t["psum"] - f0 * t["fta"]
    t["s"] = np.where(t["side"] == 0, "h", "a")
    return games, X, t


def own_team(games, X, t):
    """within-team FT make residual (luck + comp) x own FG residual, both sides summed."""
    key = games[["game_id", "seed"]]
    gid = games["game_id"].to_numpy()
    out = 0.0
    for s in ("h", "a"):
        f = t[t["s"] == s].groupby(["game_id", "seed"])[["luck", "comp"]].sum().sum(1)
        y = f.reindex(pd.MultiIndex.from_frame(key)).fillna(0.0).to_numpy()
        for k in RP.KT:
            out += RP.wcov_pooled(X[f"{k}_{s}"].to_numpy(), y, gid)
    return out


def main():
    """argv: optional extra arms as name=tapdir (default: FTn only); served restricted to the FTn games."""
    b = RP.base_rates()
    tF, gF = load(ROOT / "results/g5_channels/ftfg_tap_FTn")
    gids = set(gF["game_id"])
    tS, gS = load(ROOT / "results/g5_channels/ftfg_tap", gids)
    rep = {"n_games": len(gids)}
    arms = {"served": (tS, gS), "FTn": (tF, gF)}
    for a in sys.argv[1:]:
        nm, d = a.split("=", 1)
        arms[nm] = load(ROOT / d, gids)
    for nm, (t, g) in arms.items():
        games, X, tt = prep(t, g, b)
        st = RP.sim_tables(games, X, tt, b)
        allr = st[st["dims"] == "all"].set_index("part")["total"].to_dict()
        half = st[st["dims"] == "half"][["cell", "part", "total"]].values.tolist()
        lead = st[st["dims"] == "lead"][["cell", "part", "total"]].values.tolist()
        gid = games["game_id"].to_numpy()
        cov_pts = RP.wcov_pooled(games["home_pts"].to_numpy(float), games["away_pts"].to_numpy(float), gid)
        hp, ap_ = games["home_pts"].to_numpy(float), games["away_pts"].to_numpy(float)
        var_t = RP.wcov_pooled(hp + ap_, hp + ap_, gid)
        var_m = RP.wcov_pooled(hp - ap_, hp - ap_, gid)
        ftp = float((games["home_ftm"].sum() + games["away_ftm"].sum()) / (games["home_fta"].sum() + games["away_fta"].sum()))
        rep[nm] = {"opp_FG_x_FT": allr, "by_half": half, "by_lead": lead,
                   "own_FG_x_FT": own_team(games, X, tt), "cov_home_away_pts": cov_pts,
                   "within_var_total": var_t, "within_var_margin": var_m, "ft_pct": ftp,
                   "mean_total": float((games["home_pts"] + games["away_pts"]).mean())}
    out = ROOT / ("results/g5_channels/ftfg_ablation_v1.json" if len(sys.argv) == 1
                  else "results/g5_channels/ftfg_ablation_r16_v1.json")
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
