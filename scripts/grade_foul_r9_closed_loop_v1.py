"""grade_foul_r9_closed_loop_v1.py -- foul round 9 closed-loop HALF lines (PO experiments.md s26.3).

Reads, per tag, `half_agg.parquet` (tap v2 --agg-halves) or `poss_tap.parquet` (tap v2), reduces both to sums per
(seed, game, offence side, half), and compares with the 2025 event layer on the same games (`possessions_v2`
chances, offence side from `offense_is_home`). Lines per arm:
  FTA/FGA H1, H2, OT, pooled; |sim - actual|; paired delta of |gap| vs the reference tag with a paired game-bootstrap
  SE (200 draws, games resampled, seeds pooled); floor = max(SD over the floor tags' values [the reference's spec-
  identical reruns on other seeds, Decision 12], bootstrap SE of the paired delta); movement in floors;
  occupancy (share of possessions opened in the bonus) by half; and-ones / poss by half;
  per team (offence): FT-rate (FTA/FGA) slope over prior-season (2024) FT-rate quintiles (sim span / actual span) and
  team SD ratio (SD of team sim means / SD of team actual), paired bootstrap SE of the slope delta vs the reference;
  per game: MAE and correlation of game-side FTA/FGA (sim mean over seeds vs actual).

    grade_foul_r9_closed_loop_v1.py OUT_STEM --ref TAG --arms TAG,TAG --floors TAG,TAG [--extra TAG,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/engine_v0")
COLS = ["poss", "in_bonus", "fta", "fga", "and_one"]
NB = 200


def load_sim(tag: str) -> pd.DataFrame:
    d = R / tag
    if (d / "half_agg.parquet").exists():
        a = pd.read_parquet(d / "half_agg.parquet")
    else:
        t = pd.read_parquet(d / "poss_tap.parquet")
        t["half"] = np.where(t["period"] >= 3, 3, t["period"])
        t["and_one"] = t["trip_fouls"] - t["n_shoot_trip"] - t["n_bonus_trip"]
        t["poss"] = 1
        t["in_bonus"] = (t["def_fouls"] >= t["bonus_thr"]).astype(int)
        a = t.groupby(["seed", "game_id", "off_side", "half"])[COLS].sum().reset_index()
    return a[["seed", "game_id", "off_side", "half"] + COLS]


def load_actual(ids) -> pd.DataFrame:
    c = pd.read_parquet("data/processed/possessions_v2/chances_2025.parquet")
    c = c[c["game_id"].isin(ids)].copy()
    c["fga"] = c["fga_rim"] + c["fga_jump2"] + c["fga_3"]
    c["half"] = np.where(c["period"] >= 3, 3, c["period"])
    c["off_side"] = np.where(c["offense_is_home"], 0, 1)
    c["and_one"] = c["and_one"].astype(int)
    a = c.groupby(["game_id", "off_side", "half"]).agg(fta=("fta", "sum"), fga=("fga", "sum"),
                                                        and_one=("and_one", "sum")).reset_index()
    p = c.drop_duplicates(["game_id", "period", "poss_index"])
    a = a.merge(p.groupby(["game_id", "off_side", "half"]).size().rename("poss").reset_index(),
                on=["game_id", "off_side", "half"])
    acc = pd.read_parquet("data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet",
                          columns=["game_id", "period", "poss_index", "offense_is_home", "def_team_fouls_true"])
    acc = acc[acc["game_id"].isin(ids)]
    acc["half"] = np.where(acc["period"] >= 3, 3, acc["period"])
    acc["off_side"] = np.where(acc["offense_is_home"], 0, 1)
    acc["in_bonus"] = (acc["def_team_fouls_true"] >= 6).astype(int)
    a = a.merge(acc.groupby(["game_id", "off_side", "half"])["in_bonus"].sum().reset_index(),
                on=["game_id", "off_side", "half"], how="left")
    a["seed"] = 0
    return a


def team_map(ids):
    g = pd.read_parquet("data/processed/possessions_v2/possessions_2025.parquet",
                        columns=["game_id", "offense_team_id", "offense_is_home"]).drop_duplicates(["game_id", "offense_is_home"])
    g = g[g["game_id"].isin(ids)]
    g["off_side"] = np.where(g["offense_is_home"], 0, 1)
    p = pd.read_parquet("data/processed/possessions_v2/chances_2024.parquet",
                        columns=["offense_team_id", "fta", "fga_rim", "fga_jump2", "fga_3"])
    p["fga"] = p["fga_rim"] + p["fga_jump2"] + p["fga_3"]
    pr = p.groupby("offense_team_id")[["fta", "fga"]].sum()
    pr = (pr["fta"] / pr["fga"]).rename("prior_ftr")
    return g[["game_id", "off_side", "offense_team_id"]].merge(pr, left_on="offense_team_id", right_index=True, how="left")


def per_game(a: pd.DataFrame) -> pd.DataFrame:
    """mean over seeds of per (game, side, half) sums."""
    # sum over seeds / number of seeds: a half present in only some seeds (OT) keeps its true weight
    return a.groupby(["game_id", "off_side", "half"])[COLS].sum() / a["seed"].nunique()


def ratio(df, half=None):
    x = df if half is None else df[df.index.get_level_values("half") == half]
    return float(x["fta"].sum() / x["fga"].sum())


def team_slope(pg: pd.DataFrame, tm: pd.DataFrame, games=None) -> tuple[float, float, float]:
    x = pg.groupby(["game_id", "off_side"])[["fta", "fga"]].sum().reset_index().merge(tm, on=["game_id", "off_side"])
    if games is not None:
        x = x.set_index("game_id")
        x = x.loc[[g for g in games if g in x.index]].reset_index()
    t = x.groupby("offense_team_id").agg(fta=("fta", "sum"), fga=("fga", "sum"), prior=("prior_ftr", "first"))
    t = t.dropna()
    t["r"] = t["fta"] / t["fga"]
    q = pd.qcut(t["prior"].rank(method="first"), 5, labels=False)
    m = t.groupby(q)["r"].mean()
    return float(m.iloc[-1] - m.iloc[0]), float(t["r"].std()), m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arms", default="")
    ap.add_argument("--floors", default="")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    arms = [t for t in a.arms.split(",") if t]
    floors = [t for t in a.floors.split(",") if t]
    extra = [t for t in a.extra.split(",") if t]
    tags = [a.ref] + arms + floors + extra
    S = {t: load_sim(t) for t in tags}
    ids = np.sort(S[a.ref]["game_id"].unique())
    A = load_actual(ids).set_index(["game_id", "off_side", "half"])[COLS]
    PG = {t: per_game(s) for t, s in S.items()}
    tm = team_map(ids)
    lines = {"H1": 1, "H2": 2, "OT": 3, "pooled": None}
    act = {k: ratio(A, h) for k, h in lines.items()}
    res = {"actual": act, "n_games": int(len(ids)), "seeds": {t: int(S[t]["seed"].nunique()) for t in tags}, "arms": {}}
    rng = np.random.default_rng(20260930)
    # game-level sums for the bootstrap
    def g_sums(pg, h):
        x = pg if h is None else pg[pg.index.get_level_values("half") == h]
        return x.groupby("game_id")[["fta", "fga"]].sum().reindex(ids).fillna(0)
    Ag = {k: g_sums(A, h) for k, h in lines.items()}
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(NB)]
    act_slope, act_sd, act_q = team_slope(A, tm)
    res["actual_team"] = {"slope_span": act_slope, "team_sd": act_sd, "quintiles": act_q.round(4).tolist()}
    for t in tags:
        r = {}
        for k, h in lines.items():
            v = ratio(PG[t], h)
            r[k] = {"value": v, "gap": v - act[k]}
            r[k]["occupancy"] = float(PG[t]["in_bonus"].sum() / PG[t]["poss"].sum()) if h is None else float(
                PG[t].xs(h, level="half")["in_bonus"].sum() / PG[t].xs(h, level="half")["poss"].sum())
            r[k]["and_one_pp"] = float((PG[t]["and_one"].sum() if h is None else PG[t].xs(h, level="half")["and_one"].sum())
                                       / (PG[t]["poss"].sum() if h is None else PG[t].xs(h, level="half")["poss"].sum()))
        sl, sd, q = team_slope(PG[t], tm)
        r["team"] = {"slope_ratio": sl / act_slope, "team_sd_ratio": sd / act_sd, "quintiles": q.round(4).tolist()}
        # per game x side, pooled halves
        gs = PG[t].groupby(["game_id", "off_side"])[["fta", "fga"]].sum()
        ga = A.groupby(["game_id", "off_side"])[["fta", "fga"]].sum().reindex(gs.index)
        x, y = gs["fta"] / gs["fga"], ga["fta"] / ga["fga"]
        ok = x.notna() & y.notna() & np.isfinite(y)
        r["per_game"] = {"mae": float((x - y)[ok].abs().mean()), "corr": float(np.corrcoef(x[ok], y[ok])[0, 1])}
        res["arms"][t] = r
    # actual occupancy / and-ones
    for k, h in lines.items():
        x = A if h is None else A.xs(h, level="half")
        res["actual"][f"{k}_occupancy"] = float(x["in_bonus"].sum() / x["poss"].sum())
        res["actual"][f"{k}_and_one_pp"] = float(x["and_one"].sum() / x["poss"].sum())
    # paired deltas vs ref, floors
    for t in arms + extra:
        pt = {}
        for k, h in lines.items():
            sr, sa = g_sums(PG[a.ref], h), g_sums(PG[t], h)
            ag = Ag[k]
            d0 = abs(sa["fta"].sum() / sa["fga"].sum() - act[k]) - abs(sr["fta"].sum() / sr["fga"].sum() - act[k])
            bs = []
            for i in boots:
                ak = act[k]   # the target is fixed (these games' actual); only sim noise is resampled
                bs.append(abs(sa["fta"].to_numpy()[i].sum() / sa["fga"].to_numpy()[i].sum() - ak)
                          - abs(sr["fta"].to_numpy()[i].sum() / sr["fga"].to_numpy()[i].sum() - ak))
            se = float(np.std(bs))
            draws = [res["arms"][a.ref][k]["value"]] + [res["arms"][f][k]["value"] for f in floors]
            sd = float(np.std(draws, ddof=1)) if len(draws) > 1 else float("nan")
            fl = float(np.nanmax([sd, se]))
            pt[k] = {"delta_abs_gap": d0, "boot_se": se, "draw_sd": sd, "n_draws": len(draws), "floor": fl,
                     "floors_toward": -d0 / fl if fl > 0 else float("nan")}
        # team slope paired bootstrap
        bs = []
        for i in boots[:100]:
            gsel = ids[i]
            s1 = team_slope(PG[t], tm, gsel)[0]
            s0 = team_slope(PG[a.ref], tm, gsel)[0]
            sa_ = team_slope(A, tm, gsel)[0]
            bs.append((s1 - s0) / sa_)
        pt["team_slope_delta"] = {"delta": res["arms"][t]["team"]["slope_ratio"] - res["arms"][a.ref]["team"]["slope_ratio"],
                                  "boot_se": float(np.std(bs))}
        res["arms"][t]["paired_vs_ref"] = pt
    Path(f"{a.out}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    # text table
    L = [f"actual (event layer, {len(ids)} games): " + ", ".join(f"{k} {v:.4f}" for k, v in act.items())]
    L.append("| tag | seeds | H1 | H2 | OT | pooled | H1 occ | H1 ao/poss | H2 ao/poss | team slope ratio | team SD ratio | per-game MAE / corr |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for t in tags:
        r = res["arms"][t]
        L.append(f"| {t} | {res['seeds'][t]} | {r['H1']['value']:.4f} | {r['H2']['value']:.4f} | {r['OT']['value']:.4f} | "
                 f"{r['pooled']['value']:.4f} | {r['H1']['occupancy']:.4f} | {r['H1']['and_one_pp']:.4f} | "
                 f"{r['H2']['and_one_pp']:.4f} | {r['team']['slope_ratio']:.3f} | {r['team']['team_sd_ratio']:.3f} | "
                 f"{r['per_game']['mae']:.4f} / {r['per_game']['corr']:.3f} |")
    L.append(f"| actual | - | {act['H1']:.4f} | {act['H2']:.4f} | {act['OT']:.4f} | {act['pooled']:.4f} | "
             f"{res['actual']['H1_occupancy']:.4f} | {res['actual']['H1_and_one_pp']:.4f} | {res['actual']['H2_and_one_pp']:.4f} | 1 | 1 | - |")
    L.append("")
    L.append(f"Paired vs {a.ref} (delta |gap|, negative = closer; floor = max(draw SD over ref + floors, paired boot SE)):")
    L.append("| arm | line | delta abs gap | boot SE | draw SD (n) | floor | floors toward |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for t in arms + extra:
        for k in lines:
            p = res["arms"][t]["paired_vs_ref"][k]
            L.append(f"| {t} | {k} | {p['delta_abs_gap']:+.4f} | {p['boot_se']:.4f} | {p['draw_sd']:.4f} ({p['n_draws']}) | "
                     f"{p['floor']:.4f} | {p['floors_toward']:+.2f} |")
        ts = res["arms"][t]["paired_vs_ref"]["team_slope_delta"]
        L.append(f"| {t} | team slope ratio | {ts['delta']:+.3f} | {ts['boot_se']:.3f} | | | |")
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
