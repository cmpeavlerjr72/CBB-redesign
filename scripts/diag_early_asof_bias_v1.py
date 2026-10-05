"""Early-season as-of team-rate feature check (diagnostic, nothing fitted or served).

Question (PM, 2026-10-05): are the as-of team TOV and FTA-rate features that feed possession_outcome
biased or mis-shrunk early in the season, by days-since-start bucket, folds 1 (test 2024) and 2 (test 2025)?

Served features: possession_outcome round-2 team block (`live.features.po_team_block_r2`): possessions v2
chance tables, first-chance style source, pbp_complete universe, expanding mean centred on the league
as-of mean (`PO.build_team_form`). Target: the team's realised REST-OF-SEASON rate (this game and later),
centred on the league rate over the same date range; also the next-game realised centred rate.
Reference arm: E3 (team_rate_estimator, box-based, `team_rate_features_E3_v4.parquet`), scored against
the same first-chance targets (scale differs slightly; slope comparisons only).

Output: results/early_asof_bias/phase1_v1.csv (+ printed table). 2025-26 (season 2026) is not loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.models import possession_outcome as PO  # noqa: E402

SEASONS = [2023, 2024, 2025]          # 2023 needed only as context; tests are 2024 (F1) and 2025 (F2)
TEST = {2024: "F1", 2025: "F2"}
BUCKETS = [(0, 14, "d0-14"), (15, 45, "d15-45"), (46, 999, "d46+")]
RATES = {"tov": ("tov", "poss"), "ftr": ("fta", "fga")}
OUT = ROOT / "results" / "early_asof_bias"
RNG = np.random.default_rng(20261005)
NBOOT = 300


def load():
    uni = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
    uni = uni[uni["is_d1_game"] & ~uni["pbp_truncated"] & uni["pbp_complete"]].copy()
    uni["game_date"] = pd.to_datetime(uni["game_date"])
    keep = set(uni["game_id"])
    frames = {}
    for s in SEASONS:
        c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{s}.parquet")
        frames[s] = c[c["game_id"].isin(keep)]
    form = PO.build_team_form(frames, uni, style_source="first_chance")
    boxes = pd.concat([PO.team_game_box_first_chance(f) for f in frames.values()], ignore_index=True)
    boxes = boxes.merge(uni[["game_id", "game_date"]], on="game_id", how="left")
    return uni, form, boxes


def ros_targets(boxes: pd.DataFrame) -> pd.DataFrame:
    """Per (season, game, team): the team's rest-of-season (this game onward) and next-game rates,
    offence and defence-allowed, each centred on the league rate over the same window."""
    out = []
    cols = ["poss", "tov", "fta", "fga"]
    first = boxes.groupby("season")["game_date"].transform("min")
    boxes = boxes.assign(days=(boxes["game_date"] - first).dt.days)
    # league reverse-cumulative by date (games on or after date d)
    day = boxes.groupby(["season", "game_date"], as_index=False)[cols].sum()
    day = day.sort_values(["season", "game_date"], ascending=[True, False])
    for c in cols:
        day[f"lgros_{c}"] = day.groupby("season")[c].cumsum()
    day = day.drop(columns=cols)
    for side, key in (("off", "team_id"), ("def", "opp_id")):
        b = boxes.rename(columns={key: "_t"}).sort_values(["season", "_t", "game_date", "game_id"],
                                                          ascending=[True, True, False, False])
        g = b.groupby(["season", "_t"], sort=False)
        for c in cols:
            b[f"ros_{c}"] = g[c].cumsum()
        b = b.sort_values(["season", "_t", "game_date", "game_id"])
        g = b.groupby(["season", "_t"], sort=False)
        for c in cols:
            b[f"nxt_{c}"] = g[c].shift(-1)
        b = b.merge(day, on=["season", "game_date"], how="left")
        r = b[["season", "game_id", "_t", "days"]].rename(columns={"_t": "team_id"}).copy()
        for name, (num, den) in RATES.items():
            lg = 100 * b[f"lgros_{num}"] / b[f"lgros_{den}"]
            r[f"{side}_{name}_ros"] = (100 * b[f"ros_{num}"] / b[f"ros_{den}"].replace(0, np.nan) - lg).to_numpy()
            r[f"{side}_{name}_nxt"] = (100 * b[f"nxt_{num}"] / b[f"nxt_{den}"].replace(0, np.nan) - lg).to_numpy()
            r[f"{side}_{name}_ros_den"] = b[f"ros_{den}"].to_numpy()
        out.append(r.set_index(["season", "game_id", "team_id"]))
    t = out[0].join(out[1].drop(columns=["days"]), how="inner").reset_index()
    return t


def slope_stats(x, y, team, nboot=NBOOT):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, team = x[m], y[m], team[m]
    if len(x) < 30 or np.var(x) <= 0:
        return np.nan, np.nan, np.nan, np.nan, int(m.sum())

    def fit(ix):
        xx, yy = x[ix], y[ix]
        v = np.var(xx)
        return np.cov(xx, yy, bias=True)[0, 1] / v if v > 0 else np.nan, np.mean(xx - yy)

    b, bias = fit(np.arange(len(x)))
    teams, inv = np.unique(team, return_inverse=True)
    rows_by = [np.flatnonzero(inv == k) for k in range(len(teams))]
    bs, bb = [], []
    for _ in range(nboot):
        pick = RNG.integers(0, len(teams), len(teams))
        ix = np.concatenate([rows_by[k] for k in pick])
        s_, b_ = fit(ix)
        bs.append(s_)
        bb.append(b_)
    return b, np.nanstd(bs), bias, np.nanstd(bb), len(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    uni, form, boxes = load()
    tgt = ros_targets(boxes)
    df = form.merge(tgt, on=["season", "game_id", "team_id"], how="inner")
    e3 = pd.read_parquet(ROOT / "data/processed/team_rate_features_E3_v4.parquet")
    # E3 table carries one row per fold; take the fold whose test season matches
    e3 = e3[((e3["fold"] == "F1") & (e3["season"] == 2024)) | ((e3["fold"] == "F2") & (e3["season"] == 2025))]
    e3 = e3[["season", "game_id", "team_id", "tov_off_c", "tov_def_c", "ftr_off_c", "ftr_def_c"]]
    e3 = e3.rename(columns={c: f"E3_{c}" for c in ["tov_off_c", "tov_def_c", "ftr_off_c", "ftr_def_c"]})
    df = df.merge(e3, on=["season", "game_id", "team_id"], how="left")
    rows = []
    for season, fold in TEST.items():
        d = df[df["season"] == season]
        for name in RATES:
            for side in ("off", "def"):
                served = d[f"{side}_{name}_c"].to_numpy(float)
                e3c = 100 * d[f"E3_{name}_{side}_c"].to_numpy(float)
                for lo, hi, lab in BUCKETS + [(0, 999, "all")]:
                    m = (d["days"] >= lo) & (d["days"] <= hi)
                    for tgt_name in ("ros", "nxt"):
                        y = d[f"{side}_{name}_{tgt_name}"].to_numpy(float)
                        for arm, x in (("served_E0", served), ("E3_ref", e3c)):
                            sl, sl_se, bias, bias_se, n = slope_stats(x[m.to_numpy()], y[m.to_numpy()],
                                                                     d["team_id"].to_numpy()[m.to_numpy()])
                            rows.append(dict(fold=fold, season=season, rate=name, side=side, bucket=lab,
                                             target=tgt_name, arm=arm, n=n,
                                             mean_x=np.nanmean(x[m.to_numpy()]),
                                             mean_y=np.nanmean(y[m.to_numpy()]),
                                             bias=bias, bias_se=bias_se, slope=sl, slope_se=sl_se,
                                             sd_x=np.nanstd(x[m.to_numpy()]), n_prior_med=float(
                                                 d.loc[m, "n_prior_off"].median())))
                # by n_prior band inside d0-14 (ROS target, served only)
                for lo, hi, lab in [(0, 0, "n0"), (1, 3, "n1-3"), (4, 6, "n4-6"), (7, 999, "n7+")]:
                    m = (d["days"] <= 14) & (d["n_prior_off"] >= lo) & (d["n_prior_off"] <= hi)
                    y = d[f"{side}_{name}_ros"].to_numpy(float)
                    sl, sl_se, bias, bias_se, n = slope_stats(served[m.to_numpy()], y[m.to_numpy()],
                                                              d["team_id"].to_numpy()[m.to_numpy()])
                    rows.append(dict(fold=fold, season=season, rate=name, side=side, bucket=f"d0-14|{lab}",
                                     target="ros", arm="served_E0", n=n, mean_x=np.nanmean(served[m.to_numpy()]),
                                     mean_y=np.nanmean(y[m.to_numpy()]), bias=bias, bias_se=bias_se,
                                     slope=sl, slope_se=sl_se, sd_x=np.nanstd(served[m.to_numpy()]),
                                     n_prior_med=float(lo)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "phase1_v1.csv", index=False)

    # League level context: as-of league rate subtracted early vs the realised level of the bucket's games
    lv = []
    first = boxes.groupby("season")["game_date"].transform("min")
    bx = boxes.assign(days=(boxes["game_date"] - first).dt.days)
    for season, fold in TEST.items():
        b = bx[bx["season"] == season]
        prev = bx[bx["season"] == season - 1]
        for name, (num, den) in RATES.items():
            rec = dict(fold=fold, rate=name, prev_season=100 * prev[num].sum() / prev[den].sum(),
                       season=100 * b[num].sum() / b[den].sum())
            for lo, hi, lab in BUCKETS:
                bb = b[(b["days"] >= lo) & (b["days"] <= hi)]
                rec[lab] = 100 * bb[num].sum() / bb[den].sum()
            lv.append(rec)
    lv = pd.DataFrame(lv)
    lv.to_csv(OUT / "phase1_league_levels_v1.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = res[(res["target"] == "ros")].copy()
    print(show[["fold", "rate", "side", "bucket", "arm", "n", "mean_x", "mean_y", "bias", "bias_se",
                "slope", "slope_se", "sd_x"]].round(3).to_string(index=False))
    print(res[(res["target"] == "nxt") & (res["arm"] == "served_E0")][
        ["fold", "rate", "side", "bucket", "n", "bias", "bias_se", "slope", "slope_se"]].round(3).to_string(index=False))
    print(lv.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
