#!/usr/bin/env python
"""
diag_truth_finals_audit_v1.py -- second-source audit of game finals (Lane H, 2026-09-30).

Sources compared per game (hoopR game_id == CBBD sourceId):
  A  hoopR schedule            home_score / away_score / status_type_name
  B  hoopR team box            team_score (per side) and box-derived points 2*FGM+3PM+FTM
  C  CBBD games                homePoints / awayPoints / status / period-points lists
Writes data/processed/truth/truth_audit_unplayed_v1.parquet (one row per flagged game)
and truth_audit_unplayed_v1_summary.json.  2025-26 (season 2026, SEALED) is counted by
status / 0-0 / missing only; no score comparison, nothing model-related.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HO = ROOT / "data/raw/hoopr"
CB = ROOT / "data/raw/cbbd"
OUT = ROOT / "data/processed/truth"
SEASONS = [2022, 2023, 2024, 2025, 2026]
SEALED = 2026


def load_season(s: int) -> pd.DataFrame:
    h = pd.read_parquet(HO / f"schedules/mbb_schedule_{s}.parquet")
    h = h[["game_id", "season", "game_date", "status_type_name", "status_type_completed",
           "home_id", "away_id", "home_score", "away_score"]].rename(
        columns={"status_type_name": "h_status", "home_score": "h_home", "away_score": "h_away",
                 "home_id": "h_home_id", "away_id": "h_away_id"})
    c = pd.read_parquet(CB / f"games_{s}.parquet")
    c = c.rename(columns={"sourceId": "sid"})
    c["game_id"] = pd.to_numeric(c["sid"], errors="coerce")
    c = c[["game_id", "id", "status", "homePoints", "awayPoints", "homePeriodPoints", "awayPeriodPoints",
           "homeTeamId", "awayTeamId"]].rename(columns={"id": "cbbd_game_id", "status": "c_status",
                                                        "homePoints": "c_home", "awayPoints": "c_away"})
    df = h.merge(c, on="game_id", how="outer", indicator=True)
    tb = pd.read_parquet(HO / f"team_box/team_box_{s}.parquet")
    tb["box_pts"] = 2 * tb["field_goals_made"] + tb["three_point_field_goals_made"] + tb["free_throws_made"]
    home = tb[tb["team_home_away"] == "home"].drop_duplicates("game_id")[["game_id", "team_score", "box_pts"]]
    away = tb[tb["team_home_away"] == "away"].drop_duplicates("game_id")[["game_id", "team_score", "box_pts"]]
    home = home.rename(columns={"team_score": "b_home", "box_pts": "bx_home"})
    away = away.rename(columns={"team_score": "b_away", "box_pts": "bx_away"})
    df = df.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
    ntb = tb.groupby("game_id").size().rename("n_box_rows")
    df = df.merge(ntb, on="game_id", how="left")
    df["n_box_rows"] = df["n_box_rows"].fillna(0).astype(int)
    df["season"] = s
    return df


def n_periods_cbbd(x):
    try:
        return len(x)
    except Exception:
        return np.nan


def main() -> int:
    uni = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
    uni = uni[["game_id", "is_d1_game", "pbp_truncated", "n_periods", "has_pbp", "pbp_complete"]]
    fin = pd.read_parquet(ROOT / "data/processed/truth/game_finals_v2.parquet")[
        ["game_id", "finals_source", "finals_third_source_checked", "finals_resolution_note"]]
    rows, summ = [], {}
    for s in SEASONS:
        df = load_season(s)
        df = df.merge(uni, on="game_id", how="left").merge(fin, on="game_id", how="left")
        sealed = s == SEALED
        S = {"n_schedule": int(df["h_status"].notna().sum()), "n_cbbd": int(df["c_status"].notna().sum()),
             "only_hoopr": int((df["_merge"] == "left_only").sum()),
             "only_cbbd": int((df["_merge"] == "right_only").sum())}
        df["h_final"] = df["h_status"].eq("STATUS_FINAL")
        df["c_final"] = df["c_status"].eq("final")
        df["h_zero"] = df["h_home"].fillna(0).eq(0) & df["h_away"].fillna(0).eq(0)
        df["c_zero"] = df["c_home"].fillna(0).eq(0) & df["c_away"].fillna(0).eq(0)
        df["h_missing"] = df["h_home"].isna() | df["h_away"].isna()
        df["c_missing"] = df["c_home"].isna() | df["c_away"].isna()
        S["hoopr_status"] = df["h_status"].value_counts(dropna=False).astype(int).to_dict()
        S["cbbd_status"] = df["c_status"].value_counts(dropna=False).astype(int).to_dict()
        S["hoopr_0_0"] = int((df["h_zero"] & df["h_status"].notna()).sum())
        S["hoopr_0_0_final_status"] = int((df["h_zero"] & df["h_final"]).sum())
        S["cbbd_0_0"] = int((df["c_zero"] & df["c_status"].notna()).sum())
        S["cbbd_0_0_final_status"] = int((df["c_zero"] & df["c_final"]).sum())
        S["hoopr_missing_score"] = int((df["h_missing"] & df["h_status"].notna()).sum())
        S["cbbd_missing_score"] = int((df["c_missing"] & df["c_status"].notna()).sum())
        S["hoopr_final_no_box"] = int((df["h_final"] & (df["n_box_rows"] < 2)).sum())
        S["in_universe"] = int(df["is_d1_game"].notna().sum())
        S["universe_d1"] = int(df["is_d1_game"].fillna(False).astype(bool).sum())
        d1 = df["is_d1_game"].fillna(False).astype(bool) & ~df["pbp_truncated"].fillna(False).astype(bool)
        S["graded_universe_rule"] = int(d1.sum())
        if not sealed:
            # score comparison, both-sides present and non-zero-on-a-side excluded from "agree" only when 0-0
            both = df["h_final"] & df["c_final"] & ~df["h_missing"] & ~df["c_missing"]
            dis_home = both & (df["h_home"] != df["c_home"])
            dis_away = both & (df["h_away"] != df["c_away"])
            dis = dis_home | dis_away
            flip = both & (df["h_home"] == df["c_away"]) & (df["h_away"] == df["c_home"]) & dis
            S["both_final_scored"] = int(both.sum())
            S["disagree_any"] = int(dis.sum())
            S["agree_rate_both_final"] = float(1 - dis.sum() / both.sum())
            S["disagree_flip"] = int(flip.sum())
            S["disagree_cbbd_zero_side"] = int((dis & ((df["c_home"] == 0) | (df["c_away"] == 0))).sum())
            S["disagree_hoopr_zero_side"] = int((dis & ((df["h_home"] == 0) | (df["h_away"] == 0))).sum())
            diff = np.maximum((df["h_home"] - df["c_home"]).abs(), (df["h_away"] - df["c_away"]).abs())
            S["disagree_abs_diff_le3_not_flip"] = int((dis & ~flip & (diff <= 3)).sum())
            S["disagree_abs_diff_gt3_not_flip"] = int((dis & ~flip & (diff > 3)).sum())
            # box vs schedule
            hb = df["h_final"] & df["b_home"].notna() & df["b_away"].notna()
            bd = hb & ((df["b_home"] != df["h_home"]) | (df["b_away"] != df["h_away"]))
            S["box_vs_schedule_n"] = int(hb.sum())
            S["box_vs_schedule_disagree"] = int(bd.sum())
            bxd = hb & df["bx_home"].notna() & ((df["bx_home"] != df["h_home"]) | (df["bx_away"] != df["h_away"]))
            S["box_derived_pts_vs_schedule_disagree"] = int(bxd.sum())
            S["box_derived_pts_vs_cbbd_when_hoopr_cbbd_disagree"] = int(
                (dis & df["bx_home"].notna()).sum())
            # overtime accumulation
            df["c_nper"] = df["homePeriodPoints"].map(n_periods_cbbd)
            df["c_sum_home"] = df["homePeriodPoints"].map(lambda x: float(np.sum(x)) if x is not None and len(x) else np.nan)
            df["c_sum_away"] = df["awayPeriodPoints"].map(lambda x: float(np.sum(x)) if x is not None and len(x) else np.nan)
            cf = df["c_final"] & df["c_sum_home"].notna()
            S["cbbd_periodsum_ne_points"] = int((cf & ((df["c_sum_home"] != df["c_home"]) | (df["c_sum_away"] != df["c_away"]))).sum())
            S["cbbd_final_no_periodpoints"] = int((df["c_final"] & df["c_sum_home"].isna()).sum())
            u_np = df["n_periods"]
            ot_cmp = df["c_final"] & u_np.notna() & df["c_nper"].notna()
            S["n_periods_universe_vs_cbbd_disagree"] = int((ot_cmp & (u_np != df["c_nper"])).sum())
            S["went_ot_universe"] = int((u_np > 2).sum())
            S["went_ot_cbbd"] = int((df["c_nper"] > 2).sum())
            # OT games score agreement
            ot = both & ((u_np > 2) | (df["c_nper"] > 2))
            S["ot_games_both_final"] = int(ot.sum())
            S["ot_games_score_disagree"] = int((ot & dis).sum())
        # flagged rows
        flag = (~df["h_final"] | ~df["c_final"] | df["h_zero"] | df["c_zero"] | df["h_missing"] | df["c_missing"])
        if not sealed:
            flag = flag | dis | (df["c_sum_home"].notna() & df["c_final"] & ((df["c_sum_home"] != df["c_home"]) | (df["c_sum_away"] != df["c_away"])))
        f = df[flag].copy()
        f["cls"] = ""
        def add(mask, name):
            f.loc[mask.reindex(f.index, fill_value=False), "cls"] += name + ";"
        add(f["h_zero"] & f["h_status"].notna(), "hoopr_0_0")
        add(f["c_zero"] & f["c_status"].notna(), "cbbd_0_0")
        add(~f["h_final"] & f["h_status"].notna(), "hoopr_" + "nonfinal")
        add(~f["c_final"] & f["c_status"].notna(), "cbbd_nonfinal")
        add(f["h_status"].isna(), "not_in_hoopr")
        add(f["c_status"].isna(), "not_in_cbbd")
        add(f["h_missing"] & f["h_status"].notna(), "hoopr_missing")
        add(f["c_missing"] & f["c_status"].notna(), "cbbd_missing")
        if not sealed:
            add(dis.reindex(f.index, fill_value=False), "score_disagree")
            add(flip.reindex(f.index, fill_value=False), "flipped_sides")
            add((f["c_final"] & f["c_sum_home"].notna() & ((f["c_sum_home"] != f["c_home"]) | (f["c_sum_away"] != f["c_away"]))), "cbbd_periodsum_ne_points")
        f["sealed_season"] = sealed
        keep = ["season", "game_id", "cbbd_game_id", "game_date", "cls", "h_status", "c_status", "h_home", "h_away",
                "c_home", "c_away", "b_home", "b_away", "n_box_rows", "is_d1_game", "pbp_truncated", "n_periods",
                "finals_source", "finals_third_source_checked", "sealed_season"]
        if sealed:
            keep = [k for k in keep if k not in ("h_home", "h_away", "c_home", "c_away", "b_home", "b_away", "n_periods")]
        rows.append(f[keep])
        cls_counts = f["cls"].str.rstrip(";").str.split(";").explode().value_counts().astype(int).to_dict()
        S["flag_class_counts"] = cls_counts
        S["n_flagged"] = int(len(f))
        # flagged inside graded universe
        gd = f["is_d1_game"].fillna(False).astype(bool) & ~f["pbp_truncated"].fillna(False).astype(bool)
        S["flagged_in_graded_universe"] = int(gd.sum())
        S["graded_universe_0_0_scored_rows"] = int(((df["h_zero"]) & d1 & df["h_status"].notna()).sum())
        summ[str(s)] = S
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / "truth_audit_unplayed_v1.parquet", index=False)
    (OUT / "truth_audit_unplayed_v1_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
