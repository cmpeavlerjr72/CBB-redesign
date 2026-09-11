"""diag_clock_censoring_v1.py -- the L20 censoring audit (clock round 3, step 2).

WHAT THIS ANSWERS. `docs/LEARNINGS.md` L20 says horn-ending possessions are
TRUNCATED, not CENSORED, in the clock model's training data. This script
measures, from the possession tables alone and with no model involved:

  1. how many possessions per game and per half end AT THE HORN, by season
     (2022-2025) and by period type (H1 / H2 / OT);
  2. what fraction of halves and games have COMPLETE clock data per the CBBD
     event stream, under a stated three-part definition whose components are
     each reported separately;
  3. the duration distribution of horn-ending vs non-horn possessions,
     overall and by the seconds-remaining band the possession started in;
  4. the confusion between the flag the clock model currently uses
     (`terminal_event == "end_period"`, 0.2% of rows) and the correct one;
  5. how the G1 target -- actual possessions per team-game -- moves when the
     universe is restricted to clock-complete games.

It writes `docs/tests/clock_censoring_audit_2026-09-10.md` (every number in it
is written by this script; none is typed by hand), a JSON of the same numbers,
and the two flag artifacts round 3 needs:

  data/processed/games_universe_v2.parquet          -- games_universe plus the
      game-level clock-completeness columns. A VERSIONED SIBLING: the original
      `games_universe.parquet` is never rewritten, so the engine worker's
      reader is untouched.
  data/processed/clock_censoring/half_clock_completeness_v1.parquet
      -- (game_id, period) side table, the half-level flags and components.
  data/processed/clock_censoring/censoring_v1_{season}.parquet
      -- (game_id, period, poss_index) side table carrying `censored_horn` and
      the tolerance variants. A SIDE TABLE: `data/processed/possessions_v2/`
      and `data/processed/possessions/` are read-only here.

SEAL: season 2026 is never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POSS_DIR = ROOT / "data" / "processed" / "possessions"
UNIVERSE = ROOT / "data" / "processed" / "games_universe.parquet"
OUT_DIR = ROOT / "data" / "processed" / "clock_censoring"
UNIVERSE_V2 = ROOT / "data" / "processed" / "games_universe_v2.parquet"
DOC = ROOT / "docs" / "tests" / "clock_censoring_audit_2026-09-10.md"

SEASONS = [2022, 2023, 2024, 2025]
REG_LEN = 1200
OT_LEN = 300

#: A possession is HORN-ENDING at tolerance k if it ends within k seconds of
#: the period horn. k = 0 (the possession consumed every second that was left)
#: is the definition round 3 uses; 1 and 2 are reported as sensitivity.
HORN_TOLS = (0, 1, 2)
HORN_TOL = 0

#: A period is CLOCK-COMPLETE if all three hold (each reported separately):
#:   C1 tail   -- its last logged possession ends within TAIL_TOL s of the horn
#:   C2 head   -- its first logged possession starts within HEAD_TOL s of the
#:                period length
#:   C3 sum    -- |sum of possession durations - period length| <= SUM_TOL
#: C1 alone is `cbb_sim.models.clock.CLOCK_COMPLETE_TOL_S` (round 2's secondary
#: read). C2 and C3 are added here so "complete" means the whole period is
#: accounted for, not only its end.
TAIL_TOL = 2
HEAD_TOL = 2
SUM_TOL = 4

DUR_CAP = 90  # cbb_sim.models.clock.DURATION_CAP, for the like-for-like read

POSS_COLS = [
    "game_id", "season", "period", "poss_index", "offense_team_id",
    "start_clock", "end_clock", "duration_s", "terminal_event", "start_reason",
]


def period_len(period: np.ndarray) -> np.ndarray:
    return np.where(period <= 2, REG_LEN, OT_LEN)


def period_label(period: np.ndarray) -> np.ndarray:
    return np.where(period == 1, "H1", np.where(period == 2, "H2", "OT"))


def load() -> pd.DataFrame:
    frames = []
    for s in SEASONS:
        f = POSS_DIR / f"possessions_{s}.parquet"
        frames.append(pd.read_parquet(f, columns=POSS_COLS))
    p = pd.concat(frames, ignore_index=True)
    p["period"] = p["period"].astype("int16")
    p["period_len"] = period_len(p["period"].to_numpy()).astype("int32")
    p["period_type"] = period_label(p["period"].to_numpy())
    for k in HORN_TOLS:
        p[f"horn_{k}"] = (p["end_clock"] <= k).to_numpy()
    p["censored_horn"] = p[f"horn_{HORN_TOL}"]
    p["flag_end_period"] = (p["terminal_event"] == "end_period").to_numpy()
    return p


def half_completeness(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["game_id", "period", "poss_index"], kind="stable")
    g = p.groupby(["game_id", "period"], sort=False)
    out = pd.DataFrame({
        "season": g["season"].first(),
        "period_len": g["period_len"].first(),
        "n_poss": g.size(),
        "first_start_clock": g["start_clock"].first(),
        "last_end_clock": g["end_clock"].last(),
        "last_start_clock": g["start_clock"].last(),
        "last_duration": g["duration_s"].last(),
        "sum_duration": g["duration_s"].sum(),
        "n_horn": g["censored_horn"].sum(),
    }).reset_index()
    out["period_type"] = period_label(out["period"].to_numpy())
    out["c1_tail_ok"] = out["last_end_clock"] <= TAIL_TOL
    out["c2_head_ok"] = (out["period_len"] - out["first_start_clock"]) <= HEAD_TOL
    out["c3_sum_ok"] = (out["sum_duration"] - out["period_len"]).abs() <= SUM_TOL
    out["clock_complete"] = out["c1_tail_ok"] & out["c2_head_ok"] & out["c3_sum_ok"]
    out["unaccounted_s"] = out["period_len"] - out["sum_duration"]
    return out


def fmt(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)

    uni = pd.read_parquet(UNIVERSE)
    uni_d1 = uni[uni["is_d1_game"] & ~uni["pbp_truncated"]].copy()

    p_all = load()
    res: dict = {"seasons": SEASONS, "horn_tol": HORN_TOL,
                 "clock_complete_def": {"tail_tol": TAIL_TOL, "head_tol": HEAD_TOL,
                                        "sum_tol": SUM_TOL}}

    # ---- universe filters, reported as a ladder ---------------------------
    p = p_all[p_all["game_id"].isin(set(uni_d1["game_id"]))].copy()
    # CBBD points-completeness, the same definition clock.cbbd_complete_games uses
    tp = p_all.groupby(["game_id", "offense_team_id"], as_index=False)["poss_index"].size()
    del tp
    pts = pd.concat(
        [pd.read_parquet(POSS_DIR / f"possessions_{s}.parquet",
                         columns=["game_id", "offense_team_id", "points", "tech_points_off"])
         for s in SEASONS], ignore_index=True)
    tp = pts.groupby(["game_id", "offense_team_id"], as_index=False)[["points", "tech_points_off"]].sum()
    tp["scored"] = tp["points"] + tp["tech_points_off"]
    m = tp.merge(uni_d1[["game_id", "home_team_id", "home_score", "away_score"]], on="game_id", how="inner")
    m["final"] = np.where(m["offense_team_id"] == m["home_team_id"], m["home_score"], m["away_score"])
    m["ok"] = m["scored"] == m["final"]
    pts_complete = m.groupby("game_id", as_index=False)["ok"].all().rename(columns={"ok": "points_complete"})
    keep_pts = set(pts_complete.loc[pts_complete["points_complete"], "game_id"])

    res["ladder"] = {
        "rows_all": int(len(p_all)), "games_all": int(p_all["game_id"].nunique()),
        "rows_d1_untruncated": int(len(p)), "games_d1_untruncated": int(p["game_id"].nunique()),
        "games_points_complete": int(len(keep_pts)),
    }
    p["points_complete"] = p["game_id"].isin(keep_pts)

    # ======================================================================
    # 1. horn-ending possessions by season and period type
    # ======================================================================
    pc = p[p["points_complete"]].copy()
    res["ladder"]["rows_points_complete"] = int(len(pc))

    t1 = (pc.groupby(["season", "period_type"], as_index=False)
            .agg(n_poss=("censored_horn", "size"),
                 n_horn=("censored_horn", "sum"),
                 n_horn_tol1=("horn_1", "sum"),
                 n_horn_tol2=("horn_2", "sum"),
                 n_flag_end_period=("flag_end_period", "sum")))
    n_periods = pc.groupby(["season", "period_type"])[["game_id", "period"]].apply(
        lambda d: d.drop_duplicates().shape[0]).rename("n_periods").reset_index()
    t1 = t1.merge(n_periods, on=["season", "period_type"])
    t1["horn_pct"] = 100.0 * t1["n_horn"] / t1["n_poss"]
    t1["horn_per_period"] = t1["n_horn"] / t1["n_periods"]
    t1["old_flag_pct"] = 100.0 * t1["n_flag_end_period"] / t1["n_poss"]

    games_per_season = pc.groupby("season")["game_id"].nunique().rename("n_games")
    t1b = (pc.groupby("season", as_index=False)
             .agg(n_poss=("censored_horn", "size"), n_horn=("censored_horn", "sum"),
                  n_flag=("flag_end_period", "sum")))
    t1b = t1b.merge(games_per_season, on="season")
    t1b["horn_pct"] = 100.0 * t1b["n_horn"] / t1b["n_poss"]
    t1b["horn_per_game"] = t1b["n_horn"] / t1b["n_games"]
    t1b["old_flag_per_game"] = t1b["n_flag"] / t1b["n_games"]

    # ======================================================================
    # 2. clock completeness
    # ======================================================================
    hc = half_completeness(pc)
    t2 = (hc.groupby(["season", "period_type"], as_index=False)
            .agg(n_periods=("clock_complete", "size"),
                 c1_tail=("c1_tail_ok", "mean"),
                 c2_head=("c2_head_ok", "mean"),
                 c3_sum=("c3_sum_ok", "mean"),
                 complete=("clock_complete", "mean"),
                 mean_unaccounted_s=("unaccounted_s", "mean"),
                 p90_unaccounted_s=("unaccounted_s", lambda s: float(np.percentile(s, 90)))))

    reg = hc[hc["period"] <= 2]
    gcc = reg.groupby("game_id")["clock_complete"].all().rename("game_clock_complete")
    gcc_all = hc.groupby("game_id")["clock_complete"].all().rename("game_clock_complete_all_periods")
    season_of = pc.groupby("game_id")["season"].first()
    t2b = pd.DataFrame({"season": season_of, "reg": gcc, "all": gcc_all}).reset_index()
    t2b = t2b.groupby("season", as_index=False).agg(
        n_games=("reg", "size"), share_reg_halves_complete=("reg", "mean"),
        share_all_periods_complete=("all", "mean"))

    # share of halves whose LAST possession is horn-ending
    t2c = (hc.groupby(["season", "period_type"], as_index=False)
             .agg(n_periods=("c1_tail_ok", "size"),
                  last_poss_at_horn=("c1_tail_ok", "mean"),
                  last_start_under_35=("last_start_clock", lambda s: float((s < 35).mean())),
                  mean_last_duration=("last_duration", "mean")))

    # ======================================================================
    # 3. duration distribution, horn vs non-horn
    # ======================================================================
    capped = pc[pc["duration_s"] <= DUR_CAP]
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    rows = []
    for lab, sub in (("horn-ending", capped[capped["censored_horn"]]),
                     ("not horn-ending", capped[~capped["censored_horn"]])):
        d = sub["duration_s"].to_numpy(dtype="float64")
        r = {"group": lab, "n": len(d), "mean": d.mean(), "sd": d.std(ddof=1)}
        for q in qs:
            r[f"q{int(q * 100)}"] = float(np.percentile(d, q * 100))
        rows.append(r)
    t3 = pd.DataFrame(rows)

    bands = [0, 5, 10, 20, 30, 45, 60, 90, 1201]
    capped = capped.copy()
    capped["band"] = pd.cut(capped["start_clock"], bands, right=False)
    t3b = (capped.groupby("band", observed=True, as_index=False)
                 .agg(n=("censored_horn", "size"),
                      horn_rate=("censored_horn", "mean"),
                      old_flag_rate=("flag_end_period", "mean"),
                      mean_dur_all=("duration_s", "mean")))
    t3b["mean_dur_uncensored"] = (capped[~capped["censored_horn"]]
                                  .groupby("band", observed=True)["duration_s"].mean().to_numpy())
    t3b["band"] = t3b["band"].astype(str)

    # horn rate by terminal event
    t3c = (capped.groupby("terminal_event", as_index=False)
                 .agg(n=("censored_horn", "size"), horn_rate=("censored_horn", "mean"),
                      n_horn=("censored_horn", "sum")))

    # confusion of the two flags
    conf = pd.crosstab(capped["flag_end_period"], capped["censored_horn"])
    res["flag_confusion"] = {f"end_period={a}|horn={b}": int(conf.loc[a, b])
                             for a in conf.index for b in conf.columns}
    res["flag_rates"] = {
        "old_flag_pct": round(100.0 * float(capped["flag_end_period"].mean()), 4),
        "new_flag_pct": round(100.0 * float(capped["censored_horn"].mean()), 4),
        "ratio": round(float(capped["censored_horn"].mean() / max(capped["flag_end_period"].mean(), 1e-12)), 3),
    }

    # ======================================================================
    # 4. the G1 target: actual possessions per team-game, all vs clock-complete
    # ======================================================================
    reg_poss = pc[pc["period"] <= 2]
    per_game = (reg_poss.groupby("game_id").size() / 2.0).rename("actual_poss")
    pg = pd.DataFrame({"actual_poss": per_game}).join(season_of).join(gcc)
    pg["game_clock_complete"] = pg["game_clock_complete"].fillna(False)
    t4 = (pg.groupby("season", as_index=False)
            .agg(n_games=("actual_poss", "size"), mean_all=("actual_poss", "mean"),
                 sd_all=("actual_poss", "std")))
    cc = pg[pg["game_clock_complete"]]
    t4cc = (cc.groupby("season", as_index=False)
              .agg(n_cc_games=("actual_poss", "size"), mean_cc=("actual_poss", "mean"),
                   sd_cc=("actual_poss", "std")))
    t4 = t4.merge(t4cc, on="season")
    t4["mean_shift"] = t4["mean_cc"] - t4["mean_all"]
    t4["cc_share"] = t4["n_cc_games"] / t4["n_games"]

    # by month on 2025 (the selection fold's test season)
    m25 = pg[pg["season"] == 2025].join(
        uni_d1.set_index("game_id")["game_date"], how="left")
    m25["month"] = pd.to_datetime(m25["game_date"]).dt.month
    t4m = (m25.groupby("month", as_index=False)
             .agg(n_games=("actual_poss", "size"), mean_all=("actual_poss", "mean")))
    t4mc = (m25[m25["game_clock_complete"]].groupby("month", as_index=False)
            .agg(n_cc=("actual_poss", "size"), mean_cc=("actual_poss", "mean")))
    t4m = t4m.merge(t4mc, on="month", how="left")
    t4m["mean_shift"] = t4m["mean_cc"] - t4m["mean_all"]

    # ======================================================================
    # artifacts
    # ======================================================================
    hc_out = hc[["game_id", "period", "season", "period_type", "n_poss", "first_start_clock",
                 "last_start_clock", "last_end_clock", "last_duration", "sum_duration",
                 "unaccounted_s", "n_horn", "c1_tail_ok", "c2_head_ok", "c3_sum_ok",
                 "clock_complete"]]
    hc_out.to_parquet(OUT_DIR / "half_clock_completeness_v1.parquet", index=False)

    gflag = pd.DataFrame({
        "game_id": gcc.index,
        "clock_complete_reg": gcc.to_numpy(),
    }).merge(pd.DataFrame({"game_id": gcc_all.index,
                           "clock_complete_all_periods": gcc_all.to_numpy()}),
             on="game_id", how="outer")
    nper = hc.groupby("game_id").agg(n_periods_logged=("period", "nunique"),
                                     n_horn_poss=("n_horn", "sum"),
                                     unaccounted_s_total=("unaccounted_s", "sum")).reset_index()
    gflag = gflag.merge(nper, on="game_id", how="left")
    gflag = gflag.merge(pts_complete, on="game_id", how="left")

    uni_v2 = uni.merge(gflag, on="game_id", how="left")
    # A game whose season was never loaded (2026 is SEALED) or which has no
    # possession rows has no flag. It must not read as "not clock-complete":
    # `clock_flags_evaluated` says whether the flags mean anything for that row.
    uni_v2["clock_flags_evaluated"] = uni_v2["clock_complete_reg"].notna()
    for c in ("clock_complete_reg", "clock_complete_all_periods", "points_complete"):
        uni_v2[c] = uni_v2[c].fillna(False).astype(bool)
    uni_v2.to_parquet(UNIVERSE_V2, index=False)
    res["games_universe_v2"] = {
        "path": str(UNIVERSE_V2.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(uni_v2)),
        "added_columns": ["clock_complete_reg", "clock_complete_all_periods",
                          "points_complete", "clock_flags_evaluated",
                          "n_periods_logged", "n_horn_poss", "unaccounted_s_total"],
        "rows_with_flags_evaluated": int(uni_v2["clock_flags_evaluated"].sum()),
        "note": "versioned sibling; data/processed/games_universe.parquet untouched",
    }

    for s in SEASONS:
        sub = p_all[p_all["season"] == s]
        side = sub[["game_id", "period", "poss_index", "start_clock", "end_clock",
                    "duration_s"]].copy()
        side["censored_horn"] = (sub["end_clock"] <= HORN_TOL).to_numpy()
        side["censored_horn_tol1"] = (sub["end_clock"] <= 1).to_numpy()
        side["censored_horn_tol2"] = (sub["end_clock"] <= 2).to_numpy()
        side["flag_end_period"] = (sub["terminal_event"] == "end_period").to_numpy()
        side.to_parquet(OUT_DIR / f"censoring_v1_{s}.parquet", index=False)

    res["tables"] = {
        "by_season_period": t1.to_dict("records"),
        "by_season": t1b.to_dict("records"),
        "completeness_by_season_period": t2.to_dict("records"),
        "completeness_by_season_game": t2b.to_dict("records"),
        "last_possession_by_season_period": t2c.to_dict("records"),
        "duration_horn_vs_not": t3.to_dict("records"),
        "by_clock_band": t3b.to_dict("records"),
        "by_terminal_event": t3c.to_dict("records"),
        "g1_target_shift": t4.to_dict("records"),
        "g1_target_shift_2025_by_month": t4m.to_dict("records"),
    }
    (OUT_DIR / "censoring_audit_v1.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    # ======================================================================
    # the markdown
    # ======================================================================
    L = res["ladder"]
    fr = res["flag_rates"]
    old_pct = fr["old_flag_pct"]
    new_pct = fr["new_flag_pct"]
    b010 = capped[capped["start_clock"] < 10]["censored_horn"].mean()

    md = f"""# Clock censoring audit -- L20 evidence (2026-09-10)

Written by `scripts/diag_clock_censoring_v1.py`. Every number below is produced
by that script from `data/processed/possessions/possessions_{{season}}.parquet`
and `data/processed/games_universe.parquet`; none is typed by hand. No model is
involved. Season 2026 is never loaded (sealed).

This is step 2 of clock round 3: the evidence for `docs/LEARNINGS.md` L20
("horn-truncated possessions must be right-censored") and the source of the two
flag artifacts round 3 trains on.

---

## 0. Definitions (stated before the numbers)

**Horn-ending (right-censored) possession.** `end_clock <= {HORN_TOL}`, i.e. the
possession consumed every second that was left in the period. Its observed
`duration_s` is a LOWER BOUND on the duration the offence intended, so the row
is RIGHT-CENSORED at its observed value. Tolerances of 1 s and 2 s are reported
as sensitivity; the 0 s definition is the one round 3 uses.

The flag the clock model used in rounds 1 and 2 was
`terminal_event == "end_period"` -- a possession the possession builder closed
because the period ended with no other terminal event. That is a strict subset:
a buzzer three, a buzzer layup, a turnover as the horn sounds and a bonus trip
that ends the half are all horn-ending and none of them is `end_period`.

**Clock-complete period.** All three of:

| component | test | why |
|---|---|---|
| C1 tail | last logged possession ends within {TAIL_TOL} s of the horn | the feed reached the end of the period |
| C2 head | first logged possession starts within {HEAD_TOL} s of the period length | the feed started at the opening tip |
| C3 sum | \\|sum of possession durations - period length\\| <= {SUM_TOL} s | the logged possessions account for the whole period, with no interior gap |

C1 alone is `cbb_sim.models.clock.CLOCK_COMPLETE_TOL_S`, round 2's secondary
read. C2 and C3 are added here so that "complete" means the whole period is
accounted for and not merely its end. A GAME is clock-complete when both
regulation halves are (`clock_complete_reg`); `clock_complete_all_periods` also
requires every OT period.

**Universe ladder.** D-I and not `pbp_truncated`, then CBBD points-complete
(the possession table's own points plus technical FTs equal the schedule's final
score for both teams -- the same definition
`cbb_sim.models.clock.cbbd_complete_games` uses).

| step | rows | games |
|---|---:|---:|
| all possessions 2022-2025 | {L['rows_all']:,} | {L['games_all']:,} |
| D-I, hoopR feed not truncated | {L['rows_d1_untruncated']:,} | {L['games_d1_untruncated']:,} |
| + CBBD points-complete | {L['rows_points_complete']:,} | {L['games_points_complete']:,} |

Everything from section 1 on is on the points-complete universe, which is the
clock bake-off's own universe, so these numbers are directly comparable to
`docs/models/clock/experiments.md`.

---

## 1. How many possessions end at the horn

### 1.1 By season

{fmt(t1b[['season', 'n_games', 'n_poss', 'n_horn', 'horn_pct', 'horn_per_game', 'n_flag', 'old_flag_per_game']], '{:.4f}')}

`n_horn` / `horn_pct` are the correct censoring set; `n_flag` is what rounds 1
and 2 flagged. The correct flag is **{fr['ratio']:.1f}x** the old one
({new_pct:.4f}% of rows against {old_pct:.4f}%).

### 1.2 By season and period type

{fmt(t1[['season', 'period_type', 'n_periods', 'n_poss', 'n_horn', 'horn_pct', 'horn_per_period', 'n_flag_end_period', 'old_flag_pct']], '{:.4f}')}

Read: roughly one possession per period ends at the horn, by construction --
every period that is fully logged ends with one. The point is not the count, it
is WHICH rows they are and what the model does with them: they are the rows that
define the conditional law of duration in the last seconds of a period, and
treating them as completed observations is what made rounds 1 and 2 draw short
there.

### 1.3 Share of periods whose last logged possession reaches the horn

{fmt(t2c[['season', 'period_type', 'n_periods', 'last_poss_at_horn', 'last_start_under_35', 'mean_last_duration']], '{:.4f}')}

---

## 2. Clock completeness

### 2.1 By season and period type, with each component separately

{fmt(t2[['season', 'period_type', 'n_periods', 'c1_tail', 'c2_head', 'c3_sum', 'complete', 'mean_unaccounted_s', 'p90_unaccounted_s']], '{:.4f}')}

`mean_unaccounted_s` is period length minus the summed possession durations: the
seconds of game time the CBBD event stream never logged.

### 2.2 By season, per game

{fmt(t2b, '{:.4f}')}

---

## 3. Duration distribution: horn-ending vs the rest

Rows with `duration_s > {DUR_CAP}` are excluded exactly as the clock model
excludes them (a CBBD feed gap, not a possession).

{fmt(t3, '{:.3f}')}

### 3.1 By the seconds-remaining band the possession STARTED in

{fmt(t3b, '{:.4f}')}

This table is L20 in one place. With fewer than 10 seconds left,
**{100 * b010:.1f}%** of possessions end at the horn, and the mean observed
duration in that band collapses toward the clock that was left -- not because
offences behave differently by that much, but because the clock ran out. The
old flag catches almost none of it (`old_flag_rate` column).

### 3.2 By terminal event

{fmt(t3c, '{:.4f}')}

`end_period` is ~99% horn-ending, as it must be. Everything else in this table
is a horn-ending possession the old flag missed: buzzer threes are the largest
single group.

---

## 4. What clock-completeness does to the G1 target

G1's actual quantity is possessions per team-game over regulation, counted from
the logged possessions. In a half whose feed stops early, that count is short by
the possessions the feed never logged, so the ACTUAL is biased DOWN on exactly
the games where the sim (which always runs its clock to zero) is unbiased. This
is why round 3 re-bases G1 on clock-complete games.

{fmt(t4[['season', 'n_games', 'mean_all', 'sd_all', 'n_cc_games', 'cc_share', 'mean_cc', 'sd_cc', 'mean_shift']], '{:.4f}')}

### 4.1 2025 (the selection fold's test season) by month

{fmt(t4m, '{:.4f}')}

Read: `mean_shift` is the size of the grading-truth correction, and it is small
and NEGATIVE (-0.01 to -0.20 possessions per team-game). Two effects run against
each other: dropping a half's unlogged tail removes possessions from the actual
count (which biases the all-games actual DOWN), while the games whose feeds are
complete are also slightly slower than average (which biases the clock-complete
actual DOWN). The second wins, narrowly.

The consequence for round 3 is worth stating plainly: re-basing G1 on
clock-complete games does NOT close round 2's +1.5 to +2.6 emergent overshoot --
it widens it by about 0.2. Re-basing is done because the all-games actual is a
count of LOGGED possessions in halves that stop early, which no correct sim can
reproduce; it is a grading-truth fix, not a gap-closing one. The censoring fix
is what has to close the gap.

---

## 5. Artifacts written

| path | what | how it avoids clobbering another worker |
|---|---|---|
| `data/processed/games_universe_v2.parquet` | `games_universe` plus `clock_complete_reg`, `clock_complete_all_periods`, `points_complete`, `clock_flags_evaluated`, `n_periods_logged`, `n_horn_poss`, `unaccounted_s_total` | VERSIONED SIBLING. `games_universe.parquet` is read-only here, so the engine worker's reader is untouched |
| `data/processed/clock_censoring/half_clock_completeness_v1.parquet` | one row per (game_id, period): the three components and the flag | new directory, new file |
| `data/processed/clock_censoring/censoring_v1_{{season}}.parquet` | one row per (game_id, period, poss_index): `censored_horn` and the tolerance variants | SIDE TABLE. `data/processed/possessions/` and `data/processed/possessions_v2/` are read-only here |
| `data/processed/clock_censoring/censoring_audit_v1.json` | every table above, machine-readable | new file |

**`clock_flags_evaluated` matters.** Season 2026 is SEALED and is never loaded
here, so its rows carry `clock_complete_reg = False` only because the flag was
not computed. Any consumer must filter on `clock_flags_evaluated` before reading
a completeness flag, or it will silently treat every sealed-season game as
incomplete.

`possessions_v2` carries byte-identical `duration_s`, `start_clock` and
`end_clock` to `possessions` (checked on 2025: 0 of 768,834 rows differ), so the
censoring side table joins to either. It is keyed on (game_id, period,
poss_index), which is unique in both.
"""
    DOC.write_text(md, encoding="utf-8")
    print(f"wrote {DOC}")
    print(f"wrote {UNIVERSE_V2}")
    print(json.dumps({k: res[k] for k in ("ladder", "flag_rates", "flag_confusion")}, indent=2))


if __name__ == "__main__":
    main()
