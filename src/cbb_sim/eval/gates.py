"""
gates.py -- G1-G9, engine-agnostic (SIM_GUARDRAILS.md section 3).

This generalises `scripts/grade_control.py`'s `gate_g1` / `gate_g5` / `gate_g6`
/ `gate_g9_breakdowns` (the prototype, read but not imported: those functions
are keyed to the Control's own `summary.parquet`, which carries Control-only
columns like `rating_diff`). Every function here instead takes the CONTRACT's
`games.parquet` (`contract.py`) plus the season truth tables (`reference.py`),
so any engine that satisfies the contract gets the same gates for free.

Every gate returns a `GateResult`: one or more `GateCheck` rows, each ending
in a literal PASS / FAIL / UNDERPOWERED / NEEDS-INSTRUMENTATION, plus detail
tables for the markdown report. `UNDERPOWERED` (n < MIN_CELL_N) is never
folded into PASS or FAIL (SIM_GUARDRAILS: "Underpowered cells are labelled
underpowered, never presented as signal or absence of signal"); a gate whose
required INPUT is absent (an optional box column, players.parquet) reports
NEEDS-INSTRUMENTATION, never a fake PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from cbb_sim.eval import reference as ref_mod

DEFAULT_MIN_CELL_N = 300
TERCILE_LABELS = ("bottom_tercile", "middle_tercile", "top_tercile")


# ---------------------------------------------------------------------------
# status plumbing
# ---------------------------------------------------------------------------
def status_of(ok: bool | None) -> str:
    if ok is None:
        return "NEEDS-INSTRUMENTATION"
    return "PASS" if ok else "FAIL"


def within(value: float, target: float, tol: float) -> bool | None:
    if not np.isfinite(value) or not np.isfinite(target):
        return None
    return abs(value - target) <= tol


def cell_status(n: int, ok: bool | None, min_cell_n: int = DEFAULT_MIN_CELL_N) -> str:
    if n < min_cell_n:
        return "UNDERPOWERED"
    return status_of(ok)


@dataclass
class GateCheck:
    quantity: str
    value: str
    target: str
    tolerance: str
    status: str


@dataclass
class GateResult:
    gate: str
    title: str
    checks: list[GateCheck] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        statuses = [c.status for c in self.checks]
        if not statuses:
            return "NEEDS-INSTRUMENTATION"
        if "FAIL" in statuses:
            return "FAIL"
        if "NEEDS-INSTRUMENTATION" in statuses:
            return "NEEDS-INSTRUMENTATION"
        if all(s == "UNDERPOWERED" for s in statuses):
            return "NEEDS-INSTRUMENTATION"
        return "PASS"


# ---------------------------------------------------------------------------
# building the grading frame from contract games.parquet + truth
# ---------------------------------------------------------------------------
def summarise_games(games: pd.DataFrame) -> pd.DataFrame:
    """Per game_id: sim mean/SD margin & total, p_home, sim OT rate, seed
    count. Engine-agnostic re-implementation of
    `cbb_sim.control.simulate.summarise`'s per-game aggregation, using the
    contract's `n_periods` instead of Control's `n_ot`."""
    s = games.copy()
    s["margin"] = s["home_pts"].astype("int64") - s["away_pts"].astype("int64")
    s["total"] = s["home_pts"].astype("int64") + s["away_pts"].astype("int64")
    s["home_win"] = np.where(s["margin"] > 0, 1.0, np.where(s["margin"] < 0, 0.0, 0.5))
    s["sim_went_ot"] = s["n_periods"] > 2
    g = s.groupby("game_id")
    out = pd.DataFrame({
        "sim_margin_mean": g["margin"].mean(),
        "sim_margin_sd": g["margin"].std(),
        "sim_total_mean": g["total"].mean(),
        "sim_total_sd": g["total"].std(),
        "sim_home_pts_mean": g["home_pts"].mean(),
        "sim_away_pts_mean": g["away_pts"].mean(),
        "sim_poss_mean": g["possessions"].mean(),
        "sim_poss_sd": g["possessions"].std(),
        "p_home": g["home_win"].mean(),
        "sim_ot_rate": g["sim_went_ot"].mean(),
        "n_seeds": g.size(),
    }).reset_index()
    return out


def build_grading_frame(games: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary, raw): `summary` is one row per game_id, truth columns
    (month, neutral, home_team_id, away_team_id, margin, total, game_poss,
    n_periods) inner-joined onto the per-game sim aggregate; `raw` is the
    contract's own per-(game, seed) rows restricted to the same games, with
    margin/total attached, for gates that need the seed-level distribution
    (G1 pooled possessions, G5 dispersion/PIT)."""
    summary = summarise_games(games)
    actual = ref_mod.load_actual_games(season)
    poss = ref_mod.load_actual_possessions(season)
    actual = actual.merge(poss, on="game_id", how="left")
    merged = actual.merge(summary, on="game_id", how="inner")
    merged["is_home_win"] = (merged["margin"] > 0).astype(float)

    raw = games[games["game_id"].isin(merged["game_id"])].copy()
    raw["margin"] = raw["home_pts"].astype("int64") - raw["away_pts"].astype("int64")
    raw["total"] = raw["home_pts"].astype("int64") + raw["away_pts"].astype("int64")
    return merged.reset_index(drop=True), raw.reset_index(drop=True)


# ---------------------------------------------------------------------------
# G1 -- possessions per game
# ---------------------------------------------------------------------------
def gate_g1(summary: pd.DataFrame, raw: pd.DataFrame, season: int, tol: dict,
            min_cell_n: int = DEFAULT_MIN_CELL_N) -> GateResult:
    gref = ref_mod.load_gate_targets(season)

    def ref_val(breakdown, group, metric):
        return ref_mod.gate_target_value(gref, breakdown, group, metric)

    month = summary.set_index("game_id")["month"]
    r = raw.copy()
    r["month"] = r["game_id"].map(month)

    groups = [("season", "all", r, summary)]
    for mo in sorted(summary["month"].dropna().unique()):
        groups.append(("month", int(mo), r[r["month"] == mo], summary[summary["month"] == mo]))

    rows = []
    for breakdown, group, rr, ss in groups:
        sim_mean, sim_sd = float(rr["possessions"].mean()), float(rr["possessions"].std())
        act_mean, act_sd = float(ss["game_poss"].mean()), float(ss["game_poss"].std())
        n = int(len(ss))
        rows.append({
            "breakdown": breakdown, "group": group, "n_games": n,
            "sim_mean": sim_mean, "actual_mean": act_mean,
            "ref_mean": ref_val(breakdown, group, "poss_per_game_mean"),
            "sim_sd": sim_sd, "actual_sd": act_sd,
            "ref_sd": ref_val(breakdown, group, "poss_per_game_sd"),
            "d_mean": sim_mean - act_mean, "d_sd": sim_sd - act_sd,
            "status_mean": cell_status(n, within(sim_mean, act_mean, tol["g1_mean"]), min_cell_n),
            "status_sd": cell_status(n, within(sim_sd, act_sd, tol["g1_sd"]), min_cell_n),
        })
    tab = pd.DataFrame(rows)
    overall = tab.iloc[0]

    by_month = tab[tab["breakdown"] == "month"]
    powered = by_month[by_month["status_mean"] != "UNDERPOWERED"]
    bad = powered[(powered["status_mean"] == "FAIL") | (powered["status_sd"] == "FAIL")]

    checks = [
        GateCheck("possessions/game mean", f"{overall['sim_mean']:.3f} vs {overall['actual_mean']:.3f}",
                  f"{overall['actual_mean']:.3f}", f"+/-{tol['g1_mean']}", overall["status_mean"]),
        GateCheck("possessions/game SD", f"{overall['sim_sd']:.3f} vs {overall['actual_sd']:.3f}",
                  f"{overall['actual_sd']:.3f}", f"+/-{tol['g1_sd']}", overall["status_sd"]),
        GateCheck("by month (mean and SD)", f"{len(powered) - len(bad)}/{len(powered)} powered months inside",
                  "all inside", "see per-month table", status_of(len(bad) == 0) if len(powered) else "NEEDS-INSTRUMENTATION"),
    ]
    return GateResult(
        gate="G1", title="Possessions per game, mean and SD (overall, by month)",
        checks=checks, tables=[("by breakdown", tab)],
        notes=[f"{len(by_month) - len(powered)} month(s) below n={min_cell_n} labelled UNDERPOWERED and not scored."],
    )


# ---------------------------------------------------------------------------
# G2 -- points per possession by offense tercile x defense tercile
# ---------------------------------------------------------------------------
def gate_g2(summary: pd.DataFrame, raw: pd.DataFrame, season: int, tol: dict,
            min_cell_n: int = DEFAULT_MIN_CELL_N) -> GateResult:
    actual_games = ref_mod.load_actual_games(season)
    tiers = ref_mod.team_quality_terciles(actual_games)
    off_tier, def_tier = tiers["offense"], tiers["defense"]

    sim = raw.merge(summary[["game_id", "home_team_id", "away_team_id"]], on="game_id", how="inner")
    home_rows = sim[["game_id", "home_team_id", "away_team_id", "home_pts", "possessions"]].rename(
        columns={"home_team_id": "team_id", "away_team_id": "opp_team_id", "home_pts": "pts"})
    away_rows = sim[["game_id", "away_team_id", "home_team_id", "away_pts", "possessions"]].rename(
        columns={"away_team_id": "team_id", "home_team_id": "opp_team_id", "away_pts": "pts"})
    tg_sim = pd.concat([home_rows, away_rows], ignore_index=True)
    tg_sim["ppp"] = tg_sim["pts"] / tg_sim["possessions"]
    tg_sim["off_tier"] = tg_sim["team_id"].map(off_tier)
    tg_sim["def_tier"] = tg_sim["opp_team_id"].map(def_tier)

    box = ref_mod.load_actual_team_box(season)
    box["off_tier"] = box["team_id"].map(off_tier)
    box["def_tier"] = box["opp_team_id"].map(def_tier)

    rows = []
    for ot in TERCILE_LABELS:
        for dt in TERCILE_LABELS:
            sub_sim = tg_sim[(tg_sim["off_tier"] == ot) & (tg_sim["def_tier"] == dt)]
            sub_act = box[(box["off_tier"] == ot) & (box["def_tier"] == dt)]
            n = int(len(sub_act))
            sim_mean = float(sub_sim["ppp"].mean()) if len(sub_sim) else float("nan")
            act_mean = float(sub_act["ppp"].mean()) if len(sub_act) else float("nan")
            rows.append({
                "offense_tercile": ot, "defense_tercile": dt, "n_team_games": n,
                "sim_ppp": sim_mean, "actual_ppp": act_mean, "delta": sim_mean - act_mean,
                "status": cell_status(n, within(sim_mean, act_mean, tol["g2_ppp"]), min_cell_n),
            })
    tab = pd.DataFrame(rows)
    powered = tab[tab["status"] != "UNDERPOWERED"]
    bad = powered[powered["status"] == "FAIL"]
    checks = [GateCheck(
        "PPP by offense x defense tercile (9 cells)",
        f"{len(powered) - len(bad)}/{len(powered)} powered cells inside +/-{tol['g2_ppp']}",
        "all inside", f"+/-{tol['g2_ppp']} per cell",
        status_of(len(bad) == 0) if len(powered) else "NEEDS-INSTRUMENTATION",
    )]
    return GateResult(
        gate="G2", title="Points per possession, by offense tercile x defense tercile",
        checks=checks, tables=[("by tercile cell", tab)],
        notes=["Terciles are each team's OWN season points-scored / points-allowed average "
               "(`reference.team_quality_terciles`), grading-only, never a model feature."],
    )


# ---------------------------------------------------------------------------
# G3 -- shot mix per possession, by team
# ---------------------------------------------------------------------------
def gate_g3(summary: pd.DataFrame, raw: pd.DataFrame, season: int, tol: dict,
            box_available: dict[str, bool], min_cell_n: int = DEFAULT_MIN_CELL_N,
            truth_dir: str | None = None) -> GateResult:
    needed = ("fga3", "fga2_rim", "fga2_jump", "fta")
    missing = [s for s in needed if not box_available.get(s, False)]
    if missing:
        return GateResult(
            gate="G3", title="Shot mix per possession: 3PA share, rim share, FTA/FGA (by team)",
            checks=[GateCheck("shot mix by team", "n/a", "n/a", "+/-1.5pp", "NEEDS-INSTRUMENTATION")],
            notes=[f"games.parquet lacks optional box column pair(s) for: {', '.join(missing)}."],
        )

    r = raw.merge(summary[["game_id", "home_team_id", "away_team_id"]], on="game_id", how="inner")
    sides = []
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        d = r[["game_id", team_col, f"{side}_fga3", f"{side}_fga2_rim", f"{side}_fga2_jump", f"{side}_fta"]].copy()
        d.columns = ["game_id", "team_id", "fga3", "fga2_rim", "fga2_jump", "fta"]
        sides.append(d)
    sim = pd.concat(sides, ignore_index=True)
    sim["fga"] = sim["fga3"] + sim["fga2_rim"] + sim["fga2_jump"]
    sim["three_share"] = sim["fga3"] / sim["fga"]
    sim["rim_share"] = sim["fga2_rim"] / sim["fga"]
    sim["fta_fga"] = sim["fta"] / sim["fga"]
    sim_by_team = sim.groupby("team_id")[["three_share", "rim_share", "fta_fga"]].mean()

    box = ref_mod.load_actual_team_box(season)
    act_by_team = box.groupby("team_id").agg(
        n_games=("game_id", "size"), three_share=("three_share", "mean"), fta_fga=("ft_rate", "mean"),
    )

    rows = []
    for metric, col in (("three_pa_share", "three_share"), ("fta_per_fga", "fta_fga")):
        for team_id, act_row in act_by_team.iterrows():
            if team_id not in sim_by_team.index:
                continue
            n = int(act_row["n_games"])
            sim_v = float(sim_by_team.loc[team_id, col])
            act_v = float(act_row[col])
            rows.append({
                "metric": metric, "team_id": team_id, "n_games": n,
                "sim": sim_v, "actual": act_v, "delta_pp": (sim_v - act_v) * 100,
                "status": cell_status(n, within(sim_v, act_v, tol["g3_pp"] / 100), min_cell_n),
            })

    # rim share -- PROVISIONAL: truth_tables_v1's team_game_shots_v1.parquet
    # (event layer, `docs/tests/truth_tables_v1_2026-09-10.md`) is the first
    # time this metric has a truth source at all.
    team_truth = ref_mod.load_team_shot_truth(season, truth_dir) if truth_dir else None
    rim_check = GateCheck("rim share by team", "n/a", "n/a", f"+/-{tol['g3_pp']}pp", "NEEDS-INSTRUMENTATION")
    pooled_checks: list[GateCheck] = []
    notes = []

    if team_truth is not None and len(team_truth):
        tt = team_truth[team_truth["ev_fga"].notna() & (team_truth["ev_fga"] > 0)]
        act_rim_by_team = tt.groupby("team_id").agg(
            n_games=("game_id", "size"), rim_share=("rim_share_ev", "mean"),
        )
        for team_id, act_row in act_rim_by_team.iterrows():
            if team_id not in sim_by_team.index:
                continue
            n = int(act_row["n_games"])
            sim_v = float(sim_by_team.loc[team_id, "rim_share"])
            act_v = float(act_row["rim_share"])
            rows.append({
                "metric": "rim_share", "team_id": team_id, "n_games": n,
                "sim": sim_v, "actual": act_v, "delta_pp": (sim_v - act_v) * 100,
                "status": cell_status(n, within(sim_v, act_v, tol["g3_pp"] / 100), min_cell_n),
            })
        rim_pow = pd.DataFrame(rows)
        rim_pow = rim_pow[rim_pow["metric"] == "rim_share"]
        rim_powered = rim_pow[rim_pow["status"] != "UNDERPOWERED"]
        rim_bad = rim_powered[rim_powered["status"] == "FAIL"]
        rim_check = GateCheck(
            "rim share by team", f"{len(rim_powered) - len(rim_bad)}/{len(rim_powered)} powered teams inside",
            "all inside", f"+/-{tol['g3_pp']}pp (PROVISIONAL truth)",
            status_of(len(rim_bad) == 0) if len(rim_powered) else "NEEDS-INSTRUMENTATION",
        )

        # season-level pooled reads (genuinely powered, n = every team-game) --
        # the by-team cells above are UNDERPOWERED at ~30-40 games/team under
        # min_cell_n=300, so this is the only read that can actually PASS/FAIL.
        sim_pool_fga = sim["fga"].sum()
        sim_pooled = {
            "three_pa_share": sim["fga3"].sum() / sim_pool_fga,
            "fta_per_fga": sim["fta"].sum() / sim_pool_fga,
            "rim_share": sim["fga2_rim"].sum() / sim_pool_fga,
        }
        box_pool_fga = box["fga"].sum()
        act_pooled = {
            "three_pa_share": box["tpa"].sum() / box_pool_fga,
            "fta_per_fga": box["fta"].sum() / box_pool_fga,
            "rim_share": tt["ev_fga_rim"].sum() / tt["ev_fga"].sum(),
        }
        for metric in ("three_pa_share", "fta_per_fga", "rim_share"):
            sv, av = float(sim_pooled[metric]), float(act_pooled[metric])
            pooled_checks.append(GateCheck(
                f"{metric} (season, pooled, PROVISIONAL)", f"{sv:.4f} vs {av:.4f}",
                f"{av:.4f}", f"+/-{tol['g3_pp']}pp",
                status_of(within(sv, av, tol['g3_pp'] / 100)),
            ))
        notes.append(
            "Season-level pooled reads (sum of attempts over every team-game, not a mean of "
            "per-team means) are the only genuinely powered read here -- the by-team cells are "
            "UNDERPOWERED at ~30-40 games/team under min_cell_n=300."
        )
        notes.append(
            "Rim share truth is PROVISIONAL: `data/processed/truth/team_game_shots_v1.parquet` "
            "(CBBD possessions_v2 event layer, first wired in here 2026-09-10, "
            "`docs/tests/truth_tables_v1_2026-09-10.md`), not yet itself bake-off-validated as a "
            "grading source the way the box-derived 3PA-share/FTA-rate truth is."
        )
    else:
        notes.append("Rim-vs-jump truth requires pbp shot-location classification; rim share is "
                     "NEEDS-INSTRUMENTATION because no --truth-dir was supplied (or it has no "
                     "team_game_shots_v1.parquet for this season), even when the sim reports fga2_rim.")

    tab = pd.DataFrame(rows)
    three_fta = tab[tab["metric"].isin(("three_pa_share", "fta_per_fga"))] if len(tab) else tab
    powered = three_fta[three_fta["status"] != "UNDERPOWERED"] if len(three_fta) else three_fta
    bad = powered[powered["status"] == "FAIL"] if len(powered) else powered
    checks = [GateCheck(
        "3PA share & FTA/FGA by team", f"{len(powered) - len(bad)}/{len(powered)} powered team-metrics inside",
        "all inside", f"+/-{tol['g3_pp']}pp",
        status_of(len(bad) == 0) if len(powered) else "NEEDS-INSTRUMENTATION",
    ), rim_check, *pooled_checks]
    return GateResult(
        gate="G3", title="Shot mix per possession: 3PA share, rim share, FTA/FGA (by team)",
        checks=checks, tables=[("by team-metric", tab)] if len(tab) else [], notes=notes,
    )


# ---------------------------------------------------------------------------
# G4 -- four factors, offense and defense, by team and by tier
# ---------------------------------------------------------------------------
def gate_g4(summary: pd.DataFrame, raw: pd.DataFrame, season: int, tol: dict,
            box_available: dict[str, bool], min_cell_n: int = DEFAULT_MIN_CELL_N,
            truth_dir: str | None = None) -> GateResult:
    needed = ("fga3", "fga2_rim", "fga2_jump", "fta", "tov", "oreb", "dreb")
    # MAKE columns, added to the contract 2026-09-10 so eFG% can read
    # (`src/cbb_sim/eval/contract.py`, `docs/models/engine/model.md`). A
    # results directory built before that date (or by an engine build that
    # has not picked it up) simply lacks these -- box_available reports the
    # pair missing exactly like any other optional column, and eFG% below
    # falls back to NEEDS-INSTRUMENTATION rather than a fabricated number.
    needed_efg = ("fgm2_rim", "fgm2_jump", "fgm3", "ftm")
    missing = [s for s in needed if not box_available.get(s, False)]
    missing_efg = [s for s in needed_efg if not box_available.get(s, False)]

    efg_check = GateCheck(
        "eFG% (offense/defense)", "n/a", "n/a", f"+/-{tol['g4_pp']}pp", "NEEDS-INSTRUMENTATION",
    )
    checks: list[GateCheck] = [efg_check]
    notes: list[str] = []
    if missing_efg:
        notes.append(
            "eFG% needs MAKE counts (FGM by shot class + FTM); this results directory's "
            f"games.parquet lacks optional box column pair(s) for: {', '.join(missing_efg)} -- "
            "either a run predating the 2026-09-10 FGM/FTM contract extension "
            "(docs/models/engine/model.md) or an engine build that has not picked it up yet. "
            "TOV%/OREB%/FT-rate below are unaffected -- they only ever needed attempt counts."
        )
    if missing:
        checks.append(GateCheck(
            "TOV% / OREB% / FT rate", "n/a", "n/a", "see SIM_GUARDRAILS G4 row", "NEEDS-INSTRUMENTATION",
        ))
        notes.append(f"games.parquet also lacks optional box column pair(s) for: {', '.join(missing)}.")
        return GateResult(gate="G4", title="Four factors, offense and defense (by team, by tier)",
                           checks=checks, notes=notes)

    r = raw.merge(summary[["game_id", "home_team_id", "away_team_id"]], on="game_id", how="inner")
    efg_cols = () if missing_efg else needed_efg
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        cols = [f"{side}_{s}" for s in (*needed, *efg_cols)]
        d = r[["game_id", f"{side}_team_id", f"{opp}_dreb", *cols]].copy()
        d.columns = ["game_id", "team_id", "opp_dreb", *needed, *efg_cols]
        sides.append(d)
    sim = pd.concat(sides, ignore_index=True)
    sim["fga"] = sim["fga3"] + sim["fga2_rim"] + sim["fga2_jump"]
    poss_est = sim["fga"] - sim["oreb"] + sim["tov"] + 0.44 * sim["fta"]
    sim["tov_pct"] = sim["tov"] / poss_est
    sim["oreb_pct"] = sim["oreb"] / (sim["oreb"] + sim["opp_dreb"])
    sim["ft_rate"] = sim["fta"] / sim["fga"]
    sim_metrics = ["tov_pct", "oreb_pct", "ft_rate"]
    if not missing_efg:
        sim["fgm"] = sim["fgm2_rim"] + sim["fgm2_jump"] + sim["fgm3"]
        sim["efg_pct"] = (sim["fgm"] + 0.5 * sim["fgm3"]) / sim["fga"]
        sim_metrics.append("efg_pct")
    sim_by_team = sim.groupby("team_id")[sim_metrics].mean()

    box = ref_mod.load_actual_team_box(season)
    act_by_team = box.groupby("team_id").agg(
        n_games=("game_id", "size"), tov_pct=("tov_pct", "mean"),
        oreb_pct=("oreb_pct", "mean"), ft_rate=("ft_rate", "mean"),
    )

    rows = []
    for metric, tolkey in (("tov_pct", "g4_pp"), ("oreb_pct", "g4_pp"), ("ft_rate", "g4_ft_rate")):
        tolerance = tol[tolkey] / 100 if tolkey == "g4_pp" else tol[tolkey]
        for team_id, act_row in act_by_team.iterrows():
            if team_id not in sim_by_team.index:
                continue
            n = int(act_row["n_games"])
            sim_v = float(sim_by_team.loc[team_id, metric])
            act_v = float(act_row[metric])
            rows.append({
                "metric": metric, "team_id": team_id, "n_games": n,
                "sim": sim_v, "actual": act_v, "delta": sim_v - act_v,
                "status": cell_status(n, within(sim_v, act_v, tolerance), min_cell_n),
            })

    # eFG% -- PROVISIONAL: team_game_shots_v1.parquet's box_fgm/box_fga/box_fgm3
    # (hoopR team_box, second-sourced against the CBBD event layer,
    # `docs/tests/truth_tables_v1_2026-09-10.md`) is the ACTUAL source; the SIM
    # side needs the games.parquet FGM/FTM columns checked above. Both are now
    # available where the run and the truth dir support them.
    team_truth = ref_mod.load_team_shot_truth(season, truth_dir) if truth_dir else None
    efg_pooled_check = None
    if missing_efg:
        pass  # already noted above; nothing further to compute
    elif team_truth is None or not len(team_truth):
        notes.append(
            "games.parquet has the sim-side FGM/FTM columns, but no --truth-dir "
            f"team_game_shots_v1.parquet was found for season {season}; eFG% stays "
            "NEEDS-INSTRUMENTATION."
        )
    else:
        tt = team_truth[team_truth["box_fga"].notna() & team_truth["box_fgm"].notna()].copy()
        tt["efg_pct"] = (tt["box_fgm"] + 0.5 * tt["box_fgm3"]) / tt["box_fga"]
        act_efg_by_team = tt.groupby("team_id").agg(
            n_games=("game_id", "size"), efg_pct=("efg_pct", "mean"),
        )
        for team_id, act_row in act_efg_by_team.iterrows():
            if team_id not in sim_by_team.index:
                continue
            n = int(act_row["n_games"])
            sim_v = float(sim_by_team.loc[team_id, "efg_pct"])
            act_v = float(act_row["efg_pct"])
            rows.append({
                "metric": "efg_pct", "team_id": team_id, "n_games": n,
                "sim": sim_v, "actual": act_v, "delta": sim_v - act_v,
                "status": cell_status(n, within(sim_v, act_v, tol["g4_pp"] / 100), min_cell_n),
            })
        efg_rows = [row for row in rows if row["metric"] == "efg_pct"]
        efg_powered = [row for row in efg_rows if row["status"] != "UNDERPOWERED"]
        efg_bad = [row for row in efg_powered if row["status"] == "FAIL"]
        efg_check = GateCheck(
            "eFG% (offense/defense)",
            f"{len(efg_powered) - len(efg_bad)}/{len(efg_powered)} powered teams inside",
            "all inside", f"+/-{tol['g4_pp']}pp (PROVISIONAL truth)",
            status_of(len(efg_bad) == 0) if len(efg_powered) else "NEEDS-INSTRUMENTATION",
        )
        checks[0] = efg_check
        sim_pool_fga = sim["fga"].sum()
        sim_pool_efg = float((sim["fgm"].sum() + 0.5 * sim["fgm3"].sum()) / sim_pool_fga)
        act_pool_efg = float((tt["box_fgm"] + 0.5 * tt["box_fgm3"]).sum() / tt["box_fga"].sum())
        efg_pooled_check = GateCheck(
            "efg_pct (season, pooled, PROVISIONAL)", f"{sim_pool_efg:.4f} vs {act_pool_efg:.4f}",
            f"{act_pool_efg:.4f}", f"+/-{tol['g4_pp']}pp",
            status_of(within(sim_pool_efg, act_pool_efg, tol['g4_pp'] / 100)),
        )
        notes.append(
            "eFG% ACTUAL is data/processed/truth/team_game_shots_v1.parquet box_fgm/box_fga/box_fgm3 "
            "(hoopR team_box, second-sourced against the CBBD event layer, PROVISIONAL -- not yet "
            "itself bake-off-validated as a grading source)."
        )

    tab = pd.DataFrame(rows)
    for metric in ("tov_pct", "oreb_pct", "ft_rate"):
        sub = tab[tab["metric"] == metric]
        powered = sub[sub["status"] != "UNDERPOWERED"]
        bad = powered[powered["status"] == "FAIL"]
        checks.append(GateCheck(
            f"{metric} by team", f"{len(powered) - len(bad)}/{len(powered)} powered teams inside",
            "all inside", f"+/-{tol['g4_pp']}pp" if metric != "ft_rate" else f"+/-{tol['g4_ft_rate']}",
            status_of(len(bad) == 0) if len(powered) else "NEEDS-INSTRUMENTATION",
        ))

    # season-level pooled reads -- PROVISIONAL, gated on --truth-dir the same
    # way G3's are (tov_pct/oreb_pct/ft_rate always; eFG% too once the sim
    # side has the make columns -- efg_pooled_check is None otherwise).
    if team_truth is not None and len(team_truth):
        sim_pooled_metric = {
            "tov_pct": float(sim["tov"].sum() / poss_est.sum()),
            "oreb_pct": float(sim["oreb"].sum() / (sim["oreb"] + sim["opp_dreb"]).sum()),
            "ft_rate": float(sim["fta"].sum() / sim["fga"].sum()),
        }
        act_pooled_metric = {
            "tov_pct": float(box["tov"].sum() / (box["fga"] - box["oreb"] + box["tov"] + 0.44 * box["fta"]).sum()),
            "oreb_pct": float(box["oreb"].sum() / (box["oreb"] + box["opp_dreb"]).sum()),
            "ft_rate": float(box["fta"].sum() / box["fga"].sum()),
        }
        for metric, tolkey in (("tov_pct", "g4_pp"), ("oreb_pct", "g4_pp"), ("ft_rate", "g4_ft_rate")):
            tolerance = tol[tolkey] / 100 if tolkey == "g4_pp" else tol[tolkey]
            sim_v = sim_pooled_metric[metric]
            act_v = act_pooled_metric[metric]
            checks.append(GateCheck(
                f"{metric} (season, pooled, PROVISIONAL)", f"{sim_v:.4f} vs {act_v:.4f}",
                f"{act_v:.4f}", f"+/-{tol[tolkey]}pp" if tolkey == "g4_pp" else f"+/-{tol[tolkey]}",
                status_of(within(sim_v, act_v, tolerance)),
            ))
        if efg_pooled_check is not None:
            checks.append(efg_pooled_check)
        notes.append(
            "Season-level pooled tov_pct/oreb_pct/ft_rate reads (sum over every team-game) are "
            "genuinely powered; the by-team cells above are UNDERPOWERED at ~30-40 games/team under "
            "min_cell_n=300."
        )
    return GateResult(gate="G4", title="Four factors, offense and defense (by team, by tier)",
                       checks=checks, tables=[("by team-metric", tab)], notes=notes)


# ---------------------------------------------------------------------------
# G5 -- dispersion
# ---------------------------------------------------------------------------
def gate_g5(summary: pd.DataFrame, raw: pd.DataFrame, season: int, tol: dict,
            rng_seed: int = 20260910) -> GateResult:
    r = raw
    resid_m = float((summary["margin"] - summary["sim_margin_mean"]).std())
    resid_t = float((summary["total"] - summary["sim_total_mean"]).std())
    sim_sd_m = float(summary["sim_margin_sd"].mean())
    sim_sd_t = float(summary["sim_total_sd"].mean())

    corr_sim = float(np.corrcoef(r["home_pts"].astype(float), r["away_pts"].astype(float))[0, 1])
    corr_act = float(np.corrcoef(summary["home_score"].astype(float), summary["away_score"].astype(float))[0, 1])

    g = r.groupby("game_id")["margin"]
    actual = summary.set_index("game_id")["margin"]
    below = g.apply(lambda x: float((x < actual.loc[x.name]).mean()))
    equal = g.apply(lambda x: float((x == actual.loc[x.name]).mean()))
    rng = np.random.default_rng(rng_seed)
    pit = (below + equal * rng.random(len(below))).to_numpy()
    ks = stats.kstest(pit, "uniform")

    ratio_m = sim_sd_m / resid_m
    ratio_t = sim_sd_t / resid_t
    st_m = status_of(tol["g5_sd_ratio_lo"] <= ratio_m <= tol["g5_sd_ratio_hi"])
    st_t = status_of(tol["g5_sd_ratio_lo"] <= ratio_t <= tol["g5_sd_ratio_hi"])
    st_corr = status_of(abs(corr_sim - corr_act) <= tol["g5_corr"])
    st_pit = status_of(ks.pvalue > tol["g5_ks_p"])

    disp_tab = pd.DataFrame([
        {"quantity": "margin", "mean_sim_SD": sim_sd_m, "SD(actual - sim mean)": resid_m,
         "ratio": ratio_m, "status": st_m},
        {"quantity": "total", "mean_sim_SD": sim_sd_t, "SD(actual - sim mean)": resid_t,
         "ratio": ratio_t, "status": st_t},
    ])
    pit_hist = pd.DataFrame({
        "decile": np.arange(1, 11),
        "share": np.histogram(pit, bins=np.linspace(0, 1, 11))[0] / len(pit),
    })
    checks = [
        GateCheck("margin SD ratio", f"{ratio_m:.4f}", "1.0", "0.95-1.05", st_m),
        GateCheck("total SD ratio", f"{ratio_t:.4f}", "1.0", "0.95-1.05", st_t),
        GateCheck("home/away score correlation", f"{corr_sim:.4f} vs {corr_act:.4f}",
                  f"{corr_act:.4f}", f"+/-{tol['g5_corr']}", st_corr),
        GateCheck("PIT K-S p", f"{ks.pvalue:.3g}", "> " + str(tol["g5_ks_p"]), f"> {tol['g5_ks_p']}", st_pit),
    ]
    return GateResult(
        gate="G5", title="Dispersion: SD ratio, score correlation, PIT histogram",
        checks=checks, tables=[("SD ratio", disp_tab), ("PIT decile histogram", pit_hist)],
        notes=["SD ratio = mean(sim SD) / SD(actual - sim mean); >1 means the engine is too wide.",
               f"PIT K-S statistic D = {ks.statistic:.4f}."],
    )


# ---------------------------------------------------------------------------
# G6 -- home margin, non-neutral vs neutral
# ---------------------------------------------------------------------------
def gate_g6(summary: pd.DataFrame, tol: dict, min_cell_n: int = DEFAULT_MIN_CELL_N) -> GateResult:
    rows = []
    for label, sub in (("non-neutral", summary[summary["neutral"] == 0]),
                        ("neutral", summary[summary["neutral"] == 1])):
        sim = float(sub["sim_margin_mean"].mean())
        act = float(sub["margin"].mean())
        n = int(len(sub))
        rows.append({"site": label, "n": n, "sim": sim, "actual": act, "delta": sim - act,
                     "status": cell_status(n, within(sim, act, tol["g6_margin"]), min_cell_n)})
    tab = pd.DataFrame(rows)
    checks = [GateCheck(f"home margin ({r['site']})", f"{r['sim']:+.3f} vs {r['actual']:+.3f}",
                        f"{r['actual']:+.3f}", f"+/-{tol['g6_margin']}", r["status"])
              for r in rows]
    return GateResult(gate="G6", title="Home margin, non-neutral vs neutral, same games",
                       checks=checks, tables=[("by site", tab)])


# ---------------------------------------------------------------------------
# G7 -- overtime rate; first/second half scoring share
# ---------------------------------------------------------------------------
def gate_g7(summary: pd.DataFrame, season: int, tol: dict) -> GateResult:
    sim_ot = float(summary["sim_ot_rate"].mean())
    act_ot = float(summary["went_ot"].mean())
    st_ot = status_of(within(sim_ot, act_ot, tol["g7_ot_pp"] / 100))
    checks = [GateCheck("OT rate", f"{sim_ot:.4f} vs {act_ot:.4f}", f"{act_ot:.4f}",
                        f"+/-{tol['g7_ot_pp']}pp", st_ot)]
    notes = []
    gref = ref_mod.load_gate_targets(season)
    h1 = ref_mod.gate_target_value(gref, "season", "all", "period_share_1h_mean")
    h2 = ref_mod.gate_target_value(gref, "season", "all", "period_share_2h_mean")
    checks.append(GateCheck("first/second half scoring share", "n/a (contract has no per-period sim score)",
                            f"1H {h1:.4f} / 2H {h2:.4f}", f"+/-{tol['g7_half_pp']}pp", "NEEDS-INSTRUMENTATION"))
    notes.append("games.parquet carries only a whole-game home_pts/away_pts; a per-half score column "
                 "is not in the contract, so the half-share leg is always NEEDS-INSTRUMENTATION.")
    return GateResult(gate="G7", title="Overtime rate; first-half vs second-half scoring share",
                       checks=checks, notes=notes)


# ---------------------------------------------------------------------------
# G8 -- player layer
# ---------------------------------------------------------------------------
def gate_g8(players: pd.DataFrame | None, tol: dict, min_cell_n: int = DEFAULT_MIN_CELL_N,
            season: int | None = None, truth_dir: str | None = None) -> GateResult:
    if players is None:
        return GateResult(
            gate="G8", title="Player layer: minutes, usage share, distribution tails",
            checks=[GateCheck("player layer", "n/a", "n/a", "see SIM_GUARDRAILS G8 row", "NEEDS-INSTRUMENTATION")],
            notes=["players.parquet not found for this engine."],
        )
    per_player = players.groupby(["game_id", "seed", "athlete_id", "team_id"], as_index=False).agg(
        minutes=("minutes", "sum"), fga=("fga", "sum"),
    )
    rotation = per_player[per_player["minutes"] > 0]
    # Vectorized groupby-rank, NOT `.transform(lambda x: x.rank(...))`: the lambda form makes
    # pandas fall back to a per-group Python callback, which is fine at 5-50 seeds but is
    # ~O(n_groups) Python overhead that made a 200-seed run (2.28M (game,seed,team) groups)
    # balloon past 10GB and run for 20+ minutes without finishing (found live grading engine v1
    # F2_2025_s200_v1_clockv3c, 2026-09-11). `groupby(...)["minutes"].rank(ascending=False)`
    # calls pandas' own cythonized rank and is exactly equivalent (verified: identical boolean
    # mask on a synthetic tie-including test) -- same default tie-break method ("average"),
    # just computed without the Python-level per-group call.
    starter_cut = rotation.groupby(["game_id", "seed", "team_id"])["minutes"].rank(ascending=False) <= 5
    minutes_mean = float(rotation.loc[starter_cut, "minutes"].mean())
    minutes_sd = float(rotation.loc[starter_cut, "minutes"].std())
    team_fga = per_player.groupby(["game_id", "seed", "team_id"])["fga"].transform("sum")
    share = np.where(team_fga > 0, per_player["fga"] / team_fga, np.nan)
    top1 = per_player.assign(share=share).groupby(["game_id", "seed", "team_id"])["share"].max().mean()
    n_used = rotation.groupby(["game_id", "seed", "team_id"]).size()

    n_rows = int(len(rotation))

    truth = (ref_mod.load_player_game_truth(season, truth_dir)
              if (truth_dir and season is not None) else None)

    if truth is None:
        checks = [
            GateCheck("rotation minutes mean", f"{minutes_mean:.2f}", "n/a (needs actual box; see notes)",
                      f"+/-{tol['g8_minutes_mean']}",
                      "NEEDS-INSTRUMENTATION" if n_rows < min_cell_n else "PASS"),
            GateCheck("rotation minutes SD ratio", f"{minutes_sd:.2f}", "n/a", "0.9-1.1", "NEEDS-INSTRUMENTATION"),
            GateCheck("top-1 FGA share, mean", f"{float(top1):.4f}", "n/a", "report only", "NEEDS-INSTRUMENTATION"),
            GateCheck("players used per team-game, mean", f"{float(n_used.mean()):.2f}", "n/a", "K-S p > 0.1",
                      "NEEDS-INSTRUMENTATION"),
        ]
        notes = ["Sim-side player aggregates computed from players.parquet; the actual-box comparison "
                 "(hoopR player_box minutes/usage truth) is not wired into this harness -- box-score "
                 "truth exists (`data/raw/hoopr/player_box`) but the per-player join/name-matching layer "
                 "is out of this task's scope, so every sim-vs-actual comparison here is "
                 "NEEDS-INSTRUMENTATION even though the sim-side numbers are real."]
        return GateResult(gate="G8", title="Player layer: minutes, usage share, distribution tails",
                           checks=checks, notes=notes)

    # PROVISIONAL: player_game_v1.parquet (hoopR player_box keyed on ESPN
    # athlete_id, `docs/tests/truth_tables_v1_2026-09-10.md`), first wired
    # into G8 2026-09-10. Restrict to the games this sim run actually covers.
    sim_game_ids = set(players["game_id"].unique().tolist())
    t = truth[truth["game_id"].isin(sim_game_ids)].copy()
    t["minutes"] = t["minutes"].fillna(0)
    t_rotation = t[t["minutes"] > 0]
    # Same vectorized-rank fix as the sim side above.
    t_starter_cut = t_rotation.groupby(["game_id", "team_id"])["minutes"].rank(ascending=False) <= 5
    act_minutes_mean = float(t_rotation.loc[t_starter_cut, "minutes"].mean())
    act_minutes_sd = float(t_rotation.loc[t_starter_cut, "minutes"].std())
    t_team_fga = t.groupby(["game_id", "team_id"])["fga"].transform("sum")
    t_share = np.where(t_team_fga > 0, t["fga"] / t_team_fga, np.nan)
    act_top1 = float(t.assign(share=t_share).groupby(["game_id", "team_id"])["share"].max().mean())
    act_n_used = t_rotation.groupby(["game_id", "team_id"]).size()

    sd_ratio = minutes_sd / act_minutes_sd if act_minutes_sd else float("nan")
    ks_used = stats.ks_2samp(n_used.to_numpy(dtype="float64"), act_n_used.to_numpy(dtype="float64"))
    n_teamgames = int(len(t.groupby(["game_id", "team_id"])))

    checks = [
        GateCheck("rotation minutes mean", f"{minutes_mean:.2f} vs {act_minutes_mean:.2f}",
                  f"{act_minutes_mean:.2f}", f"+/-{tol['g8_minutes_mean']}",
                  cell_status(n_rows, within(minutes_mean, act_minutes_mean, tol["g8_minutes_mean"]), min_cell_n)),
        GateCheck("rotation minutes SD ratio", f"{sd_ratio:.4f}", "1.0",
                  f"{tol['g8_minutes_sd_ratio_lo']}-{tol['g8_minutes_sd_ratio_hi']}",
                  cell_status(n_rows,
                              tol["g8_minutes_sd_ratio_lo"] <= sd_ratio <= tol["g8_minutes_sd_ratio_hi"],
                              min_cell_n)),
        GateCheck("top-1 FGA share, mean", f"{float(top1):.4f} vs {act_top1:.4f}", f"{act_top1:.4f}",
                  "report only (no pre-registered tolerance)", "NEEDS-INSTRUMENTATION"),
        GateCheck("players used per team-game, mean", f"{float(n_used.mean()):.2f} vs {float(act_n_used.mean()):.2f}",
                  f"{float(act_n_used.mean()):.2f}", f"K-S p > {tol['g8_ks_p']}",
                  cell_status(n_teamgames, ks_used.pvalue > tol["g8_ks_p"], min_cell_n)),
    ]
    notes = [
        "PROVISIONAL: actual-box truth from `data/processed/truth/player_game_v1.parquet` (hoopR "
        "player_box keyed on ESPN athlete_id; `docs/tests/truth_tables_v1_2026-09-10.md`), first "
        f"wired into G8 2026-09-10. Restricted to the {len(sim_game_ids)} games this run covers.",
        f"players-used K-S statistic D = {ks_used.statistic:.4f} (p = {ks_used.pvalue:.3g}).",
        "top-1 FGA share has no pre-registered tolerance in docs/gates.yaml; reported, not gated.",
    ]
    return GateResult(gate="G8", title="Player layer: minutes, usage share, distribution tails",
                       checks=checks, notes=notes)


# ---------------------------------------------------------------------------
# G9 -- spread and total accuracy
# ---------------------------------------------------------------------------
def headline(summary: pd.DataFrame) -> dict:
    e_m = summary["sim_margin_mean"] - summary["margin"]
    e_t = summary["sim_total_mean"] - summary["total"]
    brier = float(((summary["p_home"] - summary["is_home_win"]) ** 2).mean())
    slope, intercept = np.polyfit(summary["sim_margin_mean"], summary["margin"], 1)
    var_mc = float((summary["sim_margin_sd"] ** 2).mean() / summary["n_seeds"].mean())
    var_pred = float(summary["sim_margin_mean"].var())
    slope_corr = float(slope * var_pred / max(var_pred - var_mc, 1e-9))
    return {
        "n": int(len(summary)), "margin_mae": float(e_m.abs().mean()), "margin_bias": float(e_m.mean()),
        "total_mae": float(e_t.abs().mean()), "total_bias": float(e_t.mean()), "brier": brier,
        "slope": float(slope), "slope_intercept": float(intercept), "slope_mc_corrected": slope_corr,
        "mc_noise_sd": float(np.sqrt(var_mc)),
    }


def brier_decile(summary: pd.DataFrame) -> pd.DataFrame:
    d = summary[["p_home", "is_home_win"]].copy()
    d["decile"] = pd.qcut(d["p_home"], 10, labels=False, duplicates="drop") + 1
    out = d.groupby("decile").agg(n=("p_home", "size"), pred=("p_home", "mean"),
                                   actual=("is_home_win", "mean")).reset_index()
    out["delta"] = out["actual"] - out["pred"]
    return out


def gate_g9_breakdowns(summary: pd.DataFrame, tol: dict, min_cell_n: int = DEFAULT_MIN_CELL_N) -> dict[str, pd.DataFrame]:
    s = summary.copy()
    tiers = ref_mod.team_quality_terciles(s.rename(columns={"home_score": "home_score", "away_score": "away_score"}))
    s["home_tier"] = s["home_team_id"].map(tiers["margin"]).astype(str)
    s["pred_total_tercile"] = pd.qcut(s["sim_total_mean"], 3, labels=list(TERCILE_LABELS)).astype(str)

    def agg(df, key):
        out = []
        for g, d in df.groupby(key, observed=True):
            e_m = d["sim_margin_mean"] - d["margin"]
            e_t = d["sim_total_mean"] - d["total"]
            slope = np.polyfit(d["sim_margin_mean"], d["margin"], 1)[0] if len(d) > 30 else np.nan
            n = int(len(d))
            out.append({
                key: g, "n": n, "margin_mae": float(e_m.abs().mean()), "margin_bias": float(e_m.mean()),
                "total_mae": float(e_t.abs().mean()), "total_bias": float(e_t.mean()), "slope": float(slope),
                "status_margin_bias": cell_status(n, within(e_m.mean(), 0.0, tol["g9_margin_bias"]), min_cell_n),
                "status_total_bias": cell_status(n, within(e_t.mean(), 0.0, tol["g9_total_bias"]), min_cell_n),
            })
        return pd.DataFrame(out)

    return {"month": agg(s, "month"), "tier": agg(s, "home_tier"), "pred_total_tercile": agg(s, "pred_total_tercile")}


def gate_g9(summary: pd.DataFrame, tol: dict, min_cell_n: int = DEFAULT_MIN_CELL_N) -> GateResult:
    h = headline(summary)
    checks = [
        GateCheck("margin bias", f"{h['margin_bias']:+.4f}", "0", f"+/-{tol['g9_margin_bias']}",
                  status_of(abs(h["margin_bias"]) <= tol["g9_margin_bias"])),
        GateCheck("total bias", f"{h['total_bias']:+.4f}", "0", f"+/-{tol['g9_total_bias']}",
                  status_of(abs(h["total_bias"]) <= tol["g9_total_bias"])),
        GateCheck("calibration slope", f"{h['slope']:.4f}", "1.0",
                  f"{tol['g9_slope_lo']}-{tol['g9_slope_hi']}",
                  status_of(tol["g9_slope_lo"] <= h["slope"] <= tol["g9_slope_hi"])),
    ]
    bd = gate_g9_breakdowns(summary, tol, min_cell_n)
    tables = [("win-probability calibration by decile", brier_decile(summary))]
    for name, tab in bd.items():
        tables.append((f"by {name}", tab))
        n_fail = int((tab["status_margin_bias"] == "FAIL").sum() + (tab["status_total_bias"] == "FAIL").sum())
        n_scored = int((tab["status_margin_bias"] != "UNDERPOWERED").sum()
                       + (tab["status_total_bias"] != "UNDERPOWERED").sum())
        checks.append(GateCheck(f"bias by {name}", f"{n_fail}/{n_scored} scored cells outside tolerance",
                                "all inside", "see per-breakdown table",
                                status_of(n_fail == 0) if n_scored else "NEEDS-INSTRUMENTATION"))
    notes = [f"margin MAE {h['margin_mae']:.4f}, total MAE {h['total_mae']:.4f}, "
             f"win-probability Brier {h['brier']:.5f}, slope_MC_corrected {h['slope_mc_corrected']:.4f} "
             f"(MC noise SD {h['mc_noise_sd']:.3f}, a diagnostic, not the gate)."]
    return GateResult(gate="G9", title="Spread and total accuracy: MAE, signed bias, calibration slope",
                       checks=checks, tables=tables, notes=notes)
