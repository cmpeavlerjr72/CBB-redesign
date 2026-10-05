"""Daily TOV-level monitor (REPORT ONLY). Season-to-date league turnovers per possession, actual vs the engine's prediction, by
days-since-season-start bucket. Nothing here adjusts, clips or recalibrates sim output (CLAUDE.md no-hand-tuning rule); it exists so a
drifting TOV level (PO section 31: descriptive window loop total bias +1.03 F1 / +0.50 F2) is seen the morning after, not after the season.

Definitions (identical formula on both sides so the ratio does not depend on the engine's own possession counter):
  poss_team = FGA - OREB + TOV + 0.475 * FTA          (box estimate; FGA = fga3 + fga2_rim + fga2_jump in the sim)
  league TOV/poss = sum(TOV, both teams) / sum(poss_team, both teams) over the games in the bucket.
Predicted = mean over seeds of each game's sim row, summed over games; actual = verified-final hoopR team_box rows for the same game ids.
The engine's own `possessions` column (one count per game) is reported as a secondary sim-only TOV/engine-possession figure.
Interval: game-level bootstrap of (sim - actual) in TOV/poss; buckets under MIN_GAMES are labelled UNDERPOWERED, never read as signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FTA_W = 0.475
BUCKETS = ((0, 6, "d00-06"), (7, 13, "d07-13"), (14, 27, "d14-27"), (28, 55, "d28-55"), (56, 10_000, "d56+"))
MIN_GAMES = 100


def _poss(fga, oreb, tov, fta):
    return fga - oreb + tov + FTA_W * fta


def sim_game_tov(root: Path, led: pd.DataFrame) -> pd.DataFrame:
    """Per graded game: sim mean TOV / FGA / OREB / FTA per team, summed to game level, from the published run's sim games.parquet
    (matched on the ledger's run_id so the earliest-published run is the one scored)."""
    out = []
    for run_id, g in led.groupby("run_id"):
        ids = set(g["game_id"].astype("int64"))
        for p in sorted(Path(root).glob(f"sim/*/{run_id}/games.parquet")):
            s = pd.read_parquet(p)
            s = s[s["game_id"].isin(ids)]
            if not len(s):
                continue
            for side in ("home", "away"):
                s[f"{side}_fga"] = s[f"{side}_fga3"] + s[f"{side}_fga2_rim"] + s[f"{side}_fga2_jump"]
            m = s.groupby("game_id").mean(numeric_only=True)
            tov = m["home_tov"] + m["away_tov"]
            poss = (_poss(m["home_fga"], m["home_oreb"], m["home_tov"], m["home_fta"])
                    + _poss(m["away_fga"], m["away_oreb"], m["away_tov"], m["away_fta"]))
            out.append(pd.DataFrame({"game_id": m.index.astype("int64"), "run_id": run_id, "sim_tov": tov.to_numpy(),
                                     "sim_poss": poss.to_numpy(), "sim_engine_poss": m["possessions"].to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["game_id", "run_id", "sim_tov", "sim_poss", "sim_engine_poss"])


def actual_game_tov(team_box: pd.DataFrame, game_ids) -> pd.DataFrame:
    """Per game: actual TOV and box-estimate possessions (both teams) from hoopR team_box, only games with exactly two team rows."""
    b = team_box[team_box["game_id"].isin(set(int(x) for x in game_ids))].copy()
    b["poss"] = _poss(b["field_goals_attempted"], b["offensive_rebounds"], b["turnovers"], b["free_throws_attempted"])
    g = b.groupby("game_id").agg(act_tov=("turnovers", "sum"), act_poss=("poss", "sum"), n_rows=("turnovers", "size")).reset_index()
    return g[g["n_rows"] == 2].drop(columns="n_rows").assign(game_id=lambda d: d["game_id"].astype("int64"))


def bucket_of(days: pd.Series) -> pd.Series:
    out = pd.Series(index=days.index, dtype=object)
    for lo, hi, name in BUCKETS:
        out[(days >= lo) & (days <= hi)] = name
    return out


def tov_level_table(d: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """d: one row per game with game_date offset `days`, sim_tov, sim_poss, act_tov, act_poss. Rows per bucket plus season-to-date."""
    rng = np.random.default_rng(seed)
    rows = []

    def one(x: pd.DataFrame, label: str):
        n = len(x)
        r = {"bucket": label, "n_games": int(n)}
        if not n:
            return {**r, "act_tov_per_poss": np.nan, "sim_tov_per_poss": np.nan, "diff": np.nan, "lo95": np.nan, "hi95": np.nan,
                    "act_tov_game": np.nan, "sim_tov_game": np.nan, "sim_tov_per_engine_poss": np.nan, "note": "no games"}
        a_t, a_p, s_t, s_p = (x[c].to_numpy(float) for c in ("act_tov", "act_poss", "sim_tov", "sim_poss"))
        act, sim = a_t.sum() / a_p.sum(), s_t.sum() / s_p.sum()
        idx = rng.integers(0, n, size=(n_boot, n))
        bd = s_t[idx].sum(1) / s_p[idx].sum(1) - a_t[idx].sum(1) / a_p[idx].sum(1)
        return {**r, "act_tov_per_poss": act, "sim_tov_per_poss": sim, "diff": sim - act,
                "lo95": float(np.quantile(bd, 0.025)), "hi95": float(np.quantile(bd, 0.975)),
                "act_tov_game": a_t.mean(), "sim_tov_game": s_t.mean(),
                "sim_tov_per_engine_poss": float(s_t.sum() / (2.0 * x["sim_engine_poss"].sum())),
                "note": "UNDERPOWERED" if n < MIN_GAMES else ""}

    for _, _, name in BUCKETS:
        rows.append(one(d[d["bucket"] == name], name))
    rows.append(one(d, "season-to-date"))
    return pd.DataFrame(rows)


def run_tov_monitor(led: pd.DataFrame, root: Path, season: int, season_start, team_box: pd.DataFrame | None = None,
                    n_boot: int = 500) -> dict:
    """led: grade ledger rows (game_id, game_date, run_id). Returns {'table', 'games', 'n_missing_sim', 'n_missing_actual'} or {'skipped': why}."""
    if not len(led):
        return {"skipped": "empty ledger"}
    if team_box is None:
        p = Path(__file__).resolve().parents[3] / f"data/raw/hoopr/team_box/team_box_{int(season)}.parquet"
        if not p.exists():
            return {"skipped": f"no team_box for {season}"}
        team_box = pd.read_parquet(p, columns=["game_id", "turnovers", "field_goals_attempted", "offensive_rebounds", "free_throws_attempted"])
    l_ = led[["game_id", "game_date", "run_id"]].drop_duplicates(["game_id", "run_id"]).copy()
    l_["game_id"] = l_["game_id"].astype("int64")
    sim = sim_game_tov(root, l_)
    act = actual_game_tov(team_box, l_["game_id"])
    d = l_.merge(sim, on=["game_id", "run_id"], how="inner").merge(act, on="game_id", how="inner")
    d["days"] = (pd.to_datetime(d["game_date"]).dt.normalize() - pd.Timestamp(season_start).normalize()).dt.days
    d["bucket"] = bucket_of(d["days"])
    return {"table": tov_level_table(d, n_boot), "games": d, "n_missing_sim": int(len(l_) - len(l_.merge(sim, on=["game_id", "run_id"]))),
            "n_missing_actual": int(len(l_) - len(l_.merge(act, on="game_id")))}


def report_block(res: dict, season_start) -> list[str]:
    """Markdown lines for the daily grade report. REPORT ONLY."""
    head = ["## TOV level monitor (report only; no output adjustment)", ""]
    if "skipped" in res:
        return head + [f"- skipped: {res['skipped']}", ""]
    t = res["table"].copy()
    lines = head + [f"Season-to-date league turnovers per possession, actual (verified hoopR box) vs the engine's prediction, by days since season start "
                    f"({pd.Timestamp(season_start).date()}). Possessions = FGA - OREB + TOV + {FTA_W} FTA on both sides. diff = sim - actual; "
                    f"95% interval = game bootstrap. Games without a sim row: {res['n_missing_sim']}; without an actual box: {res['n_missing_actual']}.", ""]
    show = t.copy()
    for c in ("act_tov_per_poss", "sim_tov_per_poss", "diff", "lo95", "hi95", "sim_tov_per_engine_poss"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.4f}" if c in ("diff", "lo95", "hi95") else f"{v:.4f}")
    for c in ("act_tov_game", "sim_tov_game"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    cols = ["bucket", "n_games", "act_tov_per_poss", "sim_tov_per_poss", "diff", "lo95", "hi95", "act_tov_game", "sim_tov_game",
            "sim_tov_per_engine_poss", "note"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in show.iterrows()]
    return lines + [""]
