"""
features.py -- the Control engine's pregame feature table.

One row per TEAM-GAME (both sides of every D-I, non-truncated game), carrying
the offence side's own pregame ratings, its opponent's, the site term, and the
day-of-season term -- plus the observed box targets the component models are
fitted against. A parallel one-row-per-GAME table carries the symmetric pace
features.

EVERY feature is pregame by construction:

  own ridge ratings   `cbb_sim.ratings.own_ratings`, joined on
                      as_of_date == the game's own date. The as-of table is
                      BUILT from games strictly before that date, so a game's
                      own result cannot influence its own features. Leak-tested
                      (change-form |corr| 0.05 / 0.06 / 0.00, gate 0.15) in
                      `data/processed/ratings/own_ratings_leak_test.csv`.
  centred KenPom      `data/processed/kenpom_snapshots.parquet` joined with
                      `pd.merge_asof(..., allow_exact_matches=False)`, the bulk
                      equivalent of `cbb_sim.data.kenpom.as_of`'s strictly-
                      BEFORE semantics (KenPom re-scrapes the morning after
                      games, so a same-day snapshot already contains the
                      result). Only the centred columns adj_o_c / adj_d_c /
                      adj_t_rel are used -- CLAUDE.md bans raw levels.
  site                site_home / site_away, neutral-site is the reference
                      level. Home/away/neutral is a first-class feature in
                      every component model (CLAUDE.md modelling rule 2).
  days_since_start    calendar days since that season's first D-I game date.

ANCHOR BUNDLES (the pre-registered L1 arms, run here for free):
  A_own   own ridge ratings only
  B_kp    centred KenPom only
  C_both  both
Each bundle always includes the site and day-of-season terms.

TEAM<->KENPOM KEY. hoopR `team_location` normalised with
`cbb_sim.data.kenpom.normalize_join_key`, with a per-season fallback to any
other spelling the same `team_id` carries in another season's team_box (hoopR
renames teams mid-history: "Massachusetts" -> "UMass", "IUPUI" -> "IU
Indianapolis"). This resolves 100% of D-I team-seasons 2022-2026; a team that
still failed to match would get NaN KenPom columns, never a fabricated value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.data.kenpom import normalize_join_key
from cbb_sim.ratings import own_ratings as orat

DEFAULT_HOOPR_DIR = Path("data/raw/hoopr")
DEFAULT_KENPOM = Path("data/processed/kenpom_snapshots.parquet")

KP_COLS: tuple[str, ...] = ("adj_o_c", "adj_d_c", "adj_t_rel")
OWN_COLS: tuple[str, ...] = ("off_c", "def_c", "tempo_rel")

SITE_COLS: tuple[str, ...] = ("site_home", "site_away")
DAY_COL = "days_since_start"

# --- anchor bundles, team-game (offence) models ----------------------------
_OWN_TEAM = ("own_off_c", "own_def_c", "own_tempo_rel",
             "opp_own_off_c", "opp_own_def_c", "opp_own_tempo_rel")
_KP_TEAM = ("own_kp_adj_o_c", "own_kp_adj_d_c", "own_kp_adj_t_rel",
            "opp_kp_adj_o_c", "opp_kp_adj_d_c", "opp_kp_adj_t_rel")

ANCHOR_TEAM_FEATURES: dict[str, tuple[str, ...]] = {
    "A_own": (*_OWN_TEAM, *SITE_COLS, DAY_COL),
    "B_kp": (*_KP_TEAM, *SITE_COLS, DAY_COL),
    "C_both": (*_OWN_TEAM, *_KP_TEAM, *SITE_COLS, DAY_COL),
}

# --- anchor bundles, game-level pace model --------------------------------
# The pace model is symmetric in the two teams by construction (sum features):
# one pace realisation per game, both teams scaled by it (CLAUDE.md modelling
# rule 3), and the own-ratings tempo model is itself additive with equal team
# weights, so a sum is the consistent form. The site term enters as the
# neutral indicator, the only site distinction a game-level quantity has.
_OWN_PACE = ("pace_own_tempo_sum", "pace_own_off_sum", "pace_own_def_sum")
_KP_PACE = ("pace_kp_t_sum", "pace_kp_o_sum", "pace_kp_d_sum")

ANCHOR_PACE_FEATURES: dict[str, tuple[str, ...]] = {
    "A_own": (*_OWN_PACE, "neutral", DAY_COL),
    "B_kp": (*_KP_PACE, "neutral", DAY_COL),
    "C_both": (*_OWN_PACE, *_KP_PACE, "neutral", DAY_COL),
}

ANCHORS: tuple[str, ...] = ("A_own", "B_kp", "C_both")


# ---------------------------------------------------------------------------
# KenPom as-of join
# ---------------------------------------------------------------------------
def build_kenpom_key_map(seasons: list[int], hoopr_dir: Path | str, kp: pd.DataFrame) -> pd.DataFrame:
    """(season, team_id) -> normalised KenPom join key.

    Prefers that season's own hoopR `team_location` spelling; falls back to any
    other spelling the same team_id carries in another season, because hoopR
    renames teams mid-history while the KenPom weekly files keep the name of
    the day.
    """
    rows = []
    for season in seasons:
        p = Path(hoopr_dir) / "team_box" / f"team_box_{int(season)}.parquet"
        tb = pd.read_parquet(p, columns=["team_id", "team_location"]).drop_duplicates("team_id")
        tb["team_id"] = pd.to_numeric(tb["team_id"], errors="coerce").astype("int64")
        tb["season"] = int(season)
        rows.append(tb)
    tb = pd.concat(rows, ignore_index=True)
    tb["key"] = tb["team_location"].map(normalize_join_key)

    kp = kp.copy()
    kp["key"] = kp["team"].map(normalize_join_key)
    season_keys = {int(s): set(d["key"]) for s, d in kp.groupby("season")}
    candidates = tb.groupby("team_id")["key"].apply(lambda x: list(dict.fromkeys(x))).to_dict()

    out = []
    for r in tb.itertuples():
        ks = season_keys.get(int(r.season), set())
        key = r.key if r.key in ks else next((c for c in candidates.get(r.team_id, []) if c in ks), None)
        out.append({"season": int(r.season), "team_id": int(r.team_id), "kp_key": key})
    return pd.DataFrame(out)


def join_kenpom_as_of(
    panel: pd.DataFrame,
    kp: pd.DataFrame,
    key_map: pd.DataFrame,
    team_col: str,
    date_col: str = "game_date",
    prefix: str = "own_kp_",
) -> pd.DataFrame:
    """Attach the latest KenPom snapshot strictly BEFORE `date_col` for the
    team in `team_col`, matched within the same season."""
    g = panel.copy()
    g["_row"] = np.arange(len(g))
    g = g.merge(
        key_map.rename(columns={"team_id": team_col, "kp_key": "_kp_key"}),
        on=["season", team_col], how="left",
    )
    g["_d"] = pd.to_datetime(g[date_col]).astype("datetime64[ns]")

    s = kp.dropna(subset=["team"]).copy()
    s["_kp_key"] = s["team"].map(normalize_join_key)
    s["_d"] = pd.to_datetime(s["snapshot_date"]).astype("datetime64[ns]")
    s = s[["_kp_key", "_d", *KP_COLS]].sort_values("_d", kind="mergesort")

    left = g.sort_values("_d", kind="mergesort")
    # `by` is the team key only, NOT (team, season): this matches
    # `cbb_sim.data.kenpom.join_as_of`'s own semantics, so a game played before
    # that season's first snapshot falls back to the previous season's last
    # snapshot -- still strictly pregame, and comparable because every snapshot
    # column is centred on its OWN snapshot's league mean.
    merged = pd.merge_asof(
        left, s, on="_d", by="_kp_key",
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("_row", kind="mergesort").drop(columns=["_row", "_d", "_kp_key"])
    return merged.rename(columns={c: f"{prefix}{c}" for c in KP_COLS}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------
def build_team_game_features(
    seasons: list[int],
    universe: pd.DataFrame | None = None,
    hoopr_dir: Path | str = DEFAULT_HOOPR_DIR,
    ratings_dir: Path | str = orat.DEFAULT_OUT_DIR,
    kenpom_path: Path | str = DEFAULT_KENPOM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (team_game, game) feature+target tables for `seasons`."""
    if universe is None:
        universe = orat.load_universe()
    tg = orat.load_team_games(universe, seasons, hoopr_dir)

    # ---- targets -----------------------------------------------------------
    tg["fg2a"] = tg["fga"] - tg["tpa"]
    tg["fg2m"] = tg["fgm"] - tg["tpm"]
    tg["points_check"] = 3 * tg["tpm"] + 2 * tg["fg2m"] + tg["ftm"]

    # ---- day of season -----------------------------------------------------
    start = tg.groupby("season")["game_date"].transform("min")
    tg[DAY_COL] = (tg["game_date"] - start).dt.days.astype("float64")

    # ---- own ridge ratings, as-of -----------------------------------------
    ratings = orat.load_ratings(sorted(set(seasons)), ratings_dir)
    tg = orat.join_as_of(tg, ratings, team_col="team_id", suffix="", cols=orat.RATING_COLS)
    tg = tg.rename(columns={c: f"own_{c}" for c in OWN_COLS} | {"n_games": "own_n_games"})
    opp_r = ratings[["season", "as_of_date", "team_id", *OWN_COLS, "n_games"]].rename(
        columns={c: f"opp_own_{c}" for c in OWN_COLS} | {"n_games": "opp_own_n_games"}
    )
    tg = tg.merge(
        opp_r.rename(columns={"team_id": "opp_team_id", "as_of_date": "game_date"}),
        on=["season", "opp_team_id", "game_date"], how="left",
    )

    # ---- centred KenPom, as-of --------------------------------------------
    kp = pd.read_parquet(kenpom_path)
    key_map = build_kenpom_key_map(sorted(set(seasons)), hoopr_dir, kp)
    tg = join_kenpom_as_of(tg, kp, key_map, team_col="team_id", prefix="own_kp_")
    tg = join_kenpom_as_of(tg, kp, key_map, team_col="opp_team_id", prefix="opp_kp_")

    # ---- game-level (pace) table ------------------------------------------
    home = tg[tg["team_id"] == tg["home_team_id"]]
    away = tg[tg["team_id"] == tg["away_team_id"]]
    hcols = ["game_id", "own_off_c", "own_def_c", "own_tempo_rel",
             "own_kp_adj_o_c", "own_kp_adj_d_c", "own_kp_adj_t_rel"]
    g = home[["game_id", "season", "game_date", "tipoff_utc", "neutral", DAY_COL,
              "game_poss", "home_team_id", "away_team_id", "n_periods", "cbbd_game_id",
              "team_score", "opp_score"]].rename(columns={"team_score": "home_score", "opp_score": "away_score"})
    g = g.merge(home[hcols].rename(columns={c: f"h_{c}" for c in hcols[1:]}), on="game_id")
    g = g.merge(away[hcols].rename(columns={c: f"a_{c}" for c in hcols[1:]}), on="game_id")

    g["pace_own_tempo_sum"] = g["h_own_tempo_rel"] + g["a_own_tempo_rel"]
    g["pace_own_off_sum"] = g["h_own_off_c"] + g["a_own_off_c"]
    g["pace_own_def_sum"] = g["h_own_def_c"] + g["a_own_def_c"]
    g["pace_kp_t_sum"] = g["h_own_kp_adj_t_rel"] + g["a_own_kp_adj_t_rel"]
    g["pace_kp_o_sum"] = g["h_own_kp_adj_o_c"] + g["a_own_kp_adj_o_c"]
    g["pace_kp_d_sum"] = g["h_own_kp_adj_d_c"] + g["a_own_kp_adj_d_c"]
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    g["month"] = pd.to_datetime(g["game_date"]).dt.month

    return tg.reset_index(drop=True), g.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fold helpers
# ---------------------------------------------------------------------------
FOLDS: dict[str, dict[str, list[int]]] = {
    "F1": {"train": [2022, 2023], "test": [2024]},
    "F2": {"train": [2022, 2023, 2024], "test": [2025]},
}


def fold_seasons(fold: str) -> tuple[list[int], list[int]]:
    f = FOLDS[fold]
    return list(f["train"]), list(f["test"])
