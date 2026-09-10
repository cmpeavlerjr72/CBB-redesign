"""
rotation.py -- L4 rotation model: who is on the floor, and for how long.

What the engine needs
---------------------
At every possession the possession engine needs the five players on the floor
for each team. `docs/SIM_GUARDRAILS.md` G8 grades the result on per-player
minutes (mean and SD), share of team minutes by top-1/top-3, players with > 0
minutes, and is broken out by final-margin band because "bench minutes in
blowouts affect props far more than game markets". This module is the
stochastic process that produces those five-player sets, plus the as-of feature
builder it consumes and the four bake-off arms of
`docs/models/rotation/experiments.md`.

Id space
--------
Everything here is in **CBBD player-id space**, because that is the only id
space the on-floor data exists in (`on_floor_h1..h5` / `on_floor_a1..a5` on
`data/processed/possessions/possessions_{season}.parquet`, 2024+ only, L13).
`cbb_sim.data.player_ids` bridges to ESPN ids for the hoopR box-score inputs
(`did_not_play`, `starter`, `minutes`) and for anything downstream that keys on
`athlete_id`.

Minutes accounting
------------------
A player's minutes in a game are the summed `duration_s` of the possessions he
is on the floor for. The on-floor set is recorded at the possession's *first
event*, so a substitution inside a possession is charged to the possession
boundary. Measured against hoopR box minutes on 2025 (via the crosswalk):
corr 0.984, mean diff +0.14 min, MAD 0.63 min. Sim and actual are computed the
same way throughout, so every comparison is like-for-like.

The four arms
-------------
R1 `dirichlet`        single-level Dirichlet over the as-of minute-share profile
                      (with the extend_profile tail fix), fitted concentration,
                      + the shared deterministic scheduler.
R2 `hier_dirichlet`   two-level Dirichlet (starters family vs bench family, then
                      within-family; mean share preserved exactly at each level,
                      the `usage_alloc.py` design of
                      `docs/postmortem/05_cfb_methodology_extract.md` section 6)
                      + the same scheduler.
R3 `stint_hazard`     per-possession logistic exit hazard (on floor) and entry
                      score (bench) conditioned on game state and the pregame
                      target.
R4 `stint_resample`   empirical stint-sequence resampling from the team's own
                      last k = 5 games with state-conditioned splicing.

Garbage time, from data rather than a rule
------------------------------------------
Every arm has to reproduce the measured collapse of starter minutes in a
blowout (2025 actual, final 8:00 of regulation: starters take 74.5% of the
five on-floor slots at |margin| <= 5, 71.8% at 6-15, and 50.6% at > 15):

* R1/R2  a **fitted state-tilt lookup table**, `TiltTables.state`, estimated on
  the training season only: the share of a team's on-floor slot-seconds taken
  by each as-of minutes-rank bucket in each (time bucket x margin bucket) cell,
  expressed as a multiplier on the baseline cell. The scheduler re-targets the
  remaining slot-seconds through that multiplier every possession, so bench
  players' remaining-minutes need rises automatically once the margin opens.
  Nothing about the multiplier is chosen by hand; it is a lookup table read off
  2024 (CLAUDE.md: "Sim loop uses lookup tables ... never live model calls").
* R3      intrinsic: `abs(margin)`, `margin`, seconds remaining and period are
  features of both hazards, so the fitted coefficients carry the blowout
  behaviour directly.
* R4      intrinsic: the splice rule picks, for each state block of the game
  being simulated, the donor game (from the team's own last five) that spent
  the most possessions in that same state band, so a blowout stretch is played
  back from a real blowout stretch of that team's own season.

Foul trouble is handled the same way: `TiltTables.foul` is the fitted
P(on floor | fouls = f, time bucket) / P(on floor | fouls <= 1, same bucket)
ratio on the training season for R1/R2, a feature for R3, and implicit in the
donor sequence for R4. Every arm simulates its own foul process from the
player's as-of fouls-per-minute, because "each player's current fouls" is a
pre-registered game-state input and reading the real foul events of the game
being simulated would be a leak.

RNG
---
`cbb_sim.control.rng.stream_keys(seed, [game_id], "rotation")` gives the
game's 64-bit key exactly as the Control and the pace sampler do, so paired
arms difference game by game and dropping a game moves no other game's draws.
A rotation game needs a few thousand *sequential* draws (a Dirichlet vector,
then a foul draw per on-floor player per possession), which the counter-based
indexed form is not shaped for -- it vectorises one draw index across many
games. So the counter-based key is expanded into that sequential stream by a
PCG64 seeded from the key itself. The (seed, game_id, "rotation") contract is
unchanged; only the expansion differs, and `tests/test_rotation.py` pins it.
Home is drawn before away from the same game stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.control import rng as crng

DEFAULT_POSS_DIR = Path("data/processed/possessions")
DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_PBP_DIR = Path("data/raw/cbbd/pbp")
DEFAULT_PLAYER_BOX_DIR = Path("data/raw/hoopr/player_box")

ON_FLOOR_H = [f"on_floor_h{i}" for i in range(1, 6)]
ON_FLOOR_A = [f"on_floor_a{i}" for i in range(1, 6)]
SLOTS = [f"p{i}" for i in range(1, 6)]

RNG_FAMILY = "rotation"

#: time buckets: 0 = 1st half, 1 = 2nd half > 8:00 left, 2 = 2nd half 8:00-2:00,
#: 3 = 2nd half final 2:00, 4 = overtime
N_TIME_BUCKETS = 5
#: margin buckets: 0 = |m| <= 5, 1 = 6-15, 2 = > 15   (the G8 blowout bands)
N_MARGIN_BUCKETS = 3
#: as-of minutes-rank buckets 1..9 individually, 10+ collapsed
N_RANK_BUCKETS = 10
#: foul states 0..5 (5 == fouled out)
N_FOUL_STATES = 6

FOUL_OUT = 5
MAX_CANDIDATES = 15


# ===========================================================================
# 1. Observed layer: team-possessions, minutes, starters, fouls
# ===========================================================================
def time_bucket(period: np.ndarray, start_clock: np.ndarray) -> np.ndarray:
    period = np.asarray(period)
    clock = np.asarray(start_clock)
    tb = np.zeros(len(period), dtype="int64")
    tb[(period == 2) & (clock > 480)] = 1
    tb[(period == 2) & (clock <= 480) & (clock > 120)] = 2
    tb[(period == 2) & (clock <= 120)] = 3
    tb[period >= 3] = 4
    return tb


def margin_bucket(margin: np.ndarray) -> np.ndarray:
    a = np.abs(np.asarray(margin))
    return np.where(a <= 5, 0, np.where(a <= 15, 1, 2)).astype("int64")


def rank_bucket(rank: np.ndarray) -> np.ndarray:
    r = np.asarray(rank)
    return np.clip(r, 1, N_RANK_BUCKETS).astype("int64") - 1


def load_team_possessions(
    season: int,
    poss_dir: Path = DEFAULT_POSS_DIR,
    universe_path: Path = DEFAULT_UNIVERSE,
    complete_only: bool = True,
) -> pd.DataFrame:
    """One row per (game_id, team_id, poss_index): the five on-floor CBBD player
    ids for that team plus the game state at the possession's start.

    `complete_only` keeps only games whose on-floor columns are complete on
    every possession (95.1% of 2025 games; the pre-registration's "2025 games
    with complete on-floor data").
    """
    cols = ["game_id", "cbbd_game_id", "season", "period", "poss_index", "duration_s",
            "start_clock", "start_score_diff", "offense_is_home"] + ON_FLOOR_H + ON_FLOOR_A
    p = pd.read_parquet(Path(poss_dir) / f"possessions_{season}.parquet", columns=cols)
    gu = pd.read_parquet(universe_path)
    gu = gu[gu["season"] == int(season)][["game_id", "home_team_id", "away_team_id",
                                          "game_date", "home_score", "away_score"]]
    p = p.merge(gu, on="game_id", how="inner")

    complete = p[ON_FLOOR_H + ON_FLOOR_A].notna().all(axis=1)
    if complete_only:
        frac = p.assign(_c=complete).groupby("game_id")["_c"].transform("mean")
        p = p[frac == 1.0]
    else:
        p = p[complete]

    p = p.copy()
    p["home_margin"] = np.where(p["offense_is_home"], p["start_score_diff"], -p["start_score_diff"])
    p["final_margin_home"] = p["home_score"] - p["away_score"]

    frames = []
    for cols_side, team_col, is_home in ((ON_FLOOR_H, "home_team_id", True),
                                         (ON_FLOOR_A, "away_team_id", False)):
        d = pd.DataFrame({
            "game_id": p["game_id"].to_numpy(),
            "season": p["season"].to_numpy(),
            "game_date": p["game_date"].to_numpy(),
            "team_id": p[team_col].to_numpy(),
            "is_home": is_home,
            "period": p["period"].to_numpy(),
            "poss_index": p["poss_index"].to_numpy(),
            "duration_s": p["duration_s"].to_numpy().astype("float64"),
            "start_clock": p["start_clock"].to_numpy(),
            "margin": p["home_margin"].to_numpy() * (1 if is_home else -1),
            "final_margin": p["final_margin_home"].to_numpy() * (1 if is_home else -1),
        })
        for k, c in enumerate(cols_side):
            d[SLOTS[k]] = p[c].to_numpy().astype("int64")
        frames.append(d)
    tp = pd.concat(frames, ignore_index=True)
    tp["time_bucket"] = time_bucket(tp["period"].to_numpy(), tp["start_clock"].to_numpy())
    tp["margin_bucket"] = margin_bucket(tp["margin"].to_numpy())
    return tp.sort_values(["game_id", "team_id", "poss_index"]).reset_index(drop=True)


def player_game_minutes(tp: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team_id, pid) -> seconds/minutes on floor, starter flag, minutes rank."""
    long = tp.melt(
        id_vars=["game_id", "season", "game_date", "team_id", "duration_s"],
        value_vars=SLOTS, value_name="pid",
    )
    pg = long.groupby(["game_id", "season", "game_date", "team_id", "pid"],
                      as_index=False)["duration_s"].sum()
    pg["minutes"] = pg["duration_s"] / 60.0

    first = (tp[tp["period"] == 1]
             .sort_values(["game_id", "team_id", "poss_index"])
             .groupby(["game_id", "team_id"], as_index=False).head(1))
    st = first.melt(id_vars=["game_id", "team_id"], value_vars=SLOTS, value_name="pid")
    st = st[["game_id", "team_id", "pid"]].drop_duplicates()
    st["is_starter"] = True
    pg = pg.merge(st, on=["game_id", "team_id", "pid"], how="left")
    pg["is_starter"] = pg["is_starter"].fillna(False).astype(bool)

    pg = pg.sort_values(["game_id", "team_id", "minutes"], ascending=[True, True, False])
    pg["minutes_rank"] = pg.groupby(["game_id", "team_id"]).cumcount() + 1
    return pg.reset_index(drop=True)


def player_game_fouls(season: int, pbp_dir: Path = DEFAULT_PBP_DIR) -> pd.DataFrame:
    """One row per personal foul: (cbbd_game_id, pid, period, seconds_remaining).

    99.93% of CBBD `PersonalFoul` rows carry a participant id in 2025.
    """
    df = pd.read_parquet(
        Path(pbp_dir) / f"plays_{season}.parquet",
        columns=["gameId", "id", "playType", "period", "secondsRemaining", "participant_1_id"],
    ).drop_duplicates(subset=["gameId", "id"])
    f = df[(df["playType"] == "PersonalFoul") & df["participant_1_id"].notna()].copy()
    f = f.rename(columns={"gameId": "cbbd_game_id", "participant_1_id": "pid",
                          "secondsRemaining": "start_clock"})
    f["pid"] = f["pid"].astype("int64")
    return f[["cbbd_game_id", "pid", "period", "start_clock"]]


# ===========================================================================
# 2. As-of pregame features (leak-safe by construction)
# ===========================================================================
#: decays offered to the fitted recency-weighted starter predictor
START_EWMA_ALPHAS = (0.15, 0.30, 0.50, 0.80)

ASOF_FEATURES = [
    "mpg_asof_raw", "mpg_asof", "share_asof", "games_played_asof", "team_games_asof",
    "start_freq_asof", "start_ewma_15", "start_ewma_30", "start_ewma_50", "start_ewma_80",
    "dnp_rate_asof", "last_game_dnp", "fouls_per_min_asof",
    "rotation_depth_asof", "minutes_rank_asof",
]


def build_asof_player_features(
    pg: pd.DataFrame,
    fouls: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    dnp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per (game_id, team_id, pid) as-of features from strictly earlier games.

    Every column is a `shift(1)` expanding statistic over the team's own game
    order, so a game's own box score can never enter its own features. The
    candidate pool for a game is exactly the set of players with at least one
    *earlier* appearance for that team in that season -- a player who joins the
    team in January is invisible to a November game by construction, which is
    why the "too short" tail is repaired with anonymous fitted tail slots
    (`extend_profile`) rather than with named future roster members.
    """
    pg = pg.copy()
    pg["game_date"] = pd.to_datetime(pg["game_date"])

    # the full (team-game x player-who-ever-appears-for-the-team) grid
    team_games = (pg[["team_id", "game_id", "game_date", "season"]]
                  .drop_duplicates()
                  .sort_values(["team_id", "game_date", "game_id"]))
    team_games["team_game_idx"] = team_games.groupby("team_id").cumcount()
    team_players = pg[["team_id", "pid"]].drop_duplicates()
    grid = team_games.merge(team_players, on="team_id", how="left")

    obs = pg[["game_id", "team_id", "pid", "minutes", "is_starter"]]
    grid = grid.merge(obs, on=["game_id", "team_id", "pid"], how="left")
    grid["minutes"] = grid["minutes"].fillna(0.0)
    grid["is_starter"] = grid["is_starter"].fillna(False).astype(float)
    grid["played"] = (grid["minutes"] > 0).astype(float)
    grid["dnp"] = 1.0 - grid["played"]

    if fouls is not None and len(fouls):
        fc = fouls.groupby(["game_id", "team_id", "pid"], as_index=False).size()
        fc = fc.rename(columns={"size": "fouls"})
        grid = grid.merge(fc, on=["game_id", "team_id", "pid"], how="left")
        grid["fouls"] = grid["fouls"].fillna(0.0)
    else:
        grid["fouls"] = 0.0

    grid = grid.sort_values(["team_id", "pid", "team_game_idx"])
    g = grid.groupby(["team_id", "pid"], sort=False)
    for src, dst in (("minutes", "cum_min"), ("played", "cum_played"),
                     ("is_starter", "cum_start"), ("dnp", "cum_dnp"),
                     ("fouls", "cum_fouls")):
        grid[dst] = g[src].cumsum().shift(1)
        grid.loc[g.cumcount() == 0, dst] = 0.0
    grid["prior_team_games"] = g.cumcount().astype(float)
    grid["last_game_dnp"] = g["dnp"].shift(1).fillna(0.0)

    grid = grid[grid["cum_played"] > 0].copy()  # candidate pool = has appeared before
    grid["games_played_asof"] = grid["cum_played"]
    grid["team_games_asof"] = grid["prior_team_games"]
    grid["mpg_asof_raw"] = grid["cum_min"] / grid["prior_team_games"].clip(lower=1)
    grid["start_freq_asof"] = grid["cum_start"] / grid["prior_team_games"].clip(lower=1)
    # Recency-weighted starter frequency. The flat expanding mean is a poor
    # predictor of tonight's starting five when a coach has changed it: it costs
    # roughly one starter in five, and the state-dependence metric counts the
    # five the model started, so a wrong fifth man moves the late-window
    # starters' share by ~7-10 pp on its own.
    gs = grid.groupby(["team_id", "pid"], sort=False)["is_starter"]
    for a in START_EWMA_ALPHAS:
        grid[f"start_ewma_{int(a * 100)}"] = (
            gs.transform(lambda x, a=a: x.shift(1).ewm(alpha=a, adjust=True).mean())
            .fillna(0.0)
        )
    grid["dnp_rate_asof"] = grid["cum_dnp"] / grid["prior_team_games"].clip(lower=1)
    grid["cum_fouls_asof"] = grid["cum_fouls"]
    grid["cum_minutes_asof"] = grid["cum_min"]
    # RAW ratio, kept for reporting only. The *modelling* foul rate is the
    # shrunk one built in `build_priors` from the two cumulative counts: an
    # unshrunk ratio gives a player with 1 prior minute and 1 foul a rate of
    # 1.0 fouls/minute, which the dev smoke run showed producing ~5x too many
    # four-foul players.
    grid["fouls_per_min_asof"] = grid["cum_fouls"] / grid["cum_min"].clip(lower=1.0)

    grid = grid.sort_values(["game_id", "team_id", "mpg_asof_raw"], ascending=[True, True, False])
    grid["minutes_rank_asof"] = grid.groupby(["game_id", "team_id"]).cumcount() + 1
    depth = (grid.assign(_r=(grid["mpg_asof_raw"] >= 10.0).astype(int))
             .groupby(["game_id", "team_id"])["_r"].transform("sum"))
    grid["rotation_depth_asof"] = depth

    # The contemporaneous columns the expanding sums were built from are DROPPED
    # here: `minutes`, `is_starter`, `played`, `dnp` and `fouls` are that game's
    # own box score, and leaving them on the returned frame is how a downstream
    # consumer accidentally trains on the answer. `tests/test_rotation.py`
    # asserts they are absent and that every remaining column is invariant to
    # corrupting the game's own rows.
    drop = ["minutes", "is_starter", "played", "dnp", "fouls", "cum_min", "cum_played",
            "cum_start", "cum_dnp", "cum_fouls", "prior_team_games"]
    grid = grid.drop(columns=[c for c in drop if c in grid.columns])
    return grid.reset_index(drop=True)


def attach_hoopr_availability(
    feats: pd.DataFrame, season: int, crosswalk: pd.DataFrame,
    pg: pd.DataFrame | None = None,
    player_box_dir: Path = DEFAULT_PLAYER_BOX_DIR,
) -> pd.DataFrame:
    """Join hoopR `did_not_play` / `starter` through the CBBD<->ESPN crosswalk.

    The as-of DNP history in `build_asof_player_features` is derived from the
    CBBD on-floor stream (zero on-floor possessions == did not play). This
    attaches the hoopR field the guardrails name as the historical availability
    source, as-of the same way, and reports coverage; where the crosswalk or the
    box row is missing, the CBBD-derived value stands.
    """
    from cbb_sim.data.player_ids import cbbd_to_espn_map

    m = cbbd_to_espn_map(crosswalk, season)
    pb = pd.read_parquet(Path(player_box_dir) / f"player_box_{season}.parquet",
                         columns=["game_id", "athlete_id", "team_id", "did_not_play", "starter"])
    pb = pb[pb["athlete_id"].notna()]
    pb = pb.rename(columns={"athlete_id": "espn_id"})
    out = feats.copy()
    out["espn_id"] = out["pid"].map(m)
    out = out.merge(pb, on=["game_id", "team_id", "espn_id"], how="left",
                    suffixes=("", "_hoopr"))
    out["hoopr_join_ok"] = out["did_not_play"].notna()
    if pg is not None:
        obs = pg[["game_id", "team_id", "pid", "minutes"]].rename(
            columns={"minutes": "onfloor_minutes"})
        out = out.merge(obs, on=["game_id", "team_id", "pid"], how="left")
        out["onfloor_minutes"] = out["onfloor_minutes"].fillna(0.0)
    return out


# ===========================================================================
# 3. Fitted objects
# ===========================================================================
@dataclass
class TiltTables:
    """State-dependence lookup tables, fitted on the training season only."""
    #: [rank_bucket, time_bucket, margin_bucket] -> multiplier vs the baseline cell
    state: np.ndarray
    #: [foul_state, time_bucket] -> multiplier vs the fouls <= 1 cell
    foul: np.ndarray

    def to_dict(self) -> dict:
        return {"state": self.state.tolist(), "foul": self.foul.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "TiltTables":
        return TiltTables(np.asarray(d["state"], dtype="float64"),
                          np.asarray(d["foul"], dtype="float64"))


@dataclass
class RotationFit:
    """Everything fitted on the training season. Written to
    `data/processed/models/rotation/rotation_fit_{arm}.json`."""
    season_train: int
    role_prior: list[float] = field(default_factory=list)   # mean minutes by as-of rank
    k0: float = 3.0                                          # shrinkage weight
    w_dnp: float = 1.0                                       # last-game-DNP multiplier
    tail_ratio: float = 0.6                                  # extend_profile decay
    n_profile: int = 13                                      # extend_profile target length
    alpha: float = 40.0                                      # R1 concentration
    alpha_family: float = 40.0                               # R2 between-family
    alpha_starters: float = 40.0                             # R2 within starters
    alpha_bench: float = 40.0                                # R2 within bench
    swap_threshold: float = 0.15                             # scheduler stickiness (share units)
    ema_horizon: float = 240.0                               # seconds, on-floor-rate tracker
    lam_deficit: float = 0.5                                 # weight on the long-run correction
    foul_rate_scale: float = 1.0
    p_play: list[float] = field(default_factory=list)        # P(plays at all) by as-of rank
    w_avail_dnp: float = 1.0                                 # last-game-DNP availability factor
    fpm_league: float = 0.085                                # league fouls per on-floor minute
    fpm_prior_min: float = 30.0                              # shrinkage strength for fpm
    min_share: float = 0.0                                   # availability share floor
    start_alpha: float = 0.30                                # starter-predictor EWMA decay
    tilt: TiltTables | None = None
    hazard_exit: dict | None = None
    hazard_enter: dict | None = None
    notes: dict = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        d = {k: v for k, v in self.__dict__.items() if k != "tilt"}
        d["tilt"] = self.tilt.to_dict() if self.tilt is not None else None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def from_json(path: Path) -> "RotationFit":
        d = json.loads(Path(path).read_text())
        tilt = TiltTables.from_dict(d.pop("tilt")) if d.get("tilt") else None
        fit = RotationFit(**{k: v for k, v in d.items() if k != "tilt"})
        fit.tilt = tilt
        return fit


def fit_role_prior(pg: pd.DataFrame, max_rank: int = MAX_CANDIDATES) -> list[float]:
    """Mean realised minutes by within-team-game minutes rank, on the train season.
    This is the prior an as-of average with few games is shrunk toward."""
    r = pg[pg["minutes_rank"] <= max_rank]
    mean_by_rank = r.groupby("minutes_rank")["minutes"].mean()
    out = [float(mean_by_rank.get(k, 0.0)) for k in range(1, max_rank + 1)]
    last = out[-1] if out[-1] > 0 else 1.0
    return [v if v > 0 else last for v in out]


def shrink_mpg(mpg_raw: np.ndarray, n_games: np.ndarray, rank: np.ndarray,
               role_prior: list[float], k0: float) -> np.ndarray:
    """(sum_minutes + k0 * role_prior[rank]) / (n_games + k0)."""
    prior = np.asarray([role_prior[min(int(r), len(role_prior)) - 1] for r in rank])
    n = np.asarray(n_games, dtype="float64")
    return (np.asarray(mpg_raw, dtype="float64") * n + k0 * prior) / (n + k0)


def fit_k0(feats: pd.DataFrame, pg: pd.DataFrame, role_prior: list[float],
           grid=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)) -> tuple[float, dict]:
    """Pick the shrinkage weight that best predicts the *next* game's minutes on
    the training season (MAE). Fitted, not assumed (L13: two seasons of lineup
    data, so shrinkage strength is a fitted parameter)."""
    truth = pg[["game_id", "team_id", "pid", "minutes"]]
    d = feats.merge(truth, on=["game_id", "team_id", "pid"], how="left")
    d["minutes"] = d["minutes"].fillna(0.0)
    scores = {}
    best, best_mae = grid[0], np.inf
    for k0 in grid:
        pred = shrink_mpg(d["mpg_asof_raw"].to_numpy(), d["team_games_asof"].to_numpy(),
                          d["minutes_rank_asof"].to_numpy(), role_prior, k0)
        mae = float(np.mean(np.abs(pred - d["minutes"].to_numpy())))
        scores[str(k0)] = mae
        if mae < best_mae:
            best, best_mae = k0, mae
    return float(best), scores


def fit_w_dnp(feats: pd.DataFrame, pg: pd.DataFrame, role_prior: list[float],
              k0: float) -> float:
    """Multiplicative correction to the as-of minutes of a player whose last game
    was a DNP, fitted as the ratio of realised to predicted minutes for that group."""
    truth = pg[["game_id", "team_id", "pid", "minutes"]]
    d = feats.merge(truth, on=["game_id", "team_id", "pid"], how="left")
    d["minutes"] = d["minutes"].fillna(0.0)
    pred = shrink_mpg(d["mpg_asof_raw"].to_numpy(), d["team_games_asof"].to_numpy(),
                      d["minutes_rank_asof"].to_numpy(), role_prior, k0)
    m = d["last_game_dnp"].to_numpy() > 0.5
    if m.sum() < 300 or pred[m].sum() <= 0:
        return 1.0
    return float(np.clip(d["minutes"].to_numpy()[m].sum() / pred[m].sum(), 0.05, 1.0))


def fit_min_share(pg: pd.DataFrame) -> float:
    """Mean share of team minutes taken by the *smallest* nonzero-minutes player
    in a team-game -- the floor `apply_min_target` gives an available player."""
    tot = pg.groupby(["game_id", "team_id"])["minutes"].transform("sum")
    sh = pg["minutes"] / tot
    return float(sh.groupby([pg["game_id"], pg["team_id"]]).min().mean())


def fit_tail(pg: pd.DataFrame) -> tuple[float, int]:
    """extend_profile parameters: the geometric decay of the minute-share tail and
    the target profile length, both read off the training season.

    The CFB defect this repairs ("too short": the raw share table sums to ~71% of
    the team's real usage because it drops the long tail) shows up here as an
    as-of candidate list that is systematically shorter than the ~9.8 players a
    team actually uses."""
    r = pg[pg["minutes_rank"].between(7, 13)]
    tot = pg.groupby(["game_id", "team_id"])["minutes"].transform("sum")
    sh = (pg["minutes"] / tot)
    by_rank = sh.groupby(pg["minutes_rank"]).mean()
    ratios = []
    for k in range(7, 13):
        a, b = by_rank.get(k), by_rank.get(k + 1)
        if a and b and a > 0:
            ratios.append(b / a)
    ratio = float(np.median(ratios)) if ratios else 0.6
    n_nonzero = pg.groupby(["game_id", "team_id"]).size()
    n_profile = int(np.ceil(n_nonzero.quantile(0.90)))
    return float(np.clip(ratio, 0.2, 0.95)), int(np.clip(n_profile, 8, MAX_CANDIDATES))


def extend_profile(share: np.ndarray, pids: np.ndarray, n_profile: int,
                   tail_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Blend a fitted geometric tail onto a short as-of share profile.

    Anchored on the profile's **own** smallest share (not a league mean), the
    way `usage_alloc.extend_profile` anchors on the team's own last real share.
    Tail slots get synthetic negative ids: they are fitted tail *mass*, not
    named players, because naming them would mean reading a roster the pregame
    feature set does not have.
    """
    share = np.asarray(share, dtype="float64")
    pids = np.asarray(pids, dtype="int64")
    if len(share) >= n_profile or len(share) == 0:
        s = share / share.sum() if share.sum() > 0 else share
        return s, pids
    anchor = float(share[-1])
    add = n_profile - len(share)
    tail = np.array([anchor * (tail_ratio ** (j + 1)) for j in range(add)], dtype="float64")
    tail_ids = np.array([-(j + 1) for j in range(add)], dtype="int64")
    out = np.concatenate([share, tail])
    return out / out.sum(), np.concatenate([pids, tail_ids])


def fit_dirichlet_alpha(pred_share: np.ndarray, real_share: np.ndarray,
                        weight: np.ndarray | None = None) -> float:
    """Method-of-moments concentration for Dirichlet(alpha * p):
    Var(s_i) = p_i (1 - p_i) / (alpha + 1)."""
    p = np.asarray(pred_share, dtype="float64")
    s = np.asarray(real_share, dtype="float64")
    keep = (p > 1e-6) & (p < 1 - 1e-6) & np.isfinite(p) & np.isfinite(s)
    if keep.sum() < 50:
        return 40.0
    p, s = p[keep], s[keep]
    var = np.mean((s - p) ** 2)
    if var <= 0:
        return 200.0
    alpha = float(np.mean(p * (1 - p)) / var - 1.0)
    return float(np.clip(alpha, 1.0, 500.0))


def fit_tilt_tables(tp: pd.DataFrame, feats: pd.DataFrame,
                    fouls_poss: pd.DataFrame | None = None) -> TiltTables:
    """The fitted state-dependence lookup tables (train season only).

    `state[rb, tb, mb]` is the share of on-floor slot-seconds taken by rank
    bucket `rb` in state cell `(tb, mb)`, divided by that bucket's
    exposure-weighted share over the whole game, so the tilt averages to 1 and
    re-shapes *when* a player's minutes fall without re-levelling *how many* he
    gets. Rank 1-5 are the five who started, ordered by as-of minutes; 6+ are
    the rest by as-of minutes -- the same construction the simulator uses, so
    the table is not attenuated by how often the as-of start prediction misses.

    `foul[f, tb]` is P(on floor | fouls == f) / P(on floor | fouls <= 1) in time
    bucket `tb`, stratified on **both** as-of rank bucket and minutes played so
    far and then pooled by exposure. Both controls are necessary: pooling
    outright says three fouls makes a player 3x MORE likely to be on the floor,
    and controlling for rank alone still says 1.6x, because within a rank the
    player who has three fouls in the second half is the one who has been
    playing all night. `fouls_poss` must therefore carry columns
    `rb`, `pf`, `fouls`, `time_bucket`, `num`, `den`.
    """
    # Rank key: the SAME construction the simulator uses -- the five who started
    # occupy positions 1-5 (ordered among themselves by as-of minutes), the rest
    # follow by as-of minutes. Keying on the as-of *predicted* start order
    # instead attenuates the table by however often the prediction misses, and
    # measured a late-close starter share of 0.72 against a real 0.78.
    first = (tp[tp["period"] == 1].sort_values(["game_id", "team_id", "poss_index"])
             .groupby(["game_id", "team_id"], as_index=False).head(1))
    st = first.melt(id_vars=["game_id", "team_id"], value_vars=SLOTS, value_name="pid")
    st = st[["game_id", "team_id", "pid"]].drop_duplicates()
    st["started"] = 1
    r = feats.merge(st, on=["game_id", "team_id", "pid"], how="left")
    r["started"] = r["started"].fillna(0).astype(int)
    r = r.sort_values(["game_id", "team_id", "started", "mpg_adj"],
                      ascending=[True, True, False, False])
    r["rank_key"] = r.groupby(["game_id", "team_id"]).cumcount() + 1
    rank = r.set_index(["game_id", "team_id", "pid"])["rank_key"]

    long = tp.melt(
        id_vars=["game_id", "team_id", "duration_s", "time_bucket", "margin_bucket"],
        value_vars=SLOTS, value_name="pid",
    )
    key = pd.MultiIndex.from_arrays([long["game_id"], long["team_id"], long["pid"]])
    long["rank"] = rank.reindex(key).to_numpy()
    long = long[np.isfinite(long["rank"])]
    long["rb"] = rank_bucket(long["rank"].to_numpy())

    tab = np.zeros((N_RANK_BUCKETS, N_TIME_BUCKETS, N_MARGIN_BUCKETS), dtype="float64")
    agg = long.groupby(["time_bucket", "margin_bucket", "rb"])["duration_s"].sum()
    for (tb, mb, rb), v in agg.items():
        tab[int(rb), int(tb), int(mb)] = float(v)
    cell_tot = tab.sum(axis=0, keepdims=True)
    share = np.divide(tab, cell_tot, out=np.zeros_like(tab), where=cell_tot > 0)
    # Normalised on the EXPOSURE-WEIGHTED mean share, not on one baseline cell,
    # so the average tilt over a game is 1: the drawn Dirichlet total is a game
    # total, and dividing by a single cell would silently re-level it.
    w = cell_tot[0] / cell_tot[0].sum()
    overall = (share * w[None, :, :]).sum(axis=(1, 2))
    overall[overall <= 0] = np.nan
    tilt = share / overall[:, None, None]
    tilt = np.nan_to_num(tilt, nan=1.0, posinf=1.0, neginf=1.0)
    tilt[:, cell_tot[0] <= 0] = 1.0
    tilt = np.clip(tilt, 0.05, 20.0)

    foul_tab = foul_tilt_from_event_study(fouls_poss)
    return TiltTables(tilt, foul_tab)


def foul_tilt_from_event_study(ev) -> np.ndarray:
    """`foul[f, tb]` from a within-player event study around each foul.

    `ev` carries, per (foul number f, time bucket of the foul), the pooled
    on-floor possessions and opportunities in the window *before* the foul and
    the window *after* it. `ratio[f, tb]` is the after/before on-floor rate --
    the coach's benching response to that specific foul -- and the cumulative
    tilt chains them, `tilt[f] = prod_{j <= f} ratio[j]`.

    A stratified estimate cannot identify this. Pooled, it says three fouls
    makes a player 3x MORE likely to be on the floor; controlling for as-of rank
    it still says 1.6x, and controlling for rank and minutes-played-so-far 1.4x
    -- because a foul count is itself a record of floor time. The event study
    compares a player with himself minutes apart, so it does not have that
    problem.
    """
    tab = np.ones((N_FOUL_STATES, N_TIME_BUCKETS), dtype="float64")
    if ev is None or not len(ev):
        tab[FOUL_OUT, :] = 0.0
        return tab
    g = ev.groupby(["fouls", "time_bucket"])[["before_on", "before_n",
                                              "after_on", "after_n"]].sum()
    ratio = np.ones((N_FOUL_STATES, N_TIME_BUCKETS), dtype="float64")
    for (f, tb), r in g.iterrows():
        f, tb = int(f), int(tb)
        if f < 1 or f >= N_FOUL_STATES or r["before_n"] < 200 or r["after_n"] < 200:
            continue
        b = r["before_on"] / r["before_n"]
        a = r["after_on"] / r["after_n"]
        if b > 0:
            ratio[f, tb] = float(np.clip(a / b, 0.02, 1.5))
    for tb in range(N_TIME_BUCKETS):
        acc = 1.0
        for f in range(1, N_FOUL_STATES):
            acc *= ratio[f, tb]
            tab[f, tb] = acc
    tab[FOUL_OUT, :] = 0.0
    return tab


# ===========================================================================
# 4. Game script + team prior (the two inputs a simulated team-game needs)
# ===========================================================================
@dataclass
class GameScript:
    """The realised possession sequence of one team-game. The rotation bake-off
    conditions on the real game script (durations, period structure, score path)
    and simulates only *who is on the floor*, so the arm is graded in isolation
    from the pace and possession-outcome models."""
    game_id: int
    team_id: int
    dur: np.ndarray
    period: np.ndarray
    start_clock: np.ndarray
    margin: np.ndarray
    time_bucket: np.ndarray
    margin_bucket: np.ndarray
    is_home: bool
    final_margin: int

    @property
    def n(self) -> int:
        return len(self.dur)

    @property
    def total_slot(self) -> float:
        return 5.0 * float(self.dur.sum())

    def remaining_slot(self) -> np.ndarray:
        rev = np.cumsum(self.dur[::-1])[::-1]
        return 5.0 * rev


@dataclass
class TeamPrior:
    """The pregame profile of one team for one game: candidates, as-of shares,
    ranks, starter identification, foul rates. Built strictly from earlier games."""
    game_id: int
    team_id: int
    pids: np.ndarray
    share: np.ndarray
    rank: np.ndarray
    srank: np.ndarray
    start_order: np.ndarray
    fpm: np.ndarray
    p_avail: np.ndarray
    n_prior_games: int

    @property
    def n(self) -> int:
        return len(self.pids)

    def starters(self) -> np.ndarray:
        return self.start_order[:5]


def add_asof_ranks(feats: pd.DataFrame, fit: RotationFit,
                   min_prior_games: int = 3) -> pd.DataFrame:
    """Shrunk as-of minutes (`mpg_adj`), the candidate truncation, and the two
    orderings the model uses: `minutes_rank_adj` (who is available, `p_play`) and
    `start_rank_asof` (who starts, and the state-tilt bucket). Shared by
    `build_priors` and `fit_tilt_tables` so the fitted table and the simulator
    bucket players identically."""
    d = feats[feats["team_games_asof"] >= min_prior_games].copy()
    mpg = shrink_mpg(d["mpg_asof_raw"].to_numpy(), d["team_games_asof"].to_numpy(),
                     d["minutes_rank_asof"].to_numpy(), fit.role_prior, fit.k0)
    mpg = mpg * np.where(d["last_game_dnp"].to_numpy() > 0.5, fit.w_dnp, 1.0)
    d["mpg_adj"] = np.maximum(mpg, 1e-3)
    d = d.sort_values(["game_id", "team_id", "mpg_adj"], ascending=[True, True, False])
    d["minutes_rank_adj"] = d.groupby(["game_id", "team_id"]).cumcount() + 1
    d = d[d["minutes_rank_adj"] <= MAX_CANDIDATES]
    col = f"start_ewma_{int(fit.start_alpha * 100)}"
    if col not in d.columns:
        col = "start_freq_asof"
    d = d.sort_values(["game_id", "team_id", col, "mpg_adj"],
                      ascending=[True, True, False, False])
    d["start_rank_asof"] = d.groupby(["game_id", "team_id"]).cumcount() + 1
    return d.sort_values(["game_id", "team_id", "minutes_rank_adj"])


def build_priors(feats: pd.DataFrame, fit: RotationFit,
                 min_prior_games: int = 3) -> dict[tuple[int, int], TeamPrior]:
    """One `TeamPrior` per (game_id, team_id) from the as-of feature table."""
    d = add_asof_ranks(feats, fit, min_prior_games)

    cf = d["cum_fouls_asof"].to_numpy(dtype="float64") if "cum_fouls_asof" in d else \
        d["fouls_per_min_asof"].to_numpy(dtype="float64") * 0.0
    cm = d["cum_minutes_asof"].to_numpy(dtype="float64") if "cum_minutes_asof" in d else \
        np.ones(len(d))
    d["fpm_shrunk"] = (cf + fit.fpm_prior_min * fit.fpm_league) / (cm + fit.fpm_prior_min)

    p_play = np.asarray(fit.p_play, dtype="float64") if fit.p_play else \
        np.full(MAX_CANDIDATES, 0.85)

    out: dict[tuple[int, int], TeamPrior] = {}
    for (gid, tid), g in d.groupby(["game_id", "team_id"], sort=False):
        pids = g["pid"].to_numpy(dtype="int64")
        mp = g["mpg_adj"].to_numpy(dtype="float64")
        share, pids_ext = extend_profile(mp / mp.sum(), pids, fit.n_profile, fit.tail_ratio)
        n_ext = len(pids_ext) - len(pids)
        fpm = np.concatenate([g["fpm_shrunk"].to_numpy(dtype="float64"),
                              np.full(n_ext, fit.fpm_league)])
        srk = np.concatenate([g["start_rank_asof"].to_numpy(dtype="float64"),
                              np.arange(len(pids) + 1, len(pids_ext) + 1, dtype="float64")])
        rank = np.arange(1, len(pids_ext) + 1, dtype="int64")
        start_order = np.argsort(srk, kind="stable").astype("int64")
        srank = np.empty(len(pids_ext), dtype="int64")
        srank[start_order] = np.arange(1, len(pids_ext) + 1)
        pa = p_play[np.clip(rank, 1, len(p_play)) - 1].copy()
        dnpf = np.concatenate([np.where(g["last_game_dnp"].to_numpy() > 0.5,
                                        fit.w_avail_dnp, 1.0), np.ones(n_ext)])
        out[(int(gid), int(tid))] = TeamPrior(
            game_id=int(gid), team_id=int(tid), pids=pids_ext, share=share, rank=rank,
            srank=srank, start_order=start_order, fpm=np.clip(fpm, 0.0, 0.30),
            p_avail=np.clip(pa * dnpf, 0.0, 1.0),
            n_prior_games=int(g["team_games_asof"].iloc[0]),
        )
    return out


def build_scripts(tp: pd.DataFrame) -> dict[tuple[int, int], GameScript]:
    scripts: dict[tuple[int, int], GameScript] = {}
    for (gid, tid), g in tp.groupby(["game_id", "team_id"], sort=False):
        scripts[(int(gid), int(tid))] = GameScript(
            game_id=int(gid), team_id=int(tid),
            dur=g["duration_s"].to_numpy(dtype="float64"),
            period=g["period"].to_numpy(dtype="int64"),
            start_clock=g["start_clock"].to_numpy(dtype="int64"),
            margin=g["margin"].to_numpy(dtype="int64"),
            time_bucket=g["time_bucket"].to_numpy(dtype="int64"),
            margin_bucket=g["margin_bucket"].to_numpy(dtype="int64"),
            is_home=bool(g["is_home"].iloc[0]),
            final_margin=int(g["final_margin"].iloc[0]),
        )
    return scripts


def draw_available(prior: TeamPrior, rng: np.random.Generator) -> np.ndarray:
    """Who dresses and is used at all, drawn from the fitted P(plays | as-of rank)
    table times the last-game-DNP factor.

    This is the pre-registered "availability (did_not_play history; the last
    game's DNP)" input doing real work. It is a shared layer, identical across
    the four arms, and it exists because no Dirichlet over the full candidate
    list can produce the exact zeros that real minutes tables are full of: the
    dev smoke run without it put 12.1 players on the floor per team-game against
    an actual 9.9. Five availabilities are guaranteed, filled in as-of rank order.
    """
    avail = rng.random(prior.n) < prior.p_avail
    if avail.sum() < 5:
        for i in np.argsort(prior.rank):
            if not avail[i]:
                avail[i] = True
            if avail.sum() >= 5:
                break
    return avail


# ===========================================================================
# 5. RNG
# ===========================================================================
def game_stream(seed: int, game_id: int, family: str = RNG_FAMILY) -> np.random.Generator:
    """The (seed, game_id, family) stream, expanded for sequential use.

    The 64-bit key comes from `cbb_sim.control.rng.stream_keys` verbatim -- the
    same contract the Control and the pace sampler use -- so a game's draws
    depend on nothing but (seed, game_id, family).
    """
    key = int(crng.stream_keys(seed, np.asarray([int(game_id)], dtype="int64"), family)[0])
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))


# ===========================================================================
# 6. Arms
# ===========================================================================
class RotationArm:
    """Base class. `simulate(prior, script, rng)` returns an (n_poss, 5) array of
    CBBD player ids and an (n_poss, n_cand) array of foul counts."""

    name = "base"
    simplicity_rank = 99

    def __init__(self, fit: RotationFit):
        self.fit = fit

    # -- targets -----------------------------------------------------------
    def draw_targets(self, prior: TeamPrior, script: GameScript,
                     rng: np.random.Generator, avail: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, prior: TeamPrior, script: GameScript,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        avail = draw_available(prior, rng)
        targets = self.draw_targets(prior, script, rng, avail)
        return run_scheduler(prior, script, targets, avail, self.fit, rng)


def apply_min_target(s: np.ndarray, avail: np.ndarray, min_share: float) -> np.ndarray:
    """Floor an available player's share at `min_share` and renormalise.

    `p_play` is fitted as P(the player records ANY minutes), so a player drawn
    available is by construction one who records minutes; without a floor the
    Dirichlet tail hands him a share so small the scheduler never brings him on,
    and the simulated nonzero-minute count lands ~0.9 players short of the real
    one. `min_share` is the training season's mean share of the *smallest*
    nonzero minutes total in a team-game."""
    out = np.where(avail, np.maximum(s, min_share), 0.0)
    tot = out.sum()
    return out / tot if tot > 0 else s


def _dirichlet(rng: np.random.Generator, p: np.ndarray, alpha: float) -> np.ndarray:
    a = np.maximum(np.asarray(p, dtype="float64") * float(alpha), 1e-6)
    g = rng.gamma(a)
    s = g.sum()
    return g / s if s > 0 else np.asarray(p, dtype="float64")


class R1Dirichlet(RotationArm):
    """R1 -- single-level Dirichlet over the extended as-of share profile."""
    name = "R1_dirichlet"
    simplicity_rank = 1

    def draw_targets(self, prior, script, rng, avail):
        p = prior.share * avail
        p = p / p.sum() if p.sum() > 0 else prior.share
        s = _dirichlet(rng, p, self.fit.alpha)
        s = apply_min_target(s, avail, self.fit.min_share)
        return s * script.total_slot


class R2HierDirichlet(RotationArm):
    """R2 -- hierarchical Dirichlet: starters family vs bench family, then within.

    Mean share is preserved exactly at both levels, which is the whole reason the
    hierarchical form exists in `usage_alloc.py`: a single-level Dirichlet with a
    varying concentration silently inflates whoever has the larger alpha.
    """
    name = "R2_hier_dirichlet"
    simplicity_rank = 2

    def draw_targets(self, prior, script, rng, avail):
        st = prior.starters()
        is_start = np.zeros(prior.n, dtype=bool)
        is_start[st] = True
        is_start &= avail
        is_bench = avail & ~is_start
        p = prior.share * avail
        p_start, p_bench = float(p[is_start].sum()), float(p[is_bench].sum())
        if p_start <= 0 or p_bench <= 0:
            q = p / p.sum() if p.sum() > 0 else prior.share
            return _dirichlet(rng, q, self.fit.alpha) * script.total_slot
        tot = p_start + p_bench
        fam = _dirichlet(rng, np.array([p_start / tot, p_bench / tot]), self.fit.alpha_family)
        out = np.zeros(prior.n, dtype="float64")
        out[is_start] = fam[0] * _dirichlet(rng, p[is_start] / p_start, self.fit.alpha_starters)
        out[is_bench] = fam[1] * _dirichlet(rng, p[is_bench] / p_bench, self.fit.alpha_bench)
        out = apply_min_target(out, avail, self.fit.min_share)
        return out * script.total_slot


class R3StintHazard(RotationArm):
    """R3 -- per-possession logistic exit hazard (on floor) and entry score (bench).

    State dependence is intrinsic: margin, |margin|, seconds remaining and period
    are features of both hazards.
    """
    name = "R3_stint_hazard"
    simplicity_rank = 4

    EXIT_FEATURES = ["deficit", "sec_since_change", "surplus", "fouls", "foul_out",
                     "mpg_asof", "is_starter", "abs_margin", "margin",
                     "sec_left_frac", "period2", "late"]
    ENTER_FEATURES = ["deficit", "sec_since_change", "surplus", "fouls", "foul_out",
                      "mpg_asof", "is_starter", "abs_margin", "margin",
                      "sec_left_frac", "period2", "late"]

    def __init__(self, fit: RotationFit):
        super().__init__(fit)
        self.exit_w = np.asarray(fit.hazard_exit["coef"], dtype="float64")
        self.exit_b = float(fit.hazard_exit["intercept"])
        self.enter_w = np.asarray(fit.hazard_enter["coef"], dtype="float64")
        self.enter_b = float(fit.hazard_enter["intercept"])

    def draw_targets(self, prior, script, rng, avail):
        p = apply_min_target(prior.share, avail, self.fit.min_share)
        return p * script.total_slot

    def simulate(self, prior, script, rng):
        avail = draw_available(prior, rng)
        targets = self.draw_targets(prior, script, rng, avail)
        return run_hazard(prior, script, targets, avail, self.fit, self, rng)


class R4StintResample(RotationArm):
    """R4 -- empirical stint-sequence resampling from the team's own last k games.

    k = 5. The game is cut into state blocks (1st half; 2nd half > 8:00; and the
    final 8:00 split by the G8 margin bands <= 5 / 6-15 / > 15). For each block
    the donor game is drawn from the team's last five games with probability
    proportional to (possessions that donor spent in the same state band + 1),
    so a blowout stretch is played back from a real blowout stretch of that
    team's own season; when no donor has any possessions in the band, the
    nearest band by |margin| ordering is used. Donor lineups are mapped onto the
    current candidate list by player identity first and by as-of minutes rank
    otherwise; a collision (two donor players mapping to one candidate) is
    filled with the next unused candidate in as-of rank order, which is what
    keeps five-on-floor exact.
    """
    name = "R4_stint_resample"
    simplicity_rank = 3
    K_DONORS = 5

    def __init__(self, fit: RotationFit, donors: dict):
        super().__init__(fit)
        self.donors = donors

    def draw_targets(self, prior, script, rng, avail):
        p = apply_min_target(prior.share, avail, self.fit.min_share)
        return p * script.total_slot

    def simulate(self, prior, script, rng):
        avail = draw_available(prior, rng)
        return run_resample(prior, script, avail, self.fit, self.donors, rng)


# ===========================================================================
# 7. The shared scheduler (R1 / R2)
# ===========================================================================
def run_scheduler(prior: TeamPrior, script: GameScript, targets: np.ndarray,
                  avail: np.ndarray, fit: RotationFit, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stint placement against a drawn, state-tilted minutes *rate*.

    Two earlier forms failed on the training season and are worth recording.
    Targeting each player's game *total* and ranking by remaining deficit
    equalises everyone toward their target by the end of the game. Targeting a
    cumulative *path* and ranking by path deficit is better but still sluggish:
    starters took 0.65 of the on-floor slots in the final eight minutes against
    an actual 0.78, below even their 0.70 game average. What a coach does is
    spend minutes at a *rate* that depends on the state, so this scheduler
    tracks the rate directly:

        q_i(state) = target_i * state_tilt[rank_i, state] * foul_tilt[fouls_i, t]
                     normalised over the available players       (sum q = 1)
        e_i       <- e_i * exp(-d/H) + (1 - exp(-d/H)) * on_floor_i
                                                     (recent on-floor fraction)
        u_i        = 5 * q_i - e_i  +  lam * (C_i - played_i) / (5 H)

    `5 q_i` is player i's target probability of being on the floor right now, so
    the first term is "how far behind his current rate is he", the second a slow
    correction that keeps game totals on the drawn target, and `sum u = 0` makes
    this a pure reallocation. Because the on-floor fraction converges to `5 q_i`,
    minutes converge to `q_i * total_slot`, which is exactly the drawn Dirichlet
    target. The five largest `u` take the floor; a swap happens only when a bench
    player's `u` exceeds an on-floor player's by more than the fitted
    `swap_threshold`, which sets the substitution rate. Period boundaries force a
    threshold-free re-evaluation and a fifth foul forces a player off. `H`
    (`ema_horizon`), `swap_threshold` and `lam` are fitted jointly on the
    training season against its substitution rate, distinct-lineup count and
    top-5 minutes share -- see `fit_scheduler_params`.
    """
    n = prior.n
    tilt_state = fit.tilt.state
    tilt_foul = fit.tilt.foul
    rb = rank_bucket(prior.srank)
    theta = fit.swap_threshold
    fscale = fit.foul_rate_scale

    H = max(fit.ema_horizon, 1.0)
    lam = fit.lam_deficit
    played = np.zeros(n, dtype="float64")
    path = np.zeros(n, dtype="float64")
    fouls = np.zeros(n, dtype="int64")
    base = np.asarray(targets, dtype="float64") * avail

    order = [i for i in prior.start_order if avail[i]]
    on = order[:5] if len(order) >= 5 else list(np.flatnonzero(avail))[:5]
    onmask = np.zeros(n, dtype=bool)
    onmask[np.asarray(on, dtype="int64")] = True
    ema = onmask.astype("float64")

    npos = script.n
    out = np.empty((npos, 5), dtype="int64")
    fouls_hist = np.zeros((npos, n), dtype="int8")
    NEG = -1e12
    prev_period = script.period[0]
    prev_cell = (int(script.time_bucket[0]), int(script.margin_bucket[0]))
    q_prev = None
    cur_prev = None

    for k in range(npos):
        # credit the possession just played, from the state that decided it --
        # everything the decision at k sees is available strictly before k, which
        # is what lets `RotationSampler.next_lineup` reproduce this loop online.
        if cur_prev is not None:
            d_prev = script.dur[k - 1]
            path += (5.0 * d_prev) * q_prev
            played[cur_prev] += d_prev
            decay = np.exp(-d_prev / H)
            ema *= decay
            ema[cur_prev] += (1.0 - decay)
            hit = rng.random(5) < prior.fpm[cur_prev] * (d_prev / 60.0) * fscale
            if hit.any():
                fouls[cur_prev[hit]] += 1
        fouls_hist[k] = np.minimum(fouls, 127)

        tb = int(script.time_bucket[k])
        mb = int(script.margin_bucket[k])
        w = base * tilt_state[rb, tb, mb] * tilt_foul[np.minimum(fouls, FOUL_OUT), tb]
        w[fouls >= FOUL_OUT] = 0.0
        tot = w.sum()
        if tot <= 0:
            w = avail.astype("float64")
            tot = max(w.sum(), 1.0)
        q = w / tot
        u = 5.0 * q - ema + lam * (path - played) / (5.0 * H)
        u[~avail] = NEG
        u[fouls >= FOUL_OUT] = NEG

        # A period boundary is a fresh start, so the floor is re-set to the best
        # available five for the CURRENT state -- the top five by `q`, not by
        # `u`. Ranking a fresh start by `u` is exactly wrong: `u` is "how far
        # below your rate are you right now", so every resting player outranks
        # every on-floor one and the coach would tip off the second half with
        # whoever happened to be sitting. A threshold-free `u` re-shuffle at
        # every state-cell change was tried for the same reason and cost 3-4 pp
        # of the starters' share in the final eight minutes; it is not used.
        eligible = avail & (fouls < FOUL_OUT)
        if script.period[k] != prev_period and eligible.sum() >= 5:
            pick = np.argsort(-np.where(eligible, q, -1.0))[:5]
            onmask[:] = False
            onmask[pick] = True
            ema = np.where(onmask, np.maximum(ema, 0.5), ema)
        prev_period = script.period[k]
        prev_cell = (tb, mb)
        on_idx = np.flatnonzero(onmask)
        bench_idx = np.flatnonzero(~onmask & eligible)
        if len(bench_idx):
            order_on = on_idx[np.argsort(u[on_idx])]
            order_bench = bench_idx[np.argsort(-u[bench_idx])]
            for j in range(min(len(order_on), len(order_bench))):
                a, b = order_on[j], order_bench[j]
                if fouls[a] < FOUL_OUT and avail[a] and not (u[b] - u[a] > theta):
                    break
                onmask[a] = False
                onmask[b] = True

        cur = np.flatnonzero(onmask)
        if len(cur) != 5:  # defensive: never let five-on-floor break
            cur = np.argsort(-u)[:5]
            onmask[:] = False
            onmask[cur] = True
        out[k] = prior.pids[cur]
        q_prev, cur_prev = q, cur
    return out, fouls_hist


# ===========================================================================
# 8. R3 hazard simulator
# ===========================================================================
HAZARD_FEATURES = [
    "deficit", "sec_since_change", "surplus", "fouls", "foul_out",
    "target_share", "is_starter", "abs_margin", "margin", "sec_left_frac",
    "period2", "late", "abs_margin_x_late", "abs_margin_x_secleft",
    # player x state interactions. Everything above that is constant across the
    # players of one possession (margin, clock, period) CANCELS in the entry
    # softmax and cannot move *who* comes in; only these four can, which is why
    # the arm was flat across margin bands in the final 8:00 without them.
    "is_starter_x_abs_margin", "is_starter_x_late",
    "target_share_x_abs_margin_x_late", "fouls_x_late",
]


def _hazard_design(prior, script, targets, played, fouls, since_change, on_idx, k, rem):
    """Feature matrix for the players in `on_idx` at possession `k`.

    Feature order is `HAZARD_FEATURES`. The player x state interactions at the
    end are load-bearing: with `abs_margin` entering only linearly and only as a
    per-possession constant, it cancels in the entry softmax and the fitted arm
    was flat across margin bands in the final eight minutes (0.66 / 0.65 / 0.62
    against an actual 0.79 / 0.72 / 0.49).
    """
    on_idx = np.asarray(on_idx, dtype="int64")
    total = max(script.total_slot, 1.0)
    d = (targets[on_idx] - played[on_idx]) / max(rem[k], 1.0)
    surplus = np.maximum(played[on_idx] - targets[on_idx], 0.0) / total
    f = fouls[on_idx].astype("float64")
    sec_left_frac = rem[k] / total
    is_start = np.zeros(prior.n, dtype="float64")
    is_start[prior.starters()] = 1.0
    am = abs(int(script.margin[k])) / 10.0
    late = 1.0 if script.time_bucket[k] >= 2 else 0.0
    m = len(on_idx)
    ts = targets[on_idx] / total * prior.n
    st = is_start[on_idx]
    return np.column_stack([
        d,
        since_change[on_idx] / 300.0,
        surplus,
        f,
        (f >= FOUL_OUT).astype("float64"),
        ts,
        st,
        np.full(m, am),
        np.full(m, int(script.margin[k]) / 10.0),
        np.full(m, sec_left_frac),
        np.full(m, 1.0 if script.period[k] >= 2 else 0.0),
        np.full(m, late),
        np.full(m, am * late),
        np.full(m, am * sec_left_frac),
        st * am,
        st * late,
        ts * am * late,
        f * late,
    ])


def _evict_fouled_out(onmask: np.ndarray, fouls: np.ndarray, avail: np.ndarray,
                      score: np.ndarray) -> np.ndarray:
    """Hard guard: nobody with five fouls stays on the floor while an eligible
    substitute exists. Returns the on-floor index array."""
    out = np.flatnonzero(onmask & (fouls >= FOUL_OUT))
    if len(out):
        pool = np.flatnonzero(~onmask & avail & (fouls < FOUL_OUT))
        if len(pool):
            pool = pool[np.argsort(-score[pool])]
            for a, b in zip(out, pool):
                onmask[a] = False
                onmask[b] = True
    return np.flatnonzero(onmask)


def run_hazard(prior: TeamPrior, script: GameScript, targets: np.ndarray,
               avail: np.ndarray, fit: RotationFit, arm, rng: np.random.Generator,
               design=None) -> tuple[np.ndarray, np.ndarray]:
    n = prior.n
    played = np.zeros(n, dtype="float64")
    fouls = np.zeros(n, dtype="int64")
    since = np.zeros(n, dtype="float64")
    onmask = np.zeros(n, dtype=bool)
    order = [i for i in prior.start_order if avail[i]]
    on = order[:5] if len(order) >= 5 else list(np.flatnonzero(avail))[:5]
    onmask[np.asarray(on, dtype="int64")] = True

    npos = script.n
    rem = script.remaining_slot()
    out = np.empty((npos, 5), dtype="int64")
    fouls_hist = np.zeros((npos, n), dtype="int8")
    fscale = fit.foul_rate_scale
    dfun = design or _hazard_design

    for k in range(npos):
        # foul state as of the START of possession k -- the same convention
        # `run_scheduler` and `actual_foul_matrix` use, so sim and actual are
        # graded on the same object.
        fouls_hist[k] = np.minimum(fouls, 127)
        on_idx = np.flatnonzero(onmask)
        bench_idx = np.flatnonzero(~onmask & avail & (fouls < FOUL_OUT))
        if len(on_idx) == 5 and len(bench_idx):
            Xe = dfun(prior, script, targets, played, fouls, since, on_idx, k, rem)
            pe = 1.0 / (1.0 + np.exp(-(Xe @ arm.exit_w + arm.exit_b)))
            pe = np.where(fouls[on_idx] >= FOUL_OUT, 1.0, pe)
            exit_mask = rng.random(5) < pe
            n_exit = int(exit_mask.sum())
            if n_exit:
                Xb = dfun(prior, script, targets, played, fouls, since, bench_idx, k, rem)
                sc = Xb @ arm.enter_w + arm.enter_b
                w = np.exp(sc - sc.max())
                w = w / w.sum() if w.sum() > 0 else np.ones(len(bench_idx)) / len(bench_idx)
                n_take = min(n_exit, len(bench_idx))
                pick = rng.choice(len(bench_idx), size=n_take, replace=False, p=w)
                # a fouled-out player leaves FIRST: with more exits drawn than
                # bench players available, taking them in index order can leave
                # a player with five fouls on the floor.
                cand_out = on_idx[exit_mask]
                cand_out = cand_out[np.argsort(-(fouls[cand_out] >= FOUL_OUT).astype(int),
                                               kind="stable")]
                leaving = cand_out[:n_take]
                onmask[leaving] = False
                onmask[bench_idx[pick]] = True
                since[leaving] = 0.0
                since[bench_idx[pick]] = 0.0

        cur = np.flatnonzero(onmask)
        if len(cur) != 5:
            score = np.where(avail & (fouls < FOUL_OUT), targets - played, -1e12)
            cur = np.argsort(-score)[:5]
            onmask[:] = False
            onmask[cur] = True
        cur = _evict_fouled_out(onmask, fouls, avail, targets - played)
        out[k] = prior.pids[cur]

        d = script.dur[k]
        played[cur] += d
        since += d
        hit = rng.random(5) < prior.fpm[cur] * (d / 60.0) * fscale
        if hit.any():
            fouls[cur[hit]] += 1
    return out, fouls_hist


# ===========================================================================
# 9. R4 empirical resampling
# ===========================================================================
def build_donor_bank(tp: pd.DataFrame, feats: pd.DataFrame | None = None,
                     k_donors: int = 5) -> dict:
    """For each (game_id, team_id): the team's last `k_donors` earlier games, each
    stored as its possession-by-possession on-floor sets plus every possession's
    state band. Strictly earlier games only -- the donor bank is a pregame object."""
    tp = tp.sort_values(["team_id", "game_date", "game_id", "poss_index"])
    per_game: dict[tuple[int, int], dict] = {}
    for (tid, gid), g in tp.groupby(["team_id", "game_id"], sort=False):
        band = _state_band(g["time_bucket"].to_numpy(), g["margin_bucket"].to_numpy())
        per_game[(int(tid), int(gid))] = {
            "lineups": g[SLOTS].to_numpy(dtype="int64"),
            "band": band,
            "date": g["game_date"].iloc[0],
        }
    order: dict[int, list[tuple]] = {}
    for (tid, gid), v in per_game.items():
        order.setdefault(tid, []).append((v["date"], gid))
    for tid in order:
        order[tid].sort()

    donors: dict[tuple[int, int], list] = {}
    lo_off = None if k_donors <= 0 else k_donors  # k_donors <= 0 means "all season to date"
    for tid, lst in order.items():
        for i, (_, gid) in enumerate(lst):
            lo = 0 if lo_off is None else max(0, i - lo_off)
            prev = [g2 for _, g2 in lst[lo:i]]
            donors[(int(gid), int(tid))] = [per_game[(tid, g2)] for g2 in prev]
    return donors


def _state_band(tb: np.ndarray, mb: np.ndarray) -> np.ndarray:
    """0 = 1st half, 1 = 2nd half > 8:00 left, 2/3/4 = final 8:00 by margin band."""
    band = np.zeros(len(tb), dtype="int64")
    band[tb == 1] = 1
    late = tb >= 2
    band[late] = 2 + mb[late]
    return band


def _reassign(chosen: list[int], blocked: np.ndarray, fouls: np.ndarray,
              order_avail: np.ndarray) -> list[int]:
    """Replace every blocked player in the donor-mapped five with the highest
    as-of-ranked available substitute who is neither blocked nor already on."""
    used = set(int(c) for c in chosen if not blocked[int(c)])
    out = []
    for c in chosen:
        c = int(c)
        if blocked[c] or (c in used and c in out):
            repl = None
            for cand in order_avail:
                cand = int(cand)
                if cand not in used and not blocked[cand] and fouls[cand] < FOUL_OUT:
                    repl = cand
                    break
            c = repl if repl is not None else c
        used.add(c)
        out.append(c)
    seen, final = set(), []
    for c in out:
        if c in seen:
            for cand in order_avail:
                cand = int(cand)
                if cand not in seen and fouls[cand] < FOUL_OUT:
                    c = cand
                    break
        seen.add(c)
        final.append(c)
    return final


def run_resample(prior: TeamPrior, script: GameScript, avail: np.ndarray,
                 fit: RotationFit, donors: dict, rng: np.random.Generator,
                 override=None) -> tuple[np.ndarray, np.ndarray]:
    bank = donors.get((prior.game_id, prior.team_id), [])
    n = prior.n
    npos = script.n
    out = np.empty((npos, 5), dtype="int64")
    fouls_hist = np.zeros((npos, n), dtype="int8")
    order_avail = np.array([i for i in np.argsort(prior.rank) if avail[i]], dtype="int64")
    if len(order_avail) < 5 or not bank:
        five = order_avail[:5] if len(order_avail) >= 5 else np.arange(min(5, n))
        out[:] = prior.pids[five]
        return out, fouls_hist

    band = _state_band(script.time_bucket, script.margin_bucket)
    rank_of_pid = {int(p): i for i, p in enumerate(prior.pids) if avail[i]}

    donor_rank: list[dict[int, int]] = []
    for d in bank:
        ids, cnt = np.unique(d["lineups"].ravel(), return_counts=True)
        ordering = ids[np.argsort(-cnt)]
        donor_rank.append({int(p): i for i, p in enumerate(ordering)})

    played = np.zeros(n, dtype="float64")
    fouls = np.zeros(n, dtype="int64")
    fscale = fit.foul_rate_scale
    ov_out = np.zeros(n, dtype=bool)          # R5: overridden onto the bench
    # one "coach tolerance" draw per player per game: the block is then a
    # monotone, persistent, self-clearing function of the state rather than an
    # independent coin flip every possession.
    tol = rng.random(n) if override is not None else None
    rem = script.remaining_slot()
    targets = prior.share * script.total_slot
    zeros = np.zeros(n, dtype="float64")

    k = 0
    while k < npos:
        b = int(band[k])
        j = k
        while j < npos and int(band[j]) == b:
            j += 1
        w = np.array([float((d["band"] == b).sum()) for d in bank]) + 1.0
        if w.sum() <= len(bank):  # nobody has this band -> nearest band
            for alt in sorted(range(5), key=lambda x: abs(x - b)):
                w2 = np.array([float((d["band"] == alt).sum()) for d in bank]) + 1.0
                if w2.sum() > len(bank):
                    w, b = w2, alt
                    break
        di = int(rng.choice(len(bank), p=w / w.sum()))
        donor = bank[di]
        src = np.flatnonzero(donor["band"] == b)
        if len(src) == 0:
            src = np.arange(len(donor["band"]))
        rmap = donor_rank[di]

        block = j - k
        pos = np.linspace(0, len(src) - 1e-9, block).astype("int64")
        for t in range(block):
            lu = donor["lineups"][src[pos[t]]]
            chosen: list[int] = []
            used: set[int] = set()
            for pid in lu:
                idx = rank_of_pid.get(int(pid))
                if idx is None or fouls[idx] >= FOUL_OUT:
                    r = rmap.get(int(pid), len(order_avail) - 1)
                    idx = int(order_avail[min(r, len(order_avail) - 1)])
                if idx in used or fouls[idx] >= FOUL_OUT:
                    for cand in order_avail:
                        if int(cand) not in used and fouls[int(cand)] < FOUL_OUT:
                            idx = int(cand)
                            break
                used.add(int(idx))
                chosen.append(int(idx))
            fouls_hist[k + t] = np.minimum(fouls, 127)
            if override is not None:
                kk = k + t
                idx_all = np.arange(n, dtype="int64")
                X = _override_design(prior, script, targets, played, fouls, zeros,
                                     idx_all, kk, rem)
                z = X[:, OVERRIDE_STATE_COLS] @ override.dw
                pb = 1.0 / (1.0 + np.exp(-(override.scale * z + override.block_logit)))
                ov_out = tol < pb
                ov_out[fouls >= FOUL_OUT] = True
                if (~ov_out & avail).sum() < 5:      # never override past five
                    keep = np.argsort(pb)
                    for i2 in keep:
                        if avail[i2] and fouls[i2] < FOUL_OUT:
                            ov_out[i2] = False
                        if (~ov_out & avail).sum() >= 5:
                            break
                chosen = _reassign(chosen, ov_out, fouls, order_avail)
            cur = np.asarray(chosen[:5], dtype="int64")
            if (fouls[cur] >= FOUL_OUT).any():
                m = np.zeros(n, dtype=bool)
                m[cur] = True
                cur = _evict_fouled_out(m, fouls, avail, -prior.rank.astype("float64"))
                if len(cur) != 5:
                    cur = np.asarray(chosen[:5], dtype="int64")
            out[k + t] = prior.pids[cur]
            d = script.dur[k + t]
            played[cur] += d
            hit = rng.random(5) < prior.fpm[cur] * (d / 60.0) * fscale
            if hit.any():
                fouls[cur[hit]] += 1
        k = j
    return out, fouls_hist


# ===========================================================================
# 10. Engine-facing sampler
# ===========================================================================
@dataclass
class RotationState:
    """The engine's view at a possession boundary."""
    period: int
    seconds_remaining: int
    score_diff: int          # home minus away
    last_possession_seconds: float = 0.0


class RotationSampler:
    """`next_lineup(state)` for the possession engine.

    Holds one team's rotation state and advances it one possession at a time,
    running exactly the `run_scheduler` decision rule. The (seed, game_id,
    "rotation") stream is created once per game and shared by both teams (home
    draws first), as the offline `simulate()` path does --
    `tests/test_rotation.py` asserts the two produce the identical sequence.
    """

    def __init__(self, arm: RotationArm, prior: TeamPrior, seed: int, game_id: int,
                 is_home: bool, expected_total_seconds: float = 2400.0,
                 rng: np.random.Generator | None = None):
        self.arm = arm
        self.fit = arm.fit
        self.prior = prior
        self.is_home = is_home
        self.rng = rng if rng is not None else game_stream(seed, game_id)
        self.total_slot = 5.0 * float(expected_total_seconds)
        stub = GameScript(
            game_id=game_id, team_id=prior.team_id,
            dur=np.array([expected_total_seconds]), period=np.array([1]),
            start_clock=np.array([1200]), margin=np.array([0]),
            time_bucket=np.array([0]), margin_bucket=np.array([0]),
            is_home=is_home, final_margin=0,
        )
        self.avail = draw_available(prior, self.rng)
        self.targets = arm.draw_targets(prior, stub, self.rng, self.avail) * self.avail
        self.played = np.zeros(prior.n, dtype="float64")
        self.path = np.zeros(prior.n, dtype="float64")
        self.fouls = np.zeros(prior.n, dtype="int64")
        self.onmask = np.zeros(prior.n, dtype=bool)
        order = [i for i in prior.start_order if self.avail[i]]
        on = order[:5] if len(order) >= 5 else list(np.flatnonzero(self.avail))[:5]
        self.onmask[np.asarray(on, dtype="int64")] = True
        self.ema = self.onmask.astype("float64")
        self.prev_period = 1
        self.prev_cell = (0, 0)
        self.q_prev = None
        self.cur_prev = None

    def next_lineup(self, state: RotationState) -> tuple[int, ...]:
        """Credit the possession just played, react to the new state, return five ids."""
        fit = self.fit
        prior = self.prior
        if self.cur_prev is not None:
            d_prev = float(state.last_possession_seconds)
            self.path += (5.0 * d_prev) * self.q_prev
            self.played[self.cur_prev] += d_prev
            decay = np.exp(-d_prev / max(fit.ema_horizon, 1.0))
            self.ema *= decay
            self.ema[self.cur_prev] += (1.0 - decay)
            hit = self.rng.random(5) <                 prior.fpm[self.cur_prev] * (d_prev / 60.0) * fit.foul_rate_scale
            if hit.any():
                self.fouls[self.cur_prev[hit]] += 1

        margin = state.score_diff if self.is_home else -state.score_diff
        tb = int(time_bucket(np.array([state.period]), np.array([state.seconds_remaining]))[0])
        mb = int(margin_bucket(np.array([margin]))[0])
        rb = rank_bucket(prior.srank)
        w = self.targets * fit.tilt.state[rb, tb, mb] * \
            fit.tilt.foul[np.minimum(self.fouls, FOUL_OUT), tb]
        w[self.fouls >= FOUL_OUT] = 0.0
        tot = w.sum()
        if tot <= 0:
            w = self.avail.astype("float64")
            tot = max(w.sum(), 1.0)
        q = w / tot
        u = (5.0 * q - self.ema
             + fit.lam_deficit * (self.path - self.played) / (5.0 * max(fit.ema_horizon, 1.0)))
        u[~self.avail] = -1e12
        u[self.fouls >= FOUL_OUT] = -1e12

        eligible = self.avail & (self.fouls < FOUL_OUT)
        if state.period != self.prev_period and eligible.sum() >= 5:
            pick = np.argsort(-np.where(eligible, q, -1.0))[:5]
            self.onmask[:] = False
            self.onmask[pick] = True
            self.ema = np.where(self.onmask, np.maximum(self.ema, 0.5), self.ema)
        self.prev_period = state.period
        self.prev_cell = (tb, mb)
        on_idx = np.flatnonzero(self.onmask)
        bench_idx = np.flatnonzero(~self.onmask & eligible)
        if len(bench_idx):
            order_on = on_idx[np.argsort(u[on_idx])]
            order_bench = bench_idx[np.argsort(-u[bench_idx])]
            for j in range(min(len(order_on), len(order_bench))):
                a, b = order_on[j], order_bench[j]
                if (self.fouls[a] < FOUL_OUT and self.avail[a]
                        and not (u[b] - u[a] > fit.swap_threshold)):
                    break
                self.onmask[a] = False
                self.onmask[b] = True
        cur = np.flatnonzero(self.onmask)
        if len(cur) != 5:
            cur = np.argsort(-u)[:5]
            self.onmask[:] = False
            self.onmask[cur] = True
        self.q_prev, self.cur_prev = q, cur
        return tuple(int(x) for x in prior.pids[cur])


# ===========================================================================
# 11. Remaining fits (scheduler threshold, foul rate, hazards)
# ===========================================================================
def fit_foul_rate_scale(tp: pd.DataFrame, feats: pd.DataFrame,
                        fouls: pd.DataFrame, fit: "RotationFit") -> float:
    """Calibrate the simulated foul hazard so simulated team fouls per game match
    the training season's. `fpm_asof` is fouls per *box* minute; the sim charges
    fouls over possession-derived on-floor minutes, so the two differ by a small
    accounting factor which is fitted here rather than assumed to be 1."""
    pgm = tp.melt(id_vars=["game_id", "team_id", "duration_s"], value_vars=SLOTS,
                  value_name="pid")
    pgm = pgm.groupby(["game_id", "team_id", "pid"], as_index=False)["duration_s"].sum()
    cols = ["game_id", "team_id", "pid", "cum_fouls_asof", "cum_minutes_asof"]
    d = pgm.merge(feats[cols], on=["game_id", "team_id", "pid"], how="inner")
    fpm = ((d["cum_fouls_asof"] + fit.fpm_prior_min * fit.fpm_league)
           / (d["cum_minutes_asof"] + fit.fpm_prior_min)).clip(0.0, 0.30)
    expected = float((fpm * d["duration_s"] / 60.0).sum())
    fc = fouls.merge(tp[["game_id", "team_id"]].drop_duplicates(), on=["game_id", "team_id"],
                     how="inner") if "team_id" in fouls.columns else fouls
    actual = float(len(fc))
    if expected <= 0:
        return 1.0
    return float(np.clip(actual / expected, 0.3, 3.0))


def fit_n_profile(p_play: list[float], mean_nonzero: float) -> int:
    """Profile length so the expected number of *available* players matches the
    training season's mean nonzero-minute count. The 90th-percentile heuristic
    it replaces produced 12 slots and 8.8 players used against an actual 9.9."""
    cum = np.cumsum(np.asarray(p_play, dtype="float64"))
    n = int(np.argmin(np.abs(cum - float(mean_nonzero)))) + 1
    return int(np.clip(n, 8, MAX_CANDIDATES))


def fit_start_predictor(feats: pd.DataFrame, pg: pd.DataFrame,
                        fit: "RotationFit") -> tuple[float, dict]:
    """Pick the starter-predictor decay that maximises the overlap between the
    model's top five and the real starting five on the training season."""
    truth = pg.loc[pg["is_starter"], ["game_id", "team_id", "pid"]].copy()
    truth["started"] = 1
    scores, best, best_ov = {}, START_EWMA_ALPHAS[0], -1.0
    for a in list(START_EWMA_ALPHAS) + [None]:
        f = RotationFit(**{**fit.__dict__, "start_alpha": a if a else -1.0})
        d = add_asof_ranks(feats, f)
        d = d[d["start_rank_asof"] <= 5]
        m = d.merge(truth, on=["game_id", "team_id", "pid"], how="left")
        m["started"] = m["started"].fillna(0)
        ov = float(m.groupby(["game_id", "team_id"])["started"].sum().mean())
        key = "expanding_mean" if a is None else f"ewma_{a}"
        scores[key] = ov
        if a is not None and ov > best_ov:
            best, best_ov = a, ov
    return float(best), scores


def fit_availability(feats: pd.DataFrame, pg: pd.DataFrame
                     ) -> tuple[list[float], float]:
    """`p_play[rank]` = P(the as-of rank-r candidate records any minutes) and the
    multiplicative factor for a player whose last game was a DNP, both read off
    the training season. This is what the pre-registered availability input buys:
    it is the only part of the model that can produce an exact zero."""
    truth = pg[["game_id", "team_id", "pid", "minutes"]]
    d = feats.merge(truth, on=["game_id", "team_id", "pid"], how="left")
    d["played"] = (d["minutes"].fillna(0.0) > 0).astype(float)
    by_rank = d.groupby(d["minutes_rank_asof"].clip(1, MAX_CANDIDATES))["played"].mean()
    p_play = [float(by_rank.get(r, np.nan)) for r in range(1, MAX_CANDIDATES + 1)]
    last = 0.5
    for i, v in enumerate(p_play):
        if not np.isfinite(v):
            p_play[i] = last
        else:
            last = v
    m = d["last_game_dnp"].to_numpy() > 0.5
    if m.sum() >= 300:
        base = d.loc[~m].groupby(d.loc[~m, "minutes_rank_asof"].clip(1, MAX_CANDIDATES))["played"].mean()
        exp = float(np.nanmean([base.get(int(min(r, MAX_CANDIDATES)), np.nan)
                                for r in d.loc[m, "minutes_rank_asof"]]))
        w = float(d.loc[m, "played"].mean() / exp) if exp and np.isfinite(exp) else 1.0
    else:
        w = 1.0
    return p_play, float(np.clip(w, 0.05, 1.0))


def fit_fpm(feats: pd.DataFrame, pg: pd.DataFrame, fouls: pd.DataFrame,
            grid=(5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 150.0)) -> tuple[float, float, dict]:
    """League fouls per on-floor minute, and the shrinkage strength `M0` in
    `fpm = (cum_fouls + M0 * mu) / (cum_minutes + M0)`, chosen on the training
    season by predicting the next game's fouls."""
    fc = fouls.groupby(["game_id", "team_id", "pid"], as_index=False).size()
    fc = fc.rename(columns={"size": "fouls"})
    d = feats.merge(pg[["game_id", "team_id", "pid", "minutes"]],
                    on=["game_id", "team_id", "pid"], how="inner")
    d = d.merge(fc, on=["game_id", "team_id", "pid"], how="left")
    d["fouls"] = d["fouls"].fillna(0.0)
    mu = float(d["fouls"].sum() / max(d["minutes"].sum(), 1.0))
    scores, best, best_mae = {}, grid[0], np.inf
    for m0 in grid:
        pred = (d["cum_fouls_asof"] + m0 * mu) / (d["cum_minutes_asof"] + m0) * d["minutes"]
        mae = float(np.mean(np.abs(pred - d["fouls"])))
        scores[str(m0)] = mae
        if mae < best_mae:
            best, best_mae = m0, mae
    return mu, float(best), scores


def observed_change_rate(tp: pd.DataFrame) -> float:
    """P(the on-floor set changes at a possession boundary), the quantity the
    scheduler's `swap_threshold` is fitted against (2025 actual: 0.1526)."""
    lu = [frozenset(r) for r in tp[SLOTS].to_numpy()]
    g = tp["game_id"].to_numpy()
    t = tp["team_id"].to_numpy()
    chg, tot = 0, 0
    for i in range(1, len(lu)):
        if g[i] == g[i - 1] and t[i] == t[i - 1]:
            tot += 1
            if lu[i] != lu[i - 1]:
                chg += 1
    return chg / max(tot, 1)


def sim_change_rate(lineups_by_game: list[np.ndarray]) -> float:
    chg, tot = 0, 0
    for lu in lineups_by_game:
        for i in range(1, len(lu)):
            tot += 1
            if set(lu[i]) != set(lu[i - 1]):
                chg += 1
    return chg / max(tot, 1)


def observed_scheduler_targets(tp: pd.DataFrame) -> dict:
    """The three training-season quantities the scheduler parameters are fitted
    against: substitution rate, distinct lineups per team-game, and the top-5
    share of team minutes."""
    lu = [tuple(sorted(int(x) for x in r)) for r in tp[SLOTS].to_numpy()]
    g = tp["game_id"].to_numpy()
    t = tp["team_id"].to_numpy()
    chg = tot = 0
    per_game: dict[tuple[int, int], set] = {}
    counts: dict[tuple[int, int], dict] = {}
    for i in range(len(lu)):
        key = (int(g[i]), int(t[i]))
        per_game.setdefault(key, set()).add(lu[i])
        c = counts.setdefault(key, {})
        c[lu[i]] = c.get(lu[i], 0) + 1
        if i and g[i] == g[i - 1] and t[i] == t[i - 1]:
            tot += 1
            chg += lu[i] != lu[i - 1]
    t1, t3 = [], []
    for key, c in counts.items():
        v = sorted(c.values(), reverse=True)
        n = float(sum(v))
        t1.append(v[0] / n)
        t3.append(sum(v[:3]) / n)
    pg = player_game_minutes(tp)
    pg = pg.sort_values(["game_id", "team_id", "minutes"], ascending=[True, True, False])
    rk = pg.groupby(["game_id", "team_id"]).cumcount() + 1
    share = pg["minutes"] / pg.groupby(["game_id", "team_id"])["minutes"].transform("sum")
    top5 = share[rk <= 5].groupby([pg.loc[rk <= 5, "game_id"],
                                   pg.loc[rk <= 5, "team_id"]]).sum().mean()
    return {
        "change_rate": chg / max(tot, 1),
        "n_lineups": float(np.mean([len(v) for v in per_game.values()])),
        "top5_share": float(top5),
        "n_nonzero": float(pg.groupby(["game_id", "team_id"]).size().mean()),
        "lu_top1": float(np.mean(t1)),
        "lu_top3": float(np.mean(t3)),
    }


def fit_scheduler_params(fit: RotationFit, feats: pd.DataFrame, scripts: dict,
                         keys: list, targets: dict, seed: int = 7,
                         horizons=(240.0, 360.0, 540.0, 800.0),
                         thetas=(0.10, 0.20, 0.32, 0.48, 0.70),
                         lams=(0.0, 0.5),
                         profiles=(12, 13, 14, 15)) -> tuple[dict, dict]:
    """Joint grid fit of the four scheduler parameters (`ema_horizon`,
    `swap_threshold`, `lam_deficit`, `n_profile`) against four training-season
    targets: substitution rate, distinct lineups per team-game, top-5 share of
    team minutes, and players with > 0 minutes. Scored by summed squared
    relative error. Run with R1, the simplest arm, so the scheduler is not tuned
    around any one target-drawing scheme; R2 then inherits it unchanged."""
    scores, best, best_err = {}, None, np.inf
    for npro in profiles:
        f0 = RotationFit(**{**fit.__dict__, "n_profile": int(npro)})
        priors = build_priors(feats, f0)
        use = [k for k in keys if k in priors and k in scripts]
        for H in horizons:
            for th in thetas:
                for lam in lams:
                    f = RotationFit(**{**f0.__dict__, "ema_horizon": float(H),
                                       "swap_threshold": float(th),
                                       "lam_deficit": float(lam)})
                    arm = R1Dirichlet(f)
                    chg = tot = 0
                    nlu, t5, nz = [], [], []
                    for (gid, tid) in use:
                        pr, sc = priors[(gid, tid)], scripts[(gid, tid)]
                        lu, _ = arm.simulate(pr, sc, game_stream(seed, gid))
                        ks = [tuple(sorted(int(x) for x in r)) for r in lu]
                        nlu.append(len(set(ks)))
                        for i in range(1, len(ks)):
                            tot += 1
                            chg += ks[i] != ks[i - 1]
                        ids, inv = np.unique(lu, return_inverse=True)
                        mins = np.zeros(len(ids))
                        np.add.at(mins, inv.reshape(lu.shape).ravel(),
                                  np.repeat(sc.dur[:, None], 5, axis=1).ravel())
                        nz.append(int((mins > 0).sum()))
                        mins.sort()
                        t5.append(mins[-5:].sum() / mins.sum())
                    got = {"change_rate": chg / max(tot, 1),
                           "n_lineups": float(np.mean(nlu)),
                           "top5_share": float(np.mean(t5)),
                           "n_nonzero": float(np.mean(nz))}
                    err = sum(((got[k] - targets[k]) / targets[k]) ** 2 for k in got)
                    scores[f"n={npro},H={H:.0f},theta={th},lam={lam}"] = {**got, "err": err}
                    if err < best_err:
                        best = {"n_profile": int(npro), "ema_horizon": float(H),
                                "swap_threshold": float(th), "lam_deficit": float(lam)}
                        best_err = err
    return best, scores


def build_hazard_training(tp: pd.DataFrame, feats: pd.DataFrame, fit: RotationFit,
                          game_ids: list[int], fouls: pd.DataFrame | None = None,
                          design=None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive substitution events from on-floor set transitions and build the two
    hazard design matrices.

    NOTE (data defect, documented in `docs/models/rotation/model.md`): CBBD pbp
    carries **zero** `Substitution` rows in season 2024 and only ~35 per game in
    2025 against ~145 per game in 2026, so the pre-registration's "fitted on 2024
    substitution events" is fitted on substitution events *derived from the
    on-floor stream* -- a player leaving the on-floor set between consecutive
    possessions is an exit, entering it is an entry. That is the same event the
    Substitution rows would carry, at possession-boundary resolution.
    """
    sel = tp[tp["game_id"].isin(game_ids)]
    scripts = build_scripts(sel)
    priors = build_priors(feats[feats["game_id"].isin(game_ids)], fit)
    actual = {(int(g), int(t)): d[SLOTS].to_numpy(dtype="int64")
              for (g, t), d in sel.groupby(["game_id", "team_id"], sort=False)}
    evg = {}
    if fouls is not None and len(fouls):
        f = fouls[fouls["game_id"].isin(game_ids)]
        evg = {(int(g), int(t)): d for (g, t), d in f.groupby(["game_id", "team_id"],
                                                             sort=False)}

    dfun = design or _hazard_design
    Xe, ye, Xn, yn = [], [], [], []
    for key, lu in actual.items():
        if key not in priors or key not in scripts:
            continue
        prior, script = priors[key], scripts[key]
        idx_of = {int(p): i for i, p in enumerate(prior.pids)}
        n = prior.n
        targets = prior.share * script.total_slot
        played = np.zeros(n)
        since = np.zeros(n)
        rem = script.remaining_slot()
        npos = min(len(lu), script.n)
        # the real per-possession foul state of this team-game. Reading the game's
        # own foul events is legitimate HERE -- this is the fit, and the label is
        # this game's own substitution -- and is never done at simulation time,
        # where fouls are simulated from the as-of rate.
        fm = actual_foul_matrix(evg.get(key), script.period, script.start_clock, idx_of)
        prev = None
        for k in range(npos):
            cur = [idx_of.get(int(p)) for p in lu[k]]
            cur = [c for c in cur if c is not None]
            fouls_v = fm[k].astype("int64") if k < fm.shape[0] else np.zeros(n, dtype="int64")
            if prev is not None and len(prev) == 5:
                prev_set, cur_set = set(prev), set(cur)
                on_idx = np.asarray(prev, dtype="int64")
                bench_idx = np.asarray([i for i in range(n) if i not in prev_set],
                                       dtype="int64")
                if len(bench_idx):
                    Xe.append(dfun(prior, script, targets, played, fouls_v,
                                   since, on_idx, k, rem))
                    ye.append(np.array([0.0 if i in cur_set else 1.0 for i in prev]))
                    Xn.append(dfun(prior, script, targets, played, fouls_v,
                                   since, bench_idx, k, rem))
                    yn.append(np.array([1.0 if i in cur_set else 0.0 for i in bench_idx]))
            d = script.dur[k]
            if cur:
                played[np.asarray(cur)] += d
            since += d
            if prev is not None:                     # reset on entry or exit
                changed = (set(cur) ^ set(prev))
                if changed:
                    since[np.asarray(sorted(changed), dtype="int64")] = 0.0
            prev = cur
    if not Xe:
        raise RuntimeError("no hazard training rows produced")
    return (np.vstack(Xe), np.concatenate(ye), np.vstack(Xn), np.concatenate(yn))


def fit_hazard_models(Xe, ye, Xn, yn) -> tuple[dict, dict]:
    from sklearn.linear_model import LogisticRegression

    out = []
    for X, y in ((Xe, ye), (Xn, yn)):
        m = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        m.fit(X, y)
        out.append({"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
                    "n": int(len(y)), "base_rate": float(np.mean(y))})
    return out[0], out[1]


# ===========================================================================
# 12. Metrics (identical code path for sim and actual -- like-for-like)
# ===========================================================================
LATE_TIME_BUCKETS = (2, 3)          # 2nd half, final 8:00 (OT excluded)
FOUL_TROUBLE = 4


def actual_foul_matrix(events: pd.DataFrame, period: np.ndarray, start_clock: np.ndarray,
                       pid_index: dict[int, int]) -> np.ndarray:
    """(n_poss, n_players) cumulative personal fouls at each possession start,
    from the real `PersonalFoul` events of that team-game."""
    n_poss, n_pl = len(period), len(pid_index)
    out = np.zeros((n_poss, n_pl), dtype="int16")
    if events is None or not len(events):
        return out
    key_poss = period.astype("int64") * 10000 - start_clock.astype("int64")
    for pid, p, c in zip(events["pid"].to_numpy(), events["period"].to_numpy(),
                         events["start_clock"].to_numpy()):
        j = pid_index.get(int(pid))
        if j is None:
            continue
        k = int(p) * 10000 - int(c)
        out[key_poss > k, j] += 1
    return out


def team_game_stats(lineups: np.ndarray, dur: np.ndarray, tb: np.ndarray, mb: np.ndarray,
                    starter_pids: set, fouls: np.ndarray | None,
                    foul_pids: np.ndarray | None, rotation_pids: set) -> dict:
    """Every per-team-game quantity the pre-registered metric table needs.

    Called with the *actual* on-floor sequence and with each arm's *simulated*
    one through the identical code path, so no metric can differ because of how
    it was computed."""
    ids, inv = np.unique(lineups, return_inverse=True)
    inv = inv.reshape(lineups.shape)
    minutes = np.zeros(len(ids), dtype="float64")
    np.add.at(minutes, inv.ravel(), np.repeat(dur[:, None], 5, axis=1).ravel())
    minutes /= 60.0
    order = np.argsort(-minutes)
    tot = minutes.sum()
    cum = np.cumsum(minutes[order])

    lu_keys = [tuple(sorted(int(x) for x in row)) for row in lineups]
    counts = pd.Series(lu_keys).value_counts().to_numpy(dtype="float64")
    n_poss = float(len(lu_keys))
    lc = np.cumsum(counts) / n_poss

    st = np.array([1.0 if int(i) in starter_pids else 0.0 for i in ids])
    starter_slot = st[inv].sum(axis=1)          # starters on floor per possession
    late = np.isin(tb, LATE_TIME_BUCKETS)
    late_bands = {}
    for band in range(N_MARGIN_BUCKETS):
        m = late & (mb == band)
        late_bands[band] = (float(starter_slot[m].sum()), float(5 * m.sum()))

    ft_on = ft_opp = ft4_on = ft4_opp = 0.0
    if fouls is not None and foul_pids is not None and len(foul_pids):
        fp_is_starter = np.array([int(p) in starter_pids for p in foul_pids])
        if fp_is_starter.any():
            sub = fouls[:, fp_is_starter]
            spids = np.asarray(foul_pids)[fp_is_starter]
            onfloor = np.zeros_like(sub, dtype=bool)
            pos = {int(p): j for j, p in enumerate(spids)}
            for k in range(lineups.shape[0]):
                for p in lineups[k]:
                    j = pos.get(int(p))
                    if j is not None:
                        onfloor[k, j] = True
            m4 = sub >= FOUL_TROUBLE
            ft_opp = float(m4.sum())
            ft_on = float(onfloor[m4].sum())
            me = sub == FOUL_TROUBLE
            ft4_opp = float(me.sum())
            ft4_on = float(onfloor[me].sum())

    rot_minutes = {int(p): float(m) for p, m in zip(ids, minutes) if int(p) in rotation_pids}
    return {
        "n_nonzero": int((minutes > 0).sum()),
        "team_minutes": float(tot),
        "top5_share": float(cum[min(4, len(cum) - 1)] / tot) if tot > 0 else np.nan,
        "top8_share": float(cum[min(7, len(cum) - 1)] / tot) if tot > 0 else np.nan,
        "lu_top1": float(lc[0]),
        "lu_top3": float(lc[min(2, len(lc) - 1)]),
        "lu_top5": float(lc[min(4, len(lc) - 1)]),
        "n_lineups": int(len(counts)),
        "late_bands": late_bands,
        "foul_trouble": (ft_on, ft_opp, ft4_on, ft4_opp),
        "rot_minutes": rot_minutes,
    }


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    r = stats.ks_2samp(np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64"))
    return float(r.statistic), float(r.pvalue)


def aggregate_stats(records: list[dict]) -> dict:
    """Pool per-team-game records into the pre-registered metric row."""
    if not records:
        return {}
    out = {
        "n_team_games": len(records),
        "n_nonzero_mean": float(np.mean([r["n_nonzero"] for r in records])),
        "top5_share": float(np.mean([r["top5_share"] for r in records])),
        "top8_share": float(np.mean([r["top8_share"] for r in records])),
        "lu_top1": float(np.mean([r["lu_top1"] for r in records])),
        "lu_top3": float(np.mean([r["lu_top3"] for r in records])),
        "lu_top5": float(np.mean([r["lu_top5"] for r in records])),
        "n_lineups_mean": float(np.mean([r["n_lineups"] for r in records])),
    }
    for band in range(N_MARGIN_BUCKETS):
        num = sum(r["late_bands"][band][0] for r in records)
        den = sum(r["late_bands"][band][1] for r in records)
        out[f"late_starter_share_b{band}"] = float(num / den) if den else float("nan")
        out[f"late_slots_b{band}"] = float(den / 5.0)
    on = sum(r["foul_trouble"][0] for r in records)
    opp = sum(r["foul_trouble"][1] for r in records)
    on4 = sum(r["foul_trouble"][2] for r in records)
    opp4 = sum(r["foul_trouble"][3] for r in records)
    out["foul_trouble_share"] = float(on / opp) if opp else float("nan")
    out["foul_trouble_opps"] = float(opp)
    out["foul4_share"] = float(on4 / opp4) if opp4 else float("nan")
    out["foul4_opps"] = float(opp4)
    return out


# ===========================================================================
# 13. Round-2 arms (R5 hybrid, R6 respecified stint hazard)
# ===========================================================================
#: R6 -- R3's feature list plus the full set of explicit interactions of
#: (fouls, |margin|, seconds remaining) with `is_starter` and `period`, which is
#: what round 1's flat entry could not represent.
R6_FEATURES = list(HAZARD_FEATURES) + [
    "fouls_x_is_starter", "fouls_x_period2", "fouls_x_sec_left_frac",
    "fouls_x_is_starter_x_late", "abs_margin_x_period2",
    "sec_left_frac_x_is_starter", "abs_margin_x_is_starter_x_late",
    "is_close_x_late", "is_close_x_late_x_is_starter",
]

#: R5 -- the two state-conditioned override hazards. (a) foul trouble:
#: fouls x period x seconds-remaining x is_starter; (b) margin: |margin| x
#: seconds-remaining x is_starter for blowouts, and the close-and-late
#: re-entry term. No hand-set thresholds: every coefficient is fitted.
OVERRIDE_FEATURES = [
    "fouls", "fouls_x_is_starter", "fouls_x_period2", "fouls_x_sec_left_frac",
    "foul_out", "is_starter", "abs_margin", "abs_margin_x_is_starter",
    "abs_margin_x_late", "abs_margin_x_sec_left_frac",
    "is_close_x_late", "is_close_x_late_x_is_starter", "period2", "sec_left_frac",
]

#: the state-dependent columns of `OVERRIDE_FEATURES` -- everything except
#: `is_starter`, `period2` and `sec_left_frac`, which are present in the neutral
#: state too and must cancel out of the deviation.
OVERRIDE_STATE_COLS = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], dtype="int64")


def _state_scalars(script: GameScript, k: int, rem: np.ndarray) -> tuple:
    total = max(script.total_slot, 1.0)
    am = abs(int(script.margin[k])) / 10.0
    late = 1.0 if script.time_bucket[k] >= 2 else 0.0
    slf = rem[k] / total
    p2 = 1.0 if script.period[k] >= 2 else 0.0
    close = 1.0 if abs(int(script.margin[k])) <= 5 else 0.0
    return total, am, late, slf, p2, close


def _r6_design(prior, script, targets, played, fouls, since_change, idx, k, rem):
    base = _hazard_design(prior, script, targets, played, fouls, since_change, idx, k, rem)
    idx = np.asarray(idx, dtype="int64")
    _, am, late, slf, p2, close = _state_scalars(script, k, rem)
    f = fouls[idx].astype("float64")
    st = np.zeros(prior.n, dtype="float64")
    st[prior.starters()] = 1.0
    st = st[idx]
    m = len(idx)
    return np.column_stack([
        base,
        f * st, f * p2, f * slf, f * st * late,
        np.full(m, am * p2),
        slf * st,
        st * am * late,
        np.full(m, close * late),
        st * close * late,
    ])


def _override_design(prior, script, targets, played, fouls, since_change, idx, k, rem):
    """Feature matrix for R5's override hazards (`OVERRIDE_FEATURES` order)."""
    idx = np.asarray(idx, dtype="int64")
    _, am, late, slf, p2, close = _state_scalars(script, k, rem)
    f = fouls[idx].astype("float64")
    st = np.zeros(prior.n, dtype="float64")
    st[prior.starters()] = 1.0
    st = st[idx]
    m = len(idx)
    return np.column_stack([
        f, f * st, f * p2, f * slf,
        (f >= FOUL_OUT).astype("float64"),
        st,
        np.full(m, am), st * am, np.full(m, am * late), np.full(m, am * slf),
        np.full(m, close * late), st * close * late,
        np.full(m, p2), np.full(m, slf),
    ])


class R6StintHazard(R3StintHazard):
    """R6 -- R3 respecified: the same per-possession exit/entry hazards, with
    fouls, |margin| and seconds remaining entered as explicit interactions with
    `is_starter` and `period`. Round 1's flat entry could not represent foul
    trouble at all: every per-possession constant cancels in the entry softmax,
    so only player x state terms can move who is on the floor."""
    name = "R6_stint_hazard_v2"
    simplicity_rank = 3

    def __init__(self, fit: RotationFit):
        RotationArm.__init__(self, fit)
        self.exit_w = np.asarray(fit.notes["r6_exit"]["coef"], dtype="float64")
        self.exit_b = float(fit.notes["r6_exit"]["intercept"])
        self.enter_w = np.asarray(fit.notes["r6_enter"]["coef"], dtype="float64")
        self.enter_b = float(fit.notes["r6_enter"]["intercept"])

    def simulate(self, prior, script, rng):
        avail = draw_available(prior, rng)
        targets = self.draw_targets(prior, script, rng, avail)
        return run_hazard(prior, script, targets, avail, self.fit, self, rng,
                          design=_r6_design)


class R5Hybrid(RotationArm):
    """R5 -- R4's own-recent-games stint-sequence resampling under two fitted,
    state-conditioned override hazards.

    The donor sequence (k fitted on the training season from {3, 5, 8, all}) says
    *which fives play together*; the overrides say *when the coach deviates*. At
    every possession boundary

        z_i     = (x_i - x_i^neutral) . (w_exit - w_enter)
        p_i     = sigmoid(scale * z_i + logit(p0))
        blocked = tol_i < p_i,   tol_i ~ U(0,1), drawn ONCE per player per game

    where `x` is the `OVERRIDE_FEATURES` row and `x^neutral` the same row with
    fouls, |margin| and the close-and-late terms zeroed. `z = 0` in an ordinary
    state, so the override does nothing there beyond the `p0` floor; the donor's
    lineup is then mapped onto the candidate list avoiding blocked players, so a
    starter in foul trouble or a starter in a blowout comes off even though the
    donor game had him on, and he comes back when the margin tightens. The
    per-game tolerance draw is what makes the block persistent and self-clearing
    instead of a per-possession coin flip.

    `scale` and `p0` are the two fitted knobs (on the training season, against
    the four state-dependence cells); `w_exit` and `w_enter` are the fitted
    override-hazard coefficients. There are no hand-set foul or margin
    thresholds anywhere. An earlier Markov-flag version (P(bench) =
    scale x sigmoid(exit), P(return) = sigmoid(entry)) is recorded in
    `experiments.md` as rejected: the fitted base rates imply a two-thirds
    blocked fraction even in an ordinary state, so it churned the lineup and the
    fit had to trade state dependence against the substitution rate.
    """
    name = "R5_hybrid"
    simplicity_rank = 2

    def __init__(self, fit: RotationFit, donors: dict):
        super().__init__(fit)
        self.donors = donors
        self.exit_w = np.asarray(fit.notes["r5_exit"]["coef"], dtype="float64")
        self.exit_b = float(fit.notes["r5_exit"]["intercept"])
        self.enter_w = np.asarray(fit.notes["r5_enter"]["coef"], dtype="float64")
        self.enter_b = float(fit.notes["r5_enter"]["intercept"])
        self.scale = float(fit.notes.get("r5_override_scale", 1.0))
        p0 = float(np.clip(fit.notes.get("r5_block_base", 0.02), 1e-6, 0.5))
        self.block_logit = float(np.log(p0 / (1.0 - p0)))
        self.dw = (self.exit_w - self.enter_w)[OVERRIDE_STATE_COLS]

    def draw_targets(self, prior, script, rng, avail):
        p = apply_min_target(prior.share, avail, self.fit.min_share)
        return p * script.total_slot

    def simulate(self, prior, script, rng):
        avail = draw_available(prior, rng)
        return run_resample(prior, script, avail, self.fit, self.donors, rng, override=self)
