"""
possession_outcome.py -- L3 POSSESSION-OUTCOME: the terminal-event mix of a
chance.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 1
(authored by the PM, 2026-09-10, before any modelling). Feature provenance:
`docs/models/possession_outcome/features.md`. Trainer:
`scripts/train_possession_outcome_v1.py`.

WHAT THIS MODEL IS. Given the offence, the defence, the site and the game
state, the probability that a chance ends in each of

    TOV, FGA_rim, FGA_jump2, FGA_3, FT_trip_shooting, FT_trip_bonus

It is the event-type MIX only. Whether a shot goes in, and how many free
throws are made, are separate models (`possession_make`, pre-registered next).
`end_period` chances are excluded from training and belong to the clock model
(L5). `unknown` chances (0.8-1.2%, the mismatch-guard residue documented in
`docs/tests/possessions_build_2026-09-10.md` section 7) are excluded as a data
gap, never imputed.

WHY A SINGLE CATEGORICAL AND NOT FOUR COUNT MODELS. L10 (`docs/LEARNINGS.md`):
the Control engine's four per-100 count models are individually calibrated but
their residuals correlate -0.56 / -0.36 / -0.29 because they compete for the
same possessions, and drawing them independently gave a margin SD ratio of
1.68 and a PIT K-S p of 2e-79. Shot mix must be drawn per possession as one
categorical outcome. That is what this model produces.

FIRST CHANCES AND CONTINUATION CHANCES ARE SEPARATE POPULATIONS. Measured on
2025: rim attempts are 25.1% of first chances and 31.6% of chances after an
offensive rebound, threes 29.3% vs 22.1%, turnovers 15.4% vs 11.4%. That is a
different distribution, not a rescaled one, so the pre-registration fits them
separately and this module carries the population as a first-class argument.

--------------------------------------------------------------------------
LEAK SAFETY
--------------------------------------------------------------------------
Every team-form feature is an EXPANDING mean over that team's own games
STRICTLY BEFORE the current game, built with a `shift(1)` inside
(season, team) ordered by (game_date, game_id) -- so a game's own events can
never enter its own features. `tests/test_possession_outcome.py` asserts this
directly by rebuilding one team's features with that game's rows deleted and
checking the value is bit-identical.

Every rate feature is expressed as a deviation from the league's own AS-OF
mean (also strictly-before), per `CLAUDE.md`: "Every rating feature is
expressed relative to its own snapshot's league mean. Raw levels are banned."
A team with no prior games in the season gets exactly 0.0 -- which IS the
league mean, not a fabricated value -- and the season-level drift that L11
measures is carried by the explicit season term of feature set B, never
smuggled in through an uncentred level.

The own ridge ratings (`cbb_sim.ratings.own_ratings`) are joined with
`join_as_of`, whose window is strictly before the game's date by construction.

--------------------------------------------------------------------------
SEAL
--------------------------------------------------------------------------
`cbb_sim.data.seal.assert_not_sealed` is called on every train slice and every
test slice this module produces. Season 2026 possession tables exist (they are
data, not a fit) but cannot enter a fold here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.data.seal import assert_not_sealed
from cbb_sim.ratings import own_ratings as orat

DEFAULT_POSS_DIR = Path("data/processed/possessions")
DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")

#: The six modelled classes, in a FIXED order. Every probability matrix in
#: this module and every artifact written by the trainer uses this order.
CLASSES: tuple[str, ...] = (
    "TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus",
)
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

#: Chance populations, fitted and scored separately (module docstring).
POPULATIONS: tuple[str, ...] = ("first", "cont")

FOLDS: dict[str, dict[str, list[int]]] = {
    "F1": {"train": [2022, 2023], "test": [2024]},
    "F2": {"train": [2022, 2023, 2024], "test": [2025]},
}
SELECTION_FOLD = "F2"


# ===========================================================================
# 1. Team form: as-of, strictly-before, league-centred
# ===========================================================================
#: The four offence rates of the pre-registration, each as (numerator,
#: denominator) so the expanding mean is a ratio of cumulative sums rather
#: than a mean of per-game ratios (stable when a team's early games are few).
RATE_DEFS: dict[str, tuple[str, str]] = {
    "3pa": ("fga_3", "poss"),        # three-point attempts per possession
    "rim": ("fga_rim", "fga"),       # rim share of field-goal attempts
    "tov": ("tov", "poss"),          # turnover rate per possession
    "ftr": ("fta", "fga"),           # free-throw rate
}
RATE_SCALE: dict[str, float] = {"3pa": 100.0, "rim": 100.0, "tov": 100.0, "ftr": 100.0}


def team_game_box(poss: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offence team): the accumulated box line this model's
    rates are built from. Both sides of every game appear."""
    p = poss.copy()
    p["fga"] = p["fga_rim"] + p["fga_jump2"] + p["fga_3"]
    p["tov"] = (p["terminal_event"] == "TOV").astype("int32")
    g = p.groupby(["season", "game_id", "offense_team_id", "defense_team_id"], as_index=False).agg(
        poss=("poss_index", "count"),
        fga=("fga", "sum"), fga_3=("fga_3", "sum"), fga_rim=("fga_rim", "sum"),
        fga_jump2=("fga_jump2", "sum"), fta=("fta", "sum"), tov=("tov", "sum"),
        points=("points", "sum"),
    )
    return g.rename(columns={"offense_team_id": "team_id", "defense_team_id": "opp_id"})


def _expanding_asof(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Cumulative sums of `cols` over rows STRICTLY BEFORE each row, within
    `keys`, in the frame's current order. This is the shift(1) expanding
    builder: `cumsum() - value` is the sum over earlier rows only."""
    grp = df.groupby(keys, sort=False)
    out = {}
    for c in cols:
        out[c] = grp[c].cumsum().to_numpy() - df[c].to_numpy()
    out["n_prior"] = grp.cumcount().to_numpy()
    return pd.DataFrame(out, index=df.index)


def build_team_form(poss_by_season: dict[int, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    """As-of, league-centred offence and defence-allowed form for every
    (game, team). One row per team-game; both sides of every game.

    Returns columns
        season, game_id, team_id, opp_id, game_date,
        off_{rate}_c, def_{rate}_c   for rate in RATE_DEFS,
        n_prior_off
    where `_c` is the team's own as-of rate MINUS the league's as-of rate on
    the same date (see LEAK SAFETY in the module docstring)."""
    boxes = pd.concat([team_game_box(p) for p in poss_by_season.values()], ignore_index=True)
    dates = universe[["game_id", "game_date"]].copy()
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    boxes = boxes.merge(dates, on="game_id", how="left")
    boxes = boxes.sort_values(["season", "game_date", "game_id"], kind="stable").reset_index(drop=True)

    num_cols = sorted({n for n, _ in RATE_DEFS.values()})
    den_cols = sorted({d for _, d in RATE_DEFS.values()})
    all_cols = sorted(set(num_cols) | set(den_cols))

    # --- offence: the team's own accumulated line -------------------------
    off = boxes.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    off_asof = _expanding_asof(off, ["season", "team_id"], all_cols)
    off_asof.columns = [f"off_{c}" for c in off_asof.columns]
    off = pd.concat([off[["season", "game_id", "team_id", "opp_id", "game_date"]], off_asof], axis=1)

    # --- defence-allowed: what opponents did against this team ------------
    dfd = boxes.rename(columns={"team_id": "_off", "opp_id": "team_id"})
    dfd = dfd.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    def_asof = _expanding_asof(dfd, ["season", "team_id"], all_cols)
    def_asof.columns = [f"def_{c}" for c in def_asof.columns]
    dfd = pd.concat([dfd[["season", "game_id", "team_id"]], def_asof], axis=1)

    form = off.merge(dfd, on=["season", "game_id", "team_id"], how="left")

    # --- league as-of: cumulative over ALL team-games strictly earlier -----
    day = boxes.groupby(["season", "game_date"], as_index=False)[all_cols].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = _expanding_asof(day, ["season"], all_cols)
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    form = form.merge(day, on=["season", "game_date"], how="left")

    def _rate(num: np.ndarray, den: np.ndarray, scale: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, scale * num / np.maximum(den, 1e-9), np.nan)
        return r

    for name, (num, den) in RATE_DEFS.items():
        s = RATE_SCALE[name]
        lg_rate = _rate(form[f"lg_{num}"].to_numpy(), form[f"lg_{den}"].to_numpy(), s)
        for sidepfx in ("off", "def"):
            own = _rate(form[f"{sidepfx}_{num}"].to_numpy(), form[f"{sidepfx}_{den}"].to_numpy(), s)
            # A team (or a league) with no prior games sits at exactly the
            # league mean: the centred feature is 0.0, never a fabricated level.
            centred = np.where(np.isnan(own) | np.isnan(lg_rate), 0.0, own - lg_rate)
            form[f"{sidepfx}_{name}_c"] = centred.astype("float32")

    form["n_prior_off"] = form["off_n_prior"].astype("int32")
    keep = ["season", "game_id", "team_id", "opp_id", "game_date", "n_prior_off"]
    keep += [f"{s}_{r}_c" for s in ("off", "def") for r in RATE_DEFS]
    return form[keep]


# ===========================================================================
# 2. Chance-level design matrix
# ===========================================================================
BASE_FEATURES: tuple[str, ...] = (
    "off_3pa_c", "off_rim_c", "off_tov_c", "off_ftr_c",          # offence form
    "opp_def_3pa_c", "opp_def_rim_c", "opp_def_tov_c", "opp_def_ftr_c",  # defence allowed
    "off_rating_off_c", "off_rating_def_c",                       # own ridge ratings, offence
    "def_rating_off_c", "def_rating_def_c",                       # own ridge ratings, defence
    "site_home", "site_away",                                     # neutral is the reference level
)
SEASON_FEATURES: tuple[str, ...] = ("season_idx", "days_since_start")
STATE_FEATURES: tuple[str, ...] = (
    "period", "seconds_remaining", "score_diff", "in_bonus", "is_transition",
)
CHANCE_NUMBER_FEATURE = "chance_number"
INTERACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("off_3pa_c", "opp_def_3pa_c"),
    ("off_rim_c", "opp_def_rim_c"),
    ("off_tov_c", "opp_def_tov_c"),
    ("off_ftr_c", "opp_def_ftr_c"),
    ("off_rating_off_c", "def_rating_def_c"),
)


def feature_set(name: str, population: str) -> list[str]:
    """The pre-registered feature bundles. `chance_number` is only included in
    the `cont` population -- it is identically 1 in `first` and a constant
    column is not a feature."""
    a = list(BASE_FEATURES)
    if name == "A_team":
        return a
    b = a + list(SEASON_FEATURES)
    if name == "B_plus_season":
        return b
    c = b + list(STATE_FEATURES)
    if population == "cont":
        c = c + [CHANCE_NUMBER_FEATURE]
    if name == "C_plus_state":
        return c
    if name == "D_plus_interactions":
        return c + [f"x_{u}__{v}" for u, v in INTERACTION_PAIRS]
    raise KeyError(f"unknown feature set {name!r}")


FEATURE_SETS: tuple[str, ...] = ("A_team", "B_plus_season", "C_plus_state", "D_plus_interactions")


def load_chances(seasons: list[int], poss_dir: Path | str = DEFAULT_POSS_DIR) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = Path(poss_dir) / f"chances_{int(s)}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing chances table: {p} (run scripts/build_possessions.py)")
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)


def build_design(
    seasons: list[int],
    poss_dir: Path | str = DEFAULT_POSS_DIR,
    universe_path: Path | str = DEFAULT_UNIVERSE,
    ratings_dir: Path | str = "data/processed/ratings",
) -> pd.DataFrame:
    """One row per modelled chance with every candidate feature attached.

    Rows dropped: `end_period` chances (the clock model's job, per the
    pre-registration) and `unknown` chances (a data gap, never imputed)."""
    seasons = [int(s) for s in seasons]
    universe = pd.read_parquet(universe_path)
    universe = universe[universe["is_d1_game"] & ~universe["pbp_truncated"]].copy()
    universe["game_date"] = pd.to_datetime(universe["game_date"])

    poss_by_season = {}
    for s in seasons:
        p = Path(poss_dir) / f"possessions_{s}.parquet"
        poss_by_season[s] = pd.read_parquet(
            p, columns=["season", "game_id", "offense_team_id", "defense_team_id", "poss_index",
                        "terminal_event", "fga_rim", "fga_jump2", "fga_3", "fta", "points"])
    form = build_team_form(poss_by_season, universe)

    ch = load_chances(seasons, poss_dir=poss_dir)
    ch = ch[ch["terminal_event"].isin(CLASSES)].copy()

    ch = ch.merge(
        universe[["game_id", "game_date", "neutral_site", "home_team_id"]],
        on="game_id", how="inner")

    # --- team form, offence side and defence side -------------------------
    off_form = form.rename(columns={"team_id": "offense_team_id"})
    off_cols = ["season", "game_id", "offense_team_id", "n_prior_off"] + \
               [f"off_{r}_c" for r in RATE_DEFS]
    ch = ch.merge(off_form[off_cols], on=["season", "game_id", "offense_team_id"], how="left")

    def_form = form.rename(columns={"team_id": "defense_team_id"})
    def_cols = ["season", "game_id", "defense_team_id"] + [f"def_{r}_c" for r in RATE_DEFS]
    def_form = def_form[def_cols].rename(columns={f"def_{r}_c": f"opp_def_{r}_c" for r in RATE_DEFS})
    ch = ch.merge(def_form, on=["season", "game_id", "defense_team_id"], how="left")

    # --- own ridge ratings, as-of, both sides -----------------------------
    ratings = orat.load_ratings(sorted(set(seasons)), out_dir=ratings_dir)
    rcols = ("off_c", "def_c")
    ch = orat.join_as_of(ch, ratings, team_col="offense_team_id", date_col="game_date",
                         suffix="__offteam", cols=rcols)
    ch = orat.join_as_of(ch, ratings, team_col="defense_team_id", date_col="game_date",
                         suffix="__defteam", cols=rcols)
    ch = ch.rename(columns={
        "off_c__offteam": "off_rating_off_c", "def_c__offteam": "off_rating_def_c",
        "off_c__defteam": "def_rating_off_c", "def_c__defteam": "def_rating_def_c",
    })

    # --- site, from the OFFENCE's point of view; neutral is the reference --
    neutral = ch["neutral_site"].to_numpy()
    off_home = ch["offense_is_home"].to_numpy()
    ch["site_home"] = ((~neutral) & off_home).astype("float32")
    ch["site_away"] = ((~neutral) & (~off_home)).astype("float32")

    # --- season / calendar ------------------------------------------------
    ch["season_idx"] = (ch["season"] - 2022).astype("float32")
    first_day = ch.groupby("season")["game_date"].transform("min")
    ch["days_since_start"] = (ch["game_date"] - first_day).dt.days.astype("float32")

    # --- game state at the chance's start ---------------------------------
    ch["period"] = ch["period"].astype("float32")
    ch["seconds_remaining"] = ch["start_clock"].astype("float32")
    ch["score_diff"] = ch["start_score_diff"].astype("float32")
    ch["in_bonus"] = ch["off_in_bonus"].astype("float32")
    ch["is_transition"] = ch["is_transition"].astype("float32")
    ch["chance_number"] = ch["chance_number"].astype("float32")

    # A handful of rows (0.03%) fail one of the form joins -- a team-season
    # whose only appearance on one side of the ball is in a game the other
    # table dropped. They fall back to exactly 0.0, which on a league-centred
    # scale IS the league mean, never a fabricated level.
    fill_cols = [f"off_{r}_c" for r in RATE_DEFS] + [f"opp_def_{r}_c" for r in RATE_DEFS] + [
        "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c"]
    for c in fill_cols:
        ch[c] = ch[c].astype("float32").fillna(0.0)

    for u, v in INTERACTION_PAIRS:
        ch[f"x_{u}__{v}"] = (ch[u].astype("float32") * ch[v].astype("float32")).astype("float32")

    ch["y"] = ch["terminal_event"].map(CLASS_INDEX).astype("int8")
    ch["population"] = np.where(ch["chance_number"].to_numpy() > 1, "cont", "first")
    return ch


def fold_slices(design: pd.DataFrame, fold: str, population: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) slices for a fold and population, with the seal guard on
    BOTH -- season 2026 can never enter a fold."""
    spec = FOLDS[fold]
    assert_not_sealed(spec["train"], context=f"{fold} train seasons")
    assert_not_sealed(spec["test"], context=f"{fold} test seasons")
    d = design[design["population"] == population]
    tr = d[d["season"].isin(spec["train"])]
    te = d[d["season"].isin(spec["test"])]
    assert_not_sealed(tr, context=f"{fold} train slice")
    assert_not_sealed(te, context=f"{fold} test slice")
    return tr, te


# ===========================================================================
# 3. Model arms
# ===========================================================================
@dataclass
class FittedArm:
    name: str
    feature_set: str
    population: str
    fold: str
    features: list[str]
    model: object
    scaler: object | None = None
    meta: dict | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


def _matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.ascontiguousarray(df[features].to_numpy(dtype="float32"))


class BaselineArm:
    """Matchup-naive floor: the league-average class shares of the MOST RECENT
    TRAINING season.

    The pre-registration says "league-average shares by season". Using the TEST
    season's own shares would be an oracle, so the honest realisation of "by
    season" is the last season you actually have at prediction time -- 2023 for
    F1, 2024 for F2. The oracle version is also reported by the trainer,
    explicitly labelled as unattainable, because the gap between the two IS the
    season-drift cost that L11 measures and that feature set B is meant to
    recover."""

    def __init__(self, oracle: bool = False):
        self.oracle = oracle
        self.p_: np.ndarray | None = None

    def fit(self, y: np.ndarray, seasons: np.ndarray) -> BaselineArm:
        last = int(np.max(seasons))
        sel = seasons == last
        counts = np.bincount(y[sel], minlength=len(CLASSES)).astype("float64")
        self.p_ = counts / counts.sum()
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.repeat(self.p_[None, :], n, axis=0)


class RidgeLogitArm:
    """Multinomial ridge logit. Features are standardised on the TRAIN slice
    only; the fitted means/SDs are persisted with the model so the sim applies
    the identical transform (`model.md` section 7)."""

    def __init__(self, C: float = 1.0, max_iter: int = 200, seed: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeLogitArm:
        from sklearn.linear_model import LogisticRegression

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        Z = (X - self.mu_) / self.sd_
        self.clf_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs", random_state=self.seed,
        ).fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu_) / self.sd_
        p = self.clf_.predict_proba(Z)
        return _align_classes(p, self.clf_.classes_)


class CascadeArm:
    """Nested binary GLMs, the pre-registered alternative decomposition:

        1. TOV?                      (turnover vs anything else)
        2. | not TOV: foul trip?     (a free-throw trip vs a field-goal attempt)
        3. | foul trip: bonus?       (FT_trip_bonus vs FT_trip_shooting)
        4. | field goal: three?      (FGA_3 vs a two-point attempt)
        5. | two-point: rim?         (FGA_rim vs FGA_jump2)

    Five binary logits on the SAME feature set, composed by the chain rule.
    This is a genuinely different functional form from the multinomial (each
    split gets its own coefficient vector and its own intercept), which is why
    the pre-registration lists it as its own model class rather than as a
    reparametrisation."""

    STEPS = ("tov", "trip", "bonus", "three", "rim")

    def __init__(self, C: float = 1.0, max_iter: int = 200, seed: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    @staticmethod
    def _targets(y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        i = CLASS_INDEX
        is_tov = y == i["TOV"]
        is_trip = np.isin(y, [i["FT_trip_shooting"], i["FT_trip_bonus"]])
        is_fg = np.isin(y, [i["FGA_rim"], i["FGA_jump2"], i["FGA_3"]])
        return {
            "tov": (np.ones_like(y, dtype=bool), is_tov),
            "trip": (~is_tov, is_trip),
            "bonus": (is_trip, y == i["FT_trip_bonus"]),
            "three": (is_fg, y == i["FGA_3"]),
            "rim": (is_fg & (y != i["FGA_3"]), y == i["FGA_rim"]),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> CascadeArm:
        from sklearn.linear_model import LogisticRegression

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        Z = (X - self.mu_) / self.sd_
        tg = self._targets(y)
        self.models_ = {}
        self.rates_ = {}
        for step in self.STEPS:
            mask, target = tg[step]
            zz, tt = Z[mask], target[mask]
            self.rates_[step] = float(tt.mean()) if len(tt) else 0.5
            if len(np.unique(tt)) < 2:
                self.models_[step] = None
                continue
            self.models_[step] = LogisticRegression(
                C=self.C, max_iter=self.max_iter, solver="lbfgs", random_state=self.seed,
            ).fit(zz, tt)
        return self

    def _p(self, step: str, Z: np.ndarray) -> np.ndarray:
        m = self.models_[step]
        if m is None:
            return np.full(len(Z), self.rates_[step])
        p = m.predict_proba(Z)
        pos = int(np.where(m.classes_ == 1)[0][0]) if 1 in m.classes_ else 1
        return p[:, pos]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu_) / self.sd_
        p_tov = self._p("tov", Z)
        p_trip = self._p("trip", Z)
        p_bonus = self._p("bonus", Z)
        p_three = self._p("three", Z)
        p_rim = self._p("rim", Z)
        rest = 1.0 - p_tov
        trip = rest * p_trip
        fg = rest * (1.0 - p_trip)
        out = np.empty((len(Z), len(CLASSES)), dtype="float64")
        i = CLASS_INDEX
        out[:, i["TOV"]] = p_tov
        out[:, i["FT_trip_bonus"]] = trip * p_bonus
        out[:, i["FT_trip_shooting"]] = trip * (1.0 - p_bonus)
        out[:, i["FGA_3"]] = fg * p_three
        out[:, i["FGA_rim"]] = fg * (1.0 - p_three) * p_rim
        out[:, i["FGA_jump2"]] = fg * (1.0 - p_three) * (1.0 - p_rim)
        return out


class LgbmArm:
    """LightGBM multiclass. Hyperparameters are held FIXED across every
    feature set and fold so that the grid compares feature sets and model
    classes, not tuning effort."""

    PARAMS = dict(
        objective="multiclass", num_class=len(CLASSES), n_estimators=400,
        learning_rate=0.06, num_leaves=63, min_child_samples=400,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        reg_lambda=1.0, verbose=-1,
    )

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> LgbmArm:
        import lightgbm as lgb

        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, **self.PARAMS)
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _align_classes(self.clf_.predict_proba(X), self.clf_.classes_)


def _align_classes(p: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Reorder a sklearn probability matrix into CLASSES order, filling any
    class absent from the training slice with zeros (then renormalised)."""
    out = np.zeros((p.shape[0], len(CLASSES)), dtype="float64")
    for j, c in enumerate(classes):
        out[:, int(c)] = p[:, j]
    s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-12)


ARMS: tuple[str, ...] = ("baseline", "ridge_logit", "cascade", "lgbm")


def fit_arm(arm: str, tr: pd.DataFrame, features: list[str], seed: int = 0):
    y = tr["y"].to_numpy()
    if arm == "baseline":
        return BaselineArm().fit(y, tr["season"].to_numpy())
    X = _matrix(tr, features)
    if arm == "ridge_logit":
        return RidgeLogitArm(seed=seed).fit(X, y)
    if arm == "cascade":
        return CascadeArm(seed=seed).fit(X, y)
    if arm == "lgbm":
        return LgbmArm(seed=seed).fit(X, y)
    raise KeyError(f"unknown arm {arm!r}")


def predict_arm(arm: str, model, te: pd.DataFrame, features: list[str]) -> np.ndarray:
    if arm == "baseline":
        return model.predict_proba(len(te))
    return model.predict_proba(_matrix(te, features))


# ===========================================================================
# 4. Metrics
# ===========================================================================
EPS = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean())


def per_class_brier(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {}
    for j, c in enumerate(CLASSES):
        o = (y == j).astype("float64")
        out[c] = float(((p[:, j] - o) ** 2).mean())
    return out


def decile_calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, dict]:
    """Per class: predicted vs actual rate inside deciles of the predicted
    probability, plus the maximum absolute gap in percentage points."""
    out = {}
    for j, c in enumerate(CLASSES):
        pj = p[:, j]
        oj = (y == j).astype("float64")
        edges = np.quantile(pj, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        idx = np.clip(np.searchsorted(edges, pj, side="right") - 1, 0, n_bins - 1)
        rows = []
        worst = 0.0
        gaps = []
        for b in range(n_bins):
            m = idx == b
            if m.sum() == 0:
                continue
            pred = float(pj[m].mean())
            act = float(oj[m].mean())
            gap = (pred - act) * 100
            rows.append({"bin": b, "n": int(m.sum()), "pred": round(pred, 5),
                         "actual": round(act, 5), "gap_pp": round(gap, 3)})
            gaps.append(gap)
            worst = max(worst, abs(gap))
        # DIAGNOSTIC, NOT A CORRECTION. A miscalibration that is the SAME in
        # every decile is a level error (the class's overall rate is wrong);
        # one that varies across deciles is a shape error (the model orders
        # chances wrongly). They point at different fixes -- a season term or a
        # recency-weighted refit for the first, a different model class or
        # feature for the second -- so `docs/SIM_GUARDRAILS.md` section 5's
        # "which sub-model is producing the wrong distribution" question needs
        # both numbers. Nothing here changes a prediction.
        level = float(np.mean(gaps)) if gaps else 0.0
        residual = max((abs(g - level) for g in gaps), default=0.0)
        out[c] = {"share_pct": round(float(oj.mean() * 100), 3),
                  "max_abs_gap_pp": round(worst, 3),
                  "level_shift_pp": round(level, 3),
                  "max_abs_gap_pp_after_level_shift": round(residual, 3),
                  "bins": rows}
    return out


RESPONSIVENESS_SPECS: tuple[tuple[str, str], ...] = (
    ("off_3pa_c", "FGA_3"),
    ("off_rim_c", "FGA_rim"),
    ("off_tov_c", "TOV"),
)


def responsiveness(te: pd.DataFrame, p: np.ndarray, n_q: int = 5) -> dict[str, dict]:
    """The matchup-specific rule (`CLAUDE.md`): bucket the test rows by the
    offence's own as-of rate quintile and check that the PREDICTED class share
    slopes with the ACTUAL one instead of sitting flat at the league mean.

    Reported per (driver feature, class): predicted and actual share per
    quintile, the number of the four quintile steps in which each moves in the
    same direction as the driver, and the predicted-vs-actual slope ratio."""
    out = {}
    y = te["y"].to_numpy()
    for feat, cls in RESPONSIVENESS_SPECS:
        j = CLASS_INDEX[cls]
        v = te[feat].to_numpy()
        edges = np.quantile(v, np.linspace(0, 1, n_q + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        q = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_q - 1)
        pred, act, ns = [], [], []
        for b in range(n_q):
            m = q == b
            ns.append(int(m.sum()))
            pred.append(float(p[m, j].mean()) if m.sum() else np.nan)
            act.append(float((y[m] == j).mean()) if m.sum() else np.nan)
        pred_a, act_a = np.array(pred), np.array(act)
        dp, da = np.diff(pred_a), np.diff(act_a)
        span_pred = float(pred_a[-1] - pred_a[0])
        span_act = float(act_a[-1] - act_a[0])
        # "Monotone" is judged against the ACTUAL direction of travel across
        # the quintiles, so an arm that slopes the right way but flatter than
        # reality still passes the SHAPE test and is caught by slope_ratio
        # instead -- flatness and wrong-direction are different failures.
        ref = np.sign(span_act) if span_act else np.sign(span_pred)
        out[f"{feat}->{cls}"] = {
            "n": ns,
            "pred_share": [round(x, 5) for x in pred],
            "actual_share": [round(x, 5) for x in act],
            "steps_agreeing": int(np.sum(np.sign(dp) == np.sign(da))),
            "pred_monotone_steps": int(np.sum(np.sign(dp) == ref)),
            "actual_monotone_steps": int(np.sum(np.sign(da) == ref)),
            "n_steps": int(len(dp)),
            "span_pred": round(span_pred, 5),
            "span_actual": round(span_act, 5),
            "slope_ratio": round(span_pred / span_act, 4) if span_act else None,
        }
    return out


#: The pre-registration says "monotone in 4 of 5 quintile steps". Five
#: quintiles give FOUR steps, so the faithful reading of "allow one violation"
#: is 3 of 4, and that is the gate the trainer applies. Both the raw step count
#: and the slope ratio are reported so the reading stays auditable.
RESPONSIVENESS_MIN_STEPS = 3


STATE_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("transition", "is_transition == 1"),
    ("half_court", "is_transition == 0"),
    ("bonus", "in_bonus == 1"),
    ("no_bonus", "in_bonus == 0"),
    ("late_clock", "seconds_remaining <= 120"),
    ("not_late_clock", "seconds_remaining > 120"),
)


def by_state_calibration(te: pd.DataFrame, p: np.ndarray) -> dict[str, dict]:
    """Predicted vs actual class share inside each pre-registered game-state
    segment, and the segment's own log loss."""
    y = te["y"].to_numpy()
    out = {}
    for name, expr in STATE_SEGMENTS:
        m = te.eval(expr).to_numpy()
        if m.sum() == 0:
            continue
        seg = {"n": int(m.sum()), "log_loss": round(log_loss(y[m], p[m]), 5)}
        for j, c in enumerate(CLASSES):
            seg[f"{c}_pred_pct"] = round(float(p[m, j].mean() * 100), 3)
            seg[f"{c}_actual_pct"] = round(float((y[m] == j).mean() * 100), 3)
            seg[f"{c}_gap_pp"] = round(seg[f"{c}_pred_pct"] - seg[f"{c}_actual_pct"], 3)
        seg["max_abs_gap_pp"] = round(max(abs(seg[f"{c}_gap_pp"]) for c in CLASSES), 3)
        out[name] = seg
    return out


def block_bootstrap_se(te: pd.DataFrame, p: np.ndarray, n_rep: int = 200, seed: int = 12345) -> float:
    """Game-level block bootstrap SE of the multiclass log loss.

    Chances inside one game are not independent -- the same lineups, the same
    officials, the same pace -- so the resampling unit is the GAME, not the
    chance. This is the noise floor for the linear arms (the pre-registration
    uses seed-varied refits for the tree arm, which has its own randomness)."""
    rng = np.random.default_rng(seed)
    y = te["y"].to_numpy()
    games = te["game_id"].to_numpy()
    order = np.argsort(games, kind="stable")
    g_sorted = games[order]
    starts = np.flatnonzero(np.concatenate([[True], g_sorted[1:] != g_sorted[:-1]]))
    ends = np.concatenate([starts[1:], [len(g_sorted)]])
    blocks = [order[s:e] for s, e in zip(starts, ends, strict=False)]
    n_g = len(blocks)
    losses = np.empty(n_rep)
    ll_row = -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))
    block_sums = np.array([ll_row[b].sum() for b in blocks])
    block_ns = np.array([len(b) for b in blocks])
    for r in range(n_rep):
        pick = rng.integers(0, n_g, n_g)
        losses[r] = block_sums[pick].sum() / block_ns[pick].sum()
    return float(losses.std(ddof=1))
