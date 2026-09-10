"""
contract.py -- the engine-agnostic results contract.

Every sim engine, regardless of internal architecture (Control's GLM cascade,
a later possession-outcome engine, anything else), writes its graded output to
`results/<engine_tag>/` in ONE shape so a single set of gate/market scripts
(`scripts/eval_gates.py`, `scripts/grade_market_games.py`,
`scripts/grade_market_props.py`) can grade any of them without engine-specific
code. This generalises `src/cbb_sim/control/simulate.py` (`simulate()` writes
the per-(game, seed) rows; `run_control.py` writes `run_meta.json`) -- that
module is the prototype and is read here, never duplicated or imported.

results/<engine_tag>/
    games.parquet     required. One row per (game_id, seed).
    players.parquet   optional. One row per (game_id, seed, athlete_id).
    run_meta.json     required.

GAMES.PARQUET -- required columns
    game_id, seed, home_pts, away_pts, possessions, n_periods

    `n_periods` is 2 for a regulation game, 3 for one overtime, etc. -- the
    same convention `data/processed/games_universe.parquet` uses. LEGACY
    SHIM: the Control prototype instead writes `n_ot` (+ `went_ot`); when
    `n_periods` is absent but `n_ot` is present, `load_games()` derives
    `n_periods = 2 + n_ot` and records a warning rather than refusing the
    file -- this is how `results/control/*` is still gradeable by this
    harness (see deliverable 5's reproduction check).

GAMES.PARQUET -- optional per-team box columns
    Each of the seven stats below is checked as a HOME/AWAY pair (the same
    home_/away_ prefix convention as the required home_pts/away_pts columns).
    A gate that needs a pair and does not find BOTH sides reports
    NEEDS-INSTRUMENTATION rather than a partial or fabricated number.

        home_fga3      / away_fga3        3-point attempts
        home_fga2_rim   / away_fga2_rim    rim/dunk/layup 2-point attempts
        home_fga2_jump  / away_fga2_jump   jump-shot 2-point attempts
        home_fta       / away_fta         free throw attempts
        home_tov       / away_tov         turnovers
        home_oreb      / away_oreb        offensive rebounds
        home_dreb      / away_dreb        defensive rebounds

    NOTE (a known, deliberate contract gap): these are all ATTEMPT/rebound
    counts, not makes. `fga2_rim + fga2_jump + fga3` gives a full FGA count,
    which is enough for G3 (shot mix) and for the TOV%/OREB%/FT-rate legs of
    G4, but NOT for eFG% (which needs makes). Until an engine also reports
    make counts, G4's eFG% sub-check is unconditionally NEEDS-INSTRUMENTATION
    -- reported honestly as an instrumentation gap, never backed into from
    points scored.

PLAYERS.PARQUET -- optional; IF the file exists, these columns are required
    game_id, seed, athlete_id, team_id, minutes, pts, reb, ast, fga, fg3a, fta

    `team_id` is not in the deliverable's literal column list but is added
    here because G8's "by player role, by team" breakdown is impossible
    without it; every other column matches the spec verbatim.

RUN_META.JSON -- required keys
    engine_tag       str
    created_at       ISO-8601 timestamp, the run's own creation time
    seeds            int (a count) or a list of seed ints
    fold             str (e.g. "F2"), or "live" for a real-time run
    backtest         bool -- True for a completed season (created_at < tipoff
                     is impossible and is not checked); False for a live run,
                     which IS checked (see `validate_tipoff_safety`)
    sealed_touched   bool, expected False. True refuses to load unless the
                     caller passes `allow_sealed=True` (mirrors
                     `cbb_sim.data.seal.assert_not_sealed`'s fail-closed
                     default; season 2026 is sealed until fold-2 selection is
                     done, CLAUDE.md "backtests must be honest").

    LEGACY SHIM: `results/control/*/run_meta.json` predates this contract and
    has neither `engine_tag` nor `sealed_touched`. `load_run_meta()` fills
    `engine_tag` from the results directory name and `sealed_touched=False`
    (Control's fold-2 test season is 2025, never the sealed 2026 season), each
    with a recorded warning, rather than refusing a well-formed legacy run.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
REQUIRED_GAME_COLUMNS: tuple[str, ...] = (
    "game_id", "seed", "home_pts", "away_pts", "possessions", "n_periods",
)

# base name -> both home_<name> and away_<name> are the pair
BOX_STATS: tuple[str, ...] = (
    "fga3", "fga2_rim", "fga2_jump", "fta", "tov", "oreb", "dreb",
)

REQUIRED_PLAYER_COLUMNS: tuple[str, ...] = (
    "game_id", "seed", "athlete_id", "team_id",
    "minutes", "pts", "reb", "ast", "fga", "fg3a", "fta",
)

REQUIRED_RUN_META_KEYS: tuple[str, ...] = (
    "engine_tag", "created_at", "seeds", "fold", "backtest", "sealed_touched",
)

# legacy Control shape, see module docstring
_LEGACY_PERIOD_SRC = "n_ot"


class ContractError(RuntimeError):
    """A results artefact violates the contract and must be refused, never
    silently patched into a fake pass."""


@dataclass
class LoadResult:
    """A loaded, validated frame plus any non-fatal warnings raised getting
    there (legacy-shim usage, defaulted run_meta keys, etc.)."""

    frame: pd.DataFrame | dict
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# games.parquet
# ---------------------------------------------------------------------------
def _apply_legacy_period_shim(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warns: list[str] = []
    if "n_periods" in df.columns:
        return df, warns
    if _LEGACY_PERIOD_SRC in df.columns:
        df = df.copy()
        df["n_periods"] = 2 + df[_LEGACY_PERIOD_SRC].fillna(0).astype("int64")
        warns.append(
            "games.parquet has no 'n_periods' column; derived it from the legacy "
            f"'{_LEGACY_PERIOD_SRC}' column as n_periods = 2 + {_LEGACY_PERIOD_SRC} "
            "(pre-contract Control shape)."
        )
    return df, warns


def validate_games_frame(df: pd.DataFrame, strict: bool = True) -> list[str]:
    """Return the list of contract problems (missing required columns).
    Raises ContractError if `strict` and the list is non-empty."""
    missing = [c for c in REQUIRED_GAME_COLUMNS if c not in df.columns]
    problems = [f"games.parquet missing required column '{c}'" for c in missing]
    if problems and strict:
        raise ContractError("; ".join(problems))
    return problems


def box_columns_available(df: pd.DataFrame) -> dict[str, bool]:
    """Per optional box stat, whether BOTH home_<stat> and away_<stat> are
    present and carry at least one non-null value. A gate must check this
    before reading a box column and report NEEDS-INSTRUMENTATION, not KeyError
    or a fabricated zero, when it is False."""
    out = {}
    for stat in BOX_STATS:
        h, a = f"home_{stat}", f"away_{stat}"
        ok = h in df.columns and a in df.columns
        if ok:
            ok = bool(df[h].notna().any() and df[a].notna().any())
        out[stat] = ok
    return out


def load_games(results_dir: Path | str) -> LoadResult:
    path = Path(results_dir) / "games.parquet"
    if not path.exists():
        raise ContractError(f"{path} does not exist")
    df = pd.read_parquet(path)
    df, warns = _apply_legacy_period_shim(df)
    problems = validate_games_frame(df, strict=False)
    if problems:
        raise ContractError(f"{path}: " + "; ".join(problems))
    for w in warns:
        warnings.warn(w, stacklevel=2)
    return LoadResult(df, warns)


# ---------------------------------------------------------------------------
# players.parquet (optional)
# ---------------------------------------------------------------------------
def validate_players_frame(df: pd.DataFrame, strict: bool = True) -> list[str]:
    missing = [c for c in REQUIRED_PLAYER_COLUMNS if c not in df.columns]
    problems = [f"players.parquet missing required column '{c}'" for c in missing]
    if problems and strict:
        raise ContractError("; ".join(problems))
    return problems


def load_players(results_dir: Path | str) -> LoadResult:
    """Returns frame=None (no warnings) if players.parquet simply does not
    exist -- that is a legitimate, expected state (G8's NEEDS-INSTRUMENTATION
    case), not a contract violation. A players.parquet that DOES exist but is
    missing a required column IS a violation and raises."""
    path = Path(results_dir) / "players.parquet"
    if not path.exists():
        return LoadResult(None, [])
    df = pd.read_parquet(path)
    problems = validate_players_frame(df, strict=False)
    if problems:
        raise ContractError(f"{path}: " + "; ".join(problems))
    return LoadResult(df, [])


# ---------------------------------------------------------------------------
# run_meta.json
# ---------------------------------------------------------------------------
def load_run_meta(results_dir: Path | str) -> LoadResult:
    results_dir = Path(results_dir)
    path = results_dir / "run_meta.json"
    if not path.exists():
        raise ContractError(f"{path} does not exist")
    meta = json.loads(path.read_text(encoding="utf-8"))
    warns: list[str] = []

    if "engine_tag" not in meta:
        meta["engine_tag"] = results_dir.name
        warns.append(
            f"run_meta.json missing 'engine_tag'; defaulted to results dir name "
            f"'{results_dir.name}' (pre-contract Control run_meta.json)."
        )
    if "sealed_touched" not in meta:
        meta["sealed_touched"] = False
        warns.append(
            "run_meta.json missing 'sealed_touched'; defaulted to False "
            "(pre-contract Control run_meta.json)."
        )

    missing = [k for k in REQUIRED_RUN_META_KEYS if k not in meta]
    if missing:
        raise ContractError(f"{path}: missing required key(s) {missing}")

    for w in warns:
        warnings.warn(w, stacklevel=2)
    return LoadResult(meta, warns)


def validate_tipoff_safety(games: pd.DataFrame, run_meta: dict) -> list[str]:
    """CLAUDE.md 'backtests must be honest': every backtest row satisfies
    created_at < tipoff, enforced in code. A completed-season backtest is
    stamped `backtest=True` (the inequality is structurally impossible, so it
    is not checked, matching `scripts/run_control.py`'s own pattern); a live
    run (`backtest=False`) MUST prove created_at < tipoff on every row."""
    problems: list[str] = []
    if "backtest" not in run_meta:
        return ["run_meta missing 'backtest' flag; cannot determine whether "
                "created_at < tipoff needs to be checked"]
    if run_meta["backtest"]:
        return problems
    if "created_at" not in games.columns or "tipoff" not in games.columns:
        return ["live run (backtest=False) but games.parquet lacks per-row "
                "'created_at'/'tipoff' columns to verify created_at < tipoff"]
    bad = int((pd.to_datetime(games["created_at"]) >= pd.to_datetime(games["tipoff"])).sum())
    if bad:
        problems.append(f"live run: {bad} row(s) violate created_at < tipoff")
    return problems


# ---------------------------------------------------------------------------
# top-level loader
# ---------------------------------------------------------------------------
@dataclass
class EngineResults:
    engine_tag: str
    results_dir: Path
    games: pd.DataFrame
    players: pd.DataFrame | None
    run_meta: dict
    warnings: list[str]
    box_available: dict[str, bool]
    has_players: bool


def load_engine_results(results_dir: Path | str, allow_sealed: bool = False) -> EngineResults:
    """The one entry point every eval/market script uses. Raises ContractError
    on anything that must be refused; collects everything else (legacy shims,
    missing-optional-data notices) into `.warnings` for the report to print."""
    results_dir = Path(results_dir)
    all_warnings: list[str] = []

    meta_res = load_run_meta(results_dir)
    all_warnings += meta_res.warnings
    run_meta = meta_res.frame

    if run_meta.get("sealed_touched", False) and not allow_sealed:
        raise ContractError(
            f"{results_dir}/run_meta.json: sealed_touched=True; refusing to grade a run "
            "that touched the sealed season (pass allow_sealed=True to override deliberately)"
        )

    games_res = load_games(results_dir)
    all_warnings += games_res.warnings
    games = games_res.frame

    tipoff_problems = validate_tipoff_safety(games, run_meta)
    if tipoff_problems:
        raise ContractError(f"{results_dir}: " + "; ".join(tipoff_problems))

    players_res = load_players(results_dir)
    all_warnings += players_res.warnings
    players = players_res.frame
    if players is None:
        all_warnings.append("players.parquet not found; player-level gates (G8) and "
                             "grade_market_props.py will report NEEDS-INSTRUMENTATION.")

    box_available = box_columns_available(games)
    missing_box = [s for s, ok in box_available.items() if not ok]
    if missing_box:
        all_warnings.append(
            "games.parquet missing optional box column pair(s) for: "
            + ", ".join(missing_box) + "; the gates that need them report NEEDS-INSTRUMENTATION."
        )

    return EngineResults(
        engine_tag=run_meta["engine_tag"],
        results_dir=results_dir,
        games=games,
        players=players,
        run_meta=run_meta,
        warnings=all_warnings,
        box_available=box_available,
        has_players=players is not None,
    )
