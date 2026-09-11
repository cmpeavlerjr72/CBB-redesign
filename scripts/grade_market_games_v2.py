"""
grade_market_games_v2.py -- the market scorecard, reading the ACCEPTED lines
source (`data/processed/lines/lines_close_v1.parquet`) instead of the raw
CBBD pull `scripts/grade_market_games.py` (v1, untouched, never overwritten)
reads directly from `data/raw/cbbd/lines_{season}.parquet`.

    .venv/Scripts/python.exe scripts/grade_market_games_v2.py \
        --results results/engine_v0/<tag> --season <season>

Why a v2 rather than an edit to v1: CLAUDE.md's scripts convention says
trainers/graders are versioned filenames, never overwritten in place; v1
stays as the historical record of the pre-lines-validation provisional runs
(e.g. `docs/tests/market_games_engine_v0_F2_2025_s5_r2event_2026-09-10.md`,
generated the same day at 5 seeds with no seed-count guard at all -- read
before this script was written, see `docs/tests/engine_seed_count_2026-09-10.md`
section 4 for why that run's ROI/Brier numbers must be struck).

FOUR things this script does that v1 does not:

  1. SEED-COUNT GUARD (CLAUDE.md "a seed-count study fixes the minimum seeds
     before any ROI number is read"). Below 200 seeds every table in the
     report is labelled PROVISIONAL. Below ~2,000 seeds (the threshold
     `docs/tests/engine_seed_count_2026-09-10.md` measured for a per-game
     ROI/Brier/calibration read) the script REFUSES to print ROI, Brier,
     calibration or edge-bucket hit-rate numbers at all -- it prints the run's
     seed count and the study's threshold instead. This is not a cosmetic
     label: at 5 seeds a game's simulated win probability can only take values
     in {0, 0.2, 0.4, 0.6, 0.8, 1.0}, and the per-game margin-mean seed noise
     (14.85 pts at 5 seeds) is larger than the quantity a Brier score or an
     ROI number is trying to measure.

  2. LINES SOURCE. Reads `data/processed/lines/lines_close_v1.parquet` --
     the validated, de-vigged, ESPN-BET-primary table built and checked in
     `docs/tests/lines_cbbd_validation_2026-09-10.md` (+ its addendum) --
     instead of the raw per-season CBBD pull. `provider == "ESPN BET"` is the
     only book present in every target season and the primary provider
     `lines_close_v1` was built around; proportional de-vig
     (`p_home_close_prop`) is the primary probability, power de-vig
     (`p_home_close_power`) is reported alongside, never used for settlement.

  3. SECOND-SOURCE TRUTH. Actual home/away scores come from
     `data/processed/truth/game_finals_v2.parquet` ONLY -- never the engine's
     own output, never a pbp-accumulated total, and never
     `games_universe`'s own score columns (used here only for schedule
     metadata: date, month, neutral site, team ids). Games `game_finals_v2`
     flags as an unresolved cross-source disagreement (a nonzero score diff
     that was never checked against a third source) are excluded and counted,
     not silently kept.

  4. CREATED_AT/TIPOFF. Reads `created_at`/`tipoff` from the results contract
     per row if the columns exist and asserts `created_at < tipoff` on every
     row; if they do not exist (true of every engine_v0 backtest run today --
     the contract only carries a run-level `created_at`), the report is
     labelled "created_at not recorded" at the top rather than silently
     skipping the check.

Everything else -- edge buckets (spread/total/moneyline), the 2025-only
open-vs-close leak cross-check, the season/month/conference/own-rating-
quintile breakdowns -- is new relative to v1, which only ever produced a
single pooled table.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import market as M  # noqa: E402
from cbb_sim.eval import reference as ref_mod  # noqa: E402
from cbb_sim.eval import report as R  # noqa: E402
from cbb_sim.eval.cli import load_tolerances, tag_from_results_dir  # noqa: E402
from cbb_sim.ratings import own_ratings as orat  # noqa: E402

OUT_DIR = Path("docs/tests")
DEFAULT_LINES_PATH = Path("data/processed/lines/lines_close_v1.parquet")
DEFAULT_CBBD_GAMES_DIR = Path("data/raw/cbbd")
PRIMARY_PROVIDER = "ESPN BET"

# ---------------------------------------------------------------------------
# seed-count guard (docs/tests/engine_seed_count_2026-09-10.md)
# ---------------------------------------------------------------------------
GATE_MIN_SEEDS = 200          # section 3: "200 seeds for the gate report"
ROI_BRIER_MIN_SEEDS = 2000    # section 3: "no ROI or Brier number is quoted from a run under ~2,000 seeds"

# Measured per-game SE at each rung, section 2's ladder table (mean over games).
# Used verbatim when n_seeds matches a rung exactly; otherwise the same
# c/sqrt(k) law the study itself uses to extrapolate ("seeds required" table)
# is applied from the nearest measured rung, and the report says so.
SEED_STUDY_SE_MEAN: dict[int, dict[str, float]] = {
    5:   {"margin": 14.850, "total": 6.434, "winprob": 0.2164},
    10:  {"margin": 10.275, "total": 4.540, "winprob": 0.1503},
    25:  {"margin": 6.364,  "total": 2.833, "winprob": 0.0938},
    50:  {"margin": 4.282,  "total": 1.874, "winprob": 0.0630},
    100: {"margin": 3.032,  "total": 1.185, "winprob": 0.0421},
}


def seed_se(n_seeds: int, quantity: str) -> tuple[float, bool]:
    """(SE, exact) -- SE of the sim mean at `n_seeds`, `exact=True` if
    `n_seeds` is one of the study's measured rungs, else an extrapolation via
    the study's own fitted c/sqrt(k) law from the nearest rung."""
    if n_seeds in SEED_STUDY_SE_MEAN:
        return SEED_STUDY_SE_MEAN[n_seeds][quantity], True
    nearest = min(SEED_STUDY_SE_MEAN, key=lambda k: abs(k - n_seeds))
    c = SEED_STUDY_SE_MEAN[nearest][quantity] * (nearest ** 0.5)
    return c / (n_seeds ** 0.5), False


# ---------------------------------------------------------------------------
# Wilson interval (not in market.py -- market.py's tables report a bootstrap
# CI on ROI; edge buckets here also want a closed-form CI on the hit RATE)
# ---------------------------------------------------------------------------
CANDIDATE_ARTIFACT_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("event.first", ("adapter_flags", "sources", "event", "first")),
    ("event.cont", ("adapter_flags", "sources", "event", "cont")),
    ("event.team_block", ("adapter_flags", "sources", "event", "team_block")),
    ("clock", ("adapter_flags", "sources", "clock")),
    ("fg_make.FGA_rim", ("adapter_flags", "sources", "fg_make", "FGA_rim")),
    ("fg_make.FGA_jump2", ("adapter_flags", "sources", "fg_make", "FGA_jump2")),
    ("fg_make.FGA_3", ("adapter_flags", "sources", "fg_make", "FGA_3")),
    ("free_throw", ("adapter_flags", "sources", "free_throw")),
    ("rebound", ("adapter_flags", "sources", "rebound")),
    ("usage", ("adapter_flags", "sources", "usage")),
    ("rotation", ("adapter_flags", "sources", "rotation")),
)


def _dig(d: dict, path: tuple[str, ...]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


#: The per-family artifact-date block `run_engine.py` has written since
#: 2026-09-11 (`Adapters._train_dates`). It is the PREFERRED source because it
#: covers EVERY family the engine served -- including rotation, and including
#: families whose `sources` entry never carried dates -- with one entry per
#: artifact, so the per-row assignment below needs no guessing.
TRAIN_DATE_BLOCK = ("adapter_flags", "max_train_date")


def collect_family_max_train_dates(run_meta: dict) -> tuple[dict, list[str]]:
    """Split the run's sub-models into those that DO carry recorded artifact
    training-window ends and those that do NOT.

    Preferred source: `adapter_flags["max_train_date"]`, which carries every
    family the engine served, keyed by population / shot class, with a
    `refit_date` and a `max_train_date` per artifact. Older runs did not write
    it, so the historical walk over `adapter_flags["sources"]` stays as the
    fallback and any family missing from BOTH is reported by name.

    A family recorded as `scheme: "static"` with a null `max_train_date` is NOT
    silently passed and NOT fabricated a date: it is reported as missing, which
    is the honest reading -- a single fitted object's training window is the
    fold's train seasons and the engine says so in `fold_train_seasons`, but
    that is not a per-artifact date this check can assert against a tipoff."""
    with_dates: dict = {}
    without_dates: list[str] = []
    block = _dig(run_meta, TRAIN_DATE_BLOCK)
    if isinstance(block, dict) and block:
        for fam, node in block.items():
            if not isinstance(node, dict):
                continue
            keys = node.get("keys")
            if not keys:
                without_dates.append(f"{fam} ({node.get('scheme', '?')}, no per-artifact dates)")
                continue
            for key, kn in keys.items():
                arts = [a for a in (kn.get("artifacts") or []) if a.get("max_train_date")]
                name = fam if str(key) in ("-", "", "None") else f"{fam}.{key}"
                if not arts:
                    without_dates.append(name)
                    continue
                with_dates[name] = {
                    "max_train_date": [a["max_train_date"] for a in arts],
                    "refit_dates": [a["refit_date"] for a in arts],
                }
        return with_dates, without_dates

    for name, path in CANDIDATE_ARTIFACT_PATHS:
        node = _dig(run_meta, path)
        if not isinstance(node, dict) or not node.get("max_train_date"):
            without_dates.append(name)
            continue
        with_dates[name] = {"max_train_date": node["max_train_date"], "refit_dates": node.get("refit_dates")}
    return with_dates, without_dates


def assign_artifact_max_train_date(game_dates: pd.Series, max_train_date, refit_dates) -> pd.Series:
    """Per-row `max_train_date` for a monthly S1 walk-forward family: the artifact deployed
    for a game is the one whose `refit_dates` entry is the latest <= the game's own date, and
    its `max_train_date` is what that artifact was actually trained through. A single static
    `max_train_date` (no `refit_dates`) applies to every row unchanged."""
    gd = pd.to_datetime(game_dates).to_numpy()
    if not refit_dates:
        return pd.Series(pd.Timestamp(max_train_date if isinstance(max_train_date, str) else max_train_date[0]),
                          index=game_dates.index)
    refit = pd.to_datetime(pd.Series(refit_dates)).to_numpy()
    mtd = pd.to_datetime(pd.Series(max_train_date)).to_numpy()
    order = np.argsort(refit)
    refit_sorted, mtd_sorted = refit[order], mtd[order]
    idx = np.searchsorted(refit_sorted, gd, side="right") - 1
    idx = np.clip(idx, 0, len(mtd_sorted) - 1)
    return pd.Series(mtd_sorted[idx], index=game_dates.index)


def build_honesty_note(run_meta: dict, graded: pd.DataFrame) -> str:
    """Rule 4, rewritten per the PM's 2026-09-10 correction: a walk-forward BACKTEST is
    honest by a per-row `max_train_date < tipoff` check on every sub-model artifact that
    carries a recorded training-window end, not by `created_at < tipoff` (which is
    structurally satisfied by construction for a completed-season backtest and is the LIVE
    check instead). Never fabricates a date for a family `run_meta` does not record."""
    backtest = bool(run_meta.get("backtest", False))
    if not backtest:
        games = graded  # live run: caller already has created_at/tipoff on the raw frame
        has_row_ts = "created_at" in games.columns and "tipoff" in games.columns
        if not has_row_ts:
            return ("**created_at not recorded** -- live run (`backtest=False`) but no per-row "
                    "`created_at`/`tipoff` columns exist to verify `created_at < tipoff`.")
        bad = int((pd.to_datetime(games["created_at"]) >= pd.to_datetime(games["tipoff"])).sum())
        return f"`created_at < tipoff` verified per row: {len(games) - bad}/{len(games)} rows pass ({bad} violation(s))."

    with_dates, without_dates = collect_family_max_train_dates(run_meta)
    if not with_dates:
        return ("**artifact dates not recorded** -- `run_meta.json` carries no per-model `max_train_date` for "
                f"any of the {len(without_dates)} known sub-model artifact paths checked ({', '.join(without_dates)}). "
                "backtest: run created post hoc by construction (`created_at` postdates every game in the fold), "
                "which is the wrong check for a walk-forward backtest; the right one (per-row artifact "
                "`max_train_date < tipoff`) cannot be computed without those dates. Not fabricated.")

    lines = []
    all_pass = True
    total_checked = 0
    for name, info in with_dates.items():
        mtd_per_row = assign_artifact_max_train_date(graded["game_date"], info["max_train_date"], info["refit_dates"])
        tipoff = pd.to_datetime(graded["tipoff_utc"])
        if getattr(tipoff.dt, "tz", None) is not None:
            tipoff = tipoff.dt.tz_convert(None)
        ok = (mtd_per_row.to_numpy() < tipoff.to_numpy()).astype(bool)
        n_pass, n_total = int(ok.sum()), len(ok)
        total_checked += 1
        all_pass = all_pass and (n_pass == n_total)
        lines.append(f"{name}: {n_pass}/{n_total} rows pass")
    verdict = ("backtest: run created post hoc by construction; artifact `max_train_date < tipoff` asserted on "
               + "; ".join(lines) + ".")
    if without_dates:
        verdict += (f" **artifact dates not recorded** for {len(without_dates)} other sub-model artifact path(s) "
                    f"({', '.join(without_dates)}) -- not checked, not fabricated, not assumed to pass.")
    if not all_pass:
        verdict = "**VIOLATION** -- " + verdict
    return verdict


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return center - half, center + half


# ---------------------------------------------------------------------------
# truth: schedule metadata from games_universe (via reference.py), SCORES
# from game_finals_v2 ONLY (rule 3 above)
# ---------------------------------------------------------------------------
def load_truth(season: int) -> tuple[pd.DataFrame, dict]:
    meta = ref_mod.load_actual_games(season).drop(
        columns=["home_score", "away_score", "margin", "total"]
    )
    finals = ref_mod.load_game_finals_truth(season, version="v2")
    if finals is None:
        raise RuntimeError("data/processed/truth/game_finals_v2.parquet not found or empty for this season")
    finals = finals.copy()
    finals["unresolved"] = (
        (finals["diff_home_score"].fillna(0) != 0) | (finals["diff_away_score"].fillna(0) != 0)
    ) & (~finals["finals_third_source_checked"].fillna(False))

    merged = meta.merge(
        finals[["game_id", "home_score", "away_score", "unresolved", "finals_source"]],
        on="game_id", how="left",
    )
    n_no_finals_row = int(merged["home_score"].isna().sum())
    n_unresolved = int(merged["unresolved"].fillna(False).sum())
    keep = merged[merged["home_score"].notna() & ~merged["unresolved"].fillna(False)].copy()
    keep["margin"] = keep["home_score"] - keep["away_score"]
    keep["total"] = keep["home_score"] + keep["away_score"]
    keep["is_home_win"] = (keep["margin"] > 0).astype(float)

    counts = {
        "n_schedule_games": int(len(meta)),
        "n_no_finals_row": n_no_finals_row,
        "n_unresolved_excluded": n_unresolved,
        "n_truth_games": int(len(keep)),
        "finals_source_counts": keep["finals_source"].value_counts(dropna=False).to_dict(),
    }
    return keep.reset_index(drop=True), counts


def attach_own_rating(truth: pd.DataFrame, season: int) -> pd.DataFrame:
    """Net own-rating (off_c - def_c, both already league-mean-centred per
    CLAUDE.md's modeling rule) as of each game's own date, home and away, and
    the game-level average quintiled into 5 team-quality buckets. This is a
    GRADING-ONLY quantity (same status as `reference.team_quality_terciles`),
    never a model feature."""
    ratings = orat.load_ratings([season])
    t = orat.join_as_of(truth, ratings, team_col="home_team_id", suffix="_home")
    t = orat.join_as_of(t, ratings, team_col="away_team_id", suffix="_away")
    t["own_rating_home"] = t["off_c_home"] - t["def_c_home"]
    t["own_rating_away"] = t["off_c_away"] - t["def_c_away"]
    t["own_rating_game"] = (t["own_rating_home"] + t["own_rating_away"]) / 2.0
    try:
        t["own_rating_quintile"] = pd.qcut(
            t["own_rating_game"], 5,
            labels=["Q1_bottom", "Q2", "Q3", "Q4", "Q5_top"], duplicates="drop",
        ).astype(str)
    except ValueError:
        t["own_rating_quintile"] = "UNAVAILABLE"
    return t


def attach_conference_flag(truth: pd.DataFrame, season: int,
                            cbbd_games_dir: Path = DEFAULT_CBBD_GAMES_DIR) -> pd.DataFrame:
    path = Path(cbbd_games_dir) / f"games_{int(season)}.parquet"
    cg = pd.read_parquet(path, columns=["id", "conferenceGame"])
    conf_map = cg.set_index("id")["conferenceGame"]
    t = truth.copy()
    t["is_conference_game"] = t["cbbd_game_id"].map(conf_map)
    return t


def load_close_lines(season: int, path: Path, provider: str = PRIMARY_PROVIDER) -> tuple[pd.DataFrame, int]:
    df = pd.read_parquet(path)
    df = df[(df["season"] == int(season)) & (df["provider"] == provider)].copy()
    dup = int(df.duplicated("game_id").sum())
    df = df.sort_values("game_id", kind="mergesort").drop_duplicates("game_id", keep="first")
    # `season` and `cbbd_game_id` are redundant with the truth frame's own columns of the
    # same name (both tables key on the same games_universe crosswalk) -- drop here so the
    # merge in main() does not silently suffix them to season_x/season_y.
    df = df.drop(columns=["season", "cbbd_game_id"])
    return df.reset_index(drop=True), dup


# ---------------------------------------------------------------------------
# edge-bucket tables (range buckets, not v1's cumulative >= thresholds)
# ---------------------------------------------------------------------------
SPREAD_TOTAL_BUCKETS: tuple[tuple[float, float], ...] = ((0, 1), (1, 2), (2, 3), (3, 5), (5, np.inf))
ML_BUCKETS_PP: tuple[tuple[float, float], ...] = ((0, 2), (2, 4), (4, 6), (6, np.inf))
MIN_CELL_N = 100


def bucket_table_range(disagree: pd.Series, cover: pd.Series,
                        buckets: tuple[tuple[float, float], ...], seed_se_value: float) -> pd.DataFrame:
    """Settled at flat -110 (spread/total; CBBD carries no per-side spread/
    total price). `disagree` = model - market (the model's side and margin of
    disagreement); `cover` = actual - market (which side actually covered)."""
    side = np.sign(disagree)
    result = np.sign(cover) * side
    rows = []
    for lo, hi in buckets:
        m = (disagree.abs() >= lo) & (disagree.abs() < hi)
        w = int(((result == 1) & m).sum())
        loss = int(((result == -1) & m).sum())
        push = int(((result == 0) & m).sum())
        n_dec = w + loss
        lo_ci, hi_ci = wilson_ci(w, n_dec)
        rows.append({
            "edge_bucket": f"[{lo:g}, {hi if np.isfinite(hi) else 'inf'})",
            "n": int(m.sum()), "wins": w, "losses": loss, "pushes": push,
            "hit_rate": (w / n_dec) if n_dec else float("nan"),
            "wilson_lo95": lo_ci, "wilson_hi95": hi_ci,
            "roi_at_-110": ((w - 1.1 * loss) / n_dec) if n_dec else float("nan"),
            "seed_noise_se": seed_se_value,
            "status": "UNDERPOWERED" if m.sum() < MIN_CELL_N else "OK",
        })
    return pd.DataFrame(rows)


def ml_bucket_table_pp(edge_pp: pd.Series, is_home_win: pd.Series, home_ml: pd.Series, away_ml: pd.Series,
                        buckets: tuple[tuple[float, float], ...], seed_se_value: float) -> pd.DataFrame:
    bet_home = edge_pp > 0
    won = np.where(bet_home, is_home_win == 1, is_home_win == 0)
    profit = M.ml_profit_if_win(np.where(bet_home, home_ml, away_ml))
    pnl = np.where(won, profit, -1.0)
    rows = []
    for lo, hi in buckets:
        m = (edge_pp.abs() >= lo) & (edge_pp.abs() < hi)
        n = int(m.sum())
        w = int(won[m].sum()) if n else 0
        lo_ci, hi_ci = wilson_ci(w, n)
        rows.append({
            "edge_bucket_pp": f"[{lo:g}, {hi if np.isfinite(hi) else 'inf'})",
            "n": n, "wins": w,
            "hit_rate": (w / n) if n else float("nan"),
            "wilson_lo95": lo_ci, "wilson_hi95": hi_ci,
            "roi_real_odds": float(pnl[m].mean()) if n else float("nan"),
            "seed_noise_se": seed_se_value,
            "status": "UNDERPOWERED" if n < MIN_CELL_N else "OK",
        })
    return pd.DataFrame(rows), pnl


def breakdown_frames(df: pd.DataFrame) -> dict[str, pd.Series]:
    conf = df["is_conference_game"]
    conf_label = pd.Series(
        np.where(conf.fillna("NA") == True, "conference",  # noqa: E712
                 np.where(conf.fillna("NA") == False, "non-conference", "unknown")),  # noqa: E712
        index=df.index,
    )
    return {
        "season": df["season"].astype(str),
        "month": df["month"].astype("Int64").astype(str),
        "conference": conf_label,
        "own_rating_quintile": df["own_rating_quintile"].astype(str),
    }


def build_edge_report(df: pd.DataFrame, disagree: pd.Series, cover: pd.Series,
                       buckets, seed_se_value: float, table_fn) -> pd.DataFrame:
    frames = []
    overall = table_fn(disagree, cover, buckets, seed_se_value)
    overall.insert(0, "group", "ALL")
    overall.insert(0, "breakdown", "overall")
    frames.append(overall)
    for dim_name, series in breakdown_frames(df).items():
        for val in sorted(pd.unique(series.dropna())):
            m = (series == val).to_numpy()
            if m.sum() == 0:
                continue
            t = table_fn(disagree[m], cover[m], buckets, seed_se_value)
            t.insert(0, "group", str(val))
            t.insert(0, "breakdown", dim_name)
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def build_ml_edge_report(df: pd.DataFrame, edge_pp: pd.Series, is_home_win: pd.Series,
                          home_ml: pd.Series, away_ml: pd.Series, seed_se_value: float) -> pd.DataFrame:
    frames = []
    overall, _ = ml_bucket_table_pp(edge_pp, is_home_win, home_ml, away_ml, ML_BUCKETS_PP, seed_se_value)
    overall.insert(0, "group", "ALL")
    overall.insert(0, "breakdown", "overall")
    frames.append(overall)
    for dim_name, series in breakdown_frames(df).items():
        for val in sorted(pd.unique(series.dropna())):
            m = (series == val).to_numpy()
            if m.sum() == 0:
                continue
            t, _ = ml_bucket_table_pp(edge_pp[m], is_home_win[m], home_ml[m], away_ml[m], ML_BUCKETS_PP, seed_se_value)
            t.insert(0, "group", str(val))
            t.insert(0, "breakdown", dim_name)
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# calibration / reliability (item 2's probability half -- Brier-adjacent,
# gated the same as Brier itself)
# ---------------------------------------------------------------------------
def calibration_20buckets(p: pd.Series, actual: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"p": p, "actual": actual}).dropna()
    d["bucket"] = pd.qcut(d["p"], 20, labels=False, duplicates="drop") + 1
    out = d.groupby("bucket").agg(n=("p", "size"), mean_p=("p", "mean"), actual_rate=("actual", "mean")).reset_index()
    out["delta"] = out["actual_rate"] - out["mean_p"]
    return out


def reliability_by_disagreement_decile(model_p: pd.Series, market_p: pd.Series, actual: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"model_p": model_p, "market_p": market_p, "actual": actual}).dropna()
    d["disagree"] = d["model_p"] - d["market_p"]
    d["actual_minus_market"] = d["actual"] - d["market_p"]
    d["decile"] = pd.qcut(d["disagree"], 10, labels=False, duplicates="drop") + 1
    out = d.groupby("decile").agg(
        n=("disagree", "size"),
        mean_sim_minus_market=("disagree", "mean"),
        mean_actual_minus_market=("actual_minus_market", "mean"),
    ).reset_index()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--gates-config", default="docs/gates.yaml")
    ap.add_argument("--lines-path", default=str(DEFAULT_LINES_PATH))
    ap.add_argument("--provider", default=PRIMARY_PROVIDER)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-sealed", action="store_true")
    ap.add_argument("--date", default="2026-09-10",
                     help="report date stamp for the output filename/header; defaults to today's project "
                          "date rather than the wall clock, matching every other doc written this session")
    args = ap.parse_args()

    tol = load_tolerances(Path(args.gates_config))
    engine = C.load_engine_results(args.results, allow_sealed=args.allow_sealed)
    tag = tag_from_results_dir(Path(args.results))
    today = args.date
    out_path = Path(args.out) if args.out else OUT_DIR / f"market_games_v2_{tag}_{today}.md"
    out_json = out_path.with_suffix(".json")

    run_meta = engine.run_meta
    n_seeds = int(run_meta.get("n_seeds") or len(run_meta.get("seeds") or []) or 0)
    gate_provisional = n_seeds < GATE_MIN_SEEDS
    calibration_ok = n_seeds >= GATE_MIN_SEEDS
    roi_ok = n_seeds >= ROI_BRIER_MIN_SEEDS

    # ---- truth: schedule metadata (games_universe) + scores (game_finals_v2 ONLY) ----
    games = engine.games
    truth, truth_counts = load_truth(args.season)
    truth = attach_own_rating(truth, args.season)
    truth = attach_conference_flag(truth, args.season)

    lines, n_dup_lines = load_close_lines(args.season, Path(args.lines_path), args.provider)

    sim = G.summarise_games(games)

    # ---- honesty guard (rule 4, per the PM's 2026-09-10 correction) --------
    graded = truth.merge(sim[["game_id"]], on="game_id", how="inner")
    tipoff_note = build_honesty_note(run_meta, graded)

    full = truth.merge(sim, on="game_id", how="inner").merge(lines, on="game_id", how="inner")

    full["market_margin"] = -full["close_spread_home"]
    full["model_margin"] = full["sim_margin_mean"]
    full["market_total"] = full["close_over_under"]
    full["model_total"] = full["sim_total_mean"]

    n_boot = args.n_boot

    L: list[str] = []
    L += [f"# Market scorecard v2 (accepted lines source) -- {engine.engine_tag} (season {args.season})", "",
          f"Generated {today} by `scripts/grade_market_games_v2.py`. Results: `{args.results}`. "
          f"Lines: `{args.lines_path}` (provider `{args.provider}`, {n_dup_lines} duplicate (game_id) rows dropped). "
          "See the module docstring for the four differences from "
          "`scripts/grade_market_games.py` (v1, unmodified).", ""]

    # ---- 1. seed-count / honesty guard --------------------------------
    L += ["## 1. Seed-count and backtest-honesty guard", ""]
    if gate_provisional:
        L += [f"**EVERY TABLE IN THIS REPORT IS PROVISIONAL.** This run has **{n_seeds} seeds**, below the "
              f"`docs/tests/engine_seed_count_2026-09-10.md` floor of {GATE_MIN_SEEDS} seeds for even the "
              "slate-level gate report.", ""]
    else:
        L += [f"This run has **{n_seeds} seeds**, at or above the {GATE_MIN_SEEDS}-seed gate-report floor.", ""]
    if not calibration_ok:
        L += [f"**Section 3 (probability calibration) is REFUSED below {GATE_MIN_SEEDS} seeds** -- this run has "
              f"**{n_seeds}**, below that floor. No calibration/reliability numbers printed.", ""]
    else:
        L += [f"This run has **{n_seeds}** seeds, at or above the {GATE_MIN_SEEDS}-seed floor: section 3's "
              f"calibration/reliability tables print below, labelled PROVISIONAL unless {n_seeds} also clears "
              f"the {ROI_BRIER_MIN_SEEDS}-seed floor.", ""]
    if not roi_ok:
        L += [f"**ROI, Brier and edge-bucket hit-rate numbers are REFUSED below {ROI_BRIER_MIN_SEEDS} seeds.** "
              f"`docs/tests/engine_seed_count_2026-09-10.md` measured that floor for a per-game ROI/Brier read "
              f"(game-level margin-mean SE is still {seed_se(n_seeds, 'margin')[0]:.3f} pts and win-prob SE "
              f"{seed_se(n_seeds, 'winprob')[0]:.4f} at {n_seeds} seeds; at 5 seeds a sim win probability can "
              "only take the values {0, 0.2, 0.4, 0.6, 0.8, 1.0}). This run has "
              f"**{n_seeds}** seeds, below that threshold, so section 3's Brier line and all of section 4 (edge "
              "buckets/ROI) print the seed count and threshold ONLY -- no numbers.", ""]
    else:
        L += [f"This run has **{n_seeds}** seeds, at or above the {ROI_BRIER_MIN_SEEDS}-seed floor for ROI/Brier "
              "reads; those numbers are computed below without a PROVISIONAL caveat.", ""]
    L += [tipoff_note, ""]

    L += ["### Second-source truth accounting (game_finals_v2 only)", ""]
    L += R.md_table(pd.DataFrame([truth_counts_row(truth_counts)])) + [""]
    L += [f"n games with a usable `{args.provider}` close line, after joining truth x sim x lines: {len(full)} "
          f"(of {truth_counts['n_truth_games']} truth games).", ""]

    # ---- 2. sim vs market: margin / total point estimates --------------
    L += ["## 2. Sim vs market: margin and total point estimates", "",
          f"{'PROVISIONAL. ' if gate_provisional else ''}Model-vs-actual and market-vs-actual MAE/bias (context), "
          "then a DIRECT model-vs-market comparison (MAE/bias/corr between the two point estimates themselves).", ""]
    core = pd.DataFrame([{
        "n": len(full),
        "model_margin_MAE_vs_actual": (full["margin"] - full["model_margin"]).abs().mean(),
        "market_margin_MAE_vs_actual": (full["margin"] - full["market_margin"]).abs().mean(),
        "model_margin_bias": (full["model_margin"] - full["margin"]).mean(),
        "market_margin_bias": (full["market_margin"] - full["margin"]).mean(),
        "model_total_MAE_vs_actual": (full["total"] - full["model_total"]).abs().mean(),
        "market_total_MAE_vs_actual": (full["total"] - full["market_total"]).abs().mean(),
        "model_total_bias": (full["model_total"] - full["total"]).mean(),
        "market_total_bias": (full["market_total"] - full["total"]).mean(),
    }])
    L += R.md_table(core) + [""]
    direct = pd.DataFrame([{
        "n": len(full),
        "margin_MAE(model,market)": (full["model_margin"] - full["market_margin"]).abs().mean(),
        "margin_bias(model-market)": (full["model_margin"] - full["market_margin"]).mean(),
        "margin_corr(model,market)": float(full["model_margin"].corr(full["market_margin"])),
        "total_MAE(model,market)": (full["model_total"] - full["market_total"]).abs().mean(),
        "total_bias(model-market)": (full["model_total"] - full["market_total"]).mean(),
        "total_corr(model,market)": float(full["model_total"].corr(full["market_total"])),
    }])
    L += R.md_table(direct) + [""]
    L += [f"seed-noise SE of the sim mean at {n_seeds} seeds: margin "
          f"{seed_se(n_seeds, 'margin')[0]:.3f} pts, total {seed_se(n_seeds, 'total')[0]:.3f} pts "
          f"({'exact study rung' if seed_se(n_seeds, 'margin')[1] else 'extrapolated via the study''s c/sqrt(k) law'}).", ""]

    # ---- 3. probability calibration (allowed >= 200 seeds, PROVISIONAL; Brier stays gated at 2,000) ---
    L += ["## 3. Sim win probability vs de-vigged close probability", ""]
    if not calibration_ok:
        L += [f"**REFUSED**: {n_seeds} seeds < {GATE_MIN_SEEDS}-seed floor for even a PROVISIONAL per-game "
              "probability read. No calibration/reliability/Brier numbers printed.", ""]
    else:
        if not roi_ok:
            L += [f"**PROVISIONAL** ({n_seeds} seeds, below the {ROI_BRIER_MIN_SEEDS}-seed floor for a fully "
                  "trusted read) -- calibration/reliability tables below are directional only.", ""]
        pm = full.dropna(subset=["p_home_close_prop", "close_ml_home", "close_ml_away"]).copy()
        mean_abs_prop_power = float((pm["p_home_close_prop"] - pm["p_home_close_power"]).abs().mean())
        L += [f"n = {len(pm)}. mean |prop - power| de-vig difference: {mean_abs_prop_power:.5f} "
              "(proportional is primary throughout; power is reported for comparison only).", "",
              "### Calibration vs realised outcome (20 buckets, sim probability)", ""]
        L += R.md_table(calibration_20buckets(pm["p_home"], pm["is_home_win"])) + [""]
        L += ["### Calibration vs realised outcome (20 buckets, market probability)", ""]
        L += R.md_table(calibration_20buckets(pm["p_home_close_prop"], pm["is_home_win"])) + [""]
        L += ["### Reliability: (sim prob - market prob) deciles vs (realised - market prob)", ""]
        L += R.md_table(reliability_by_disagreement_decile(pm["p_home"], pm["p_home_close_prop"], pm["is_home_win"])) + [""]
        if not roi_ok:
            L += [f"**Brier REFUSED**: {n_seeds} seeds < {ROI_BRIER_MIN_SEEDS}-seed floor. No Brier number printed.", ""]
        else:
            model_brier = float(((pm["p_home"] - pm["is_home_win"]) ** 2).mean())
            market_brier = float(((pm["p_home_close_prop"] - pm["is_home_win"]) ** 2).mean())
            L += [f"Brier: model {model_brier:.5f} vs market (proportional de-vig) {market_brier:.5f}.", ""]

    # ---- 4. edge buckets (ROI -- gated) --------------------------------
    L += ["## 4. Edge buckets: spread, total, moneyline (settled at real -110 / real posted odds)", ""]
    if not roi_ok:
        L += [f"**REFUSED**: {n_seeds} seeds < {ROI_BRIER_MIN_SEEDS}-seed floor. No ROI or hit-rate numbers "
              "printed. Breakdowns (season / month / conference / own-rating quintile) are likewise withheld.", ""]
    else:
        se_margin, _ = seed_se(n_seeds, "margin")
        se_total, _ = seed_se(n_seeds, "total")
        se_wp, _ = seed_se(n_seeds, "winprob")

        disagree_s = full["model_margin"] - full["market_margin"]
        cover_s = full["margin"] - full["market_margin"]
        L += ["### Spread edge buckets", ""]
        L += R.md_table(build_edge_report(full, disagree_s, cover_s, SPREAD_TOTAL_BUCKETS, se_margin, bucket_table_range)) + [""]

        dt = full.dropna(subset=["market_total"])
        disagree_t = dt["model_total"] - dt["market_total"]
        cover_t = dt["total"] - dt["market_total"]
        L += ["### Total edge buckets", ""]
        L += R.md_table(build_edge_report(dt, disagree_t, cover_t, SPREAD_TOTAL_BUCKETS, se_total, bucket_table_range)) + [""]

        dm = full.dropna(subset=["close_ml_home", "close_ml_away", "p_home_close_prop"])
        edge_pp = (dm["p_home"] - dm["p_home_close_prop"]) * 100.0
        L += ["### Moneyline edge buckets (pp)", ""]
        L += R.md_table(build_ml_edge_report(dm, edge_pp, dm["is_home_win"], dm["close_ml_home"], dm["close_ml_away"], se_wp)) + [""]

    # ---- 5. leak cross-check (2025 only, opens) ------------------------
    L += ["## 5. Leak cross-check: does edge-at-open predict open-to-close movement?", "",
          "CLAUDE.md: \"an edge that beats the close but cannot predict line movement is presumed leaked.\" "
          "Only 2025 has usable `open_spread_home` coverage (`docs/tests/lines_cbbd_validation_2026-09-10.md` "
          "addendum A: 0% in 2023-2024, 33.5% in 2025).", ""]
    if int(args.season) != 2025:
        L += [f"season {args.season} != 2025 -- skipped (no usable opens).", ""]
    else:
        do = full.dropna(subset=["open_spread_home"]).copy()
        do["open_margin"] = -do["open_spread_home"]
        do["movement"] = do["market_margin"] - do["open_margin"]
        do["edge_at_open"] = do["model_margin"] - do["open_margin"]
        moved = do[do["movement"] != 0]
        corr = float(do["edge_at_open"].corr(do["movement"])) if len(do) > 2 else float("nan")
        L += [f"n games with an open: {len(do)}; n with a nonzero close-vs-open move: {len(moved)}.",
              f"corr(edge at open, open-to-close movement) = {corr:.4f} "
              f"({'PROVISIONAL -- ' if gate_provisional else ''}the sim mean feeding `edge_at_open` carries "
              f"{seed_se(n_seeds, 'margin')[0]:.2f} pts of seed noise at {n_seeds} seeds, which attenuates any "
              "correlation read here toward zero).", ""]
        tab = pd.crosstab(np.sign(moved["edge_at_open"]).map({-1: "edge<0 (away)", 0: "edge=0", 1: "edge>0 (home)"}),
                           np.sign(moved["movement"]).map({-1: "moved away", 1: "moved home"}))
        L += ["2x2 (edge sign at open vs movement sign, nonzero moves only):", ""]
        L += R.md_table(tab.reset_index().rename(columns={"index": "edge_sign"})) + [""]
        agree = float((np.sign(moved["edge_at_open"]) == np.sign(moved["movement"])).mean()) if len(moved) else float("nan")
        L += [f"CLV sign agreement: {agree:.4f} on {len(moved)} moved lines "
              f"(gate {tol['g10_clv_agreement']}: near-coinflip = presumed-leaked signature if the surprise "
              "correlation above is also real).", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")

    with_dates, without_dates = collect_family_max_train_dates(run_meta)
    summary_json = {
        "engine_tag": engine.engine_tag, "results": str(args.results), "season": args.season,
        "n_seeds": n_seeds, "gate_provisional": gate_provisional, "calibration_ok": calibration_ok, "roi_ok": roi_ok,
        "gate_min_seeds": GATE_MIN_SEEDS, "roi_brier_min_seeds": ROI_BRIER_MIN_SEEDS,
        "honesty_note": tipoff_note,
        "run_meta_backtest": run_meta.get("backtest"), "run_meta_created_at": run_meta.get("created_at"),
        "artifact_families_with_max_train_date": list(with_dates.keys()),
        "artifact_families_without_max_train_date": without_dates,
        "lines_path": str(args.lines_path), "provider": args.provider, "n_dup_lines_dropped": n_dup_lines,
        "truth_counts": truth_counts,
        "n_games_graded": int(len(full)),
    }
    out_json.write_text(json.dumps(summary_json, indent=2, default=str), encoding="utf-8")

    print(f"wrote {out_path} ({len(L)} lines)")
    print(f"wrote {out_json}")
    print(f"n_seeds={n_seeds} gate_provisional={gate_provisional} roi_ok={roi_ok}")
    return 0


def truth_counts_row(counts: dict) -> dict:
    return {k: v for k, v in counts.items() if k != "finals_source_counts"} | {
        "finals_source_counts": json.dumps(counts["finals_source_counts"])
    }


if __name__ == "__main__":
    raise SystemExit(main())
