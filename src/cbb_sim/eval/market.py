"""
market.py -- G10, the market scorecard (SIM_GUARDRAILS.md section 3, row G10;
`docs/postmortem/05_cfb_methodology_extract.md` sections 4 and 6).

Generalises `scripts/grade_control.py`'s `gate_g10` (read, not imported: that
function is keyed to Control's `summary.parquet` and hardcodes the ESPN-BET-
only 2025 join). Shared by `scripts/grade_market_games.py` (games) and
`scripts/grade_market_props.py` (props import the settle/de-vig/bootstrap
primitives below rather than re-implementing them).

Two rules this module enforces, both non-negotiable per the methodology doc:

  1. SETTLEMENT VS DE-VIG. Bets settle at the REAL posted line/odds. De-vigging
     (removing the vig from a two-sided moneyline) is used ONLY to build a
     probability for calibration/Brier comparison -- never to move a
     settlement price. ("de-vig is for the probability comparison only, never
     for settlement.")

  2. LEAK-SUSPECT. surprise_corr = corr(model-vs-close disagreement,
     actual-vs-close surprise) is the "beats the close" headline number.
     clv_agreement = P(sign(model disagreement at the OPEN) ==
     sign(close - open)) on games where the line actually moved -- ~0.50 is
     the coin-flip/leak signature. If surprise_corr is real (> the gate) AND
     clv_agreement is near coin-flip (< the gate), the run is LEAK-SUSPECT:
     "information that beats the close but never moves the line is leaked
     information."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cbb_sim.eval import reference as ref_mod

EDGE_BUCKETS: tuple[tuple[float, float], ...] = ((0.0, 0.025), (0.025, 0.05), (0.05, 0.10), (0.10, 1.0))
ATS_THRESHOLDS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)


def ml_to_prob(ml: pd.Series) -> np.ndarray:
    ml = ml.astype("float64")
    return np.where(ml > 0, 100.0 / (ml + 100.0), -ml / (-ml + 100.0))


def ml_profit_if_win(ml: pd.Series | np.ndarray) -> np.ndarray:
    """Decimal profit per unit stake if the bet at these American odds wins."""
    ml = np.asarray(ml, dtype="float64")
    return np.where(ml > 0, ml / 100.0, 100.0 / (-ml))


# ---------------------------------------------------------------------------
# generic ATS/OU bucket table at flat -110
# ---------------------------------------------------------------------------
def bucket_table(disagree: pd.Series, cover: pd.Series, thresholds: tuple[float, ...] = ATS_THRESHOLDS) -> pd.DataFrame:
    side = np.sign(disagree)
    result = np.sign(cover) * side  # +1 win, -1 loss, 0 push
    rows = []
    for thr in thresholds:
        m = disagree.abs() >= thr
        w = int(((result == 1) & m).sum())
        loss = int(((result == -1) & m).sum())
        push = int(((result == 0) & m).sum())
        n_dec = w + loss
        rows.append({
            "bucket": f">= {thr:.0f}", "n": int(m.sum()), "wins": w, "losses": loss, "pushes": push,
            "win_pct": (w / n_dec) if n_dec else float("nan"),
            "roi_at_-110": ((w - 1.1 * loss) / n_dec) if n_dec else float("nan"),
        })
    return pd.DataFrame(rows)


def decided_pnl(disagree: pd.Series, cover: pd.Series, threshold: float) -> np.ndarray:
    """+1/-1.1/nan (push, dropped) per game at flat -110, for bets meeting
    `threshold` points of disagreement -- the per-row input to
    `bootstrap_roi`."""
    side = np.sign(disagree)
    result = np.sign(cover) * side
    m = disagree.abs() >= threshold
    pnl = np.where(result[m] == 1, 1.0, np.where(result[m] == -1, -1.1, np.nan))
    return pnl[~np.isnan(pnl)]


def bootstrap_roi(pnl: np.ndarray, n_boot: int = 1000, seed: int = 20260910) -> dict:
    """Game-clustered bootstrap CI on ROI. Each element of `pnl` is already
    one game's single settled bet, so resampling rows IS resampling games
    (whole-game clustering, never resampling individual sides of the same
    game independently)."""
    n = len(pnl)
    if n == 0:
        return {"mean": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
                "p_roi_le_0": float("nan"), "n_boot": n_boot, "n_games": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = pnl[idx].mean(axis=1)
    return {
        "mean": float(boots.mean()), "lo95": float(np.percentile(boots, 2.5)),
        "hi95": float(np.percentile(boots, 97.5)), "p_roi_le_0": float((boots <= 0).mean()),
        "n_boot": n_boot, "n_games": n,
    }


# ---------------------------------------------------------------------------
# moneyline edge buckets at REAL odds
# ---------------------------------------------------------------------------
def ml_edge_table(edge: pd.Series, is_home_win: pd.Series, home_ml: pd.Series, away_ml: pd.Series,
                   buckets: tuple[tuple[float, float], ...] = EDGE_BUCKETS) -> pd.DataFrame:
    bet_home = edge > 0
    chosen_odds = np.where(bet_home, home_ml, away_ml)
    won = np.where(bet_home, is_home_win == 1, is_home_win == 0)
    profit = ml_profit_if_win(chosen_odds)
    pnl = np.where(won, profit, -1.0)
    rows = []
    for lo, hi in buckets:
        m = (edge.abs() >= lo) & (edge.abs() < hi)
        n = int(m.sum())
        rows.append({
            "edge_bucket": f"[{lo:.3f}, {hi:.3f})", "n": n,
            "win_pct": float(won[m].mean()) if n else float("nan"),
            "roi": float(pnl[m].mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows), pnl


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def calibration_deciles(model_p: pd.Series, market_p: pd.Series, actual: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"model_p": model_p, "market_p": market_p, "actual": actual}).dropna()
    d["decile"] = pd.qcut(d["model_p"], 10, labels=False, duplicates="drop") + 1
    out = d.groupby("decile").agg(
        n=("model_p", "size"), model_p=("model_p", "mean"),
        market_p=("market_p", "mean"), actual_rate=("actual", "mean"),
    ).reset_index()
    out["model_delta"] = out["actual_rate"] - out["model_p"]
    out["market_delta"] = out["actual_rate"] - out["market_p"]
    return out


# ---------------------------------------------------------------------------
# G10 result container
# ---------------------------------------------------------------------------
def leak_verdict(surprise_corr: float, clv_agreement: float, tol: dict) -> str:
    """The INV-45/47 leak screen, standalone and pure so it is unit-testable
    without touching disk (`tests/test_eval.py`): "information that beats the
    close but never moves the line is leaked information." `surprise_corr`
    real (> gate) AND `clv_agreement` near coin-flip (< gate) => LEAK-SUSPECT.
    Real `surprise_corr` with no opening lines to check (`clv_agreement`
    NaN) => NEEDS-INSTRUMENTATION, never a silent PASS."""
    leaky = surprise_corr > tol["g10_surprise_corr"]
    if not leaky:
        return "PASS"
    if not np.isfinite(clv_agreement):
        return "NEEDS-INSTRUMENTATION"
    return "LEAK-SUSPECT" if clv_agreement < tol["g10_clv_agreement"] else "PASS"


@dataclass
class MarketResult:
    season: int
    n_with_line: int
    provider_counts: dict
    model_margin_mae: float
    close_margin_mae: float
    model_margin_bias: float
    close_margin_bias: float
    model_total_mae: float
    close_total_mae: float
    model_total_bias: float
    close_total_bias: float
    ats_table: pd.DataFrame
    ou_table: pd.DataFrame
    ats_bootstrap: dict
    ou_bootstrap: dict
    ml_edge_table: pd.DataFrame
    ml_edge_bootstrap: dict
    n_ml: int
    model_brier_ml_subset: float
    market_brier: float
    mean_vig: float
    calibration: pd.DataFrame
    surprise_corr: float
    n_clv: int
    clv_agreement: float
    leak_status: str
    notes: list[str] = field(default_factory=list)


def gate_g10(summary: pd.DataFrame, season: int, tol: dict, lines: pd.DataFrame | None = None,
             n_boot: int = 1000) -> MarketResult:
    """`summary` must carry (at minimum): cbbd_game_id, margin, total,
    sim_margin_mean, sim_total_mean, p_home, is_home_win -- exactly the
    columns `gates.build_grading_frame` already produces."""
    if lines is None:
        lines = ref_mod.load_lines(season)
    d = summary.merge(lines, on="cbbd_game_id", how="inner").dropna(subset=["spread"]).copy()
    provider_counts = d["provider_used"].value_counts().to_dict()

    # CBBD `spread` is home-perspective (negative == home favoured); verified
    # in `docs/tests/control_engine_F2_2026-09-10.md` (corr(spread, actual
    # margin) ~ -0.65), so the market's expected home margin is -spread.
    d["market_margin"] = -d["spread"]
    d["model_margin"] = d["sim_margin_mean"]

    model_margin_mae = float((d["margin"] - d["model_margin"]).abs().mean())
    close_margin_mae = float((d["margin"] - d["market_margin"]).abs().mean())
    model_margin_bias = float((d["model_margin"] - d["margin"]).mean())
    close_margin_bias = float((d["market_margin"] - d["margin"]).mean())

    dt = d.dropna(subset=["overUnder"])
    model_total_mae = float((dt["total"] - dt["sim_total_mean"]).abs().mean())
    close_total_mae = float((dt["total"] - dt["overUnder"]).abs().mean())
    model_total_bias = float((dt["sim_total_mean"] - dt["total"]).mean())
    close_total_bias = float((dt["overUnder"] - dt["total"]).mean())

    # ---- ATS by disagreement bucket, settled at the REAL spread, -110 -----
    disagree = d["model_margin"] - d["market_margin"]
    cover = d["margin"] - d["market_margin"]
    ats_table = bucket_table(disagree, cover)
    ats_boot = bootstrap_roi(decided_pnl(disagree, cover, ATS_THRESHOLDS[0]), n_boot)

    # ---- OU by disagreement bucket, settled at the REAL total, -110 -------
    if len(dt):
        disagree_t = dt["sim_total_mean"] - dt["overUnder"]
        cover_t = dt["total"] - dt["overUnder"]
        ou_table = bucket_table(disagree_t, cover_t)
        ou_boot = bootstrap_roi(decided_pnl(disagree_t, cover_t, ATS_THRESHOLDS[0]), n_boot)
    else:
        ou_table = pd.DataFrame()
        ou_boot = bootstrap_roi(np.array([]), n_boot)

    # ---- Brier + calibration vs de-vigged moneyline, settle at REAL odds ---
    dm = d.dropna(subset=["homeMoneyline", "awayMoneyline"]).copy()
    ph = ml_to_prob(dm["homeMoneyline"])
    pa = ml_to_prob(dm["awayMoneyline"])
    dm["market_p_home"] = ph / (ph + pa)  # de-vig: probability comparison ONLY
    dm["vig"] = ph + pa - 1.0
    y = dm["is_home_win"]
    model_brier = float(((dm["p_home"] - y) ** 2).mean())
    market_brier = float(((dm["market_p_home"] - y) ** 2).mean())
    mean_vig = float(dm["vig"].mean())
    calib = calibration_deciles(dm["p_home"], dm["market_p_home"], y)

    edge = dm["p_home"] - dm["market_p_home"]
    ml_table, ml_pnl = ml_edge_table(edge, y, dm["homeMoneyline"], dm["awayMoneyline"])
    ml_boot = bootstrap_roi(ml_pnl, n_boot)

    # ---- leak screen --------------------------------------------------
    surprise_corr = float(np.corrcoef(disagree, cover)[0, 1])
    dc = d.dropna(subset=["spreadOpen"]).copy()
    if len(dc) >= 100:
        dc["open_margin"] = -dc["spreadOpen"]
        move = dc["market_margin"] - dc["open_margin"]
        dis_open = dc["model_margin"] - dc["open_margin"]
        moved = move != 0
        n_clv = int(moved.sum())
        clv_agreement = float((np.sign(dis_open[moved]) == np.sign(move[moved])).mean())
    else:
        n_clv = int(len(dc))
        clv_agreement = float("nan")

    leak_status = leak_verdict(surprise_corr, clv_agreement, tol)

    notes = []
    if leak_status == "LEAK-SUSPECT":
        notes.append("DO NOT TRUST THIS RUN'S EDGE NUMBERS -- surprise correlation is real "
                     "(beats the close) but CLV sign agreement is near coin-flip: the edge does "
                     "not predict which way the market itself later moves, the signature of a "
                     "leaked (post-hoc-available) feature, not real information.")

    return MarketResult(
        season=season, n_with_line=int(len(d)), provider_counts=provider_counts,
        model_margin_mae=model_margin_mae, close_margin_mae=close_margin_mae,
        model_margin_bias=model_margin_bias, close_margin_bias=close_margin_bias,
        model_total_mae=model_total_mae, close_total_mae=close_total_mae,
        model_total_bias=model_total_bias, close_total_bias=close_total_bias,
        ats_table=ats_table, ou_table=ou_table, ats_bootstrap=ats_boot, ou_bootstrap=ou_boot,
        ml_edge_table=ml_table, ml_edge_bootstrap=ml_boot, n_ml=int(len(dm)),
        model_brier_ml_subset=model_brier, market_brier=market_brier, mean_vig=mean_vig,
        calibration=calib, surprise_corr=surprise_corr, n_clv=n_clv, clv_agreement=clv_agreement,
        leak_status=leak_status, notes=notes,
    )
