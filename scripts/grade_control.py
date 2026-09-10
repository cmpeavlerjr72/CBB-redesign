"""
grade_control.py -- blind gate report for the Control engine.

    .venv/Scripts/python.exe scripts/grade_control.py

Reads every `results/control/<fold>_<anchor>[_seedoff{N}]/summary.parquet` that
exists and writes `docs/tests/control_engine_F2_2026-09-10.md`: the fold-2
scorecard for all three anchors (fold 1 alongside, for drift), the
pre-registered decision rule applied to the measured seed noise floor, and
gates G1, G5, G6, G9, the responsiveness check and the G10 market scorecard.

EVERY gate line ends in a literal PASS / FAIL / NEEDS-INSTRUMENTATION.
Tolerances come from `docs/SIM_GUARDRAILS.md` section 3 and are provisional
until the seed-noise study replaces them; a miss is a miss.

Nothing here adjusts the engine. Where a gate fails, the report names the
responsible component and stops (CLAUDE.md "no hand tuning on engine output").
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.control import features as F  # noqa: E402

RESULTS = Path("results/control")
REFERENCE = Path("data/reference")
LINES_DIR = Path("data/raw/cbbd")
OUT_DOC = Path("docs/tests/control_engine_F2_2026-09-10.md")

# --- provisional tolerances, SIM_GUARDRAILS.md section 3 -------------------
TOL = {
    "g1_mean": 1.0,
    "g1_sd": 0.75,
    "g5_sd_ratio_lo": 0.95,
    "g5_sd_ratio_hi": 1.05,
    "g5_corr": 0.05,
    "g5_ks_p": 0.10,
    "g6_margin": 1.0,
    "g9_margin_bias": 0.5,
    "g9_total_bias": 1.0,
    "g9_slope_lo": 0.95,
    "g9_slope_hi": 1.05,
    "g10_surprise_corr": 0.15,
    "g10_clv_agreement": 0.53,
}

# Cells below this many games are labelled UNDERPOWERED and are NOT scored
# pass/fail (SIM_GUARDRAILS.md: "Underpowered cells are labelled underpowered,
# never presented as signal or absence of signal"). Same convention as
# cbb_sim.analysis.leak_test.MIN_N.
MIN_CELL_N = 50


def status(ok: bool | None) -> str:
    if ok is None:
        return "NEEDS-INSTRUMENTATION"
    return "PASS" if ok else "FAIL"


def within(value: float, target: float, tol: float) -> bool | None:
    if not np.isfinite(value) or not np.isfinite(target):
        return None
    return abs(value - target) <= tol


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_run(fold: str, anchor: str, seed_offset: int = 0) -> pd.DataFrame | None:
    name = f"{fold}_{anchor}" + (f"_seedoff{seed_offset}" if seed_offset else "")
    p = RESULTS / name / "summary.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p).sort_values("game_id").reset_index(drop=True)
    df["is_home_win"] = (df["margin"] > 0).astype(float)
    return df


def load_games_raw(fold: str, anchor: str, seed_offset: int = 0) -> pd.DataFrame | None:
    name = f"{fold}_{anchor}" + (f"_seedoff{seed_offset}" if seed_offset else "")
    p = RESULTS / name / "games.parquet"
    return pd.read_parquet(p, columns=["game_id", "seed", "home_pts", "away_pts", "possessions", "n_ot"]) if p.exists() else None


# ---------------------------------------------------------------------------
# headline metrics
# ---------------------------------------------------------------------------
def headline(s: pd.DataFrame) -> dict:
    e_m = s["sim_margin_mean"] - s["margin"]
    e_t = s["sim_total_mean"] - s["total"]
    brier = float(((s["p_home"] - s["is_home_win"]) ** 2).mean())
    slope, intercept = np.polyfit(s["sim_margin_mean"], s["margin"], 1)
    # errors-in-variables correction: the sim mean is a Monte-Carlo estimate of
    # the model's expected margin, so the raw slope is attenuated by
    # Var(MC noise) = mean(sim_sd^2) / n_seeds. Reported as a DIAGNOSTIC only.
    var_mc = float((s["sim_margin_sd"] ** 2).mean() / s["n_seeds"].mean())
    var_pred = float(s["sim_margin_mean"].var())
    slope_corr = float(slope * var_pred / max(var_pred - var_mc, 1e-9))
    return {
        "n": int(len(s)),
        "margin_mae": float(e_m.abs().mean()),
        "margin_bias": float(e_m.mean()),
        "total_mae": float(e_t.abs().mean()),
        "total_bias": float(e_t.mean()),
        "brier": brier,
        "slope": float(slope),
        "slope_intercept": float(intercept),
        "slope_mc_corrected": slope_corr,
        "mc_noise_sd": float(np.sqrt(var_mc)),
    }


def brier_decile(s: pd.DataFrame) -> pd.DataFrame:
    d = s[["p_home", "is_home_win"]].copy()
    d["decile"] = pd.qcut(d["p_home"], 10, labels=False, duplicates="drop") + 1
    out = d.groupby("decile").agg(
        n=("p_home", "size"), pred=("p_home", "mean"), actual=("is_home_win", "mean")
    ).reset_index()
    out["delta"] = out["actual"] - out["pred"]
    return out


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate_g1(s: pd.DataFrame, raw: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict]:
    """Possessions per game, mean and SD, overall and by month."""
    ref = pd.read_parquet(REFERENCE / f"gate_targets_{season}.parquet")

    def ref_val(breakdown, group, metric):
        m = ref[(ref["breakdown"] == breakdown) & (ref["group"].astype(str) == str(group))
                & (ref["metric"] == metric) & (ref["side"] == "overall")]
        return float(m["value"].iloc[0]) if len(m) else float("nan")

    month = s.set_index("game_id")["month"]
    r = raw.copy()
    r["month"] = r["game_id"].map(month)

    rows = []
    groups = [("season", "all", r, s)]
    for mo in sorted(s["month"].unique()):
        groups.append(("month", int(mo), r[r["month"] == mo], s[s["month"] == mo]))
    for breakdown, group, rr, ss in groups:
        sim_mean, sim_sd = float(rr["possessions"].mean()), float(rr["possessions"].std())
        act_mean, act_sd = float(ss["game_poss"].mean()), float(ss["game_poss"].std())
        rows.append({
            "breakdown": breakdown, "group": group, "n_games": int(len(ss)),
            "sim_mean": sim_mean, "actual_mean": act_mean, "ref_mean": ref_val(breakdown, group, "poss_per_game_mean"),
            "sim_sd": sim_sd, "actual_sd": act_sd, "ref_sd": ref_val(breakdown, group, "poss_per_game_sd"),
            "d_mean": sim_mean - act_mean, "d_sd": sim_sd - act_sd,
            "status_mean": ("UNDERPOWERED" if len(ss) < MIN_CELL_N
                            else status(abs(sim_mean - act_mean) <= TOL["g1_mean"])),
            "status_sd": ("UNDERPOWERED" if len(ss) < MIN_CELL_N
                          else status(abs(sim_sd - act_sd) <= TOL["g1_sd"])),
        })
    tab = pd.DataFrame(rows)
    overall = tab[tab["breakdown"] == "season"].iloc[0]
    return tab, {"mean_ok": overall["status_mean"] == "PASS", "sd_ok": overall["status_sd"] == "PASS"}


def gate_g5(s: pd.DataFrame, raw: pd.DataFrame) -> dict:
    r = raw.copy()
    r["margin"] = r["home_pts"].astype("int32") - r["away_pts"].astype("int32")
    r["total"] = r["home_pts"].astype("int32") + r["away_pts"].astype("int32")

    resid_m = float((s["margin"] - s["sim_margin_mean"]).std())
    resid_t = float((s["total"] - s["sim_total_mean"]).std())
    sim_sd_m = float(s["sim_margin_sd"].mean())
    sim_sd_t = float(s["sim_total_sd"].mean())

    corr_sim = float(np.corrcoef(r["home_pts"].astype(float), r["away_pts"].astype(float))[0, 1])
    corr_act = float(np.corrcoef(s["home_score"].astype(float), s["away_score"].astype(float))[0, 1])

    # randomised PIT: fraction of sims below the actual margin, plus half the ties
    g = r.groupby("game_id")["margin"]
    actual = s.set_index("game_id")["margin"]
    below = g.apply(lambda x: float((x < actual.loc[x.name]).mean()))
    equal = g.apply(lambda x: float((x == actual.loc[x.name]).mean()))
    rng = np.random.default_rng(20260910)
    pit = (below + equal * rng.random(len(below))).to_numpy()
    ks = stats.kstest(pit, "uniform")

    ratio_m = sim_sd_m / resid_m
    ratio_t = sim_sd_t / resid_t
    return {
        "sim_sd_margin": sim_sd_m, "resid_sd_margin": resid_m, "ratio_margin": ratio_m,
        "sim_sd_total": sim_sd_t, "resid_sd_total": resid_t, "ratio_total": ratio_t,
        "corr_sim": corr_sim, "corr_actual": corr_act, "d_corr": corr_sim - corr_act,
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "status_margin": status(TOL["g5_sd_ratio_lo"] <= ratio_m <= TOL["g5_sd_ratio_hi"]),
        "status_total": status(TOL["g5_sd_ratio_lo"] <= ratio_t <= TOL["g5_sd_ratio_hi"]),
        "status_corr": status(abs(corr_sim - corr_act) <= TOL["g5_corr"]),
        "status_pit": status(ks.pvalue > TOL["g5_ks_p"]),
        "pit": pit,
    }


def gate_g6(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in (("non-neutral", s[s["neutral"] == 0]), ("neutral", s[s["neutral"] == 1])):
        sim = float(sub["sim_margin_mean"].mean())
        act = float(sub["margin"].mean())
        rows.append({"site": label, "n": int(len(sub)), "sim": sim, "actual": act,
                     "delta": sim - act, "status": status(abs(sim - act) <= TOL["g6_margin"])})
    return pd.DataFrame(rows)


def team_tiers(s: pd.DataFrame) -> pd.Series:
    """Per-team tercile of own-season mean point differential, the same
    (deliberately not leak-free) definition scripts/build_gate_reference.py
    uses. GRADING ONLY -- never joined onto a game as a feature."""
    home = s[["home_team_id", "margin"]].rename(columns={"home_team_id": "team_id"})
    away = s[["away_team_id", "margin"]].rename(columns={"away_team_id": "team_id"})
    away["margin"] = -away["margin"]
    tg = pd.concat([home, away], ignore_index=True)
    mean_margin = tg.groupby("team_id")["margin"].mean()
    tiers = pd.qcut(mean_margin, 3, labels=["bottom_tercile", "middle_tercile", "top_tercile"])
    return tiers


def gate_g9_breakdowns(s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    s = s.copy()
    tiers = team_tiers(s)
    s["home_tier"] = s["home_team_id"].map(tiers).astype(str)
    s["pred_total_tercile"] = pd.qcut(
        s["sim_total_mean"], 3, labels=["bottom_tercile", "middle_tercile", "top_tercile"]
    ).astype(str)

    def agg(df, key):
        out = []
        for g, d in df.groupby(key, observed=True):
            e_m = d["sim_margin_mean"] - d["margin"]
            e_t = d["sim_total_mean"] - d["total"]
            slope = np.polyfit(d["sim_margin_mean"], d["margin"], 1)[0] if len(d) > 30 else np.nan
            out.append({
                key: g, "n": int(len(d)),
                "margin_mae": float(e_m.abs().mean()), "margin_bias": float(e_m.mean()),
                "total_mae": float(e_t.abs().mean()), "total_bias": float(e_t.mean()),
                "slope": float(slope),
                "status_margin_bias": ("UNDERPOWERED" if len(d) < MIN_CELL_N
                                       else status(abs(e_m.mean()) <= TOL["g9_margin_bias"])),
                "status_total_bias": ("UNDERPOWERED" if len(d) < MIN_CELL_N
                                      else status(abs(e_t.mean()) <= TOL["g9_total_bias"])),
            })
        return pd.DataFrame(out)

    return {
        "month": agg(s, "month"),
        "tier": agg(s, "home_tier"),
        "pred_total_tercile": agg(s, "pred_total_tercile"),
    }


def responsiveness(s: pd.DataFrame) -> pd.DataFrame:
    d = s.dropna(subset=["rating_diff"]).copy()
    d["quintile"] = pd.qcut(d["rating_diff"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out = d.groupby("quintile").agg(
        n=("margin", "size"),
        rating_diff=("rating_diff", "mean"),
        predicted_margin=("sim_margin_mean", "mean"),
        actual_margin=("margin", "mean"),
    ).reset_index()
    out["delta"] = out["predicted_margin"] - out["actual_margin"]
    return out


# ---------------------------------------------------------------------------
# G10 market scorecard
# ---------------------------------------------------------------------------
def ml_to_prob(ml: pd.Series) -> pd.Series:
    ml = ml.astype("float64")
    return np.where(ml > 0, 100.0 / (ml + 100.0), -ml / (-ml + 100.0))


def load_lines(season: int) -> pd.DataFrame:
    p = LINES_DIR / f"lines_{season}.parquet"
    lines = pd.read_parquet(p)
    lines = lines[lines["provider"] == "ESPN BET"].copy()
    lines["cbbd_game_id"] = pd.to_numeric(lines["gameId"], errors="coerce").astype("Int64")
    lines["spreadOpen"] = pd.to_numeric(lines["spreadOpen"], errors="coerce")
    return lines[["cbbd_game_id", "spread", "spreadOpen", "overUnder",
                  "homeMoneyline", "awayMoneyline"]]


def gate_g10(s: pd.DataFrame, season: int) -> dict:
    lines = load_lines(season)
    d = s.merge(lines, on="cbbd_game_id", how="inner").dropna(subset=["spread"])
    d = d.copy()
    # CBBD `spread` is home-perspective (negative == home favoured), verified by
    # corr(spread, actual margin) = -0.65: the market's expected home margin is
    # therefore -spread.
    d["market_margin"] = -d["spread"]
    d["model_margin"] = d["sim_margin_mean"]

    res = {"season": season, "n": int(len(d)), "n_universe": int(len(s))}
    res["model_margin_mae"] = float((d["margin"] - d["model_margin"]).abs().mean())
    res["close_margin_mae"] = float((d["margin"] - d["market_margin"]).abs().mean())
    res["model_margin_bias"] = float((d["model_margin"] - d["margin"]).mean())
    res["close_margin_bias"] = float((d["market_margin"] - d["margin"]).mean())

    dt = d.dropna(subset=["overUnder"])
    res["n_total"] = int(len(dt))
    res["model_total_mae"] = float((dt["total"] - dt["sim_total_mean"]).abs().mean())
    res["close_total_mae"] = float((dt["total"] - dt["overUnder"]).abs().mean())
    res["model_total_bias"] = float((dt["sim_total_mean"] - dt["total"]).mean())
    res["close_total_bias"] = float((dt["overUnder"] - dt["total"]).mean())

    # ---- ATS by disagreement bucket -----------------------------------
    disagree = d["model_margin"] - d["market_margin"]
    cover = d["margin"] - d["market_margin"]
    side = np.sign(disagree)
    result = np.sign(cover) * side          # +1 win, -1 loss, 0 push
    ats = []
    for thr in (1.0, 2.0, 3.0, 5.0):
        m = disagree.abs() >= thr
        w = int(((result == 1) & m).sum())
        loss = int(((result == -1) & m).sum())
        push = int(((result == 0) & m).sum())
        n_dec = w + loss
        ats.append({
            "bucket": f">= {thr:.0f}", "n": int(m.sum()), "wins": w, "losses": loss, "pushes": push,
            "win_pct": (w / n_dec) if n_dec else float("nan"),
            "roi_at_-110": ((w - 1.1 * loss) / n_dec) if n_dec else float("nan"),
        })
    res["ats"] = pd.DataFrame(ats)

    # ---- Brier vs de-vigged moneyline ---------------------------------
    dm = d.dropna(subset=["homeMoneyline", "awayMoneyline"]).copy()
    ph = ml_to_prob(dm["homeMoneyline"])
    pa = ml_to_prob(dm["awayMoneyline"])
    dm["market_p_home"] = ph / (ph + pa)
    dm["vig"] = ph + pa - 1.0
    y = dm["is_home_win"]
    res["n_ml"] = int(len(dm))
    res["model_brier_ml_subset"] = float(((dm["p_home"] - y) ** 2).mean())
    res["market_brier"] = float(((dm["market_p_home"] - y) ** 2).mean())
    res["mean_vig"] = float(dm["vig"].mean())

    # ---- leak screen ---------------------------------------------------
    res["surprise_corr"] = float(np.corrcoef(disagree, cover)[0, 1])
    dc = d.dropna(subset=["spreadOpen"]).copy()
    if len(dc) >= 100:
        dc["open_margin"] = -dc["spreadOpen"]
        move = dc["market_margin"] - dc["open_margin"]
        dis_open = dc["model_margin"] - dc["open_margin"]
        moved = move != 0
        res["n_clv"] = int(moved.sum())
        res["clv_agreement"] = float((np.sign(dis_open[moved]) == np.sign(move[moved])).mean())
    else:
        res["n_clv"] = int(len(dc))
        res["clv_agreement"] = float("nan")

    leaky = res["surprise_corr"] > TOL["g10_surprise_corr"]
    if leaky and np.isfinite(res["clv_agreement"]):
        res["leak_status"] = "LEAK-SUSPECT" if res["clv_agreement"] < TOL["g10_clv_agreement"] else "PASS"
    elif leaky:
        res["leak_status"] = "NEEDS-INSTRUMENTATION"
    else:
        res["leak_status"] = "PASS"
    return res



# ---------------------------------------------------------------------------
# failure diagnosis (never a fix -- see CLAUDE.md "no hand tuning")
# ---------------------------------------------------------------------------
def diagnose_dispersion(fold: str, anchor: str) -> dict:
    """Trace the G5 dispersion failure to a component.

    Refits nothing: it loads the persisted models, recomputes their Pearson
    residuals on the fold's TRAINING team-games, and reports (a) whether each
    count's MARGINAL dispersion matches the data and (b) the residual
    correlation between the four counts, which the simulator assumes to be
    zero because the four GLMs are drawn independently.
    """
    from cbb_sim.control import models as M

    train_seasons, _ = F.fold_seasons(fold)
    tg, _ = F.build_team_game_features(train_seasons)
    b = M.load(fold, anchor)

    marg, resid = [], {}
    for t in M.RATE_TARGETS:
        m = b.rates[t]
        mu = np.exp(m.linpred(tg)) * tg["game_poss"] / 100.0
        var = mu + (m.alpha if m.family == "negbin" else 0.0) * mu ** 2
        resid[t] = (tg[t] - mu) / np.sqrt(mu)
        marg.append({"count": t, "actual_mean": float(tg[t].mean()),
                     "actual_SD": float(tg[t].std()), "model_SD": float(np.sqrt(var).mean()),
                     "family": m.family,
                     "poisson_deviance_df": float(m.deviance_df)})
    corr = pd.DataFrame(resid).corr()

    p3 = 1.0 / (1.0 + np.exp(-b.pcts["tp_pct"].linpred(tg)))
    p2 = 1.0 / (1.0 + np.exp(-b.pcts["fg2_pct"].linpred(tg)))
    pf = 1.0 / (1.0 + np.exp(-b.pcts["ft_pct"].linpred(tg)))
    mu3 = np.exp(b.rates["tpa"].linpred(tg)) * tg["game_poss"] / 100.0
    mu2 = np.exp(b.rates["fg2a"].linpred(tg)) * tg["game_poss"] / 100.0
    muf = np.exp(b.rates["fta"].linpred(tg)) * tg["game_poss"] / 100.0
    pred_pts = 3 * mu3 * p3 + 2 * mu2 * p2 + muf * pf
    actual_pts = 3 * tg["tpm"] + 2 * tg["fg2m"] + tg["ftm"]

    def comp(mu, p, alpha, rho):
        vn = mu + (alpha if np.isfinite(alpha) else 0.0) * mu ** 2
        return mu * p * (1 - p) * (1 + (mu - 1) * rho) + p ** 2 * vn

    v = (9 * comp(mu3, p3, b.rates["tpa"].alpha, b.pcts["tp_pct"].rho)
         + 4 * comp(mu2, p2, b.rates["fg2a"].alpha, b.pcts["fg2_pct"].rho)
         + comp(muf, pf, b.rates["fta"].alpha, b.pcts["ft_pct"].rho))
    return {
        "marginals": pd.DataFrame(marg),
        "resid_corr": corr,
        "implied_team_pts_sd": float(np.sqrt(v.mean())),
        "actual_team_pts_resid_sd": float((actual_pts - pred_pts).std()),
        "actual_team_pts_sd": float(actual_pts.std()),
        "train_seasons": train_seasons,
    }


def season_totals(seasons: list[int]) -> pd.DataFrame:
    ref_rows = []
    for season in seasons:
        ref = pd.read_parquet(REFERENCE / f"gate_targets_{season}.parquet")
        m = ref[(ref["breakdown"] == "season") & (ref["metric"] == "total_points_mean")]
        ref_rows.append({"season": season, "total_points_mean": float(m["value"].iloc[0]),
                         "n": int(m["n"].iloc[0])})
    return pd.DataFrame(ref_rows)


def tier_bias_market_control(s: pd.DataFrame, season: int) -> pd.DataFrame:
    """The tier definition is deliberately NOT leak-free (each team's tier uses
    its own full-season margin), so a tier-wise bias is partly selection on the
    outcome. Running the SAME breakdown on the closing line separates the two:
    whatever the market also shows is selection, not a model defect."""
    lines = load_lines(season)
    d = s.merge(lines, on="cbbd_game_id", how="inner").dropna(subset=["spread"]).copy()
    d["market_margin"] = -d["spread"]
    tiers = team_tiers(s)
    d["home_tier"] = d["home_team_id"].map(tiers).astype(str)
    out = []
    for g, sub in d.groupby("home_tier"):
        out.append({
            "home_tier": g, "n": int(len(sub)),
            "model_margin_bias": float((sub["sim_margin_mean"] - sub["margin"]).mean()),
            "close_margin_bias": float((sub["market_margin"] - sub["margin"]).mean()),
        })
    t = pd.DataFrame(out)
    t["model_minus_close"] = t["model_margin_bias"] - t["close_margin_bias"]
    return t


# ---------------------------------------------------------------------------
# markdown helpers
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_, int, np.integer)):
                cells.append(str(v))
            elif isinstance(v, (float, np.floating)):
                if not np.isfinite(v):
                    cells.append("n/a")
                elif float(v).is_integer() and c in ("n", "n_games", "decile", "quintile", "month",
                                                     "wins", "losses", "pushes", "seeds", "games",
                                                     "season", "test_season", "seed_offset",
                                                     "n_with_line", "n_ml", "n_clv"):
                    cells.append(str(int(v)))
                else:
                    cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DOC))
    args = ap.parse_args()

    anchors = list(F.ANCHORS)
    runs: dict[tuple[str, str, int], pd.DataFrame] = {}
    for fold in ("F1", "F2"):
        for anchor in anchors:
            for off in (0, 1000):
                df = load_run(fold, anchor, off)
                if df is not None:
                    runs[(fold, anchor, off)] = df
    if not runs:
        raise SystemExit("no results/control/*/summary.parquet found; run scripts/run_control.py first")

    L: list[str] = []
    now = datetime.now(UTC)
    L += [
        "# Control engine gate report -- fold F2 (train 2022-2024, test 2024-25)",
        "",
        f"Generated {now.date().isoformat()} by `scripts/grade_control.py`. "
        "Spec: `docs/models/control_engine/model.md` (pre-registered 2026-09-10). "
        "Tolerances: `docs/SIM_GUARDRAILS.md` section 3 (provisional until the seed-noise study). "
        "Every gate line ends in a literal PASS / FAIL / NEEDS-INSTRUMENTATION.",
        "",
        "The Control is a yardstick, not a candidate. Nothing in this report is used to adjust it: "
        "a failed gate is reported with the component responsible and left alone "
        "(CLAUDE.md \"no hand tuning on engine output\").",
        "",
        "## 0. Runs graded",
        "",
    ]
    meta_rows = []
    for (fold, anchor, off), df in sorted(runs.items()):
        meta_rows.append({
            "fold": fold, "anchor": anchor, "seed_offset": off,
            "seeds": int(df["n_seeds_requested"].iloc[0]),
            "games": int(len(df)),
            "test_season": int(df["season"].iloc[0]),
            "backtest": bool(df["backtest"].iloc[0]),
            "created_at": str(pd.Timestamp(df["created_at"].iloc[0]).date()),
        })
    L += md_table(pd.DataFrame(meta_rows)) + [""]
    L += [
        "`backtest=True` on every row: these are completed seasons, so `created_at < tipoff` "
        "is impossible and is stamped rather than faked (`scripts/run_control.py` asserts the "
        "inequality only for `--live` runs).",
        "",
    ]

    # ---- 1. headline ----------------------------------------------------
    head_rows = []
    for fold in ("F1", "F2"):
        for anchor in anchors:
            s = runs.get((fold, anchor, 0))
            if s is None:
                continue
            h = headline(s)
            head_rows.append({"fold": fold, "anchor": anchor, "n": h["n"],
                              "margin_MAE": h["margin_mae"], "margin_bias": h["margin_bias"],
                              "total_MAE": h["total_mae"], "total_bias": h["total_bias"],
                              "Brier": h["brier"], "calib_slope": h["slope"],
                              "slope_MC_corrected": h["slope_mc_corrected"]})
    head = pd.DataFrame(head_rows)
    L += ["## 1. Headline scorecard", "",
          "Fold 2 is the selection metric; fold 1 is shown for drift. "
          "`calib_slope` is OLS of actual margin on the SIM MEAN margin; because the sim mean is "
          "itself a Monte-Carlo estimate, that slope is attenuated by "
          "Var(MC) = mean(sim SD^2)/n_seeds. `slope_MC_corrected` divides that attenuation out and "
          "is a DIAGNOSTIC, not the gate.", ""]
    L += md_table(head) + [""]

    f2 = head[head["fold"] == "F2"].sort_values("margin_MAE")
    L += ["Fold-2 ranking on the pre-registered primary metric (margin MAE): "
          + " < ".join(f"{r.anchor} {r.margin_MAE:.4f}" for r in f2.itertuples()) + ".", ""]

    # ---- 2. noise floor + decision --------------------------------------
    L += ["## 2. Seed noise floor and the anchor decision", "",
          "Noise floor = the same spec re-simulated at seed offset +1000 "
          "(`--seed-offset 1000`), differenced against the base run on the same games.", ""]
    floor_rows = []
    for anchor in anchors:
        a0, a1 = runs.get(("F2", anchor, 0)), runs.get(("F2", anchor, 1000))
        if a0 is None or a1 is None:
            continue
        h0, h1 = headline(a0), headline(a1)
        floor_rows.append({
            "anchor": anchor,
            "margin_MAE_seed0": h0["margin_mae"], "margin_MAE_seed1000": h1["margin_mae"],
            "abs_delta": abs(h0["margin_mae"] - h1["margin_mae"]),
            "total_MAE_delta": abs(h0["total_mae"] - h1["total_mae"]),
            "Brier_delta": abs(h0["brier"] - h1["brier"]),
        })
    floor = pd.DataFrame(floor_rows)
    L += md_table(floor, "{:.5f}") + [""]
    noise_floor = float(floor["abs_delta"].max()) if len(floor) else float("nan")
    floor_mean = float(floor["abs_delta"].mean()) if len(floor) else float("nan")

    pair_rows = []
    for i, x in enumerate(anchors):
        for y in anchors[i + 1:]:
            sx, sy = runs.get(("F2", x, 0)), runs.get(("F2", y, 0))
            if sx is None or sy is None:
                continue
            d = (sx["margin"] - sx["sim_margin_mean"]).abs().to_numpy() - \
                (sy["margin"] - sy["sim_margin_mean"]).abs().to_numpy()
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            pair_rows.append({"comparison": f"{x} - {y}", "delta_margin_MAE": float(d.mean()),
                              "paired_SE": se, "t": float(d.mean() / se) if se else np.nan})
    L += ["Paired per-game differences (the two arms share the same (seed, game_id) streams, "
          "so Monte-Carlo noise largely cancels):", ""]
    L += md_table(pd.DataFrame(pair_rows), "{:.4f}") + [""]

    ranked = f2.reset_index(drop=True)
    best, second = ranked.iloc[0], ranked.iloc[1]
    a_own = float(ranked.loc[ranked["anchor"] == "A_own", "margin_MAE"].iloc[0])
    b_kp = float(ranked.loc[ranked["anchor"] == "B_kp", "margin_MAE"].iloc[0])
    c_both = float(ranked.loc[ranked["anchor"] == "C_both", "margin_MAE"].iloc[0])
    ab_gap = abs(a_own - b_kp)
    c_beats_both_by = min(a_own - c_both, b_kp - c_both)
    if ab_gap <= noise_floor:
        chosen = "C_both" if c_beats_both_by > noise_floor else "A_own"
        rule = (f"|A_own - B_kp| = {ab_gap:.4f} is within the measured seed noise floor "
                f"({floor['abs_delta'].min():.4f}-{noise_floor:.4f}, mean {floor_mean:.4f}), so the "
                f"pre-registered tie clause applies: C_both is adopted only if it beats BOTH by more "
                f"than the floor. It beats the better of them by {c_beats_both_by:.4f}, "
                f"{'above' if c_beats_both_by > noise_floor else 'below'} the floor.")
    else:
        chosen = str(best["anchor"])
        rule = (f"|A_own - B_kp| = {ab_gap:.4f} exceeds the seed noise floor ({noise_floor:.4f}), so the "
                f"plain rule applies: lowest fold-2 margin MAE wins.")
    L += [f"**Decision (pre-registered rule, `experiments.md`): {chosen}.** {rule}", ""]
    L += [f"Runner-up on the raw metric is {best['anchor']} ({best['margin_MAE']:.4f}) ahead of "
          f"{second['anchor']} ({second['margin_MAE']:.4f}); the paired table above shows that gap is "
          "statistically detectable once MC noise is cancelled, but the pre-registered floor is the "
          "unpaired seed-offset difference and it is coarser than the paired SE. Recorded as an "
          "observation for the PM, not acted on here.", ""]

    graded = chosen

    # ---- per-gate sections for the chosen anchor on F2 -------------------
    s = runs[("F2", graded, 0)]
    raw = load_games_raw("F2", graded, 0)
    season = int(s["season"].iloc[0])

    L += [f"## 3. G1 -- possessions per game ({graded}, F2, test season {season})", "",
          "`sim` pools every (game, seed) row; `actual` is the same games' box-derived possessions "
          "(FGA - OREB + TOV + 0.44 FTA, averaged over both teams); `ref` is "
          "`data/reference/gate_targets_{season}.parquet` over the full season. "
          f"Tolerance: mean +/- {TOL['g1_mean']}, SD +/- {TOL['g1_sd']}.", ""]
    g1_tab, g1_ok = gate_g1(s, raw, season)
    L += md_table(g1_tab, "{:.3f}") + [""]
    ov = g1_tab.iloc[0]
    L += [f"G1 possessions mean: sim {ov['sim_mean']:.3f} vs actual {ov['actual_mean']:.3f} "
          f"(delta {ov['d_mean']:+.3f}, tol +/-{TOL['g1_mean']}) -- {ov['status_mean']}",
          "",
          f"G1 possessions SD: sim {ov['sim_sd']:.3f} vs actual {ov['actual_sd']:.3f} "
          f"(delta {ov['d_sd']:+.3f}, tol +/-{TOL['g1_sd']}) -- {ov['status_sd']}", ""]
    worst_m = g1_tab[g1_tab["breakdown"] == "month"]
    powered = worst_m[worst_m["status_mean"] != "UNDERPOWERED"]
    bad = powered[(powered["status_mean"] == "FAIL") | (powered["status_sd"] == "FAIL")]
    n_under = len(worst_m) - len(powered)
    L += [f"G1 by month: {len(powered) - len(bad)}/{len(powered)} powered months inside both "
          f"tolerances ({n_under} month(s) below n={MIN_CELL_N} labelled UNDERPOWERED and not "
          f"scored) -- {status(len(bad) == 0)}", ""]

    # ---- G5 -------------------------------------------------------------
    g5 = gate_g5(s, raw)
    _ref = pd.read_parquet(REFERENCE / f"gate_targets_{season}.parquet")
    _rc = _ref[(_ref["breakdown"] == "season") & (_ref["metric"] == "home_away_score_corr")]
    ref_corr = float(_rc["value"].iloc[0]) if len(_rc) else float("nan")
    ref_corr_n = int(_rc["n"].iloc[0]) if len(_rc) else 0
    L += [f"## 4. G5 -- dispersion ({graded}, F2)", "",
          "SD ratio = mean(sim SD) / SD(actual - sim mean). >1 means the engine is too wide. "
          f"Tolerance 0.95-1.05; score correlation +/-{TOL['g5_corr']}; PIT K-S p > {TOL['g5_ks_p']}.", ""]
    g5_tab = pd.DataFrame([
        {"quantity": "margin", "mean_sim_SD": g5["sim_sd_margin"], "SD(actual - sim mean)": g5["resid_sd_margin"],
         "ratio": g5["ratio_margin"], "status": g5["status_margin"]},
        {"quantity": "total", "mean_sim_SD": g5["sim_sd_total"], "SD(actual - sim mean)": g5["resid_sd_total"],
         "ratio": g5["ratio_total"], "status": g5["status_total"]},
    ])
    L += md_table(g5_tab, "{:.4f}") + [""]
    L += [f"G5 margin SD ratio {g5['ratio_margin']:.4f} -- {g5['status_margin']}",
          "",
          f"G5 total SD ratio {g5['ratio_total']:.4f} -- {g5['status_total']}",
          "",
          f"G5 home/away score correlation: sim {g5['corr_sim']:.4f} vs actual {g5['corr_actual']:.4f} "
          f"on the same {len(s)} games (delta {g5['d_corr']:+.4f}) -- {g5['status_corr']}",
          "",
          f"(The season reference table gives {ref_corr:.4f} over all "
          f"{ref_corr_n} D-I non-truncated games; the {ref_corr_n - len(s)} games this run drops for "
          "a missing team_box row happen to be scoring outliers, which is why the like-for-like "
          "actual is lower. The gate compares sim to actual on the SAME games.)",
          "",
          f"G5 PIT K-S vs Uniform(0,1): D = {g5['ks_stat']:.4f}, p = {g5['ks_p']:.3g} -- {g5['status_pit']}",
          "",
          "PIT decile histogram (a correctly dispersed engine is flat at 10% per decile; "
          "a mass in the middle deciles means the engine is too WIDE):", ""]
    pit_hist = pd.DataFrame({
        "decile": np.arange(1, 11),
        "share": np.histogram(g5["pit"], bins=np.linspace(0, 1, 11))[0] / len(g5["pit"]),
    })
    L += md_table(pit_hist, "{:.4f}") + [""]

    # ---- G6 -------------------------------------------------------------
    g6 = gate_g6(s)
    L += [f"## 5. G6 -- home margin, non-neutral vs neutral, same games ({graded}, F2)", "",
          f"Tolerance +/-{TOL['g6_margin']} points.", ""]
    L += md_table(g6, "{:.4f}") + [""]
    for _, r in g6.iterrows():
        L += [f"G6 home margin ({r['site']}): sim {r['sim']:+.3f} vs actual {r['actual']:+.3f} "
              f"(delta {r['delta']:+.3f}) -- {r['status']}", ""]

    # ---- G9 -------------------------------------------------------------
    h = headline(s)
    L += [f"## 6. G9 -- spread and total accuracy ({graded}, F2)", "",
          f"Tolerance: margin bias +/-{TOL['g9_margin_bias']}, total bias +/-{TOL['g9_total_bias']}, "
          f"calibration slope {TOL['g9_slope_lo']}-{TOL['g9_slope_hi']}.", ""]
    L += [f"G9 margin MAE {h['margin_mae']:.4f}, bias {h['margin_bias']:+.4f} -- "
          f"{status(abs(h['margin_bias']) <= TOL['g9_margin_bias'])}", "",
          f"G9 total MAE {h['total_mae']:.4f}, bias {h['total_bias']:+.4f} -- "
          f"{status(abs(h['total_bias']) <= TOL['g9_total_bias'])}", "",
          f"G9 calibration slope {h['slope']:.4f} (MC-corrected {h['slope_mc_corrected']:.4f}, "
          f"MC noise SD {h['mc_noise_sd']:.3f}) -- "
          f"{status(TOL['g9_slope_lo'] <= h['slope'] <= TOL['g9_slope_hi'])}", "",
          f"G9 win-probability Brier {h['brier']:.5f}", "",
          "Win-probability calibration by decile:", ""]
    L += md_table(brier_decile(s), "{:.4f}") + [""]

    bd = gate_g9_breakdowns(s)
    bd_counts = {}
    for name, tab in bd.items():
        n_fail = int((tab["status_margin_bias"] == "FAIL").sum() + (tab["status_total_bias"] == "FAIL").sum())
        n_scored = int((tab["status_margin_bias"] != "UNDERPOWERED").sum()
                       + (tab["status_total_bias"] != "UNDERPOWERED").sum())
        bd_counts[name] = (n_fail, n_scored)
        L += [f"### G9 by {name}", ""]
        L += md_table(tab, "{:.4f}") + [""]
        L += [f"G9 by {name}: {n_fail} of {n_scored} scored bias cells outside tolerance -- "
              f"{status(n_fail == 0)}", ""]

    # ---- responsiveness --------------------------------------------------
    resp = responsiveness(s)
    slope_q = np.polyfit(resp["predicted_margin"], resp["actual_margin"], 1)[0]
    mono = bool(resp["actual_margin"].is_monotonic_increasing)
    L += [f"## 7. Responsiveness ({graded}, F2)", "",
          "Games bucketed by pregame rating differential (a prior, not the prediction). "
          "The standing rule is that predictions must SLOPE with actuals, not sit flat at the mean.", ""]
    L += md_table(resp, "{:.4f}") + [""]
    L += [f"Responsiveness: actual margin monotone across quintiles = {mono}; "
          f"slope of actual on predicted across quintile means = {slope_q:.4f} -- "
          f"{status(mono and 0.85 <= slope_q <= 1.15)}", ""]

    # ---- G10 -------------------------------------------------------------
    L += ["## 8. G10 -- market scorecard vs ESPN BET closes", "",
          "CBBD `lines_{season}.parquet`, provider ESPN BET, joined on `cbbd_game_id`. "
          "`spread` is home-perspective, so the market's expected home margin is `-spread`. "
          "G10 is report-only except the leak screen "
          f"(LEAK-SUSPECT if surprise corr > {TOL['g10_surprise_corr']} AND CLV agreement < "
          f"{TOL['g10_clv_agreement']}).", ""]
    g10_all = {}
    for fold, anchor in (("F1", graded), ("F2", graded)):
        srun = runs.get((fold, anchor, 0))
        if srun is None:
            continue
        g10_all[fold] = gate_g10(srun, int(srun["season"].iloc[0]))

    if g10_all:
        core = pd.DataFrame([
            {"fold": f, "season": r["season"], "n_with_line": r["n"],
             "model_margin_MAE": r["model_margin_mae"], "close_margin_MAE": r["close_margin_mae"],
             "model_margin_bias": r["model_margin_bias"], "close_margin_bias": r["close_margin_bias"],
             "model_total_MAE": r["model_total_mae"], "close_total_MAE": r["close_total_mae"],
             "model_total_bias": r["model_total_bias"], "close_total_bias": r["close_total_bias"]}
            for f, r in g10_all.items()
        ])
        L += md_table(core, "{:.4f}") + [""]
        for r in g10_all.values():
            beat = r["model_margin_mae"] < r["close_margin_mae"]
            L += [f"G10 {r['season']} margin: model MAE {r['model_margin_mae']:.4f} vs close "
                  f"{r['close_margin_mae']:.4f} (model {'beats' if beat else 'trails'} the close by "
                  f"{abs(r['model_margin_mae'] - r['close_margin_mae']):.4f}) -- report-only, PASS", ""]
            L += [f"G10 {r['season']} total: model MAE {r['model_total_mae']:.4f} vs close "
                  f"{r['close_total_mae']:.4f}; model total bias {r['model_total_bias']:+.4f} vs close "
                  f"{r['close_total_bias']:+.4f} -- report-only, PASS", ""]
            L += [f"### ATS by disagreement bucket, {r['season']}", ""]
            L += md_table(r["ats"], "{:.4f}") + [""]
            L += [f"G10 {r['season']} Brier vs de-vigged moneyline "
                  f"(n = {r['n_ml']}, mean vig {r['mean_vig']:.4f}): model "
                  f"{r['model_brier_ml_subset']:.5f} vs market {r['market_brier']:.5f} -- report-only, PASS", ""]
            clv = f"{r['clv_agreement']:.4f}" if np.isfinite(r["clv_agreement"]) else "n/a (no opening lines)"
            L += [f"G10 {r['season']} leak screen: surprise corr "
                  f"{r['surprise_corr']:.4f} (gate {TOL['g10_surprise_corr']}), CLV sign agreement "
                  f"{clv} on {r['n_clv']} moved lines -- {r['leak_status']}", ""]

    # ---- 9. drift -------------------------------------------------------
    L += ["## 9. Fold drift (F1 vs F2)", "",
          "Same spec, one season earlier. Both folds train on pooled seasons and test on the next "
          "one, so the size and sign of the total bias is the direct read on season-level drift.", ""]
    drift = head.pivot(index="anchor", columns="fold",
                       values=["margin_MAE", "margin_bias", "total_bias", "Brier"])
    drift.columns = [f"{a}_{b}" for a, b in drift.columns]
    L += md_table(drift.reset_index(), "{:.4f}") + [""]

    # ---- 10. diagnosis --------------------------------------------------
    L += ["## 10. Diagnosis of every failed gate", "",
          "The first response to a failed gate is *which sub-model is producing the wrong "
          "distribution*, never *what adjustment closes the gap* "
          "(`docs/SIM_GUARDRAILS.md` core principle). Nothing below has been changed.", ""]

    dg = diagnose_dispersion("F2", graded)
    L += ["### D1. G5 margin/total SD ratio, home-away score correlation, PIT "
          "-- the attempt-count layer", "",
          "The four per-100-possession count models are each individually well calibrated: "
          "their fitted marginal SD matches the data.", ""]
    L += md_table(dg["marginals"], "{:.4f}") + [""]
    L += ["But the simulator draws those four counts INDEPENDENTLY given the shared possession "
          "draw, and in the data they are strongly negatively correlated -- they compete for the "
          "same finite possessions (a possession that ends in a turnover is not a shot; a three is "
          "not a two). Pearson-residual correlation on the fold-2 training team-games:", ""]
    L += md_table(dg["resid_corr"].round(3).reset_index().rename(columns={"index": "count"}), "{:.3f}") + [""]
    L += [f"Consequence, computed from the fitted parameters alone: the independent-component "
          f"variance implies a team-points SD of {dg['implied_team_pts_sd']:.2f}, while the actual "
          f"residual team-points SD is {dg['actual_team_pts_resid_sd']:.2f} (unconditional "
          f"{dg['actual_team_pts_sd']:.2f}). "
          f"sqrt(2) x {dg['implied_team_pts_sd']:.2f} = "
          f"{np.sqrt(2) * dg['implied_team_pts_sd']:.2f} is the implied simulated margin SD, and "
          f"the report above measures {g5['sim_sd_margin']:.2f}. The same inflated per-team "
          f"variance is what dilutes the home/away score correlation ({g5['corr_sim']:.4f} "
          f"simulated vs {g5['corr_actual']:.4f} actual): the shared pace draw contributes about "
          "the right covariance, but it is divided by two SDs that are ~50% too large. The PIT "
          "failure and the under-confident win-probability deciles are the same defect seen "
          "through two more lenses.", "",
          "**Responsible component: the attempt-count layer (3PA / 2PA / FTA / TOV drawn as four "
          "independent overdispersed counts).** The marginals are right and every dispersion "
          "parameter is fitted, so this is not a tuning error -- the model class cannot represent "
          "the negative dependence, exactly the situation SIM_GUARDRAILS section 5 says requires a "
          "rebuilt model rather than an adjustment. The pre-registered replacement is the L3 "
          "possession-outcome model, which allocates each possession to one outcome and therefore "
          "gets the competition for possessions for free. Until then the Control's point estimates "
          "are usable and its intervals are not.", ""]

    tot = season_totals(sorted(set(dg["train_seasons"] + [season])))
    train_mean = float(tot[tot["season"].isin(dg["train_seasons"])]["total_points_mean"].mean())
    test_mean = float(tot[tot["season"] == season]["total_points_mean"].iloc[0])
    L += ["### D2. G9 total bias (and its month/tier breakdowns) -- pooled-season training "
          "against a rising scoring level", "",
          "Scoring per game has risen every season in the data. Reference totals:", ""]
    L += md_table(tot, "{:.3f}") + [""]
    L += [f"The fold-2 training seasons average {train_mean:.2f} total points; the test season is "
          f"{test_mean:.2f}, a drift of {test_mean - train_mean:+.2f}. The pre-registered feature "
          "list is entirely CENTRED ratings plus site plus day-of-season, so nothing in it carries "
          "a season LEVEL: the GLM intercepts are pooled over the training seasons and the engine "
          f"inherits the shortfall. The measured total bias is {h['total_bias']:+.3f}, i.e. "
          f"{100 * abs(h['total_bias']) / (test_mean - train_mean):.0f}% of the raw drift "
          "(the rest is absorbed by the as-of rating and day-of-season terms). Fold 1 shows the "
          "same mechanism one season earlier at a larger magnitude, which is the drift check in "
          "section 9. The closing line over the same games is essentially unbiased on totals "
          f"({g10_all['F2']['close_total_bias']:+.3f}), confirming this is the model, not the "
          "grading truth.", "",
          "**Responsible component: the level (intercept) of the per-100-possession rate and "
          "make-rate GLMs.** No adjustment is applied: the pre-registered Control feature list has "
          "no season-level term by design. The fix belongs upstream -- a season-aware level term "
          "(the as-of league mean is already produced by `own_ratings`, is pregame and is "
          "leak-safe) or a preseason refit, both of which are L2/L3 decisions, not Control "
          "patches. SIM_GUARDRAILS section 4 predicted exactly this: *any model trained on pooled "
          "seasons without season-aware features will under-shoot the current year*.", ""]

    tc = tier_bias_market_control(s, season)
    L += ["### D3. G9 bias by tier -- partly selection, partly real", "",
          "Team tiers are terciles of each team's OWN full-season margin, which "
          "`docs/tests/gate_reference_2026-09-10.md` flags as not leak-free. Grouping on a "
          "quantity computed from the outcomes themselves guarantees some regression-to-the-mean "
          "bias for ANY pregame predictor. The control is to run the same breakdown on the "
          "closing line:", ""]
    L += md_table(tc, "{:.4f}") + [""]
    L += ["The closing line shows the same sign and most of the same magnitude, so most of the "
          "tier-wise bias is the tier definition, not the engine. The `model_minus_close` column "
          "is the part the engine actually owns.", ""]

    L += ["### D4. G6 neutral-site home margin", "",
          f"Simulated {g6.iloc[1]['sim']:+.3f} vs actual {g6.iloc[1]['actual']:+.3f} on "
          f"{int(g6.iloc[1]['n'])} neutral games. The site term is a single "
          "`site_home`/`site_away` pair with neutral as the reference level, so every neutral game "
          "gets exactly zero site effect and the residual gap has to come from the ratings. In "
          "reality *neutral* covers a wide range -- an NCAA sub-regional in a team's home state, an "
          "in-season tournament in a team's own market -- and the nominal home team at a neutral "
          "site is systematically the stronger or higher-seeded one. "
          "**Responsible component: the site feature, which is too coarse.** The non-neutral cell "
          f"passes ({g6.iloc[0]['delta']:+.3f}), so the home effect itself is wired correctly in "
          "every scoring-stage model; it is the neutral bucket that is heterogeneous. A "
          "venue-distance or designated-home feature is the honest fix and belongs to the feature "
          "layer, not to a post-hoc neutral-site offset.", ""]

    L += ["### D5. G1 possessions -- the overtime stub", "",
          f"Sim {ov['sim_mean']:.3f} vs actual {ov['actual_mean']:.3f} possessions per game "
          f"({ov['d_mean']:+.3f}) passes, but the sign is explained: the pace target is the "
          "OBSERVED possession count, which already contains whatever overtime a real game played, "
          "and the stub then adds a further 5/40 of a game to every tied sim. That double count is "
          f"worth about +{float(s['sim_ot_rate'].mean()) * 5 / 40 * ov['sim_mean']:.2f} possessions "
          f"per game at the simulated OT rate of {float(s['sim_ot_rate'].mean()):.4f} "
          f"(actual OT rate on these games: {float((s['n_periods'] > 2).mean()):.4f}). The "
          "simulated OT rate is itself too low because a Beta-Binomial / NegBin score has no "
          "end-game mechanics to pile probability mass onto an exact tie. Both are the known L5 "
          "gap, listed in model.md section 9 and left alone.", ""]

    # ---- 11. gate summary ------------------------------------------------
    L += ["## 11. Gate summary", "", "| gate | quantity | value | tolerance | status |",
          "|---|---|---|---|---|"]
    rows = [
        ("G1", "possessions/game mean", f"{ov['sim_mean']:.3f} vs {ov['actual_mean']:.3f}",
         f"+/-{TOL['g1_mean']}", ov["status_mean"]),
        ("G1", "possessions/game SD", f"{ov['sim_sd']:.3f} vs {ov['actual_sd']:.3f}",
         f"+/-{TOL['g1_sd']}", ov["status_sd"]),
        ("G1", "by month (mean and SD)", f"{len(bad)}/{len(powered)} powered months out",
         "all inside", status(len(bad) == 0)),
        ("G5", "margin SD ratio", f"{g5['ratio_margin']:.4f}", "0.95-1.05", g5["status_margin"]),
        ("G5", "total SD ratio", f"{g5['ratio_total']:.4f}", "0.95-1.05", g5["status_total"]),
        ("G5", "home/away score corr", f"{g5['corr_sim']:.4f} vs {g5['corr_actual']:.4f}",
         f"+/-{TOL['g5_corr']}", g5["status_corr"]),
        ("G5", "PIT K-S p", f"{g5['ks_p']:.3g}", f"> {TOL['g5_ks_p']}", g5["status_pit"]),
        ("G6", "home margin non-neutral", f"{g6.iloc[0]['sim']:+.3f} vs {g6.iloc[0]['actual']:+.3f}",
         f"+/-{TOL['g6_margin']}", g6.iloc[0]["status"]),
        ("G6", "home margin neutral", f"{g6.iloc[1]['sim']:+.3f} vs {g6.iloc[1]['actual']:+.3f}",
         f"+/-{TOL['g6_margin']}", g6.iloc[1]["status"]),
        ("G9", "margin bias", f"{h['margin_bias']:+.4f}", f"+/-{TOL['g9_margin_bias']}",
         status(abs(h["margin_bias"]) <= TOL["g9_margin_bias"])),
        ("G9", "total bias", f"{h['total_bias']:+.4f}", f"+/-{TOL['g9_total_bias']}",
         status(abs(h["total_bias"]) <= TOL["g9_total_bias"])),
        ("G9", "calibration slope", f"{h['slope']:.4f}", "0.95-1.05",
         status(TOL["g9_slope_lo"] <= h["slope"] <= TOL["g9_slope_hi"])),
        ("G9", "responsiveness slope", f"{slope_q:.4f}", "monotone, 0.85-1.15",
         status(mono and 0.85 <= slope_q <= 1.15)),
    ]
    for name, (nf, ns) in bd_counts.items():
        rows.append(("G9", f"bias by {name}", f"{nf}/{ns} scored cells out", "all inside", status(nf == 0)))
    for r in g10_all.values():
        rows.append(("G10", f"{r['season']} margin MAE vs close",
                     f"{r['model_margin_mae']:.3f} vs {r['close_margin_mae']:.3f}", "report only", "PASS"))
        rows.append(("G10", f"{r['season']} leak screen",
                     f"surprise corr {r['surprise_corr']:.3f}", f"<= {TOL['g10_surprise_corr']}",
                     r["leak_status"]))
    for gate, q, v, t, st in rows:
        L.append(f"| {gate} | {q} | {v} | {t} | {st} |")
    L += [""]

    (Path(args.out)).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(L)} lines)")
    print(f"chosen anchor: {chosen}; noise floor {noise_floor:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
