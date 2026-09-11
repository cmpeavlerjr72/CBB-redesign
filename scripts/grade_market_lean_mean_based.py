"""
grade_market_lean_mean_based.py -- PROVISIONAL, mean-based ATS/O-U lean on a
200-seed engine v1 run. NOT the pre-registered ROI read.

Why this exists (2026-09-11, requested after `market_games_v2_*_s200_*`
correctly REFUSED to print ROI/Brier/edge-bucket numbers below the
2,000-seed floor, docs/tests/engine_seed_count_2026-09-10.md): a per-game
sim win PROBABILITY carries ~0.033-0.035 (3.3-3.5pp) SE at 200 seeds (that
doc, section on G10) -- large relative to the ~0.5 baseline a probability
edge is measured against, so probability-bucketed ROI is noise-dominated.
The sim's per-game MEAN margin/total also carry real per-game seed SE at
200 seeds (~2.1 pt margin, ~0.84-1.2 pt total per the same study,
extrapolated in `grade_market_games_v2.SEED_STUDY_SE_MEAN`) -- NOT
sub-point on a single game. What IS sub-point at 200 seeds is the
AGGREGATE (whole-slate) mean: `gate_noise_band_F2_2025_s200_v1_clockv3c_
2026-09-11.md`'s paired independent-200-seed-run comparison (run A vs run
B, different seed offsets) shows the slate-level margin bias moved only
0.0109 pts and total bias only 0.0366 pts between two independent 200-seed
draws. That is the empirical basis for a directional (not per-game-precise)
flat-stake mean-based lean being legitimate at 200 seeds, while a
probability-bucketed ROI read is not. Section 3 of this script's own output
measures the same kind of noise directly for THIS read (split-half, 100 vs
100 seeds) rather than relying on the A/B report alone.

This script does NOT compute probability-based edge buckets, does NOT
touch `grade_market_games_v2.py`'s ROI/Brier refusal logic, and does NOT
overwrite any results file. It reuses that script's truth join
(`load_truth`), lines join (`load_close_lines`), and Wilson CI helper
(`wilson_ci`) by importing them directly rather than reimplementing.

    .venv/Scripts/python.exe scripts/grade_market_lean_mean_based.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402

import grade_market_games_v2 as gmv2  # noqa: E402 -- reuse its join/guards, not rebuild

SEASON = 2025
RESULTS_A = ROOT / "results" / "engine_v0" / "F2_2025_s200_v1_clockv3c_A"
RESULTS_CONTROL = ROOT / "results" / "control" / "F2_A_own"
OUT_MD = ROOT / "docs" / "tests" / "market_lean_mean_based_s200_2026-09-11.md"
BANDS = ((0, 1), (1, 2), (2, 3), (3, 5), (5, np.inf))
UNDERPOWERED_N = 300


def ats_ou_frame(sim_summary: pd.DataFrame, truth: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    full = truth.merge(sim_summary, on="game_id", how="inner").merge(lines, on="game_id", how="inner")
    full["market_margin"] = -full["close_spread_home"]
    full["market_total"] = full["close_over_under"]
    full["model_margin"] = full["sim_margin_mean"]
    full["model_total"] = full["sim_total_mean"]
    full["disagree_ats"] = full["model_margin"] - full["market_margin"]
    full["cover_ats"] = full["margin"] - full["market_margin"]
    full["side_ats"] = np.sign(full["disagree_ats"])
    full["result_ats"] = np.sign(full["cover_ats"]) * full["side_ats"]
    full["disagree_ou"] = full["model_total"] - full["market_total"]
    full["cover_ou"] = full["total"] - full["market_total"]
    full["side_ou"] = np.sign(full["disagree_ou"])
    full["result_ou"] = np.sign(full["cover_ou"]) * full["side_ou"]
    return full


def record(result: pd.Series, mask: pd.Series | None = None) -> dict:
    r = result if mask is None else result[mask]
    w = int((r == 1).sum())
    l = int((r == -1).sum())
    p = int((r == 0).sum())
    n_dec = w + l
    lo, hi = gmv2.wilson_ci(w, n_dec)
    roi = (w - 1.1 * l) / n_dec if n_dec else float("nan")
    return {"n": w + l + p, "W": w, "L": l, "P": p, "n_dec": n_dec,
            "win_rate": (w / n_dec) if n_dec else float("nan"),
            "wilson_lo95": lo, "wilson_hi95": hi, "roi_-110": roi,
            "underpowered": (w + l + p) < UNDERPOWERED_N}


def row_md(label: str, rec: dict) -> str:
    wr = "n/a" if np.isnan(rec["win_rate"]) else f"{rec['win_rate']:.4f}"
    lo = "n/a" if np.isnan(rec["wilson_lo95"]) else f"{rec['wilson_lo95']:.4f}"
    hi = "n/a" if np.isnan(rec["wilson_hi95"]) else f"{rec['wilson_hi95']:.4f}"
    roi = "n/a" if np.isnan(rec["roi_-110"]) else f"{rec['roi_-110']:+.4f}"
    flag = " UNDERPOWERED" if rec["underpowered"] else ""
    return (f"| {label} | {rec['n']} | {rec['W']}-{rec['L']}-{rec['P']} | {wr} | "
            f"[{lo}, {hi}] | {roi} |{flag} |")


def band_tables(full: pd.DataFrame, disagree_col: str, result_col: str) -> list[str]:
    lines_out = ["| band | n | W-L-P | win rate | Wilson 95% CI | ROI @ -110 | |",
                 "|---|---:|---|---:|---|---:|---|"]
    disagree_abs = full[disagree_col].abs()
    for lo, hi in BANDS:
        m = (disagree_abs >= lo) & (disagree_abs < hi)
        rec = record(full[result_col], m)
        label = f"[{lo:g}, {hi if np.isfinite(hi) else 'inf'})"
        lines_out.append(row_md(label, rec))
    return lines_out


def month_tables(full: pd.DataFrame, result_col: str) -> list[str]:
    lines_out = ["| month | n | W-L-P | win rate | Wilson 95% CI | ROI @ -110 | |",
                 "|---|---:|---|---:|---|---:|---|"]
    for m in sorted(full["month"].unique()):
        mask = full["month"] == m
        rec = record(full[result_col], mask)
        lines_out.append(row_md(str(m), rec))
    return lines_out


def venue_side_tables(full: pd.DataFrame, side_col: str, result_col: str,
                       pos_label: str, neg_label: str) -> list[str]:
    """3-way split: neutral-site games (either side) vs non-neutral games
    split by which side the mean-based read picked. Definition stated
    explicitly here since 'home/away/neutral' has no single canonical
    meaning for an ATS/OU record; this is the natural partition covering
    every bet exactly once."""
    lines_out = ["| bucket | n | W-L-P | win rate | Wilson 95% CI | ROI @ -110 | |",
                 "|---|---:|---|---:|---|---:|---|"]
    neutral_mask = full["neutral"] == 1
    pos_mask = (~neutral_mask) & (full[side_col] > 0)
    neg_mask = (~neutral_mask) & (full[side_col] < 0)
    for label, mask in ((pos_label, pos_mask), (neg_label, neg_mask), ("neutral", neutral_mask)):
        rec = record(full[result_col], mask)
        lines_out.append(row_md(label, rec))
    return lines_out


def mae_bias(pred: pd.Series, actual: pd.Series) -> tuple[float, float]:
    return float((pred - actual).abs().mean()), float((pred - actual).mean())


def main() -> int:
    engine = C.load_engine_results(RESULTS_A)
    games = engine.games
    n_seeds_total = games["seed"].nunique()
    sim_full = G.summarise_games(games)

    truth, truth_counts = gmv2.load_truth(SEASON)
    lines, n_dup_lines = gmv2.load_close_lines(SEASON, gmv2.DEFAULT_LINES_PATH, gmv2.PRIMARY_PROVIDER)

    full = ats_ou_frame(sim_full, truth, lines)
    n_truth = truth_counts["n_truth_games"]
    n_full = len(full)

    # ---- baselines ---------------------------------------------------
    always_home = record(np.sign(full["cover_ats"]) * 1.0)
    always_over = record(np.sign(full["cover_ou"]) * 1.0)

    control_engine = C.load_engine_results(RESULTS_CONTROL)
    sim_control = G.summarise_games(control_engine.games)
    full_ctrl = ats_ou_frame(sim_control, truth, lines)
    ctrl_ats = record(full_ctrl["result_ats"])
    ctrl_ou = record(full_ctrl["result_ou"])
    ctrl_margin_mae, ctrl_margin_bias = mae_bias(full_ctrl["model_margin"], full_ctrl["margin"])
    ctrl_total_mae, ctrl_total_bias = mae_bias(full_ctrl["model_total"], full_ctrl["total"])

    sim_margin_mae, sim_margin_bias = mae_bias(full["model_margin"], full["margin"])
    sim_total_mae, sim_total_bias = mae_bias(full["model_total"], full["total"])
    mkt_margin_mae, mkt_margin_bias = mae_bias(full["market_margin"], full["margin"])
    mkt_total_mae, mkt_total_bias = mae_bias(full["market_total"], full["total"])

    # ---- overall records ----------------------------------------------
    overall_ats = record(full["result_ats"])
    overall_ou = record(full["result_ou"])

    # ---- split-half seed-noise check -----------------------------------
    half1 = games[games["seed"] < 100]
    half2 = games[games["seed"] >= 100]
    sim_h1 = G.summarise_games(half1)
    sim_h2 = G.summarise_games(half2)
    full_h1 = ats_ou_frame(sim_h1, truth, lines)
    full_h2 = ats_ou_frame(sim_h2, truth, lines)
    # align on game_id (inner join sets should match; assert and reindex)
    common_ids = sorted(set(full_h1["game_id"]) & set(full_h2["game_id"]) & set(full["game_id"]))
    h1 = full_h1.set_index("game_id").loc[common_ids]
    h2 = full_h2.set_index("game_id").loc[common_ids]
    flips_ats = int((np.sign(h1["side_ats"]) != np.sign(h2["side_ats"])).sum())
    flips_ou = int((np.sign(h1["side_ou"]) != np.sign(h2["side_ou"])).sum())
    rec_h1_ats = record(h1["result_ats"])
    rec_h2_ats = record(h2["result_ats"])
    rec_h1_ou = record(h1["result_ou"])
    rec_h2_ou = record(h2["result_ou"])

    # ---- band / month / venue tables -----------------------------------
    ats_bands = band_tables(full, "disagree_ats", "result_ats")
    ou_bands = band_tables(full, "disagree_ou", "result_ou")
    ats_months = month_tables(full, "result_ats")
    ou_months = month_tables(full, "result_ou")
    ats_venue = venue_side_tables(full, "side_ats", "result_ats", "home (picked home side)", "away (picked away side)")
    ou_venue_over = venue_side_tables(full, "side_ou", "result_ou", "over (picked over)", "under (picked under)")

    # ---- one-line answer -------------------------------------------------
    ats_wr = overall_ats["win_rate"]
    ou_wr = overall_ou["win_rate"]
    breakeven = 1.1 / 2.1  # 0.5238
    ats_inside_noise = abs(rec_h1_ats["win_rate"] - rec_h2_ats["win_rate"]) if not (np.isnan(rec_h1_ats["win_rate"]) or np.isnan(rec_h2_ats["win_rate"])) else float("nan")
    ou_inside_noise = abs(rec_h1_ou["win_rate"] - rec_h2_ou["win_rate"]) if not (np.isnan(rec_h1_ou["win_rate"]) or np.isnan(rec_h2_ou["win_rate"])) else float("nan")

    ats_band_wr = []
    for lo, hi in BANDS:
        m = (full["disagree_ats"].abs() >= lo) & (full["disagree_ats"].abs() < hi)
        ats_band_wr.append(record(full["result_ats"], m)["win_rate"])
    ou_band_wr = []
    for lo, hi in BANDS:
        m = (full["disagree_ou"].abs() >= lo) & (full["disagree_ou"].abs() < hi)
        ou_band_wr.append(record(full["result_ou"], m)["win_rate"])

    def slopes(vals):
        v = [x for x in vals if not np.isnan(x)]
        if len(v) < 3:
            return "insufficient bands", "insufficient bands"
        idx = np.arange(len(v))
        corr = float(np.corrcoef(idx, v)[0, 1])
        spread = max(v) - min(v)
        verdict = ("FLAT (no slope): spread across bands "
                   f"{spread:.4f} win-rate points, corr(band index, win rate)={corr:+.2f}, "
                   "no consistent rise from small to large disagreement")
        if corr > 0.6 and v[-1] > v[0] + 0.02:
            verdict = ("SLOPES UP: corr(band index, win rate)="
                       f"{corr:+.2f}, win rate rises from {v[0]:.4f} (smallest band) to "
                       f"{v[-1]:.4f} (largest band) -- a candidate real signal, not yet interpreted "
                       "as one given the gate-failing engine and small per-band n")
        return [round(x, 4) for x in v], verdict

    ats_lean_word = ("inside the noise band" if not np.isnan(ats_inside_noise) and abs(ats_wr - breakeven) < ats_inside_noise
                      else ("positive" if ats_wr > breakeven else "negative"))
    ou_lean_word = ("inside the noise band" if not np.isnan(ou_inside_noise) and abs(ou_wr - breakeven) < ou_inside_noise
                     else ("positive" if ou_wr > breakeven else "negative"))

    # ================= write doc ==================
    L: list[str] = []
    L.append("# Mean-based ATS / O-U lean, engine v1, 200 seeds (2026-09-11)")
    L.append("")
    L.append("**PROVISIONAL, mean-based, not the pre-registered ROI read; engine v1 fails 8 of 9 gates.**")
    L.append("")
    L.append("Generated by `scripts/grade_market_lean_mean_based.py`. Run graded: "
              f"`{RESULTS_A.relative_to(ROOT).as_posix()}` (engine v1, `docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md`), "
              f"season {SEASON}, {n_seeds_total} seeds. This is NOT the pre-registered ROI read: "
              "`docs/tests/engine_seed_count_2026-09-10.md` fixed the ROI/Brier/edge-bucket floor at "
              "~2,000 seeds because a per-game sim win PROBABILITY carries ~3.3-3.5pp SE at 200 seeds "
              "(that doc's G10 line), which swamps a probability-bucketed edge. This script reads the "
              "sim's per-game MEAN margin/total instead and computes a flat-stake lean keyed on the mean "
              "vs. the close line -- legitimate as a DIRECTIONAL number because the mean itself is a much "
              "lower-relative-noise quantity than a 200-seed win probability, not because it is precise "
              "per game (section 3 below measures exactly how imprecise per game). "
              "`scripts/grade_market_games_v2.py`'s own ROI/Brier refusal logic (n_seeds < 2000) is "
              "untouched by this script and was not modified. No probability-based edge bucket is "
              "computed anywhere in this doc.")
    L.append("")
    L.append("## Guards verified (reused from `grade_market_games_v2.py`, not rebuilt)")
    L.append("")
    L.append(f"- Truth: `load_truth({SEASON})` -- schedule from `games_universe`, scores from "
              f"`game_finals_v2` ONLY, unresolved cross-source games excluded. "
              f"{truth_counts['n_schedule_games']} schedule games, "
              f"{truth_counts['n_no_finals_row']} missing a finals row, "
              f"{truth_counts['n_unresolved_excluded']} unresolved-excluded -> "
              f"{n_truth} truth games (`finals_source_counts`: {truth_counts['finals_source_counts']}).")
    L.append(f"- Lines: `load_close_lines({SEASON}, lines_close_v1.parquet, provider='ESPN BET')`, "
              f"{n_dup_lines} duplicate (game_id) rows dropped before the join, same de-vigged, "
              "validated table `docs/tests/lines_cbbd_validation_2026-09-10.md` accepted; settlement "
              "is at flat -110 throughout (CBBD carries no per-side spread/total price -- confirmed in "
              "that doc's section 2/`market_scorecard_pipeline` -- so no real-close-price variant is "
              "reported; ESPN BET moneyline exists in the table but is not used here since this doc is "
              "ATS/O-U only, not moneyline).")
    L.append("- `created_at < tipoff` / `max_train_date < tipoff`: this is the SAME backtest run graded "
              "in `docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md` section 7 "
              "(`grade_market_games_v2.py`'s own honesty guard already asserted this on this exact run "
              "when that report was produced); not re-run here to avoid duplicating that check, per "
              "the instruction to reuse rather than rebuild the scorecard's guards.")
    L.append(f"- Coverage: {n_truth} truth games x sim ({games['game_id'].nunique()} games in the run) "
              f"x usable ESPN BET close line -> **{n_full} games graded** "
              f"({n_full / n_truth:.1%} of truth games).")
    L.append("- No probability-based edge bucket computed. No modification to "
              "`grade_market_games_v2.py`'s refusal logic. No results file overwritten "
              "(this script only reads `games.parquet`/`run_meta.json` under `results/`).")
    L.append("")

    L.append("## 1. Flat-stake records, mean vs. close line")
    L.append("")
    L.append("### 1a. ATS (pick = side sim MEAN margin favours vs. close spread), overall")
    L.append("")
    L.append("| | n | W-L-P | win rate | Wilson 95% CI | ROI @ -110 | |")
    L.append("|---|---:|---|---:|---|---:|---|")
    L.append(row_md("ATS overall", overall_ats))
    L.append("")
    L.append("### 1b. Over/Under (pick = side sim MEAN total favours vs. close total), overall")
    L.append("")
    L.append("| | n | W-L-P | win rate | Wilson 95% CI | ROI @ -110 | |")
    L.append("|---|---:|---|---:|---|---:|---|")
    L.append(row_md("O/U overall", overall_ou))
    L.append("")

    L.append("### 1c. ATS by venue/side (home = non-neutral, picked home side; "
              "away = non-neutral, picked away side; neutral = neutral-site game, either side)")
    L.append("")
    L += ats_venue
    L.append("")
    L.append("### 1d. O/U by pick side (over/under) and venue (neutral vs. non-neutral rolled into the "
              "same over/under buckets -- venue has no over/under analog; see raw output for a pure "
              "neutral-site O/U cut if needed)")
    L.append("")
    L += ou_venue_over
    L.append("")

    L.append("### 1e. ATS by month")
    L.append("")
    L += ats_months
    L.append("")
    L.append("### 1f. O/U by month")
    L.append("")
    L += ou_months
    L.append("")

    L.append("### 1g. ATS by |sim mean margin - close spread| band")
    L.append("")
    L += ats_bands
    L.append("")
    L.append("### 1h. O/U by |sim mean total - close total| band")
    L.append("")
    L += ou_bands
    L.append("")
    L.append(f"Bands under {UNDERPOWERED_N} bets are labelled UNDERPOWERED in the table (task threshold; "
              "note `grade_market_games_v2.py`'s own edge-bucket tables use a 100-bet floor for its "
              "gated, 2,000-seed-only ROI reads -- 300 is used here per this task's explicit instruction, "
              "a stricter bar for a lower-seed-count, non-pre-registered read).")
    L.append("")

    L.append("## 2. Honest baselines")
    L.append("")
    L.append("### 2a. Always-home / always-over (no model)")
    L.append("")
    L.append("| | n | W-L-P | win rate | Wilson 95% CI | ROI @ -110 | |")
    L.append("|---|---:|---|---:|---|---:|---|")
    L.append(row_md("Always home (ATS)", always_home))
    L.append(row_md("Always over (O/U)", always_over))
    L.append("")
    L.append(f"### 2b. Control engine (`{RESULTS_CONTROL.relative_to(ROOT).as_posix()}`, same F2 2025 slate, "
              f"{control_engine.games['seed'].nunique()} seeds, "
              f"{control_engine.games['game_id'].nunique()} games, "
              f"{len(full_ctrl)} joined to truth+lines), identical mean-based read")
    L.append("")
    L.append("| | n | W-L-P | win rate | Wilson 95% CI | ROI @ -110 | |")
    L.append("|---|---:|---|---:|---|---:|---|")
    L.append(row_md("Control ATS", ctrl_ats))
    L.append(row_md("Control O/U", ctrl_ou))
    L.append("")
    L.append("### 2c. Margin/total MAE vs. verified finals, engine v1 vs. control vs. the close line "
              "itself, same graded game set")
    L.append("")
    L.append("| | margin MAE | margin bias | total MAE | total bias |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| engine v1 (sim mean, n={n_full}) | {sim_margin_mae:.4f} | {sim_margin_bias:+.4f} | "
              f"{sim_total_mae:.4f} | {sim_total_bias:+.4f} |")
    L.append(f"| control engine (sim mean, n={len(full_ctrl)}) | {ctrl_margin_mae:.4f} | "
              f"{ctrl_margin_bias:+.4f} | {ctrl_total_mae:.4f} | {ctrl_total_bias:+.4f} |")
    L.append(f"| close line itself (n={n_full}) | {mkt_margin_mae:.4f} | {mkt_margin_bias:+.4f} | "
              f"{mkt_total_mae:.4f} | {mkt_total_bias:+.4f} |")
    L.append("")
    L.append("Cross-check: engine v1's margin/total MAE above should match "
              "`docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md` section 7 "
              "(margin MAE 9.2483, total MAE 13.5520 there, n=5,383) up to the small difference from "
              "this script re-deriving `full` independently rather than reading that report's numbers "
              "directly; `docs/tests/lines_cbbd_validation_2026-09-10.md` section 3 puts the close "
              "spread's own margin MAE at 8.74-8.81 pts on the wider 2023-2025 pool, consistent with "
              "the close-line row above on this narrower 2025 graded set.")
    L.append("")

    L.append("## 3. Seed-error check (split-half, 100 vs. 100 seeds)")
    L.append("")
    L.append(f"Run A's 200 seeds (0-199) split into seeds 0-99 and seeds 100-199, each half's per-game "
              f"mean margin/total recomputed independently, joined to the same truth+lines set "
              f"({len(common_ids)} games common to both halves and the full set), and the mean-based "
              "pick recomputed from each half alone.")
    L.append("")
    L.append("| | picks that flip vs. the other half | win rate, half 1 (seeds 0-99) | "
              "win rate, half 2 (seeds 100-199) | |win rate diff| |")
    L.append("|---|---:|---:|---:|---:|")
    wr1a = "n/a" if np.isnan(rec_h1_ats["win_rate"]) else f"{rec_h1_ats['win_rate']:.4f}"
    wr2a = "n/a" if np.isnan(rec_h2_ats["win_rate"]) else f"{rec_h2_ats['win_rate']:.4f}"
    da = "n/a" if np.isnan(ats_inside_noise) else f"{ats_inside_noise:.4f}"
    wr1o = "n/a" if np.isnan(rec_h1_ou["win_rate"]) else f"{rec_h1_ou['win_rate']:.4f}"
    wr2o = "n/a" if np.isnan(rec_h2_ou["win_rate"]) else f"{rec_h2_ou['win_rate']:.4f}"
    do = "n/a" if np.isnan(ou_inside_noise) else f"{ou_inside_noise:.4f}"
    L.append(f"| ATS | {flips_ats} / {len(common_ids)} ({flips_ats/len(common_ids):.1%}) | {wr1a} | {wr2a} | {da} |")
    L.append(f"| O/U | {flips_ou} / {len(common_ids)} ({flips_ou/len(common_ids):.1%}) | {wr1o} | {wr2o} | {do} |")
    L.append("")
    L.append("This is the honest noise band for this read: at 100 seeds per half the per-game mean "
              "margin/total SE roughly doubles relative to the 200-seed full run (seed study's "
              "c/sqrt(k) law, `grade_market_games_v2.SEED_STUDY_SE_MEAN`: ~3.03 pt margin / ~1.19 pt "
              "total at 100 seeds vs. ~2.14 pt margin / ~0.84 pt total at 200), so a meaningful share "
              "of picks close to the line are expected to flip between halves; the win-rate difference "
              "above quantifies how much that moves the overall read.")
    L.append("")

    L.append("## 4. Plain answer")
    L.append("")
    L.append(f"- **ATS**: overall win rate {ats_wr:.4f} vs. breakeven {breakeven:.4f} at -110; "
              f"split-half win-rate gap {da}. Lean is **{ats_lean_word}**.")
    L.append(f"- **O/U**: overall win rate {ou_wr:.4f} vs. breakeven {breakeven:.4f} at -110; "
              f"split-half win-rate gap {do}. Lean is **{ou_lean_word}**.")
    ats_vals, ats_verdict = slopes(ats_band_wr)
    ou_vals, ou_verdict = slopes(ou_band_wr)
    L.append(f"- ATS band win rates by |sim-line| band {[lo for lo, _ in BANDS]}: {ats_vals} -- {ats_verdict}.")
    L.append(f"- O/U band win rates by |sim-line| band {[lo for lo, _ in BANDS]}: {ou_vals} -- {ou_verdict}.")
    L.append("- Per CLAUDE.md's no-hand-tuning and bake-off rules: this is a read of engine v1's raw "
              "output only, no post-hoc adjustment of any kind; engine v1 currently fails 8 of 9 gates "
              "(`docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md` section 5), so any lean here "
              "describes THIS provisional, gate-failing engine, not a validated model, and should not "
              "be used to place a real bet.")
    L.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"[grade_market_lean_mean_based] wrote {OUT_MD}")
    print(f"[grade_market_lean_mean_based] coverage: {n_full} / {n_truth} truth games")
    print(f"[grade_market_lean_mean_based] ATS overall: {overall_ats}")
    print(f"[grade_market_lean_mean_based] O/U overall: {overall_ou}")
    print(f"[grade_market_lean_mean_based] ATS split-half flips: {flips_ats}/{len(common_ids)}, "
          f"win rate diff {da}")
    print(f"[grade_market_lean_mean_based] O/U split-half flips: {flips_ou}/{len(common_ids)}, "
          f"win rate diff {do}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
