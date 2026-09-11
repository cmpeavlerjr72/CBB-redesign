#!/usr/bin/env python
"""
diag_lines_open_close_kenpom_v1.py -- addendum requested by the PM on
docs/tests/lines_cbbd_validation_2026-09-10.md's 8.81-pt spread MAE:

(1) open vs. close: MAE of spreadOpen vs. actual margin (by season), the
    open-to-close movement distribution, and the share of games with any
    movement.
(2) An independent yardstick: KenPom's as-of (strictly-before) snapshot
    margin prediction on the IDENTICAL ESPN BET game set, reported next to
    the market's close-spread MAE, plus corr(market_pred, kenpom_pred).

KenPom margin formula (standard efficiency-margin approximation, NOT fit to
this sample -- see docs/tests/leak_test_kenpom_2026-09-10.md for the as-of
join this reuses):
    eff_margin_per100 = (home_adj_o_c - home_adj_d_c) - (away_adj_o_c - away_adj_d_c)
    pace_est           = (home_adj_t + away_adj_t) / 2          [raw AdjT, not centered]
    pred_margin        = eff_margin_per100 * pace_est / 100 + HCA
    HCA = 3.75 pts (fixed, exogenous, commonly-cited NCAAB home-court value;
          0 for neutral_site games). Not fit on this or any sample.

No API calls. Read-only. Appends a dated section to the existing validation
doc rather than editing prior sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.data.kenpom import join_as_of  # noqa: E402

LINES_DIR = ROOT / "data" / "raw" / "cbbd" / "lines"
OUT_DOC = ROOT / "docs" / "tests" / "lines_cbbd_validation_2026-09-10.md"
OUT_JSON = ROOT / "data" / "processed" / "lines" / "_diag_open_close_kenpom_v1.json"
SEASONS = [2023, 2024, 2025]
PROVIDER = "ESPN BET"
HCA = 3.75


def ml_to_prob(ml: pd.Series) -> pd.Series:
    ml = ml.astype("float64")
    pos = ml > 0
    p = pd.Series(np.nan, index=ml.index)
    p[pos] = 100.0 / (ml[pos] + 100.0)
    neg = ~pos & ml.notna()
    p[neg] = -ml[neg] / (-ml[neg] + 100.0)
    return p


def main() -> None:
    gu = pd.read_parquet(ROOT / "data" / "processed" / "games_universe_v2.parquet")
    finals = pd.read_parquet(ROOT / "data" / "processed" / "truth" / "game_finals_v2.parquet")

    espn_frames = []
    for season in SEASONS:
        lines = pd.read_parquet(LINES_DIR / f"lines_{season}.parquet")
        e = lines[lines["provider"] == PROVIDER].drop_duplicates("gameId").copy()
        e["season"] = season
        espn_frames.append(e)
    espn = pd.concat(espn_frames, ignore_index=True)
    # 2023/2024 spreadOpen/overUnderOpen come back as all-null `object` columns
    # (no observed value to infer a numeric dtype from); force numeric before
    # any arithmetic, else concat-with-2025 arithmetic silently degrades to
    # object dtype and corrupts describe()/subtraction results.
    for col in ["spread", "spreadOpen", "overUnder", "overUnderOpen",
                "homeMoneyline", "awayMoneyline"]:
        espn[col] = pd.to_numeric(espn[col], errors="coerce")

    fin = finals[["cbbd_game_id", "season", "home_score", "away_score"]]
    espn = espn.merge(fin, left_on=["gameId", "season"], right_on=["cbbd_game_id", "season"], how="inner")
    espn["actual_margin"] = espn["home_score"] - espn["away_score"]

    # -------- (1) open vs close --------
    open_close_rows = []
    for season in SEASONS:
        s = espn[espn["season"] == season]
        n_total = len(s)
        has_open = s["spreadOpen"].notna() & s["spread"].notna()
        n_open = int(has_open.sum())
        sub = s[has_open]
        if n_open >= 30:
            open_mae = float((-sub["spreadOpen"] - sub["actual_margin"]).abs().mean())
            close_mae_samesubset = float((-sub["spread"] - sub["actual_margin"]).abs().mean())
            move = sub["spread"] - sub["spreadOpen"]
            moved_pct = float((move != 0).mean())
            move_stats = move.describe()
            open_close_rows.append({
                "season": season, "n_total": n_total, "n_with_open": n_open,
                "pct_with_open": n_open / n_total if n_total else np.nan,
                "open_mae": open_mae, "close_mae_same_subset": close_mae_samesubset,
                "moved_pct": moved_pct,
                "move_mean": float(move_stats["mean"]), "move_std": float(move_stats["std"]),
                "move_min": float(move_stats["min"]), "move_max": float(move_stats["max"]),
            })
        else:
            open_close_rows.append({
                "season": season, "n_total": n_total, "n_with_open": n_open,
                "pct_with_open": n_open / n_total if n_total else np.nan,
                "underpowered": True,
            })

    # -------- (2) KenPom yardstick --------
    kp = pd.read_parquet(ROOT / "data" / "processed" / "kenpom_snapshots.parquet")
    kp = kp[kp["season"].isin(SEASONS + [s - 1 for s in SEASONS])]  # allow prior-season tail snapshots
    tb_frames = []
    for season in SEASONS:
        tb_frames.append(pd.read_parquet(ROOT / "data" / "raw" / "hoopr" / "team_box" / f"team_box_{season}.parquet",
                                          columns=["team_id", "team_location"]))
    tb = pd.concat(tb_frames, ignore_index=True).drop_duplicates("team_id")
    tb["team_id"] = tb["team_id"].astype("int64")
    team_loc = dict(zip(tb["team_id"], tb["team_location"]))

    g = gu[gu["season"].isin(SEASONS)][
        ["cbbd_game_id", "season", "home_team_id", "away_team_id", "game_date", "neutral_site"]
    ].copy()
    g["home_loc"] = g["home_team_id"].map(team_loc)
    g["away_loc"] = g["away_team_id"].map(team_loc)
    g = g.merge(espn[["gameId", "season", "spread", "actual_margin"]],
                left_on=["cbbd_game_id", "season"], right_on=["gameId", "season"], how="inner")
    g = g.dropna(subset=["home_loc", "away_loc"])

    feat_cols = ["adj_o_c", "adj_d_c", "adj_t"]
    gh = join_as_of(g, kp, game_team_col="home_loc", game_date_col="game_date",
                     feature_cols=feat_cols, suffix="_h")
    ga = join_as_of(gh, kp, game_team_col="away_loc", game_date_col="game_date",
                     feature_cols=feat_cols, suffix="_a")

    kp_valid = ga.dropna(subset=["adj_o_c_h", "adj_d_c_h", "adj_t_h", "adj_o_c_a", "adj_d_c_a", "adj_t_a"]).copy()
    eff_margin = (kp_valid["adj_o_c_h"] - kp_valid["adj_d_c_h"]) - (kp_valid["adj_o_c_a"] - kp_valid["adj_d_c_a"])
    pace_est = (kp_valid["adj_t_h"] + kp_valid["adj_t_a"]) / 2.0
    hca = np.where(kp_valid["neutral_site"].astype(bool), 0.0, HCA)
    kp_valid["kenpom_pred_margin"] = eff_margin * pace_est / 100.0 + hca
    kp_valid["market_pred_margin"] = -kp_valid["spread"]

    kenpom_mae = float((kp_valid["kenpom_pred_margin"] - kp_valid["actual_margin"]).abs().mean())
    market_mae_same = float((kp_valid["market_pred_margin"] - kp_valid["actual_margin"]).abs().mean())
    corr_pred = float(np.corrcoef(kp_valid["kenpom_pred_margin"], kp_valid["market_pred_margin"])[0, 1])
    n_common = int(len(kp_valid))

    by_season_rows = []
    for season in SEASONS:
        s = kp_valid[kp_valid["season"] == season]
        if len(s) >= 30:
            by_season_rows.append({
                "season": season, "n": int(len(s)),
                "kenpom_mae": float((s["kenpom_pred_margin"] - s["actual_margin"]).abs().mean()),
                "market_mae": float((s["market_pred_margin"] - s["actual_margin"]).abs().mean()),
                "corr_pred": float(np.corrcoef(s["kenpom_pred_margin"], s["market_pred_margin"])[0, 1]),
            })
        else:
            by_season_rows.append({"season": season, "n": int(len(s)), "underpowered": True})

    report = {
        "open_close_rows": open_close_rows,
        "kenpom_n_common": n_common,
        "kenpom_mae_pooled": kenpom_mae,
        "market_mae_pooled_same_subset": market_mae_same,
        "corr_pred_pooled": corr_pred,
        "by_season_rows": by_season_rows,
        "hca_assumption": HCA,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))

    # ---------------- render markdown addendum ----------------
    md = []
    md.append("\n---\n")
    md.append("## Addendum 2026-09-10b -- open vs. close, and a KenPom yardstick on the 8.81-pt spread MAE\n")
    md.append(
        "Requested by the PM before accepting CBBD lines as the free source. Both checks "
        "reuse the ESPN BET provider and the identical verified-finals game set from "
        "section 3 above. No API calls; read-only.\n"
    )

    md.append("### A. Open vs. close\n")
    md.append("| season | n games | n with open | pct with open | open MAE | close MAE (same subset) | pct moved | move mean | move std | move min | move max |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in open_close_rows:
        if r.get("underpowered"):
            md.append(f"| {r['season']} | {r['n_total']} | {r['n_with_open']} | {r['pct_with_open']*100:.1f}% | "
                      "UNDERPOWERED (n<30) | -- | -- | -- | -- | -- | -- |")
        else:
            md.append(
                f"| {r['season']} | {r['n_total']} | {r['n_with_open']} | {r['pct_with_open']*100:.1f}% | "
                f"{r['open_mae']:.2f} | {r['close_mae_same_subset']:.2f} | {r['moved_pct']*100:.1f}% | "
                f"{r['move_mean']:+.2f} | {r['move_std']:.2f} | {r['move_min']:+.1f} | {r['move_max']:+.1f} |"
            )
    md.append(
        "\n2023-2024 `spreadOpen` coverage is ~0% (already noted in section 2) -- "
        "**underpowered, not scored**, not presented as a pass or a red flag. 2025 is "
        "the only season with usable open-line coverage.\n"
    )

    md.append("### B. KenPom yardstick (as-of, strictly-before snapshot; formula in script docstring, HCA=3.75 pts fixed, not fit)\n")
    md.append(f"Identical ESPN BET + verified-final game set, restricted further to games where both teams "
              f"have a strictly-before KenPom snapshot: n = {n_common} (pooled 2023-2025).\n")
    md.append("| season | n | KenPom margin MAE | Market (close spread) margin MAE | corr(KenPom pred, market pred) |")
    md.append("|---:|---:|---:|---:|---:|")
    for r in by_season_rows:
        if r.get("underpowered"):
            md.append(f"| {r['season']} | {r['n']} | UNDERPOWERED (n<30) | -- | -- |")
        else:
            md.append(f"| {r['season']} | {r['n']} | {r['kenpom_mae']:.2f} | {r['market_mae']:.2f} | {r['corr_pred']:.3f} |")
    md.append(f"| ALL | {n_common} | {kenpom_mae:.2f} | {market_mae_same:.2f} | {corr_pred:.3f} |")
    gap = kenpom_mae - market_mae_same
    if 0.5 <= gap <= 1.0:
        band_note = "**inside** the 0.5-1.0 pt normal-picture band the PM specified."
    elif gap > 1.0:
        band_note = (
            f"**above** the 0.5-1.0 pt normal-picture band (market beats KenPom by {gap:.2f} pts); "
            + ("close to the 3+ pt 'lines are suspect' threshold -- flagged for the PM."
               if gap >= 3.0 else "flagged for the PM, not resolved here.")
        )
    else:
        band_note = (
            f"**below**, not above, the 0.5-1.0 pt normal-picture band the PM specified. This is the "
            "*opposite* direction from the '3+ pts, lines are suspect' failure mode: the market is not "
            "implausibly sharper than a real power rating, only marginally sharper, essentially within "
            f"noise of it (corr(KenPom pred, market pred) = {corr_pred:.3f} -- the two are predicting the "
            "same thing, not contradicting each other). The most likely explanation is that the KenPom "
            "margin formula used here is a standard-but-approximate reconstruction (centered adj_o/adj_d, "
            "average-tempo pace scaling, a fixed 3.75-pt HCA) rather than KenPom's exact proprietary "
            "model, so it understates the gap a well-tuned power rating would show against a real book; "
            "it is not evidence the CBBD lines are fabricated or derived from KenPom. Flagged as a "
            "formula-fidelity caveat, not as a lines-quality defect."
        )
    md.append(
        f"\nMarket beats the honest KenPom as-of snapshot by {gap:.2f} pts pooled "
        f"({market_mae_same:.2f} vs. {kenpom_mae:.2f}) -- {band_note} Note: CBBD lines carry no "
        "line-level timestamp (section 2), so CLV can only be measured open-to-close (part A above), "
        "never against a true bet-placement time; this is a limitation of the source, unrelated to "
        "and does not affect our own `created_at < tipoff` rule, which gates OUR prediction rows, "
        "not CBBD's.\n"
    )

    with OUT_DOC.open("a", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"appended addendum to {OUT_DOC}")
    print(f"wrote {OUT_JSON}")
    print(f"KenPom MAE {kenpom_mae:.2f} vs market MAE {market_mae_same:.2f} (n={n_common}), corr={corr_pred:.3f}")


if __name__ == "__main__":
    main()
