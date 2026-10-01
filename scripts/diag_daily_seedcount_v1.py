#!/usr/bin/env python
"""
diag_daily_seedcount_v1.py -- seed-count study for the daily chain (lane F, 2026-10-01). Reads an existing fold-2 run; no sim is run.

Question: how many seeds per game before a published win probability, spread-cover probability and total-over probability are stable
enough to read an edge. Input: the adopted served stack at full size (5,710 games x 200 seeds, `results/engine_v0/v3full_COMB9GCTKD_s200_o0`)
and the close spread / total (`reference.load_lines`, the harness's provider preference). Nothing is fitted or adjusted; every number is a
direct statistic of the per-seed rows.

For each seed count k, the 200 seeds are cut into floor(200/k) disjoint blocks (independent runs of k seeds, RNG streams are keyed on
seed, so blocks are independent draws). Per game and quantity, p_hat(block). Reported per k and quantity:
  se_rms        RMS over games of the binomial standard error sqrt(p(1-p)/k) with p = the game's 200-seed estimate (analytic)
  move_sd       empirical: RMS over games of the SD of p_hat across blocks (run-to-run movement of one run); = se_rms if streams are binomial
  d_p95         95th percentile over games of |p_hat(block a) - p_hat(block b)|, a, b consecutive blocks (movement between two runs)
  share_lt_1pp / share_lt_05pp   share of game-quantities whose |difference between two runs| is < 1 pp / < 0.5 pp
  lean_flip     share of games whose lean (p_hat > 0.5) differs between two runs   (cover / over / win)
  edge_flip     share of games whose "edge >= 3 pp from 0.5" status differs between two runs (the unit a bucketed ROI table reads)
and, in points, the standard error of the sim MEAN margin and MEAN total (what the ATS / O-U disagreement buckets read): sd / sqrt(k).
Standard-error extrapolation beyond k = 100 uses the binomial 1/sqrt(k) law checked against the empirical columns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cbb_sim.eval import reference as ref  # noqa: E402

RUN = REPO / "results/engine_v0/v3full_COMB9GCTKD_s200_o0"
KS = [5, 10, 20, 25, 40, 50, 100]


def load():
    g = pd.read_parquet(RUN / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts"])
    u = pd.read_parquet(REPO / "data/processed/games_universe.parquet", columns=["game_id", "cbbd_game_id"])
    ln = ref.load_lines(2025)
    ln = ln.merge(u.assign(cbbd_game_id=u["cbbd_game_id"].astype("Int64")), on="cbbd_game_id", how="inner")
    g = g.merge(ln[["game_id", "spread", "overUnder"]], on="game_id", how="inner")
    g["margin"] = g["home_pts"].astype("int32") - g["away_pts"].astype("int32")
    g["total"] = g["home_pts"].astype("int32") + g["away_pts"].astype("int32")
    g["win"] = (g["margin"] > 0).astype("float32")
    # home covers when margin + spread > 0 (CBBD spread is home-perspective, negative = home favoured); a push counts half
    cv = g["margin"] + g["spread"]
    g["cover"] = np.where(cv > 0, 1.0, np.where(cv == 0, 0.5, 0.0)).astype("float32")
    ov = g["total"] - g["overUnder"]
    g["over"] = np.where(ov > 0, 1.0, np.where(ov == 0, 0.5, 0.0)).astype("float32")
    g.loc[g["overUnder"].isna(), "over"] = np.nan
    return g.sort_values(["game_id", "seed"]).reset_index(drop=True)


def arr(g, col):
    n = g["game_id"].nunique()
    return g[col].to_numpy(dtype="float64").reshape(n, -1)       # (games, 200), seeds sorted within game


def main():
    g = load()
    n_games = g["game_id"].nunique()
    assert (g.groupby("game_id").size() == 200).all()
    print(f"games with a close line: {n_games}; with a total line: {int(g.dropna(subset=['over'])['game_id'].nunique())}")
    out = {"n_games_spread": int(n_games), "run": str(RUN.relative_to(REPO)), "rows": []}
    for q in ("win", "cover", "over"):
        gg = g.dropna(subset=[q]) if q == "over" else g
        x = arr(gg, q)
        p200 = x.mean(1)
        for k in KS:
            nb = 200 // k
            blocks = x[:, : nb * k].reshape(x.shape[0], nb, k).mean(2)            # (games, blocks)
            se_rms = float(np.sqrt(np.mean(p200 * (1 - p200)) / k))
            move_sd = float(np.sqrt(np.mean(blocks.var(1, ddof=1))))
            d = np.abs(blocks[:, 0::2][:, : nb // 2] - blocks[:, 1::2][:, : nb // 2]).ravel() if nb >= 2 else np.array([np.nan])
            a, b = blocks[:, 0::2][:, : nb // 2], blocks[:, 1::2][:, : nb // 2]
            out["rows"].append({"q": q, "k": k, "n_blocks": nb, "n_games": int(x.shape[0]),
                                "se_rms_pp": 100 * se_rms, "move_sd_pp": 100 * move_sd,
                                "d_p95_pp": float(100 * np.nanpercentile(d, 95)),
                                "share_lt_1pp": float(np.mean(d < 0.01)), "share_lt_05pp": float(np.mean(d < 0.005)),
                                "lean_flip": float(np.mean((a > 0.5) != (b > 0.5))),
                                "edge_flip": float(np.mean((np.abs(a - 0.5) >= 0.03) != (np.abs(b - 0.5) >= 0.03)))})
        # analytic requirement from the 200-seed per-game p (binomial law): k such that RMS SE < target
        v = float(np.mean(p200 * (1 - p200)))
        out[f"{q}_mean_p_var"] = v
        out[f"{q}_k_for_rms_se_1pp"] = int(np.ceil(v / 0.01 ** 2))
        out[f"{q}_k_for_rms_se_05pp"] = int(np.ceil(v / 0.005 ** 2))
        # a "worst game" requirement (p = 0.5)
        out[f"{q}_k_worst_1pp"], out[f"{q}_k_worst_05pp"] = 2500, 10000
        # games with p in the 0.35-0.65 band (where a read of an edge happens)
        band = (p200 > 0.35) & (p200 < 0.65)
        out[f"{q}_share_band_035_065"] = float(band.mean())
    # points-scale: SE of the sim mean margin and mean total
    m = arr(g, "margin")
    t = arr(g, "total")
    sdm, sdt = float(np.sqrt(np.mean(m.var(1, ddof=1)))), float(np.sqrt(np.mean(t.var(1, ddof=1))))
    out["margin_sd_within_game"], out["total_sd_within_game"] = sdm, sdt
    out["points_se"] = {int(k): {"margin_se": sdm / np.sqrt(k), "total_se": sdt / np.sqrt(k)} for k in (25, 50, 100, 200, 400, 500, 1000, 2000)}
    out["k_margin_se_05pt"], out["k_margin_se_025pt"] = int(np.ceil((sdm / 0.5) ** 2)), int(np.ceil((sdm / 0.25) ** 2))
    out["k_total_se_05pt"], out["k_total_se_025pt"] = int(np.ceil((sdt / 0.5) ** 2)), int(np.ceil((sdt / 0.25) ** 2))
    # empirical check on the means (200 seeds -> blocks)
    for k in (25, 50, 100):
        nb = 200 // k
        mb = m[:, : nb * k].reshape(m.shape[0], nb, k).mean(2)
        out[f"margin_mean_move_sd_k{k}"] = float(np.sqrt(np.mean(mb.var(1, ddof=1))))
    df = pd.DataFrame(out["rows"])
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print({k: v for k, v in out.items() if k not in ("rows",)})
    (REPO / "results/seedcount_study.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    df.to_csv(REPO / "results/seedcount_study.csv", index=False)


if __name__ == "__main__":
    main()
