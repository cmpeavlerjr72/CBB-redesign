"""ops_pair_bootstrap_v1.py -- AWS operator 2026-10-01. Decision 12 floors for paired full-size closed-loop reads. Decides nothing.

    CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 \
        --ref v3full_S0_s200_o0 --floors v3full_S0f1_s200_o1000,... --arms v3full_COMB_s200_o0,... \
        --out-json <path> --out-md <path> [--n-boot 2000]

Per line, on the games common to ref and arm (verified truth via cbb_sim.eval.gates.build_grading_frame):
  move        = arm - ref (full sample)
  draw SD     = SD (n-1) of the line over [ref, floors...] (seed-offset reruns of the reference, unpaired)
  boot CI     = 95% interval of (arm - ref) under a paired game bootstrap (games resampled with replacement; each
                game keeps all its paired seeds in both arms; 2,000 draws, numpy seed 20261001)
  floor       = max(2 x draw SD, half-width of the boot CI)   (Decision 12: "the floor is the max")
  flag        = BEYOND if |move| > floor, else inside
Line definitions follow cbb_sim.eval.gates: G5 ratio = mean_g(sim SD_g) / SD_g(actual_g - sim mean_g), reported with both
components (within = numerator, resid = denominator) and corr(actual, sim mean) as the accuracy term; home/away corr over
all (game, seed) rows; G9 slope = OLS slope of actual margin on sim margin mean; bias = mean(sim mean - actual).
G4 pooled lines here are ratios of sums over all team-games (TOV% = TOV/(FGA-OREB+TOV+0.44 FTA), OREB% = OREB/(OREB+DREB),
eFG% = (FGM+0.5 FGM3)/FGA, FTA/FGA); TOV is ALSO reported as a count per team-game (guardrails 2026-09-30 (b)).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402

COLS = ["game_id", "seed", "home_pts", "away_pts", "possessions", "n_periods", "home_fga3", "away_fga3", "home_fga2_rim",
        "away_fga2_rim", "home_fga2_jump", "away_fga2_jump", "home_fta", "away_fta", "home_tov", "away_tov", "home_oreb",
        "away_oreb", "home_dreb", "away_dreb", "home_fgm2_rim", "away_fgm2_rim", "home_fgm2_jump", "away_fgm2_jump",
        "home_fgm3", "away_fgm3"]

LINES = [  # key, gate, label
    ("poss_mean", "G1", "possessions/game mean (sim)"),
    ("poss_sd_within", "G1", "possessions within-game SD (sim, mean over games)"),
    ("g5_margin_ratio", "G5", "margin SD ratio"),
    ("g5_margin_within", "G5", "  margin component: mean within-game sim SD"),
    ("g5_margin_resid", "G5", "  margin component: SD(actual - sim mean)"),
    ("g5_total_ratio", "G5", "total SD ratio"),
    ("g5_total_within", "G5", "  total component: mean within-game sim SD"),
    ("g5_total_resid", "G5", "  total component: SD(actual - sim mean)"),
    ("corr_total", "G5", "  accuracy: corr(actual total, sim total mean)"),
    ("corr_margin", "G5", "  accuracy: corr(actual margin, sim margin mean)"),
    ("home_away_corr", "G5", "home/away score correlation (sim)"),
    ("ot_rate", "G7", "OT rate (sim)"),
    ("margin_bias", "G9", "margin bias (sim - actual)"),
    ("total_bias", "G9", "total bias (sim - actual)"),
    ("slope", "G9", "calibration slope"),
    ("tov_pct", "G4", "TOV% pooled (ratio of sums)"),
    ("tov_per_team_game", "G4", "TOV count per team-game"),
    ("oreb_pct", "G4", "OREB% pooled"),
    ("efg_pct", "G4", "eFG% pooled"),
    ("fta_per_fga", "G4", "FTA/FGA pooled"),
]


def per_game(results_dir: Path, tag: str) -> pd.DataFrame:
    g = pd.read_parquet(results_dir / tag / "games.parquet", columns=COLS)
    summ, _ = G.build_grading_frame(g, 2025)
    for s in ("home", "away"):
        g[f"{s}_fga"] = g[f"{s}_fga3"] + g[f"{s}_fga2_rim"] + g[f"{s}_fga2_jump"]
        g[f"{s}_fgm"] = g[f"{s}_fgm3"] + g[f"{s}_fgm2_rim"] + g[f"{s}_fgm2_jump"]
    hp, ap = g["home_pts"].astype(float), g["away_pts"].astype(float)
    a = pd.DataFrame({
        "game_id": g["game_id"], "n": 1.0, "sh": hp, "sa": ap, "shh": hp * hp, "saa": ap * ap, "sha": hp * ap,
        "ot": (g["n_periods"] > 2).astype(float),
        "tov": g["home_tov"] + g["away_tov"],
        "pest": g["home_fga"] + g["away_fga"] - g["home_oreb"] - g["away_oreb"] + g["home_tov"] + g["away_tov"]
        + 0.44 * (g["home_fta"] + g["away_fta"]),
        "oreb": g["home_oreb"] + g["away_oreb"], "orebd": g["home_oreb"] + g["away_oreb"] + g["home_dreb"] + g["away_dreb"],
        "fga": g["home_fga"] + g["away_fga"], "fgm": g["home_fgm"] + g["away_fgm"], "fgm3": g["home_fgm3"] + g["away_fgm3"],
        "fta": g["home_fta"] + g["away_fta"]}).groupby("game_id").sum()
    s = summ.set_index("game_id")[["margin", "total", "sim_margin_mean", "sim_margin_sd", "sim_total_mean", "sim_total_sd",
                                   "sim_poss_mean", "sim_poss_sd"]]
    return s.join(a, how="inner").sort_index().astype(float)


def _wmean(W, x):
    return (W @ x) / W.sum(axis=1)


def _wvar(W, x):
    n = W.sum(axis=1)
    m = (W @ x) / n
    return ((W @ (x * x)) - n * m * m) / (n - 1)


def _wcov(W, x, y):
    n = W.sum(axis=1)
    return ((W @ (x * y)) - n * ((W @ x) / n) * ((W @ y) / n)) / (n - 1)


def stats(d: pd.DataFrame, W: np.ndarray) -> dict[str, np.ndarray]:
    """W: (B, n_games) nonnegative game weights (a row of ones = the full sample)."""
    c = {k: d[k].to_numpy() for k in d.columns}
    rm, rt = c["margin"] - c["sim_margin_mean"], c["total"] - c["sim_total_mean"]
    o = {}
    o["poss_mean"] = _wmean(W, c["sim_poss_mean"])
    o["poss_sd_within"] = _wmean(W, c["sim_poss_sd"])
    o["g5_margin_within"] = _wmean(W, c["sim_margin_sd"])
    o["g5_margin_resid"] = np.sqrt(_wvar(W, rm))
    o["g5_margin_ratio"] = o["g5_margin_within"] / o["g5_margin_resid"]
    o["g5_total_within"] = _wmean(W, c["sim_total_sd"])
    o["g5_total_resid"] = np.sqrt(_wvar(W, rt))
    o["g5_total_ratio"] = o["g5_total_within"] / o["g5_total_resid"]
    o["corr_total"] = _wcov(W, c["total"], c["sim_total_mean"]) / np.sqrt(_wvar(W, c["total"]) * _wvar(W, c["sim_total_mean"]))
    o["corr_margin"] = _wcov(W, c["margin"], c["sim_margin_mean"]) / np.sqrt(_wvar(W, c["margin"]) * _wvar(W, c["sim_margin_mean"]))
    N = W @ c["n"]
    mh, ma = (W @ c["sh"]) / N, (W @ c["sa"]) / N
    vh, va = (W @ c["shh"]) / N - mh * mh, (W @ c["saa"]) / N - ma * ma
    o["home_away_corr"] = ((W @ c["sha"]) / N - mh * ma) / np.sqrt(vh * va)
    o["ot_rate"] = (W @ c["ot"]) / N
    o["margin_bias"] = _wmean(W, -rm)
    o["total_bias"] = _wmean(W, -rt)
    o["slope"] = _wcov(W, c["sim_margin_mean"], c["margin"]) / _wvar(W, c["sim_margin_mean"])
    o["tov_pct"] = (W @ c["tov"]) / (W @ c["pest"])
    o["tov_per_team_game"] = (W @ c["tov"]) / (2 * N)
    o["oreb_pct"] = (W @ c["oreb"]) / (W @ c["orebd"])
    o["efg_pct"] = (W @ (c["fgm"] + 0.5 * c["fgm3"])) / (W @ c["fga"])
    o["fta_per_fga"] = (W @ c["fta"]) / (W @ c["fga"])
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/engine_v0")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--floors", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    R = Path(a.results_dir)
    floors = [t for t in a.floors.split(",") if t]
    arms = [t for t in a.arms.split(",") if t]
    labels = a.labels.split(",") if a.labels else arms
    ref = per_game(R, a.ref)
    one = lambda d: {k: float(v[0]) for k, v in stats(d, np.ones((1, len(d)))).items()}
    s_ref = one(ref)
    draws = [s_ref] + [one(per_game(R, f).reindex(ref.index).dropna()) for f in floors]
    draw_sd = {k: float(np.std([dd[k] for dd in draws], ddof=1)) if len(draws) > 1 else float("nan") for k in s_ref}
    out = {"ref": a.ref, "floors": floors, "n_boot": a.n_boot, "ref_values": s_ref,
           "floor_draw_values": draws, "draw_sd": draw_sd, "arms": {}}
    md = [f"Paired game bootstrap (Decision 12), ref `{a.ref}`, floor draws {', '.join('`'+f+'`' for f in floors)}; "
          f"{a.n_boot} game resamples; floor = max(2 x draw SD, bootstrap 95% half-width); BEYOND = |move| > floor.\n"]
    for arm, lab in zip(arms, labels):
        d = per_game(R, arm)
        common = ref.index.intersection(d.index)
        r_, d_ = ref.loc[common], d.loc[common]
        rng = np.random.default_rng(20261001)
        idx = rng.integers(0, len(common), size=(a.n_boot, len(common)))
        W = np.zeros((a.n_boot, len(common)), dtype=np.float64)
        for b in range(a.n_boot):
            W[b] = np.bincount(idx[b], minlength=len(common))
        sa_, sr_ = stats(d_, W), stats(r_, W)
        fa, fr = one(d_), one(r_)
        res = {"n_games": int(len(common)), "lines": {}}
        md += [f"\n### {lab} (`{arm}`) vs ref, {len(common)} common games\n",
               "| gate | line | ref | arm | move | draw SD | 2 x draw SD | boot 95% CI of move | floor | move / floor | flag |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, gate, label in LINES:
            diff = sa_[k] - sr_[k]
            lo, hi = float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))
            mv = fa[k] - fr[k]
            hw = (hi - lo) / 2
            fl = max(2 * draw_sd[k], hw)
            flag = "BEYOND" if abs(mv) > fl else "inside"
            res["lines"][k] = {"ref": fr[k], "arm": fa[k], "move": mv, "draw_sd": draw_sd[k], "boot_ci95": [lo, hi],
                               "boot_halfwidth": hw, "floor": fl, "flag": flag}
            md.append(f"| {gate} | {label} | {fr[k]:.4f} | {fa[k]:.4f} | {mv:+.4f} | {draw_sd[k]:.4f} | {2*draw_sd[k]:.4f} | "
                      f"[{lo:+.4f}, {hi:+.4f}] | {fl:.4f} | {mv/fl if fl else float('nan'):+.2f} | {flag} |")
        out["arms"][lab] = res
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=1, default=float))
    Path(a.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
