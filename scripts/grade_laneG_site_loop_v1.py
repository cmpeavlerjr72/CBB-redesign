"""grade_laneG_site_loop_v1.py -- Lane G (2026-09-30): SITE lines for a paired
full-size closed loop (fg_make G4 vs S0). Sibling of ops_pair_bootstrap_v1.py
(not edited), same Decision-12 floor: floor = max(2 x SD over [ref, floor
draws], half-width of the paired game-bootstrap 95% CI of the move). Decides
nothing.

    CBB_TRUTH=verified_v1 python scripts/grade_laneG_site_loop_v1.py --results-dir results/engine_v0 \
        --ref v3full_S0_s200_o0 --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
        --arms v3full_G4_s200_o0 --out-json <path> --out-md <path> [--n-boot 500]

Lines (sim per-game means vs verified actual; "real HCA" = team-FE-adjusted):
  g6_margin_ha        mean margin, home/away games        (G6 non-neutral line)
  g6_margin_neu       mean margin, neutral games          (listed-home line, NOT home advantage)
  hca_fe              FE margin model b_home (non-neutral): margin = theta_h - theta_a + b_home*nn + b_neu*neu
  hca_fe_bias         hca_fe(sim) - hca_fe(actual)        (actual 3.06 on 2024-25)
  neu_fe              FE b_neu (listed home at neutral)
  team_part_ha_bias   mean over home/away games of FE team part, sim - actual (the strength compression)
  g9_slope, g9_bias   OLS slope of actual on sim mean; mean(sim - actual)
  bias_ha, bias_neu   margin bias by site
  efg_home, efg_away, efg_neu, efg_hma   sim eFG% by offence site (pooled sums), and home-minus-away
  efg_hma_bias        sim home-minus-away eFG minus actual's
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
from cbb_sim.eval import reference as R  # noqa: E402

COLS = ["game_id", "seed", "home_pts", "away_pts", "possessions", "n_periods",
        "home_fga3", "away_fga3", "home_fga2_rim", "away_fga2_rim", "home_fga2_jump", "away_fga2_jump",
        "home_fgm2_rim", "away_fgm2_rim", "home_fgm2_jump", "away_fgm2_jump", "home_fgm3", "away_fgm3"]
LINES = ["g6_margin_ha", "g6_margin_neu", "hca_fe", "hca_fe_bias", "neu_fe", "team_part_ha_bias",
         "g9_slope", "g9_bias", "bias_ha", "bias_neu", "efg_home", "efg_away", "efg_neu", "efg_hma",
         "efg_hma_bias"]


def actual_efg() -> pd.DataFrame:
    b = R.load_actual_team_box(2025)
    b["ev"] = b["fgm"] + 0.5 * b["tpm"]
    h = b[b["team_home_away"] == "home"].set_index("game_id")[["ev", "fga"]].add_prefix("ah_")
    a = b[b["team_home_away"] == "away"].set_index("game_id")[["ev", "fga"]].add_prefix("aa_")
    return h.join(a, how="inner")


def per_game(results_dir: Path, tag: str, act: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_parquet(results_dir / tag / "games.parquet", columns=COLS)
    summ, _ = G.build_grading_frame(g, 2025)
    for s in ("home", "away"):
        g[f"{s}_fga"] = g[f"{s}_fga3"] + g[f"{s}_fga2_rim"] + g[f"{s}_fga2_jump"]
        g[f"{s}_ev"] = g[f"{s}_fgm2_rim"] + g[f"{s}_fgm2_jump"] + 1.5 * g[f"{s}_fgm3"]
    a = g.groupby("game_id")[["home_fga", "home_ev", "away_fga", "away_ev"]].mean()
    s = summ.set_index("game_id")[["margin", "sim_margin_mean", "neutral", "home_team_id", "away_team_id"]]
    d = s.join(a, how="inner").join(act, how="inner").sort_index()
    return d


def fe_design(d: pd.DataFrame) -> np.ndarray:
    teams = np.unique(np.concatenate([d["home_team_id"], d["away_team_id"]]))
    ti = {t: i for i, t in enumerate(teams)}
    n, T = len(d), len(teams)
    X = np.zeros((n, 2 + T))
    X[:, 0] = (d["neutral"] == 0).astype(float)
    X[:, 1] = (d["neutral"] > 0).astype(float)
    X[np.arange(n), 2 + d["home_team_id"].map(ti).to_numpy()] += 1
    X[np.arange(n), 2 + d["away_team_id"].map(ti).to_numpy()] -= 1
    return X


def fe_fit(X, y, w):
    reg = np.zeros(X.shape[1]); reg[2:] = 1e-6 * w.mean()
    Xw = X * w[:, None]
    beta = np.linalg.solve(X.T @ Xw + np.diag(reg), Xw.T @ y)
    return beta, X[:, 2:] @ beta[2:]


def stats(d: pd.DataFrame, X: np.ndarray, w: np.ndarray) -> dict[str, float]:
    nn = (d["neutral"] == 0).to_numpy()
    x, y = d["sim_margin_mean"].to_numpy(), d["margin"].to_numpy()
    wm = lambda v, m=None: float((w * v)[m].sum() / w[m].sum()) if m is not None else float((w * v).sum() / w.sum())  # noqa: E731
    o = {"g6_margin_ha": wm(x, nn), "g6_margin_neu": wm(x, ~nn)}
    bx, tx = fe_fit(X, x, w)
    by, ty = fe_fit(X, y, w)
    o["hca_fe"] = float(bx[0]); o["hca_fe_bias"] = float(bx[0] - by[0]); o["neu_fe"] = float(bx[1])
    o["team_part_ha_bias"] = wm(tx - ty, nn)
    mx, my = wm(x), wm(y)
    o["g9_slope"] = float((w * (x - mx) * (y - my)).sum() / (w * (x - mx) ** 2).sum())
    o["g9_bias"] = wm(x - y); o["bias_ha"] = wm(x - y, nn); o["bias_neu"] = wm(x - y, ~nn)
    ev = lambda c, m: float((w * d[c].to_numpy())[m].sum() / (w * d[c.replace("ev", "fga")].to_numpy())[m].sum())  # noqa: E731
    o["efg_home"] = ev("home_ev", nn); o["efg_away"] = ev("away_ev", nn)
    o["efg_neu"] = float(((w * (d["home_ev"] + d["away_ev"]).to_numpy())[~nn].sum())
                         / (w * (d["home_fga"] + d["away_fga"]).to_numpy())[~nn].sum())
    o["efg_hma"] = o["efg_home"] - o["efg_away"]
    o["efg_hma_bias"] = o["efg_hma"] - (ev("ah_ev", nn) - ev("aa_ev", nn))
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/engine_v0")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--floors", default="")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    Rd = Path(a.results_dir)
    act = actual_efg()
    ref = per_game(Rd, a.ref, act)
    Xr = fe_design(ref)
    s_ref = stats(ref, Xr, np.ones(len(ref)))
    draws = [s_ref]
    for f in [t for t in a.floors.split(",") if t]:
        df = per_game(Rd, f, act)
        draws.append(stats(df, fe_design(df), np.ones(len(df))))
    draw_sd = {k: float(np.std([dd[k] for dd in draws], ddof=1)) if len(draws) > 1 else float("nan") for k in LINES}
    out = {"ref": a.ref, "floors": a.floors, "ref_values": s_ref, "draws": draws, "draw_sd": draw_sd, "arms": {}}
    md = [f"Lane G site lines, ref `{a.ref}`, {len(draws) - 1} floor draws, {a.n_boot} paired game resamples; "
          "floor = max(2 x draw SD, bootstrap 95% half-width).\n"]
    for arm in [t for t in a.arms.split(",") if t]:
        d = per_game(Rd, arm, act)
        common = ref.index.intersection(d.index)
        r_, d_ = ref.loc[common], d.loc[common]
        X = fe_design(r_)
        fr, fa = stats(r_, X, np.ones(len(common))), stats(d_, X, np.ones(len(common)))
        rng = np.random.default_rng(20261001)
        diffs = {k: [] for k in LINES}
        for _ in range(a.n_boot):
            w = np.bincount(rng.integers(0, len(common), len(common)), minlength=len(common)).astype(float)
            sa, sr = stats(d_, X, w), stats(r_, X, w)
            for k in LINES:
                diffs[k].append(sa[k] - sr[k])
        res = {}
        md += [f"\n### `{arm}` vs `{a.ref}`, {len(common)} common games\n",
               "| line | ref | arm | move | 2 x draw SD | boot 95% CI | floor | move/floor | flag |",
               "|---|---|---|---|---|---|---|---|---|"]
        for k in LINES:
            lo, hi = np.quantile(diffs[k], [0.025, 0.975])
            mv = fa[k] - fr[k]
            fl = max(2 * draw_sd[k] if draw_sd[k] == draw_sd[k] else 0.0, (hi - lo) / 2)
            flag = "BEYOND" if abs(mv) > fl else "inside"
            res[k] = {"ref": fr[k], "arm": fa[k], "move": mv, "floor": fl, "ci": [float(lo), float(hi)], "flag": flag}
            md.append(f"| {k} | {fr[k]:.4f} | {fa[k]:.4f} | {mv:+.4f} | {2*draw_sd[k]:.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                      f"{fl:.4f} | {mv/fl if fl else float('nan'):+.2f} | {flag} |")
        out["arms"][arm] = res
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=1, default=float))
    Path(a.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
