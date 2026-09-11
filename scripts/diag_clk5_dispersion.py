"""diag_clk5_dispersion.py -- where the clock's WITHIN-GAME dispersion shortfall lives.

Round 5's measurement step. It FITS NOTHING and SCORES NO ARM. It answers one
question with three candidate answers, on the 2025 (F2 test) clock-complete
regulation possessions and on the SERVED arm's own predictive law:

  the engine's per-team-game possession SD is 3.743 where 4.972 is needed
  (`docs/tests/engine_v1_variance_ot_diag_2026-09-11.md` section 2). Is that

  (a) the per-possession CONDITIONAL law being too narrow -- the served cell
      pmf's own variance smaller than the residual variance the real
      possessions carry around the same conditional mean;
  (b) a missing positive WITHIN-GAME correlation -- a game-level duration
      latent the iid inverse-CDF draw cannot produce, measured as the excess
      variance of the per-game mean residual over its iid sampling variance,
      and cross-checked by the within-game lag-1/lag-2 residual
      autocorrelation; or
  (c) the state COMPOSITION the engine visits (L34's prev_end mix), which
      moves the mixture variance even with the conditional law held fixed.

THE BRIDGE, stated once because every number below rides on it.
A regulation game's possessions tile 2400 s exactly on a clock-complete game
(round 4's tiling identity, 768,834 of 768,834 rows). So per-team-game
possessions P and the game's realised mean duration Dbar satisfy

    P = 1200 / Dbar      exactly, and     dP/dDbar = -P/Dbar,

hence   SD(P) ~= (Pbar / mu) * SD(Dbar),  mu = pooled mean duration.

That is a first-order delta-method step on an EXACT identity, and it is the
only approximation in the decomposition; it is checked numerically against the
realised per-game P on the same games (`bridge_check`).

Every "needed" quantity is the actual residual around the SERVED MODEL's own
conditional mean, which is the offline mirror of the engine's
SD(actual - sim per-game mean). Every "produced" quantity is the served law's
own variance under iid inverse-CDF sampling, which is what
`clock.sample_from_pmf` does today. Neither is a re-simulation and both are
labelled as arithmetic on a measured decomposition.

Usage:
    .venv/Scripts/python.exe scripts/diag_clk5_dispersion.py \
        --season 2025 --mode v3c_srfloor_P3_s1 --out data/processed/models/clock/v5_diag
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import clock as CK  # noqa: E402

CK_DIR = ROOT / "data" / "processed" / "models" / "clock"
UNIV_V2 = ROOT / "data" / "processed" / "games_universe_v2.parquet"
DESIGN = CK_DIR / "design_v2.parquet"
CHUNK = 40_000


def load_schedule(mode: str) -> list[dict]:
    from cbb_sim.engine.clock_adapter_v3 import V3C_MODES
    spec = V3C_MODES[mode]
    doc = json.loads((CK_DIR / spec["manifest"]).read_text(encoding="utf-8"))
    out = []
    for m in doc["months"]:
        with open(CK_DIR / m["model_file"], "rb") as f:
            out.append({"refit_date": pd.Timestamp(m["refit_date"]),
                        "arm": pickle.load(f)})
    out.sort(key=lambda r: r["refit_date"])
    return out


def moments(arm, df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """(E[T], Var[T], E[min(T,R)], Var[min(T,R)]) per row, from the arm's pmf.

    min(T,R) is exactly what loop.py subtracts from the clock (L20), so the
    consumed moments are the ones the possession COUNT is a function of."""
    grid = np.arange(CK.DURATION_CAP + 1, dtype=np.float64)
    n = len(df)
    e_t = np.empty(n); v_t = np.empty(n); e_m = np.empty(n); v_m = np.empty(n)
    for lo in range(0, n, CHUNK):
        blk = df.iloc[lo:lo + CHUNK]
        pmf = np.asarray(arm.pmf(blk.reset_index(drop=True)), dtype=np.float64)
        hi = lo + len(blk)
        r = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
        mt = pmf @ grid
        e_t[lo:hi] = mt
        v_t[lo:hi] = (pmf @ (grid ** 2)) - mt ** 2
        c = np.minimum(grid[None, :], r)
        mm = (pmf * c).sum(axis=1)
        e_m[lo:hi] = mm
        v_m[lo:hi] = (pmf * c ** 2).sum(axis=1) - mm ** 2
    return e_t, v_t, e_m, v_m


def lag_autocorr(d: pd.DataFrame, col: str,
                 lags=(1, 2, 3, 4, 6, 8, 12, 20, 40)) -> dict:
    """Within-(game, period) autocorrelation of a residual column.

    Possessions alternate offences, so lag 1 is the OPPONENT's next possession
    and lag 2 is the same offence's next one. Both are reported: a game-level
    tempo latent lifts every lag, an offence-specific one lifts only the even
    lags."""
    d = d.sort_values(["game_id", "period", "poss_index"], kind="stable")
    g = d["game_id"].to_numpy(); p = d["period"].to_numpy()
    x = d[col].to_numpy(dtype=np.float64)
    x = x - x.mean()
    out = {}
    for L in lags:
        ok = (g[:-L] == g[L:]) & (p[:-L] == p[L:])
        a = x[:-L][ok]; b = x[L:][ok]
        out[f"lag{L}"] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else float("nan")
        out[f"lag{L}_n"] = int(len(a))
    return out


def _mean_sq(s) -> float:
    return float(np.mean(np.asarray(s, dtype=np.float64) ** 2))


def game_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per clock-complete game: realised and modelled mean duration."""
    g = d.groupby("game_id")
    t = pd.DataFrame({
        "M": g.size(),
        "dbar": g["duration_s"].mean(),
        "mbar": g["e_min"].mean(),
        "sum_v": g["v_min"].sum(),
        "sum_dur": g["duration_s"].sum(),
        "resid_var_actual": g["resid"].apply(_mean_sq),
        "tempo": g["tempo_prior_game"].first(),
    }).reset_index()
    t["rbar"] = t["dbar"] - t["mbar"]
    t["iid_var_dbar"] = t["sum_v"] / t["M"] ** 2
    t["P"] = 1200.0 / t["dbar"]
    return t


def decompose(d: pd.DataFrame, gt: pd.DataFrame, label: str) -> dict:
    mu = float(d["duration_s"].mean())
    Pbar = float(gt["P"].mean())
    k = Pbar / mu                       # bridge factor: possessions per second of Dbar

    v_model = float(d["v_min"].mean())              # the served law's conditional variance
    v_act = _mean_sq(d["resid"])                    # actual variance around the same mean
    Mbar = float(gt["M"].mean())

    var_rbar = float(np.var(gt["rbar"].to_numpy(), ddof=0))
    iid_var = float(gt["iid_var_dbar"].mean())      # iid sampling of the served law
    iid_var_actual_law = float(np.mean(gt["resid_var_actual"].to_numpy()
                                       / gt["M"].to_numpy()))
    tau2 = var_rbar - iid_var_actual_law            # game-level latent, net of (a)

    gap_var = var_rbar - iid_var                    # whole Dbar-variance shortfall
    ch_a = iid_var_actual_law - iid_var             # conditional law too narrow
    ch_b = tau2                                     # missing within-game correlation

    return {
        "label": label,
        "n_possessions": int(len(d)), "n_games": int(len(gt)),
        "mu_actual_mean_duration": mu,
        "model_mean_consumed": float(d["e_min"].mean()),
        "Pbar_realised": Pbar,
        "P_sd_realised": float(np.std(gt["P"].to_numpy(), ddof=0)),
        "bridge_factor_k_poss_per_s": k,
        "Mbar_possessions_per_game": Mbar,
        "sd_model_conditional": float(np.sqrt(v_model)),
        "sd_actual_residual": float(np.sqrt(v_act)),
        "conditional_sd_ratio": float(np.sqrt(v_model / v_act)),
        "sd_actual_marginal": float(d["duration_s"].std(ddof=0)),
        "sd_model_marginal": float(np.sqrt(float(np.var(d["e_min"].to_numpy(), ddof=0))
                                           + v_model)),
        "var_rbar_needed": var_rbar,
        "var_dbar_iid_produced": iid_var,
        "var_dbar_iid_with_actual_law": iid_var_actual_law,
        "tau2_game_latent": tau2,
        "sd_rbar_needed_s": float(np.sqrt(max(var_rbar, 0.0))),
        "sd_dbar_produced_s": float(np.sqrt(max(iid_var, 0.0))),
        "tau_game_latent_s": float(np.sqrt(max(tau2, 0.0))),
        "tau_cv_pct": float(100.0 * np.sqrt(max(tau2, 0.0)) / mu),
        "P_sd_needed_bridge": k * float(np.sqrt(max(var_rbar, 0.0))),
        "P_sd_produced_bridge": k * float(np.sqrt(max(iid_var, 0.0))),
        "gap_var_dbar": gap_var,
        "channel_a_var": ch_a,
        "channel_a_share": float(ch_a / gap_var) if gap_var else float("nan"),
        "channel_b_var": ch_b,
        "channel_b_share": float(ch_b / gap_var) if gap_var else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--mode", default="v3c_srfloor_P3_s1")
    ap.add_argument("--out", default="data/processed/models/clock/v5_diag")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    univ = pd.read_parquet(UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))

    design = pd.read_parquet(DESIGN)
    d = design[(design["season"] == a.season) & (design["period"] <= 2.0)
               & (design["game_id"].isin(cc))].copy()
    d = d.sort_values(["game_id", "period", "poss_index"],
                      kind="stable").reset_index(drop=True)

    sched = load_schedule(a.mode)
    cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
    seg = np.searchsorted(cuts, pd.to_datetime(d["game_date"]).to_numpy(), side="right") - 1
    assert (seg >= 0).all(), "a game tips before the first refit date"

    e_t = np.empty(len(d)); v_t = np.empty(len(d))
    e_m = np.empty(len(d)); v_m = np.empty(len(d))
    for k in np.unique(seg):
        r = np.flatnonzero(seg == k)
        a1, a2, a3, a4 = moments(sched[int(k)]["arm"], d.iloc[r])
        e_t[r] = a1; v_t[r] = a2; e_m[r] = a3; v_m[r] = a4
    d["e_t"] = e_t; d["v_t"] = v_t; d["e_min"] = e_m; d["v_min"] = v_m
    d["resid"] = d["duration_s"].to_numpy(dtype=np.float64) - d["e_min"]
    d["z"] = d["resid"] / np.sqrt(np.maximum(d["v_min"], 1e-9))
    d["r2_bucket"] = np.asarray(CK.R2_SR_LABELS, dtype=object)[
        CK.r2_bucket_id(d["seconds_remaining"].to_numpy())]
    d["period_cell"] = np.where(d["period"] <= 1.0, "H1", "H2")

    gt = game_table(d)
    gt["tempo_quintile"] = pd.qcut(gt["tempo"], 5,
                                   labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    d = d.merge(gt[["game_id", "tempo_quintile", "M"]], on="game_id", how="left")

    rep: dict = {"season": a.season, "mode": a.mode,
                 "n_clock_complete_games": int(len(gt))}
    rep["overall"] = decompose(d, gt, "all_cc")

    kk = rep["overall"]["bridge_factor_k_poss_per_s"]
    rep["bridge_check"] = {
        "P_sd_realised": rep["overall"]["P_sd_realised"],
        "P_sd_from_bridge_on_realised_dbar": kk * float(np.std(gt["dbar"].to_numpy(), ddof=0)),
        "sd_dbar_realised_s": float(np.std(gt["dbar"].to_numpy(), ddof=0)),
        "note": "delta method on an exact identity; ratio should be ~1.00",
    }
    rep["bridge_check"]["ratio"] = (rep["bridge_check"]["P_sd_from_bridge_on_realised_dbar"]
                                    / rep["bridge_check"]["P_sd_realised"])

    rep["autocorr_resid"] = lag_autocorr(d, "resid")
    rep["autocorr_z"] = lag_autocorr(d, "z")

    # Is the latent SHARED by the two offences in a game, or the offence's own?
    # Possessions alternate, so the two offences' mean residuals inside one game
    # are two draws on the same night. Their correlation separates a game-level
    # pace realisation (CLAUDE.md's "one pace realisation per simulated game")
    # from an offence-level one, and the two need different variances to move
    # the possession count by the same amount.
    ho = (d.assign(off=np.where(d["offense_is_home"] > 0, "H", "A"))
           .groupby(["game_id", "off"])
           .agg(r=("resid", "mean"), n=("resid", "size"),
                v=("v_min", "sum")).reset_index())
    w = ho.pivot(index="game_id", columns="off", values=["r", "n", "v"]).dropna()
    rh = w[("r", "H")].to_numpy(); ra = w[("r", "A")].to_numpy()
    nh = w[("n", "H")].to_numpy(); na = w[("n", "A")].to_numpy()
    vh = w[("v", "H")].to_numpy() / nh ** 2
    va_ = w[("v", "A")].to_numpy() / na ** 2
    cov = float(np.mean((rh - rh.mean()) * (ra - ra.mean())))
    var_h = float(np.var(rh, ddof=0)); var_a = float(np.var(ra, ddof=0))
    # the iid sampling floor subtracts from each side's variance but NOT from
    # the covariance, so the shared component is the covariance itself
    rep["latent_shared_vs_team"] = {
        "n_games": int(len(w)),
        "var_rbar_home_offence": var_h, "var_rbar_away_offence": var_a,
        "iid_floor_home": float(np.mean(vh)), "iid_floor_away": float(np.mean(va_)),
        "cov_home_away_offence": cov,
        "corr_home_away_offence": float(cov / np.sqrt(var_h * var_a)),
        "shared_game_latent_var": cov,
        "offence_own_latent_var_home": var_h - float(np.mean(vh)) - cov,
        "offence_own_latent_var_away": var_a - float(np.mean(va_)) - cov,
        "note": ("cov is the game-SHARED latent variance; each side's excess "
                 "over its iid floor minus cov is that offence's own latent."),
    }

    # (c) composition: the mixture variance under the engine's own prev_end mix
    pe = d.groupby("prev_end").agg(n=("duration_s", "size"),
                                   mean_act=("duration_s", "mean"),
                                   var_act=("duration_s", "var"),
                                   mean_mod=("e_min", "mean"),
                                   v_mod=("v_min", "mean"),
                                   resid_var=("resid", _mean_sq)).reset_index()
    pe["share_actual"] = pe["n"] / pe["n"].sum()
    # L34 / round 4 section 15.4: the engine starts 2.50 pp FEWER possessions
    # after a made FG and 2.10 pp MORE after a defensive rebound. The residual
    # 0.40 pp is absorbed pro rata by the other levels. This is an IMPORTED
    # measurement, not one this script makes.
    shift = {"made_FG": -0.0250, "DREB": +0.0210}
    s = pe["share_actual"].to_numpy().copy()
    lev = pe["prev_end"].to_numpy()
    named = np.isin(lev, list(shift))
    for i, l in enumerate(lev):
        if l in shift:
            s[i] += shift[l]
    rem = 1.0 - s.sum()
    if (~named).any():
        base = pe["share_actual"].to_numpy()[~named]
        s[~named] += rem * base / base.sum()
    pe["share_engine_L34"] = s
    m_a = pe["mean_act"].to_numpy(); v_a = pe["var_act"].to_numpy()

    def mix_var(w):
        m = float((w * m_a).sum())
        return float((w * (v_a + m_a ** 2)).sum() - m ** 2), m

    va, ma = mix_var(pe["share_actual"].to_numpy())
    ve, me = mix_var(pe["share_engine_L34"].to_numpy())
    rep["composition_channel_c"] = {
        "mix_mean_actual": ma, "mix_mean_engine": me, "mix_mean_delta_s": me - ma,
        "mix_var_actual": va, "mix_var_engine": ve, "mix_var_delta": ve - va,
        "mix_sd_actual": float(np.sqrt(va)), "mix_sd_engine": float(np.sqrt(ve)),
        "note": ("prev_end shares shifted by L34's measured engine mix error "
                 "(made_FG -2.50 pp, DREB +2.10 pp); per-cell laws held at their "
                 "actual values. Marginal mixture variance only -- an iid mixture "
                 "term, which enters Var(Dbar) divided by M."),
    }
    rep["composition_channel_c"]["var_dbar_effect"] = (
        (ve - va) / rep["overall"]["Mbar_possessions_per_game"])
    pe.to_csv(out / "v5_prev_end.csv", index=False)

    def seg_table(by: str) -> pd.DataFrame:
        g = d.groupby(by, dropna=False)
        t = g.agg(n=("duration_s", "size"),
                  actual_mean=("duration_s", "mean"),
                  model_mean=("e_min", "mean"),
                  actual_sd=("duration_s", "std"),
                  model_cond_var=("v_min", "mean"),
                  resid_ms=("resid", _mean_sq),
                  z_sd=("z", lambda s: float(np.std(np.asarray(s), ddof=0)))).reset_index()
        t["model_cond_sd"] = np.sqrt(t["model_cond_var"])
        t["resid_sd"] = np.sqrt(t["resid_ms"])
        t["cond_sd_ratio"] = t["model_cond_sd"] / t["resid_sd"]
        t["underpowered"] = t["n"] < 300
        return t

    for by in ("prev_end", "r2_bucket", "period_cell", "tempo_quintile", "terminal_event"):
        seg_table(by).to_csv(out / f"v5_seg_{by}.csv", index=False)

    rows = []
    for q, sub in d.groupby("tempo_quintile"):
        sg = gt[gt["tempo_quintile"] == q]
        rows.append(decompose(sub, sg, str(q)))
    pd.DataFrame(rows).to_csv(out / "v5_decomp_by_tempo_quintile.csv", index=False)
    rep["by_tempo_quintile"] = rows

    gt.to_parquet(out / "v5_games.parquet", index=False)
    pcts = [1, 5, 25, 50, 75, 95, 99]
    rep["rbar_percentiles"] = {f"p{p}": float(v) for p, v in
                               zip(pcts, np.percentile(gt["rbar"].to_numpy(), pcts))}
    rep["z_percentiles"] = {f"p{p}": float(v) for p, v in
                            zip(pcts, np.percentile(d["z"].to_numpy(), pcts))}
    rep["quantiles_duration"] = {
        "actual": {f"p{p}": float(np.percentile(d["duration_s"], p))
                   for p in (5, 10, 25, 50, 75, 90, 95, 99)},
    }

    (out / "v5_dispersion_report.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
