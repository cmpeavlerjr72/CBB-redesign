"""
grade_clk3c_closed_loop.py -- the round-3c Decision-10 closed-loop table.

Pre-registration: `docs/models/clock/experiments.md` section 12.4-12.6. One
blind grading path scores every arm: the arm's identity is read from
`run_meta.json` and is used only to label the row.

    .venv/Scripts/python.exe scripts/grade_clk3c_closed_loop.py --season 2025

Every arm is scored on the SAME games (asserted), against the SAME actuals
computed on that subset -- never against the season-wide figure.

Nothing here adjusts anything; it only reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import reference as REF  # noqa: E402
from cbb_sim.models import clock as CK  # noqa: E402

RESULTS = Path("results/engine_v0")
UNIVERSE_V2 = Path("data/processed/games_universe_v2.parquet")
#: The subset's own clock-complete end-of-half truth, written once by the
#: round-3c harness from `clock_v3.actual_end_of_half_cc` + `clock.eoh_stats`
#: -- the same two functions the offline gate uses, so the engine row and the
#: offline row describe the same statistic.
EOH_TRUTH = Path("data/processed/models/clock/v3c_eoh_truth.json")
INPUT_DIR = Path("data/processed/models/engine")
MONTH_MIN_GAMES = CK.G1_MONTH_MIN_GAMES          # 100


def load_actual(season: int, game_ids: np.ndarray) -> pd.DataFrame:
    act = REF.load_actual_games(season)[
        ["game_id", "game_date", "home_team_id", "away_team_id", "margin", "total",
         "home_score", "away_score", "went_ot"]].rename(
        columns={"margin": "act_margin", "total": "act_total"})
    poss = REF.load_actual_possessions(season).rename(columns={"game_poss": "act_poss"})
    cc = pd.read_parquet(UNIVERSE_V2)[["game_id", "clock_complete_reg"]]
    a = act.merge(poss, on="game_id", how="left").merge(cc, on="game_id", how="left")
    a = a[a["game_id"].isin(game_ids)].reset_index(drop=True)
    a["month"] = pd.to_datetime(a["game_date"]).dt.to_period("M").astype(str)
    return a


def tempo_prior(season: int, fold: str = "F2") -> pd.DataFrame:
    import json as _j
    g = pd.read_parquet(INPUT_DIR / f"games_{fold}_{season}.parquet")
    z = np.load(INPUT_DIR / f"arrays_{fold}_{season}.npz")
    names = _j.loads((INPUT_DIR / f"names_{fold}_{season}.json").read_text(encoding="utf-8"))
    j = names["team_names"]["tempo_prior_game"]
    return pd.DataFrame({"game_id": g["game_id"].to_numpy(),
                         "tempo_prior": z["team_static"][:, :, j].mean(axis=1)})


def score(res: Path, act: pd.DataFrame, tempo: pd.DataFrame) -> dict:
    raw = pd.read_parquet(res / "games.parquet")
    meta = json.loads((res / "run_meta.json").read_text(encoding="utf-8"))
    raw = raw[raw["game_id"].isin(act["game_id"])].copy()
    raw["margin"] = raw["home_pts"].astype("int64") - raw["away_pts"].astype("int64")
    raw["total"] = raw["home_pts"].astype("int64") + raw["away_pts"].astype("int64")

    g = raw.groupby("game_id")
    pg = pd.DataFrame({"margin": g["margin"].mean(), "total": g["total"].mean(),
                       "possessions": g["possessions"].mean()}).reset_index()
    pg = pg.merge(act, on="game_id", how="inner").merge(tempo, on="game_id", how="left")
    rw = raw.merge(act[["game_id", "act_poss", "clock_complete_reg", "month"]],
                   on="game_id", how="left")

    def g1(rows: pd.DataFrame, a: pd.DataFrame) -> dict:
        """Mean over every (game, seed); SD across GAMES within a seed, averaged
        over seeds -- the single-realisation dispersion the offline chain
        reports, so the two reads are the same quantity."""
        if not len(rows) or not len(a):
            return {"n_games": 0}
        sd_by_seed = rows.groupby("seed")["possessions"].std(ddof=1)
        am = a["act_poss"].to_numpy(float)
        am = am[np.isfinite(am)]
        return {
            "n_games": int(a["game_id"].nunique()),
            "sim_mean": float(rows["possessions"].mean()),
            "sim_sd": float(sd_by_seed.mean()),
            "actual_mean": float(am.mean()), "actual_sd": float(am.std(ddof=1)),
            "mean_delta": float(rows["possessions"].mean() - am.mean()),
            "sd_delta": float(sd_by_seed.mean() - am.std(ddof=1)),
        }

    cc_rows = rw[rw["clock_complete_reg"].fillna(False).astype(bool)]
    cc_act = act[act["clock_complete_reg"].fillna(False).astype(bool)]
    out = {
        "arm": meta.get("arm"), "freeze": bool(meta.get("freeze_score_diff", False)),
        "tag": res.name, "n_seeds": int(meta.get("n_seeds", 0)),
        "seed_offset": int(meta.get("seed_offset", 0)),
        "flags": {k: v for k, v in meta.get("adapter_flags", {}).items()
                  if k.startswith("ENGINE_")},
        "runtime_s": meta.get("runtime_s"),
        "G1_cc": g1(cc_rows, cc_act),
        "G1_all": g1(rw, act),
        "margin_sd": float(raw["margin"].std(ddof=1)),
        "total_sd": float(raw["total"].std(ddof=1)),
        "corr_home_away": float(np.corrcoef(raw["home_pts"].astype(float),
                                            raw["away_pts"].astype(float))[0, 1]),
        "margin_mean": float(pg["margin"].mean()),
        "margin_bias": float(pg["margin"].mean() - pg["act_margin"].mean()),
        "total_mean": float(pg["total"].mean()),
        "total_bias": float(pg["total"].mean() - pg["act_total"].mean()),
        "total_mae": float(np.abs(pg["total"] - pg["act_total"]).mean()),
        "margin_mae": float(np.abs(pg["margin"] - pg["act_margin"]).mean()),
        "ppp": float((pg["total"] / (2 * pg["possessions"])).mean()),
        "ot_rate": float((raw["n_periods"] > 2).mean()),
        "end_of_half": meta.get("end_of_half", {}).get("REG", {}),
        "end_of_half_by_period": meta.get("end_of_half", {}),
        "horn_truncated_per_game": (
            meta.get("diagnostics", {}).get("possessions_censored_by_period_end", 0)
            / max(len(raw), 1)),
    }

    # per month
    months = []
    for m, sub in rw.groupby("month", sort=True):
        a = act[act["month"] == m]
        r = g1(sub, a)
        r["month"] = m
        r["powered"] = bool(r.get("n_games", 0) >= MONTH_MIN_GAMES)
        months.append(r)
    out["months"] = months

    # responsiveness: possessions by PREGAME tempo-prior quintile of the game.
    # Team-level quintiles are underpowered at 500 games (about 2.8 appearances
    # per team), so the driver is the game's own pregame tempo prior, which is
    # the quantity the clock model is supposed to respond to.
    q = pg.dropna(subset=["tempo_prior", "act_poss"]).copy()
    q["q"] = pd.qcut(q["tempo_prior"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    gq = q.groupby("q", observed=True)
    qt = pd.DataFrame({"n": gq.size(), "sim": gq["possessions"].mean(),
                       "act": gq["act_poss"].mean()}).sort_index()
    sp_s = float(qt["sim"].iloc[-1] - qt["sim"].iloc[0])
    sp_a = float(qt["act"].iloc[-1] - qt["act"].iloc[0])
    out["quintiles"] = [{"q": int(i), "n": int(r["n"]), "sim": float(r["sim"]),
                         "act": float(r["act"]), "delta": float(r["sim"] - r["act"])}
                        for i, r in qt.iterrows()]
    out["slope_ratio"] = sp_s / sp_a if sp_a else float("nan")
    out["slope_span_sim"] = sp_s
    out["slope_span_act"] = sp_a
    out["monotone_steps"] = int(np.sum(np.sign(np.diff(qt["sim"]))
                                       == np.sign(np.diff(qt["act"]))))
    out["game_ids"] = raw["game_id"].nunique()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--pattern", default="clock3c_*")
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--out", default="data/processed/models/clock/v3c_closed_loop.json")
    a = ap.parse_args()

    dirs = sorted(p for p in Path(a.results_dir).glob(a.pattern)
                  if (p / "games.parquet").exists())
    if not dirs:
        raise SystemExit(f"no results under {a.results_dir}/{a.pattern}")
    ids = pd.read_parquet(dirs[0] / "games.parquet")["game_id"].unique()
    act = load_actual(a.season, ids)
    tempo = tempo_prior(a.season)

    rows = {}
    for d in dirs:
        rows[d.name] = score(d, act, tempo)
    seen = {r["game_ids"] for r in rows.values()}
    assert len(seen) == 1, f"arms were scored on different game sets: {seen}"

    actual = {
        "n_games": int(len(act)),
        "poss_mean_all": float(act["act_poss"].mean()),
        "poss_sd_all": float(act["act_poss"].std(ddof=1)),
        "n_cc": int(act["clock_complete_reg"].fillna(False).sum()),
        "poss_mean_cc": float(act.loc[act["clock_complete_reg"].fillna(False).astype(bool),
                                      "act_poss"].mean()),
        "poss_sd_cc": float(act.loc[act["clock_complete_reg"].fillna(False).astype(bool),
                                    "act_poss"].std(ddof=1)),
        "margin_sd": float(act["act_margin"].std(ddof=1)),
        "total_sd": float(act["act_total"].std(ddof=1)),
        "corr_home_away": float(np.corrcoef(act["home_score"].astype(float),
                                            act["away_score"].astype(float))[0, 1]),
        "margin_mean": float(act["act_margin"].mean()),
        "total_mean": float(act["act_total"].mean()),
        "ot_rate": float(act["went_ot"].mean()),
    }
    actual["ppp"] = actual["total_mean"] / (2 * actual["poss_mean_all"])

    print("=" * 118)
    print(f"CLOCK ROUND 3c -- DECISION-10 CLOSED-LOOP GATE  (season {a.season}, "
          f"{actual['n_games']} games, {actual['n_cc']} clock-complete)")
    print("=" * 118)
    hdr = (f"{'tag':<26}{'sd':>4}{'G1cc mean':>11}{'G1cc SD':>9}{'G1all mean':>12}"
           f"{'G1all SD':>10}{'marginSD':>10}{'corr':>8}{'totbias':>9}{'PPP':>8}")
    print(hdr)
    for k, r in rows.items():
        print(f"{k:<26}{r['n_seeds']:>4}"
              f"{r['G1_cc'].get('mean_delta', float('nan')):>+11.3f}"
              f"{r['G1_cc'].get('sd_delta', float('nan')):>+9.3f}"
              f"{r['G1_all'].get('mean_delta', float('nan')):>+12.3f}"
              f"{r['G1_all'].get('sd_delta', float('nan')):>+10.3f}"
              f"{r['margin_sd']:>10.3f}{r['corr_home_away']:>8.3f}"
              f"{r['total_bias']:>+9.3f}{r['ppp']:>8.4f}")
    print(f"{'ACTUAL (same games)':<26}{'--':>4}{actual['poss_mean_cc']:>11.3f}"
          f"{actual['poss_sd_cc']:>9.3f}{actual['poss_mean_all']:>12.3f}"
          f"{actual['poss_sd_all']:>10.3f}{actual['margin_sd']:>10.3f}"
          f"{actual['corr_home_away']:>8.3f}{0.0:>+9.3f}{actual['ppp']:>8.4f}")
    print("  (the ACTUAL row prints LEVELS for G1; every arm row prints the DELTA "
          "against those levels)")

    print("\nEND OF HALF (regulation, engine-side accumulator)")
    print(f"{'tag':<26}{'share<35s':>11}{'mean last dur':>15}{'mean poss dur':>15}"
          f"{'period-ending n':>17}")
    for k, r in rows.items():
        e = r["end_of_half"]
        print(f"{k:<26}{e.get('share_last_starts_under_35s', float('nan')):>11.4f}"
              f"{e.get('mean_last_possession_duration_s', float('nan')):>15.3f}"
              f"{e.get('mean_possession_duration_s', float('nan')):>15.3f}"
              f"{e.get('n_period_ending', 0):>17,}")
    if EOH_TRUTH.exists():
        t = json.loads(EOH_TRUTH.read_text(encoding="utf-8"))
        print(f"{'ACTUAL (clock-complete)':<26}"
              f"{t['actual_share_last_poss_under_35s']:>11.4f}"
              f"{t['actual_mean_last_poss_duration_s']:>15.3f}{'--':>15}"
              f"{t['n_clock_complete_halves']:>17,}")
        print("  (the sim row is every simulated regulation half; the truth is only "
              "measurable on clock-complete halves, which is the same asymmetry the "
              "offline gate carries)")
        for r in rows.values():
            e = r["end_of_half"]
            r["eoh_share_gap"] = (e.get("share_last_starts_under_35s", float("nan"))
                                  - t["actual_share_last_poss_under_35s"])
            r["eoh_duration_gap"] = (e.get("mean_last_possession_duration_s", float("nan"))
                                     - t["actual_mean_last_poss_duration_s"])
        actual["end_of_half"] = t

    print("\nRESPONSIVENESS -- possessions by PREGAME tempo-prior quintile "
          "(flat = FAIL; team-level quintiles are underpowered at 500 games)")
    for k, r in rows.items():
        print(f"  {k:<24} slope ratio {r['slope_ratio']:.3f} "
              f"(sim span {r['slope_span_sim']:+.2f} vs actual {r['slope_span_act']:+.2f}), "
              f"monotone {r['monotone_steps']}/4")

    print("\nPER MONTH, all games (cells under "
          f"{MONTH_MIN_GAMES} games are UNDERPOWERED and excluded from the decision)")
    for k, r in rows.items():
        parts = [f"{m['month']}:{m['mean_delta']:+.2f}"
                 f"{'' if m['powered'] else '*'}({m['n_games']})" for m in r["months"]]
        print(f"  {k:<24} " + "  ".join(parts))
    print("  * = underpowered")

    payload = {"season": a.season, "actual": actual,
               "month_min_games": MONTH_MIN_GAMES, "arms": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
