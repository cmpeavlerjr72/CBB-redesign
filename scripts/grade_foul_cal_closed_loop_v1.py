#!/usr/bin/env python
"""grade_foul_cal_closed_loop_v1.py -- paired closed-loop lines for the foul-accrual calendar round
(possession_outcome experiments.md s32.4). ONE grader for every arm: ref = served v3 tap run, arm = same games,
same seeds, ENGINE_FOUL_CAL set. Verified truth (`cbb_sim.eval.reference`).

Lines: total bias by days bucket (d0-14 / d15-45 / d46+, the engine inputs' days_since_start), FTA/FGA by bucket,
first-half in-bonus share by bucket vs actual (`foul_accrual_poss_v2.off_in_bonus_true`), H2 bonus trips per
in-bonus possession vs actual (chances_<season> FT_trip_bonus), plus the eval_gates lines (G5, G9, G4 ft_rate)
parsed from each run's eval_gates report. Paired game-bootstrap SE (200 reps) of arm - ref.

    .venv/Scripts/python.exe scripts/grade_foul_cal_closed_loop_v1.py --season 2025 --input-dir data/processed/models/engine_v3 \
        --ref results/engine_v0/<ref> --arm A2dbk=results/engine_v0/<arm> --gates-ref <md> --gates-arm <md> --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

BK = [("d0-14", 0, 14), ("d15-45", 15, 45), ("d46+", 46, 10_000)]
FINE = [("d0-7", 0, 7), ("d8-14", 8, 14), ("d15-30", 15, 30), ("d31-45", 31, 45), ("d46+", 46, 10_000)]
NB = 200


def sim_game(p: Path) -> pd.DataFrame:
    g = pd.read_parquet(p / "games.parquet")
    fga = sum(g[f"{s}_{k}"] for s in ("home", "away") for k in ("fga3", "fga2_rim", "fga2_jump"))
    out = pd.DataFrame({"game_id": g["game_id"], "total": g.home_pts + g.away_pts,
                        "fta": g.home_fta + g.away_fta, "fga": fga})
    return out.groupby("game_id").mean()


def half_game(p: Path) -> pd.DataFrame:
    a = pd.read_parquet(p / "half_agg.parquet")
    return a.groupby(["game_id", "half"])[["poss", "in_bonus", "n_bonus_trip", "fta"]].sum().unstack("half")


def gate_lines(md: Path) -> dict:
    t = md.read_text(encoding="utf-8")
    out = {}
    pats = {"G5_margin_sd_ratio": r"\| margin SD ratio \| ([\d.]+) \|", "G5_total_sd_ratio": r"\| total SD ratio \| ([\d.]+) \|",
            "G5_corr": r"\| home/away score correlation \| ([-\d.]+) vs", "G9_total_bias": r"\| total bias \| ([-+\d.]+) \|",
            "G9_margin_bias": r"\| margin bias \| ([-+\d.]+) \|", "G9_slope": r"\| calibration slope \| ([\d.]+) \|",
            "G4_ft_rate": r"\| ft_rate \(season, pooled, PROVISIONAL\) \| ([\d.]+) vs ([\d.]+)"}
    for k, p in pats.items():
        m = re.search(p, t)
        if m:
            out[k] = float(m.group(1))
            if k == "G4_ft_rate":
                out["G4_ft_rate_actual"] = float(m.group(2))
    m = re.search(r"\| home/away score correlation \| [-\d.]+ vs ([-\d.]+)", t)
    if m:
        out["G5_corr_actual"] = float(m.group(1))
    out["status"] = dict(re.findall(r"^\| (G\d) \| (PASS|FAIL|NEEDS-INSTRUMENTATION) \|$", t, flags=re.M))
    return out


def actual_half(season: int, ids) -> pd.DataFrame:
    acc = pd.read_parquet(ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet",
                          columns=["game_id", "season", "period", "poss_index", "off_in_bonus_true"])
    acc = acc[(acc.season == season) & acc.game_id.isin(ids)]
    ch = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{season}.parquet",
                         columns=["game_id", "period", "poss_index", "terminal_event"])
    ch = ch[ch.game_id.isin(ids)]
    bt = (ch.assign(b=(ch.terminal_event == "FT_trip_bonus").astype(int))
          .groupby(["game_id", "period", "poss_index"])["b"].sum().rename("n_bonus_trip").reset_index())
    acc = acc.merge(bt, on=["game_id", "period", "poss_index"], how="left").fillna({"n_bonus_trip": 0})
    acc["half"] = np.where(acc.period >= 3, 3, acc.period)
    acc["poss"] = 1
    acc["in_bonus"] = acc["off_in_bonus_true"].astype(int)
    return acc.groupby(["game_id", "half"])[["poss", "in_bonus", "n_bonus_trip"]].sum().unstack("half")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--gates-ref", default=None)
    ap.add_argument("--gates-arm", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert_not_sealed(a.season)
    fold = a.fold or ("F2" if a.season == 2025 else "F1")
    inp = EngineInputs.load(a.input_dir, f"{fold}_{a.season}")
    dss = pd.Series(inp.team_static[:, 0, inp.team_names["days_since_start"]], index=inp.games["game_id"].to_numpy())
    runs = {"ref": ROOT / a.ref, **{k: ROOT / v for k, v in (x.split("=", 1) for x in a.arm)}}
    G = {k: sim_game(v) for k, v in runs.items()}
    H = {k: half_game(v) for k, v in runs.items()}
    act = R.load_actual_games(a.season).set_index("game_id")
    ids = sorted(set.intersection(*[set(v.index) for v in G.values()]) & set(act.index))
    A = act.loc[ids]
    d = dss.loc[ids].to_numpy()
    tb = R.load_actual_team_box(a.season)
    tb = tb[tb.game_id.isin(ids)].groupby("game_id")[["fta", "fga"]].sum().reindex(ids)
    AH = actual_half(a.season, ids).reindex(ids)
    rng = np.random.default_rng(20261007)
    out = {"season": a.season, "fold": fold, "n_games": len(ids), "runs": {k: str(v) for k, v in runs.items()},
           "seeds_note": "paired seeds (same seed list in every run, see run_meta)", "arms": {}}

    def ratio_line(num, den, idx):
        return num[idx].sum() / den[idx].sum()

    def boot(fn_arm, fn_ref, idx):
        v = []
        for _ in range(NB):
            b = idx[rng.integers(0, len(idx), len(idx))]
            v.append(fn_arm(b) - fn_ref(b))
        return float(np.std(v, ddof=1))

    def per_run(k):
        g = G[k].loc[ids]
        h = H[k].reindex(ids)
        return {"bias": g.total.to_numpy() - A.total.to_numpy(), "fta": g.fta.to_numpy(), "fga": g.fga.to_numpy(),
                "h1p": h[("poss", 1)].to_numpy(), "h1b": h[("in_bonus", 1)].to_numpy(),
                "h2p": h[("poss", 2)].to_numpy(), "h2b": h[("in_bonus", 2)].to_numpy(),
                "h2t": h[("n_bonus_trip", 2)].to_numpy(), "h1t": h[("n_bonus_trip", 1)].to_numpy()}

    P = {k: per_run(k) for k in runs}
    a_h1p, a_h1b = AH[("poss", 1)].to_numpy(float), AH[("in_bonus", 1)].to_numpy(float)
    a_h2p, a_h2b, a_h2t = (AH[("poss", 2)].to_numpy(float), AH[("in_bonus", 2)].to_numpy(float),
                           AH[("n_bonus_trip", 2)].to_numpy(float))
    ok_a = np.isfinite(a_h1p) & np.isfinite(a_h2p)
    gl = {"ref": gate_lines(Path(a.gates_ref)) if a.gates_ref else {}}
    for x in a.gates_arm:
        k, v = x.split("=", 1)
        gl[k] = gate_lines(Path(v))
    out["gates_ref"] = gl["ref"]
    for arm in [k for k in runs if k != "ref"]:
        r, x = P["ref"], P[arm]
        res = {}
        for lab, lo, hi in BK + [("all", 0, 10_000)]:
            idx = np.where((d >= lo) & (d <= hi))[0]
            ia = idx[ok_a[idx]]
            res[lab] = {
                "n_games": int(len(idx)),
                "total_bias_ref": float(r["bias"][idx].mean()), "total_bias_arm": float(x["bias"][idx].mean()),
                "total_bias_move_se": boot(lambda b: x["bias"][b].mean(), lambda b: r["bias"][b].mean(), idx),
                "total_bias_se_level": float(r["bias"][idx].std(ddof=1) / np.sqrt(len(idx))),
                "fta_fga_ref": float(ratio_line(r["fta"], r["fga"], idx)), "fta_fga_arm": float(ratio_line(x["fta"], x["fga"], idx)),
                "fta_fga_actual": float(np.nansum(tb.fta.to_numpy()[idx]) / np.nansum(tb.fga.to_numpy()[idx])),
                "n_games_no_box": int(np.isnan(tb.fta.to_numpy()[idx]).sum()),
                "fta_fga_move_se": boot(lambda b: ratio_line(x["fta"], x["fga"], b), lambda b: ratio_line(r["fta"], r["fga"], b), idx),
                "H1_inbonus_actual": float(a_h1b[ia].sum() / a_h1p[ia].sum()),
                "H1_inbonus_ref": float(ratio_line(r["h1b"], r["h1p"], ia)), "H1_inbonus_arm": float(ratio_line(x["h1b"], x["h1p"], ia)),
                "H1_inbonus_move_se": boot(lambda b: ratio_line(x["h1b"], x["h1p"], b), lambda b: ratio_line(r["h1b"], r["h1p"], b), ia),
                "H2_inbonus_actual": float(a_h2b[ia].sum() / a_h2p[ia].sum()),
                "H2_inbonus_ref": float(ratio_line(r["h2b"], r["h2p"], ia)), "H2_inbonus_arm": float(ratio_line(x["h2b"], x["h2p"], ia)),
                "H2_trips_per_inbonus_actual": float(a_h2t[ia].sum() / a_h2b[ia].sum()),
                "H2_trips_per_inbonus_ref": float(ratio_line(r["h2t"], r["h2b"], ia)),
                "H2_trips_per_inbonus_arm": float(ratio_line(x["h2t"], x["h2b"], ia)),
            }
        for lab, lo, hi in FINE:
            idx = np.where((d >= lo) & (d <= hi))[0]
            ia = idx[ok_a[idx]]
            res[f"fine_{lab}"] = {"n_games": int(len(idx)), "H1_inbonus_actual": float(a_h1b[ia].sum() / a_h1p[ia].sum()),
                                  "H1_inbonus_ref": float(ratio_line(r["h1b"], r["h1p"], ia)),
                                  "H1_inbonus_arm": float(ratio_line(x["h1b"], x["h1p"], ia))}
        res["gates_arm"] = gl.get(arm, {})
        out["arms"][arm] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for arm, res in out["arms"].items():
        print(f"== {arm} ({out['n_games']} games)")
        for lab in ("d0-14", "d15-45", "d46+", "all"):
            v = res[lab]
            print(f"  {lab:6s} n {v['n_games']:5d} total bias {v['total_bias_ref']:+.2f} -> {v['total_bias_arm']:+.2f} "
                  f"(se {v['total_bias_move_se']:.3f}) | FTA/FGA {v['fta_fga_ref']:.4f} -> {v['fta_fga_arm']:.4f} act {v['fta_fga_actual']:.4f}"
                  f" | H1 inb {v['H1_inbonus_ref']:.4f} -> {v['H1_inbonus_arm']:.4f} act {v['H1_inbonus_actual']:.4f}"
                  f" | H2 inb {v['H2_inbonus_ref']:.4f} -> {v['H2_inbonus_arm']:.4f} act {v['H2_inbonus_actual']:.4f}"
                  f" | H2 trips/inb {v['H2_trips_per_inbonus_ref']:.4f} -> {v['H2_trips_per_inbonus_arm']:.4f} act {v['H2_trips_per_inbonus_actual']:.4f}")
        print("  gates ref", {k: v for k, v in out["gates_ref"].items() if k != "status"}, out["gates_ref"].get("status"))
        print("  gates arm", {k: v for k, v in res["gates_arm"].items() if k != "status"}, res["gates_arm"].get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
