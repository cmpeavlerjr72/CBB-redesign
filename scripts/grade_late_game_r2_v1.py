"""grade_late_game_r2_v1.py -- late-game ROUND 2: the ONE grader for every closed-loop arm.

Pre-registration: `docs/models/late_game/experiments.md` section 4. One code
path per run; the arm's identity comes from `run_meta.json` and only labels the
row. Nothing here adjusts anything.

    .venv/Scripts/python.exe scripts/grade_late_game_r2_v1.py \
        --ref lg2_R_s25 --floor lg2_Rfloor_s25 --arms lg2_W_C2_s25 lg2_W_D_s25 ... \
        --out results/late_game/round2/grade_stride.json
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

from cbb_sim.eval import gates as G                                   # noqa: E402
from cbb_sim.eval import reference as REF                             # noqa: E402

RES = ROOT / "results/engine_v0"
TARGET_RATIO, TARGET_OT = 1.546, 0.0557
OT_BAND = (0.046, 0.055)
ROLES = ("trailing", "tied", "leading")
BUCKETS = ("(0,10]", "(10,30]", "(30,60]", "(60,120]")


def load(tag: str) -> dict:
    d = RES / tag
    g = pd.read_parquet(d / "games.parquet")
    t = pd.read_parquet(d / "tap_sims.parquet")
    w = np.load(d / "tap_window.npz")
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    g = g.merge(t, on=["game_id", "seed"], how="left", validate="one_to_one")
    return {"tag": tag, "g": g, "clk": w["clock"], "ev": w["event"], "meta": meta}


def lines(run: dict, season: int = 2025) -> dict:
    g = run["g"]
    ot = g["n_periods"].to_numpy() > 2
    fm = (g["home_pts"] - g["away_pts"]).abs().to_numpy()
    reg = np.where(ot, 0, fm)
    out: dict = {"tag": run["tag"], "arm": run["meta"].get("arm"), "n_sims": int(len(g))}
    out["P0"] = float(ot.mean())
    out["P1"] = float((reg == 1).mean())
    out["P0_over_P1"] = out["P0"] / out["P1"]
    out["ot_rate"] = out["P0"]
    out["n_reg_ties"] = int(ot.sum())
    for k in range(6):
        out[f"P_abs_m_{k}"] = float((reg == k).mean())
    # --- whole-game gates on the run's own games -------------------------
    summary, raw = G.build_grading_frame(g, season)
    out["G1_poss_mean"] = float(raw["possessions"].mean())
    out["G1_poss_sd"] = float(raw["possessions"].std())
    out["act_poss_mean"] = float(summary["game_poss"].mean())
    out["act_poss_sd"] = float(summary["game_poss"].std())
    resid_m = float((summary["margin"] - summary["sim_margin_mean"]).std())
    resid_t = float((summary["total"] - summary["sim_total_mean"]).std())
    out["G5_margin_sd_ratio"] = float(summary["sim_margin_sd"].mean()) / resid_m
    out["G5_total_sd_ratio"] = float(summary["sim_total_sd"].mean()) / resid_t
    h = G.headline(summary)
    out["G9_margin_bias"] = h["margin_bias"]
    out["G9_total_bias"] = h["total_bias"]
    out["G7_ot_rate_graded"] = float(summary["sim_ot_rate"].mean())
    out["act_ot_rate_sample"] = float(summary["went_ot"].mean())
    am = (summary["margin"].abs()).to_numpy()
    aot = summary["went_ot"].to_numpy().astype(bool)
    out["act_P0_sample"] = float(aot.mean())
    out["act_P1_sample"] = float(((~aot) & (am == 1)).mean())
    out["n_graded_games"] = int(len(summary))
    # --- half share ------------------------------------------------------
    tot = (g["home_pts"] + g["away_pts"]).to_numpy().astype(float)
    h1 = (g["h1_home_pts"] + g["h1_away_pts"]).to_numpy().astype(float)
    out["half1_share"] = float(np.mean(h1 / tot))
    gref = REF.load_gate_targets(season)
    out["act_half1_share"] = float(REF.gate_target_value(gref, "season", "all", "period_share_1h_mean"))
    # --- late-game reported lines ---------------------------------------
    reach = g["home_margin_120"].notna().to_numpy()
    m120 = g["home_margin_120"].abs().to_numpy()
    close = reach & (m120 <= 6)
    out["n_sims_close_at_120"] = int(close.sum())
    out["share_close_at_120"] = float(close.mean())
    fta_end = np.where(ot, g["fta_reg_end_if_ot"].to_numpy(),
                       (g["home_fta"] + g["away_fta"]).to_numpy())
    f2 = fta_end - g["fta_at_120"].to_numpy()
    out["final2_fta_per_game"] = float(np.nanmean(f2[reach]))
    out["final2_fta_per_game_close"] = float(np.nanmean(f2[close]))
    out["window_poss_per_sim"] = float(g["window_poss"].mean())
    out["window_poss_per_close_sim"] = float(g["window_poss"].to_numpy()[close].mean())
    c, e = run["clk"], run["ev"]
    dur = {}
    for ri, rn in enumerate(ROLES):
        for bi, bn in enumerate(BUCKETS):
            n = c[ri, bi, 0]
            dur[f"{rn}|{bn}"] = {"n": int(n), "intended": float(c[ri, bi, 1] / n) if n else None,
                                 "consumed": float(c[ri, bi, 2] / n) if n else None}
    out["window_duration"] = dur
    ev = {}
    for ri, rn in enumerate(ROLES):
        n = e[ri, :, 0].sum()
        ev[rn] = {"n_first_chances": int(n),
                  "bonus_ft_rate": float(e[ri, :, 1].sum() / n) if n else None,
                  "three_share_fga": float(e[ri, :, 2].sum() / e[ri, :, 3].sum()) if n else None}
    out["window_event_by_role"] = ev
    if ev["leading"]["bonus_ft_rate"] is not None and ev["trailing"]["bonus_ft_rate"] is not None:
        out["split_bonus_ft_lead_minus_trail"] = ev["leading"]["bonus_ft_rate"] - ev["trailing"]["bonus_ft_rate"]
        out["split_three_trail_minus_lead"] = ev["trailing"]["three_share_fga"] - ev["leading"]["three_share_fga"]
    return out


def paired_boot(ref: pd.DataFrame, arm: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> dict:
    """Game-block bootstrap SE of the paired delta in P0, P1 and P0/P1."""
    def per_game(g):
        ot = (g["n_periods"] > 2).to_numpy()
        fm = (g["home_pts"] - g["away_pts"]).abs().to_numpy()
        d = pd.DataFrame({"game_id": g["game_id"].to_numpy(), "p0": ot.astype(float),
                          "p1": ((~ot) & (fm == 1)).astype(float)})
        return d.groupby("game_id")[["p0", "p1"]].sum()
    a, b = per_game(ref), per_game(arm)
    b = b.loc[a.index]
    rng = np.random.default_rng(seed)
    k = len(a)
    A, B = a.to_numpy(), b.to_numpy()
    deltas = []
    for _ in range(n_boot):
        ix = rng.integers(0, k, k)
        sa, sb = A[ix].sum(0), B[ix].sum(0)
        deltas.append([(sb[0] - sa[0]) / (k * 25), (sb[1] - sa[1]) / (k * 25),
                       sb[0] / max(sb[1], 1) - sa[0] / max(sa[1], 1)])
    d = np.array(deltas)
    return {"se_dP0": float(d[:, 0].std()), "se_dP1": float(d[:, 1].std()),
            "se_dratio": float(d[:, 2].std())}


def first_half_identical(ref: pd.DataFrame, arm: pd.DataFrame) -> dict:
    cols = ["h1_home_pts", "h1_away_pts", "h1_poss"]
    m = ref[["game_id", "seed"] + cols].merge(arm[["game_id", "seed"] + cols],
                                              on=["game_id", "seed"], suffixes=("_r", "_a"))
    bad = np.zeros(len(m), dtype=bool)
    for c in cols:
        bad |= m[f"{c}_r"].to_numpy() != m[f"{c}_a"].to_numpy()
    return {"n_sims": int(len(m)), "n_mismatch": int(bad.sum()), "identical": bool(not bad.any())}


VETO_LINES = {
    "G1_poss_mean": ("act_poss_mean", "abs_err"),
    "G1_poss_sd": ("act_poss_sd", "abs_err"),
    "G5_margin_sd_ratio": (1.0, "abs_err"),
    "G5_total_sd_ratio": (1.0, "abs_err"),
    "G9_margin_bias": (0.0, "abs_err"),
    "G9_total_bias": (0.0, "abs_err"),
    "half1_share": ("act_half1_share", "abs_err"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, F = load(a.ref), load(a.floor)
    LR, LF = lines(R), lines(F)
    floors = {k: abs(LF[k] - LR[k]) for k in
              ["P0", "P1", "P0_over_P1"] + list(VETO_LINES)}
    rows = [LR, LF]
    verdicts = []
    for tag in a.arms:
        A = load(tag)
        LA = lines(A)
        rows.append(LA)
        v = {"tag": tag, "arm": LA["arm"]}
        for k in ("P0", "P1", "P0_over_P1"):
            v[f"d_{k}"] = LA[k] - LR[k]
            v[f"floors_{k}"] = (LA[k] - LR[k]) / floors[k] if floors[k] else None
        v["ot_in_band"] = OT_BAND[0] <= LA["ot_rate"] <= OT_BAND[1]
        v["ratio_ge_1"] = LA["P0_over_P1"] >= 1.0
        v.update(paired_boot(R["g"], A["g"]))
        v["first_half"] = first_half_identical(R["g"], A["g"])
        vetoes = {}
        for k, (tgt, _) in VETO_LINES.items():
            t_r = LR[tgt] if isinstance(tgt, str) else tgt
            t_a = LA[tgt] if isinstance(tgt, str) else tgt
            er, ea = abs(LR[k] - t_r), abs(LA[k] - t_a)
            vetoes[k] = {"ref": LR[k], "arm": LA[k], "target": t_a, "abs_err_ref": er,
                         "abs_err_arm": ea, "floor": floors[k],
                         "worse_in_floors": (ea - er) / floors[k] if floors[k] else None,
                         "pass": bool(ea - er <= floors[k])}
        vetoes["first_half_bit_identical"] = {"pass": v["first_half"]["identical"]}
        v["vetoes"] = vetoes
        v["all_vetoes_pass"] = all(x["pass"] for x in vetoes.values())
        verdicts.append(v)
    ref_fh = first_half_identical(R["g"], F["g"])
    out = {"ref": a.ref, "floor": a.floor, "floors": floors, "lines": rows,
           "verdicts": verdicts, "target": {"P0_over_P1": TARGET_RATIO, "ot": TARGET_OT,
                                            "ot_band": OT_BAND},
           "power": {"floor_ratio": floors["P0_over_P1"], "ref_reg_ties": LR["n_reg_ties"],
                     "adequate": bool(floors["P0_over_P1"] <= 0.10 and LR["n_reg_ties"] >= 300)},
           "sanity_ref_vs_floor_first_half": ref_fh}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    hdr = f"{'tag':28s} {'P0':>7s} {'P1':>7s} {'P0/P1':>7s} {'fl':>6s} {'G1m':>7s} {'G1sd':>6s} {'G5m':>6s} {'G5t':>6s} {'G9m':>7s} {'G9t':>7s} {'h1sh':>6s} {'f2FTA':>6s} vetoes"
    print(hdr)
    for L in rows:
        v = next((x for x in verdicts if x["tag"] == L["tag"]), None)
        fl = f"{v['floors_P0_over_P1']:+.2f}" if v else "  --"
        vt = ("PASS" if v["all_vetoes_pass"] else "FAIL:" + ",".join(
            k for k, x in v["vetoes"].items() if not x["pass"])) if v else ""
        print(f"{L['tag']:28s} {L['P0']:.4f} {L['P1']:.4f} {L['P0_over_P1']:7.3f} {fl:>6s} "
              f"{L['G1_poss_mean']:7.3f} {L['G1_poss_sd']:6.3f} {L['G5_margin_sd_ratio']:6.3f} "
              f"{L['G5_total_sd_ratio']:6.3f} {L['G9_margin_bias']:+7.3f} {L['G9_total_bias']:+7.3f} "
              f"{L['half1_share']:6.4f} {L['final2_fta_per_game']:6.2f} {vt}")
    print("floors:", {k: round(x, 5) for k, x in floors.items()})
    print("power:", out["power"], "ref-vs-floor first half:", ref_fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
