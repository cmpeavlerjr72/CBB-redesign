"""diag_late_foul_accrual_v1.py -- late foul accrual: where the trailing team's fouls go missing (reported lines).

Actual: `possession_outcome/round6/foul_accrual_poss_v2.parquet` (engine-definition team-foul counts, the
round-7/9 training table), seasons 2024 (fold-1 test) and 2025 (fold-2 test). Sim: a round-4 tap run with a
300 s possession log (defence fouls committed in a possession = the same team's count at the next possession
of the same period minus its count at this one). Served accrual LUT (`lut_acc_A2_F2`) evaluated on the
ACTUAL rows to show what the served table predicts there and whether those rows were in its fit window.

    .venv/Scripts/python.exe scripts/diag_late_foul_accrual_v1.py --run lg7_R9_s25 --out results/late_game/round7/foul_diag.json
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
RES = ROOT / "results/engine_v0"
ACC = ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet"
LUT = ROOT / "data/processed/models/possession_outcome/round7/lut_acc_A2_F2.npz"
CLK = ((-1, 30, "(0,30]"), (30, 60, "(30,60]"), (60, 120, "(60,120]"), (120, 300, "(120,300]"), (300, 9999, ">300"))
BONUS = 6   # prior team fouls at which the opponent shoots bonus FTs (engine-definition count; train_foul_joint_v1)


def role_tier(om):
    """DEFENCE perspective: the defence trails when the offence leads."""
    a = np.abs(om)
    tier = np.where(a == 0, "tied", np.where(a <= 3, "1-3", np.where(a <= 6, "4-6", "7+")))
    side = np.where(om > 0, "def_trailing", np.where(om < 0, "def_leading", "tied"))
    return np.where(side == "tied", "tied", np.char.add(np.char.add(side.astype(str), "|"), tier.astype(str)))


def lut_p(per, sec, margin, dfl, ofl, site):
    t = np.load(LUT)
    pi = np.where(per <= 1, 0, np.where(per == 2, 1, 2))
    ci = np.searchsorted(t["clock_cuts"], sec, side="right")
    mi = np.searchsorted(t["margin_cuts"], margin, side="right")
    m = int(t["max_fouls"])
    return t["lut"][pi, ci, mi, np.clip(dfl, 0, m), np.clip(ofl, 0, m), site]


def act_frame(season):
    d = pd.read_parquet(ACC)
    d = d[(d["season"] == season) & (d["period"] <= 2)]
    site = np.where(d["neutral_site"], 0, np.where(d["offense_is_home"], 1, 2))
    return pd.DataFrame({"game_id": d["game_id"].to_numpy(), "seed": -1, "per": d["period"].to_numpy(),
                         "sec": d["start_clock"].to_numpy(), "om": d["start_score_diff"].to_numpy(),
                         "dfl": d["def_team_fouls_true"].to_numpy(), "ofl": d["off_team_fouls_true"].to_numpy(),
                         "nontrip": d["def_silent"].to_numpy(), "trip": d["def_trip"].to_numpy(),
                         "in_fit": d["in_fit_window"].to_numpy(),
                         "p_lut": lut_p(d["period"].to_numpy(), d["start_clock"].to_numpy(), d["start_score_diff"].to_numpy(),
                                        d["def_team_fouls_true"].to_numpy(), d["off_team_fouls_true"].to_numpy(), site)})


def sim_frame(tag):
    t = pd.read_parquet(RES / tag / "tap_poss.parquet").sort_values(["game_id", "seed", "per", "sec"],
                                                                        ascending=[True, True, True, False])
    t = t.reset_index(drop=True)
    sgn = np.where(t["off"] == 0, 1, -1)
    om = (t["hp"] - t["ap"]).to_numpy() * sgn
    nxt_same = ((t["game_id"].shift(-1) == t["game_id"]) & (t["seed"].shift(-1) == t["seed"])
                & (t["per"].shift(-1) == t["per"])).to_numpy()
    committed = np.where(nxt_same, t["tf_off"].shift(-1).fillna(0).to_numpy() - t["tf_def"].to_numpy(), np.nan)
    trips = (t["d_fta"].to_numpy() > 0).astype(int)
    return pd.DataFrame({"game_id": t["game_id"], "seed": t["seed"], "per": t["per"], "sec": t["sec"], "om": om,
                         "dfl": t["tf_def"], "ofl": t["tf_off"], "committed": committed,
                         "nontrip": np.where(np.isnan(committed), np.nan, np.clip(committed - trips, 0, None)),
                         "trip": trips, "bon": t["bon"]})


def accrual_table(f: pd.DataFrame, max_sec: int) -> dict:
    f = f[f["sec"] <= max_sec]
    rt = role_tier(f["om"].to_numpy())
    out = {}
    for per in (1, 2):
        for lo, hi, cn in CLK:
            if lo >= max_sec:
                continue
            for r in np.unique(rt):
                m = (f["per"] == per).to_numpy() & (f["sec"] > lo).to_numpy() & (f["sec"] <= hi).to_numpy() & (rt == r)
                x = f[m]
                n = int(len(x))
                if n < 50:
                    continue
                nt = x["nontrip"].dropna()
                row = {"n": n, "label": "UNDERPOWERED" if n < 200 else "",
                       "nontrip_mean": float(nt.mean()), "p_nontrip_ge1": float((nt >= 1).mean()),
                       "p_nontrip_ge2": float((nt >= 2).mean()), "trip_rate": float(x["trip"].mean())}
                if "p_lut" in x:
                    row["served_lut_p_ge1"] = float(x["p_lut"].mean())
                    row["share_in_fit_window"] = float(x["in_fit"].mean())
                out[f"H{per}|{cn}|{r}"] = row
    return out


def fouls_at(f: pd.DataFrame) -> dict:
    """H2: the TRAILING team's team fouls at the first possession starting <= T, by margin tier; and whether the
    leader would be in the bonus (trailing team's count >= BONUS)."""
    f = f[f["per"] == 2].sort_values(["game_id", "seed", "sec"], ascending=[True, True, False])
    out = {}
    for T in (240, 120, 60, 30):
        x = f[f["sec"] <= T].groupby(["game_id", "seed"]).head(1)
        om = x["om"].to_numpy()
        trail = np.where(om > 0, x["dfl"].to_numpy(), np.where(om < 0, x["ofl"].to_numpy(), np.nan))
        a = np.abs(om)
        for nm, lo, hi in (("1-3", 1, 3), ("4-6", 4, 6), ("7-10", 7, 10)):
            m = (a >= lo) & (a <= hi)
            v = trail[m]
            out[f"T{T}|{nm}"] = {"n": int(m.sum()), "trail_fouls_mean": float(np.nanmean(v)) if m.any() else None,
                                 "leader_in_bonus": float(np.nanmean(v >= BONUS)) if m.any() else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    act = {s: act_frame(s) for s in (2024, 2025)}
    sim = sim_frame(a.run)
    out = {"run": a.run,
           "accrual_act_2025": accrual_table(act[2025], 300), "accrual_act_2024": accrual_table(act[2024], 300),
           "accrual_sim": accrual_table(sim, 300),
           "fouls_at_act_2025": fouls_at(act[2025]), "fouls_at_act_2024": fouls_at(act[2024]), "fouls_at_sim": fouls_at(sim)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("non-trip defensive fouls per possession: mean / P(>=1) / P(>=2) | served LUT P(>=1) on actual rows, share in its fit window")
    for k, r in out["accrual_act_2025"].items():
        s = out["accrual_sim"].get(k)
        r4 = out["accrual_act_2024"].get(k, {})
        if not (k.startswith("H2|(0,30]") or k.startswith("H2|(30,60]") or k.startswith("H2|(60,120]") or k.startswith("H2|(120,300]") or k.startswith("H1|(0,30]")):
            continue
        ss = f"{s['nontrip_mean']:.3f}/{s['p_nontrip_ge1']:.3f}/{s['p_nontrip_ge2']:.3f} n={s['n']}" if s else "-"
        print(f" {k:34s} act25 {r['nontrip_mean']:.3f}/{r['p_nontrip_ge1']:.3f}/{r['p_nontrip_ge2']:.3f} n={r['n']:5d} "
              f"| act24 {r4.get('nontrip_mean', float('nan')):.3f} | LUT {r['served_lut_p_ge1']:.3f} fit {r['share_in_fit_window']:.2f} | sim {ss}")
    print("\nH2 trailing team's fouls at T (mean, leader in bonus): sim | act 2025 | act 2024")
    for k in out["fouls_at_sim"]:
        s, x, y = out["fouls_at_sim"][k], out["fouls_at_act_2025"][k], out["fouls_at_act_2024"][k]
        print(f" {k:10s} {s['trail_fouls_mean']:.2f} {s['leader_in_bonus']:.3f} (n {s['n']}) | {x['trail_fouls_mean']:.2f} {x['leader_in_bonus']:.3f} (n {x['n']}) | {y['trail_fouls_mean']:.2f} {y['leader_in_bonus']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
