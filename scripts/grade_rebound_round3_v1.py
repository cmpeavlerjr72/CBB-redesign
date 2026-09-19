#!/usr/bin/env python
"""
grade_rebound_round3_v1.py -- the ONE renderer for L3 REBOUND round 3.

Reads every cell JSON written by `scripts/train_rebound_v3_round3.py` and emits
the results tables. It applies the pre-registered decision rule
(`docs/models/rebound/experiments.md` sections 9.7 and 10.6) mechanically and
prints the verdict; it ADOPTS nothing and changes no default.

    .venv/Scripts/python.exe scripts/grade_rebound_round3_v1.py > out.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CELLS = _ROOT / "data/processed/models/rebound/round3/cells"

#: the already-published floors for this exact cell (experiments.md section 3)
PUBLISHED_SEED_SD = 6.7e-05
PUBLISHED_BLOCK_BOOTSTRAP_SE = 0.001412
REF = "A0B0C0"
#: simplicity order for the tie-break (9.7 rule 2, extended by 10.6 rule 10)
SIMPLICITY = {"A0B0C0": 0, "A1": 1, "A4": 2, "A2_h60": 3, "A2_h120": 3, "A2_h240": 3,
              "A3_1s": 4, "A3_2s": 4, "A5": 5, "B1": 1, "C3": 1, "C1": 2, "C2": 3,
              "D1": 2, "D2": 3, "D3": 1}


def load() -> dict:
    out = {}
    for p in sorted(CELLS.glob("*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        out[(g["stage"], g["fold"], g["arm"], g["seed"])] = g
    return out


def tbl(rows, cols):
    return "\n".join(["| " + " | ".join(cols) + " |",
                      "|" + "|".join("---" for _ in cols) + "|",
                      *["| " + " | ".join("" if r.get(c) is None else str(r.get(c, ""))
                                          for c in cols) + " |" for r in rows]])


def row_of(g: dict) -> dict:
    tq = g.get("team_quintile", {})
    return {
        "arm": g["arm"], "why": g["why"][:64],
        "n_feat": g["n_features"], "n_fits": g["n_fits"],
        "log_loss": g["log_loss"], "brier": g["brier"],
        "L1_pp": g["L1_level_pp"],
        "L2_B0_pp": g["L2_by_feed_pp"].get("B0_zero"),
        "L2_B3_pp": g["L2_by_feed_pp"].get("B3_draw"),
        "calib": "PASS" if g["calib_pass"] else "FAIL",
        "gap_pp": g["calib_worst_gap_pp"],
        "resp": "PASS" if g["resp_pass"] else "FAIL",
        "slope_q": tq.get("slope_ratio"),
        "mono": tq.get("monotone_steps"),
        "fit_s": g["fit_seconds"],
    }


def seg_row(g: dict, key: str, want: list[str]) -> dict:
    cells = g.get(key, {}).get("cells", {})
    r = {"arm": g["arm"]}
    for w in want:
        c = cells.get(w)
        r[w] = ("UNDERPOWERED" if (c and c.get("UNDERPOWERED"))
                else (c.get("level_pp") if c else "n/a"))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, help="0 = both")
    args = ap.parse_args()
    C = load()
    L: list[str] = []

    stages = [1, 2] if args.stage == 0 else [args.stage]
    for st in stages:
        for fold in ("F2", "F1"):
            keys = [k for k in C if k[0] == st and k[1] == fold and k[3] == 0]
            if not keys:
                continue
            rows = sorted((row_of(C[k]) for k in keys), key=lambda r: r["log_loss"])
            sch = C[keys[0]]["scheme"]
            L.append(f"\n**Stage {st} / {fold} / `{sch}`"
                     + (" (SELECTION)" if fold == "F2" else "") + "**\n")
            L.append(tbl(rows, ["arm", "n_feat", "n_fits", "log_loss", "brier", "L1_pp",
                                "L2_B0_pp", "L2_B3_pp", "calib", "gap_pp", "resp",
                                "slope_q", "mono", "fit_s", "why"]))

    # --- floors ------------------------------------------------------------
    L.append("\n### Noise floor\n")
    for st in stages:
        a = C.get((st, "F2", REF, 0))
        b = C.get((st, "F2", REF, 1))
        if a and b:
            sp = abs(a["log_loss"] - b["log_loss"])
            lv = abs(a["L1_level_pp"] - b["L1_level_pp"])
            L.append(f"- stage {st}, `{REF}` second-seed retrain on F2: log-loss spread "
                     f"**{sp:.6f}**, L1 spread {lv:.4f} pp. Operative floor = max(spread, "
                     f"published 5-seed SD {PUBLISHED_SEED_SD}) = "
                     f"**{max(sp, PUBLISHED_SEED_SD):.6f}**.")
        else:
            L.append(f"- stage {st}: second-seed retrain of `{REF}` NOT RUN.")
    L.append(f"- game-clustered block-bootstrap SE for this cell (published, section 3): "
             f"{PUBLISHED_BLOCK_BOOTSTRAP_SE}.")

    # --- decision ----------------------------------------------------------
    for st in stages:
        ref = C.get((st, "F2", REF, 0))
        if ref is None:
            continue
        s1 = C.get((st, "F2", REF, 1))
        floor = max(abs(ref["log_loss"] - s1["log_loss"]) if s1 else 0.0, PUBLISHED_SEED_SD)
        L.append(f"\n### Decision rule applied mechanically, stage {st} "
                 f"(floor {floor:.6f})\n")
        rows = []
        for (stg, fold, arm, seed), g in sorted(C.items()):
            if stg != st or fold != "F2" or seed != 0 or arm == REF:
                continue
            gain = ref["log_loss"] - g["log_loss"]
            f1 = C.get((st, "F1", arm, 0))
            ref1 = C.get((st, "F1", REF, 0))
            gain1 = (ref1["log_loss"] - f1["log_loss"]) if (f1 and ref1) else None
            tq_ref = ref.get("team_quintile", {}).get("slope_ratio")
            tq = g.get("team_quintile", {}).get("slope_ratio")
            seg_ok = True
            for lab in ("Nov-Dec", "Jan", "Feb-Apr"):
                a = ref.get(f"team_quintile_{lab}", {}).get("slope_ratio")
                b = g.get(f"team_quintile_{lab}", {}).get("slope_ratio")
                if a is not None and b is not None and b < a - 1e-9:
                    seg_ok = False
            rows.append({
                "arm": arm,
                "F2 gain": round(gain, 6), "x floor": round(gain / floor, 1),
                "F1 gain": (round(gain1, 6) if gain1 is not None else "NOT RUN"),
                "rule1 (beats ref)": gain > floor,
                "rule3 (gates)": bool(g["calib_pass"] and g["resp_pass"]),
                "rule4 (L1 not worse)": abs(g["L1_level_pp"]) <= abs(ref["L1_level_pp"]) + 1e-9,
                "L1_pp": g["L1_level_pp"],
                "rule5 (slope not reduced)": seg_ok and (tq is not None and tq_ref is not None
                                                         and tq >= tq_ref - 1e-9),
                "slope_q": tq,
            })
        rows.sort(key=lambda r: -r["F2 gain"])
        L.append(tbl(rows, ["arm", "F2 gain", "x floor", "F1 gain", "rule1 (beats ref)",
                            "rule3 (gates)", "rule4 (L1 not worse)", "L1_pp",
                            "rule5 (slope not reduced)", "slope_q"]))
        elig = [r for r in rows if r["rule1 (beats ref)"] and r["rule3 (gates)"]
                and r["rule4 (L1 not worse)"] and r["rule5 (slope not reduced)"]]
        if elig:
            best = min(elig, key=lambda r: -r["F2 gain"])
            tied = [r for r in elig if best["F2 gain"] - r["F2 gain"] <= floor]
            win = min(tied, key=lambda r: SIMPLICITY.get(r["arm"], 99))
            L.append(f"\nEligible: {[r['arm'] for r in elig]}. "
                     f"**Stage-{st} leader by the rule: `{win['arm']}`** "
                     "(ties inside one floor broken by simplicity).")
        else:
            L.append(f"\n**No arm clears every rule at stage {st}.**")

    # --- segment evidence ---------------------------------------------------
    L.append("\n### Multi-level evidence (stage-2 cells if present, else stage 1)\n")
    st = 2 if any(k[0] == 2 for k in C) else 1
    keys = sorted([k for k in C if k[0] == st and k[1] == "F2" and k[3] == 0],
                  key=lambda k: C[k]["log_loss"])
    if keys:
        for name, key, want in (
            ("by miss type (level pp)", "by_miss_type", ["rim", "jump2", "three", "ft"]),
            ("by month (level pp)", "by_month", ["11", "12", "1", "2", "3", "4"]),
            ("by site (level pp)", "by_site", ["home", "away", "neutral"]),
            ("by conference game (level pp)", "by_conf", ["conf", "nonconf"]),
            ("by period (level pp)", "by_period", ["1", "2", "3", "4", "5"]),
        ):
            L.append(f"\n{name}\n")
            L.append(tbl([seg_row(C[k], key, want) for k in keys], ["arm"] + want))
        L.append("\nper-team prior-quintile slope ratio, by season segment "
                 "(the G4 responsiveness defect; served engine 0.561 / 0.799 / 0.738)\n")
        rr = []
        for k in keys:
            g = C[k]
            rr.append({"arm": g["arm"],
                       "all": g.get("team_quintile", {}).get("slope_ratio"),
                       "Nov-Dec": g.get("team_quintile_Nov-Dec", {}).get("slope_ratio"),
                       "Jan": g.get("team_quintile_Jan", {}).get("slope_ratio"),
                       "Feb-Apr": g.get("team_quintile_Feb-Apr", {}).get("slope_ratio"),
                       "gap_pp_by_q": g.get("team_quintile", {}).get("gap_pp_by_q")})
        L.append(tbl(rr, ["arm", "all", "Nov-Dec", "Jan", "Feb-Apr", "gap_pp_by_q"]))
        L.append("\nper-game level error (pp) and the first four weeks of conference play\n")
        rr = []
        for k in keys:
            g = C[k]
            pg = g["per_game"]
            rr.append({"arm": g["arm"], "n_games": pg["n_games"], "mean": pg["mean_pp"],
                       "median": pg["median_pp"], "sd": pg["sd_pp"], "mae": pg["mae_pp"],
                       "P(pred<act)": pg["p_sim_below_actual"],
                       "conf4_n": g["conf4"]["n"],
                       "conf4_level_pp": g["conf4"].get("level_pp", "UNDERPOWERED")})
        L.append(tbl(rr, ["arm", "n_games", "mean", "median", "sd", "mae", "P(pred<act)",
                          "conf4_n", "conf4_level_pp"]))

    # --- the engine feed table ---------------------------------------------
    L.append("\n### Block B: the `blocked_f` engine feed (all feeds, same trained model)\n")
    for st2 in stages:
        g = C.get((st2, "F2", REF, 0))
        if not g:
            continue
        rr = [{"feed": k, "level_pp": v,
               "log_loss": g["log_loss_by_feed"].get(k)}
              for k, v in g["L2_by_feed_pp"].items()]
        L.append(f"\nstage {st2}, `{REF}` on F2 (actual live OREB "
                 f"{g['actual_live_oreb']})\n")
        L.append(tbl(rr, ["feed", "level_pp", "log_loss"]))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
