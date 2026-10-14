"""diag_ca1_trailer_slope_v1.py -- LABELLED DIAGNOSTIC, not a selection read (PM ruling 2026-10-14 on clock s47).

CA1 (s46 block A ratio cell law) won both block-A primaries but failed the team-prior trailer-slope gate
(`slope_team_trail` 0.025 -> -0.120, floor 0.084) and overshot the z_off / defence-z quintile slopes (2.10 / 3.04
vs control 0.688 / 0.786). This script answers, offline on the period-2 window rows of `grade_endgame_clock_v2.py`
(the SAME truth frame, predictions and quintile definitions, imported unchanged):

  (a) the ACTUAL trailer first-chance-seconds response to offence prior strength (`off_rating_off_c`, quintiles over
      all period-2 window rows, as the gate): per-quintile means (actual, control CA0, CA1), the actual's own slope
      (seconds per rating point, OLS of quintile means on quintile mean strength) and Q5 - Q1, with floors; plus the
      gate's form (model quintile means on actual quintile means) for control and CA1.
  (b) per (trailer band x time bucket) cell: the cell's contribution to the trailer Q5 - Q1 difference
      (share(Q5, c) x mean(Q5, c) - share(Q1, c) x mean(Q1, c)), actual / control / CA1, and the within-cell
      strength slope; cells < 3,000 rows are UNDERPOWERED for slopes (s46 3).
  (c) a between-cell (composition) / within-cell decomposition of the quintile spread (Q5 - Q1) for the z_off,
      defence-z and trailer-strength quintiles: between = sum_c (share(Q5,c) - share(Q1,c)) x mean_all(c) [the
      actual cell mean, the same weights for every source], within = the rest. Shows whether the overshoots share
      one mechanism.
Floors: 2 x SD of the game-block bootstrap (200 draws, numpy seed 12345), per line; arm-minus-control lines paired.

    set CBB_TRUTH=verified_v1
    .venv/Scripts/python.exe scripts/diag_ca1_trailer_slope_v1.py
Writes results/endgame_clock_v2/diag_ca1/offline_diag.json and offline_diag_tables.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("CBB_TRUTH", "verified_v1")

import grade_endgame_clock_v2 as GC  # noqa: E402

OUT = ROOT / "results/endgame_clock_v2/diag_ca1"
SRC = ("act", "CA0", "CA1")
W = GC.W
EG = GC.EG


def build(fold: str):
    t = GC.truth_A(fold)
    e = {}
    for arm in ("CA0", "CA1"):
        P = GC.load_pred("A", fold, arm, t)
        e[arm] = GC.rows_A(t, P)["e"]
    sel = t["period"].to_numpy() == 2
    t = t[sel].reset_index(drop=True)
    vals = {"act": t["y"].to_numpy(float), "CA0": e["CA0"][sel], "CA1": e["CA1"][sel]}
    return t, vals


def per_game(t, vals, key, K):
    g, gi = np.unique(t["game_id"].to_numpy(), return_inverse=True)
    G = len(g)
    one = np.ones(len(t))
    S = {"n": EG.gsum(gi, key, one, G, K)}
    for s in SRC:
        S[s] = EG.gsum(gi, key, vals[s], G, K)
    S["x"] = EG.gsum(gi, key, t["strength"].to_numpy(float), G, K)
    return S, G


def boot(G):
    return EG.boot_weights(G)


def table_a(t, vals):
    """Trailers by offence-strength quintile (quintiles over all period-2 rows, as LinesA)."""
    sq = EG.quintile(t["strength"].to_numpy())
    role = W.role(t["m"].to_numpy())
    out = {}
    for lab, r in (("trail", -1), ("lead", 1)):
        key = np.where(role == r, sq, 5)
        S, G = per_game(t, vals, key, 6)

        def lines(w):
            n = w @ S["n"]
            xm = (w @ S["x"]) / n
            o = {}
            for s in SRC:
                m = (w @ S[s]) / n
                o[f"{s}_q"] = m[:5]
                o[f"{s}_slope_per_pt"] = EG.ols_slope(xm[:5], m[:5])
                o[f"{s}_q5_q1"] = float(m[4] - m[0])
            for s in ("CA0", "CA1"):
                o[f"{s}_gate_slope"] = EG.ols_slope(o["act_q"], o[f"{s}_q"])
            o["n_q"] = n[:5]
            return o
        base = lines(np.ones(G))
        B = [lines(w) for w in boot(G)]
        fl = {}
        for k in ("act_slope_per_pt", "act_q5_q1", "CA0_slope_per_pt", "CA1_slope_per_pt", "CA0_q5_q1", "CA1_q5_q1",
                  "CA0_gate_slope", "CA1_gate_slope"):
            fl[k] = 2 * float(np.nanstd([b[k] for b in B]))
        for s in ("CA0", "CA1"):
            fl[f"{s}_minus_act_slope_per_pt"] = 2 * float(np.nanstd([b[f"{s}_slope_per_pt"] - b["act_slope_per_pt"] for b in B]))
        fl["CA1_minus_CA0_gate_slope"] = 2 * float(np.nanstd([b["CA1_gate_slope"] - b["CA0_gate_slope"] for b in B]))
        out[lab] = {"lines": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in base.items()},
                    "floors": fl, "strength_q_mean": ((S["x"].sum(0) / S["n"].sum(0))[:5]).tolist()}
    return out


def decomposition(t, vals, qv, rows_mask, label):
    """Between-cell / within-cell split of Q5 - Q1 of mean seconds (quintile of qv over all period-2 rows)."""
    q = EG.quintile(qv)
    cell = (t["rb"].to_numpy() * W.N_TB + t["tb"].to_numpy())
    key = np.where(rows_mask, q * 45 + cell, 5 * 45)
    S, G = per_game(t, vals, key, 5 * 45 + 1)

    def lines(w):
        n = (w @ S["n"])[:225].reshape(5, 45)
        ca = (w @ S["act"])[:225].reshape(5, 45).sum(0) / np.maximum(n.sum(0), 1e-12)   # actual cell mean (pooled q)
        sh = n / np.maximum(n.sum(1, keepdims=True), 1e-12)
        o = {}
        for s in SRC:
            m = (w @ S[s])[:225].reshape(5, 45)
            qm = m.sum(1) / n.sum(1)
            tot = qm[4] - qm[0]
            betw = float(((sh[4] - sh[0]) * ca).sum())
            o[f"{s}_total"], o[f"{s}_between"], o[f"{s}_within"] = float(tot), betw, float(tot - betw)
            with np.errstate(invalid="ignore", divide="ignore"):
                cm = m / n
            o[f"{s}_contrib"] = (sh[4] * np.nan_to_num(cm[4]) - sh[0] * np.nan_to_num(cm[0]))
            o[f"{s}_cellq"] = cm
        o["n"] = n
        return o
    base = lines(np.ones(G))
    B = [lines(w) for w in boot(G)]
    fl = {}
    for s in SRC:
        for p in ("total", "between", "within"):
            fl[f"{s}_{p}"] = 2 * float(np.nanstd([b[f"{s}_{p}"] for b in B]))
    for s in ("CA0", "CA1"):
        for p in ("total", "within"):
            fl[f"{s}_minus_act_{p}"] = 2 * float(np.nanstd([b[f"{s}_{p}"] - b[f"act_{p}"] for b in B]))
    cfl = {s: 2 * np.nanstd(np.array([b[f"{s}_contrib"] for b in B]), axis=0) for s in SRC}
    cfl["CA1_minus_CA0"] = 2 * np.nanstd(np.array([b["CA1_contrib"] - b["CA0_contrib"] for b in B]), axis=0)
    return {"label": label, "base": base, "floors": fl, "contrib_floors": cfl}


def cells_trailer(t, vals, dec):
    """Per trailer (band x time bucket) cell: contribution to the trailer Q5 - Q1 and the within-cell slope."""
    b, fl = dec["base"], dec["contrib_floors"]
    rows = []
    for rb in range(4, 8):
        for tb in range(5):
            c = rb * W.N_TB + tb
            n = b["n"][:, c]
            ntot = int(n.sum())
            r = {"cell": f"{W.ROLEBAND_LABELS[rb]} | {W.BUCKET_LABELS[tb]}", "n": ntot,
                 "n_q1": int(n[0]), "n_q5": int(n[4]), "slope_power": "ok" if ntot >= GC.MIN_SLOPE else "UNDERPOWERED"}
            for s in SRC:
                r[f"contrib_{s}"] = float(b[f"{s}_contrib"][c])
                r[f"contrib_floor_{s}"] = float(fl[s][c])
                cq = b[f"{s}_cellq"][:, c]
                r[f"q1_{s}"], r[f"q5_{s}"] = float(cq[0]), float(cq[4])
                r[f"within_q5_q1_{s}"] = float(cq[4] - cq[0]) if n[0] > 0 and n[4] > 0 else None
            r["contrib_floor_CA1_minus_CA0"] = float(fl["CA1_minus_CA0"][c])
            rows.append(r)
    return rows


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"label": "DIAGNOSTIC, not a selection read (PM ruling 2026-10-14)", "folds": {}}
    md = ["# CA1 trailer-slope diagnostic, offline tables (DIAGNOSTIC, not a selection read)", ""]
    for fold in ("F2", "F1"):
        t, vals = build(fold)
        A = table_a(t, vals)
        role = W.role(t["m"].to_numpy())
        dz = decomposition(t, vals, t["zoff"].to_numpy(), np.ones(len(t), bool), "z_off quintile (all period-2 rows)")
        dd = decomposition(t, vals, t["defz"].to_numpy(), np.ones(len(t), bool), "defence-z quintile (all rows)")
        dt = decomposition(t, vals, t["strength"].to_numpy(), role < 0, "offence strength quintile, trailers")
        dl = decomposition(t, vals, t["strength"].to_numpy(), role > 0, "offence strength quintile, leaders")
        cells = cells_trailer(t, vals, dt)
        rep["folds"][fold] = {"n_rows_p2": int(len(t)), "a": A, "cells_trailer": cells,
                              "decomp": {d["label"]: {"base": {k: v for k, v in d["base"].items() if k.endswith(("total", "between", "within"))},
                                                      "floors": d["floors"]} for d in (dz, dd, dt, dl)}}
        a = A["trail"]
        L, F = a["lines"], a["floors"]
        md += [f"## {fold}: period-2 window rows {len(t):,}", "", "### (a) trailer first-chance seconds by offence prior-strength quintile",
               "", "| quintile | strength mean | rows | actual | control CA0 | CA1 |", "|---|---:|---:|---:|---:|---:|"]
        for k in range(5):
            md.append(f"| Q{k + 1} | {a['strength_q_mean'][k]:+.2f} | {int(L['n_q'][k]):,} | {L['act_q'][k]:.3f} | "
                      f"{L['CA0_q'][k]:.3f} | {L['CA1_q'][k]:.3f} |")
        md += ["", "| line | actual | control CA0 | CA1 |", "|---|---|---|---|",
               f"| seconds per rating point (floor) | {L['act_slope_per_pt']:+.4f} ({F['act_slope_per_pt']:.4f}) | "
               f"{L['CA0_slope_per_pt']:+.4f} ({F['CA0_slope_per_pt']:.4f}); minus actual floor {F['CA0_minus_act_slope_per_pt']:.4f} | "
               f"{L['CA1_slope_per_pt']:+.4f} ({F['CA1_slope_per_pt']:.4f}); minus actual floor {F['CA1_minus_act_slope_per_pt']:.4f} |",
               f"| Q5 - Q1 seconds (floor) | {L['act_q5_q1']:+.3f} ({F['act_q5_q1']:.3f}) | {L['CA0_q5_q1']:+.3f} ({F['CA0_q5_q1']:.3f}) | "
               f"{L['CA1_q5_q1']:+.3f} ({F['CA1_q5_q1']:.3f}) |",
               f"| gate form: model q-means on actual q-means (floor) | 1 (target) | {L['CA0_gate_slope']:+.3f} ({F['CA0_gate_slope']:.3f}) | "
               f"{L['CA1_gate_slope']:+.3f} ({F['CA1_gate_slope']:.3f}); CA1 - CA0 floor {F['CA1_minus_CA0_gate_slope']:.3f} |", ""]
        la, fa = A["lead"]["lines"], A["lead"]["floors"]
        md += [f"Leaders, same lines: actual {la['act_slope_per_pt']:+.4f} s/pt ({fa['act_slope_per_pt']:.4f}), control "
               f"{la['CA0_slope_per_pt']:+.4f}, CA1 {la['CA1_slope_per_pt']:+.4f}; gate form control {la['CA0_gate_slope']:+.3f}, CA1 "
               f"{la['CA1_gate_slope']:+.3f}.", "",
               "### (b) trailer cells: contribution to the trailer Q5 - Q1 (seconds; floor) and within-cell Q5 - Q1", "",
               "| cell | rows (Q1/Q5) | contrib actual | contrib CA0 | contrib CA1 | within actual | within CA0 | within CA1 | power |",
               "|---|---|---|---|---|---:|---:|---:|---|"]
        for r in cells:
            md.append(f"| {r['cell']} | {r['n']:,} ({r['n_q1']}/{r['n_q5']}) | {fmt(r['contrib_act'])} ({r['contrib_floor_act']:.3f}) | "
                      f"{fmt(r['contrib_CA0'])} ({r['contrib_floor_CA0']:.3f}) | {fmt(r['contrib_CA1'])} ({r['contrib_floor_CA1']:.3f}) | "
                      f"{fmt(r['within_q5_q1_act'], 2)} | {fmt(r['within_q5_q1_CA0'], 2)} | {fmt(r['within_q5_q1_CA1'], 2)} | {r['slope_power']} |")
        md += ["", "### (c) Q5 - Q1 decomposition: between-cell (composition at actual cell means) / within-cell (floor)", "",
               "| quintile line | source | total | between | within |", "|---|---|---|---|---|"]
        for d in (dz, dd, dt, dl):
            for s in SRC:
                b, fl = d["base"], d["floors"]
                md.append(f"| {d['label']} | {s} | {b[f'{s}_total']:+.3f} ({fl[f'{s}_total']:.3f}) | "
                          f"{b[f'{s}_between']:+.3f} ({fl[f'{s}_between']:.3f}) | {b[f'{s}_within']:+.3f} ({fl[f'{s}_within']:.3f}) |")
        md.append("")
    (OUT / "offline_diag.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)),
                                           encoding="utf-8")
    (OUT / "offline_diag_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
