"""
grade_po4b_closed_loop.py -- the possession-outcome round-5 SHIP GATE lines
that `eval_gates.py` does not compute.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 12.4.
`eval_gates.py` scores G1-G9 and `diag_pair_gate_reports.py` pairs the reports
against the seed-offset floor run. This script adds, on the same four runs:

  * **the PM's responsiveness line** -- the team-prior quintile slope, read
    CLOSED-LOOP. Team-games are bucketed by the OFFENCE team's own as-of driver
    value taken from the SERVED round-2 team block, which is identical in every
    arm, so all four runs are cut on exactly the same rows;
  * the pre-registered SEGMENT cells (weeks 0-3 / 4-7 / 8+, conference /
    non-conference), with n and an UNDERPOWERED label;
  * per-game paired deltas, per-team bias, and the per-possession shot/turnover
    mix against the event layer's own actuals.

One path, every arm, the arm's identity read from `run_meta.json` and used only
to label the row. Nothing here adjusts anything.

    .venv/Scripts/python.exe scripts/grade_po4b_closed_loop.py \
        --ref po4b_R_s25 --floor po4b_R_s25_floor --arms po4b_G2_s25 po4b_G3_s25
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
from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

RESULTS = Path("results/engine_v0")
ENGINE_DIR = Path("data/processed/models/engine")
#: Decision 8's own driver/class pairs, imported rather than restated.
DRIVERS = PO.RESPONSIVENESS_SPECS
#: driver -> (sim numerator column, actual numerator column AS RENAMED by
#: `actual_team_long`, which source the actual comes from)
DRIVER_STATS = {
    "off_3pa_c":  ("fga3",      "a_tpa",      "box"),
    "off_rim_c":  ("fga2_rim",  "ev_fga_rim", "event"),
    "off_tov_c":  ("tov",       "a_tov",      "box"),
}
MIN_CELL_N = 300          # docs/gates.yaml min_cell_n
MIN_SPAN_PP = 2.0         # Decision 8's narrow-span exemption


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def engine_games(fold: str = "F2", season: int = 2025) -> pd.DataFrame:
    g = pd.read_parquet(ENGINE_DIR / f"games_{fold}_{season}.parquet")
    g["game_date"] = pd.to_datetime(g["game_date"])
    return g


def served_team_block(fold: str = "F2", season: int = 2025) -> tuple[np.ndarray, dict]:
    """The SERVED round-2 team block and its name map. Every arm is bucketed on
    THIS block's raw-centred drivers, never on its own shrunk columns: the
    buckets must be the same rows in every arm or the slopes are not
    comparable."""
    d = ENGINE_DIR / f"event_round2_s1_{fold}_{season}"
    idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
    z = np.load(d / "team_block.npz")
    return z["team_block"], {c: i for i, c in enumerate(idx["team_cols"])}


def sim_team_long(res: Path, game_ids: np.ndarray, eg: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team_id): the arm's per-team counts averaged over
    seeds first WITHIN a game, so a game contributes once whatever its seed
    count."""
    raw = pd.read_parquet(res / "games.parquet")
    raw = raw[raw["game_id"].isin(game_ids)]
    ids = eg[["game_id", "home_team_id", "away_team_id"]]
    raw = raw.merge(ids, on="game_id", how="left")
    out = []
    for side, tcol in (("home", "home_team_id"), ("away", "away_team_id")):
        o = "away" if side == "home" else "home"
        out.append(pd.DataFrame({
            "game_id": raw["game_id"].to_numpy(),
            "seed": raw["seed"].to_numpy(),
            "team_id": raw[tcol].to_numpy(),
            "poss": raw["possessions"].astype(float).to_numpy(),
            "pts": raw[f"{side}_pts"].astype(float).to_numpy(),
            "opp_pts": raw[f"{o}_pts"].astype(float).to_numpy(),
            "fga3": raw[f"{side}_fga3"].astype(float).to_numpy(),
            "fga2_rim": raw[f"{side}_fga2_rim"].astype(float).to_numpy(),
            "fga2_jump": raw[f"{side}_fga2_jump"].astype(float).to_numpy(),
            "fta": raw[f"{side}_fta"].astype(float).to_numpy(),
            "tov": raw[f"{side}_tov"].astype(float).to_numpy(),
            "oreb": raw[f"{side}_oreb"].astype(float).to_numpy(),
        }))
    lo = pd.concat(out, ignore_index=True)
    num = ["poss", "pts", "opp_pts", "fga3", "fga2_rim", "fga2_jump", "fta", "tov", "oreb"]
    per = lo.groupby(["game_id", "team_id"], as_index=False)[num].mean()
    per["n_seeds"] = lo.groupby(["game_id", "team_id"]).size().to_numpy()
    return per


def actual_team_long(season: int, game_ids: np.ndarray) -> pd.DataFrame:
    box = REF.load_actual_team_box(season)
    box = box[box["game_id"].isin(game_ids)][
        ["game_id", "team_id", "poss_team", "team_score", "fga", "tpa", "tov", "fta", "oreb"]]
    shot = REF.load_team_shot_truth(season)
    if shot is not None:
        shot = shot[shot["game_id"].isin(game_ids)][
            ["game_id", "team_id", "ev_fga_rim", "ev_fga_3", "ev_fta"]]
        box = box.merge(shot, on=["game_id", "team_id"], how="left")
    return box.rename(columns={"poss_team": "a_poss", "team_score": "a_pts",
                               "tpa": "a_tpa", "tov": "a_tov", "fta": "a_fta",
                               "fga": "a_fga", "oreb": "a_oreb"})


# ---------------------------------------------------------------------------
# the responsiveness line
# ---------------------------------------------------------------------------
def responsiveness(sim: pd.DataFrame, act: pd.DataFrame, drv: pd.DataFrame,
                   n_q: int = 5) -> list[dict]:
    """Section 12.4's closed-loop reading of the matchup-specific rule.

    `slope_ratio = (sim Q5 - sim Q1) / (actual Q5 - actual Q1)`, the span-ratio
    `train_possession_outcome_v3.responsiveness_by` uses, on the SAME quintile
    cut for every arm."""
    m = sim.merge(act, on=["game_id", "team_id"], how="inner").merge(
        drv, on=["game_id", "team_id"], how="inner")
    rows = []
    for feat, cls in DRIVERS:
        stem, acol, src = DRIVER_STATS[feat]
        d = m[np.isfinite(m[feat]) & (m["a_poss"] > 0) & (m["poss"] > 0)]
        d = d[np.isfinite(d[acol])]
        if len(d) < 50:
            rows.append({"driver": feat, "class": cls, "status": "underpowered",
                         "n": int(len(d))})
            continue
        v = d[feat].to_numpy()
        edges = np.quantile(v, np.linspace(0, 1, n_q + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        q = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_q - 1)
        sim_r, act_r, ns = [], [], []
        for b in range(n_q):
            k = q == b
            sim_r.append(float((d[stem].to_numpy()[k] / d["poss"].to_numpy()[k]).mean()))
            act_r.append(float((d[acol].to_numpy()[k] / d["a_poss"].to_numpy()[k]).mean()))
            ns.append(int(k.sum()))
        span_s = sim_r[-1] - sim_r[0]
        span_a = act_r[-1] - act_r[0]
        steps = int(np.sum(np.sign(np.diff(sim_r)) == np.sign(np.diff(act_r))))
        exempt = abs(span_a) * 100.0 < MIN_SPAN_PP
        ratio = float(span_s / span_a) if abs(span_a) > 1e-9 else float("nan")
        rows.append({
            "driver": feat, "class": cls, "actual_source": src, "n": int(len(d)), "n_q": ns,
            "sim": [round(x, 6) for x in sim_r], "act": [round(x, 6) for x in act_r],
            "span_sim_pp": round(span_s * 100, 4), "span_act_pp": round(span_a * 100, 4),
            "slope_ratio": (round(ratio, 4) if np.isfinite(ratio) else None),
            "steps_with_actual": steps, "n_steps": n_q - 1,
            "exempt_narrow_span": bool(exempt),
            "status": "underpowered" if min(ns) < 50 else "scored",
        })
    return rows


def headline_slope(rows: list[dict]) -> float | None:
    """`train_possession_outcome_v4.quintile_slope_worst`'s rule: the
    non-exempt driver slope furthest from 1.0."""
    vals = [r["slope_ratio"] for r in rows
            if r.get("slope_ratio") is not None and not r.get("exempt_narrow_span", True)]
    if not vals:
        return None
    return float(min(vals, key=lambda v: -abs(v - 1.0)))


# ---------------------------------------------------------------------------
# segments, per-game, per-team, mix
# ---------------------------------------------------------------------------
def segment_table(sim: pd.DataFrame, act: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    m = sim.merge(act, on=["game_id", "team_id"], how="inner").merge(
        seg, on="game_id", how="left")
    m["e_pts"] = m["pts"] - m["a_pts"]
    m["e_poss"] = m["poss"] - m["a_poss"]
    m["ppp"] = m["pts"] / m["poss"]
    m["a_ppp"] = m["a_pts"] / m["a_poss"]
    out = []
    for name, mask in (("weeks 0-3", m["week_bucket"] == "wk0_3"),
                       ("weeks 4-7", m["week_bucket"] == "wk4_7"),
                       ("weeks 8+", m["week_bucket"] == "wk8plus"),
                       ("conference", m["is_conf_game"] == True),        # noqa: E712
                       ("non-conference", m["is_conf_game"] == False),   # noqa: E712
                       ("ALL", m["game_id"].notna())):
        d = m[mask]
        n = int(len(d))
        out.append({
            "cell": name, "n_team_games": n, "n_games": int(d["game_id"].nunique()),
            "pts_bias": float(d["e_pts"].mean()) if n else np.nan,
            "pts_mae": float(d["e_pts"].abs().mean()) if n else np.nan,
            "poss_bias": float(d["e_poss"].mean()) if n else np.nan,
            "ppp_sim": float(d["ppp"].mean()) if n else np.nan,
            "ppp_act": float(d["a_ppp"].mean()) if n else np.nan,
            "fga3_rate": float((d["fga3"] / d["poss"]).mean()) if n else np.nan,
            "fga3_rate_act": float((d["a_tpa"] / d["a_poss"]).mean()) if n else np.nan,
            "tov_rate": float((d["tov"] / d["poss"]).mean()) if n else np.nan,
            "tov_rate_act": float((d["a_tov"] / d["a_poss"]).mean()) if n else np.nan,
            "ftr": float((d["fta"] / (d["fga3"] + d["fga2_rim"] + d["fga2_jump"])).mean())
                   if n else np.nan,
            "ftr_act": float((d["a_fta"] / d["a_fga"]).mean()) if n else np.nan,
            "status": "UNDERPOWERED" if n < MIN_CELL_N else "scored",
        })
    return pd.DataFrame(out)


def per_game_delta(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Paired per-game movement between two arms, on the game level."""
    ka = a.groupby("game_id").agg(pts=("pts", "sum"), poss=("poss", "mean")).reset_index()
    kb = b.groupby("game_id").agg(pts=("pts", "sum"), poss=("poss", "mean")).reset_index()
    m = ka.merge(kb, on="game_id", suffixes=("_a", "_b"))
    dt = m["pts_b"] - m["pts_a"]
    dp = m["poss_b"] - m["poss_a"]
    return {"n_games": int(len(m)),
            "d_total_mean": float(dt.mean()), "d_total_sd": float(dt.std(ddof=1)),
            "d_total_p05": float(dt.quantile(0.05)), "d_total_p95": float(dt.quantile(0.95)),
            "d_poss_mean": float(dp.mean()), "d_poss_sd": float(dp.std(ddof=1))}


def per_team_bias(sim: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    m = sim.merge(act, on=["game_id", "team_id"], how="inner")
    m["e_pts"] = m["pts"] - m["a_pts"]
    g = m.groupby("team_id").agg(n=("game_id", "size"), pts_bias=("e_pts", "mean"))
    return g.reset_index()


# ---------------------------------------------------------------------------
def build_segments(eg: pd.DataFrame, season: int) -> pd.DataFrame:
    start = eg["game_date"].min()
    wk = ((eg["game_date"] - start).dt.days // 7).astype(int)
    bucket = np.where(wk <= 3, "wk0_3", np.where(wk <= 7, "wk4_7", "wk8plus"))
    conf = CF.build_conference_flags([season])[["game_id", "is_conf_game"]]
    seg = pd.DataFrame({"game_id": eg["game_id"].to_numpy(), "week": wk.to_numpy(),
                        "week_bucket": bucket})
    return seg.merge(conf, on="game_id", how="left")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--floor", default=None)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--out", default="results/engine_v0/po4b_grade")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    tags = [args.ref] + ([args.floor] if args.floor else []) + list(args.arms)
    metas = {t: json.loads((rd / t / "run_meta.json").read_text(encoding="utf-8"))
             for t in tags}
    gid_sets = {t: sorted(int(x) for x in metas[t]["game_ids"]) for t in tags}
    base = gid_sets[args.ref]
    for t in tags:
        if gid_sets[t] != base:
            raise SystemExit(f"{t} was run on a different game set than {args.ref}")
    game_ids = np.array(base)

    eg = engine_games(args.fold, args.season)
    eg = eg[eg["game_id"].isin(game_ids)].reset_index(drop=True)
    seg = build_segments(engine_games(args.fold, args.season), args.season)
    seg = seg[seg["game_id"].isin(game_ids)]

    block, names = served_team_block(args.fold, args.season)
    allg = engine_games(args.fold, args.season)
    pos = {int(g): i for i, g in enumerate(allg["game_id"].to_numpy())}
    drv_rows = []
    for _, r in eg.iterrows():
        i = pos[int(r["game_id"])]
        for sidx, tcol in ((0, "home_team_id"), (1, "away_team_id")):
            rec = {"game_id": int(r["game_id"]), "team_id": int(r[tcol])}
            for feat, _cls in DRIVERS:
                rec[feat] = float(block[i, sidx, names[feat]])
            drv_rows.append(rec)
    drv = pd.DataFrame(drv_rows)

    act = actual_team_long(args.season, game_ids)
    sims = {t: sim_team_long(rd / t, game_ids, allg) for t in tags}

    report: dict = {"ref": args.ref, "floor": args.floor, "arms": list(args.arms),
                    "n_games": int(len(game_ids)),
                    "seeds": {t: metas[t]["n_seeds"] for t in tags},
                    "arm_flag": {t: metas[t]["adapter_flags"]["ENGINE_EVENT"] for t in tags},
                    "responsiveness": {}, "headline_slope": {}, "segments": {},
                    "per_game": {}, "per_team": {}}

    for t in tags:
        rows = responsiveness(sims[t], act, drv)
        report["responsiveness"][t] = rows
        report["headline_slope"][t] = headline_slope(rows)
        report["segments"][t] = segment_table(sims[t], act, seg).to_dict("records")
        pt = per_team_bias(sims[t], act)
        report["per_team"][t] = {
            "n_teams": int(len(pt)),
            "sd_of_team_pts_bias": float(pt["pts_bias"].std(ddof=1)),
            "mean_abs_team_pts_bias": float(pt["pts_bias"].abs().mean()),
            "n_teams_underpowered": int((pt["n"] < 3).sum()),
        }
        if t != args.ref:
            report["per_game"][t] = per_game_delta(sims[args.ref], sims[t])

    # ---- the responsiveness line, priced in floors ------------------------
    lines = []
    ref_rows = {r["driver"]: r for r in report["responsiveness"][args.ref]}
    fl_rows = ({r["driver"]: r for r in report["responsiveness"][args.floor]}
               if args.floor else {})
    for feat, _cls in DRIVERS:
        r0 = ref_rows.get(feat, {})
        rf = fl_rows.get(feat, {})
        s0, sf = r0.get("slope_ratio"), rf.get("slope_ratio")
        floor = abs(sf - s0) if (s0 is not None and sf is not None) else None
        for t in args.arms:
            ra = {r["driver"]: r for r in report["responsiveness"][t]}.get(feat, {})
            sa = ra.get("slope_ratio")
            if s0 is None or sa is None:
                continue
            fall = s0 - sa                       # positive = the arm FELL below the reference
            lines.append({
                "quantity": f"slope_ratio[{feat}]", "arm": t,
                "ref": s0, "arm_value": sa, "floor_run": sf, "floor": floor,
                "signed_fall": round(fall, 5),
                "fall_in_floors": (round(fall / floor, 2) if floor else None),
                "d_abs_dev_from_1": round(abs(sa - 1.0) - abs(s0 - 1.0), 5),
                "exempt": bool(r0.get("exempt_narrow_span", False)),
                "status": r0.get("status", "scored"),
            })
    h0, hf = report["headline_slope"][args.ref], (
        report["headline_slope"].get(args.floor) if args.floor else None)
    hfloor = abs(hf - h0) if (h0 is not None and hf is not None) else None
    for t in args.arms:
        ha = report["headline_slope"][t]
        if h0 is None or ha is None:
            continue
        lines.append({"quantity": "headline slope (worst non-exempt driver)", "arm": t,
                      "ref": h0, "arm_value": ha, "floor_run": hf, "floor": hfloor,
                      "signed_fall": round(h0 - ha, 5),
                      "fall_in_floors": (round((h0 - ha) / hfloor, 2) if hfloor else None),
                      "d_abs_dev_from_1": round(abs(ha - 1.0) - abs(h0 - 1.0), 5),
                      "exempt": False, "status": "scored"})
    report["responsiveness_lines"] = lines

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "po4b_lines.json").write_text(json.dumps(report, indent=1, default=str),
                                         encoding="utf-8")
    pd.DataFrame(lines).to_csv(out / "responsiveness_lines.csv", index=False)
    segs = []
    for t in tags:
        s = pd.DataFrame(report["segments"][t])
        s.insert(0, "run", t)
        segs.append(s)
    pd.concat(segs, ignore_index=True).to_csv(out / "segments.csv", index=False)

    print(f"\n=== responsiveness, closed loop ({len(game_ids)} games) ===")
    for t in tags:
        print(f"\n{t}  ({metas[t]['adapter_flags']['ENGINE_EVENT']}, "
              f"{metas[t]['n_seeds']} seeds)   headline slope "
              f"{report['headline_slope'][t]}")
        for r in report["responsiveness"][t]:
            if "slope_ratio" not in r:
                print(f"   {r['driver']:12s} {r.get('status')}")
                continue
            print(f"   {r['driver']:12s} slope {r['slope_ratio']!s:>8}  "
                  f"span_sim {r['span_sim_pp']:+.3f}pp  span_act {r['span_act_pp']:+.3f}pp  "
                  f"steps {r['steps_with_actual']}/{r['n_steps']}  n {r['n']}"
                  f"{'  EXEMPT' if r['exempt_narrow_span'] else ''}")
    print("\n=== the line ===")
    for ln in lines:
        print(f"   {ln['quantity']:44s} {ln['arm']:16s} ref {ln['ref']:.4f} -> "
              f"{ln['arm_value']:.4f}  fall {ln['signed_fall']:+.4f}  "
              f"floor {ln['floor']}  = {ln['fall_in_floors']} floors")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
