"""diag_g4_oreb_fta_v1.py -- gate G4 cause tracing for OREB% (-1.6 pp) and
FTA/FGA (-1.2 pp) on fold 2 (2025).

DIAGNOSTIC ONLY. Reads data and existing sim output; writes nothing under
`src/cbb_sim/`, changes no served default, fits nothing, tunes nothing.

Parts (``--part`` may be repeated; default: all):

  truth   box (hoopR team_box, the grader's ACTUAL) vs pbp event layer, per
          team-game, for OREB / DREB / FTA / FGA.  Quantifies the
          grading-source channel BEFORE any model is blamed.
  actual  the actual 2025 rebound opportunity mix and P(OREB | miss type,
          blocked), and the actual FTA supply split by trip kind.
  sim     pooled sim four factors from a results dir, plus the sim's own
          rebound-opportunity mix derived from the box counts and the
          engine's own fixed dead shares.
  reb     OFFLINE calibration of the SERVED rebound arm (s1_weekly dated
          schedule) on the fold-2 test rows, scored twice: with the true
          `blocked_f` and with `blocked_f` forced to 0.0, which is what
          `engine/loop.py` feeds it.

Usage:
    .venv/Scripts/python.exe scripts/diag_g4_oreb_fta_v1.py \
        --part truth --part actual --part sim --part reb \
        --run results/engine_v0/F2_2025_s200_v5b_A \
        --out results/g4_diag
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEASON = 2025
MISS_TYPES = ("rim", "jump2", "three", "ft")
OUT: dict = {}


def _p(*a):
    print(*a, flush=True)


def _hdr(t: str):
    _p("\n" + "=" * 78)
    _p(t)
    _p("=" * 78)


# ---------------------------------------------------------------------------
# truth: box vs pbp event layer
# ---------------------------------------------------------------------------
def part_truth() -> dict:
    from cbb_sim.eval import reference as ref_mod

    _hdr("PART truth -- grading source: hoopR box vs pbp event layer, 2025")
    box = ref_mod.load_actual_team_box(SEASON)
    box = box[["game_id", "team_id", "opp_team_id", "fga", "tpa", "fta", "ftm",
               "fgm", "tpm", "oreb", "dreb", "tov", "poss_team"]].copy()
    _p(f"box team-games: {len(box)}  games: {box.game_id.nunique()}")

    # hoopR team_box carries more rebound columns than load_team_games keeps;
    # read them directly so "does the box OREB include team rebounds" is
    # answered from the source rather than assumed.
    tb = pd.read_parquet(ROOT / f"data/raw/hoopr/team_box/team_box_{SEASON}.parquet")
    cols = [c for c in tb.columns if "rebound" in c.lower()]
    _p(f"hoopR team_box rebound-ish columns: {cols}")
    tb["game_id"] = pd.to_numeric(tb["game_id"], errors="coerce").astype("int64")
    tb["team_id"] = pd.to_numeric(tb["team_id"], errors="coerce").astype("int64")
    sub = tb[["game_id", "team_id"] + cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.merge(box[["game_id", "team_id"]], on=["game_id", "team_id"], how="inner")
    if {"offensive_rebounds", "defensive_rebounds", "total_rebounds"} <= set(cols):
        gap = sub["total_rebounds"] - sub["offensive_rebounds"] - sub["defensive_rebounds"]
        _p(f"total_rebounds - (oreb+dreb): mean {gap.mean():.4f}  "
           f"nonzero {int((gap != 0).sum())}/{len(gap)}  sum {gap.sum():.0f}")
        _p("  -> team/dead-ball rebounds are NOT inside the box oreb/dreb the grader reads"
           if abs(gap.mean()) < 1e-9 else
           "  -> box total_rebounds carries rebounds outside oreb+dreb")

    # pbp event layer: rebound opportunities + chance-level FTA/FGA
    ev = pd.read_parquet(ROOT / "data/processed/models/rebound/events_v1.parquet",
                         columns=["game_id", "season", "off_team_id", "def_team_id",
                                  "miss_type", "blocked", "outcome"])
    ev = ev[ev.season == SEASON]
    o = ev[ev.outcome == "OREB"].groupby(["game_id", "off_team_id"]).size().rename("pbp_oreb")
    d = ev[ev.outcome == "DREB"].groupby(["game_id", "def_team_id"]).size().rename("pbp_dreb")
    o.index.names = ["game_id", "team_id"]
    d.index.names = ["game_id", "team_id"]

    ch = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{SEASON}.parquet",
                         columns=["game_id", "offense_team_id", "fga_rim", "fga_jump2",
                                  "fga_3", "fta", "ftm"])
    ch["fga"] = ch.fga_rim + ch.fga_jump2 + ch.fga_3
    pbp_tg = ch.groupby(["game_id", "offense_team_id"]).agg(
        pbp_fga=("fga", "sum"), pbp_fta=("fta", "sum"), pbp_ftm=("ftm", "sum")).reset_index()
    pbp_tg = pbp_tg.rename(columns={"offense_team_id": "team_id"})

    m = box.merge(pbp_tg, on=["game_id", "team_id"], how="inner")
    m = m.merge(o, on=["game_id", "team_id"], how="left").merge(d, on=["game_id", "team_id"], how="left")
    m[["pbp_oreb", "pbp_dreb"]] = m[["pbp_oreb", "pbp_dreb"]].fillna(0.0)
    _p(f"team-games with BOTH box and pbp: {len(m)} "
       f"({len(m) / len(box) * 100:.1f}% of box team-games)")

    res = {"n_team_games_box": int(len(box)), "n_team_games_matched": int(len(m))}
    for name, bcol, pcol in (("OREB", "oreb", "pbp_oreb"), ("DREB", "dreb", "pbp_dreb"),
                             ("FTA", "fta", "pbp_fta"), ("FGA", "fga", "pbp_fga")):
        diff = m[pcol] - m[bcol]
        _p(f"{name:5s}  box/tg {m[bcol].mean():7.3f}  pbp/tg {m[pcol].mean():7.3f}  "
           f"pbp-box {diff.mean():+7.4f}  |d|>0 {float((diff != 0).mean()) * 100:5.1f}%  "
           f"season sum box {m[bcol].sum():.0f} pbp {m[pcol].sum():.0f}")
        res[f"{name}_box_per_tg"] = float(m[bcol].mean())
        res[f"{name}_pbp_per_tg"] = float(m[pcol].mean())
        res[f"{name}_pbp_minus_box_per_tg"] = float(diff.mean())

    # the two graded ratios computed from each source, on the SAME team-games
    mm = m.merge(m[["game_id", "team_id", "dreb", "pbp_dreb"]].rename(
        columns={"team_id": "opp_team_id", "dreb": "opp_dreb_b", "pbp_dreb": "opp_dreb_p"}),
        on=["game_id", "opp_team_id"], how="left")
    ok = mm["opp_dreb_b"].notna()
    mm = mm[ok]
    box_oreb_pct = float(mm.oreb.sum() / (mm.oreb + mm.opp_dreb_b).sum())
    pbp_oreb_pct = float(mm.pbp_oreb.sum() / (mm.pbp_oreb + mm.opp_dreb_p).sum())
    box_ftr = float(mm.fta.sum() / mm.fga.sum())
    pbp_ftr = float(mm.pbp_fta.sum() / mm.pbp_fga.sum())
    _p(f"\npooled OREB%   box {box_oreb_pct:.5f}   pbp {pbp_oreb_pct:.5f}   "
       f"pbp-box {(pbp_oreb_pct - box_oreb_pct) * 100:+.3f} pp")
    _p(f"pooled FTA/FGA box {box_ftr:.5f}   pbp {pbp_ftr:.5f}   "
       f"pbp-box {(pbp_ftr - box_ftr) * 100:+.3f} pp")
    # numerator-only and denominator-only views of the FT rate gap
    ftr_num_only = float(mm.pbp_fta.sum() / mm.fga.sum())
    _p(f"  FT rate with pbp FTA over BOX FGA: {ftr_num_only:.5f} "
       f"({(ftr_num_only - box_ftr) * 100:+.3f} pp -> the FTA-count channel)")
    _p(f"  remaining {(pbp_ftr - ftr_num_only) * 100:+.3f} pp is the FGA-count channel")
    res.update(box_oreb_pct=box_oreb_pct, pbp_oreb_pct=pbp_oreb_pct,
               box_ft_rate=box_ftr, pbp_ft_rate=pbp_ftr,
               ft_rate_pbpfta_boxfga=ftr_num_only)
    return res


# ---------------------------------------------------------------------------
# actual: rebound mix / rates, FTA supply split
# ---------------------------------------------------------------------------
def part_actual() -> dict:
    _hdr("PART actual -- 2025 rebound opportunity mix, P(OREB|.), FTA supply")
    ev = pd.read_parquet(ROOT / "data/processed/models/rebound/events_v1.parquet")
    ev = ev[ev.season == SEASON].copy()
    res: dict = {"n_opportunity_rows": int(len(ev))}
    _p("outcome shares over ALL opportunities:")
    _p(ev.outcome.value_counts(normalize=True).to_string())
    live = ev[ev.outcome.isin(["OREB", "DREB"])].copy()
    live["o"] = (live.outcome == "OREB").astype(float)
    res["n_live"] = int(len(live))
    res["oreb_pct_pbp"] = float(live.o.mean())

    tab = live.groupby("miss_type").agg(n=("o", "size"), oreb=("o", "mean"))
    tab["mix"] = tab.n / tab.n.sum()
    _p("\nlive opportunities by miss type:")
    _p(tab.to_string())
    res["actual_mix"] = {k: float(v) for k, v in tab["mix"].items()}
    res["actual_rate"] = {k: float(v) for k, v in tab["oreb"].items()}

    _p("\nP(OREB) by miss type x blocked:")
    bt = live.groupby(["miss_type", "blocked"]).agg(n=("o", "size"), oreb=("o", "mean"))
    _p(bt.to_string())
    res["blocked_share_overall"] = float(live.blocked.mean())
    res["blocked_share_by_type"] = {k: float(v) for k, v in live.groupby("miss_type").blocked.mean().items()}
    r_unb = live[~live.blocked].groupby("miss_type").o.mean()
    naive = float((tab["mix"] * r_unb).sum())
    _p(f"\nmarginal blocked channel (unblocked rates at the true mix): {naive:.5f} "
       f"vs actual {live.o.mean():.5f}  ({(naive - live.o.mean()) * 100:+.3f} pp)")
    res["oreb_pct_unblocked_rates_true_mix"] = naive

    # chance-level FTA supply
    _hdr("PART actual -- 2025 FTA supply by trip kind (chances table)")
    ch = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{SEASON}.parquet")
    ch["fga"] = ch.fga_rim + ch.fga_jump2 + ch.fga_3
    ch["fgm"] = ch.fgm_rim + ch.fgm_jump2 + ch.fgm_3
    n_poss = ch.groupby(["game_id", "poss_index", "offense_team_id"]).ngroups
    _p(f"chances {len(ch)}  possessions {n_poss}  games {ch.game_id.nunique()}")
    _p("terminal_event shares:")
    _p(ch.terminal_event.value_counts(normalize=True).to_string())

    fta_tot = float(ch.fta.sum())
    ao = ch[ch.and_one]
    trip = ch[ch.terminal_event.astype(str).str.startswith("FT_trip")]
    _p(f"\ntotal FTA (event layer) {fta_tot:.0f}")
    _p(f"  and_one chances {len(ao)}  their FTA {ao.fta.sum():.0f} "
       f"({ao.fta.sum() / fta_tot * 100:.1f}%)")
    for te, g in trip.groupby(trip.terminal_event.astype(str)):
        _p(f"  {te:22s} n {len(g):7d}  FTA {g.fta.sum():8.0f} "
           f"({g.fta.sum() / fta_tot * 100:5.1f}%)  FTA/trip {g.fta.mean():.3f}")
    other = ch[~ch.and_one & ~ch.terminal_event.astype(str).str.startswith("FT_trip")]
    _p(f"  other chances with FTA n {int((other.fta > 0).sum())} FTA {other.fta.sum():.0f}")
    res["actual_fta_total_pbp"] = fta_tot
    res["actual_fga_total_pbp"] = float(ch.fga.sum())
    res["actual_and_one_fta"] = float(ao.fta.sum())
    res["actual_trip_fta"] = {str(k): float(v.fta.sum())
                              for k, v in trip.groupby(trip.terminal_event.astype(str))}
    res["actual_trip_n"] = {str(k): int(len(v))
                            for k, v in trip.groupby(trip.terminal_event.astype(str))}
    res["actual_and_one_n"] = int(len(ao))
    res["n_possessions_pbp"] = int(n_poss)

    # the final-2:00 window, which the late-game lane owns
    ch["late"] = (ch.period == 2) & (ch.start_clock <= 120)
    lat = ch.groupby("late").agg(n=("fta", "size"), fta=("fta", "sum"), fga=("fga", "sum"))
    _p("\nFTA / FGA inside vs outside the final 2:00 of regulation (actual):")
    _p(lat.assign(ft_rate=lat.fta / lat.fga, fta_share=lat.fta / lat.fta.sum()).to_string())
    res["actual_late_fta"] = float(lat.loc[True, "fta"])
    res["actual_late_fga"] = float(lat.loc[True, "fga"])
    res["actual_rest_fta"] = float(lat.loc[False, "fta"])
    res["actual_rest_fga"] = float(lat.loc[False, "fga"])
    return res


# ---------------------------------------------------------------------------
# sim: pooled four factors + derived rebound mix
# ---------------------------------------------------------------------------
def _sim_team_frame(run: Path) -> pd.DataFrame:
    g = pd.read_parquet(run / "games.parquet")
    cols = ("fga3", "fga2_rim", "fga2_jump", "fta", "tov", "oreb", "dreb",
            "fgm2_rim", "fgm2_jump", "fgm3", "ftm")
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = g[["game_id", "seed", f"{opp}_dreb"] + [f"{side}_{c}" for c in cols]].copy()
        d.columns = ["game_id", "seed", "opp_dreb"] + list(cols)
        out.append(d)
    s = pd.concat(out, ignore_index=True)
    s["fga"] = s.fga3 + s.fga2_rim + s.fga2_jump
    s["fgm"] = s.fgm2_rim + s.fgm2_jump + s.fgm3
    return s


def part_sim(runs: list[Path]) -> dict:
    _hdr("PART sim -- pooled four factors and derived rebound mix")
    dead = {"rim": 0.004562129620919951, "jump2": 0.006432451059221214,
            "three": 0.009225635700264665, "ft": 0.007985358472046133}
    res: dict = {}
    for run in runs:
        s = _sim_team_frame(run)
        meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
        tag = run.name
        oreb_pct = float(s.oreb.sum() / (s.oreb + s.opp_dreb).sum())
        ft_rate = float(s.fta.sum() / s.fga.sum())
        m_rim = float((s.fga2_rim - s.fgm2_rim).sum())
        m_jmp = float((s.fga2_jump - s.fgm2_jump).sum())
        m_thr = float((s.fga3 - s.fgm3).sum())
        live_fg = (1 - dead["rim"]) * m_rim + (1 - dead["jump2"]) * m_jmp + (1 - dead["three"]) * m_thr
        live_tot = float((s.oreb + s.dreb).sum())
        live_ft = live_tot - live_fg
        m_ft = live_ft / (1 - dead["ft"])
        mix = {"rim": (1 - dead["rim"]) * m_rim / live_tot,
               "jump2": (1 - dead["jump2"]) * m_jmp / live_tot,
               "three": (1 - dead["three"]) * m_thr / live_tot,
               "ft": live_ft / live_tot}
        ns = int(meta["n_seeds"])
        _p(f"\n{tag}  n_seeds {ns}  n_games {meta['n_games']}  clock {meta['adapter_flags']['ENGINE_CLOCK']}")
        _p(f"  OREB%   {oreb_pct:.5f}")
        _p(f"  FTA/FGA {ft_rate:.5f}")
        _p(f"  per team-game: FGA {s.fga.sum() / len(s):.3f}  FTA {s.fta.sum() / len(s):.3f}  "
           f"OREB {s.oreb.sum() / len(s):.3f}  DREB {s.dreb.sum() / len(s):.3f}")
        _p(f"  derived FT last-misses/team-game {m_ft / len(s):.3f}; live-opportunity mix: "
           + "  ".join(f"{k} {v:.4f}" for k, v in mix.items()))
        res[tag] = {"n_seeds": ns, "oreb_pct": oreb_pct, "ft_rate": ft_rate,
                    "mix": {k: float(v) for k, v in mix.items()},
                    "fga_per_tg": float(s.fga.sum() / len(s)),
                    "fta_per_tg": float(s.fta.sum() / len(s)),
                    "fgm_per_tg": float(s.fgm.sum() / len(s)),
                    "fgm2_rim_per_tg": float(s.fgm2_rim.sum() / len(s)),
                    "fgm2_jump_per_tg": float(s.fgm2_jump.sum() / len(s)),
                    "fgm3_per_tg": float(s.fgm3.sum() / len(s)),
                    "oreb_per_tg": float(s.oreb.sum() / len(s)),
                    "dreb_per_tg": float(s.dreb.sum() / len(s)),
                    "n_team_games": int(len(s))}
    return res


# ---------------------------------------------------------------------------
# reb: offline calibration of the SERVED arm on fold-2 test rows
# ---------------------------------------------------------------------------
def part_reb() -> dict:
    _hdr("PART reb -- served rebound arm (s1_weekly) offline on fold-2 rows")
    import joblib

    from cbb_sim.models import rebound as RB

    events = pd.read_parquet(ROOT / "data/processed/models/rebound/events_v1.parquet")
    des = RB.build_design([SEASON], events=events[events.season == SEASON])
    _p(f"design rows (2025, resolved): {len(des)}  unresolved dropped: {des.attrs['n_unresolved']}")

    man_path = ROOT / "data/processed/models/rebound/s1_confirm/S1_weekly/F2/manifest.json"
    obj = json.loads(man_path.read_text(encoding="utf-8"))
    arts = sorted(obj["artifacts"], key=lambda a: pd.Timestamp(a["refit_date"]))
    _p(f"S1_weekly artifacts: {len(arts)}  refit dates "
       f"{arts[0]['refit_date'][:10]} .. {arts[-1]['refit_date'][:10]}")
    feats = RB.feature_set("C_plus_state")

    # per-row artifact choice: the latest refit whose date is <= the game date,
    # the same rule ArtifactManifest._select applies per game.
    dates = np.array([pd.Timestamp(a["refit_date"]).value for a in arts])
    gd = pd.to_datetime(des["game_date"]).astype("int64").to_numpy()
    seg = np.clip(np.searchsorted(dates, gd, side="right") - 1, 0, len(arts) - 1)
    des = des.assign(_seg=seg)

    X = des[feats].to_numpy(dtype=np.float64)
    ib = feats.index("blocked_f")
    X0 = X.copy()
    X0[:, ib] = 0.0
    p_true = np.zeros((len(des), 3))
    p_zero = np.zeros((len(des), 3))
    for k in sorted(set(seg.tolist())):
        w = joblib.load(man_path.parent / arts[k]["path"])
        if list(w["features"]) != feats:
            raise ValueError("artifact feature order differs from C_plus_state")
        mdl = w["model"]
        try:
            mdl.clf_.set_params(n_jobs=1)
        except Exception:                                        # noqa: BLE001
            pass
        r = np.flatnonzero(seg == k)
        p_true[r] = mdl.predict_proba(X[r])
        p_zero[r] = mdl.predict_proba(X0[r])

    io, id_ = RB.CLASS_INDEX["OREB"], RB.CLASS_INDEX["DREB"]

    def binary(p):
        return p[:, io] / np.maximum(p[:, io] + p[:, id_], 1e-12)

    b_true, b_zero = binary(p_true), binary(p_zero)
    y = des["y"].to_numpy()
    live = (y == io) | (y == id_)
    act = (y[live] == io).astype(float)
    mt = des["miss_type"].to_numpy()[live]
    blk = des["blocked"].to_numpy()[live].astype(bool)

    res = {"n_design": int(len(des)), "n_live": int(live.sum())}
    _p(f"\nlive rows {int(live.sum())}")
    _p(f"  ACTUAL      P(OREB) {act.mean():.5f}")
    _p(f"  MODEL true  P(OREB) {b_true[live].mean():.5f}   "
       f"({(b_true[live].mean() - act.mean()) * 100:+.3f} pp = fold-2 calibration error)")
    _p(f"  MODEL blk=0 P(OREB) {b_zero[live].mean():.5f}   "
       f"({(b_zero[live].mean() - b_true[live].mean()) * 100:+.3f} pp = the engine's blocked_f=0 feed)")
    res.update(actual_oreb=float(act.mean()),
               model_true_oreb=float(b_true[live].mean()),
               model_blk0_oreb=float(b_zero[live].mean()))

    rows = []
    for m in MISS_TYPES:
        s = mt == m
        rows.append({"miss_type": m, "n": int(s.sum()), "mix": float(s.mean()),
                     "actual": float(act[s].mean()),
                     "model_true": float(b_true[live][s].mean()),
                     "model_blk0": float(b_zero[live][s].mean()),
                     "blocked_share": float(blk[s].mean())})
    t = pd.DataFrame(rows)
    t["cal_err_pp"] = (t.model_true - t.actual) * 100
    t["blk0_pp"] = (t.model_blk0 - t.model_true) * 100
    _p("\nby miss type:")
    _p(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    res["by_miss_type"] = t.to_dict("records")

    _p("\nby blocked, within the blocked rows only:")
    for m in ("rim", "jump2", "three"):
        s = (mt == m) & blk
        if s.sum() < 300:
            _p(f"  {m:6s} n {int(s.sum()):6d}  UNDERPOWERED")
            continue
        _p(f"  {m:6s} n {int(s.sum()):6d}  actual {act[s].mean():.5f}  "
           f"model_true {b_true[live][s].mean():.5f}  model_blk0 {b_zero[live][s].mean():.5f}  "
           f"blk0 effect {(b_zero[live][s].mean() - b_true[live][s].mean()) * 100:+.3f} pp")

    # responsiveness: does the model's own prediction slope with the offence's
    # as-of OREB% quintile, and does forcing blocked_f=0 flatten it?
    q = pd.qcut(des.loc[live, "off_oreb_c"], 5, labels=False, duplicates="drop")
    rq = pd.DataFrame({"q": q.to_numpy(), "actual": act,
                       "model_true": b_true[live], "model_blk0": b_zero[live]})
    _p("\nresponsiveness by offence as-of OREB% quintile:")
    _p(rq.groupby("q").agg(n=("actual", "size"), actual=("actual", "mean"),
                           model_true=("model_true", "mean"),
                           model_blk0=("model_blk0", "mean")).to_string())
    res["responsiveness"] = rq.groupby("q").mean().to_dict()

    # --- is the level error SEASON DRIFT the pooled fit averages away? ------
    _p("\nseason drift in the target (live OREB%, event layer):")
    allev = events[events.outcome.isin(["OREB", "DREB"])].copy()
    allev["o"] = (allev.outcome == "OREB").astype(float)
    sd = allev.groupby("season").agg(n=("o", "size"), oreb=("o", "mean"))
    _p(sd.to_string())
    res["season_drift"] = sd.reset_index().to_dict("records")

    _p("\nfold-2 calibration error BY MONTH of 2025 (does the weekly refit catch up?):")
    mo = pd.to_datetime(des["game_date"]).dt.month.to_numpy()[live]
    md = pd.DataFrame({"month": mo, "actual": act, "model_true": b_true[live],
                       "model_blk0": b_zero[live]}).groupby("month").agg(
        n=("actual", "size"), actual=("actual", "mean"),
        model_true=("model_true", "mean"), model_blk0=("model_blk0", "mean"))
    md["cal_err_pp"] = (md.model_true - md.actual) * 100
    md["blk0_err_pp"] = (md.model_blk0 - md.actual) * 100
    md["underpowered"] = md.n < 300
    _p(md.to_string(float_format=lambda v: f"{v:.5f}"))
    res["cal_by_month"] = md.reset_index().to_dict("records")

    _p("\ntraining-pool level at each weekly refit vs the 2025 season level "
       f"({allev[allev.season == SEASON].o.mean():.5f}):")
    allev["game_date"] = pd.to_datetime(allev["game_date"])
    prows = []
    for k, aobj in enumerate(arts):
        mt = pd.Timestamp(aobj["max_train_date"])
        pool = allev[allev.game_date <= mt]
        prows.append({"refit": aobj["refit_date"][:10], "max_train": str(mt.date()),
                      "n_pool": len(pool), "pool_oreb": float(pool.o.mean()),
                      "share_2025": float((pool.season == SEASON).mean())})
    pt = pd.DataFrame(prows)
    _p(pt.iloc[::4].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    res["train_pool_level"] = pt.to_dict("records")

    # calibration deciles on the served (blocked_f=0) feed
    dec = pd.qcut(b_zero[live], 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"d": dec, "p": b_zero[live], "y": act}).groupby("d").agg(
        n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
    cal["gap_pp"] = (cal.pred - cal.obs) * 100
    _p("\ncalibration deciles of the SERVED (blocked_f=0) prediction:")
    _p(cal.to_string())
    res["calibration_blk0"] = cal.reset_index().to_dict("records")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", default=None)
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--out", default="results/g4_diag")
    a = ap.parse_args()
    parts = a.part or ["truth", "actual", "sim", "reb"]
    runs = [Path(r) for r in (a.run or ["results/engine_v0/F2_2025_s200_v5b_A",
                                        "results/engine_v0/F2_2025_s200_v1_clockv3c_A"])]
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    if "truth" in parts:
        OUT["truth"] = part_truth()
    if "actual" in parts:
        OUT["actual"] = part_actual()
    if "sim" in parts:
        OUT["sim"] = part_sim(runs)
    if "reb" in parts:
        OUT["reb"] = part_reb()
    p = outd / ("g4_offline_" + "_".join(parts) + ".json")
    p.write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")
    _p(f"\nwrote {p}")


if __name__ == "__main__":
    main()
