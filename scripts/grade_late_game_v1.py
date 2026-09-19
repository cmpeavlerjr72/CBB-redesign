"""grade_late_game_v1.py -- the ONE blind grader for L7 late-game round 1.

Pre-registration: `docs/models/late_game/experiments.md` section 1 (metric 1.3,
floors 1.5, gates 1.6, decision rule 1.7) and the section-2 amendment.

BLIND means: this script reads cell ids and probability matrices, applies the
IDENTICAL code path to every one of them, and has no branch anywhere on which
arm produced a cell.  It is the only place a metric is computed; neither
trainer computes one.

It scores both halves of the regime:

  EVENT    `train_late_game_v1.py` -> (n, 6) class probabilities.
           Primary: multiclass log loss on the SELECTION gate
           (`period == 2`, `seconds_remaining <= 120`, `|score_diff| <= 6`)
           of the fold's test season.
  DURATION `train_late_game_clock_v1.py` -> (n, 91) duration PMFs.
           Primary: CRPS of the truncated law on uncensored window rows, with
           the censored log-likelihood beside it.

FLOORS.  Two, and the BINDING floor is the larger:
  * the seed floor, `|loss(seed 0) - loss(seed 1)|` from spec-identical
    retrains -- real for the LightGBM arms, identically zero for the
    deterministic ones (the cascade's logits, the Kaplan-Meier cell law);
  * the game-level block bootstrap SE of the same metric -- the instrument
    `possession_outcome`'s own pre-registration uses for deterministic arms.
    Chances inside a game are not independent, so the resampling unit is the
    game.

UNDERPOWERED.  Every cell below 200 rows is labelled and is never read as
signal or as absence of signal (experiments.md 1.6).

Usage: grade_late_game_v1.py [--out results/late_game/round1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from cbb_sim.models import clock as ck                               # noqa: E402
from cbb_sim.models import clock_v3 as c3                            # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

EV_DIR = ROOT / "data/processed/models/late_game/round1"
CK_DIR = ROOT / "data/processed/models/late_game/round1_clock"
SELECTION_FOLD = "F2"
PPG: pd.Series | None = None
MIN_CELL = 200

CLASSES = list(PO.CLASSES)
J3 = PO.CLASS_INDEX["FGA_3"]
JB = PO.CLASS_INDEX["FT_trip_bonus"]
FGA_J = [PO.CLASS_INDEX[c] for c in ("FGA_rim", "FGA_jump2", "FGA_3")]

#: The 2024-25 actual role splits this round must reproduce the SIGN and at
#: least HALF the magnitude of (experiments.md gates R1 and R2).
#:
#: SIGN.  `experiments.md` R1 says the three-point split is NEGATIVE, quoting
#: "-0.158".  The evidence table it quotes has the trailing offence at 0.4788
#: and the leading offence at 0.3213, so trailing MINUS leading is +0.1575: the
#: minus sign is a transcription error and the substantive gate -- the trailing
#: team shoots MORE threes -- is unchanged.  The amendment records the
#: correction; this constant carries the sign the data has.  Both constants are
#: recomputed from the test fold's own rows by the grader anyway
#: (`act[gate]["three_split"]`); these are the pre-registered TARGETS.
ACT_3PA_SPLIT = +0.1575     # trailing minus leading, three-point share of FGA
ACT_FOUL_SPLIT = +0.2652    # leading minus trailing, bonus-FT rate

GATES = {"main": "in_window", "sens1": "in_window_sens1", "sens2": "in_window_sens2"}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ===========================================================================
# shared metric machinery -- one code path for every cell
# ===========================================================================
def block_se(values: np.ndarray, games: np.ndarray, n_rep: int = 400,
             seed: int = 20260918) -> float:
    """Game-level block bootstrap SE of the MEAN of a per-row score."""
    ok = np.isfinite(values)
    values, games = values[ok], games[ok]
    if not len(values):
        return float("nan")
    order = np.argsort(games, kind="stable")
    gs = games[order]
    starts = np.flatnonzero(np.concatenate([[True], gs[1:] != gs[:-1]]))
    ends = np.concatenate([starts[1:], [len(gs)]])
    sums = np.array([values[order[s:e]].sum() for s, e in zip(starts, ends, strict=False)])
    ns = np.array([e - s for s, e in zip(starts, ends, strict=False)], dtype="float64")
    rng = np.random.default_rng(seed)
    n_g = len(sums)
    out = np.empty(n_rep)
    for r in range(n_rep):
        pick = rng.integers(0, n_g, n_g)
        out[r] = sums[pick].sum() / ns[pick].sum()
    return float(out.std(ddof=1))


def ll_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))


def three_share(p: np.ndarray) -> float:
    """AGGREGATE three-point share of field-goal attempts over a set of rows:
    `sum(P(FGA_3)) / sum(P(any FGA))`.

    Aggregate, not a mean of per-row ratios, so that the predicted quantity and
    the realised one are the SAME functional -- on the realised side a
    turnover row has no field-goal attempt at all and a per-row ratio is
    undefined there, which is exactly how a mean-of-ratios silently becomes a
    different statistic on the two sides."""
    fga = p[:, FGA_J].sum()
    return float(p[:, J3].sum() / max(fga, 1e-12))


def cell_label(n: int) -> str:
    return "UNDERPOWERED" if n < MIN_CELL else ""


# ===========================================================================
# 1. EVENT half
# ===========================================================================
DESIGN = ROOT / "data/processed/models/late_game/design_v1.parquet"


def team_ppg() -> pd.Series:
    """Season points per game of the OFFENCE team, read from the FULL design,
    not from the scored sample -- the scored rows are a 60k sample of the test
    season and points-per-sampled-group is not points-per-game.

    A BUCKETING variable for the responsiveness cut only (exactly as the
    2026-09-11 actual table used it); it is never a feature, and gate R3 names
    it, so it is computed as named."""
    d = pd.read_parquet(DESIGN, columns=["season", "game_id", "offense_team_id", "points"])
    g = d.groupby(["season", "offense_team_id", "game_id"], as_index=False)["points"].sum()
    return g.groupby(["season", "offense_team_id"])["points"].mean()


def quintiles(v: np.ndarray, sel: np.ndarray, n_q: int = 5) -> np.ndarray:
    edges = np.nanquantile(v[sel], np.linspace(0, 1, n_q + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_q - 1)


def grade_event(fold: str, cells: list[dict], pop_graded: str = "first"
                ) -> tuple[pd.DataFrame, dict]:
    """`pop_graded` restricts EVERY mask to one chance population, so every
    arm is scored on exactly the same rows.  Round 1 grades FIRST chances
    (86.4% of window rows): the continuation cascade fit was dropped under a
    wall-clock deadline, and the first-chance predictions are bit-identical
    either way because each population is fitted on its own rows only.  The
    restriction is applied to the REFERENCE rows too, so R6 is on the same
    population as the primary."""
    te = pd.read_parquet(EV_DIR / f"window_test_{fold}.parquet")
    y = te["y"].to_numpy()
    games = te["game_id"].to_numpy()
    # The population restriction is a MASK, never a row drop: the prediction
    # matrices are indexed by the full canonical order and must stay aligned.
    pop_ok = (te["population"].to_numpy() == pop_graded if pop_graded
              else np.ones(len(te), dtype=bool))
    masks = {k: te[v].to_numpy() & ~te["is_reference"].to_numpy() & pop_ok
             for k, v in GATES.items()}
    ref = te["is_reference"].to_numpy() & pop_ok
    sel = masks["main"]
    role = te["role"].to_numpy()
    pop = te["population"].to_numpy()
    sec = te["seconds_remaining"].to_numpy()

    key = pd.MultiIndex.from_arrays([te["season"], te["offense_team_id"]])
    tv = PPG.reindex(key).to_numpy()
    quint = quintiles(tv, sel)
    # A SECOND responsiveness cut on a genuine as-of PRIOR -- the offence's own
    # league-centred ridge offensive rating, strictly before the game.  Season
    # PPG is what gate R3 names and is contemporaneous; CLAUDE.md's
    # matchup-specific rule asks for a PRIOR quintile, so both are reported and
    # a winner has to slope on both.
    quint_prior = quintiles(te["off_rating_off_c"].to_numpy(), sel)

    # --- the ACTUAL side of every gate, once -------------------------------
    act = {}
    for gname, m in masks.items():
        t, l = m & (role < 0), m & (role > 0)
        onehot = np.zeros((len(y), len(CLASSES)))
        onehot[np.arange(len(y)), y] = 1.0
        act[gname] = {
            "n": int(m.sum()), "n_trail": int(t.sum()), "n_lead": int(l.sum()),
            "three_split": three_share(onehot[t]) - three_share(onehot[l]),
            "foul_split": float(onehot[l][:, JB].mean() - onehot[t][:, JB].mean()),
        }
    act_q = [float((y[sel & (quint == q)] == JB).mean()) for q in range(5)]
    act_qp = [float((y[sel & (quint_prior == q)] == JB).mean()) for q in range(5)]
    act_ref_rate = {c: float((y[ref] == j).mean()) for j, c in enumerate(CLASSES)}

    rows, detail = [], {}
    for c in cells:
        p = np.load(EV_DIR / "pred" / f"{c['cell_id']}.npy").astype("float64")
        lr = ll_rows(y, np.nan_to_num(p, nan=1.0))
        r = {k: c[k] for k in ("cell_id", "arm", "bundle", "fold", "seed",
                               "cadence", "scope")}
        r["n_selection"] = int(sel.sum())
        r["log_loss"] = float(lr[sel].mean())
        r["block_se"] = block_se(np.where(sel, lr, np.nan), games)
        for pname in ("first", "cont"):
            m = sel & (pop == pname)
            r[f"log_loss_{pname}"] = float(lr[m].mean()) if m.sum() else float("nan")
            r[f"n_{pname}"] = int(m.sum())
        for gname in ("sens1", "sens2"):
            r[f"log_loss_{gname}"] = float(lr[masks[gname]].mean())

        # --- R1 / R2 on all three gates --------------------------------------
        for gname, m in masks.items():
            t, l = m & (role < 0), m & (role > 0)
            ts = three_share(p[t]) - three_share(p[l])
            fs = float(p[l][:, JB].mean() - p[t][:, JB].mean())
            r[f"R1_three_split_{gname}"] = ts
            r[f"R2_foul_split_{gname}"] = fs
        r["R1_pass"] = bool(r["R1_three_split_main"] > 0
                            and r["R1_three_split_main"] >= ACT_3PA_SPLIT / 2)
        r["R2_pass"] = bool(r["R2_foul_split_main"] > 0
                            and r["R2_foul_split_main"] >= ACT_FOUL_SPLIT / 2)
        r["R7_pass"] = bool(
            all(r[f"R1_three_split_{g}"] > 0 for g in GATES)
            and all(r[f"R2_foul_split_{g}"] > 0 for g in GATES))

        # --- R3 team responsiveness on the fouling channel -------------------
        pq = [float(p[sel & (quint == q)][:, JB].mean()) for q in range(5)]
        nq = [int((sel & (quint == q)).sum()) for q in range(5)]
        span_p, span_a = pq[-1] - pq[0], act_q[-1] - act_q[0]
        steps = int(np.sum(np.sign(np.diff(pq)) == np.sign(np.diff(act_q))))
        r["R3_pred_quintiles"] = [round(x, 5) for x in pq]
        r["R3_span_pred"] = span_p
        r["R3_span_actual"] = span_a
        r["R3_slope_ratio"] = span_p / span_a if span_a else None
        r["R3_steps_agreeing"] = steps
        r["R3_pass"] = bool(span_a != 0 and np.sign(span_p) == np.sign(span_a)
                            and steps >= 3)
        r["R3_min_cell_n"] = min(nq)
        pqp = [float(p[sel & (quint_prior == q)][:, JB].mean()) for q in range(5)]
        sp_p, sp_a = pqp[-1] - pqp[0], act_qp[-1] - act_qp[0]
        r["R3b_prior_span_pred"] = sp_p
        r["R3b_prior_span_actual"] = sp_a
        r["R3b_prior_slope_ratio"] = sp_p / sp_a if sp_a else None
        r["R3b_prior_steps_agreeing"] = int(np.sum(np.sign(np.diff(pqp))
                                                   == np.sign(np.diff(act_qp))))
        r["R3b_pass"] = bool(sp_a != 0 and np.sign(sp_p) == np.sign(sp_a)
                             and r["R3b_prior_steps_agreeing"] >= 3)

        # --- R6 no upstream damage (full-scope arms only) --------------------
        if c["scope"] == "full" and np.isfinite(p[ref]).all():
            gaps = {cl: float(p[ref][:, j].mean() - act_ref_rate[cl])
                    for j, cl in enumerate(CLASSES)}
            r["R6_ref_max_abs_gap_pp"] = 100 * max(abs(v) for v in gaps.values())
            r["R6_ref_log_loss"] = float(lr[ref].mean())
            r["R6_scope"] = "measured"
        else:
            r["R6_ref_max_abs_gap_pp"] = None
            r["R6_ref_log_loss"] = None
            r["R6_scope"] = "gated: this arm is served only inside the window"

        # --- per-possession-type and per-role evidence -----------------------
        seg = {}
        for name, m in (("trailing", sel & (role < 0)), ("tied", sel & (role == 0)),
                        ("leading", sel & (role > 0)),
                        ("first", sel & (pop == "first")), ("cont", sel & (pop == "cont")),
                        ("bonus", sel & (te["in_bonus"].to_numpy() > 0)),
                        ("no_bonus", sel & (te["in_bonus"].to_numpy() == 0)),
                        ("sec_60_120", sel & (sec > 60)), ("sec_30_60", sel & (sec > 30) & (sec <= 60)),
                        ("sec_10_30", sel & (sec > 10) & (sec <= 30)), ("sec_0_10", sel & (sec <= 10))):
            n = int(m.sum())
            seg[name] = {"n": n, "label": cell_label(n),
                         "log_loss": float(lr[m].mean()) if n else None,
                         "pred_bonusFT": float(p[m][:, JB].mean()) if n else None,
                         "actual_bonusFT": float((y[m] == JB).mean()) if n else None,
                         "pred_3pa_share": three_share(p[m]) if n else None,
                         "actual_3pa_share": three_share(
                             np.eye(len(CLASSES))[y[m]]) if n else None}
        detail[c["cell_id"]] = {"segments": seg, "R3_quintile_n": nq,
                                "R3_actual_quintiles": act_q,
                                "R3_pred_quintiles": [round(x, 5) for x in pq],
                                "R3b_actual_prior_quintiles": act_qp,
                                "R3b_pred_prior_quintiles": [round(x, 5) for x in pqp]}

        # --- per-team spread (the multi-level rule) --------------------------
        tdf = pd.DataFrame({"team": te["offense_team_id"].to_numpy()[sel],
                            "pred": p[sel][:, JB], "act": (y[sel] == JB).astype(float)})
        tg = tdf.groupby("team").agg(n=("pred", "size"), pred=("pred", "mean"),
                                     act=("act", "mean"))
        tg = tg[tg["n"] >= 20]
        r["per_team_n_teams_ge20"] = int(len(tg))
        r["per_team_bonusFT_mae_pp"] = float(100 * (tg["pred"] - tg["act"]).abs().mean())
        r["per_team_bonusFT_corr"] = (float(np.corrcoef(tg["pred"], tg["act"])[0, 1])
                                      if len(tg) > 2 else None)
        rows.append(r)

    df = pd.DataFrame(rows)
    # --- the seed floor: |loss(seed 0) - loss(seed 1)| per spec ------------
    k = ["arm", "bundle", "fold", "cadence", "scope"]
    piv = df.pivot_table(index=k, columns="seed", values="log_loss")
    if 0 in piv.columns and 1 in piv.columns:
        floor = (piv[0] - piv[1]).abs().rename("seed_floor")
        df = df.merge(floor.reset_index(), on=k, how="left")
    else:
        df["seed_floor"] = np.nan
    df["binding_floor"] = df[["seed_floor", "block_se"]].max(axis=1)
    return df, {"actual": act, "actual_quintiles": act_q,
                "actual_reference_rates": act_ref_rate, "detail": detail}


# ===========================================================================
# 2. DURATION half
# ===========================================================================
def grade_clock(fold: str, cells: list[dict]) -> tuple[pd.DataFrame, dict]:
    te = pd.read_parquet(CK_DIR / f"window_test_{fold}.parquet")
    y = te["duration_s"].to_numpy(dtype="int64")
    cen = te["censored"].to_numpy(dtype=bool)
    rleft = te["seconds_remaining"].to_numpy(dtype="float64")
    games = te["game_id"].to_numpy()
    sd = te["score_diff"].to_numpy()
    role = np.sign(sd)
    sec = te["seconds_remaining"].to_numpy()
    unc = ~cen

    buckets = [("sec_60_120", sec > 60), ("sec_30_60", (sec > 30) & (sec <= 60)),
               ("sec_10_30", (sec > 10) & (sec <= 30)), ("sec_0_10", sec <= 10)]
    rows, detail = [], {}
    for c in cells:
        pmf = np.load(CK_DIR / "pmf" / f"{c['cell_id']}.npy").astype("float64")
        tp, _ = c3.truncate_pmf(pmf, rleft)
        crps_t = np.full(len(te), np.nan)
        crps_t[unc] = ck.crps(tp[unc], y[unc])
        ll = c3.censored_loglik_rows(pmf, y, cen, rleft)
        mean_pred, _ = ck.pmf_mean_sd(pmf)
        # The apples-to-apples quantity for R4.  `mean_pred` is the INTENDED
        # duration of the untruncated law; the actual on an uncensored row is a
        # duration that fitted inside the time left.  Inside 10 seconds those
        # are different quantities by construction (L20: the adapter returns the
        # intended duration and `loop.py` applies `min(dur, left)`), so the
        # profile is read on the TRUNCATED law's mean, which is conditioned on
        # the same event the actual row satisfies.
        mean_trunc, _ = ck.pmf_mean_sd(tp)
        r = {k: c[k] for k in ("cell_id", "arm", "scope", "feature_set",
                               "sr_floor_bucket", "fold", "n_train", "n_test",
                               "level_used_mean")}
        r["crps_trunc"] = float(np.nanmean(crps_t))
        r["crps_block_se"] = block_se(crps_t, games)
        r["censored_loglik"] = float(ll.mean())
        r["loglik_block_se"] = block_se(ll, games)
        r["pred_mean_duration"] = float(mean_pred.mean())
        r["actual_mean_duration_unc"] = float(y[unc].mean())

        # --- R4 clock profile: mean duration by bucket, and by bucket x role --
        # Predicted and actual are BOTH read on the uncensored rows of the cell,
        # so the two sides are the same population.  Both are a bound: the
        # actual `duration_s` is post-outcome (L5) and censoring is informative.
        prof, rr = {}, {}
        for name, m in buckets:
            n = int((m & unc).sum())
            prof[name] = {"n": n, "label": cell_label(n),
                          "pred": float(mean_trunc[m & unc].mean()) if n else None,
                          "pred_untruncated": float(mean_pred[m & unc].mean()) if n else None,
                          "actual_unc": float(y[m & unc].mean()) if n else None}
            for rl, tag in ((role < 0, "trail"), (role > 0, "lead")):
                mm = m & rl & unc
                nn = int(mm.sum())
                rr[f"{name}_{tag}"] = {
                    "n": nn, "label": cell_label(nn),
                    "pred": float(mean_trunc[mm].mean()) if nn else None,
                    "pred_untruncated": float(mean_pred[mm].mean()) if nn else None,
                    "actual_unc": float(y[mm].mean()) if nn else None}
        detail[c["cell_id"]] = {"profile": prof, "profile_by_role": rr}
        r["R4_max_abs_bucket_gap_s"] = max(
            abs(v["pred"] - v["actual_unc"]) for v in prof.values()
            if v["pred"] is not None and v["actual_unc"] is not None)
        t60 = rr["sec_0_10_trail"]
        l60 = rr["sec_0_10_lead"]
        r["R4_role_gap_pred_0_10"] = (t60["pred"] - l60["pred"]
                                      if t60["pred"] is not None and l60["pred"] is not None
                                      else None)
        r["R4_role_gap_actual_0_10"] = (t60["actual_unc"] - l60["actual_unc"]
                                        if t60["actual_unc"] is not None
                                        and l60["actual_unc"] is not None else None)

        # --- R5 offline proxy: mean duration by |margin| at the row ----------
        mg = []
        for k in range(0, 7):
            m = np.abs(sd) == k
            n = int(m.sum())
            nu = int((m & unc).sum())
            mg.append({"abs_margin": k, "n": n, "n_uncensored": nu,
                       "label": cell_label(nu),
                       "pred_mean_dur": float(mean_trunc[m & unc].mean()) if nu else None,
                       "actual_mean_dur_unc": float(y[m & unc].mean()) if nu else None})
        detail[c["cell_id"]]["by_abs_margin"] = mg
        ok = [x for x in mg if x["pred_mean_dur"] is not None]
        r["R5_proxy_span_pred_s"] = ok[0]["pred_mean_dur"] - ok[-1]["pred_mean_dur"]
        r["R5_proxy_span_actual_s"] = ok[0]["actual_mean_dur_unc"] - ok[-1]["actual_mean_dur_unc"]
        rows.append(r)
    return pd.DataFrame(rows), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/late_game/round1"))
    ap.add_argument("--pop", default="first",
                    help="chance population every arm is graded on; '' = both")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    global PPG
    PPG = team_ppg()
    seen, cells = set(), []
    for f in sorted(EV_DIR.glob("cells_shard*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r["cell_id"] not in seen:
                seen.add(r["cell_id"])
                cells.append(r)
    log(f"event cells: {len(cells)}")
    ev_rows, ev_extra = [], {}
    for fold in sorted({c["fold"] for c in cells}):
        df, extra = grade_event(fold, [c for c in cells if c["fold"] == fold],
                                pop_graded=a.pop)
        ev_rows.append(df)
        ev_extra[fold] = extra
        log(f"  graded event fold {fold}: {len(df)} cells")
    ev = pd.concat(ev_rows, ignore_index=True)
    ev.to_csv(out / "event_results.csv", index=False)
    (out / "event_detail.json").write_text(json.dumps(ev_extra, indent=1, default=str),
                                           encoding="utf-8")

    ck_rows, ck_extra = [], {}
    cpath = CK_DIR / "cells.json"
    if cpath.exists():
        ccells = json.loads(cpath.read_text(encoding="utf-8"))
        for fold in sorted({c["fold"] for c in ccells}):
            df, extra = grade_clock(fold, [c for c in ccells if c["fold"] == fold])
            ck_rows.append(df)
            ck_extra[fold] = extra
            log(f"  graded clock fold {fold}: {len(df)} cells")
        cl = pd.concat(ck_rows, ignore_index=True)
        cl.to_csv(out / "clock_results.csv", index=False)
        (out / "clock_detail.json").write_text(json.dumps(ck_extra, indent=1, default=str),
                                               encoding="utf-8")

    # ---- the printed table, selection fold only --------------------------
    s = ev[(ev["fold"] == SELECTION_FOLD) & (ev["seed"] == 0)].copy()
    if not len(s) or not (s["arm"] == "A").any():
        log("selection fold has no arm-A cell yet; printing nothing and exiting")
        return 0
    base = s[s["arm"] == "A"]["log_loss"].iloc[0]
    s["delta_vs_A"] = s["log_loss"] - base
    s["floors_beaten"] = -s["delta_vs_A"] / s["binding_floor"].replace(0, np.nan)
    cols = ["arm", "bundle", "log_loss", "delta_vs_A", "seed_floor", "block_se",
            "binding_floor", "floors_beaten", "R1_three_split_main",
            "R2_foul_split_main", "R1_pass", "R2_pass", "R3_slope_ratio", "R3_pass",
            "R3b_prior_slope_ratio", "R3b_pass", "R7_pass", "R6_ref_max_abs_gap_pp"]
    print("")
    print("=== FOLD 2 (SELECTION), event half, seed 0 ===")
    print(s[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    s.to_csv(out / "event_fold2_table.csv", index=False)
    if ck_rows:
        c2 = cl[cl["fold"] == SELECTION_FOLD]
        base_c = c2[c2["arm"] == "A_clk"]["crps_trunc"].iloc[0]
        c2 = c2.assign(delta_vs_A=c2["crps_trunc"] - base_c,
                       floors_beaten=lambda x: -(x["crps_trunc"] - base_c)
                       / x["crps_block_se"])
        print("")
        print("=== FOLD 2 (SELECTION), duration half ===")
        print(c2[["arm", "scope", "feature_set", "sr_floor_bucket", "n_train",
                  "crps_trunc", "delta_vs_A", "crps_block_se", "floors_beaten",
                  "censored_loglik", "loglik_block_se", "pred_mean_duration",
                  "actual_mean_duration_unc", "R4_max_abs_bucket_gap_s",
                  "R4_role_gap_pred_0_10", "R4_role_gap_actual_0_10",
                  "R5_proxy_span_pred_s", "R5_proxy_span_actual_s"]
                 ].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        c2.to_csv(out / "clock_fold2_table.csv", index=False)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
