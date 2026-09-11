"""L3 POSSESSION-OUTCOME round 4: early-season shrinkage of the as-of style rates,
plus the two tree ALIGNMENT cells round 3 left NOT RUN.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 8, committed
BEFORE this file was written to run anything.

What this trainer does and does NOT do
--------------------------------------
* The GRADER is round 3's, imported and not edited: `R1.score` (via
  `train_possession_outcome_v1`), and `conf_window_calibration` / `responsiveness_by`
  from `train_possession_outcome_v3`. Round 4 ADDS one evidence cell -- a per-week-of-season
  calibration and residual table -- as a NEW function here, and applies it to every arm
  INCLUDING the re-scored round-3 reference.
* The arms differ from round 2's reference by the SHRINKAGE and by nothing else: each shrunk
  column is a weight times ROUND 2'S OWN cached column, so the unshrunk component is
  bit-identical across arms.
* The shrinkage weight k_r is a fitted parameter of the FEATURE BUILDER, estimated by
  method-of-moments empirical Bayes from COMPLETED PRIOR SEASONS ONLY. It never sees the
  test season, never sees a score, and is not tuned. This is a walk-forward feature
  definition, not a post-hoc adjustment to model output.
* Round 3's artifacts are READ, never rewritten (worker discipline). Round 4 writes to
  `data/processed/models/possession_outcome/round4/`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("CBB_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import train_possession_outcome_v1 as R1  # noqa: E402
import train_possession_outcome_v3 as R3  # noqa: E402

from cbb_sim.analysis import leak_test as LT  # noqa: E402
from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

R3_DIR = _ROOT / "data/processed/models/possession_outcome/round3"
OUT_DIR = _ROOT / "data/processed/models/possession_outcome/round4"
SEASONS = [2022, 2023, 2024, 2025]
POP_ARM = {"first": "lgbm", "cont": "cascade"}
N_JOBS = int(_THREADS)

#: round 3's recorded reference, the number the stage-0 re-score must reproduce
R3_REF_LOG_LOSS = 1.515428

FEATURE_ARMS: tuple[str, ...] = ("G0", "G1", "G2", "G3", "G4")
#: complexity rank; G1 and G4 are TIED at 1 and G1 wins the tie (pre-registration 8.1)
FEATURE_SIMPLICITY = {"G0": 0, "G1": 1, "G4": 1, "G2": 2, "G3": 3}
FEATURE_TIEBREAK = {"G0": 0, "G1": 1, "G4": 2, "G2": 3, "G3": 4}
SHRUNK_SUFFIX = {"G1": "g1", "G2": "g2", "G3": "g3"}
NPRIOR_FEATURES = ["off_n_prior_g", "def_n_prior_g"]
REFERENCE_CELL = ("G0", "S1_monthly")
SEGMENT_THRESHOLD_PP = 0.25


# ---------------------------------------------------------------------------
# 1. Feature bundles
# ---------------------------------------------------------------------------
def feature_set_v4(arm: str, population: str) -> list[str]:
    f0 = PO.feature_set("C_plus_state", population)
    if arm == "G0":
        return f0
    if arm == "G4":
        return f0 + NPRIOR_FEATURES
    suf = SHRUNK_SUFFIX[arm]
    out = []
    for c in f0:
        style = ((c.startswith("off_") and c.endswith("_c") and not c.startswith("off_rating_"))
                 or (c.startswith("opp_def_") and c.endswith("_c")))
        out.append(c[:-1] + suf if style else c)
    return out


# ---------------------------------------------------------------------------
# 2. As-of reliability, prior-season target, and the EB weight
# ---------------------------------------------------------------------------
def _asof_panel(boxes: pd.DataFrame) -> pd.DataFrame:
    """Per (season, game_id, team_id): the as-of accumulated numerator and denominator
    mass STRICTLY BEFORE that team's own game, on both the offence and the
    defence-allowed side, plus the league's accumulation on the same window and the
    number of completed games behind each. Transcribed from `PO.build_team_form` so the
    windows are the same windows round 2's columns were built on."""
    num_cols = sorted({n for n, _ in PO.RATE_DEFS.values()})
    den_cols = sorted({d for _, d in PO.RATE_DEFS.values()})
    all_cols = sorted(set(num_cols) | set(den_cols))
    b = boxes.copy()
    b["game_date"] = pd.to_datetime(b["game_date"])
    b = b.sort_values(["season", "game_date", "game_id"], kind="stable").reset_index(drop=True)

    off = b.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    oa = PO._expanding_asof(off, ["season", "team_id"], all_cols)
    oa.columns = [f"off_{c}" for c in oa.columns]
    off = pd.concat([off[["season", "game_id", "team_id", "game_date"]], oa], axis=1)

    dfd = b.rename(columns={"team_id": "_off", "opp_id": "team_id"})
    dfd = dfd.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    da = PO._expanding_asof(dfd, ["season", "team_id"], all_cols)
    da.columns = [f"def_{c}" for c in da.columns]
    dfd = pd.concat([dfd[["season", "game_id", "team_id"]], da], axis=1)

    form = off.merge(dfd, on=["season", "game_id", "team_id"], how="left")
    day = b.groupby(["season", "game_date"], as_index=False)[all_cols].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PO._expanding_asof(day, ["season"], all_cols)
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    return form.merge(day, on=["season", "game_date"], how="left")


def _rate(num: np.ndarray, den: np.ndarray, scale: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, scale * num / np.maximum(den, 1e-9), np.nan)


def fit_k(boxes: pd.DataFrame, seasons_pool: list[int]) -> dict:
    """Method-of-moments empirical-Bayes k_r = s2_r / tau2_r, in units of denominator
    mass, from COMPLETED SEASONS in `seasons_pool` only (pre-registration 8.1).

    s2   : per-unit-of-denominator sampling variance of a single game's rate around the
           team's own season rate, pooled over teams with at least two games.
    tau2 : denominator-weighted between-team variance of season rates around the league
           season rate, MINUS the mean sampling variance those season rates carry.
    """
    out = {}
    b = boxes[boxes["season"].isin(seasons_pool)]
    for side in ("off", "def"):
        key_team = "team_id" if side == "off" else "opp_id"
        for r, (num, den) in PO.RATE_DEFS.items():
            scale = PO.RATE_SCALE[r]
            g = b[[key_team, "season", num, den]].rename(columns={key_team: "tid"})
            g = g[g[den] > 0]
            if not len(g):
                out[f"{side}|{r}"] = {"k": float("nan"), "s2": None, "tau2": None, "n_teams": 0}
                continue
            ts = g.groupby(["season", "tid"], as_index=False)[[num, den]].sum()
            ts["n_games"] = g.groupby(["season", "tid"]).size().to_numpy()
            ts["rate"] = _rate(ts[num].to_numpy(), ts[den].to_numpy(), scale)
            lgs = g.groupby("season", as_index=False)[[num, den]].sum()
            lgs["lg_rate"] = _rate(lgs[num].to_numpy(), lgs[den].to_numpy(), scale)
            ts = ts.merge(lgs[["season", "lg_rate"]], on="season", how="left")

            gg = g.merge(ts[["season", "tid", "rate", "n_games"]], on=["season", "tid"], how="left")
            gg["rate_g"] = _rate(gg[num].to_numpy(), gg[den].to_numpy(), scale)
            gg["sq"] = gg[den].to_numpy() * (gg["rate_g"].to_numpy() - gg["rate"].to_numpy()) ** 2
            multi = gg[gg["n_games"] >= 2]
            per_team = multi.groupby(["season", "tid"]).agg(ss=("sq", "sum"),
                                                            n=("sq", "size")).reset_index()
            dfree = (per_team["n"] - 1).to_numpy()
            s2 = float(per_team["ss"].to_numpy().sum() / max(dfree.sum(), 1))

            w = ts[den].to_numpy().astype("float64")
            dev = ts["rate"].to_numpy() - ts["lg_rate"].to_numpy()
            mu = float(np.average(dev, weights=w))
            var_obs = float(np.average((dev - mu) ** 2, weights=w))
            samp = float(np.average(s2 / np.maximum(w, 1e-9), weights=w))
            tau2 = var_obs - samp
            k = s2 / max(tau2, 1e-9)
            k = float(min(max(k, 0.0), 1e6))
            out[f"{side}|{r}"] = {"k": round(k, 3), "s2": round(s2, 5),
                                  "tau2_raw": round(var_obs, 5),
                                  "tau2_net": round(tau2, 5),
                                  "n_team_seasons": int(len(ts)),
                                  "seasons_pool": seasons_pool}
    return out


def build_design_v4(cache: Path | None = None) -> tuple[pd.DataFrame, dict]:
    meta_path = Path(cache).with_suffix(".meta.json") if cache else None
    if cache and Path(cache).exists():
        print(f"reusing design cache {cache}", flush=True)
        return pd.read_parquet(cache), json.loads(meta_path.read_text())

    print("reading round-3 design and boxes (READ ONLY) ...", flush=True)
    d = pd.read_parquet(R3_DIR / "design_v3.parquet")
    boxes = pd.read_parquet(R3_DIR / "boxes_first_chance.parquet")
    panel = _asof_panel(boxes)

    # --- reproduce round 2's raw centred column, as the pre-registered check -------
    checks = {}
    for side in ("off", "def"):
        for r, (num, den) in PO.RATE_DEFS.items():
            s = PO.RATE_SCALE[r]
            own = _rate(panel[f"{side}_{num}"].to_numpy(), panel[f"{side}_{den}"].to_numpy(), s)
            lgr = _rate(panel[f"lg_{num}"].to_numpy(), panel[f"lg_{den}"].to_numpy(), s)
            panel[f"{side}_rawc_{r}"] = np.where(np.isnan(own) | np.isnan(lgr), 0.0,
                                                 own - lgr).astype("float32")
            panel[f"{side}_D_{r}"] = panel[f"{side}_{den}"].to_numpy().astype("float64")
    panel["off_n_prior_g"] = panel["off_n_prior"].astype("float32")
    panel["def_n_prior_g"] = panel["def_n_prior"].astype("float32")

    # --- prior-season target and its reliability ----------------------------------
    prior = {}
    for side in ("off", "def"):
        key_team = "team_id" if side == "off" else "opp_id"
        for r, (num, den) in PO.RATE_DEFS.items():
            s = PO.RATE_SCALE[r]
            g = boxes[[key_team, "season", num, den]].rename(columns={key_team: "tid"})
            ts = g.groupby(["season", "tid"], as_index=False)[[num, den]].sum()
            lgs = g.groupby("season", as_index=False)[[num, den]].sum()
            lgs["lg_rate"] = _rate(lgs[num].to_numpy(), lgs[den].to_numpy(), s)
            ts = ts.merge(lgs[["season", "lg_rate"]], on="season", how="left")
            ts["rate"] = _rate(ts[num].to_numpy(), ts[den].to_numpy(), s)
            ts[f"{side}_priorc_{r}"] = (ts["rate"] - ts["lg_rate"]).astype("float32")
            ts[f"{side}_Dprev_{r}"] = ts[den].astype("float64")
            ts["season"] = ts["season"] + 1          # available to the NEXT season only
            prior[f"{side}|{r}"] = ts[["season", "tid", f"{side}_priorc_{r}",
                                       f"{side}_Dprev_{r}"]]

    keep = ["season", "game_id", "team_id", "off_n_prior_g", "def_n_prior_g"]
    keep += [f"{s}_rawc_{r}" for s in ("off", "def") for r in PO.RATE_DEFS]
    keep += [f"{s}_D_{r}" for s in ("off", "def") for r in PO.RATE_DEFS]
    panel = panel[keep]
    for side in ("off", "def"):
        for r in PO.RATE_DEFS:
            p = prior[f"{side}|{r}"].rename(columns={"tid": "team_id"})
            panel = panel.merge(p, on=["season", "team_id"], how="left")
            panel[f"{side}_priorc_{r}"] = panel[f"{side}_priorc_{r}"].fillna(0.0).astype("float32")
            panel[f"{side}_Dprev_{r}"] = panel[f"{side}_Dprev_{r}"].fillna(0.0).astype("float64")

    # --- the EB weight, fitted from COMPLETED PRIOR SEASONS ONLY -------------------
    ks = {}
    pooled = fit_k(boxes, [s for s in SEASONS if s < max(SEASONS)])
    for s in SEASONS:
        pool = [x for x in SEASONS if x < s]
        ks[s] = fit_k(boxes, pool) if pool else {kk: dict(v, fallback="pooled (no prior season "
                                                          "in panel; 2022 is train-only in both "
                                                          "folds)")
                                                 for kk, v in pooled.items()}

    # --- join the panel onto the design, offence side then defence side ------------
    off_cols = [c for c in panel.columns if c.startswith("off_")]
    def_cols = [c for c in panel.columns if c.startswith("def_")]
    o = panel[["season", "game_id", "team_id"] + off_cols].rename(
        columns={"team_id": "offense_team_id"})
    f = panel[["season", "game_id", "team_id"] + def_cols].rename(
        columns={"team_id": "defense_team_id"})
    d = d.merge(o, on=["season", "game_id", "offense_team_id"], how="left")
    d = d.merge(f, on=["season", "game_id", "defense_team_id"], how="left")
    for c in off_cols + def_cols:
        d[c] = d[c].fillna(0.0)

    # --- the pre-registered reproduction assertion ---------------------------------
    for r in PO.RATE_DEFS:
        for design_col, panel_col in ((f"off_{r}_c", f"off_rawc_{r}"),
                                      (f"opp_def_{r}_c", f"def_rawc_{r}")):
            diff = float(np.nanmax(np.abs(d[design_col].to_numpy().astype("float64")
                                          - d[panel_col].to_numpy().astype("float64"))))
            checks[design_col] = round(diff, 6)
            assert diff < 1e-3, (f"round-4 rebuild of {design_col} does not reproduce round 2's "
                                 f"cached column (max abs diff {diff}); the arms would then "
                                 f"differ by more than the shrinkage. ABORT.")

    # --- the shrunk columns --------------------------------------------------------
    season_arr = d["season"].to_numpy()
    for side, pfx in (("off", "off"), ("def", "opp_def")):
        for r in PO.RATE_DEFS:
            kmap = {int(sn): float(ks[int(sn)][f"{side}|{r}"]["k"]) for sn in SEASONS}
            kk = pd.Series(season_arr).map(kmap).to_numpy().astype("float64")
            D = d[f"{side}_D_{r}"].to_numpy().astype("float64")
            Dprev = d[f"{side}_Dprev_{r}"].to_numpy().astype("float64")
            w = D / np.maximum(D + kk, 1e-9)
            w2 = Dprev / np.maximum(Dprev + kk, 1e-9)
            raw = d[f"{pfx}_{r}_c"].to_numpy().astype("float64")
            pri = d[f"{side}_priorc_{r}"].to_numpy().astype("float64")
            d[f"{pfx}_{r}_g1"] = (w * raw).astype("float32")
            d[f"{pfx}_{r}_g2"] = (w * raw + (1.0 - w) * pri).astype("float32")
            d[f"{pfx}_{r}_g3"] = (w * raw + (1.0 - w) * w2 * pri).astype("float32")
            d[f"{side}_w_{r}"] = w.astype("float32")

    cols: set[str] = set()
    for pop in PO.POPULATIONS:
        for arm in FEATURE_ARMS:
            cols |= set(feature_set_v4(arm, pop))
    cols |= {f for f, _c in PO.RESPONSIVENESS_SPECS}
    cols |= {"y", "season", "game_id", "game_date", "population", "offense_team_id",
             "defense_team_id", "ncss", R3.CONF_FEATURE}
    cols |= {f"{s}_w_{r}" for s in ("off", "def") for r in PO.RATE_DEFS}
    d = d[sorted(c for c in cols if c in d.columns)]

    meta = {"reproduction_max_abs_diff": checks, "k_by_season": ks,
            "source_design": str(R3_DIR / "design_v3.parquet"),
            "source_boxes": str(R3_DIR / "boxes_first_chance.parquet")}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(cache, index=False)
        meta_path.write_text(json.dumps(meta, indent=1, default=str))
    return d, meta


# ---------------------------------------------------------------------------
# 3. The evidence cell round 4 adds (applied to EVERY arm, reference included)
# ---------------------------------------------------------------------------
WEEK_BUCKETS = (("wk0", 0, 0), ("wk1", 1, 1), ("wk2", 2, 2), ("wk3", 3, 3),
                ("wk4_7", 4, 7), ("wk8plus", 8, 99))


def per_week_table(te: pd.DataFrame, p: np.ndarray) -> list[dict]:
    """Per-week-of-season calibration and signed per-class residual. New in round 4,
    written as a NEW function so nothing rounds 1-3 scored through is edited."""
    dates = pd.to_datetime(te["game_date"])
    wk = ((dates - dates.min()).dt.days // 7).to_numpy()
    y = te["y"].to_numpy()
    rows = []
    for name, lo, hi in WEEK_BUCKETS:
        m = (wk >= lo) & (wk <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        under = n < 20_000
        gap = None
        if not under:
            cal = PO.decile_calibration(y[m], p[m])
            gated = [v["max_abs_gap_pp"] for v in cal.values()
                     if v["share_pct"] >= R1.CAL_GATE_MIN_SHARE]
            gap = round(float(max(gated)), 3) if gated else None
        resid = {c: round(float(p[m, j].mean() - (y[m] == j).mean()) * 100, 3)
                 for c, j in PO.CLASS_INDEX.items()}
        rows.append({"week": name, "n": n, "underpowered": bool(under),
                     "worst_gated_gap_pp": gap, "signed_residual_pp": resid,
                     "mean_shrink_w_off_3pa": (round(float(te["off_w_3pa"].to_numpy()[m].mean()), 4)
                                               if "off_w_3pa" in te.columns else None)})
    return rows


def quintile_slope_worst(extra: dict) -> float | None:
    """Headline responsiveness number: the own-driver slope ratio furthest from 1.0
    among the drivers Decision 8 does NOT exempt for a narrow span."""
    vals = [r.get("slope_ratio") for r in extra.get("responsiveness_own", [])
            if not r.get("exempt_narrow_span", False) and r.get("slope_ratio") is not None]
    if not vals:
        return None
    return float(min(vals, key=lambda v: -abs(v - 1.0)))



def leak_test_round4(design: pd.DataFrame) -> pd.DataFrame:
    """The standing INV-45 change-form leak test on every column round 4 adds, built on
    the same offence-side panel `R3.leak_test_adjusted` uses. Round 2's raw-centred
    columns are tested in the same run so the shrunk numbers are read next to columns
    already accepted."""
    u = pd.read_parquet(PO.DEFAULT_UNIVERSE)
    u = u[u["season"].isin(SEASONS) & u["is_d1_game"]].copy()
    u["game_date"] = pd.to_datetime(u["game_date"])
    home = u[["game_id", "season", "game_date", "home_team_id", "home_score", "away_score"]].rename(
        columns={"home_team_id": "team", "home_score": "own", "away_score": "opp"})
    away = u[["game_id", "season", "game_date", "away_team_id", "away_score", "home_score"]].rename(
        columns={"away_team_id": "team", "away_score": "own", "home_score": "opp"})
    marg = pd.concat([home, away], ignore_index=True)
    marg["margin"] = marg["own"] - marg["opp"]
    cols = ([f"off_{r}_c" for r in PO.RATE_DEFS]
            + [f"off_{r}_{sfx}" for r in PO.RATE_DEFS for sfx in SHRUNK_SUFFIX.values()]
            + NPRIOR_FEATURES)
    cols = [c for c in cols if c in design.columns]
    feat = (design.groupby(["season", "game_id", "offense_team_id"], as_index=False)[cols]
            .first().rename(columns={"offense_team_id": "team"}))
    panel = marg.merge(feat, on=["season", "game_id", "team"], how="inner")
    rows = []
    for c in cols:
        r = LT.leak_test_column(panel, c, team_col="team")
        r["column"] = c
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Cells
# ---------------------------------------------------------------------------
def planned_cells() -> list[dict]:
    cells: list[dict] = []

    def add(stage, pop, arm, fold, fa, sa, role, seed=0):
        cells.append({"stage": stage, "population": pop, "arm": arm, "fold": fold,
                      "feature_arm": fa, "scheme": sa, "role": role, "seed": seed})

    # Stage 1 -- the cheap cascade cells.
    for fa in FEATURE_ARMS:
        for fold in ("F1", "F2"):
            add(1, "cont", "cascade", fold, fa, "S1_monthly", "selection")
        add(1, "first", "cascade", "F2", fa, "S1_monthly", "interaction probe (not selection)")
    for sa in ("S1_conf_aligned", "S1_weekly"):
        add(1, "cont", "cascade", "F2", "G0", sa, "selection")
        add(1, "first", "cascade", "F2", "G0", sa, "interaction probe (not selection)")
    # Stages 2-5 -- the tree shrinkage ladder at the reference scheme, fold 2.
    for st, fa in ((2, "G1"), (3, "G4"), (4, "G2"), (5, "G3")):
        add(st, "first", "lgbm", "F2", fa, "S1_monthly", "selection")
    # Stage 6 -- the noise floor.
    add(6, "first", "lgbm", "F2", *REFERENCE_CELL, "noise floor", seed=1)
    # Stages 7-8 -- the alignment cells round 3 did not run (Decision 9c).
    add(7, "first", "lgbm", "F2", "G0", "S1_conf_aligned", "selection")
    add(8, "first", "lgbm", "F2", "G0", "S1_weekly", "selection")
    return cells


def cell_key(c: dict) -> str:
    return (f"{c['population']}|{c['fold']}|{c['arm']}|{c['feature_arm']}|{c['scheme']}"
            f"|s{c.get('seed', 0)}")


def grade(c: dict, te: pd.DataFrame, p: np.ndarray, meta: dict, dt: float) -> tuple[dict, dict, dict]:
    """One blind grading pass. Every number here comes from round 3's imported scorer
    except `per_week`, the cell round 4 adds, which every arm goes through."""
    s = R1.score(te, p)
    extra = {
        "conf_window": R3.conf_window_calibration(te, p, grade.firsts),
        "responsiveness_ncss": [R3.responsiveness_by(te, p, d_, cl) for d_, cl in R3.NCSS_SPECS],
        "responsiveness_own": [R3.responsiveness_by(te, p, d_, cl)
                               for d_, cl in PO.RESPONSIVENESS_SPECS],
        "per_week": per_week_table(te, p),
    }
    w = extra["conf_window"]["first4_conf_weeks"]
    nc = extra["conf_window"]["non_conference"]
    row = {"stage": c["stage"], "role": c["role"], "population": c["population"],
           "fold": c["fold"], "arm": c["arm"], "feature_arm": c["feature_arm"],
           "scheme": c["scheme"], "seed": c.get("seed", 0),
           "n_fits": meta.get("n_fits"), "n_test": len(te),
           "log_loss": round(s["log_loss"], 6),
           "worst_gated_gap_pp": round(s["worst_gated_gap_pp"], 3),
           "worst_gated_level_pp": round(s["worst_gated_level_pp"], 3),
           "worst_gated_shape_pp": round(s["worst_gated_shape_pp"], 3),
           "wk03_gap_pp": extra["conf_window"]["first4_season_weeks"]["worst_gated_gap_pp"],
           "nonconf_gap_pp": nc["worst_gated_gap_pp"],
           "conf4_gap_pp": w["worst_gated_gap_pp"],
           "quintile_slope_worst": quintile_slope_worst(extra),
           "calibration_pass": s["calibration_pass"],
           "responsiveness_pass": s["responsiveness_pass"],
           "ncss_slope_pass": all(r.get("pass", False) for r in extra["responsiveness_ncss"]),
           **{f"brier_{k}": round(v, 6) for k, v in s["brier"].items()},
           "fit_seconds": round(dt, 1)}
    return row, s, extra


class Checkpoint:
    def __init__(self, path: Path, enabled: bool = True):
        self.path, self.enabled = Path(path), enabled
        self.d: dict = {"grid": {}, "detail": {}, "meta": {}, "extra": {}}
        if enabled and self.path.exists():
            loaded = json.loads(self.path.read_text())
            self.d.update({k: loaded.get(k, v) for k, v in self.d.items()})
            print(f"resuming from {self.path}: {len(self.d['grid'])} cells", flush=True)

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str))
        os.replace(tmp, self.path)

    def put(self, key, row, detail, meta, extra) -> None:
        self.d["grid"][key] = row
        self.d["detail"][key] = detail
        self.d["meta"][key] = meta
        self.d["extra"][key] = extra
        self.save()


# ---------------------------------------------------------------------------
# 5. Decision rule (pre-registration 8.5)
# ---------------------------------------------------------------------------
def decide_v4(grid: pd.DataFrame, population: str, nf: float, fold: str = "F2") -> dict:
    sub = grid[(grid["population"] == population) & (grid["fold"] == fold)
               & (grid["arm"] == POP_ARM[population])].copy()
    if not len(sub):
        return {"status": "no cells"}
    ref = sub[(sub["feature_arm"] == REFERENCE_CELL[0]) & (sub["scheme"] == REFERENCE_CELL[1])
              & (sub["seed"] == 0)]
    if not len(ref):
        return {"status": "no reference cell"}
    ref = ref.iloc[0]

    def _g(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    def ladder(rows: pd.DataFrame, label: str, order: dict, key: str) -> dict:
        out = []
        for _, r in rows.iterrows():
            ll_gain = float(ref["log_loss"]) - float(r["log_loss"])
            wk = _g(r["wk03_gap_pp"]); wk_ref = _g(ref["wk03_gap_pp"])
            nc = _g(r["nonconf_gap_pp"]); nc_ref = _g(ref["nonconf_gap_pp"])
            wk_gain = None if (wk is None or wk_ref is None) else wk_ref - wk
            nc_gain = None if (nc is None or nc_ref is None) else nc_ref - nc
            gates = bool(r["calibration_pass"]) and bool(r["responsiveness_pass"])
            beats = gates and (ll_gain > nf
                               or (wk_gain is not None and wk_gain > SEGMENT_THRESHOLD_PP)
                               or (nc_gain is not None and nc_gain > SEGMENT_THRESHOLD_PP))
            # the no-shuffling clause: an adopted arm may not give back more than the
            # threshold on the other pre-registered segment
            shuffles = ((wk_gain is not None and wk_gain < -SEGMENT_THRESHOLD_PP)
                        or (nc_gain is not None and nc_gain < -SEGMENT_THRESHOLD_PP))
            out.append({key: r[key], "seed": int(r["seed"]), "log_loss": float(r["log_loss"]),
                        "gain_vs_reference": round(ll_gain, 6),
                        "wk03_gap_pp": wk, "wk03_gain_pp": (None if wk_gain is None
                                                            else round(wk_gain, 3)),
                        "nonconf_gap_pp": nc, "nonconf_gain_pp": (None if nc_gain is None
                                                                  else round(nc_gain, 3)),
                        "quintile_slope_worst": _g(r["quintile_slope_worst"]),
                        "gates_pass": gates,
                        "moves_error_between_segments": bool(shuffles),
                        "beats_reference_beyond_floor": bool(beats and not shuffles)})
        winners = [o for o in out if o["beats_reference_beyond_floor"]]
        if not winners:
            return {"ladder": label, "rows": out, "winner": REFERENCE_CELL[0 if key == "feature_arm" else 1],
                    "reason": (f"the simplest arm stands -- no more complex arm beat it by more "
                               f"than the noise floor {nf:.5f} on log loss or by more than "
                               f"{SEGMENT_THRESHOLD_PP} pp on the weeks-0-3 or non-conference "
                               f"gap without giving the other segment back. Under the "
                               f"pre-registration that is a RESULT, not a failure.")}
        best = min(winners, key=lambda o: o["log_loss"])
        eligible = [o for o in winners if o["log_loss"] <= best["log_loss"] + nf]
        win = min(eligible, key=lambda o: (order.get(o[key], 99),
                                           FEATURE_TIEBREAK.get(o[key], 99)))
        return {"ladder": label, "rows": out, "winner": win[key],
                "reason": (f"{win[key]} beats the reference beyond the floor and is the simplest "
                           f"arm within the floor of the best beater ({best['log_loss']:.6f}).")}

    fl = sub[sub["scheme"] == REFERENCE_CELL[1]].sort_values("feature_arm")
    sl = sub[(sub["feature_arm"] == REFERENCE_CELL[0])].sort_values("scheme")
    return {"population": population, "fold": fold, "noise_floor": nf,
            "reference": {"feature_arm": REFERENCE_CELL[0], "scheme": REFERENCE_CELL[1],
                          "log_loss": float(ref["log_loss"]),
                          "wk03_gap_pp": _g(ref["wk03_gap_pp"]),
                          "nonconf_gap_pp": _g(ref["nonconf_gap_pp"])},
            "feature_ladder": ladder(fl, "feature", FEATURE_SIMPLICITY, "feature_arm"),
            "scheme_ladder": ladder(sl, "scheme", R3.SCHEME_SIMPLICITY, "scheme")}


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--ckpt", default="checkpoint.json")
    ap.add_argument("--stop-at", default="", help="ET wall clock HH:MM; a tree cell is NOT "
                                                  "STARTED if the clock is past it")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--merge", default="", help="comma-separated checkpoint filenames to merge")
    ap.add_argument("--no-summary", action="store_true", help="worker process: write only "
                    "its own checkpoint, so two concurrent workers never race on the "
                    "shared summary files. The render pass writes those.")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    stop_at = None
    if a.stop_at:
        hh, mm = a.stop_at.split(":")
        now = pd.Timestamp.now()
        stop_at = now.normalize() + pd.Timedelta(hours=int(hh), minutes=int(mm))

    design, dmeta = build_design_v4(OUT_DIR / "design_v4.parquet")
    conf = CF.build_conference_flags(SEASONS)
    firsts = CF.first_conference_game_dates(conf)
    firsts = firsts[firsts["season"] == PO.FOLDS[PO.SELECTION_FOLD]["test"][0]][
        ["team_id", "first_conf_date"]]
    grade.firsts = firsts
    kbs = dmeta["k_by_season"]
    print("k_by_season:", json.dumps({str(sn): {k: v["k"]
                                                for k, v in kbs.get(sn, kbs.get(str(sn), {})).items()}
                                      for sn in SEASONS}, default=str)[:900], flush=True)

    ckpt = Checkpoint(OUT_DIR / a.ckpt, enabled=True)
    rows, detail, metas, extras = [], {}, {}, {}
    not_run: list[dict] = []

    # --- stage 0: re-score round 3's reference through the round-4 grader ----------
    ref_npy = R3_DIR / "ref_pred_first_F2_seed0.npy"
    if ref_npy.exists():
        _tr, te_ref = PO.fold_slices(design, "F2", "first")
        p_ref = np.load(ref_npy).astype("float64")
        assert len(p_ref) == len(te_ref), (f"stored reference predictions ({len(p_ref)}) do not "
                                           f"match the fold-2 `first` test slice ({len(te_ref)})")
        c0 = {"stage": 0, "role": "reference (re-scored from round 3)", "population": "first",
              "fold": "F2", "arm": "lgbm", "feature_arm": "G0", "scheme": "S1_monthly", "seed": 0}
        row0, s0, extra0 = grade(c0, te_ref, p_ref, {"n_fits": 6}, 0.0)
        repro = abs(row0["log_loss"] - R3_REF_LOG_LOSS)
        assert repro < 1e-5, (f"the re-scored reference gives {row0['log_loss']} against round 3's "
                              f"recorded {R3_REF_LOG_LOSS}; the grader is not additive. ABORT.")
        print(f"stage 0 reference re-score OK: log loss {row0['log_loss']:.6f} reproduces round 3 "
              f"to {repro:.2e}", flush=True)
        k0 = cell_key(c0)
        rows.append(row0); detail[k0], metas[k0], extras[k0] = s0, {"n_fits": 6}, extra0
        ckpt.put(k0, row0, s0, {"n_fits": 6, "reproduction_abs_diff": repro}, extra0)

    stages = {int(x) for x in a.stages.split(",") if x.strip()}
    cells = [c for c in planned_cells() if c["stage"] in stages]
    if not a.render_only:
        for c in cells:
            key = cell_key(c)
            if key in ckpt.d["grid"]:
                print(f"  {key:52s} (checkpoint)", flush=True)
                continue
            if stop_at is not None and pd.Timestamp.now() >= stop_at and c["arm"] == "lgbm":
                not_run.append({**c, "reason": f"wall clock past the pre-registered stop "
                                               f"{a.stop_at} ET"})
                print(f"  {key:52s} NOT RUN (clock)", flush=True)
                continue
            feats = feature_set_v4(c["feature_arm"], c["population"])
            tr, te = PO.fold_slices(design, c["fold"], c["population"])
            cuts = R3.refit_dates(c["scheme"], te, conf)
            t0 = time.time()
            p, meta = R3.fit_predict_walkforward(c["arm"], tr, te, feats, cuts,
                                                 seed=c.get("seed", 0))
            row, s, extra = grade(c, te, p, meta, time.time() - t0)
            rows.append(row); detail[key], metas[key], extras[key] = s, meta, extra
            ckpt.put(key, row, s, meta, extra)
            print(f"  {key:52s} n_fits {row['n_fits']:3d}  logloss {row['log_loss']:.6f}  "
                  f"wk03 {row['wk03_gap_pp']}  nonconf {row['nonconf_gap_pp']}  "
                  f"[{row['fit_seconds']:.0f}s]", flush=True)

    # --- merge every checkpoint this round wrote ----------------------------------
    merged = dict(ckpt.d["grid"])
    mdetail, mmeta, mextra = dict(ckpt.d["detail"]), dict(ckpt.d["meta"]), dict(ckpt.d["extra"])
    for name in [x for x in a.merge.split(",") if x.strip()]:
        p_ = OUT_DIR / name
        if p_.exists() and p_.name != a.ckpt:
            other = json.loads(p_.read_text())
            merged.update(other.get("grid", {})); mdetail.update(other.get("detail", {}))
            mmeta.update(other.get("meta", {})); mextra.update(other.get("extra", {}))
    if a.no_summary:
        print(f"worker done: {len(merged)} cells in {a.ckpt}  "
              f"runtime={(time.time()-t_start)/3600:.2f}h", flush=True)
        return
    grid = pd.DataFrame(list(merged.values()))
    if not len(grid):
        print("no cells", flush=True)
        return
    # every pre-registered cell the merged grid does not hold is NOT RUN, never a result
    have = set(merged)
    for c in planned_cells():
        if cell_key(c) not in have and not any(cell_key(c) == cell_key(x) for x in not_run):
            not_run.append({**c, "reason": "not reached inside the pre-registered wall clock "
                                           "(round 4 section 8.7 drop order)"})

    # --- noise floor ---------------------------------------------------------------
    floors = {}
    for pop in ("first", "cont"):
        sub = grid[(grid["population"] == pop) & (grid["fold"] == "F2")
                   & (grid["arm"] == POP_ARM[pop]) & (grid["feature_arm"] == REFERENCE_CELL[0])
                   & (grid["scheme"] == REFERENCE_CELL[1])]
        b = sub[sub["seed"] == 0]
        alt = sub[sub["seed"] == 1]
        if not len(b):
            continue
        b = b.iloc[0]
        _tr, te_ = PO.fold_slices(design, "F2", pop)
        se = None
        if pop == "first" and ref_npy.exists():
            se = round(PO.block_bootstrap_se(te_, np.load(ref_npy).astype("float64"),
                                             n_rep=R1.N_BOOTSTRAP), 6)
        spread = (round(abs(float(b["log_loss"]) - float(alt.iloc[0]["log_loss"])), 6)
                  if len(alt) else None)
        f = {"population": pop, "seed0_log_loss": float(b["log_loss"]),
             "seed1_log_loss": float(alt.iloc[0]["log_loss"]) if len(alt) else None,
             "seed_spread": spread,
             "seed0_wk03_gap_pp": b["wk03_gap_pp"],
             "seed1_wk03_gap_pp": alt.iloc[0]["wk03_gap_pp"] if len(alt) else None,
             "seed0_nonconf_gap_pp": b["nonconf_gap_pp"],
             "seed1_nonconf_gap_pp": alt.iloc[0]["nonconf_gap_pp"] if len(alt) else None,
             "block_bootstrap_se": se, "n_seeds": 2 if len(alt) else 1,
             "partial_reason": ("2 seeds against the 5 round 1 pre-registered; the round-4 "
                                "pre-registration asks for one second seed and the run is "
                                "wall-clock bound" if len(alt) else
                                "SECOND SEED NOT RUN -- the floor falls back to the block "
                                "bootstrap SE alone and is labelled PARTIAL")}
        cands = [x for x in (spread, se) if x is not None]
        f["applied"] = max(cands) if cands else None
        floors[pop] = f

    verdicts = {}
    for pop in ("first", "cont"):
        nf = floors.get(pop, {}).get("applied")
        verdicts[pop] = decide_v4(grid, pop, float(nf) if nf is not None else 0.0)

    leak = leak_test_round4(design)
    leak.to_csv(OUT_DIR / "leak_test_round4.csv", index=False)

    grid.to_csv(OUT_DIR / "grid_results.csv", index=False)
    (OUT_DIR / "metrics_detail.json").write_text(json.dumps(mdetail, indent=1, default=str))
    (OUT_DIR / "scheme_meta.json").write_text(json.dumps(mmeta, indent=1, default=str))
    (OUT_DIR / "extra_metrics.json").write_text(json.dumps(mextra, indent=1, default=str))
    (OUT_DIR / "noise_floor.json").write_text(json.dumps(floors, indent=1, default=str))
    (OUT_DIR / "verdict.json").write_text(json.dumps(verdicts, indent=1, default=str))
    (OUT_DIR / "not_run.json").write_text(json.dumps(not_run, indent=1, default=str))
    (OUT_DIR / "shrinkage_k.json").write_text(json.dumps(dmeta, indent=1, default=str))
    (OUT_DIR / "run_meta.json").write_text(json.dumps(
        {"run_at": time.strftime("%Y-%m-%d %H:%M"), "stages": sorted(stages),
         "ckpt": a.ckpt, "threads": _THREADS,
         "runtime_hours": round((time.time() - t_start) / 3600.0, 3),
         "n_cells": len(grid)}, indent=1, default=str))
    print(f"\nwrote {OUT_DIR}  cells={len(grid)}  "
          f"runtime={(time.time()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
