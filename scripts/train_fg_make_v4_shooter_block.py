#!/usr/bin/env python
"""
train_fg_make_v4_shooter_block.py -- fg_make ROUND 4: the SHOOTER BLOCK,
re-baked from scratch on the corrected `shot_shooter_id` label.

    .venv/Scripts/python.exe scripts/train_fg_make_v4_shooter_block.py

Pre-registration: `docs/models/fg_make/experiments.md` section 19, committed
BEFORE this ran. Evidence: `docs/tests/fg_make_shooter_skill_2026-09-10.md`
(`scripts/diag_fg_make_shooter_skill_v1.py`).

WHAT IS FIXED (19.1): shooter key `shot_shooter_id`; LightGBM per class with
round 1's frozen F1-only ladder; the S-C engine-safe state block; the team
block; the S1 monthly schedule; F2 selection; `FG.score()` unchanged; v2
possessions.

WHAT IS SELECTED (19.2): the SHOOTER REPRESENTATION, six arms
B0 < B1 < B2 < BR < B3 < B4.

THE ONE FITTED PARAMETER (19.3): the shrinkage strength `m` per class, fitted
on FOLD 1's test season (2024) by minimising the log loss of the shrunk rate
used directly as a probability, then FROZEN. F2 never sees the fit.

`fg_make.py` is READ, never modified: the arms are assembled with
`FG.design_matrix` on explicit feature lists and fitted with `FG.LgbmArm`, so
this round adds no feature-set constants to the shared module and cannot
collide with another lane.

Artifacts (a new directory; nothing existing is overwritten):
    data/processed/models/fg_make/round4/<arm>/<class>_<refit_date>.joblib
    data/processed/models/fg_make/round4/<arm>/manifest_<class>.json
    data/processed/models/fg_make/round4/run_report.json
    data/processed/models/fg_make/round4/m_fitted.json
    data/processed/models/fg_make/round4/slot_source_v1.parquet   (engine slots)
    data/processed/models/fg_make/design_v4_extra.parquet          (cache)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.analysis import leak_test as LK  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.pbp import events as PBP  # noqa: E402

FG_DIR = Path("data/processed/models/fg_make")
R4 = FG_DIR / "round4"
DESIGN = FG_DIR / "design_v2_shotshooter.parquet"
EVENTS = FG_DIR / "events_v2_shotshooter.parquet"
EXTRA_CACHE = FG_DIR / "design_v4_extra_v2.parquet"
SEASONS = [2022, 2023, 2024, 2025]
SHOOTER_KEY = "shot_shooter_id"

#: 19.3, fixed in the pre-registration.
M_GRID = (5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 1000, 2000)
#: The method-of-moments values from the step-2 evidence, reported as an
#: independent check and NEVER used (they are pooled over 2022-2025).
MOM_M = {"FGA_rim": 50.8, "FGA_jump2": 152.77, "FGA_3": 298.94}

COMMON = list(FG.TEAM_FEATURES) + list(FG.R2_SAFE_STATE)
SHOOTER_BLOCK: dict[str, list[str]] = {
    "B0": [],
    "B1": ["shooter_shrunk_dev_c"],
    "B2": ["shooter_shrunk_dev_c", "shooter_att_c", "prior_season_make_c",
           "prior_season_att_c", "has_prior_season"],
    "BR": list(FG.SHOOTER_FEATURES),
    "B3": ["shooter_shrunk_dev_c", "shooter_att_c", "prior_season_make_c",
           "prior_season_att_c", "has_prior_season",
           "sh_share_rim", "sh_share_jump2", "sh_share_three", "sh_assisted_share"],
    "B4": ["shooter_shrunk_dev_c", "shooter_att_c", "prior_season_make_c",
           "prior_season_att_c", "has_prior_season",
           "sh_share_rim", "sh_share_jump2", "sh_share_three", "sh_assisted_share",
           "five_shrunk_dev_rim", "five_shrunk_dev_three"],
}
ARMS = ("B0", "B1", "B2", "BR", "B3", "B4")
SIMPLICITY = {"B0": 0, "B1": 1, "B2": 2, "BR": 3, "B3": 4, "B4": 5}
FLOOR_ARM = "B1"
#: 19.6: B4 needs a runtime lineup aggregate the engine cannot express.
UNSERVABLE = frozenset({"B4"})
LABEL = {"B0": "R4_B0_no_shooter", "B1": "R4_B1_shrunk", "B2": "R4_B2_prior_counts",
         "BR": "R4_BR_incumbent", "B3": "R4_B3_mix", "B4": "R4_B4_spacing"}

#: The columns the engine must carry per SLOT for an arm to be servable.
SLOT_EXPORT_PER_CLASS = ("shooter_shrunk_dev_c", "prior_season_att_c")
SLOT_EXPORT_SHARED = ("sh_share_rim", "sh_share_jump2", "sh_share_three",
                      "sh_assisted_share")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ===========================================================================
# 1. the extra shooter features
# ===========================================================================
def assisted_asof(events: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Per (season, shooter, game): as-of share of the shooter's MADE field
    goals that were assisted, expanding over games strictly before this one.

    Built from the feed's own `shot_assisted` flag, keyed on `shot_shooter_id`,
    at the (game, player) level -- so it never has to be aligned to the event
    table's row order and cannot silently drift from it. Strictly pre-shot: the
    expanding sum excludes the current game entirely.

    THE MISSINGNESS LEAK THIS CONSTRUCTION EXISTS TO AVOID (round 4, found and
    fixed before the closed loop). The first version of this function built the
    panel from MADE field goals only. A (game, shooter) pair then had a row if
    and only if the shooter MADE at least one field goal in THAT game, so on
    the 3.55% of design rows whose shooter went 0-for-the-game the join missed
    and the feature fell back to 0.0 -- and those rows have a realised make
    rate of EXACTLY 0.00000 by construction. The value was honest and the
    change-form leak test passed at |corr| <= 0.026; the outcome information
    travelled through the PRESENCE OF THE ROW, which no value-based test can
    see. It was worth 451 / 839 noise floors of log loss on rim / three.

    The fix is to build the panel over every ATTEMPT, so a row exists exactly
    when the shooter took a field goal in that game -- which is precisely the
    condition under which the design has a row to join it to. `mk` is then
    allowed to be 0 for a game, which is what a 0-for-N night is."""
    frames = []
    for s in SEASONS:
        p = pd.read_parquet(
            Path(PBP.DEFAULT_PBP_DIR) / f"plays_{s}.parquet",
            columns=["gameId", "id", "shot_shooter_id", "shot_assisted", "shot_made"])
        p = p.drop_duplicates(subset=["gameId", "id"], keep="first")
        sh = pd.to_numeric(p["shot_shooter_id"], errors="coerce")
        made = p["shot_made"]
        if made.dtype == object:
            made = made.map({True: True, False: False})
        made = made.astype("boolean").fillna(False).to_numpy()
        asst = p["shot_assisted"]
        if asst.dtype == object:
            asst = asst.map({True: True, False: False})
        asst = asst.astype("boolean").fillna(False).to_numpy()
        # EVERY attempt with a shooter id, not only the makes.
        keep = np.isfinite(sh.to_numpy())
        frames.append(pd.DataFrame({
            "season": s,
            "cbbd_game_id": p.loc[keep, "gameId"].to_numpy(),
            "shooter_id": sh[keep].to_numpy().astype("int64"),
            "fga": 1.0,
            "mk": made[keep].astype("float64"),
            "mk_asst": (made[keep] & asst[keep]).astype("float64"),
        }))
        log(f"  assisted flag: season {s}, {int(keep.sum()):,} attempts "
            f"({int(made[keep].sum()):,} made)", t0)
    a = pd.concat(frames, ignore_index=True)
    a = a.groupby(["season", "cbbd_game_id", "shooter_id"], as_index=False)[
        ["fga", "mk", "mk_asst"]].sum()

    dates = (events[["season", "cbbd_game_id", "game_date"]]
             .drop_duplicates(subset=["season", "cbbd_game_id"]))
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    a = a.merge(dates, on=["season", "cbbd_game_id"], how="inner")
    a = a.sort_values(["season", "shooter_id", "game_date", "cbbd_game_id"],
                      kind="stable").reset_index(drop=True)
    asof = PM.expanding_asof(a, ["season", "shooter_id"], ["fga", "mk", "mk_asst"])
    out = pd.concat([a[["season", "shooter_id", "cbbd_game_id"]],
                     asof[["fga", "mk", "mk_asst"]].rename(
                         columns={"fga": "fga_asof", "mk": "mk_asof",
                                  "mk_asst": "mk_asst_asof"})], axis=1)
    out["sh_assisted_share"] = np.where(out["mk_asof"].to_numpy() > 0,
                                        out["mk_asst_asof"] / out["mk_asof"].clip(lower=1),
                                        0.0)
    return out[["season", "shooter_id", "cbbd_game_id", "sh_assisted_share",
                "mk_asof", "fga_asof"]]


def build_extra(design: pd.DataFrame, events: pd.DataFrame, m: dict[str, float],
                t0: float) -> pd.DataFrame:
    """Every round-4 feature that is not already in the design."""
    d = design
    cls = d["shot_class"].to_numpy(dtype=object)
    mvec = np.array([m[c] for c in cls], dtype="float64")
    team = d["off_make_raw"].to_numpy(dtype="float64")
    mk = d["shooter_mk_c"].to_numpy(dtype="float64")
    att = d["shooter_att_c"].to_numpy(dtype="float64")
    shrunk = (mvec * team + mk) / (mvec + att)
    out = pd.DataFrame(index=d.index)
    out["shooter_shrunk_dev_c"] = (shrunk - team).astype("float32")

    # ---- as-of shot mix -------------------------------------------------
    sf = FG.shooter_form(events)
    share_cols = ["att_rim", "att_jump2", "att_three", "att_all"]
    sm = sf[["season", "shooter_id", "game_id", *share_cols]].copy()
    tot = sm["att_all"].to_numpy(dtype="float64")
    for k in ("rim", "jump2", "three"):
        v = sm[f"att_{k}"].to_numpy(dtype="float64")
        sm[f"sh_share_{k}"] = np.where(tot > 0, v / np.maximum(tot, 1e-9), 0.0)
    j = d[["season", "shooter_id", "game_id"]].merge(
        sm[["season", "shooter_id", "game_id", "sh_share_rim", "sh_share_jump2",
            "sh_share_three"]], on=["season", "shooter_id", "game_id"], how="left")
    for k in ("rim", "jump2", "three"):
        out[f"sh_share_{k}"] = j[f"sh_share_{k}"].fillna(0.0).to_numpy().astype("float32")
    log("shot-mix shares joined", t0)

    # ---- assisted share --------------------------------------------------
    asst = assisted_asof(events, t0)
    j = d[["season", "shooter_id", "cbbd_game_id"]].merge(
        asst, on=["season", "shooter_id", "cbbd_game_id"], how="left")
    miss = j["sh_assisted_share"].isna().to_numpy()
    cover = float((~miss).mean() * 100)
    # THE MISSINGNESS GUARD (see `assisted_asof`). A join that misses on rows
    # selected by the OUTCOME is a leak no value-based test can see, so the
    # residual miss rate and the realised make rate on the missed rows are
    # asserted here, in code, rather than eyeballed.
    y_miss = float(d["y"].to_numpy()[miss].mean()) if miss.any() else float("nan")
    log(f"assisted share joined, {cover:.4f}% of design rows matched; "
        f"make rate on the {int(miss.sum()):,} unmatched rows {y_miss:.5f} "
        f"(class mean {float(d['y'].mean()):.5f})", t0)
    if miss.mean() > 0.001:
        raise AssertionError(
            f"sh_assisted_share misses {miss.mean() * 100:.3f}% of design rows; the panel "
            "must have one row per (shooter, game) in which he ATTEMPTED a field goal, "
            "or the missingness itself carries the outcome (round-4 leak)")
    out["sh_assisted_share"] = j["sh_assisted_share"].fillna(0.0).to_numpy().astype("float32")

    # ---- B4 spacing: the other four on the floor -------------------------
    # Plain merges rather than a MultiIndex reindex: the id columns arrive from
    # three different tables with three different dtypes and a silent dtype
    # mismatch in a reindex would return all-NaN and read as "no teammates on
    # the floor", which is exactly the kind of quiet defect this project keeps
    # finding. Every join rate is reported.
    tf = FG.team_shot_form(events, ES.load_universe(require_pbp_complete=True))
    tf = tf.drop_duplicates(subset=["game_id", "team_id"])
    sfu = sf.drop_duplicates(subset=["game_id", "shooter_id"])
    off_home = d["offense_is_home"].to_numpy().astype(bool)
    gid = d["game_id"].to_numpy()
    own = d["shooter_id"].to_numpy(dtype="int64")
    base = pd.DataFrame({"game_id": gid, "team_id": d["off_team_id"].to_numpy()})
    n_join = 0
    for key in ("rim", "three"):
        t = tf[["game_id", "team_id", f"off_mk_{key}", f"off_att_{key}"]].copy()
        t["game_id"] = t["game_id"].astype("int64")
        t["team_id"] = pd.to_numeric(t["team_id"], errors="coerce").astype("int64")
        b = base.copy()
        b["game_id"] = b["game_id"].astype("int64")
        b["team_id"] = pd.to_numeric(b["team_id"], errors="coerce").astype("int64")
        tj = b.merge(t, on=["game_id", "team_id"], how="left")
        t_att = tj[f"off_att_{key}"].to_numpy(dtype="float64")
        t_mk = tj[f"off_mk_{key}"].to_numpy(dtype="float64")
        cls_mean = float(d.loc[d["class_key"] == key, "y"].mean())
        t_rate = np.where(np.isfinite(t_att) & (t_att > 0),
                          t_mk / np.maximum(t_att, 1e-9), cls_mean)
        mm = float(m["FGA_rim" if key == "rim" else "FGA_3"])
        p = sfu[["game_id", "shooter_id", f"mk_{key}", f"att_{key}"]].copy()
        p["game_id"] = p["game_id"].astype("int64")
        p["shooter_id"] = p["shooter_id"].astype("int64")
        p = p.rename(columns={"shooter_id": "pid"})
        acc = np.zeros(len(d), dtype="float64")
        cnt = np.zeros(len(d), dtype="float64")
        for i in range(1, 6):
            pid = np.where(off_home, d[f"home_on_{i}"].to_numpy(dtype="float64"),
                           d[f"away_on_{i}"].to_numpy(dtype="float64"))
            ok = np.isfinite(pid) & (pid != own.astype("float64"))
            q = pd.DataFrame({"game_id": gid.astype("int64"),
                              "pid": np.where(ok, pid, -1).astype("int64")})
            qj = q.merge(p, on=["game_id", "pid"], how="left")
            p_mk = qj[f"mk_{key}"].to_numpy(dtype="float64")
            p_att = qj[f"att_{key}"].to_numpy(dtype="float64")
            good = ok & np.isfinite(p_mk) & np.isfinite(p_att)
            dev = (mm * t_rate + np.nan_to_num(p_mk)) / (mm + np.nan_to_num(p_att)) - t_rate
            acc += np.where(good, dev, 0.0)
            cnt += good
            n_join += int(good.sum())
        out[f"five_shrunk_dev_{key}"] = np.where(
            cnt > 0, acc / np.maximum(cnt, 1), 0.0).astype("float32")
        out[f"five_n_{key}"] = cnt.astype("float32")
        log(f"  spacing {key}: mean teammates resolved {cnt.mean():.2f} of 4, "
            f"{(cnt == 0).mean() * 100:.2f}% of rows with none", t0)
    log(f"B4 spacing block built ({n_join:,} teammate lookups resolved)", t0)
    return out


# ===========================================================================
# 2. the one fitted parameter: m, on FOLD 1's test season
# ===========================================================================
def fit_m(design: pd.DataFrame) -> dict:
    """19.3. One `m` per class, chosen on fold 1's TEST season by the log loss
    of the shrunk rate used directly as a probability. F2 never sees this."""
    _, te = FG.fold_slices(design, "F1")
    out: dict = {}
    for c in FG.SHOT_CLASSES:
        s = FG.class_slice(te, c)
        y = s["y"].to_numpy()
        team = s["off_make_raw"].to_numpy(dtype="float64")
        mk = s["shooter_mk_c"].to_numpy(dtype="float64")
        att = s["shooter_att_c"].to_numpy(dtype="float64")
        rows = []
        for m in M_GRID:
            p = (m * team + mk) / (m + att)
            p = np.clip(p, FG.EPS_P, 1 - FG.EPS_P)
            ll = PM.log_loss(y, np.column_stack([1 - p, p]))
            rows.append({"m": float(m), "f1_test_log_loss": round(float(ll), 6)})
        best = min(rows, key=lambda r: r["f1_test_log_loss"])
        out[c] = {"m": best["m"], "f1_test_log_loss": best["f1_test_log_loss"],
                  "grid": rows, "n_f1_test": int(len(s)),
                  "mom_reference_not_used": MOM_M[c]}
    return out


# ===========================================================================
# 3. one arm's S1 schedule
# ===========================================================================
def fit_s1(design: pd.DataFrame, arm: str, params: dict, seed: int,
           export: bool, m: dict) -> tuple[dict, dict]:
    tr_all, te_all = FG.fold_slices(design, "F2")
    scores, segs = {}, {}
    out_dir = R4 / arm
    if export:
        out_dir.mkdir(parents=True, exist_ok=True)
    for c in FG.SHOT_CLASSES:
        feats = COMMON + SHOOTER_BLOCK[arm]
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        te_dates = pd.to_datetime(te["game_date"])
        tr_dates = pd.to_datetime(tr["game_date"])
        cuts = PO.month_boundaries(te_dates)
        cols = [*feats, "y"]
        p_s1 = np.zeros((len(te), len(FG.CLASSES)), dtype="float64")
        segments = []
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None
                   else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if not seg.any():
                continue
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, cols]
            rows = tr[cols] if not len(prior) else pd.concat(
                [tr[cols], prior], ignore_index=True)
            t1 = time.time()
            X = FG.design_matrix(rows, feats)
            mdl = FG.LgbmArm(seed=seed, params=params[c]).fit(X, rows["y"].to_numpy())
            p_s1[seg] = mdl.predict_proba(FG.design_matrix(te.loc[seg], feats))
            max_train = (tr_dates.max() if not before.any()
                         else max(tr_dates.max(), te_dates[before].max()))
            info = {"refit_date": str(cut.date()), "n_train": int(len(rows)),
                    "n_train_from_test_season": int(len(prior)),
                    "n_scored": int(seg.sum()),
                    "max_train_date": str(pd.Timestamp(max_train).date()),
                    "fit_s": round(time.time() - t1, 1)}
            if export:
                fname = f"{c}_{cut.date()}.joblib"
                joblib.dump({"arm": "lgbm", "round4_arm": arm, "scheme": "S1",
                             "shooter_key": SHOOTER_KEY, "feature_set": LABEL[arm],
                             "features": feats, "model": mdl, "fold": "F2",
                             "shot_class": c, "adopted": False,
                             "refit_date": str(cut.date()),
                             "max_train_date": info["max_train_date"],
                             "possessions_version": "v2",
                             "shrinkage_m": m[c]["m"] if arm != "B0" else None,
                             "servable": arm not in UNSERVABLE,
                             "note": "fg_make round 4, experiments.md section 19: "
                                     f"shooter-block arm {arm} ({LABEL[arm]})"},
                            out_dir / fname)
                info["path"] = fname
            segments.append(info)
        if (p_s1.sum(axis=1) == 0).any():
            raise AssertionError("S1 left test rows unscored; month partition is not a cover")
        scores[c] = FG.score(te, p_s1)
        segs[c] = segments
        if export:
            (out_dir / f"manifest_{c}.json").write_text(json.dumps({
                "model": "fg_make", "scheme": "S1", "fold": "F2", "season": 2025,
                "key": c, "arm": arm, "feature_set": LABEL[arm], "features": feats,
                "shooter_key": SHOOTER_KEY,
                "shrinkage_m": m[c]["m"] if arm != "B0" else None,
                "servable": arm not in UNSERVABLE,
                "artifacts": [{k: v for k, v in s.items()
                               if k in ("refit_date", "path", "max_train_date", "n_train")}
                              for s in segments],
            }, indent=2), encoding="utf-8")
        print(f"  {arm} {c}: ll {scores[c]['log_loss']:.6f} "
              f"calib {scores[c]['calib_worst_gap_pp']:.3f}pp "
              f"D8 {'PASS' if scores[c]['resp_pass_decision8'] else 'FAIL'} "
              f"({len(segments)} refits, {sum(s['fit_s'] for s in segments):.0f}s)",
              flush=True)
    return scores, segs


# ===========================================================================
# 4. the leak test on the NEW features
# ===========================================================================
def leak(design: pd.DataFrame) -> dict:
    """Change-form leak test (INV-45 / `CLAUDE.md`'s honest-backtest rule) on
    every NEW round-4 feature, grouped on the SHOOTER rather than the team.

    One panel row per (shooter, game); the feature is constant within a game by
    construction. The margin is the **verified final margin from the shooting
    team's point of view**, taken from `eval.reference.load_actual_games` -- NOT
    from the design's own `score_diff`, which L27 proved is the post-shot score
    on the attempt's own row and would put an outcome column on both sides of
    the correlation."""
    from cbb_sim.eval import reference as REF
    cols = ["shooter_shrunk_dev_c", "prior_season_att_c", "sh_share_rim",
            "sh_share_jump2", "sh_share_three", "sh_assisted_share",
            "five_shrunk_dev_rim", "five_shrunk_dev_three"]
    act = pd.concat([REF.load_actual_games(s)[["game_id", "home_score", "away_score"]]
                     for s in SEASONS], ignore_index=True).drop_duplicates("game_id")
    out = {}
    for c in FG.SHOT_CLASSES:
        s = FG.class_slice(design, c)
        panel = s.groupby(["season", "shooter_id", "game_id"], as_index=False).agg(
            {**{k: "first" for k in cols}, "game_date": "first",
             "offense_is_home": "first"})
        panel = panel.merge(act, on="game_id", how="inner")
        hs = panel["home_score"].to_numpy(dtype="float64")
        as_ = panel["away_score"].to_numpy(dtype="float64")
        panel["margin"] = np.where(panel["offense_is_home"].to_numpy().astype(bool),
                                   hs - as_, as_ - hs)
        panel = panel.rename(columns={"shooter_id": "team"})
        res = LK.run_leak_test(panel, cols, team_col="team", season_col="season",
                               date_col="game_date", margin_col="margin")
        out[c] = res[res["season"] == "ALL"].to_dict(orient="records")
        out[f"{c}__by_season"] = res.to_dict(orient="records")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--merge-report", default=None,
                    help="carry arms (and the noise floor) forward from an EARLIER "
                         "round-4 report for every arm not named in --arms, so a "
                         "feature repair can re-run only the affected arms without "
                         "re-fitting the ones it cannot have touched")
    ap.add_argument("--report-name", default="run_report.json")
    a = ap.parse_args()
    t0 = time.time()
    R4.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="fg_make round 4")

    ladder = json.loads((FG_DIR / f"lgbm_ladder_{a.version}.json").read_text(encoding="utf-8"))
    params = {c: dict(v) for c, v in ladder["frozen_params"].items()}

    # `add_round2_state` is deliberately NOT called: no round-4 arm carries
    # `score_diff_pre` or the round-2 indicators, and building them would put a
    # column derived from the post-outcome `score_diff` into the frame for no
    # reason (L27).
    design = pd.read_parquet(DESIGN)
    log(f"design {design.shape} (shot_shooter_id)", t0)

    m = fit_m(design)
    (R4 / "m_fitted.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    for c in FG.SHOT_CLASSES:
        print(f"  fitted m[{c}] = {m[c]['m']:g} "
              f"(F1-test ll {m[c]['f1_test_log_loss']:.6f}; "
              f"MoM reference {m[c]['mom_reference_not_used']}, not used)", flush=True)

    if EXTRA_CACHE.exists():
        extra = pd.read_parquet(EXTRA_CACHE)
        log(f"extra-feature cache hit: {EXTRA_CACHE}", t0)
    else:
        events = pd.read_parquet(EVENTS)
        extra = build_extra(design, events, {c: m[c]["m"] for c in FG.SHOT_CLASSES}, t0)
        extra.to_parquet(EXTRA_CACHE, index=False)
        del events
    for c in extra.columns:
        design[c] = extra[c].to_numpy()
    log(f"extra features attached: {list(extra.columns)}", t0)

    # ---- the engine's slot source, exported once ------------------------
    slot = design[["shooter_id", "game_date", "shot_class", "class_key",
                   *SLOT_EXPORT_PER_CLASS, *SLOT_EXPORT_SHARED]].copy()
    slot = slot.drop_duplicates(subset=["shooter_id", "game_date", "class_key"])
    wide = None
    for key in ("rim", "jump2", "three"):
        s = slot[slot["class_key"] == key][
            ["shooter_id", "game_date", *SLOT_EXPORT_PER_CLASS, *SLOT_EXPORT_SHARED]]
        s = s.rename(columns={c: f"{c}__{key}" for c in SLOT_EXPORT_PER_CLASS})
        wide = s if wide is None else wide.merge(
            s.drop(columns=list(SLOT_EXPORT_SHARED)),
            on=["shooter_id", "game_date"], how="outer")
    wide.to_parquet(R4 / "slot_source_v2.parquet", index=False)
    log(f"slot source exported: {wide.shape} -> {R4}/slot_source_v2.parquet", t0)

    lk = leak(design)
    (R4 / "leak_test.json").write_text(json.dumps(lk, indent=2, default=str),
                                       encoding="utf-8")
    worst = {}
    for c in FG.SHOT_CLASSES:
        for r in lk[c]:
            worst[r["column"]] = max(worst.get(r["column"], 0.0),
                                     abs(float(r["corr_asjoined"]))
                                     if np.isfinite(r["corr_asjoined"]) else 0.0)
    print("  leak test (worst |change-form corr| over classes, gate 0.15): "
          + json.dumps({k: round(v, 4) for k, v in worst.items()}), flush=True)

    arms = [x for x in a.arms.split(",") if x]
    report: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(),
                    "shooter_key": SHOOTER_KEY, "scheme": "S1", "fold": "F2",
                    "m_fitted": {c: m[c]["m"] for c in FG.SHOT_CLASSES},
                    "m_mom_reference_not_used": MOM_M,
                    "arms": {}, "leak_worst_abs_corr": worst,
                    "common_features": COMMON, "shooter_blocks": SHOOTER_BLOCK}

    prior: dict = {}
    if a.merge_report:
        prior = json.loads((R4 / a.merge_report).read_text(encoding="utf-8"))
        report["merged_from"] = a.merge_report
        report["re_run_arms"] = arms

    all_scores = {}
    for arm in arms:
        print(f"=== arm {arm} ({LABEL[arm]}), "
              f"{len(COMMON) + len(SHOOTER_BLOCK[arm])} features ===", flush=True)
        sc, sg = fit_s1(design, arm, params, seed=0, export=True, m=m)
        all_scores[arm] = sc
        report["arms"][arm] = {
            "label": LABEL[arm], "simplicity": SIMPLICITY[arm],
            "servable": arm not in UNSERVABLE,
            "n_features": len(COMMON) + len(SHOOTER_BLOCK[arm]),
            "shooter_block": SHOOTER_BLOCK[arm],
            "by_class": {c: {k: sc[c][k] for k in
                             ("n", "log_loss", "brier", "calib_pass",
                              "calib_worst_gap_pp", "resp_pass_decision8",
                              "resp_failed_drivers_decision8", "pred_make_rate",
                              "actual_make_rate")} for c in FG.SHOT_CLASSES},
            "detail": {c: {"resp_decision8": sc[c]["resp_decision8"],
                           "by_chance": sc[c]["by_chance"],
                           "calibration": sc[c]["calibration"]}
                       for c in FG.SHOT_CLASSES},
            "segments": sg,
        }

    # ---- arms carried forward unchanged from an earlier report -----------
    carried = []
    for arm in ARMS:
        if arm in arms or arm not in prior.get("arms", {}):
            continue
        blk = prior["arms"][arm]
        if blk["shooter_block"] != SHOOTER_BLOCK[arm]:
            raise AssertionError(
                f"{arm}'s shooter block changed since {a.merge_report}; it cannot be "
                "carried forward and must be re-run")
        report["arms"][arm] = blk
        all_scores[arm] = {c: {**blk["by_class"][c], **blk["detail"][c]}
                           for c in FG.SHOT_CLASSES}
        carried.append(arm)
    if carried:
        arms = sorted(set(arms) | set(carried), key=lambda x: SIMPLICITY[x])
        report["carried_forward_arms"] = carried
        print(f"carried forward unchanged from {a.merge_report}: {carried}", flush=True)

    # ---- noise floor: a second-seed refit of B1 -------------------------
    if FLOOR_ARM in a.arms.split(","):
        print(f"=== noise floor: {FLOOR_ARM} refit at seed 1 ===", flush=True)
        sc1, _ = fit_s1(design, FLOOR_ARM, params, seed=1, export=False, m=m)
        floor = {c: abs(all_scores[FLOOR_ARM][c]["log_loss"] - sc1[c]["log_loss"])
                 for c in FG.SHOT_CLASSES}
        report["noise_floor"] = {
            "arm": FLOOR_ARM, "method": "second-seed refit of the whole S1 schedule",
            "by_class": {c: round(floor[c], 8) for c in FG.SHOT_CLASSES},
            "seed1_log_loss": {c: sc1[c]["log_loss"] for c in FG.SHOT_CLASSES},
            "round3_second_seed_floor": {"FGA_rim": 8.0e-5, "FGA_jump2": 4.2e-5,
                                         "FGA_3": 6.0e-6},
            "round1_bootstrap_floor": {"FGA_rim": 0.000539, "FGA_jump2": 0.000578,
                                       "FGA_3": 0.000849},
        }
    elif prior.get("noise_floor"):
        report["noise_floor"] = dict(prior["noise_floor"])
        report["noise_floor"]["carried_forward_from"] = a.merge_report
        floor = {c: float(prior["noise_floor"]["by_class"][c]) for c in FG.SHOT_CLASSES}
        print(f"noise floor carried forward from {a.merge_report}: {floor}", flush=True)
    else:
        floor = {c: float("nan") for c in FG.SHOT_CLASSES}

    # ---- the offline decision (19.7 steps 1-2 only; step 3 is the engine)
    decision = {}
    for c in FG.SHOT_CLASSES:
        rows = []
        for arm in arms:
            s = all_scores[arm][c]
            rows.append({"arm": arm, "simplicity": SIMPLICITY[arm],
                         "log_loss": s["log_loss"],
                         "calib_pass": bool(s["calib_pass"]),
                         "d8_pass": bool(s["resp_pass_decision8"]),
                         "d8_failed": s["resp_failed_drivers_decision8"],
                         "eligible": bool(s["calib_pass"] and s["resp_pass_decision8"])})
        elig = [r for r in rows if r["eligible"]]
        if not elig:
            decision[c] = {"offline_winner": "B0", "why": "no arm passed calibration "
                           "and Decision 8; rule 19.7 step 4", "rows": rows}
            continue
        best = min(elig, key=lambda r: r["log_loss"])
        simpler = [r for r in elig if r["simplicity"] < best["simplicity"]
                   and r["log_loss"] - best["log_loss"] <= floor[c]]
        pick = min(simpler, key=lambda r: r["simplicity"]) if simpler else best
        decision[c] = {
            "offline_winner": pick["arm"],
            "best_log_loss_arm": best["arm"],
            "beaten_by_floors": {r["arm"]: (round((r["log_loss"] - best["log_loss"]) / floor[c], 2)
                                            if floor[c] else None) for r in rows},
            "why": ("lowest log loss beyond every simpler arm's floor"
                    if pick["arm"] == best["arm"]
                    else f"simplicity tie-break: inside one floor of {best['arm']}"),
            "rows": rows,
        }
        print(f"{c}: offline winner {pick['arm']} (best ll {best['arm']}), "
              f"floor {floor[c]:.2e}", flush=True)
    report["offline_decision"] = decision
    report["runtime_s"] = round(time.time() - t0, 1)
    (R4 / "run_report.json").write_text(json.dumps(report, indent=2, default=str),
                                        encoding="utf-8")
    print(f"\nwrote {R4}/run_report.json ({report['runtime_s']}s)")
    print("OFFLINE ONLY. The Decision-10 closed loop (19.6) is a separate step: "
          "build the round-4 engine slots, then run_engine.py per arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
