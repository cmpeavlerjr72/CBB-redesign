"""
build_engine_event_round4b.py -- persist a possession-outcome ROUND-4b
SHRINKAGE ARM (`G2` or `G3`) as engine artifacts, including the S1 monthly
refit schedule.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 12
(the SHIP GATE closed loop). Arms are section 8.1's:

    G2   off_{r}_g2 = w*raw_c + (1-w)*prior_c
    G3   off_{r}_g3 = w*raw_c + (1-w)*w2*prior_c

WHAT THIS SCRIPT IS, AND WHAT IT IS NOT
---------------------------------------
It is `scripts/build_engine_event_round2.py` run against a different feature
set. Every structural decision -- the S1 schedule replay, the per-GAME leak
assertion, the as-of-backward fallback for a team-game the design does not
cover, the `(G, 2, 16)` team block, the `index.json` contract the adapter
loads -- is that script's, IMPORTED from it rather than copied, so the served
arm and the test arms cannot drift apart. `build_engine_event_round2.py` is
NOT edited and its output directory is NOT touched.

It is not a refit of the bake-off. The shrunk columns are READ out of
`round4/design_v4.parquet` exactly as round 4b's grade scored them (round 3c's
standing rule: the fitted object is READ, never reimplemented). The fitted
`k_r` are already baked into those columns and are not re-estimated here.

THE PRE-REGISTERED PRECONDITION (12.1)
---------------------------------------
An arm must differ from the served reference ONLY by the shrinkage. So before
anything is fitted, the sixteen `TEAM_COLS` of `design_v4.parquet` are checked
against `round2/design.parquet` team-game for team-game on the test season:
the eight NON-style columns and the eight RAW style columns must agree to 0.0
EXACTLY. A failure aborts the build -- it would mean the arms differ by
something other than the shrinkage and no closed-loop number from them would
be interpretable.

The S1 refit dates are likewise asserted equal to the served model's own, read
out of `event_round2_s1_{fold}_{season}/index.json`, so the arm is scored by a
schedule identical to the reference's rather than one this script re-derived.

Usage:
    .venv/Scripts/python.exe scripts/build_engine_event_round4b.py \
        --arm G2 --fold F2 --season 2025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_engine_event_round2 as B2  # noqa: E402
import train_possession_outcome_v4 as V4  # noqa: E402

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

R2_DESIGN = B2.R2_DESIGN
V4_DESIGN = Path("data/processed/models/possession_outcome/round4/design_v4.parquet")
V4_DESIGN_META = Path("data/processed/models/possession_outcome/round4/design_v4.meta.json")
ENGINE_DIR = B2.ENGINE_DIR
FOLD_TRAIN = B2.FOLD_TRAIN
FOLD_TEST = B2.FOLD_TEST

ARMS = ("G2", "G3")

#: The served model's own sixteen team columns, in order. An arm's block
#: carries the SAME sixteen quantities with the eight style ones replaced by
#: their shrunk siblings; positions and meanings are otherwise identical.
TEAM_COLS_R2: tuple[str, ...] = B2.TEAM_COLS

#: The eight style columns the arms transform, and the eight they must not.
STYLE_COLS: tuple[str, ...] = tuple(
    c for c in TEAM_COLS_R2
    if (c.startswith("off_") and c.endswith("_c") and not c.startswith("off_rating_"))
    or (c.startswith("opp_def_") and c.endswith("_c"))
)
NONSTYLE_COLS: tuple[str, ...] = tuple(c for c in TEAM_COLS_R2 if c not in STYLE_COLS)


def team_cols_for(arm: str) -> tuple[str, ...]:
    """`TEAM_COLS_R2` with the eight style names mapped to the arm's shrunk
    siblings. Built by the TRAINER's own `feature_set_v4` mapping rule rather
    than by re-deriving the suffix here, so the engine block and the offline
    design agree by construction."""
    suf = V4.SHRUNK_SUFFIX[arm]
    return tuple(c[:-1] + suf if c in STYLE_COLS else c for c in TEAM_COLS_R2)


def assert_arms_differ_only_by_shrinkage(v4: pd.DataFrame, season: int, t0: float) -> dict:
    """Section 12.1's pre-registered precondition, re-asserted here rather than
    trusted from the read-only check that preceded the pre-registration."""
    key = ["game_id", "offense_team_id"]
    cols = list(TEAM_COLS_R2)
    r2 = pd.read_parquet(R2_DESIGN, columns=key + ["season"] + cols)
    r2 = r2[r2["season"] == season].drop_duplicates(key)
    a = v4[v4["season"] == season].drop_duplicates(key)[key + cols]
    m = r2.merge(a, on=key, suffixes=("_r2", "_v4"))
    if len(m) != len(r2) or len(m) != len(a):
        raise AssertionError(
            f"round-2 design has {len(r2)} season-{season} team-games and design_v4 has {len(a)}; "
            f"{len(m)} joined. The arms would not be cut on the same rows. ABORT.")
    diffs = {}
    for c in cols:
        d = float(np.nanmax(np.abs(m[f"{c}_r2"].to_numpy("float64")
                                   - m[f"{c}_v4"].to_numpy("float64"))))
        diffs[c] = d
        if d != 0.0:
            raise AssertionError(
                f"design_v4 column {c!r} differs from round 2's by {d:.3e} on season {season}; "
                f"an arm built on it would differ from the reference by more than the "
                f"shrinkage. ABORT.")
    B2.log(f"precondition PASS: all {len(cols)} TEAM_COLS identical to round 2 "
           f"on {len(m)} team-games (max abs diff 0.0)", t0)
    return {"n_team_games": int(len(m)), "max_abs_diff": diffs}


def build(arm: str, fold: str, season: int, seed: int = 0) -> Path:
    t0 = time.time()
    if arm not in ARMS:
        raise SystemExit(f"--arm must be one of {ARMS}, not {arm!r}")
    if season != FOLD_TEST[fold]:
        raise SystemExit(f"fold {fold} tests season {FOLD_TEST[fold]}, not {season}")
    train_seasons = FOLD_TRAIN[fold]
    for s in [*train_seasons, season]:
        assert_not_sealed(s)

    ref_dir = ENGINE_DIR / f"event_round2_s1_{fold}_{season}"
    if not ref_dir.exists():
        raise SystemExit(f"{ref_dir} missing; the served reference must exist before an arm "
                         f"is built against it")
    ref_idx = json.loads((ref_dir / "index.json").read_text(encoding="utf-8"))
    win = B2.winners_from_verdict()
    B2.log(f"arm {arm}: population arms held at round 2's winners "
           f"first={win['first']['arm']}, cont={win['cont']['arm']}", t0)

    tcols = team_cols_for(arm)
    out = ENGINE_DIR / f"event_round4b_{arm}_{fold}_{season}"
    out.mkdir(parents=True, exist_ok=True)

    games = pd.read_parquet(ENGINE_DIR / f"games_{fold}_{season}.parquet")
    gdate = pd.to_datetime(games["game_date"] if "game_date" in games else games["date"])
    G = len(games)
    B2.log(f"engine universe: {G} games, {gdate.min().date()} .. {gdate.max().date()}", t0)

    need = set(tcols) | set(TEAM_COLS_R2) | {
        "game_id", "offense_team_id", "defense_team_id", "season", "game_date",
        "population", "y"}
    for pop in ("first", "cont"):
        need |= set(V4.feature_set_v4(arm, pop))
    design = pd.read_parquet(V4_DESIGN, columns=sorted(need))
    design["game_date"] = pd.to_datetime(design["game_date"])
    B2.log(f"design_v4: {len(design):,} rows, {design.season.nunique()} seasons", t0)

    precheck = assert_arms_differ_only_by_shrinkage(design, season, t0)

    # ---- 1. the arm's team-form block, on the engine's game order ----------
    # Identical construction to build_engine_event_round2.build, on the arm's
    # own sixteen columns.
    form = design[["game_id", "offense_team_id", "season", "game_date", *tcols]]
    form = form.drop_duplicates(["game_id", "offense_team_id"])
    fs = form[form["season"] == season]
    gpos = {int(g): i for i, g in enumerate(games["game_id"].to_numpy())}
    team_block = np.full((G, 2, len(tcols)), np.nan, dtype=np.float32)
    gi = fs["game_id"].map(gpos)
    ok = gi.notna().to_numpy()
    gi = gi[ok].to_numpy().astype(np.int64)
    tid = fs.loc[ok, "offense_team_id"].to_numpy()
    home = games["home_team_id"].to_numpy()[gi]
    side = np.where(tid == home, 0, 1).astype(np.int64)
    team_block[gi, side] = fs.loc[ok, list(tcols)].to_numpy(dtype=np.float32)

    miss = ~np.isfinite(team_block).all(axis=2)
    n_missing = int(miss.sum())
    n_filled = 0
    if n_missing:
        side_team = np.where(
            np.arange(2)[None, :] == 0,
            games["home_team_id"].to_numpy()[:, None],
            games["away_team_id"].to_numpy()[:, None])
        rowsel = np.flatnonzero(miss.ravel())
        want = pd.DataFrame({
            "row": rowsel,
            "team_id": side_team.ravel()[rowsel],
            "game_date": np.repeat(gdate.to_numpy(), 2)[rowsel],
        }).sort_values("game_date")
        have = (form.rename(columns={"offense_team_id": "team_id"})
                    .assign(game_date=lambda d: pd.to_datetime(d["game_date"]))
                    .sort_values("game_date"))
        got = pd.merge_asof(want, have[["team_id", "game_date", *tcols]],
                            on="game_date", by="team_id", direction="backward")
        vals = got[list(tcols)].to_numpy(dtype=np.float32)
        good = np.isfinite(vals).all(axis=1)
        flat = team_block.reshape(G * 2, -1)
        flat[got.loc[good, "row"].to_numpy()] = vals[good]
        n_filled = int(good.sum())
        team_block = flat.reshape(G, 2, -1)
    n_zero = int((~np.isfinite(team_block).all(axis=2)).sum())
    team_block = np.nan_to_num(team_block, nan=0.0).astype(np.float32)
    B2.log(f"team block: {n_missing} team-games uncovered, {n_filled} filled as-of backward, "
           f"{n_zero} left at the league mean", t0)

    # The same three counts as the served reference, or the block covers a
    # different set of games than the reference does and the runs are not
    # paired on the same information.
    rp = ref_idx["team_block_provenance"]
    for k, v in (("n_team_games_uncovered", n_missing), ("n_filled_asof_backward", n_filled),
                 ("n_left_at_league_mean", n_zero)):
        if int(rp[k]) != int(v):
            raise AssertionError(
                f"team-block coverage differs from the served reference: {k} is {v} here and "
                f"{rp[k]} there. The arm and the reference would not see the same games. ABORT.")
    B2.log("team-block coverage identical to the served reference", t0)

    # ---- 2. the S1 refit schedule, asserted equal to the reference's -------
    cuts = PO.month_boundaries(design.loc[design["season"] == season, "game_date"])
    got_dates = [str(c.date()) for c in cuts]
    if got_dates != list(ref_idx["refit_dates"]):
        raise AssertionError(
            f"S1 refit dates from design_v4 {got_dates} differ from the served reference's "
            f"{ref_idx['refit_dates']}; the arm would be scored on a different schedule. ABORT.")
    B2.log(f"S1 refit dates match the reference: {got_dates}", t0)

    cut_ns = np.array([np.datetime64(c, "ns") for c in cuts])
    seg_of_game = np.searchsorted(cut_ns, gdate.to_numpy(), side="right") - 1
    if (seg_of_game < 0).any():
        raise AssertionError("a game in the engine universe precedes the first S1 refit date")
    ref_seg = np.load(ref_dir / "team_block.npz")["seg_of_game"]
    if not np.array_equal(ref_seg.astype(np.int64), seg_of_game.astype(np.int64)):
        raise AssertionError("per-game S1 segment assignment differs from the served reference")

    index: dict = {
        "fold": fold, "season": season, "seed": seed, "scheme": "S1",
        "arm": arm,
        "round": "possession_outcome round 5 ship gate "
                 "(docs/models/possession_outcome/experiments.md section 12)",
        "source": f"possession_outcome round 4b arm {arm} (experiments.md 8.1, 11.3-11.5). "
                  f"train_possession_outcome_v4.py persists no booster, so the engine refits the "
                  f"arm's OWN spec through cbb_sim.models.possession_outcome, exactly as "
                  f"build_engine_event_round2.py does for the served round-2 winners. The SHRUNK "
                  f"COLUMNS ARE READ from design_v4.parquet and are never re-derived.",
        "adopted": False,
        "winners": win,
        "team_cols": list(tcols),
        "team_cols_reference": list(TEAM_COLS_R2),
        "team_block_provenance": {
            "design": str(V4_DESIGN), "design_meta": str(V4_DESIGN_META),
            "style_source": "first_chance", "require_pbp_complete": True,
            "possessions_version": "v2",
            "n_team_games_uncovered": n_missing, "n_filled_asof_backward": n_filled,
            "n_left_at_league_mean": n_zero,
            "differs_from_reference_only_by_shrinkage": precheck,
            "why": f"round-4b arm {arm} shrinks the eight as-of style columns toward the team's "
                   f"own prior-season centred rate; the other eight team columns are bit-"
                   f"identical to the served reference's (asserted, max abs diff 0.0)",
        },
        "refit_dates": got_dates,
        "populations": {},
    }

    for pop in ("first", "cont"):
        model_arm = win[pop]["arm"]
        feats = V4.feature_set_v4(arm, pop)
        missing = [c for c in feats if c not in design.columns]
        if missing:
            raise AssertionError(f"design_v4 lacks {missing} for {pop}/{arm}")
        tr = design[(design["season"].isin(train_seasons)) & (design["population"] == pop)]
        te = design[(design["season"] == season) & (design["population"] == pop)]
        te_dates = te["game_date"]
        fit_cols = [*feats, "y", "season"]
        segs = []
        for cut in cuts:
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, fit_cols]
            rows = tr[fit_cols] if not len(prior) else pd.concat(
                [tr[fit_cols], prior], ignore_index=True)
            model = PO.fit_arm(model_arm, rows, feats, seed=seed)
            for b in (getattr(model, "clf_", None),):
                try:
                    b.set_params(n_jobs=1)
                except Exception:                                  # noqa: BLE001
                    pass
            p = out / f"{pop}_{cut.date()}.joblib"
            joblib.dump({"model": model, "arm": model_arm, "features": feats,
                         "refit_date": str(cut.date())}, p, compress=3)
            mt = (tr["game_date"].max() if not before.any()
                  else max(tr["game_date"].max(), te_dates[before].max()))
            segs.append({"refit_date": str(cut.date()), "file": p.name,
                         "n_train": int(len(rows)),
                         "n_train_from_test_season": int(len(prior)),
                         "max_train_date": str(pd.Timestamp(mt).date()),
                         "size_mb": round(p.stat().st_size / 1e6, 2)})
            B2.log(f"  {arm}/{pop} {cut.date()}: fit {model_arm} on {len(rows):,} rows "
                   f"(through {pd.Timestamp(mt).date()}) -> {p.name}", t0)
        B2._assert_no_leak(gdate, seg_of_game,
                           [pd.Timestamp(s["max_train_date"]) for s in segs],
                           f"{arm}/{pop}/S1")
        ref_n = [s["n_train"] for s in ref_idx["populations"][pop]["segments"]]
        if [s["n_train"] for s in segs] != ref_n:
            raise AssertionError(
                f"{pop}: training row counts {[s['n_train'] for s in segs]} differ from the "
                f"served reference's {ref_n}; the arm saw a different training set. ABORT.")
        index["populations"][pop] = {
            "arm": model_arm, "feature_set": f"C_plus_state[{arm}]", "features": feats,
            "segments": segs}

    np.savez_compressed(out / "team_block.npz", team_block=team_block,
                        seg_of_game=seg_of_game.astype(np.int16))
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    B2.log(f"wrote {out} ({sum(f.stat().st_size for f in out.iterdir()) / 1e6:.1f} MB)", t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--fold", default="F2", choices=sorted(FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.arm, a.fold, a.season, a.seed)


if __name__ == "__main__":
    main()
