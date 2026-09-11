#!/usr/bin/env python
"""
build_engine_inputs_shotshooter.py -- the engine's fg_make SHOOTER slot block,
rebuilt with the shooter keyed on `shot_shooter_id`.

    .venv/Scripts/python.exe scripts/build_engine_inputs_shotshooter.py \
        --fold F2 --season 2025

WHY THIS EXISTS
---------------
`scripts/build_engine_inputs.py` builds the per-slot fg_make shooter block from
`data/processed/models/fg_make/events_v2.parquet`, which is keyed on
`participant_1_id` -- the column L28 proved is the ASSISTER on ~half of all
assisted made field goals. Serving a model TRAINED on `shot_shooter_id`
(fg_make round 3's interim arm, and every round-4 arm) against slot features
BUILT on `participant_1_id` is a train/serve skew of exactly the kind L27
recorded for round 1's `score_diff`: the corrected feature correlates only
0.694 / 0.802 / 0.335 (rim / jumper / three) with the one the engine would feed
(`docs/tests/fg_make_shooter_key_2026-09-10.md` 1.4).

So a closed-loop gate on a corrected-label arm is only meaningful against
corrected engine inputs, and this script produces them.

WHAT IT DOES AND DOES NOT TOUCH
-------------------------------
It reads the EXISTING inputs (`arrays_<tag>.npz`, `games_<tag>.parquet`,
`names_<tag>.json`) and rewrites ONLY the twelve fg_make shooter slot columns:

    shooter_make_c__{rim,jump2,three}
    shooter_att_c__{rim,jump2,three}
    prior_season_make_c__{rim,jump2,three}
    has_prior_season_fg, pos_G, pos_F, pos_C,
    shooter_games_asof, shooter_fga_asof

using exactly `build_engine_inputs.asof_backward` and the same source frame
construction, imported from that script rather than reimplemented. Every other
array (team_static, rosters, rotation, usage, rebound, free-throw slot columns)
is copied through BYTE FOR BYTE and the script asserts it.

The TEAM columns `off_make_c__*` / `def_allow_c__*` are NOT rebuilt because
they cannot move: `fg_make.team_shot_form` aggregates per-(game, team) attempt
and make COUNTS, which are conserved whichever player is credited. The script
verifies this claim numerically rather than asserting it, and prints the
result.

WORKER DISCIPLINE: writes to a NEW directory (`--out-dir`, default
`data/processed/models/engine_fgm4`). Nothing under
`data/processed/models/engine/` is written, so the five other workers running
against the stock inputs are unaffected. `run_engine.py --input-dir` points at
the new directory; the adapters' own joblib paths are unchanged and still
resolve to `data/processed/models/engine/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_engine_inputs import SHOT_KEY, asof_backward  # noqa: E402

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

IN_DIR = Path("data/processed/models/engine")
OUT_DIR = Path("data/processed/models/engine_fgm4")
FG_DIR = Path("data/processed/models/fg_make")
EVENTS_SHOTSHOOTER = FG_DIR / "events_v2_shotshooter.parquet"
EVENTS_STOCK = FG_DIR / "events_v2.parquet"
DESIGN_CACHE = FG_DIR / "design_v2_shotshooter.parquet"

FOLD_TRAIN = {"F1": [2022, 2023], "F2": [2022, 2023, 2024]}

#: The columns this script rewrites. Everything else is copied through.
PER_CLASS = ("shooter_make_c", "shooter_att_c", "prior_season_make_c")
SHARED = ("has_prior_season_fg", "pos_G", "pos_F", "pos_C",
          "shooter_games_asof", "shooter_fga_asof")

#: ROUND 4 (`experiments.md` section 19): columns that do not exist in the
#: stock inputs at all and are APPENDED to the slot block, never inserted, so
#: every existing `FeaturePlan` that indexes `slot_names` keeps its positions.
#: Source: `data/processed/models/fg_make/round4/slot_source_v1.parquet`,
#: written by `scripts/train_fg_make_v4_shooter_block.py` AFTER the shrinkage
#: strength `m` is fitted on fold 1 -- which is why this is a second pass and
#: not part of the first.
R4_SLOT_SOURCE = FG_DIR / "round4" / "slot_source_v2.parquet"
R4_PER_CLASS = ("shooter_shrunk_dev_c", "prior_season_att_c")
R4_SHARED = ("sh_share_rim", "sh_share_jump2", "sh_share_three", "sh_assisted_share")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_design(seasons: list[int], t0: float) -> pd.DataFrame:
    if DESIGN_CACHE.exists():
        log(f"design cache hit: {DESIGN_CACHE}", t0)
        return pd.read_parquet(DESIGN_CACHE)
    if not EVENTS_SHOTSHOOTER.exists():
        raise FileNotFoundError(
            f"{EVENTS_SHOTSHOOTER} missing; run scripts/train_fg_make_v3_shooter.py")
    ev = pd.read_parquet(EVENTS_SHOTSHOOTER)
    uni = ES.load_universe(require_pbp_complete=True)
    d = FG.build_design(seasons, universe=uni, version="v2", events=ev)
    d.to_parquet(DESIGN_CACHE, index=False)
    log(f"fg_make design (shot_shooter_id): {len(d):,} rows -> {DESIGN_CACHE}", t0)
    return d


def team_form_unchanged(seasons: list[int], season: int, t0: float) -> dict:
    """Verify the claim that team-level as-of form cannot move with the shooter
    key, instead of asserting it. Counts are conserved; only the credited
    player moves."""
    if not EVENTS_STOCK.exists():
        return {"checked": False, "why": f"{EVENTS_STOCK} missing"}
    uni = ES.load_universe(require_pbp_complete=True)
    a = FG.team_shot_form(pd.read_parquet(EVENTS_STOCK), uni)
    b = FG.team_shot_form(pd.read_parquet(EVENTS_SHOTSHOOTER), uni)
    key = ["season", "game_id", "team_id"]
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    num = [c for c in a.columns if c not in (*key, "game_date")]
    same = bool(len(a) == len(b) and np.allclose(
        a[num].to_numpy(dtype="float64"), b[num].to_numpy(dtype="float64"),
        rtol=0, atol=0, equal_nan=True))
    out = {"checked": True, "n_rows_participant1": int(len(a)),
           "n_rows_shotshooter": int(len(b)), "identical": same}
    log(f"team form identical under both keys: {same}", t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2", choices=sorted(FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--in-dir", default=str(IN_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--skip-team-check", action="store_true")
    ap.add_argument("--with-round4", action="store_true",
                    help="also APPEND the round-4 shooter slot columns from "
                         "data/processed/models/fg_make/round4/slot_source_v1.parquet")
    args = ap.parse_args()
    t0 = time.time()

    from cbb_sim.data.seal import assert_not_sealed
    season = int(args.season)
    train = FOLD_TRAIN[args.fold]
    assert_not_sealed([season], context="engine inputs (shot_shooter_id)")
    assert_not_sealed(train, context="engine inputs (shot_shooter_id) train")
    all_seasons = sorted(set(train + [season]))

    tag = f"{args.fold}_{season}"
    ind, outd = Path(args.in_dir), Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    games = pd.read_parquet(ind / f"games_{tag}.parquet")
    z = dict(np.load(ind / f"arrays_{tag}.npz"))
    names = json.loads((ind / f"names_{tag}.json").read_text(encoding="utf-8"))
    slot_names = {k: int(v) for k, v in names["slot_names"].items()}
    G, S = len(games), z["roster_cbbd"].shape[2]
    log(f"loaded stock inputs: {G} games, {S} slots, "
        f"slot_static {z['slot_static'].shape}", t0)

    design = build_design(all_seasons, t0)

    # ---- the same per-slot target frame `build_engine_inputs` builds --------
    flat = pd.DataFrame({
        "row": np.repeat(np.arange(G), 2 * S),
        "side": np.tile(np.repeat([0, 1], S), G),
        "slot": np.tile(np.arange(S), 2 * G),
        "pid": z["roster_cbbd"].reshape(-1),
        "game_date": np.repeat(pd.to_datetime(games["game_date"]).to_numpy(), 2 * S),
    })
    real = flat[flat["pid"] > 0].copy()
    log(f"per-slot join: {len(real):,} named roster slots of {len(flat):,}", t0)

    slot_static = z["slot_static"].copy()
    before = z["slot_static"]
    touched = []

    def put_slot(col: str, frame: pd.DataFrame, value_col: str) -> None:
        j = slot_names[col]
        slot_static[:, :, :, j] = 0.0            # same zero base as the builder
        v = frame[value_col].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        slot_static[frame["row"].to_numpy()[ok], frame["side"].to_numpy()[ok],
                    frame["slot"].to_numpy()[ok], j] = v[ok].astype(np.float32)
        touched.append(col)

    for cls_name, key in SHOT_KEY.items():
        sl = design[design["shot_class"] == cls_name]
        src = (sl[["shooter_id", "game_date", *PER_CLASS]]
               .rename(columns={"shooter_id": "pid"})
               .drop_duplicates(subset=["pid", "game_date"]))
        src["game_date"] = pd.to_datetime(src["game_date"])
        src["pid"] = src["pid"].astype("int64")
        got = asof_backward(real, src, ["pid"], "game_date", list(PER_CLASS))
        for c in PER_CLASS:
            put_slot(f"{c}__{key}", got, c)
    src = (design[["shooter_id", "game_date", "has_prior_season", "pos_G", "pos_F",
                   "pos_C", "shooter_games_asof", "shooter_fga_asof"]]
           .rename(columns={"shooter_id": "pid"})
           .drop_duplicates(subset=["pid", "game_date"]))
    src["game_date"] = pd.to_datetime(src["game_date"])
    src["pid"] = src["pid"].astype("int64")
    got = asof_backward(real, src, ["pid"], "game_date",
                        ["has_prior_season", "pos_G", "pos_F", "pos_C",
                         "shooter_games_asof", "shooter_fga_asof"])
    put_slot("has_prior_season_fg", got, "has_prior_season")
    for c in ("pos_G", "pos_F", "pos_C", "shooter_games_asof", "shooter_fga_asof"):
        put_slot(c, got, c)
    log(f"rewrote {len(touched)} fg_make shooter slot columns", t0)

    # ---- round-4 columns, APPENDED (never inserted) ------------------------
    r4_added: list[str] = []
    if args.with_round4:
        if not R4_SLOT_SOURCE.exists():
            raise FileNotFoundError(
                f"{R4_SLOT_SOURCE} missing; run "
                "scripts/train_fg_make_v4_shooter_block.py first (it fits m, then "
                "exports the slot source)")
        src4 = pd.read_parquet(R4_SLOT_SOURCE)
        src4["game_date"] = pd.to_datetime(src4["game_date"])
        src4["pid"] = src4["shooter_id"].astype("int64")
        new_cols = [f"{c}__{k}" for c in R4_PER_CLASS
                    for k in ("rim", "jump2", "three")] + list(R4_SHARED)
        missing = [c for c in new_cols if c not in src4.columns]
        if missing:
            raise KeyError(f"{R4_SLOT_SOURCE} is missing {missing}")
        src4 = (src4[["pid", "game_date", *new_cols]]
                .drop_duplicates(subset=["pid", "game_date"]))
        got4 = asof_backward(real, src4, ["pid"], "game_date", new_cols)
        width = slot_static.shape[3]
        pad = np.zeros((*slot_static.shape[:3], len(new_cols)), dtype=np.float32)
        slot_static = np.concatenate([slot_static, pad], axis=3)
        for i, col in enumerate(new_cols):
            slot_names[col] = width + i
            v = got4[col].to_numpy(dtype="float64")
            ok = np.isfinite(v)
            slot_static[got4["row"].to_numpy()[ok], got4["side"].to_numpy()[ok],
                        got4["slot"].to_numpy()[ok], width + i] = v[ok].astype(np.float32)
            r4_added.append(col)
        log(f"appended {len(r4_added)} round-4 slot columns "
            f"(slot block {width} -> {slot_static.shape[3]})", t0)

    # ---- how much the engine's own inputs moved ----------------------------
    valid = z["roster_valid"]
    move: dict = {}
    for col in touched:
        j = slot_names[col]
        a = before[:, :, :, j][valid].astype("float64")
        b = slot_static[:, :, :, j][valid].astype("float64")
        d = b - a
        finite = np.isfinite(a) & np.isfinite(b)
        c = float(np.corrcoef(a[finite], b[finite])[0, 1]) if finite.sum() > 2 else float("nan")
        move[col] = {
            "n_valid_slots": int(finite.sum()),
            "corr_old_new": round(c, 4),
            "mean_abs_delta": round(float(np.abs(d[finite]).mean()), 6),
            "pct_moved_ge_1pp": round(float((np.abs(d[finite]) >= 0.01).mean() * 100), 3),
            "mean_old": round(float(a[finite].mean()), 6),
            "mean_new": round(float(b[finite].mean()), 6),
        }
    for col in slot_names:
        if col in touched or col in r4_added:
            continue
        j = slot_names[col]
        assert np.array_equal(before[:, :, :, j], slot_static[:, :, :, j]), col

    # every other array copied through byte for byte
    out_arrays = {k: (slot_static if k == "slot_static" else v) for k, v in z.items()}
    for k, v in z.items():
        if k != "slot_static":
            assert out_arrays[k] is v

    np.savez_compressed(outd / f"arrays_{tag}.npz", **out_arrays)
    shutil.copy2(ind / f"games_{tag}.parquet", outd / f"games_{tag}.parquet")
    names_out = dict(names)
    names_out["slot_names"] = {k: int(v) for k, v in slot_names.items()}
    meta = dict(names_out.get("meta") or {})
    meta["fg_make_shooter_key"] = "shot_shooter_id"
    meta["fg_make_shooter_block_rebuilt_by"] = "scripts/build_engine_inputs_shotshooter.py"
    meta["fg_make_shooter_block_source"] = str(EVENTS_SHOTSHOOTER)
    if r4_added:
        meta["fg_make_round4_slot_columns"] = r4_added
        meta["fg_make_round4_slot_source"] = str(R4_SLOT_SOURCE)
    names_out["meta"] = meta
    (outd / f"names_{tag}.json").write_text(json.dumps(names_out, indent=1, default=str),
                                            encoding="utf-8")

    report = {
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "fold": args.fold, "season": season, "tag": tag,
        "in_dir": str(ind), "out_dir": str(outd),
        "shooter_key": "shot_shooter_id",
        "columns_rewritten": touched,
        "columns_appended_round4": r4_added,
        "columns_copied": [c for c in slot_names if c not in touched and c not in r4_added],
        "slot_feature_movement": move,
        "team_form_check": ({"checked": False, "why": "--skip-team-check"}
                            if args.skip_team_check
                            else team_form_unchanged(all_seasons, season, t0)),
    }
    (outd / f"rebuild_report_{tag}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"wrote {outd}/arrays_{tag}.npz and rebuild_report_{tag}.json", t0)
    for col, m in move.items():
        print(f"  {col:<28} corr {m['corr_old_new']:+.4f}  "
              f"mean|d| {m['mean_abs_delta']:.5f}  moved>=1pp {m['pct_moved_ge_1pp']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
