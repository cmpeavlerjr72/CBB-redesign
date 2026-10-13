"""diag_clock_env_q1_wiring_v1.py -- clock round 9 (experiments.md section 41.0 item 3) Q1 WIRING PRE-CHECK.

Read-only. Run BEFORE any round-9 arm is fitted. Question: are the rating columns a round-9 arm reads
(`off_rating_off_c`, `off_rating_def_c`, `def_rating_off_c`, `def_rating_def_c`: the offence's off_c / def_c and the
defence's off_c / def_c) produced identically by
  (A) the offline design join (`r6_L2/design_v2.parquet`, built by `clock.build_design` through
      `own_ratings.join_as_of` on the batch files `data/processed/ratings/own_ratings_{season}.parquet`),
  (B) the live path (`cbb_sim.live.features.rating_site_block`, the code `build_engine_inputs_live.py` runs for the
      2027 day-1 slate against `data/processed/ratings_asof/2026-11-02`), and
  (C) the fold-2 engine inputs the closed loop serves (`engine_v3/arrays_F2_2025.npz` team_static)?
Same centring (rating minus the snapshot's own league mean over the season's teams), same as-of lag
(`as_of_date == game date` = fitted on games strictly before that date).

Checks:
 1. design columns == a fresh `join_as_of` on the batch files (seasons 2022-2025), max abs diff, NaN-fill counts.
 2. the LIVE code path on three historical 2025 slates (first day, 2025-01-16, 2025-03-04), fed (i) the batch files and
    (ii) the live as-of builder's own output (`build_own_ratings_asof_v1.asof_ratings`, teams `tg`), vs the design.
 3. F2 engine inputs team_static vs the design (per game, offence side), 2025.
 4. 2027 day-1 snapshot: schema, as_of_date, created_at < first tip, centring (team mean of off_c / def_c), SDs,
    hyperparameters and prior-carry rule vs the batch definition; team-set source recorded.
Writes results/clock_env/q1_wiring.json (gitignored); the table goes into docs/tests/clock_env_round_2026-10-13.md.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.live import features as LF  # noqa: E402
from cbb_sim.ratings import own_ratings as orat  # noqa: E402

COLS4 = ["off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c"]
OUT = ROOT / "results/clock_env"
LIVE = ROOT / "data/processed/ratings_asof/2026-11-02/own_ratings_2027.parquet"


def recompute(design: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    g = design[["season", "game_id", "game_date", "offense_team_id", "defense_team_id", *COLS4]].copy()
    g = orat.join_as_of(g, ratings, team_col="offense_team_id", suffix="__o", cols=("off_c", "def_c"))
    g = orat.join_as_of(g, ratings, team_col="defense_team_id", suffix="__d", cols=("off_c", "def_c"))
    g["n_nan"] = g[["off_c__o", "def_c__o", "off_c__d", "def_c__d"]].isna().any(axis=1)
    for a, b in (("off_c__o", "off_rating_off_c"), ("def_c__o", "off_rating_def_c"),
                 ("off_c__d", "def_rating_off_c"), ("def_c__d", "def_rating_def_c")):
        g[f"re_{b}"] = g[a].astype("float32").fillna(0.0)
    return g


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    design = pd.read_parquet(ROOT / "data/processed/models/clock/r6_L2/design_v2.parquet",
                             columns=["season", "game_id", "game_date", "offense_team_id", "defense_team_id",
                                      "offense_is_home", *COLS4])
    tg = design.drop_duplicates(["game_id", "offense_team_id"]).reset_index(drop=True)
    ratings = orat.load_ratings(sorted(tg["season"].unique().tolist()))
    # 1. design vs fresh batch join
    g = recompute(tg, ratings)
    r1 = {"n_team_games": int(len(g)), "n_nan_filled_zero": int(g["n_nan"].sum())}
    for c in COLS4:
        r1[f"max_abs_diff_{c}"] = float(np.abs(g[c].to_numpy(np.float64) - g[f"re_{c}"].to_numpy(np.float64)).max())
    by = g.groupby("season")["n_nan"].agg(["sum", "size"])
    r1["nan_by_season"] = {int(k): [int(v["sum"]), int(v["size"])] for k, v in by.iterrows()}
    rep["1_design_vs_batch_join"] = r1
    print("1", json.dumps(r1), flush=True)

    # 2. live code path on historical 2025 slates
    uni = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
    uni["game_date"] = pd.to_datetime(uni["game_date"])
    u25 = uni[(uni["season"] == 2025) & uni["is_d1_game"]]
    first_day = u25["game_date"].min()
    dates = [first_day, pd.Timestamp("2025-01-16"), pd.Timestamp("2025-03-04")]
    import build_own_ratings_asof_v1 as E  # noqa: E402
    cache: dict = {}
    r2 = []
    d25 = tg[tg["season"] == 2025].copy()
    d25["game_date"] = pd.to_datetime(d25["game_date"])
    for D in dates:
        slate = u25[u25["game_date"] == D].copy()
        if "neutral" not in slate.columns:
            slate["neutral"] = slate["neutral_site"].fillna(False).astype(bool)
        slate = slate[LF.SLATE_COLS]
        as_of = pd.to_datetime(slate["tipoff_utc"], utc=True).min() - pd.Timedelta(hours=1)
        ref = d25[d25["game_date"] == D].set_index(["game_id", "offense_team_id"])[COLS4]
        row = {"date": str(D.date()), "n_slate_games": int(len(slate))}
        with tempfile.TemporaryDirectory() as tmp:
            new, prov = E.asof_ratings(2025, str(D.date()), ROOT, "tg", 2022, cache)
            new.to_parquet(Path(tmp) / "own_ratings_2025.parquet", index=False)
            for label, rdir in (("batch", str(ROOT / "data/processed/ratings")), ("live_builder", tmp)):
                ctx = LF.build_ctx(slate, as_of, 2025, uni, ratings_dir=rdir, strict_finish=False)
                rs = LF.rating_site_block(ctx).set_index(["game_id", "team_id"])[COLS4]
                rs.index = rs.index.set_names(["game_id", "offense_team_id"])
                j = ref.join(rs, how="inner", rsuffix="_live")
                row[f"{label}_n_matched"] = int(len(j))
                row[f"{label}_max_abs_diff"] = float(max(np.abs(j[c].to_numpy(np.float64) -
                                                                j[f"{c}_live"].to_numpy(np.float64)).max()
                                                         for c in COLS4)) if len(j) else None
        r2.append(row)
        print("2", json.dumps(row), flush=True)
    rep["2_live_path_vs_design_2025"] = r2

    # 3. F2 engine inputs vs design
    z = np.load(ROOT / "data/processed/models/engine_v3/arrays_F2_2025.npz")
    names = json.loads((ROOT / "data/processed/models/engine_v3/names_F2_2025.json").read_text(encoding="utf-8"))
    tn = names["team_names"]
    tn = {c: i for i, c in enumerate(tn)} if isinstance(tn, list) else tn
    gm = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")
    ts = z["team_static"]
    rows = []
    for side, tcol in ((0, "home_team_id"), (1, "away_team_id")):
        f = pd.DataFrame({"game_id": gm["game_id"].to_numpy(), "offense_team_id": gm[tcol].to_numpy()})
        for c in COLS4:
            f[c] = ts[:, side, int(tn[c])]
        rows.append(f)
    eng = pd.concat(rows, ignore_index=True).set_index(["game_id", "offense_team_id"])
    j = d25.set_index(["game_id", "offense_team_id"])[COLS4].join(eng, how="inner", rsuffix="_eng")
    r3 = {"n_matched_team_games": int(len(j)), "n_engine_team_games": int(len(eng))}
    for c in COLS4:
        dd = np.abs(j[c].to_numpy(np.float64) - j[f"{c}_eng"].to_numpy(np.float64))
        r3[f"max_abs_diff_{c}"] = float(dd.max())
        r3[f"n_diff_gt_1e-4_{c}"] = int((dd > 1e-4).sum())
    rep["3_engine_F2_vs_design"] = r3
    print("3", json.dumps(r3), flush=True)

    # 4. 2027 day-1 snapshot
    live = pd.read_parquet(LIVE)
    prov = json.loads((LIVE.parent / "provenance.json").read_text(encoding="utf-8"))
    man = json.loads((ROOT / "data/processed/ratings/own_ratings_manifest.json").read_text(encoding="utf-8"))
    b25 = ratings[(ratings["season"] == 2025)].copy()
    b25["as_of_date"] = pd.to_datetime(b25["as_of_date"])
    b25d1 = b25[b25["as_of_date"] == b25["as_of_date"].min()]
    tips = None
    tp = ROOT / "data/processed/ingest/tip_times_2027.parquet"
    if tp.exists():
        t = pd.read_parquet(tp)
        tcol = [c for c in t.columns if "tip" in c.lower()]
        if tcol:
            tt = pd.to_datetime(t[tcol[0]], utc=True, errors="coerce")
            dcol = [c for c in t.columns if c in ("game_date", "date")]
            if dcol:
                tt = tt[pd.to_datetime(t[dcol[0]]) == pd.Timestamp("2026-11-02")]
            tips = tt.min()
    batch_cols = [c for c in ratings.columns]
    r4 = {
        "schema_live": list(live.columns), "schema_batch": batch_cols,
        "live_has_all_batch_cols": bool(set(batch_cols) <= set(live.columns)),
        "as_of_dates": sorted({str(pd.Timestamp(x).date()) for x in live["as_of_date"]}),
        "n_teams": int(len(live)), "n_games_max": int(live["n_games"].max()),
        "created_at_max": str(pd.to_datetime(live["created_at"], utc=True).max()),
        "first_tip_2026-11-02": None if tips is None or pd.isna(tips) else str(tips),
        "created_at_before_first_tip": (None if tips is None or pd.isna(tips)
                                        else bool(pd.to_datetime(live["created_at"], utc=True).max() < tips)),
        "team_mean_off_c_live": float(live["off_c"].mean()), "team_mean_def_c_live": float(live["def_c"].mean()),
        "team_mean_off_c_batch2025_day1": float(b25d1["off_c"].mean()),
        "team_mean_def_c_batch2025_day1": float(b25d1["def_c"].mean()),
        "sd_off_c_live": float(live["off_c"].std()), "sd_def_c_live": float(live["def_c"].std()),
        "sd_off_c_batch2025_day1": float(b25d1["off_c"].std()), "sd_def_c_batch2025_day1": float(b25d1["def_c"].std()),
        "hyper_live": prov.get("hyperparameters"), "hyper_batch_manifest": {k: man.get(k) for k in man
                                                                            if "lambda" in k or "prior_weight" in k},
        "prior_carry_live": prov.get("prior_carry"), "teams_source_live": prov.get("teams_source"),
        "policy_live": prov.get("policy"), "choices_live": prov.get("choices"),
    }
    rep["4_live_2027_day1_snapshot"] = r4
    print("4", json.dumps(r4, default=str), flush=True)
    (OUT / "q1_wiring.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
