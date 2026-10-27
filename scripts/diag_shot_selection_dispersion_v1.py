"""diag_shot_selection_dispersion_v1.py -- shot-location round 2, block V inputs (docs/models/shot_location/
experiments.md section 3.3; PRE-REGISTERED at commit 5f6d6b0 BEFORE any fit).

Block V is a DIAGNOSTIC (no arm). This script only assembles the inputs the blind grader
(`grade_shot_location_v2.py`) scores; it computes no ratio and no verdict.

  class part   the SERVED possession-outcome artifacts (`data/processed/models/engine/event_round2_s1_F2_2025/`,
               monthly S1 refits, NO refit here) score every `first` / `cont` chance of the 2025 test season on the
               round-2 design's own columns (C_plus_state). One row per chance: game, offence, y (realised terminal
               class), six class probabilities. F1 (2024): the engine holds no F1 artifact set, so the class part is
               NOT RUN for F1 (a replay would be a refit).
  band part    round-1 B2 band probabilities (readB/<fold>/preds.parquet) on every LOCATED jump2 / three attempt of
               the test season, with the actual band, aligned to (game, offence) via the round-1 design table.
  context      one row per (game, offence team): realised final margin from the VERIFIED finals, site, chance count,
               the opponent's as-of defence profile columns (class: opp_def_*_c; band: def_*_c).

    .venv/Scripts/python.exe scripts/diag_shot_selection_dispersion_v1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.eval import reference as REF  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

R1 = ROOT / "data/processed/models/shot_location/round1"
OUT = ROOT / "data/processed/models/shot_location/round2/V"
ENGINE = ROOT / "data/processed/models/engine/event_round2_s1_F2_2025"
PO_DESIGN = ROOT / "data/processed/models/possession_outcome/round2/design.parquet"
BANDS = {"FGA_jump2": ("short2", "long2"), "FGA_3": ("corner3", "wing3", "top3")}
FOLD_TEST = {"F1": 2024, "F2": 2025}


def log(m, t0):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def class_part(t0):
    idx = json.loads((ENGINE / "index.json").read_text(encoding="utf-8"))
    cuts = [pd.Timestamp(c) for c in idx["refit_dates"]]
    d = pd.read_parquet(PO_DESIGN)
    d = d[d["season"] == 2025].copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    out = []
    for pop in ("first", "cont"):
        info = idx["populations"][pop]
        feats = info["features"]
        te = d[d["population"] == pop]
        seg = np.searchsorted(np.array([np.datetime64(c, "ns") for c in cuts]),
                              te["game_date"].to_numpy(), side="right") - 1
        assert (seg >= 0).all()
        P = np.full((len(te), 6), np.nan)
        for i, s in enumerate(info["segments"]):
            m = seg == i
            if not m.any():
                continue
            w = joblib.load(ENGINE / s["file"])
            P[m] = PO.predict_arm(info["arm"], w["model"], te.loc[m], feats)
            log(f"  {pop} refit {s['refit_date']}: scored {int(m.sum()):,}", t0)
        assert np.isfinite(P).all()
        o = te[["game_id", "offense_team_id", "defense_team_id", "offense_is_home", "neutral_site", "game_date",
                "poss_index", "chance_number", "y"]].copy()
        o["population"] = pop
        for k, c in enumerate(PO.CLASSES):
            o["p_" + c] = P[:, k].astype("float32")
        out.append(o)
    ch = pd.concat(out, ignore_index=True)
    ch.to_parquet(OUT / "chance_probs_F2_2025.parquet", index=False)
    ctx = d.drop_duplicates(["game_id", "offense_team_id"])[
        ["game_id", "offense_team_id", "offense_is_home", "neutral_site", "off_3pa_c", "off_rim_c", "off_tov_c",
         "opp_def_3pa_c", "opp_def_rim_c", "opp_def_tov_c"]]
    ctx.to_parquet(OUT / "class_context_F2_2025.parquet", index=False)
    log(f"class part: {len(ch):,} chances, {ch[['game_id', 'offense_team_id']].drop_duplicates().shape[0]:,} team-games",
        t0)


def band_part(fold, t0):
    season = FOLD_TEST[fold]
    assert_not_sealed(season, context="shot_location round 2 band part")
    loc = pd.read_parquet(R1 / "design_loc_cols_v1.parquet",
                          columns=["game_id", "off_team_id", "def_team_id", "season", "game_date", "shot_class",
                                   "band", "located", "off_short2_c", "off_corner3_c", "off_wing3_c", "off_top3_c",
                                   "def_short2_c", "def_corner3_c", "def_wing3_c", "def_top3_c"])
    pr = pd.read_parquet(R1 / "readB" / fold / "preds.parquet")
    pr = pr[pr["located"]]
    cols = ["fg_row", "B2__p_short2", "B2__p_long2", "B2__p_corner3", "B2__p_wing3", "B2__p_top3"]
    x = pr[cols].rename(columns=lambda c: c.replace("B2__", ""))
    x = x.join(loc, on="fg_row")
    x = x.drop(columns=["located"])
    assert (x["band"] != "").all()
    x.drop(columns=["fg_row"]).to_parquet(OUT / f"band_probs_{fold}.parquet", index=False)
    log(f"band part {fold}: {len(x):,} located attempts", t0)


def context(t0):
    rows = []
    for s in (2024, 2025):
        a = REF.load_actual_games(s)[["game_id", "home_team_id", "away_team_id", "home_score", "away_score", "neutral"]]
        for side, tc, sc, oc in (("home", "home_team_id", "home_score", "away_score"),
                                 ("away", "away_team_id", "away_score", "home_score")):
            rows.append(pd.DataFrame({"game_id": a["game_id"], "team_id": a[tc], "season": s,
                                      "margin": (a[sc] - a[oc]).astype(float), "is_home": side == "home",
                                      "neutral": a["neutral"].astype(float)}))
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "margins.parquet", index=False)
    log("margins written (verified finals)", t0)


def main():
    t0 = time.time()
    assert_not_sealed([2024, 2025], context="shot_location round 2 block V")
    OUT.mkdir(parents=True, exist_ok=True)
    context(t0)
    band_part("F2", t0)
    band_part("F1", t0)
    class_part(t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
