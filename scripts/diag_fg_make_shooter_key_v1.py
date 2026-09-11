#!/usr/bin/env python
"""
diag_fg_make_shooter_key_v1.py -- fg_make round 3, step 1: quantify the
shooter-key defect ON FG_MAKE'S OWN POPULATION before anything is retrained.

Context. `docs/models/change_ledger.md` ("CBBD's `participant_1_id` is the
ASSISTER...") and `docs/tests/shooter_key_audit_2026-09-10.md` measured the
defect for `usage` (2024-2025 only, usage's own window) and flagged fg_make as
"STILL ON THE WRONG COLUMN and needs a re-run" without measuring fg_make's own
2022-2025 window or its own three shot classes (`FGA_rim`/`FGA_jump2`/`FGA_3`,
under the v2 rim-location override, L16) at the assisted/unassisted split. This
script measures exactly that, at the levels `CLAUDE.md` requires (overall, per
season, per class, per team), and then measures how much fg_make's shooter
as-of features actually move when the design is rebuilt on `shot_shooter_id`
instead of `participant_1_id` (`fg_make.build_fg_events(shooter_key=...)`,
added this round -- see `fg_make.py` module docstring "THE SHOOTER-KEY DEFECT").

2026 is SEALED and is never read here.

Usage:
    .venv/Scripts/python.exe scripts/diag_fg_make_shooter_key_v1.py

Writes:
    data/processed/models/fg_make/shooter_key_audit_v1.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.pbp.events import PLAY_COLUMNS, classify_frame, load_plays  # noqa: E402

SEASONS = (2022, 2023, 2024, 2025)          # 2026 sealed
OUT_DIR = Path("data/processed/models/fg_make")
AUDIT_COLUMNS: tuple[str, ...] = (
    *PLAY_COLUMNS, "participant_1_id", "participant_2_id",
    "shot_assisted", "shot_assisted_by_id",
)


def _bool(col: pd.Series) -> np.ndarray:
    s = col
    if s.dtype == object:
        s = s.map({True: True, False: False})
    return s.astype("boolean").fillna(False).to_numpy(dtype=bool)


def season_class_frame(season: int, universe: pd.DataFrame, max_ft: float) -> pd.DataFrame:
    """One row per FGA of `season`, fg_make's own universe and classes."""
    u = universe[universe["season"] == season]
    gids = set(u["cbbd_game_id"].dropna().astype("int64").tolist())
    plays = load_plays(season, game_ids=gids, columns=AUDIT_COLUMNS)
    cls = classify_frame(plays, rim_override_max_ft=max_ft).to_numpy(dtype=object)
    is_fga = np.isin(cls, FG.SHOT_CLASSES)
    plays = plays.loc[is_fga].reset_index(drop=True)
    cls = cls[is_fga]
    made = _bool(plays["shot_made"]) | _bool(plays["scoringPlay"])
    assist_known = plays["shot_assisted"].notna().to_numpy()
    assisted = _bool(plays["shot_assisted"])
    p1 = pd.to_numeric(plays["participant_1_id"], errors="coerce").to_numpy()
    sh = pd.to_numeric(plays["shot_shooter_id"], errors="coerce").to_numpy()
    return pd.DataFrame({
        "season": season,
        "team_id": pd.to_numeric(plays["teamId"], errors="coerce").to_numpy(),
        "shot_class": cls,
        "made": made,
        "assist_known": assist_known,
        "assisted": assisted,
        "p1": p1,
        "shooter": sh,
    })


def agg(d: pd.DataFrame) -> dict:
    n = len(d)
    if not n:
        return {"n": 0}
    p1, sh = d["p1"].to_numpy(), d["shooter"].to_numpy()
    has_p1, has_sh = np.isfinite(p1), np.isfinite(sh)
    both = has_p1 & has_sh
    agree = both & (p1 == sh)
    return {
        "n": int(n),
        "p1_present_pct": round(float(has_p1.mean() * 100), 4),
        "shooter_present_pct": round(float(has_sh.mean() * 100), 4),
        "shooter_missing_pct": round(float((~has_sh).mean() * 100), 4),
        "both_present_n": int(both.sum()),
        "agree_pct": round(float(agree.sum() / both.sum() * 100), 4) if both.any() else None,
        "mismatch_pct": round(float(100 - agree.sum() / both.sum() * 100), 4) if both.any() else None,
        "mismatch_n": int(both.sum() - agree.sum()),
    }


def per_team_drop(d: pd.DataFrame, min_rows: int = 50) -> dict:
    """Distribution across teams of the `shot_shooter_id`-missing rate (the
    DROP rate this round applies, never imputed)."""
    tid = d["team_id"].to_numpy()
    miss = ~np.isfinite(d["shooter"].to_numpy())
    ok = np.isfinite(tid)
    g = pd.DataFrame({"t": tid[ok], "miss": miss[ok]}).groupby("t")["miss"].agg(["size", "mean"])
    big = g[g["size"] >= min_rows]
    if not len(big):
        return {"teams": 0, "underpowered_teams": int(len(g))}
    q = big["mean"] * 100
    return {"teams": int(len(big)), "underpowered_teams": int(len(g) - len(big)),
            "min_pct": round(float(q.min()), 4), "median_pct": round(float(q.median()), 4),
            "p95_pct": round(float(q.quantile(0.95)), 4), "max_pct": round(float(q.max()), 4),
            "sd_pp": round(float(q.std(ddof=1)), 4) if len(big) > 1 else 0.0}


def feature_movement(seasons: list[int], universe: pd.DataFrame) -> dict:
    """Rebuild fg_make's design under both shooter keys and measure how much
    the shooter as-of features (`shooter_make_c`, `shooter_att_c`,
    `shooter_games_asof`) move, attempt for attempt.

    `build_fg_events`/`_season_events` do not filter or reorder rows on
    `shooter_key` (module docstring, "THE SHOOTER-KEY DEFECT") -- only the
    `shooter_id` VALUE changes on FGA rows -- so the two event tables are the
    same length in the same row order and a positional `attempt_row_id`
    survives `build_design`'s merges untouched, which is what makes the
    before/after join exact rather than approximate."""
    ev_p1 = FG.build_fg_events(seasons, universe=universe, version="v2",
                               shooter_key="participant_1_id")
    ev_sh = FG.build_fg_events(seasons, universe=universe, version="v2",
                               shooter_key="shot_shooter_id")
    assert len(ev_p1) == len(ev_sh), "shooter_key changed row count at the event layer"
    reassigned = (np.isfinite(ev_p1["shooter_id"].to_numpy())
                 & np.isfinite(ev_sh["shooter_id"].to_numpy())
                 & (ev_p1["shooter_id"].to_numpy() != ev_sh["shooter_id"].to_numpy()))
    ev_p1 = ev_p1.copy()
    ev_sh = ev_sh.copy()
    ev_p1["attempt_row_id"] = np.arange(len(ev_p1))
    ev_sh["attempt_row_id"] = np.arange(len(ev_sh))
    ev_p1["reassigned"] = reassigned
    ev_sh["reassigned"] = reassigned

    d_p1 = FG.build_design(seasons, universe=universe, version="v2", events=ev_p1,
                           with_ratings=False)
    d_sh = FG.build_design(seasons, universe=universe, version="v2", events=ev_sh,
                           with_ratings=False)

    keep = ["attempt_row_id", "shot_class", "reassigned", "shooter_make_c",
            "shooter_att_c", "shooter_games_asof"]
    m = d_p1[keep].merge(d_sh[keep], on="attempt_row_id", how="inner",
                         suffixes=("_p1", "_sh"))
    assert (m["shot_class_p1"] == m["shot_class_sh"]).all()
    assert (m["reassigned_p1"] == m["reassigned_sh"]).all()

    out: dict = {"n_events_p1": int(len(ev_p1)), "n_events_sh": int(len(ev_sh)),
                "n_reassigned_shooter_rows": int(reassigned.sum()),
                "n_design_p1": int(len(d_p1)), "n_design_sh": int(len(d_sh)),
                "n_joined": int(len(m)), "by_class": {}}
    for c in FG.SHOT_CLASSES:
        mc = m[m["shot_class_p1"] == c]
        if not len(mc):
            continue
        delta = (mc["shooter_make_c_sh"] - mc["shooter_make_c_p1"]).to_numpy()
        mc_r = mc[mc["reassigned_p1"]]
        delta_r = (mc_r["shooter_make_c_sh"] - mc_r["shooter_make_c_p1"]).to_numpy()
        out["by_class"][c] = {
            "n": int(len(mc)), "n_reassigned": int(len(mc_r)),
            "corr_shooter_make_c": round(
                float(np.corrcoef(mc["shooter_make_c_p1"], mc["shooter_make_c_sh"])[0, 1]), 6),
            "mean_abs_delta_pp": round(float(np.abs(delta).mean() * 100), 4),
            "rms_delta_pp": round(float(np.sqrt((delta ** 2).mean()) * 100), 4),
            "pct_moved_ge_1pp": round(float((np.abs(delta) >= 0.01).mean() * 100), 4),
            "pct_moved_ge_3pp": round(float((np.abs(delta) >= 0.03).mean() * 100), 4),
            "reassigned_rows": {
                "mean_abs_delta_pp": round(float(np.abs(delta_r).mean() * 100), 4)
                if len(delta_r) else None,
                "pct_moved_ge_1pp": round(float((np.abs(delta_r) >= 0.01).mean() * 100), 4)
                if len(delta_r) else None,
                "pct_moved_ge_3pp": round(float((np.abs(delta_r) >= 0.03).mean() * 100), 4)
                if len(delta_r) else None,
            },
            "shooter_att_c_corr": round(
                float(np.corrcoef(mc["shooter_att_c_p1"], mc["shooter_att_c_sh"])[0, 1]), 6),
            "shooter_games_asof_corr": round(
                float(np.corrcoef(mc["shooter_games_asof_p1"], mc["shooter_games_asof_sh"])[0, 1]), 6),
        }
    return out


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = ES.load_universe(FG.DEFAULT_UNIVERSE, require_pbp_complete=True)
    max_ft = ES.rim_override_for_version("v2")
    print(f"v2 rim override: {max_ft} ft", flush=True)

    out: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(),
                "seasons": list(SEASONS), "rim_override_max_ft": max_ft,
                "by_season_class": {}, "pooled_by_class": {}, "per_team_drop": {},
                "assisted_split": {}}

    frames = []
    for s in SEASONS:
        d = season_class_frame(s, universe, max_ft)
        frames.append(d)
        print(f"{s}: {len(d):,} FGA rows ({time.time() - t0:.0f}s)", flush=True)
    allrows = pd.concat(frames, ignore_index=True)

    for s in SEASONS:
        sd = allrows[allrows["season"] == s]
        for c in FG.SHOT_CLASSES:
            key = f"{s}|{c}"
            cd = sd[sd["shot_class"] == c]
            out["by_season_class"][key] = agg(cd)
            out["by_season_class"][key]["assisted"] = agg(
                cd[cd["assist_known"] & cd["assisted"]])
            out["by_season_class"][key]["unassisted"] = agg(
                cd[cd["assist_known"] & ~cd["assisted"]])
            out["by_season_class"][key]["assist_flag_missing"] = agg(
                cd[~cd["assist_known"]])
            out["per_team_drop"][key] = per_team_drop(cd)

    for c in FG.SHOT_CLASSES:
        cd = allrows[allrows["shot_class"] == c]
        out["pooled_by_class"][c] = agg(cd)
        out["pooled_by_class"][c]["assisted"] = agg(cd[cd["assist_known"] & cd["assisted"]])
        out["pooled_by_class"][c]["unassisted"] = agg(cd[cd["assist_known"] & ~cd["assisted"]])
        out["per_team_drop"][f"pooled|{c}"] = per_team_drop(cd)
        n = len(cd)
        n_ast = int((cd["assist_known"] & cd["assisted"]).sum())
        out["assisted_split"][c] = {
            "n": int(n), "assisted_pct": round(float(n_ast / n * 100), 4) if n else None,
        }
    print(f"raw-feed mismatch pass done ({time.time() - t0:.0f}s)", flush=True)

    out["feature_movement"] = feature_movement(list(SEASONS), universe)
    print(f"feature-movement pass done ({time.time() - t0:.0f}s)", flush=True)

    out["runtime_s"] = round(time.time() - t0, 1)
    jp = OUT_DIR / "shooter_key_audit_v1.json"
    jp.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {jp} ({out['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
