#!/usr/bin/env python
"""
chain_day1_2027_v1.py -- the day-1 (season >= 2027) branches of the chain's `ratings` and `inputs` stages (lane F2, 2026-09-30).
Used by `chain_daily_v3.py` ONLY when the slate's season is >= 2027 (the season switch); fold-2 replay never reaches this module.

NO MODEL CHOICE. The ratings stage reads `data/overrides/ratings_day1_choices.json` (six choices, each marked SERVED DEFAULT or PROPOSED in the
file's `_status` block). `prior_weight_policy` is the one switch: `manifest_uniform` (SERVED: w = 0.8 uniform carry, `build_own_ratings_asof_v1`) or
`arm_C_conference` (selected offline, PENDING as a retrain-set dimension: `build_own_ratings_asof_C_v1`). The default is the served one.

SEAL. Reading the 2026 prior chain needs `seal_lift_approved: true` in the choices file (PM). This module never sets CBB_UNSEAL otherwise; it sets it only
inside `V2.unsealed()` and only when the chain really reaches season 2026 (`season >= 2026` for ratings, any 2027 slate for inputs, whose prior-season
tables are 2026). With the flag false the stages return `blocked` naming the flag.

The inputs branch builds the live engine inputs for the slate (day-1 as-of features: every as-of table is cut to date < slate date, the prior season's
tables carry into the shooter / FT / usage / rebound / rotation priors exactly as `build_engine_inputs_live.build_live` defines them), writes only a census
(the sim stage builds its own copy of the inputs at its own clock), and refuses when the ratings for the slate date do not exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_daily_v2 as V2

REPO = V2.REPO
CHOICES_PATH = V2.CHOICES_PATH
PRESEASON = V2.PRESEASON
OPTIONS = {**V2.DAY1_CHOICES, "prior_weight_policy": ["manifest_uniform", "arm_C_conference"]}
POLICY_DIR = {"manifest_uniform": "ratings_asof", "arm_C_conference": "ratings_asof_C"}


def check_choices(path: Path = CHOICES_PATH, season: int = 2027) -> tuple[dict, list]:
    have = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    need = dict(OPTIONS)
    if season < 2026:                       # the chain does not reach the sealed season
        need.pop("seal_lift_approved")
    miss = [k for k in need if have.get(k) not in need[k]]
    return have, miss


def ratings_out_dir(slate_date, policy: str, dry_run: bool, root: Path = REPO) -> Path:
    base = (root / "results/daily_dry") if dry_run else (root / "data/processed")
    return base / POLICY_DIR[policy] / str(slate_date)


def stage_ratings(ctx, CD, a, season: int | None = None, choices_path: Path = CHOICES_PATH, root: Path = REPO) -> dict:
    season = season or CD.current_season(ctx.slate_date)
    have, miss = check_choices(choices_path, season)
    if miss:
        return {"_status": "blocked", "needs_choices_file": str(Path(choices_path).name),
                "missing_or_invalid": {k: {"implemented_options": OPTIONS[k], "given": have.get(k)} for k in miss},
                "note": "2026 prior chain NOT read (no CBB_UNSEAL)" if "seal_lift_approved" in miss else ""}
    policy = have["prior_weight_policy"]
    src = {"cbbd": f"cbbd:{PRESEASON / 'games_2027.parquet'}|{REPO / 'data/reference/team_crosswalk_v2.parquet'}",
           "schedule": "schedule", "tg": "tg"}[have["teams_source"]]
    if policy == "manifest_uniform":
        import build_own_ratings_asof_v1 as E
        fn = lambda: E.asof_ratings(season, str(ctx.slate_date), root, src, 2022, {})      # noqa: E731
    else:
        import build_own_ratings_asof_C_v1 as EC
        fn = lambda: EC.asof_ratings_C(season, str(ctx.slate_date), root, src, 2022, {})   # noqa: E731
    if season >= 2026:
        with V2.unsealed():
            out, prov = fn()
    else:
        out, prov = fn()
    d = ratings_out_dir(ctx.slate_date, policy, ctx.dry_run, root)
    d.mkdir(parents=True, exist_ok=True)
    out = out.assign(created_at=pd.Timestamp.now("UTC"))
    out.to_parquet(d / f"own_ratings_{season}.parquet", index=False)
    (d / "provenance.json").write_text(json.dumps({**prov, "policy": policy, "choices": {k: have[k] for k in OPTIONS if k in have}},
                                                  indent=2, default=str), encoding="utf-8")
    ctx.state["ratings_dir"] = str(d)
    return {"rows": int(len(out)), "policy": policy, "out": str(d), "dry_run": bool(ctx.dry_run)}


def stage_inputs(ctx, CD, a, SIM, season: int | None = None, pass_name: str = "evening", tips: str = "table") -> dict:
    """Guard first (created_at = the chain clock must precede every simulated game's tip), then the day-1 build."""
    import build_engine_inputs_live as BL
    season = season or CD.current_season(ctx.slate_date)
    slate = SIM.load_slate(str(ctx.slate_date), season, "cbbd", str(PRESEASON / "games_2027.parquet"),
                           str(REPO / "data/reference/team_crosswalk_v2.parquet"), tips)
    if not len(slate):
        return {"_status": "blocked", "reason": f"no slate rows for {ctx.slate_date}"}
    from cbb_sim.live import tips as TP
    ok, late = TP.select_for_pass(slate, ctx.now, pass_name)
    out = {"slate_date": str(ctx.slate_date), "pass": pass_name, "clock": str(ctx.now), "slate_games_mapped": int(len(slate)),
           "slate_games_unmapped_non_d1": len(slate.attrs.get("unmapped", [])), "would_simulate": int(len(ok)),
           "would_refuse": int(len(late)), "placeholder_tips_in_simulate": int(ok["tip_time_is_placeholder"].sum()) if len(ok) else 0}
    if not len(ok):
        return {**out, "_status": "blocked", "blocked_on": ["no game passes the pass filter at this clock"]}
    rd = ctx.state.get("ratings_dir")
    if not rd:
        return {**out, "_status": "blocked",
                "blocked_on": ["own ratings as of the slate date (ratings stage blocked; see its missing_or_invalid)"]}
    cols = ["game_id", "cbbd_game_id", "season", "game_date", "tipoff_utc", "home_team_id", "away_team_id", "neutral"]
    import contextlib
    cm = V2.unsealed() if season >= 2027 else contextlib.nullcontext()
    with cm:                                  # the prior-season carry reads the 2026 tables (needs seal_lift_approved, checked by the ratings stage)
        inp, diag = BL.build_live(ok[cols], ctx.now, season, a.fold, created_at=ctx.now,
                                  season_start=SIM.season_start_of(season, "cbbd", str(PRESEASON / "games_2027.parquet")),
                                  strict_finish=True, ratings_dir=rd)
    ts = inp.team_static
    out.update(inputs_built=True, n_games=int(inp.n_games), rotation_fallback_team_games=int(diag.get("rotation_fallback_team_games", -1)),
               candidates_per_team_game_mean=round(float(diag.get("candidates_per_team_game_mean", 0.0)), 2),
               share_zero_team_static_cells=round(float((ts == 0).mean()), 4), ratings_dir=rd)
    return out
