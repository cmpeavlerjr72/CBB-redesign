#!/usr/bin/env python
"""
build_ao_team_prior_v2.py -- versioned builder of the R9ao3 and-one team-prior table, one served season at a time.

The stored table `ao_team_prior_v1.parquet` was written by `train_foul_r9_v1.run_ao` as
`team_prior_table(ao_design())` over seasons 2022-2025: row (season s, team) holds season s-1 rates (offence and-ones per made FG,
defence and-ones conceded per made FG, each centred on that season's own league mean and shrunk with k=200 made FGs).
Seasons are independent in `team_prior_table`, so the row set for served season S needs ONLY season S-1's made-FG chances.

    .venv/Scripts/python.exe scripts/build_ao_team_prior_v2.py --season 2027                  # reads 2025-26: SEALED -> SealedSeasonError
    .venv/Scripts/python.exe scripts/build_ao_team_prior_v2.py --season 2027 --out data/processed/models/possession_outcome/round9/ao_team_prior_v2.parquet
    .venv/Scripts/python.exe scripts/build_ao_team_prior_v2.py --validate                     # regenerate served seasons 2025 and 2026 from 2024 / 2025
                                                                                               # and compare to the stored v1 table

Design (the made-FG chance set) for source season S-1: possessions_v2 chances, made FGs, inner-joined to the foul-accrual possession
keys (`build_foul_accrual_design_v2._season`, rebuilt for the source season when `foul_accrual_poss_v2.parquet` does not hold it; the
accrual columns are not used by the prior, only the row set). The output is the v1 table rows (verbatim) plus the new season's rows
(`--out` default: round9/ao_team_prior_v2.parquet). The PM switches the served path; this script never touches v1.

SEAL: the source season is passed through `assert_not_sealed` BEFORE any read; --season 2027 needs 2026 (2025-26) and stops unless
CBB_UNSEAL=1 (the seal-week runbook sets it only inside the lifted window).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

R9D = ROOT / "data/processed/models/possession_outcome/round9"
R6 = ROOT / "data/processed/models/possession_outcome/round6"
V1 = R9D / "ao_team_prior_v1.parquet"
V2 = R9D / "ao_team_prior_v2.parquet"
KEYS = ["game_id", "season", "period", "poss_index"]


def accrual_keys(src: int, source: str = "auto") -> pd.DataFrame:
    """Possession keys of the source season (stored accrual table when it holds the season, else a rebuild from pbp)."""
    if source in ("auto", "stored"):
        a = pd.read_parquet(R6 / "foul_accrual_poss_v2.parquet", columns=KEYS)
        a = a[a["season"] == src]
        if len(a) or source == "stored":
            return a
    import build_foul_accrual_design_v2 as AC
    df, _ = AC._season(src)
    u = pd.read_parquet(ROOT / "data/processed/games_universe.parquet", columns=["game_id", "season"])
    return df[KEYS].merge(u, on=["game_id", "season"], how="inner")[KEYS]


def made_fg_design(src: int, acc: pd.DataFrame) -> pd.DataFrame:
    """The columns `team_prior_table` reads (season, offense_team_id, defense_team_id, y_ao) for made-FG chances of season `src`
    that have an accrual possession (same filter as `train_foul_r9_v1.ao_design`)."""
    c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{src}.parquet")
    c = c[(c["fgm_rim"] + c["fgm_jump2"] + c["fgm_3"]) > 0].copy()
    c["season"] = src
    c = c.merge(acc, on=KEYS, how="inner")
    return pd.DataFrame({"season": c["season"].to_numpy(), "offense_team_id": c["offense_team_id"].to_numpy(),
                         "defense_team_id": c["defense_team_id"].to_numpy(), "y_ao": c["and_one"].astype("int8").to_numpy()})


def build_season(season: int, acc_source: str = "auto") -> pd.DataFrame:
    """Rows for served `season` (= source season `season - 1` rates). Seal check first, before any read."""
    src = int(season) - 1
    assert_not_sealed(src, context=f"ao_team_prior_v2: served season {season} needs season {src} and-one data")
    import train_foul_r9_v1 as R9
    d = made_fg_design(src, accrual_keys(src, acc_source))
    t = R9.team_prior_table(d)
    assert set(t["season"].unique()) == {int(season)}
    return t


def validate(acc_source: str = "auto") -> dict:
    """Regenerate served seasons 2025 (from 2024) and 2026 (from 2025) and compare to the stored v1 table."""
    stored = pd.read_parquet(V1)
    rep = {}
    for s in (2025, 2026):
        new = build_season(s, acc_source).sort_values("team_id").reset_index(drop=True)
        old = stored[stored["season"] == s].sort_values("team_id").reset_index(drop=True)
        same_keys = bool(len(new) == len(old) and (new["team_id"].to_numpy() == old["team_id"].to_numpy()).all())
        diffs = {c: float((new[c] - old[c]).abs().max()) for c in ("ao_off_prior_c", "ao_def_prior_c")} if same_keys else None
        rep[s] = {"rows_new": int(len(new)), "rows_stored": int(len(old)), "same_team_ids": same_keys, "max_abs_diff": diffs,
                  "bit_identical": bool(same_keys and all(v == 0.0 for v in diffs.values()))}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, help="served season to add (reads season-1 data only)")
    ap.add_argument("--out", default=str(V2), help="output parquet (v1 rows plus the new season); default round9/ao_team_prior_v2.parquet")
    ap.add_argument("--acc-source", choices=("auto", "stored", "rebuild"), default="auto")
    ap.add_argument("--validate", action="store_true", help="regenerate 2025 and 2026 rows from their source seasons and compare to v1")
    a = ap.parse_args(argv)
    if a.validate:
        rep = validate("stored" if a.acc_source == "auto" else a.acc_source)
        print(json.dumps(rep, indent=1))
        return 0 if all(r["bit_identical"] or (r["max_abs_diff"] and max(r["max_abs_diff"].values()) < 1e-12) for r in rep.values()) else 1
    if a.season is None:
        ap.error("--season or --validate required")
    new = build_season(a.season, a.acc_source)
    base = pd.read_parquet(V1)
    base = base[base["season"] != a.season]
    out = pd.concat([base, new[base.columns]], ignore_index=True).sort_values(["season", "team_id"]).reset_index(drop=True)
    out_path = Path(a.out)
    if out_path.resolve() == V1.resolve():
        raise SystemExit("refusing to overwrite the served v1 table; write a versioned sibling")
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: seasons {sorted(out['season'].unique().tolist())}, new season {a.season} rows {len(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
