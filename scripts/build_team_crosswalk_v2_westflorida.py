"""
build_team_crosswalk_v2_westflorida.py -- versioned sibling of `data/reference/team_crosswalk.parquet`
that adds West Florida (new ASUN D-I team in 2027; CBBD id 1073, ESPN id 2697).

    .venv/Scripts/python.exe scripts/build_team_crosswalk_v2_westflorida.py

The original file is never touched. The new row's ids come from the CBBD `/teams?season=2027`
pull already on disk (`data/raw/preseason/2027_v2_20260930/teams_2027.parquet`: `id` = CBBD id,
`sourceId` = ESPN id); the same ESPN id 2697 is the `home_id`/`away_id` on the hoopR 2027 schedule.
`kenpom_name` is left null: the name KenPom will use for a first-year D-I team is not knowable
here and is not guessed.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/reference/team_crosswalk.parquet"
OUT = ROOT / "data/reference/team_crosswalk_v2.parquet"
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from cbb_sim.live.preseason import preseason_dir as _preseason_dir, preseason_rel as _preseason_rel  # noqa: E402,F401
TEAMS = _preseason_dir() / "teams_2027.parquet"


def main() -> int:
    cw = pd.read_parquet(SRC)
    t = pd.read_parquet(TEAMS)
    wf = t[t["school"] == "West Florida"]
    assert len(wf) == 1, wf
    r = wf.iloc[0]
    espn, cbbd = int(r["sourceId"]), int(r["id"])
    assert espn not in set(cw["espn_team_id"]) and cbbd not in set(cw["cbbd_team_id"])
    hp = pd.read_parquet(_preseason_dir() / "hoopr_mbb_schedule_2027.parquet",
                         columns=["home_id", "home_display_name", "away_id", "away_display_name"])
    names = set(hp.loc[hp["home_id"].astype("int64") == espn, "home_display_name"]) | \
        set(hp.loc[hp["away_id"].astype("int64") == espn, "away_display_name"])
    assert names == {"West Florida Argonauts"}, names
    row = {"espn_team_id": espn, "espn_name": "West Florida Argonauts",
           "espn_names_by_season": json.dumps({"2027": "West Florida Argonauts"}),
           "first_season": 2027, "last_season": 2027, "cbbd_team_id": cbbd, "cbbd_name": "West Florida",
           "kenpom_name": None, "match_method": "cbbd_teams_2027_sourceId", "match_confidence": 1.0}
    out = pd.concat([cw, pd.DataFrame([row])[cw.columns]], ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(cw)} -> {len(out)} rows (added espn {espn} / cbbd {cbbd})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
