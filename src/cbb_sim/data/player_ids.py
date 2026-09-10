"""
player_ids.py -- CBBD <-> ESPN **player** id crosswalk.

Why this exists
---------------
`docs/SIM_GUARDRAILS.md` section 4 ("Lineups"): hoopR pbp has no on-floor field,
CBBD pbp carries all ten players per play, so every lineup / rotation feature is
built in **CBBD player-id space** (`on_floor_h1..h5` / `on_floor_a1..a5` on
`data/processed/possessions/possessions_{season}.parquet`). Every *box-score*
quantity the rotation model needs as a pregame input -- minutes, `starter`,
`did_not_play` (the historical availability field, hoopR `active` being a
placeholder before 2025-26) -- lives in hoopR `player_box`, keyed on the **ESPN**
`athlete_id`. The bridge between the two id spaces is a prerequisite for L4 and
"its match rate is a reported number".

Matching methodology
--------------------
- **Primary, exact: `source_id`.** CBBD's `/teams/roster` returns, per player,
  both its own `id` (the id that appears in pbp `onFloor` / `participants`) and
  a `sourceId` which is the ESPN athlete id verbatim (visible on the Duke 2025
  roster sample already on disk: CBBD 204 "Khaman Maluach" -> sourceId
  "5203685", that player's ESPN id; confirmed at scale here by the fraction of
  `sourceId`s that land on a hoopR `player_box.athlete_id` in the same season).
  One `/teams/roster?season=S` call returns **every** team's roster for the
  season, so the whole crosswalk costs one API call per season. Rows whose
  `sourceId` is also observed in that season's hoopR `player_box` are
  `source_id_verified`; rows whose `sourceId` is present but never seen in
  hoopR that season (a rostered player who never dressed, or a hoopR gap) are
  `source_id_unverified` -- the id is still an ESPN id by construction, it just
  has no box-score rows to attach to.

- **Fallback 1: `name_team_jersey`.** Normalized name + ESPN team id + jersey,
  for CBBD roster rows with a null/blank `sourceId`. The ESPN team id comes
  from `data/reference/team_crosswalk.parquet` (`cbbd_team_id` -> `espn_team_id`).

- **Fallback 2: `name_team`.** Same without the jersey, accepted only when the
  (normalized name, espn_team_id, season) key is unique on both sides.

- **Fallback 3: `name_team_pbp`.** CBBD player ids that appear in pbp
  `onFloor` but on **no** roster row at all (mid-season additions, roster
  endpoint gaps). Their name and team come from the pbp payload itself and are
  matched by normalized name + ESPN team id + season against hoopR `player_box`.

Anything still unresolved is written out with `match_method = "unmatched"`
rather than guessed at, and both the roster-row match rate and the
**possession-weighted** on-floor match rate are reported (the second is the one
that matters: a crosswalk that misses only players who never take the floor
costs the rotation model nothing).

Outputs
-------
`data/processed/player_crosswalk.parquet` -- one row per
(season, cbbd_player_id), plus `data/processed/player_crosswalk_report.json`.

Entry point: `scripts/build_player_crosswalk.py`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROSTER_DIR = Path("data/raw/cbbd/rosters")
DEFAULT_PLAYER_BOX_DIR = Path("data/raw/hoopr/player_box")
DEFAULT_POSSESSIONS_DIR = Path("data/processed/possessions")
DEFAULT_TEAM_CROSSWALK = Path("data/reference/team_crosswalk.parquet")
DEFAULT_OUT = Path("data/processed/player_crosswalk.parquet")

#: generational suffixes stripped before name comparison
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

#: Letters with a stroke or a ligature do NOT decompose under NFKD, so a bare
#: combining-mark strip DELETES them ("Nikola Đurišić" -> "nikola urisic", which
#: matches nothing). Transliterated explicitly instead.
_TRANSLIT = str.maketrans({
    "Đ": "Dj", "đ": "dj",   # D/d with stroke
    "Ø": "O", "ø": "o",     # O/o with stroke
    "Ł": "L", "ł": "l",     # L/l with stroke
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "ß": "ss",
    "İ": "I", "ı": "i",     # Turkish dotted/dotless i
    "Ð": "D", "ð": "d",     # eth
    "Þ": "Th", "þ": "th",   # thorn
})

ON_FLOOR_COLS = [f"on_floor_h{i}" for i in range(1, 6)] + [f"on_floor_a{i}" for i in range(1, 6)]


def normalize_name(name: object) -> str:
    """Lowercase, strip accents, punctuation and generational suffixes.

    'Bobby Pettiford Jr.' -> 'bobby pettiford'; "D'Andre Jackson" -> 'dandre jackson'.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", "", s)
    parts = [p for p in s.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def normalize_jersey(j: object) -> str:
    """'03' and '3' are the same shirt; '' and None mean 'no jersey known'."""
    if j is None or (isinstance(j, float) and np.isnan(j)):
        return ""
    s = str(j).strip()
    if s in {"", "nan", "None"}:
        return ""
    if s.isdigit():
        return str(int(s))
    return s


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_cbbd_rosters(seasons, roster_dir: Path = DEFAULT_ROSTER_DIR) -> pd.DataFrame:
    """One row per (season, cbbd_player_id) from the pulled `/teams/roster` dumps."""
    frames = []
    for season in seasons:
        path = Path(roster_dir) / f"roster_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run scripts/build_player_crosswalk.py --pull first"
            )
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["season", "cbbd_player_id"], keep="first")


def load_hoopr_players(seasons, player_box_dir: Path = DEFAULT_PLAYER_BOX_DIR) -> pd.DataFrame:
    """One row per (season, athlete_id, team_id) with the box-score footprint used
    for matching and for reporting how much box data a matched id actually has."""
    frames = []
    for season in seasons:
        path = Path(player_box_dir) / f"player_box_{season}.parquet"
        pb = pd.read_parquet(
            path,
            columns=["season", "athlete_id", "athlete_display_name", "team_id",
                     "athlete_jersey", "minutes", "did_not_play"],
        )
        pb = pb[pb["athlete_id"].notna()].copy()
        pb["minutes"] = pd.to_numeric(pb["minutes"], errors="coerce")
        g = (
            pb.groupby(["season", "athlete_id", "team_id"], as_index=False)
            .agg(
                espn_name=("athlete_display_name", "first"),
                espn_jersey=("athlete_jersey", "first"),
                box_games=("minutes", "size"),
                box_minutes=("minutes", "sum"),
            )
        )
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    df["espn_name_norm"] = df["espn_name"].map(normalize_name)
    df["espn_jersey_norm"] = df["espn_jersey"].map(normalize_jersey)
    df = df.rename(columns={"athlete_id": "espn_athlete_id", "team_id": "espn_team_id"})
    df["espn_athlete_id"] = df["espn_athlete_id"].astype("int64")
    df["espn_team_id"] = df["espn_team_id"].astype("int64")
    df["season"] = df["season"].astype("int64")
    return df


def load_team_crosswalk(path: Path = DEFAULT_TEAM_CROSSWALK) -> pd.DataFrame:
    tc = pd.read_parquet(path)[["espn_team_id", "cbbd_team_id", "espn_name", "cbbd_name"]]
    tc = tc[tc["cbbd_team_id"].notna()]
    return tc.astype({"cbbd_team_id": "int64", "espn_team_id": "int64"})


def onfloor_player_weights(seasons, poss_dir: Path = DEFAULT_POSSESSIONS_DIR) -> pd.DataFrame:
    """(season, cbbd_player_id) -> possessions that player is on the floor for.
    This is the weight the match rate that matters is computed under."""
    rows = []
    for season in seasons:
        path = Path(poss_dir) / f"possessions_{season}.parquet"
        p = pd.read_parquet(path, columns=ON_FLOOR_COLS)
        vals = p.to_numpy(dtype="float64").ravel()
        vals = vals[~np.isnan(vals)]
        ids, cnt = np.unique(vals.astype("int64"), return_counts=True)
        rows.append(pd.DataFrame({"season": int(season), "cbbd_player_id": ids,
                                  "on_floor_poss": cnt}))
    return pd.concat(rows, ignore_index=True)


def pbp_player_names(seasons, pbp_dir: Path = Path("data/raw/cbbd/pbp")) -> pd.DataFrame:
    """(season, cbbd_player_id) -> (name, team) harvested from the pbp
    `on_floor_json` payload -- the only name source for a CBBD id on no roster row."""
    out = []
    for season in seasons:
        path = Path(pbp_dir) / f"plays_{season}.parquet"
        df = pd.read_parquet(path, columns=["gameId", "id", "on_floor_json"])
        df = df.drop_duplicates(subset=["gameId", "id"])
        df = df[df["on_floor_json"].notna()]
        df = df.groupby("gameId", group_keys=False).head(4)
        seen: dict[int, tuple[str, str]] = {}
        for payload in df["on_floor_json"].to_numpy():
            try:
                items = json.loads(payload)
            except (TypeError, ValueError):
                continue
            for it in items or []:
                pid = it.get("id")
                if pid is None:
                    continue
                pid = int(pid)
                if pid in seen:
                    continue
                seen[pid] = (it.get("name") or "", it.get("team") or "")
        out.append(pd.DataFrame({
            "season": int(season),
            "cbbd_player_id": list(seen.keys()),
            "pbp_name": [v[0] for v in seen.values()],
            "pbp_team": [v[1] for v in seen.values()],
        }))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Crosswalk build
# ---------------------------------------------------------------------------
def _unique_key_map(hoopr: pd.DataFrame, cols: list[str]) -> dict:
    """Map from a tuple key to espn_athlete_id, dropping any key that is not
    unique on the hoopR side (two same-named team-mates -> no guess)."""
    h = hoopr.copy()
    h["_k"] = list(zip(*[h[c] for c in cols]))
    h = h.drop_duplicates(subset=["_k", "espn_athlete_id"])
    h = h.drop_duplicates(subset=["_k"], keep=False)
    return dict(zip(h["_k"], h["espn_athlete_id"]))


def build_crosswalk(
    seasons,
    roster_dir: Path = DEFAULT_ROSTER_DIR,
    player_box_dir: Path = DEFAULT_PLAYER_BOX_DIR,
    poss_dir: Path = DEFAULT_POSSESSIONS_DIR,
    team_crosswalk_path: Path = DEFAULT_TEAM_CROSSWALK,
    pbp_dir: Path = Path("data/raw/cbbd/pbp"),
) -> tuple[pd.DataFrame, dict]:
    seasons = [int(s) for s in seasons]
    rosters = load_cbbd_rosters(seasons, roster_dir).copy()
    hoopr = load_hoopr_players(seasons, player_box_dir)
    teams = load_team_crosswalk(team_crosswalk_path)

    rosters["season"] = rosters["season"].astype("int64")
    rosters = rosters.merge(teams[["cbbd_team_id", "espn_team_id"]], on="cbbd_team_id", how="left")
    rosters["name_norm"] = rosters["name"].map(normalize_name)
    rosters["jersey_norm"] = rosters["jersey"].map(normalize_jersey)

    # --- primary: sourceId --------------------------------------------------
    src = pd.to_numeric(rosters["source_id"], errors="coerce")
    rosters["espn_athlete_id"] = src.astype("float64")
    rosters["match_method"] = np.where(src.notna(), "source_id_unverified", "")
    rosters["match_confidence"] = np.where(src.notna(), 0.90, 0.0)

    hoopr_keys = set(zip(hoopr["season"].tolist(), hoopr["espn_athlete_id"].tolist()))
    verified = np.array([
        (not np.isnan(a)) and (int(s), int(a)) in hoopr_keys
        for s, a in zip(rosters["season"].to_numpy(), rosters["espn_athlete_id"].to_numpy())
    ])
    rosters.loc[verified, "match_method"] = "source_id_verified"
    rosters.loc[verified, "match_confidence"] = 1.0

    # --- fallback 1: name + team + jersey -----------------------------------
    need = rosters["match_method"] == ""
    if need.any():
        uj = _unique_key_map(hoopr, ["season", "espn_team_id", "espn_name_norm", "espn_jersey_norm"])
        keys = list(zip(rosters.loc[need, "season"], rosters.loc[need, "espn_team_id"],
                        rosters.loc[need, "name_norm"], rosters.loc[need, "jersey_norm"]))
        hit = pd.Series([uj.get(k, np.nan) for k in keys], index=rosters.index[need]).dropna()
        rosters.loc[hit.index, "espn_athlete_id"] = hit.values
        rosters.loc[hit.index, "match_method"] = "name_team_jersey"
        rosters.loc[hit.index, "match_confidence"] = 0.95

    # --- fallback 2: name + team --------------------------------------------
    need = rosters["match_method"] == ""
    if need.any():
        un = _unique_key_map(hoopr, ["season", "espn_team_id", "espn_name_norm"])
        keys = list(zip(rosters.loc[need, "season"], rosters.loc[need, "espn_team_id"],
                        rosters.loc[need, "name_norm"]))
        hit = pd.Series([un.get(k, np.nan) for k in keys], index=rosters.index[need]).dropna()
        rosters.loc[hit.index, "espn_athlete_id"] = hit.values
        rosters.loc[hit.index, "match_method"] = "name_team"
        rosters.loc[hit.index, "match_confidence"] = 0.85

    rosters.loc[rosters["match_method"] == "", "match_method"] = "unmatched"

    keep = ["season", "cbbd_player_id", "name", "name_norm", "jersey_norm", "position",
            "cbbd_team_id", "espn_team_id", "espn_athlete_id", "match_method", "match_confidence"]
    cw = rosters[keep].copy()

    # --- fallback 3: pbp-only ids (on no roster row at all) -----------------
    weights = onfloor_player_weights(seasons, poss_dir)
    known = set(zip(cw["season"].astype("int64").tolist(),
                    cw["cbbd_player_id"].astype("int64").tolist()))
    mask = np.array([(int(s), int(p)) not in known
                     for s, p in zip(weights["season"], weights["cbbd_player_id"])])
    missing = weights[mask].copy()
    if len(missing):
        names = pbp_player_names(seasons, pbp_dir)
        missing = missing.merge(names, on=["season", "cbbd_player_id"], how="left")
        missing["name_norm"] = missing["pbp_name"].map(normalize_name)
        tname = teams.copy()
        tname["cbbd_name_key"] = tname["cbbd_name"].map(normalize_name)
        missing["team_key"] = missing["pbp_team"].map(normalize_name)
        missing = missing.merge(tname[["cbbd_name_key", "espn_team_id", "cbbd_team_id"]],
                                left_on="team_key", right_on="cbbd_name_key", how="left")
        un = _unique_key_map(hoopr, ["season", "espn_team_id", "espn_name_norm"])
        keys = list(zip(missing["season"], missing["espn_team_id"], missing["name_norm"]))
        missing["espn_athlete_id"] = [un.get(k, np.nan) for k in keys]
        ok = missing["espn_athlete_id"].notna()
        missing["match_method"] = np.where(ok, "name_team_pbp", "unmatched")
        missing["match_confidence"] = np.where(ok, 0.80, 0.0)
        missing["name"] = missing["pbp_name"]
        missing["jersey_norm"] = ""
        missing["position"] = None
        cw = pd.concat([cw, missing[keep]], ignore_index=True)

    cw = cw.merge(weights, on=["season", "cbbd_player_id"], how="left")
    cw["on_floor_poss"] = cw["on_floor_poss"].fillna(0).astype("int64")
    cw["espn_athlete_id"] = cw["espn_athlete_id"].astype("float64")
    cw = cw.sort_values(["season", "cbbd_player_id"]).reset_index(drop=True)

    return cw, crosswalk_report(cw)


def crosswalk_report(cw: pd.DataFrame) -> dict:
    """Row-level and possession-weighted match rates, overall and per season."""
    def block(df: pd.DataFrame) -> dict:
        m = df["espn_athlete_id"].notna()
        of = df["on_floor_poss"] > 0
        w = df["on_floor_poss"].to_numpy(dtype="float64")
        return {
            "rows": int(len(df)),
            "matched_rows": int(m.sum()),
            "row_match_rate": float(m.mean()) if len(df) else float("nan"),
            "on_floor_players": int(of.sum()),
            "on_floor_matched": int((m & of).sum()),
            "on_floor_player_match_rate": float(m[of].mean()) if of.any() else float("nan"),
            "possession_weighted_match_rate": (
                float(w[m.to_numpy()].sum() / w.sum()) if w.sum() else float("nan")
            ),
            "by_method": {str(k): int(v) for k, v in df["match_method"].value_counts().items()},
        }

    rep = {"overall": block(cw)}
    for season, g in cw.groupby("season"):
        rep[str(int(season))] = block(g)
    unmatched = cw[cw["espn_athlete_id"].isna() & (cw["on_floor_poss"] > 0)]
    rep["unmatched_on_floor_top20"] = (
        unmatched.sort_values("on_floor_poss", ascending=False)
        .head(20)[["season", "cbbd_player_id", "name", "on_floor_poss"]]
        .to_dict("records")
    )
    return rep


def load_crosswalk(path: Path = DEFAULT_OUT) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/build_player_crosswalk.py")
    return pd.read_parquet(path)


def cbbd_to_espn_map(cw: pd.DataFrame, season: int) -> dict[int, int]:
    """{cbbd_player_id: espn_athlete_id} for one season."""
    g = cw[(cw["season"] == int(season)) & cw["espn_athlete_id"].notna()]
    return dict(zip(g["cbbd_player_id"].astype("int64"), g["espn_athlete_id"].astype("int64")))
