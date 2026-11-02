"""Tests for scripts/export_site_day_v1.py: the site bundle for one slate day matches the football files key for key
(days in place of weeks), paths resolve, A_win_prob is mean(home > away) with ties split, and a rebuild merges the season index.

Synthetic 3-game fixture (one game with tie seeds, one neutral site, one placeholder tip, one unnamed athlete, one game with a
stored line, assists zero for everyone like the 2026-11-02 slate) into a
tmp daily root + tmp stage root; no engine run, no network. The football reference key sets are pinned here from
cfb-props-sim/data/processed/site_stage/cfb-sims-2026/2026/ (season_index.json, weeks/week08/) so the test runs without that repo;
when the football checkout is present its live files are checked too."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import export_site_day_v1 as X  # noqa: E402

DAY = "2026-11-02"
RUN = "s40_o0"
CFB_REF = Path(r"C:\Users\devuser\cfb-props-sim\data\processed\site_stage\cfb-sims-2026\2026")
CFB_WEEK = "week08"

# football key sets (reference copies, 2026-10-29)
FB_SEASON_INDEX = {"sport", "season", "convention", "layout", "weeks"}
FB_SEASON_ENTRY = {"week", "id", "dir", "label", "legacy_key", "n_games", "has_open"}
FB_LAYOUT = {"season_index", "week_index", "game_summary", "game_compact", "game_players"}
FB_INDEX = {"sport", "season", "week", "label", "folder_original", "legacy_key", "convention", "n_games", "has_open",
            "updated", "source_tag", "engine", "games"}
FB_INDEX_GAME = {"game_id", "slug", "teamA", "teamB", "A_espn_id", "B_espn_id", "date", "time_utc", "n_sims",
                 "summary_path", "compact_path", "players_path", "players_dist_path", "seeds_path"}
CBB_INDEX_GAME = FB_INDEX_GAME | {"source_tag"}          # per-game run (multi-run union, 2026-11-02)
FB_ENGINE = {"tag", "model", "git", "seeds", "rulebook", "note"}
FB_SUMMARY = {"convention", "sport", "season", "week", "game_id", "slug", "teamA", "teamB", "A_espn_id", "B_espn_id",
              "neutral_site", "date", "time_utc", "time_et", "A_win_prob", "median_margin", "median_total", "p25_margin",
              "p75_margin", "p25_total", "p75_total", "mean_margin", "mean_total", "mean_A_pts", "mean_B_pts", "median_A_pts",
              "median_B_pts", "nsims", "updated", "finalA", "finalB", "odds", "source_tag"}
FB_ODDS = {"spread_open", "over_under_open", "spread_current", "over_under_current", "open_date", "provider_count"}
FB_COMPACT = {"convention", "n_sims", "quantile_levels", "quantiles", "hist", "A_pts", "B_pts", "totals", "spreads", "game_id",
              "slug", "teamA", "teamB", "updated"}
FB_COMPACT_Q = {"margin", "total", "teamA_pts", "teamB_pts"}
FB_COMPACT_H = {"margin", "total"}
FB_PLAYERS = {"convention", "game_id", "slug", "teamA", "teamB", "n_sims", "updated", "stat_keys", "percentiles", "players"}
FB_PLAYER_HEAD = {"player", "team", "role", "n_sims", "sims_used"}
FB_STAT = {"mean", "p10", "p25", "p50", "p75", "p90"}
FB_DIST = {"convention", "game_id", "slug", "teamA", "teamB", "nsims", "updated", "stat_keys", "players"}
FB_DIST_PLAYER = {"player", "team", "role", "stats"}
FB_SEEDS = {"convention", "game_id", "slug", "teamA", "teamB", "nsims", "updated", "seed_order", "stat_keys", "A_pts", "B_pts",
            "players"}


def _day_for_week(keys: set) -> set:
    return (keys - {"week"}) | {"day"}


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory) -> tuple[Path, dict]:
    """A tiny daily root: 3 games x 40 seeds, 4 players a side, one publish with one stored line."""
    root = tmp_path_factory.mktemp("daily")
    rng = np.random.default_rng(7)
    seeds = np.arange(40)
    teams = {101: "Cincinnati", 102: "American University", 103: "Saint Mary's", 104: "Illinois State", 105: "Hawai'i", 106: "Seattle U"}
    slate = pd.DataFrame({
        "game_id": [401000001, 401000002, 401000003], "cbbd_game_id": [1, 2, 3], "season": 2027,
        "game_date": pd.Timestamp(DAY),
        "tipoff_utc": pd.to_datetime([f"{DAY}T23:00:00Z", "2026-11-03T00:30:00Z", "2026-11-03T03:00:00Z"]),
        "home_team_id": [101, 103, 105], "away_team_id": [102, 104, 106], "neutral": [False, True, False],
        "home_name": [teams[101], teams[103], teams[105]], "away_name": [teams[102], teams[104], teams[106]],
        "tip_source": "cbbd", "tip_time_is_placeholder": [False, False, True]})        # game 3: date-only feed row
    grows, prows = [], []
    for g in slate.itertuples(index=False):
        for s in seeds:
            h, a = int(rng.normal(75, 9)), int(rng.normal(70, 9))
            if g.game_id == 401000001 and s < 4:
                a = h                                      # 4 tie seeds: A_win_prob must split them
            grows.append({"game_id": g.game_id, "seed": s, "home_pts": h, "away_pts": a, "possessions": 68.0, "n_periods": 2,
                          "tipoff_utc": g.tipoff_utc, "created_at": pd.Timestamp("2026-10-29T13:00:00Z")})
            for side, tid in (("h", g.home_team_id), ("a", g.away_team_id)):
                for k in range(4):
                    aid = tid * 100 + k
                    if k == 3 and s % 2:
                        continue                           # 4th man sits half the seeds (DNP seeds zero-filled)
                    fgm2r, fgm2j, fgm3, ftm = rng.integers(0, 5), rng.integers(0, 3), rng.integers(0, 4), rng.integers(0, 5)
                    prows.append({"game_id": g.game_id, "seed": s, "athlete_id": aid, "cbbd_id": aid, "team_id": tid,
                                  "minutes": float(rng.uniform(12, 36)) if k < 3 else 0.4, "pts": 2 * (fgm2r + fgm2j) + 3 * fgm3 + ftm,
                                  "reb": rng.integers(0, 10), "ast": 0, "fga": fgm2r + fgm2j + fgm3 + rng.integers(0, 6),
                                  "fg3a": fgm3 + rng.integers(0, 4), "fta": ftm + rng.integers(0, 3), "fgm2_rim": fgm2r,
                                  "fgm2_jump": fgm2j, "fgm3": fgm3, "fouls": rng.integers(0, 5)})
    sd = root / "sim" / DAY / RUN
    sd.mkdir(parents=True)
    pd.DataFrame(grows).to_parquet(sd / "games.parquet")
    pd.DataFrame(prows).to_parquet(sd / "players.parquet")
    slate.to_parquet(sd / "slate.parquet")
    (sd / "run_meta.json").write_text(json.dumps({"engine_tag": f"daily/{DAY}/{RUN}", "seeds": list(map(int, seeds))}))
    (sd / "_DONE.json").write_text("{}")
    pub = root / "publish" / DAY / RUN / "abc123"
    pub.mkdir(parents=True)
    ps = slate.copy()
    ps["provider"] = [None, "DraftKings", None]
    ps["spread"] = [np.nan, -3.5, np.nan]
    ps["total"] = [np.nan, 145.5, np.nan]
    ps["spread_open"] = [np.nan, -2.5, np.nan]
    ps["total_open"] = [np.nan, np.nan, np.nan]
    ps["line_fetched_at"] = pd.Timestamp("2026-11-02T12:00:00Z")
    ps.to_parquet(pub / "slate.parquet")
    pd.DataFrame({"cbbd_game_id": [2], "provider": ["DraftKings"], "spread": [-3.5], "total": [145.5]}).to_parquet(pub / "lines.parquet")
    (pub / "publish_meta.json").write_text(json.dumps({"publish_id": "abc123", "published_at": "2026-10-29 13:00:00+00:00"}))
    # roster: every athlete named except 10503 (the Saint Mary's 4th man, who sits on minutes anyway) and 10601 (plays; unnamed)
    ros = pd.DataFrame([{"espn_player_id": tid * 100 + k, "name": f"{teams[tid].split()[0]} Player{k}",
                         "position": ["Guard", "Forward", "Center", "Guard"][k]}
                        for tid in teams for k in range(4) if tid * 100 + k not in (10303, 10501)])
    rp = root / "roster.parquet"
    ros.to_parquet(rp)
    return root, {"slate": slate, "roster": rp, "espn_ids": {101, 102, 103, 104, 106}}   # 105 Hawai'i: no ESPN id on purpose


@pytest.fixture(scope="module")
def built(fixture_root, tmp_path_factory):
    root, info = fixture_root
    stage = tmp_path_factory.mktemp("stage")
    rep = X.build_day(DAY, RUN, root=root, stage_root=stage, roster=X.load_roster(info["roster"], espn_roster=Path("nope")),
                      espn_ids=info["espn_ids"])
    return stage, rep, info


def _load(stage: Path, rel: str) -> dict:
    return json.loads((stage / "2027" / rel).read_text(encoding="utf-8"))


def test_season_index_keys(built):
    stage, rep, _ = built
    si = _load(stage, "season_index.json")
    assert set(si) == (FB_SEASON_INDEX - {"weeks"}) | {"days"}
    assert set(si["layout"]) == (FB_LAYOUT - {"week_index"}) | {"day_index"}
    assert si["sport"] == "cbb" and si["season"] == 2027 and si["convention"] == X.CONVENTION
    assert len(si["days"]) == 1
    e = si["days"][0]
    assert set(e) == _day_for_week(FB_SEASON_ENTRY) | {"updated", "engine"}
    assert e["day"] == e["id"] == DAY and e["dir"] == f"days/{DAY}" and e["n_games"] == 3 and e["has_open"] is True
    assert e["label"] == "Mon Nov 2"


def test_day_index_keys_and_paths_resolve(built):
    stage, rep, info = built
    idx = _load(stage, f"days/{DAY}/index.json")
    assert set(idx) == _day_for_week(FB_INDEX)
    assert set(idx["engine"]) == FB_ENGINE and idx["engine"]["seeds"] == 40 and idx["engine"]["tag"] == f"daily/{DAY}/{RUN}"
    assert idx["source_tag"] == RUN and idx["n_games"] == 3 and idx["day"] == DAY
    for g in idx["games"]:
        assert set(g) == CBB_INDEX_GAME and g["source_tag"] == RUN
        assert g["slug"] == X.make_slug(g["teamA"], g["teamB"])
        for k in ("summary_path", "compact_path", "players_path", "players_dist_path", "seeds_path"):
            assert g[k].startswith(f"days/{DAY}/games/{g['slug']}/") and (stage / "2027" / g[k]).exists(), g[k]
    slugs = {g["slug"] for g in idx["games"]}
    assert slugs == {"cincinnati_americanuniversity", "saintmarys_illinoisstate", "hawaii_seattleu"}
    by = {g["slug"]: g for g in idx["games"]}
    assert by["hawaii_seattleu"]["A_espn_id"] is None and by["hawaii_seattleu"]["B_espn_id"] == 106
    assert by["cincinnati_americanuniversity"]["time_utc"] == f"{DAY}T23:00:00Z"
    # placeholder tip: no time published, date = the slate day
    assert by["hawaii_seattleu"]["time_utc"] is None and by["hawaii_seattleu"]["date"] == DAY
    assert rep["gaps"]["placeholder_tip"] == [401000003] and rep["gaps"]["dropped_stats"] == ["ast", "pra"]


def test_summary_keys_win_prob_and_odds(built):
    stage, rep, info = built
    s = _load(stage, f"days/{DAY}/games/cincinnati_americanuniversity/summary.json")
    assert set(s) == _day_for_week(FB_SUMMARY) and set(s["odds"]) == FB_ODDS
    assert s["sport"] == "cbb" and s["season"] == 2027 and s["day"] == DAY and s["convention"] == X.CONVENTION
    assert s["finalA"] is None and s["finalB"] is None and s["source_tag"] == RUN
    # A_win_prob = mean(home > away) with ties split, straight from the compact series (same seeds)
    c = _load(stage, f"days/{DAY}/games/cincinnati_americanuniversity/compact.json")
    A, B = np.array(c["A_pts"]), np.array(c["B_pts"])
    assert (A == B).sum() >= 4                                        # the 4 forced ties (plus any natural ones)
    assert s["A_win_prob"] == pytest.approx(float((A > B).mean() + 0.5 * (A == B).mean()), abs=1e-4)
    assert s["mean_margin"] == pytest.approx(float((A - B).mean()), abs=1e-4)
    assert s["time_et"] == "18:00" and s["date"] == DAY and s["neutral_site"] is False
    assert s["odds"]["provider_count"] == 0 and s["odds"]["spread_current"] is None
    s2 = _load(stage, f"days/{DAY}/games/saintmarys_illinoisstate/summary.json")
    assert s2["neutral_site"] is True and s2["time_et"] == "19:30"
    s3 = _load(stage, f"days/{DAY}/games/hawaii_seattleu/summary.json")
    assert s3["time_utc"] is None and s3["time_et"] == "TBD" and s3["date"] == DAY
    assert s2["odds"] == {"spread_open": -2.5, "over_under_open": 145.5, "spread_current": -3.5, "over_under_current": 145.5,
                          "open_date": "2026-11-02", "provider_count": 1}


def test_lines_json_matches_football_shape_and_sim_probs(built):
    stage, _, _ = built
    lj = _load(stage, f"days/{DAY}/lines.json")
    assert {"season", "week", "generated_at", "convention", "spread_total_price", "games"} <= set(lj)
    assert lj["season"] == 2027 and lj["week"] == 20261102 and lj["spread_total_price"] == -110
    assert list(lj["games"]) == ["saintmarys_illinoisstate"]          # only the game with a stored line
    g = lj["games"]["saintmarys_illinoisstate"]
    assert set(g) == {"game_id", "teamA", "teamB", "spread", "total", "ml", "sim"}
    assert g["spread"] == {"open": -2.5, "close": -3.5, "n_open": 1, "n_close": 1}
    assert g["total"]["close"] == 145.5 and g["total"]["open"] == 145.5
    c = _load(stage, f"days/{DAY}/games/saintmarys_illinoisstate/compact.json")
    A, B = np.array(c["A_pts"]), np.array(c["B_pts"])
    assert g["sim"]["n"] == 40
    assert g["sim"]["spread"]["close"]["home"] == pytest.approx(float(((A - B) - 3.5 > 0).mean()), abs=1e-4)
    assert g["sim"]["total"]["close"]["over"] == pytest.approx(float(((A + B) > 145.5).mean()), abs=1e-4)
    s = _load(stage, f"days/{DAY}/games/saintmarys_illinoisstate/summary.json")
    assert g["sim"]["ml"]["home"] == pytest.approx(s["A_win_prob"], abs=1e-4)


def test_compact_keys(built):
    stage, _, _ = built
    c = _load(stage, f"days/{DAY}/games/hawaii_seattleu/compact.json")
    assert set(c) == FB_COMPACT and set(c["quantiles"]) == FB_COMPACT_Q and set(c["hist"]) == FB_COMPACT_H
    assert c["n_sims"] == 40 == len(c["A_pts"]) == len(c["spreads"]) and c["quantile_levels"] == list(range(1, 100))
    assert c["spreads"] == [-(a - b) for a, b in zip(c["A_pts"], c["B_pts"])]
    assert sum(h["count"] for h in c["hist"]["total"]) == 40


def test_players_dist_seeds_keys_and_alignment(built):
    stage, rep, _ = built
    p = _load(stage, f"days/{DAY}/games/hawaii_seattleu/players.json")
    d = _load(stage, f"days/{DAY}/games/hawaii_seattleu/players_dist.json")
    sd = _load(stage, f"days/{DAY}/games/hawaii_seattleu/seeds.json")
    assert set(p) == FB_PLAYERS and set(d) == FB_DIST and set(sd) == FB_SEEDS
    # assists are zero for everyone on the day -> `ast` and the derived `pra` leave every file; nothing else does
    keys = sorted(k for k in X.STAT_KEYS if k not in ("ast", "pra"))
    assert p["stat_keys"] == keys == d["stat_keys"] and p["percentiles"] == ["p10", "p25", "p50", "p75", "p90"]
    assert sd["stat_keys"] == ["pts", "reb", "fg3m"]
    assert all(col.rsplit("|", 1)[1] in ("pts", "reb", "fg3m") for col in sd["players"])
    # 3 men a side (the 4th averages < 1 minute and is dropped); home team first
    assert len(p["players"]) == 6 and [e["team"] for e in p["players"]] == ["Hawai'i"] * 3 + ["Seattle U"] * 3
    for e in p["players"]:
        assert set(e) == FB_PLAYER_HEAD | set(keys)
        assert all(set(e[k]) == FB_STAT for k in keys)
        assert e["n_sims"] == 40 and e["sims_used"] == 40
    assert {e["role"] for e in p["players"]} == {"Guard", "Forward", "Center", ""}
    unnamed = [e for e in p["players"] if e["player"].startswith("Player ")]
    assert len(unnamed) == 1 and unnamed[0]["player"] == "Player 10501" and unnamed[0]["role"] == ""
    assert 10501 in rep["gaps"]["unnamed"]
    for de in d["players"]:
        assert set(de) == FB_DIST_PLAYER and set(de["stats"]) == set(keys)
        for st in keys:
            assert sum(de["stats"][st]["pmf"].values()) == 40
    # seeds.json: every column 40 long, mean equals players.json mean, PMF rebuilt from the series equals players_dist
    by, dby = {e["player"]: e for e in p["players"]}, {e["player"]: e for e in d["players"]}
    assert len(sd["A_pts"]) == 40 and sd["A_pts"] == _load(stage, f"days/{DAY}/games/hawaii_seattleu/compact.json")["A_pts"]
    for col, arr in sd["players"].items():
        player, st = col.rsplit("|", 1)
        assert st in sd["stat_keys"] and len(arr) == 40
        assert round(float(np.mean(arr)), 2) == by[player][st]["mean"]
        vals, counts = np.unique(np.array(arr), return_counts=True)
        assert {str(int(a)): int(b) for a, b in zip(vals, counts)} == dby[player]["stats"][st]["pmf"]


def test_active_stats_keeps_everything_when_nothing_is_zero_and_drops_pra_with_ast():
    rows = pd.DataFrame({"minutes": [30.0, 20.0], "pts": [10, 4], "reb": [3, 0], "ast": [2, 1], "fga": [8, 3], "fg3a": [2, 1],
                         "fta": [2, 0], "fgm2_rim": [2, 1], "fgm2_jump": [1, 0], "fgm3": [1, 0], "fouls": [0, 0]})
    stats, dropped = X.active_stats(rows)
    assert dropped == ["fouls"] and "fouls" not in stats.values() and "pra" in stats.values()
    stats, dropped = X.active_stats(rows.assign(ast=0, fouls=2))
    assert dropped == ["ast", "pra"] and set(stats.values()) == set(X.STAT_KEYS) - {"ast", "pra"}
    assert X.active_stats(rows.iloc[:0])[1] == X.STAT_KEYS


def test_rebuild_merges_season_index_without_duplicating_the_day(fixture_root, tmp_path):
    root, info = fixture_root
    stage = tmp_path / "stage"
    roster = X.load_roster(info["roster"], espn_roster=Path("nope"))
    X.build_day(DAY, RUN, root=root, stage_root=stage, roster=roster, espn_ids=info["espn_ids"])
    # another day already on the stage must survive, the rebuilt day must appear once
    si = json.loads((stage / "2027" / "season_index.json").read_text())
    si["days"].append({"day": "2026-11-03", "id": "2026-11-03", "dir": "days/2026-11-03", "label": "Tue Nov 3", "legacy_key": "2026-11-03",
                       "n_games": 5, "has_open": False, "updated": "x", "engine": {}})
    (stage / "2027" / "season_index.json").write_text(json.dumps(si))
    stale = stage / "2027" / "days" / DAY / "games" / "stale_slug" / "summary.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    rep = X.build_day(DAY, RUN, root=root, stage_root=stage, roster=roster, espn_ids=info["espn_ids"])
    si2 = json.loads((stage / "2027" / "season_index.json").read_text())
    assert [d["day"] for d in si2["days"]] == [DAY, "2026-11-03"]
    assert si2["days"][0]["n_games"] == 3 and si2["days"][0]["updated"] == rep["updated"]
    assert not stale.exists()                                       # the day folder is rebuilt from scratch
    assert X.newest_run_id(root, DAY) == RUN and X.newest_publish_dir(root, DAY, RUN).name == "abc123"


def _clone_run(root: Path, src: str, dst: str, keep_games: list[int], mtime: float) -> None:
    """Copy sim run `src` to `dst` keeping only `keep_games`; stamp _DONE.json so run order is deterministic."""
    import os
    import shutil
    sd, dd = root / "sim" / DAY / src, root / "sim" / DAY / dst
    shutil.copytree(sd, dd)
    for f in ("games.parquet", "players.parquet", "slate.parquet"):
        df = pd.read_parquet(dd / f)
        df[df["game_id"].isin(keep_games)].to_parquet(dd / f)
    os.utime(dd / "_DONE.json", (mtime, mtime))


def test_multi_run_union_newest_run_wins_and_finals_from_verified_file(fixture_root, tmp_path):
    import os
    import shutil
    root0, info = fixture_root
    root = tmp_path / "daily"
    shutil.copytree(root0, root)
    os.utime(root / "sim" / DAY / RUN / "_DONE.json", (1_000, 1_000))      # oldest run: all 3 games
    # morning-style run: skipped the first (already tipped) game
    _clone_run(root, RUN, "s40_o0_morning", [401000002, 401000003], 2_000)
    # a smoke run without players.parquet is never used
    junk = root / "sim" / DAY / "s4_o0"
    junk.mkdir()
    (junk / "_DONE.json").write_text("{}")
    os.utime(junk / "_DONE.json", (3_000, 3_000))
    assert X.finished_runs(root, DAY) == ["s40_o0_morning", RUN]
    verified = tmp_path / "finals_verified.parquet"
    base = {"hoopr_status": "STATUS_FINAL", "cbbd_status": "final"}
    pd.DataFrame([
        {"game_id": 401000001, "hoopr_home": 80, "hoopr_away": 71, "cbbd_home": 80, "cbbd_away": 71.0, **base},
        {"game_id": 401000002, "hoopr_home": 70, "hoopr_away": 60, "cbbd_home": 70, "cbbd_away": 61.0, **base},   # disagree
        {"game_id": 401000003, "hoopr_home": 50, "hoopr_away": 40, "cbbd_home": 50, "cbbd_away": 40.0,
         "hoopr_status": "STATUS_IN_PROGRESS", "cbbd_status": "in_progress"},                                  # not final
    ]).to_parquet(verified)
    stage = tmp_path / "stage"
    roster = X.load_roster(info["roster"], espn_roster=Path("nope"))
    rep = X.build_day(DAY, None, root=root, stage_root=stage, roster=roster, espn_ids=info["espn_ids"], verified_finals=verified)
    day_dir = stage / "2027" / "days" / DAY
    idx = json.loads((day_dir / "index.json").read_text())
    assert idx["n_games"] == 3 and rep["n_games"] == 3 and idx["source_tag"] == "s40_o0_morning"
    tags = {g["slug"]: g["source_tag"] for g in idx["games"]}
    assert tags == {"cincinnati_americanuniversity": RUN, "saintmarys_illinoisstate": "s40_o0_morning",
                    "hawaii_seattleu": "s40_o0_morning"}
    assert [g["game_id"] for g in idx["games"]] == [401000001, 401000002, 401000003]       # tip order
    summ = {}
    for slug, tag in tags.items():
        summ[slug] = json.loads((day_dir / "games" / slug / "summary.json").read_text())
        assert summ[slug]["source_tag"] == tag
        assert all((day_dir / "games" / slug / f).exists() for f in X.GAME_FILES)
    cin = summ["cincinnati_americanuniversity"]
    assert (cin["finalA"], cin["finalB"]) == (80, 71)
    assert summ["saintmarys_illinoisstate"]["finalA"] is None and summ["hawaii_seattleu"]["finalA"] is None   # disagree / live
    # the grade ledger is the first source: it overrides the verified file
    (root / "grade").mkdir()
    pd.DataFrame({"game_id": [401000001], "home_score": [90], "away_score": [88]}).to_parquet(root / "grade" / "ledger.parquet")
    assert X.load_finals(root, [401000001, 401000002], verified) == {401000001: (90, 88)}
    # an explicit run id exports that run alone
    rep1 = X.build_day(DAY, "s40_o0_morning", root=root, stage_root=tmp_path / "stage1", roster=roster,
                       espn_ids=info["espn_ids"], verified_finals=verified)
    assert rep1["n_games"] == 2


def test_slug_matches_football_convention():
    assert X.make_slug("Texas A&M", "Hawai'i") == "texasam_hawaii"
    assert X.make_slug("Saint Mary's", "UConn") == "saintmarys_uconn"


@pytest.mark.skipif(not (CFB_REF / "season_index.json").exists(), reason="football reference stage not on this machine")
def test_pinned_football_key_sets_match_the_live_reference_files():
    si = json.loads((CFB_REF / "season_index.json").read_text())
    assert set(si) == FB_SEASON_INDEX and set(si["layout"]) == FB_LAYOUT and set(si["weeks"][0]) >= FB_SEASON_ENTRY
    idx = json.loads((CFB_REF / "weeks" / CFB_WEEK / "index.json").read_text())
    assert set(idx) == FB_INDEX and set(idx["games"][0]) == FB_INDEX_GAME and set(idx["engine"]) >= FB_ENGINE
    gdir = CFB_REF / "weeks" / CFB_WEEK / "games" / idx["games"][0]["slug"]
    s = json.loads((gdir / "summary.json").read_text())
    assert set(s) == FB_SUMMARY and set(s["odds"]) == FB_ODDS
    c = json.loads((gdir / "compact.json").read_text())
    assert set(c) == FB_COMPACT and set(c["quantiles"]) == FB_COMPACT_Q and set(c["hist"]) == FB_COMPACT_H
    p = json.loads((gdir / "players.json").read_text())
    assert set(p) == FB_PLAYERS and set(p["players"][0]) >= FB_PLAYER_HEAD
    d = json.loads((gdir / "players_dist.json").read_text())
    assert set(d) == FB_DIST and set(d["players"][0]) == FB_DIST_PLAYER
    sd = json.loads((gdir / "seeds.json").read_text())
    assert set(sd) == FB_SEEDS
