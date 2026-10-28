"""Round 3 per-FGA context table (shot_location experiments.md R3.2-R3.4).

Usage: python scripts/build_shot_context_v1.py SEASON   (2024 or 2025; 2026 is sealed and refused)
Adapted from the partner b_features.py (Heuermann study), with our flattened CBBD columns, neutral site from
the games file (hard error on a failed join), D1-vs-D1 from games-file conferences, and a score-convention check.
"""
import sys, json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/models/shot_location/round3"
SHOT_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot"}
TIMEOUTS = ("ShortTimeOut", "RegularTimeOut", "OfficialTVTimeOut")
HX, HY = 52.5, 250.0
COLS = ["gameId", "id", "tournament", "period", "secondsRemaining", "playType", "team",
        "homeScore", "awayScore", "playText", "shot_made", "shot_range", "shot_location_x", "shot_location_y"]


def possession_context(g):
    """Partner walker: for each FGA return (origin, seconds since possession start, after-timeout)."""
    teams = [t for t in g.team.dropna().unique()]
    other = {teams[0]: teams[1], teams[1]: teams[0]} if len(teams) == 2 else {}
    poss, origin, start, ato, cur_period = None, "PERIOD_START", None, False, None
    st = g.playType == "Steal"
    steal_at = set(zip(g.period[st], g.secondsRemaining[st]))
    res = {}
    for r in g.itertuples(index=False):
        if r.period != cur_period:
            cur_period, poss, origin, ato = r.period, None, "PERIOD_START", False
            start = 1200.0 if r.period <= 2 else 300.0
        t, pt, sec = r.team, r.playType, float(r.secondsRemaining)
        if pt in SHOT_TYPES:
            o = origin if poss == t else ("UNKNOWN" if poss is not None else origin)
            res[r.id] = (o, max(0.0, (start if start is not None else sec) - sec), ato)
            if r.shot_made is True or r.shot_made == True:  # noqa: E712
                if t in other:
                    poss, origin, start, ato = other[t], "OPP_MAKE", sec, False
        elif pt == "Defensive Rebound" and t in other:
            poss, origin, start, ato = t, "DRB", sec, False
        elif pt == "Offensive Rebound" and t in other:
            poss, origin, start, ato = t, "ORB", sec, False
        elif pt == "Dead Ball Rebound" and t in other:
            if poss != t:
                poss, origin, start, ato = t, "DEAD", sec, False
        elif pt == "Lost Ball Turnover" and t in other:
            live = (r.period, r.secondsRemaining) in steal_at
            poss, origin, start, ato = other[t], ("STEAL" if live else "DEAD_TO"), sec, False
        elif pt == "Steal" and t in other:
            poss, origin, start, ato = t, "STEAL", sec, False
        elif pt == "MadeFreeThrow" and t in other and "missed" not in str(r.playText).lower():
            poss, origin, start, ato = other[t], "OPP_MAKE", sec, False
        elif pt in TIMEOUTS:
            ato = True
    return res


def main(season):
    if season not in (2024, 2025):
        raise SystemExit("only 2024 and 2025 allowed (2026 sealed)")
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / f"data/raw/cbbd/pbp/{season}").glob("plays_*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    n_raw = len(d)
    d = d.drop_duplicates("id").sort_values(["gameId", "id"]).reset_index(drop=True)
    rep = {"season": season, "files": len(files), "rows_raw": n_raw, "rows_dedup": len(d)}

    gm = pd.read_parquet(ROOT / f"data/raw/cbbd/games_{season}.parquet",
                         columns=["id", "homeTeam", "awayTeam", "neutralSite", "homeConference", "awayConference", "startDate"])
    if not d.gameId.isin(gm.id).all():
        raise SystemExit("games join failed for some pbp gameId")
    gm = gm.set_index("id")
    gm["d1"] = gm.homeConference.notna() & gm.awayConference.notna()

    # score convention check (score AFTER the play)
    d["tot"] = d.homeScore + d.awayScore
    d["prev_tot"] = d.groupby("gameId").tot.shift(1).fillna(0)
    d["next_tot"] = d.groupby("gameId").tot.shift(-1)
    sh = d[d.playType.isin(SHOT_TYPES) & d.shot_made.notna()]
    val = np.where(sh.shot_range == "three_pointer", 3, 2)
    mk = sh.shot_made.astype(bool).values
    rep["score_conv"] = {
        "made_after_eq_value": float(((sh.tot - sh.prev_tot).values[mk] == val[mk]).mean()),
        "missed_after_eq_zero": float(((sh.tot - sh.prev_tot).values[~mk] == 0).mean()),
        "made_nextrow_eq_value": float(((sh.next_tot - sh.tot).values[mk] == val[mk]).mean())}

    ctx = {}
    for _, g in d.groupby("gameId", sort=False):
        ctx.update(possession_context(g))

    fga = d[d.playType.isin(SHOT_TYPES) & d.shot_made.notna()].copy()
    fga["made"] = fga.shot_made.astype(bool)
    fga["range"] = fga.shot_range
    fga["x"] = pd.to_numeric(fga.shot_location_x, errors="coerce").astype(float); fga["y"] = pd.to_numeric(fga.shot_location_y, errors="coerce").astype(float)
    rep["fga_all"] = len(fga)
    rep["located_share_all"] = float(fga.x.notna().mean())
    right = fga.x > 470
    fga["fx"] = np.where(right, 940 - fga.x, fga.x)
    fga["fy"] = np.where(right, 500 - fga.y, fga.y)
    fga["dist_ft"] = np.hypot(fga.fx - HX, fga.fy - HY) / 10.0
    fga["angle_deg"] = np.degrees(np.arctan2(np.abs(fga.fy - HY), fga.fx - HX))
    fga["corner3"] = (fga.range == "three_pointer") & (fga.fx < 140)
    cls = fga.playType.map({"DunkShot": "dunk", "LayUpShot": "layup", "TipShot": "tip"})
    cls = cls.fillna(fga.range.map({"three_pointer": "three", "jumper": "jumper2", "rim": "rim_other"}))
    fga["shot_class"] = cls
    fga["value"] = np.where(fga.range == "three_pointer", 3, 2)
    fga["pts"] = fga.value * fga.made.astype(int)
    c = fga.id.map(ctx)
    fga["origin"] = c.map(lambda z: z[0] if isinstance(z, tuple) else "UNKNOWN")
    fga["poss_sec"] = c.map(lambda z: z[1] if isinstance(z, tuple) else np.nan)
    fga["after_timeout"] = c.map(lambda z: z[2] if isinstance(z, tuple) else False)

    g = gm.loc[fga.gameId]
    if g.neutralSite.isna().any():
        raise SystemExit("neutralSite null")
    fga["neutral"] = g.neutralSite.values.astype(bool)
    is_home = (fga.team.values == g.homeTeam.values)
    is_away = (fga.team.values == g.awayTeam.values)
    if not (is_home | is_away).all():
        raise SystemExit("team not in game")
    fga["home"] = is_home & ~fga.neutral
    fga["d1"] = g.d1.values
    fga["team_conf"] = np.where(is_home, g.homeConference.values, g.awayConference.values)
    fga["opp_team"] = np.where(is_home, g.awayTeam.values, g.homeTeam.values)
    fga["opp_conf"] = np.where(is_home, g.awayConference.values, g.homeConference.values)
    fga["margin"] = np.where(is_home, fga.homeScore - fga.awayScore, fga.awayScore - fga.homeScore) - fga.pts  # score before the shot
    fga["game_sec_left"] = np.where(fga.period <= 2, (2 - fga.period) * 1200 + fga.secondsRemaining, fga.secondsRemaining)
    fga["abs_script_index"] = (fga.margin / np.maximum(fga.game_sec_left / 24.0, 5.0 / 24.0)).abs()
    fga["heave"] = (fga.secondsRemaining <= 3) & (fga.dist_ft >= 35)
    fga["date"] = pd.to_datetime(g.startDate.values, utc=True).tz_convert("US/Eastern").tz_localize(None).normalize()
    fga["tournament"] = fga.tournament.fillna("")
    keep_mask = fga.d1 & (fga.tournament != "CBI")
    rep["fga_d1_noncbi"] = int(keep_mask.sum())
    rep["cbi_fga"] = int((fga.tournament == "CBI").sum())
    f = fga[keep_mask]
    loc = f.x.notna()
    rep["located_share_d1"] = float(loc.mean())
    rep["heaves_located"] = int(f.heave.sum())
    rep["unlocated_last3s_upper_bound_undetected_heaves"] = int((~loc & (f.secondsRemaining <= 3)).sum())
    t = f[loc & (f["range"] == "three_pointer")]
    j = f[loc & (f["range"] == "jumper")]
    rep["fold_sanity"] = {"threes_inside_20.75ft": float((t.dist_ft < 20.75).mean()), "jumpers_beyond_22.15ft": float((j.dist_ft > 22.15).mean())}
    rep["origin_share"] = f.origin.value_counts(normalize=True).round(4).to_dict()
    rep["class_share"] = f.shot_class.value_counts(normalize=True).round(4).to_dict()
    rep["n_games_d1"] = int(f.gameId.nunique())
    keep = ["gameId", "date", "id", "team", "opp_team", "team_conf", "opp_conf", "period", "secondsRemaining", "game_sec_left", "home",
            "neutral", "margin", "shot_class", "range", "value", "made", "pts", "x", "y", "fx", "fy", "dist_ft", "angle_deg",
            "corner3", "origin", "poss_sec", "after_timeout", "abs_script_index", "heave", "tournament"]
    f[keep].to_parquet(OUT / f"shot_context_{season}.parquet", index=False)
    (OUT / f"build_report_{season}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main(int(sys.argv[1]))
