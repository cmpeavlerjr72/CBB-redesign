"""diag_event_layer_v4_verify_v1.py -- verification of event layer v4 against v2.

Lane B, 2026-09-30.  DIAGNOSTIC ONLY; reads possessions_v2 / possessions_v4,
the hoopR box (via own_ratings.load_team_games), the universe, and re-runs the
v4 machine through the unknown-class tap.  Seasons 2022-2025 are analysed;
2026 (sealed) contributes ROW COUNTS ONLY.  Writes
results/g1g5_diag/event_layer_v4_verify.json.

(a) conservation: per team-game FGA, FGM, 3PA, 3PM, FTA, FTM, points,
    technical points, TOV-terminal possessions, OREB (chance continuations),
    DREB-started possessions; v2 vs v4, and against the box where it exists.
(b) possessions per team-game minus the box estimator, game by game; its
    regression on the v2 per-game counts of each former class (v2 and v4 on
    the same regressors).
(c) the `unknown` class table under v4 (same classifier as the diagnostic).
(d) regulation possession duration by start type, clock-complete games.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.pbp import possessions as PZ  # noqa: E402
from cbb_sim.pbp.events import load_plays  # noqa: E402
from cbb_sim.ratings import own_ratings as orat  # noqa: E402

OUT = ROOT / "results/g1g5_diag"


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


UNK = _mod("unk", "diag_g1_unknown_poss_v1.py")
SMP = _mod("smp", "diag_g1_unknown_sample_v1.py")
SEASONS = [2022, 2023, 2024, 2025]
TYPES = ["period_start", "DREB", "TOV", "made_FG", "made_FT", "other"]


def per_tg(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["fga"] = p["fga_rim"] + p["fga_jump2"] + p["fga_3"]
    p["fgm"] = p["fgm_rim"] + p["fgm_jump2"] + p["fgm_3"]
    p["tov"] = (p["terminal_event"] == "TOV").astype(int)
    p["dreb_start"] = (p["start_reason"] == "DREB").astype(int)
    p["tech_pts"] = p["tech_points_off"] + p["tech_points_def"]
    g = p.groupby("game_id")
    cols = {"possessions": g.size() / 2.0}
    for c in ("fga", "fgm", "fga_3", "fgm_3", "fta", "ftm", "points", "tech_pts", "tov", "oreb_count", "dreb_start"):
        cols[c] = g[c].sum() / 2.0
    return pd.DataFrame(cols)


def v4_unknown(season: int) -> pd.DataFrame:
    u = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
    u = u[u["is_d1_game"] & ~u["pbp_truncated"] & (u["season"] == season) & u["cbbd_game_id"].notna()]
    meta = {int(r.cbbd_game_id): {"game_id": int(r.game_id), "cbbd_game_id": int(r.cbbd_game_id),
                                  "season": int(r.season), "home_team_id": int(r.home_team_id),
                                  "away_team_id": int(r.away_team_id)} for r in u.itertuples()}
    plays = load_plays(season, game_ids=set(meta))
    ev = PZ._prepare_events(plays, with_on_floor=False)
    games = ev["game"]
    b = np.flatnonzero(np.concatenate([[True], games[1:] != games[:-1], [True]]))
    rec = []
    for k in range(len(b) - 1):
        lo, hi = int(b[k]), int(b[k + 1])
        gm = meta.get(int(games[lo]))
        if gm is None:
            continue
        sub = {c: ev[c][lo:hi] for c in ("cls", "team", "sec", "period", "hs", "as_", "made", "stolen")}
        sub["on_floor"] = None
        m = UNK.TapMachine(gm, sub, tech_lookahead=True, **PZ.VERSION_EVENT_FIXES["v4"])
        m.run()
        rec += m.rec
    R = pd.DataFrame(rec)
    if len(R):
        R["dur"] = R["start_clock"] - R["end_clock"]
        R["cls"] = SMP.classify(R)
    return R


def main():
    rep = {"seasons": {}}
    univ = orat.load_universe()
    U2 = pd.read_parquet(OUT / "unknown_poss_classified.parquet")
    uv = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
    for s in SEASONS:
        p2 = pd.read_parquet(ROOT / f"data/processed/possessions_v2/possessions_{s}.parquet")
        p4 = pd.read_parquet(ROOT / f"data/processed/possessions_v4/possessions_{s}.parquet")
        p3 = pd.read_parquet(ROOT / f"data/processed/possessions_v3/possessions_{s}.parquet")
        a2, a3, a4 = per_tg(p2), per_tg(p3), per_tg(p4)
        ids = a2.index.intersection(a4.index)
        ids3 = a3.index.intersection(a4.index)
        tg = orat.load_team_games(univ, [s])
        box = tg.groupby("game_id")[["fga", "fgm", "tpa", "tpm", "fta", "ftm", "tov", "oreb", "dreb"]].mean()
        est = tg.drop_duplicates("game_id").set_index("game_id")["game_poss"]
        bid = ids.intersection(box.index)
        cons = {}
        pairs = {"fga": "fga", "fgm": "fgm", "fga_3": "tpa", "fgm_3": "tpm", "fta": "fta", "ftm": "ftm",
                 "tov": "tov", "oreb_count": "oreb", "dreb_start": "dreb", "points": None, "tech_pts": None,
                 "possessions": None}
        for c, bc in pairs.items():
            d = (a4.loc[ids, c] - a2.loc[ids, c])
            row = {"v2": float(a2.loc[ids, c].mean()), "v4": float(a4.loc[ids, c].mean()),
                   "games_changed": int((d != 0).sum()), "max_abs_change": float(d.abs().max())}
            if bc:
                row["box"] = float(box.loc[bid, bc].mean())
                row["mae_vs_box_v2"] = float((a2.loc[bid, c] - box.loc[bid, bc]).abs().mean())
                row["mae_vs_box_v4"] = float((a4.loc[bid, c] - box.loc[bid, bc]).abs().mean())
            # v3 -> v4 isolates the three phantom switches (v3 = v2 + technical lookahead)
            d3 = (a4.loc[ids3, c] - a3.loc[ids3, c])
            row["v3"] = float(a3.loc[ids3, c].mean())
            row["games_changed_v3_to_v4"] = int((d3 != 0).sum())
            row["max_abs_change_v3_to_v4"] = float(d3.abs().max())
            cons[c] = row
        # (b) box gap and slopes
        X = U2[U2["season"] == s].pivot_table(index="game_id", columns="cls", values="period",
                                               aggfunc="size", fill_value=0) / 2.0
        A = pd.read_parquet(OUT / f"andone_{s}.parquet")
        A["restart"] = (~A["ft_made"]) & (A["next_cls"] == "OREB") & (A["next_team"] == A["shooter_side"])
        nt2 = p2.groupby("game_id")["offense_team_id"].nunique()
        gid = [g for g in ids if g in est.index and pd.notna(est[g]) and nt2.get(g, 0) == 2]
        X = X.reindex(gid, fill_value=0)
        X["E_andone_missFT_OREB_restart"] = (A.groupby("game_id")["restart"].sum() / 2.0).reindex(gid, fill_value=0)
        M = np.column_stack([np.ones(len(gid)), X.to_numpy()])
        gap = {}
        for lab, a in (("v2", a2), ("v4", a4)):
            y = (a.loc[gid, "possessions"] - est.loc[gid]).to_numpy()
            bb, *_ = np.linalg.lstsq(M, y, rcond=None)
            r = y - M @ bb
            se = np.sqrt(np.diag(np.linalg.inv(M.T @ M)) * r.var())
            gap[lab] = {"mean_gap": float(y.mean()), "sd_gap": float(y.std()), "intercept": float(bb[0]),
                        "slopes": {c: [float(bb[j + 1]), float(se[j + 1])] for j, c in enumerate(X.columns)}}
        # (c) unknown classes under v4
        R4 = v4_unknown(s)
        cls4 = R4["cls"].value_counts().to_dict() if len(R4) else {}
        cls2 = U2[U2["season"] == s]["cls"].value_counts().to_dict()
        same_team = {}
        for lab, p in (("v2", p2), ("v4", p4)):
            q = p.sort_values(["game_id", "period", "poss_index"])
            prev = q.groupby(["game_id", "period"])["offense_team_id"].shift(1)
            same_team[lab] = int((prev == q["offense_team_id"]).sum())
        # (d) durations by start type, clock-complete regulation
        cc = uv.index[uv["clock_complete_reg"].fillna(False) & (uv["season"] == s)]
        dur = {}
        for lab, p in (("v2", p2), ("v4", p4)):
            q = p[p["game_id"].isin(cc) & (p["period"] <= 2)]
            t = q.groupby("start_reason")["duration_s"].agg(["size", "mean", "median"])
            t["share"] = t["size"] / t["size"].sum()
            t["share_le2s"] = q.groupby("start_reason")["duration_s"].apply(lambda d: float((d <= 2).mean()))
            dur[lab] = t.reindex(TYPES).to_dict("index")
            dur[lab + "_all_mean"] = float(q["duration_s"].mean())
            dur[lab + "_n"] = int(len(q))
            dur[lab + "_per_team_game"] = float(len(q) / 2.0 / max(1, q["game_id"].nunique()))
        rep["seasons"][s] = {"n_games": int(len(ids)), "conservation": cons, "box_gap": gap,
                             "unknown_classes_v2": cls2, "unknown_classes_v4": cls4,
                             "unknown_total": {"v2": int((p2["terminal_event"] == "unknown").sum()),
                                               "v4": int((p4["terminal_event"] == "unknown").sum())},
                             "same_team_consecutive": same_team, "duration_cc": dur}
        print(s, "done", flush=True)
    # 2026: row counts only
    p26 = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2026.parquet")
    c26 = pd.read_parquet(ROOT / "data/processed/possessions_v4/chances_2026.parquet")
    rep["season_2026_row_counts_only"] = {"possessions": int(len(p26)), "chances": int(len(c26))}
    (OUT / "event_layer_v4_verify.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
