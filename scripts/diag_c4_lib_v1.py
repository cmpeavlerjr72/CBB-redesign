"""diag_c4_lib_v1.py -- shared extraction for the four 2026-10-09 diagnostics (D1-D4). DIAGNOSTIC ONLY.

Sim side: per-possession trajectory parquet (cbb_sim.engine.trajectory, served stack v3, fold 2 / season 2025).
Real side: hoopR/CBBD possession table `possessions_v4otc` (season 2025 only; 2025-26 is never read).
Both are converted to ONE standardised possession frame and every aggregate below is computed by the same code.

Standard columns: game_id, seed, k (possession order), period, cs, ce (clock start/end, s left in period), dur,
off_side (0 home offence, 1 away), fga2, fgm2, fga3, fgm3, fta, ftm, tov, oreb, pts (offence points), post (home margin
after the possession), pre (home margin before), ofl, dfl (offence / defence team fouls at open), bonus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
OUT = ROOT / "results" / "diag_c4"
SEASON = 2025


def elapsed(period, clock):
    period = np.asarray(period).astype(np.int64)
    clock = np.asarray(clock).astype(np.int64)
    return np.where(period <= 2, (period - 1) * 1200 + 1200 - clock, 2400 + (period - 3) * 300 + 300 - clock)


def std_from_sim(t: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values(["game_id", "seed", "poss_idx"], kind="stable").reset_index(drop=True)
    d = pd.DataFrame({
        "game_id": t.game_id.to_numpy(), "seed": t.seed.to_numpy().astype(np.int32), "k": t.poss_idx.to_numpy(),
        "period": t.period.to_numpy(), "cs": t.clock_start.to_numpy(), "ce": t.clock_end.to_numpy(),
        "dur": t.duration_s.to_numpy(), "off_side": t.off_side.to_numpy(),
        "fga2": t.fga2.to_numpy(), "fgm2": t.fgm2.to_numpy(), "fga3": t.fga3.to_numpy(), "fgm3": t.fgm3.to_numpy(),
        "fta": t.fta.to_numpy(), "ftm": t.ftm.to_numpy(), "tov": t.tov.to_numpy(), "oreb": t.n_oreb.to_numpy(),
        "pts": t.points.to_numpy(), "post": t.margin.to_numpy().astype(np.int16),
        "ofl": t.off_team_fouls.to_numpy(), "dfl": t.def_team_fouls.to_numpy(), "bonus": t.off_in_bonus.to_numpy()})
    key = d.game_id.to_numpy().astype(np.int64) * 1000 + d.seed.to_numpy()
    first = np.r_[True, key[1:] != key[:-1]]
    pre = np.r_[0, d.post.to_numpy()[:-1]]
    d["pre"] = np.where(first, 0, pre).astype(np.int16)
    return d


def std_from_real(game_ids) -> pd.DataFrame:
    import diag_trajectory_vs_pbp_v1 as T
    p = T.real_paths(game_ids, SEASON)
    cols = ["game_id", "poss_index", "terminal_event", "fga_rim", "fgm_rim", "fga_jump2", "fgm_jump2", "fga_3", "fgm_3",
            "fta", "ftm", "oreb_count", "off_team_fouls", "def_team_fouls", "off_in_bonus"]
    full = pd.read_parquet(ROOT / f"data/processed/possessions_v4otc/possessions_{SEASON}.parquet", columns=cols)
    p = p[["game_id", "poss_index", "period", "start_clock", "end_clock", 
           "offense_is_home", "points", "post", "pre"]].merge(full, on=["game_id", "poss_index"], how="left")
    p["duration_s"] = p["start_clock"] - p["end_clock"]
    d = pd.DataFrame({
        "game_id": p.game_id.to_numpy(), "seed": np.zeros(len(p), np.int32), "k": p.poss_index.to_numpy(),
        "period": p.period.to_numpy(), "cs": p.start_clock.to_numpy(), "ce": p.end_clock.to_numpy(),
        "dur": p.duration_s.to_numpy(), "off_side": np.where(p.offense_is_home, 0, 1),
        "fga2": (p.fga_rim + p.fga_jump2).to_numpy(), "fgm2": (p.fgm_rim + p.fgm_jump2).to_numpy(),
        "fga3": p.fga_3.to_numpy(), "fgm3": p.fgm_3.to_numpy(), "fta": p.fta.to_numpy(), "ftm": p.ftm.to_numpy(),
        "tov": (p.terminal_event == "TOV").astype(int).to_numpy(), "oreb": p.oreb_count.to_numpy(),
        "pts": p.points.to_numpy(), "post": p.post.to_numpy().astype(np.int16), "pre": p.pre.to_numpy().astype(np.int16),
        "ofl": p.off_team_fouls.to_numpy(), "dfl": p.def_team_fouls.to_numpy(),
        "bonus": p.off_in_bonus.astype(int).to_numpy(), "terminal": p.terminal_event.to_numpy()})
    return d.sort_values(["game_id", "k"], kind="stable").reset_index(drop=True)


def poss_class(d: pd.DataFrame) -> np.ndarray:
    """identical rule both sides: TOV > FT trip (fta>0) > 3PA > 2PA > NONE."""
    return np.select([d.tov > 0, d.fta > 0, d.fga3 > 0, d.fga2 > 0], ["TOV", "FT", "3PA", "2PA"], "NONE")


def team_game(d: pd.DataFrame) -> pd.DataFrame:
    """regulation (period <= 2) channel sums per (game, seed, offence side)."""
    r = d[d.period <= 2]
    g = r.groupby(["game_id", "seed", "off_side"], sort=False)
    out = g[["fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "tov", "oreb", "pts", "dur"]].sum()
    out["poss"] = g.size()
    r = r.assign(h1=(r.period == 1).astype(int), h2=(r.period == 2).astype(int))
    out["poss_h1"] = r.groupby(["game_id", "seed", "off_side"], sort=False).h1.sum()
    out["poss_h2"] = r.groupby(["game_id", "seed", "off_side"], sort=False).h2.sum()
    out["dur_h1"] = r.assign(d1=r.dur * r.h1).groupby(["game_id", "seed", "off_side"], sort=False).d1.sum()
    out["dur_h2"] = r.assign(d2=r.dur * r.h2).groupby(["game_id", "seed", "off_side"], sort=False).d2.sum()
    return out.reset_index()


def game_seed(d: pd.DataFrame) -> pd.DataFrame:
    """per (game, seed): lead changes, ties, time of decision, reg-end margin, OT flag, totals."""
    d = d.reset_index(drop=True)
    key = d.game_id.to_numpy().astype(np.int64) * 1000 + d.seed.to_numpy().astype(np.int64)
    reg = d.period.to_numpy() <= 2
    e1 = elapsed(d.period, d.ce)
    post = d.post.to_numpy().astype(int)
    s = np.sign(post)
    sr = pd.Series(np.where(reg, s, 0)).replace(0, np.nan)
    ffv = sr.groupby(key).ffill().to_numpy()
    prev = pd.Series(ffv).groupby(key).shift(1).to_numpy()
    flip = (~np.isnan(ffv)) & (~np.isnan(prev)) & (ffv != prev) & reg
    firstrow = np.r_[True, key[1:] != key[:-1]]
    prevpost = np.where(firstrow, 0, np.r_[0, post[:-1]])
    tie = reg & (post == 0) & (prevpost != 0)
    df = pd.DataFrame({"key": key, "reg": reg, "post": post, "flip": flip, "e1": e1, "pts": d.pts.to_numpy(),
                       "period": d.period.to_numpy(), "game_id": d.game_id.to_numpy(), "seed": d.seed.to_numpy(),
                       "tie": tie})
    g = df.groupby("key", sort=False)
    out = pd.DataFrame({"game_id": g.game_id.first(), "seed": g.seed.first(), "final_margin": g.post.last(),
                        "max_period": g.period.max(), "lead_changes": g.flip.sum(), "ties": g.tie.sum()})
    reg_last = df[df.reg].groupby("key", sort=False).post.last()
    out["reg_margin"] = reg_last.reindex(out.index).fillna(0).astype(int)
    out["ot"] = (out.max_period >= 3).astype(int)
    out["total_pts"] = g.pts.sum()
    w = np.sign(out.reg_margin).reindex(df.key).to_numpy()
    notahead = df.reg.to_numpy() & (np.sign(df.post.to_numpy()) * w <= 0)
    t = pd.Series(np.where(notahead, df.e1.to_numpy(), 0)).groupby(key).max()
    out["t_decision"] = np.where(out.reg_margin == 0, 2400, t.reindex(out.index).to_numpy())
    out["largest_lead"] = (df[df.reg].assign(a=lambda x: x.post.abs()).groupby("key", sort=False).a.max()
                           .reindex(out.index).fillna(0))
    return out.reset_index(drop=True)


def late_frame(d: pd.DataFrame, within=240) -> pd.DataFrame:
    """regulation possessions that START in the last `within` seconds of a half, with margin state and fouls delta."""
    d = d.reset_index(drop=True)
    key = (d.game_id.to_numpy().astype(np.int64) * 1000 + d.seed.to_numpy().astype(np.int64)) * 10 + d.period.to_numpy()
    nxt_ofl = pd.Series(d.ofl.to_numpy()).groupby(key).shift(-1).to_numpy()
    nxt_off = pd.Series(d.off_side.to_numpy()).groupby(key).shift(-1).to_numpy()
    dfoul = nxt_ofl - d.dfl.to_numpy()
    dfoul = np.where(nxt_off == d.off_side.to_numpy(), np.nan, dfoul)
    d = d.assign(dfoul=dfoul)
    x = d[(d.period <= 2) & (d.cs <= within)].copy()
    sg = np.where(x.off_side == 0, 1, -1)
    x["moff"] = x.pre.to_numpy() * sg
    x["cls"] = poss_class(x)
    return x.drop(columns=[c for c in ("terminal",) if c in x.columns])
