"""diag_trajectory_vs_pbp_v1.py -- HOW games get to their score: simulated trajectories vs real hoopR play-by-play.

DIAGNOSTIC ONLY. Reads a sim trajectory parquet (cbb_sim.engine.trajectory; fold-2 / season 2025 replay) and the real
possession table parsed from pbp (`cbb_sim.pbp.possessions`, version v4otc), builds the same per-possession path on both
sides, and compares path statistics. Nothing is fitted, adjusted or fixed here.

    .venv/Scripts/python.exe scripts/diag_trajectory_vs_pbp_v1.py --traj results/trajectories/_runs/replay394_s50 ...
Season 2025 (2024-25) only; 2025-26 is never read.

Definitions (identical on both sides)
  elapsed seconds e0/e1 = start/end of a possession on the regulation clock (OT continues after 2400).
  margin = home - away, updated at the END of each possession (sim: engine score; real: the next possession's
  start_score_diff, i.e. the pbp scoreboard, which also carries technicals).  margin(T) = margin after the last possession
  with e1 <= T (0 before the first).  All path statistics are over REGULATION possessions (period <= 2) except the winner,
  which uses the final score including overtime.
  lead change = sign flip of the nonzero margin; tie = a possession that leaves the margin at 0 after a nonzero margin.
  decided at mark M = the team leading with M seconds left (strictly) never loses the lead again in regulation and wins.
  comeback = trailing by 10+ at the half (1200 s) and winning the game.
  last-2-minutes points = points of possessions that START in the final 120 s of regulation, both teams.
Uncertainty: seed band = 2.5-97.5 percentile, across seeds, of the sim statistic over the common game set; diff CI =
cluster bootstrap over games (300 reps) of real minus sim.  A cell is UNDERPOWERED when its real denominator (games, or
eligible games for a conditional rate) is below --min-n (default 60).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKS = list(range(240, 2401, 240))          # 4-minute marks, elapsed seconds
REG = 2400
NSEG = 10


# ---------------------------------------------------------------------------------------------------------------
# path statistics for ONE game-seed
# ---------------------------------------------------------------------------------------------------------------
def game_stats(per, e0, e1, post, ph, pa):
    """per, e0, e1, post, ph, pa: aligned 1-d arrays of one game-seed in possession order. Returns dict of scalars."""
    out = {}
    reg = per <= 2
    r_e0, r_e1, r_post = e0[reg], e1[reg], post[reg]
    final = int(post[-1]) if len(post) else 0
    # margin at marks
    idx = np.searchsorted(r_e1, MARKS, side="right") - 1
    mk = np.where(idx >= 0, r_post[np.maximum(idx, 0)], 0) if len(r_post) else np.zeros(len(MARKS), int)
    for T, m in zip(MARKS, mk):
        out[f"abs_m{T}"] = abs(int(m))
        out[f"abs_m{T}_sq"] = float(int(m) ** 2)
        out[f"home_m{T}"] = int(m)
    # lead changes / ties / largest lead
    s = np.sign(r_post)
    nz = s[s != 0]
    out["lead_changes"] = int((nz[1:] != nz[:-1]).sum()) if len(nz) > 1 else 0
    prev = np.concatenate([[0], r_post[:-1]]) if len(r_post) else r_post
    out["ties"] = int(((r_post == 0) & (prev != 0)).sum())
    out["largest_lead"] = int(np.abs(r_post).max()) if len(r_post) else 0
    # largest scoring run (unanswered points), regulation
    best = cur = 0
    side = 0
    for h, a in zip(ph[reg], pa[reg]):
        if h > 0 and a == 0:
            cur = cur + h if side == 1 else h
            side = 1
        elif a > 0 and h == 0:
            cur = cur + a if side == -1 else a
            side = -1
        elif h > 0 and a > 0:
            cur, side = 0, 0
        if cur > best:
            best = cur
    out["largest_run"] = int(best)
    # decided by mark (seconds left 480/240/120)
    for left in (480, 240, 120):
        T = REG - left
        i = np.searchsorted(r_e1, T, side="right") - 1
        m = int(r_post[i]) if i >= 0 else 0
        ok = 0
        if m != 0:
            L = 1 if m > 0 else -1
            later = r_post[i + 1:]
            ok = int((np.sign(later) == L).all() and np.sign(final) == L) if True else 0
        out[f"decided_{left}"] = ok
    # comeback
    i = np.searchsorted(r_e1, 1200, side="right") - 1
    mh = int(r_post[i]) if i >= 0 else 0
    down = abs(mh) >= 10
    out["down10_half"] = int(down)
    out["comeback_num"] = int(down and np.sign(final) == -np.sign(mh))
    out["comeback_den"] = int(down)
    out["half_margin"] = mh
    # halves
    p = ph + pa
    out["pts_1h"] = int(p[per == 1].sum())
    out["pts_2h"] = int(p[per == 2].sum())
    out["went_ot"] = int((per >= 3).any())
    out["total_points"] = int(p.sum())
    out["final_abs_margin"] = abs(final)
    # last two minutes (regulation) by margin at 2:00
    i = np.searchsorted(r_e1, REG - 120, side="right") - 1
    m2 = int(r_post[i]) if i >= 0 else 0
    l2 = int(p[reg][r_e0 >= REG - 120].sum())
    close = abs(m2) <= 5
    out["l2m_close_num"], out["l2m_close_den"] = (l2 if close else 0), int(close)
    out["l2m_far_num"], out["l2m_far_den"] = (0 if close else l2), int(not close)
    out["l2m_all"] = l2
    out["close_at_2m"] = int(close)
    # pace by 4-minute segment
    seg = np.minimum((r_e0 // 240).astype(int), NSEG - 1)
    cnt = np.bincount(seg, minlength=NSEG)
    for k in range(NSEG):
        out[f"poss_seg{k}"] = int(cnt[k])
    return out


# ---------------------------------------------------------------------------------------------------------------
# frames -> per game-seed stats
# ---------------------------------------------------------------------------------------------------------------
def elapsed(period, clock):
    period = np.asarray(period).astype(np.int64)
    clock = np.asarray(clock).astype(np.int64)
    return np.where(period <= 2, (period - 1) * 1200 + 1200 - clock, 2400 + (period - 3) * 300 + 300 - clock)


def stats_from_sim(traj: pd.DataFrame) -> pd.DataFrame:
    t = traj.sort_values(["game_id", "seed", "poss_idx"], kind="stable")
    e0 = elapsed(t.period.to_numpy(), t.clock_start.to_numpy())
    e1 = elapsed(t.period.to_numpy(), t.clock_end.to_numpy())
    post = t.margin.to_numpy().astype(int)
    side = t.off_side.to_numpy()
    pts = t.points.to_numpy().astype(int)
    ph, pa = np.where(side == 0, pts, 0), np.where(side == 1, pts, 0)
    per = t.period.to_numpy()
    key = t.game_id.to_numpy() * 1000 + t.seed.to_numpy()
    cuts = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(t)]])
    gid, sd = t.game_id.to_numpy(), t.seed.to_numpy()
    rows = []
    for a, b in zip(starts, ends):
        r = game_stats(per[a:b], e0[a:b], e1[a:b], post[a:b], ph[a:b], pa[a:b])
        r["game_id"], r["seed"] = int(gid[a]), int(sd[a])
        rows.append(r)
    return pd.DataFrame(rows)


def real_paths(game_ids, season: int, poss_version: str = "v4otc") -> pd.DataFrame:
    p = pd.read_parquet(ROOT / f"data/processed/possessions_{poss_version}/possessions_{season}.parquet",
                        columns=["game_id", "period", "poss_index", "offense_is_home", "start_clock", "end_clock",
                                 "start_score_diff", "points"])
    p = p[p.game_id.isin(set(int(g) for g in game_ids))].sort_values(["game_id", "poss_index"], kind="stable")
    sgn = np.where(p.offense_is_home, 1, -1)
    p["pre"] = p.start_score_diff.to_numpy() * sgn
    p["ph"] = np.where(p.offense_is_home, p.points, 0)
    p["pa"] = np.where(~p.offense_is_home, p.points, 0)
    p["post"] = p.groupby("game_id")["pre"].shift(-1)
    p["post"] = p["post"].fillna(p["pre"] + p["ph"] - p["pa"]).astype(int)
    p["e0"] = elapsed(p.period.to_numpy(), p.start_clock.to_numpy())
    p["e1"] = elapsed(p.period.to_numpy(), p.end_clock.to_numpy())
    return p


def stats_from_real(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, x in p.groupby("game_id", sort=True):
        r = game_stats(x.period.to_numpy(), x.e0.to_numpy(), x.e1.to_numpy(), x.post.to_numpy(),
                       x.ph.to_numpy(), x.pa.to_numpy())
        r["game_id"], r["seed"] = int(gid), 0
        r["final_post"] = int(x.post.iloc[-1])
        r["n_poss"] = len(x)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# metric table: (label, numerator column(s), denominator column or None, combiner)
# ---------------------------------------------------------------------------------------------------------------
def build_metrics():
    M = []   # (key, label, num, den, kind)   kind: 'mean' | 'rate' (conditional) | derived handled separately
    for T in MARKS:
        m = T // 60
        M.append((f"abs_m{T}", f"|margin| at {m} min (mean)", f"abs_m{T}", None, "mean"))
    for T in MARKS:
        M.append((f"home_m{T}", f"home margin at {T // 60} min (mean)", f"home_m{T}", None, "mean"))
    M += [
        ("lead_changes", "lead changes / game", "lead_changes", None, "mean"),
        ("ties", "ties / game", "ties", None, "mean"),
        ("largest_lead", "largest lead (either team)", "largest_lead", None, "mean"),
        ("largest_run", "largest scoring run (pts)", "largest_run", None, "mean"),
        ("decided_480", "decided by 8:00 left (share)", "decided_480", None, "mean"),
        ("decided_240", "decided by 4:00 left (share)", "decided_240", None, "mean"),
        ("decided_120", "decided by 2:00 left (share)", "decided_120", None, "mean"),
        ("down10_half", "down 10+ at half (share of games)", "down10_half", None, "mean"),
        ("comeback", "comeback rate | down 10+ at half", "comeback_num", "comeback_den", "rate"),
        ("pts_1h", "1H points / game (both teams)", "pts_1h", None, "mean"),
        ("pts_2h", "2H points / game (both teams)", "pts_2h", None, "mean"),
        ("went_ot", "went to OT (share)", "went_ot", None, "mean"),
        ("total_points", "total points / game", "total_points", None, "mean"),
        ("l2m_close", "last-2-min points | |margin|<=5 at 2:00", "l2m_close_num", "l2m_close_den", "rate"),
        ("l2m_far", "last-2-min points | |margin|>5 at 2:00", "l2m_far_num", "l2m_far_den", "rate"),
        ("close_at_2m", "|margin|<=5 at 2:00 (share)", "close_at_2m", None, "mean"),
    ]
    for k in range(NSEG):
        M.append((f"poss_seg{k}", f"possessions, min {4 * k}-{4 * k + 4} (both teams)", f"poss_seg{k}", None, "mean"))
    return M


def _ratio(num, den):
    return num / np.where(den == 0, np.nan, den)


class Cube:
    """Per-game sums of a statistic split by bucket: arrays (G, B) for real and sim, plus (S, G, B) for the seed band."""

    def __init__(self, G, B, S):
        self.rn = np.zeros((G, B)); self.rd = np.zeros((G, B))
        self.sn = np.zeros((G, B)); self.sd = np.zeros((G, B))
        self.sn_s = np.zeros((S, G, B)); self.sd_s = np.zeros((S, G, B))


def accumulate(real: pd.DataFrame, sim: pd.DataFrame, gids, seeds, bucket_real, bucket_sim, B, num, den):
    """bucket_*: int array per row (-1 = excluded)."""
    G, S = len(gids), len(seeds)
    gi_r = real["_gi"].to_numpy(); gi_s = sim["_gi"].to_numpy(); si_s = sim["_si"].to_numpy()
    c = Cube(G, B, S)
    nr = real[num].to_numpy(float); dr = np.ones(len(real)) if den is None else real[den].to_numpy(float)
    ns = sim[num].to_numpy(float); ds = np.ones(len(sim)) if den is None else sim[den].to_numpy(float)
    ok = bucket_real >= 0
    np.add.at(c.rn, (gi_r[ok], bucket_real[ok]), nr[ok]); np.add.at(c.rd, (gi_r[ok], bucket_real[ok]), dr[ok])
    ok = bucket_sim >= 0
    np.add.at(c.sn, (gi_s[ok], bucket_sim[ok]), ns[ok]); np.add.at(c.sd, (gi_s[ok], bucket_sim[ok]), ds[ok])
    np.add.at(c.sn_s, (si_s[ok], gi_s[ok], bucket_sim[ok]), ns[ok])
    np.add.at(c.sd_s, (si_s[ok], gi_s[ok], bucket_sim[ok]), ds[ok])
    return c


def summarize(c: Cube, b: int, rng, nboot=300, min_n=60):
    rn, rd, sn, sd = c.rn[:, b], c.rd[:, b], c.sn[:, b], c.sd[:, b]
    real = rn.sum() / rd.sum() if rd.sum() else np.nan
    sim = sn.sum() / sd.sum() if sd.sum() else np.nan
    G = len(rn)
    diffs = []
    for _ in range(nboot):
        ix = rng.integers(0, G, G)
        a, bb = rd[ix].sum(), sd[ix].sum()
        if a and bb:
            diffs.append(rn[ix].sum() / a - sn[ix].sum() / bb)
    lo, hi = (np.nanpercentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    ss = c.sn_s[:, :, b].sum(1) / np.where(c.sd_s[:, :, b].sum(1) == 0, np.nan, c.sd_s[:, :, b].sum(1))
    band = np.nanpercentile(ss, [2.5, 97.5]) if np.isfinite(ss).any() else (np.nan, np.nan)
    n_real = int(rd.sum())                 # eligible real units (games, or games meeting a conditional)
    n_games = int((rd > 0).sum())
    return dict(real=real, sim=sim, diff=real - sim, lo=lo, hi=hi, band_lo=band[0], band_hi=band[1], n=n_real,
                n_games=n_games, underpowered=bool(n_real < min_n), sig=bool(lo > 0 or hi < 0))


def derived_sd(cubeA: Cube, cubeB: Cube, b: int):
    """SD of |margin| from mean and mean-square cubes (point estimate real / sim / seed band only)."""
    def sd(m1n, m1d, m2n, m2d):
        m1 = m1n.sum() / m1d.sum(); m2 = m2n.sum() / m2d.sum()
        return float(np.sqrt(max(m2 - m1 ** 2, 0)))
    real = sd(cubeA.rn[:, b], cubeA.rd[:, b], cubeB.rn[:, b], cubeB.rd[:, b])
    sim = sd(cubeA.sn[:, b], cubeA.sd[:, b], cubeB.sn[:, b], cubeB.sd[:, b])
    S = cubeA.sn_s.shape[0]
    ss = []
    for s in range(S):
        m1 = cubeA.sn_s[s, :, b].sum() / cubeA.sd_s[s, :, b].sum(); m2 = cubeB.sn_s[s, :, b].sum() / cubeB.sd_s[s, :, b].sum()
        ss.append(np.sqrt(max(m2 - m1 ** 2, 0)))
    return real, sim, float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5))


# ---------------------------------------------------------------------------------------------------------------
def pregame_tables(game_ids, season):
    """spread (consensus close, home) and home-team own-rating net (as of the day before)."""
    g = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")
    g = g[g.game_id.isin(game_ids)][["game_id", "game_date", "home_team_id", "away_team_id"]]
    ln = pd.read_parquet(ROOT / "data/processed/lines/lines_close_v2_verified.parquet")
    ln = ln[ln.game_id.isin(game_ids) & ln.close_spread_home.notna()][["game_id", "close_spread_home"]]
    g = g.merge(ln.drop_duplicates("game_id"), on="game_id", how="left")
    r = pd.read_parquet(ROOT / f"data/processed/ratings/own_ratings_{season}.parquet",
                        columns=["as_of_date", "team_id", "off_c", "def_c"])
    r["as_of_date"] = pd.to_datetime(r.as_of_date).astype("datetime64[ns]")
    r = r.sort_values("as_of_date")
    out = []
    for side in ("home", "away"):
        left = g[["game_id", "game_date", f"{side}_team_id"]].rename(columns={f"{side}_team_id": "team_id"}).copy()
        left["key"] = (pd.to_datetime(left.game_date) - pd.Timedelta(days=1)).astype("datetime64[ns]")
        m = pd.merge_asof(left.sort_values("key"), r.rename(columns={"as_of_date": "key"}), on="key", by="team_id",
                          direction="backward")
        out.append(m.set_index("game_id")[["off_c", "def_c"]].add_prefix(f"{side}_"))
    g = g.set_index("game_id").join(out[0]).join(out[1]).reset_index()
    best = None
    for s in (1, -1):
        net_h = g.home_off_c + s * g.home_def_c
        net_a = g.away_off_c + s * g.away_def_c
        c = np.corrcoef((net_h - net_a)[g.close_spread_home.notna() & net_h.notna() & net_a.notna()],
                        (-g.close_spread_home)[g.close_spread_home.notna() & net_h.notna() & net_a.notna()])[0, 1]
        if best is None or c > best[0]:
            best = (c, s)
    g["home_net"] = g.home_off_c + best[1] * g.home_def_c
    g["away_net"] = g.away_off_c + best[1] * g.away_def_c
    g["_rating_sign"], g["_rating_corr"] = best[1], best[0]
    return g


def quint(x):
    x = pd.Series(x)
    q = pd.qcut(x.rank(method="first"), 5, labels=False)
    return q.fillna(-1).astype(int).to_numpy()


def fmt(v, nd=2):
    return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def row_md(label, s, nd=2, extra=""):
    flag = "UNDERPOWERED" if s["underpowered"] else ("diff CI excludes 0" if s["sig"] else "")
    label = label.replace("|", "\\|")
    return (f"| {label} | {s['n']:,} | {fmt(s['real'], nd)} | {fmt(s['sim'], nd)} | {fmt(s['diff'], nd)} "
            f"[{fmt(s['lo'], nd)}, {fmt(s['hi'], nd)}] | [{fmt(s['band_lo'], nd)}, {fmt(s['band_hi'], nd)}] | {flag}{extra} |")


HDR = ("| cell | n (real) | real | sim | real - sim [95% CI] | sim seed band | flag |\n"
       "|---|---:|---:|---:|---|---|---|")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="dir with trajectory.parquet (a run under results/trajectories/)")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--min-n", type=int, default=60)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--out", default="docs/tests/trajectory_vs_pbp_2026-10-09.md")
    ap.add_argument("--json", default=None)
    ap.add_argument("--read-file", default=None, help="markdown with the read and the candidate list, split on a line '====='")
    a = ap.parse_args()
    if a.season >= 2026:
        raise SystemExit("season 2026 is SEALED")
    rng = np.random.default_rng(0)
    traj = pd.read_parquet(Path(a.traj) / "trajectory.parquet")
    sim = stats_from_sim(traj)
    n_seeds = sim.seed.nunique()
    gids_all = np.sort(sim.game_id.unique())
    rp = real_paths(gids_all, a.season)
    real = stats_from_real(rp)
    fin = pd.read_parquet(ROOT / "data/processed/truth/game_finals_v2.parquet")[["game_id", "home_score", "away_score"]]
    real = real.merge(fin, on="game_id", how="left")
    real["verified_margin"] = real.home_score - real.away_score
    n0 = len(real)
    bad = real[(real.verified_margin.isna()) | (real.final_post != real.verified_margin) | (real.n_poss < 90)]
    real = real[~real.game_id.isin(bad.game_id)]
    gids = np.sort(real.game_id.unique())
    sim = sim[sim.game_id.isin(gids)]
    # a game must have every seed to count (paired)
    gids = np.sort(np.intersect1d(gids, sim.game_id.unique()))
    real = real[real.game_id.isin(gids)].reset_index(drop=True)
    sim = sim[sim.game_id.isin(gids)].reset_index(drop=True)
    seeds = np.sort(sim.seed.unique())
    gi = {int(g): i for i, g in enumerate(gids)}
    real["_gi"] = real.game_id.map(gi); sim["_gi"] = sim.game_id.map(gi)
    sim["_si"] = sim.seed.map({int(s): i for i, s in enumerate(seeds)})
    G, S = len(gids), len(seeds)

    pg = pregame_tables(set(int(g) for g in gids), a.season).set_index("game_id").reindex(gids)
    home_q = np.where(pg.home_net.notna(), quint(pg.home_net.fillna(pg.home_net.median())), -1)
    spr = pg.close_spread_home.abs()
    spr_q = np.where(spr.notna(), quint(spr.fillna(spr.median())), -1)
    sgn_spr = pg.close_spread_home

    def b_pre(arr):
        return arr[real._gi.to_numpy()], arr[sim._gi.to_numpy()]
    hb_r, hb_s = b_pre(home_q)
    sb_r, sb_s = b_pre(spr_q)
    # margin-at-half buckets (game-seed own state): |margin| 0-4, 5-9, 10-14, 15+
    half_edges = [0, 5, 10, 15, 999]
    def hbucket(df):
        return np.clip(np.digitize(np.abs(df.half_margin.to_numpy()), half_edges[1:-1]), 0, 3)
    mh_r, mh_s = hbucket(real), hbucket(sim)
    zero_r, zero_s = np.zeros(len(real), int), np.zeros(len(sim), int)

    M = build_metrics()
    cubes = {}
    for key, label, num, den, kind in M:
        cubes[key] = {
            "all": accumulate(real, sim, gids, seeds, zero_r, zero_s, 1, num, den),
            "home_q": accumulate(real, sim, gids, seeds, hb_r, hb_s, 5, num, den),
            "spread_q": accumulate(real, sim, gids, seeds, sb_r, sb_s, 5, num, den),
            "half": accumulate(real, sim, gids, seeds, mh_r, mh_s, 4, num, den),
        }
    cubes["abs_sq"] = {T: accumulate(real, sim, gids, seeds, zero_r, zero_s, 1, f"abs_m{T}_sq", None) for T in MARKS}

    lines = []
    W = lines.append
    W("# Trajectory vs real play-by-play: how simulated games unfold, 2026-10-09\n")
    W("DIAGNOSTIC ONLY (post-freeze round candidates are listed at the end). Generated by "
      "`scripts/diag_trajectory_vs_pbp_v1.py`; sim trajectories from `scripts/run_engine_window_v2.py` "
      f"(`{a.traj}`), served stack v3, `engine_v3` inputs, fold 2 / season 2025 (2024-25). 2025-26 is not read.\n")
    W("## 0. Frame\n")
    W(f"- Replay window: games on 2025-02-11, 02-15, 02-25, 03-01, 03-04 (the three serving-replay dates of "
      f"`run_daily_replay_f2_v1.sh` plus the two Saturdays 02-15 and 03-01 to reach 300+ games). Candidate games: {n0}. "
      f"Kept: **{G} games** x **{S} seeds** = {G * S:,} simulated game-seeds ({len(traj):,} sim possessions).")
    W(f"- Dropped {n0 - G} games: real pbp path final margin != verified final (`game_finals_v2`), or fewer than 90 parsed "
      "possessions, or missing sim. Real side: `possessions_v4otc` (parsed from hoopR/CBBD pbp), margin from the pbp scoreboard.")
    W(f"- Pre-game spread: close (ESPN BET, the only 2025 provider) from `lines_close_v2_verified`; rating prior: own as-of ratings the day before "
      f"(net = off_c {'+' if pg._rating_sign.iloc[0] > 0 else '-'} def_c, orientation chosen by correlation with the close "
      f"spread, r = {pg._rating_corr.iloc[0]:.2f}).")
    W(f"- Seed band: 2.5-97.5 percentile over {S} seeds of the sim statistic on these {G} games. Diff CI: cluster bootstrap "
      f"over games ({a.nboot} reps). Underpowered = real n < {a.min_n}. 'diff CI excludes 0' is a flag, not a verdict: about "
      "1 in 20 cells will do so by chance, and the sim side has no real-game sampling noise because it is seed-averaged.\n")

    W("## 1. Overall\n")
    W(HDR)
    keys_overall = [m for m in M]
    for key, label, num, den, kind in keys_overall:
        nd = 3 if key.startswith(("decided", "down10", "comeback", "went_ot", "close_at")) else 2
        W(row_md(label, summarize(cubes[key]["all"], 0, rng, a.nboot, a.min_n), nd))
    W("\nSD of |margin| by 4-minute mark (point estimates; seed band from per-seed SD):\n")
    W("| mark (min) | real SD | sim SD | sim seed band |\n|---:|---:|---:|---|")
    for T in MARKS:
        r_, s_, lo_, hi_ = derived_sd(cubes[f"abs_m{T}"]["all"], cubes["abs_sq"][T], 0)
        W(f"| {T // 60} | {r_:.2f} | {s_:.2f} | [{lo_:.2f}, {hi_:.2f}] |")
    h1 = summarize(cubes["pts_1h"]["all"], 0, rng, a.nboot, a.min_n)
    h2 = summarize(cubes["pts_2h"]["all"], 0, rng, a.nboot, a.min_n)
    W(f"\nSecond half minus first half points (both teams, per game): real {h2['real'] - h1['real']:.2f}, "
      f"sim {h2['sim'] - h1['sim']:.2f}, difference {(h2['real'] - h1['real']) - (h2['sim'] - h1['sim']):.2f}.\n")

    key_slice = [("lead_changes", "lead changes", 2), ("largest_lead", "largest lead", 2), ("abs_m1200", "|margin| at half", 2),
                 ("abs_m2160", "|margin| at 4:00 left", 2), ("decided_240", "decided by 4:00 left", 3),
                 ("comeback", "comeback | down 10+ at half", 3), ("l2m_close", "last-2-min pts | close at 2:00", 2),
                 ("total_points", "total points", 2)]
    W("## 2. By quintile of the home team's prior rating (Q1 = weakest home team)\n")
    for key, nm, nd in key_slice:
        W(f"**{nm}**\n\n{HDR}")
        for q in range(5):
            W(row_md(f"home-rating Q{q + 1}", summarize(cubes[key]["home_q"], q, rng, a.nboot, a.min_n), nd))
        W("")
    W("## 3. By margin at the half (own state in each frame: |margin| 0-4, 5-9, 10-14, 15+)\n")
    W("Unpaired: each frame is bucketed by its own half margin, so a difference here mixes path shape with how often a "
      "game reaches that state; read together with section 1 (`|margin| at 20 min`).\n")
    names = ["|m|<5", "5-9", "10-14", "15+"]
    for key, nm, nd in [("lead_changes", "lead changes", 2), ("abs_m2160", "|margin| at 4:00 left", 2),
                        ("decided_240", "decided by 4:00 left", 3), ("l2m_close", "last-2-min pts | close at 2:00", 2),
                        ("pts_2h", "2H points / game", 2)]:
        W(f"**{nm}**\n\n{HDR}")
        for q in range(4):
            W(row_md(f"half {names[q]}", summarize(cubes[key]["half"], q, rng, a.nboot, a.min_n), nd))
        W("")

    W("## 4. Responsiveness: lead changes and largest lead by REAL pre-game |spread| quintile (Q1 = closest games)\n")
    W("The sim's counts should slope with the real ones across quintiles (competitive games change leads more).\n")
    resp = {}
    for key, nm in (("lead_changes", "lead changes"), ("largest_lead", "largest lead"), ("total_points", "total points")):
        W(f"**{nm}**\n\n{HDR}")
        rs, ss_ = [], []
        for q in range(5):
            s = summarize(cubes[key]["spread_q"], q, rng, a.nboot, a.min_n)
            rs.append(s["real"]); ss_.append(s["sim"])
            W(row_md(f"|spread| Q{q + 1} (mean {spr[spr_q == q].mean():.1f})", s, 2))
        x = np.arange(5)
        resp[key] = (float(np.polyfit(x, rs, 1)[0]), float(np.polyfit(x, ss_, 1)[0]))
        W(f"\nSlope across quintiles (per quintile step): real {resp[key][0]:.3f}, sim {resp[key][1]:.3f}.\n")

    # continuous slope with bootstrap: lead changes on |spread|
    ok_g = spr.notna().to_numpy()
    xs = spr.to_numpy()
    lc_r = real.set_index("_gi").lead_changes.reindex(range(G)).to_numpy(float)
    lc_s = sim.groupby("_gi").lead_changes.mean().reindex(range(G)).to_numpy(float)
    sl = []
    for _ in range(a.nboot):
        ix = rng.integers(0, G, G); ix = ix[ok_g[ix]]
        sl.append((np.polyfit(xs[ix], lc_r[ix], 1)[0], np.polyfit(xs[ix], lc_s[ix], 1)[0]))
    sl = np.array(sl)
    W(f"Continuous slope of lead changes on |spread| (per point of spread): real {np.polyfit(xs[ok_g], lc_r[ok_g], 1)[0]:.4f}, "
      f"sim {np.polyfit(xs[ok_g], lc_s[ok_g], 1)[0]:.4f}; bootstrap SD real {sl[:, 0].std():.4f}, sim {sl[:, 1].std():.4f}; "
      f"difference CI [{np.percentile(sl[:, 0] - sl[:, 1], 2.5):.4f}, {np.percentile(sl[:, 0] - sl[:, 1], 97.5):.4f}].\n")

    W("## 5. Plain-language read\n")
    W("READ_PLACEHOLDER")
    W("\n## 6. Post-freeze round candidates (not fixed here)\n")
    W("CANDIDATES_PLACEHOLDER")
    text = "\n".join(lines) + "\n"
    if a.read_file and Path(a.read_file).exists():
        rd, cand = Path(a.read_file).read_text(encoding="utf-8").split("\n=====\n")
        text = text.replace("READ_PLACEHOLDER", rd.strip()).replace("CANDIDATES_PLACEHOLDER", cand.strip())
    Path(a.out).write_text(text, encoding="utf-8")
    if a.json:
        flat = {}
        for key, label, num, den, kind in M:
            flat[key] = {b: summarize(cubes[key]["all"], 0, rng, a.nboot, a.min_n)}
        Path(a.json).write_text(json.dumps({k: {str(b): v for b, v in d.items()} for k, d in flat.items()}, default=float, indent=1))
    print(f"wrote {a.out}: {G} games x {S} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
