"""
export_live_grid_v1.py -- per-game `live.json` (game level) for the CBB scoreboard's live-odds layer.

Reads a run's `trajectory.parquet` (one row per possession per (game, seed)) plus the run's games.parquet, bins the stored ensemble
(arm chosen by the live_grid bake-off, docs/models/live_grid/experiments.md) and writes the CFB-contract file
`days/<day>/games/<slug>/live.json` plus the basketball additions the site client reads (monte_site commit 6d2de56, `atStateBb`):
`clock`, `remaining.total_edges`, total-keyed `remaining.total_sf`, `remaining.late`, `remaining.end`.

No fit, no adjustment: every number is a count over the run's own seeds. Contract + sizes: docs/ops/live_grid_exporter_2026-11-02.md.
Library: src/cbb_sim/live_grid.py. Standalone use:

    python scripts/export_live_grid_v1.py --traj results/trajectories/daily_2026-11-02_s2500_o0/trajectory.parquet \
        --games results/daily/sim/2026-11-02/s2500_o0/games.parquet --slate results/daily/sim/2026-11-02/s2500_o0/slate.parquet \
        --out-dir /tmp/live_out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim import live_grid as LG  # noqa: E402

DEFAULT_ARM = "D_row"          # B0 winner at 100 seeds, confirmation, and the 2,000-seed scale read (docs/models/live_grid/experiments.md s9)
CONVENTION = "teamA = home, teamB = away, margin = home minus away; spread is in HOME perspective (negative = home favored)"
CLOCK = {"regulation_s": LG.REG_S, "period_s": LG.PERIOD_S, "n_periods": 2, "ot_period_s": 300}
GRID_TB, GRID_N_TIME, GRID_MB, GRID_MIN, GRID_MAX, GRID_N_MARGIN = 120, 20, 2, -28, 28, 29
MARGIN_RULE = ("half-away-from-zero on margin/2: bucket 14 == tied, bucket 14+k (k>0) == margins {2k-1, 2k}, bucket 14-k == "
               "margins {-2k, -2k+1}; margins beyond +/-28 are clipped into the ends")
TOP_KEYS = ("convention", "game_id", "slug", "teamA", "teamB", "n_sims", "updated", "source_tag", "orientation", "grid", "remaining",
            "uncond", "clock")


def grid_bucket(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m)
    k = np.sign(m) * ((np.abs(m) + 1) // 2)
    return np.clip(14 + k, 0, GRID_N_MARGIN - 1).astype(np.int64)


def curve(pmf: np.ndarray, offset: int) -> dict | None:
    """{k0, v}: v[j] = per-mille P(X >= k0 + j); k0 = smallest value with mass; 1000 below k0, 0 past the end."""
    nz = np.nonzero(pmf > 0)[0]
    if len(nz) == 0:
        return None
    lo, hi = int(nz[0]), int(nz[-1])
    sf = np.cumsum(pmf[::-1])[::-1]                       # P(X >= idx)
    v = np.rint(1000.0 * sf[lo:hi + 1]).astype(int).tolist()
    v.append(0)
    v[0] = 1000
    return {"k0": lo - offset, "v": v}


def _mcurve(pmf):
    return curve(pmf, LG.RM_OFF)


def _tcurve(pmf):
    return curve(pmf, 0)


def build_grid(gs: LG.GameSeeds, min_n: int) -> dict:
    """CFB `grid` block: for each seed and (2-min bin, 2-pt margin bucket) cell it ever entered, mean over the seeds' FINALS."""
    S, P = gs.E.shape
    valid = np.isfinite(gs.E)
    tb = np.minimum((np.where(valid, gs.E, 0) // GRID_TB).astype(np.int64), GRID_N_TIME - 1)
    mb = grid_bucket(gs.Mg)
    seed_i = np.repeat(np.arange(S)[:, None], P, axis=1)
    cell = tb * GRID_N_MARGIN + mb
    key = (seed_i * GRID_N_TIME * GRID_N_MARGIN + cell)[valid]
    key = np.unique(key)                                  # each seed counts at most once per cell
    sidx, cidx = key // (GRID_N_TIME * GRID_N_MARGIN), key % (GRID_N_TIME * GRID_N_MARGIN)
    nc = GRID_N_TIME * GRID_N_MARGIN
    n = np.bincount(cidx, minlength=nc)
    win = np.bincount(cidx, weights=(gs.fm[sidx] > 0).astype(float), minlength=nc)
    tot = np.bincount(cidx, weights=gs.ft[sidx].astype(float), minlength=nc)
    mar = np.bincount(cidx, weights=gs.fm[sidx].astype(float), minlength=nc)
    ok = n >= min_n
    with np.errstate(invalid="ignore", divide="ignore"):
        pw, mt, mm = win / n, tot / n, mar / n

    def mat(a, nd):
        out = []
        for t in range(GRID_N_TIME):
            out.append([round(float(a[t * GRID_N_MARGIN + j]), nd) if ok[t * GRID_N_MARGIN + j] else None for j in range(GRID_N_MARGIN)])
        return out
    return {"time_bin_s": GRID_TB, "n_time": GRID_N_TIME, "margin_bin": GRID_MB, "margin_min": GRID_MIN, "margin_max": GRID_MAX,
            "n_margin": GRID_N_MARGIN, "min_n": int(min_n), "margin_rule": MARGIN_RULE,
            "p_home_win": mat(pw, 4), "mean_total": mat(mt, 2), "mean_margin": mat(mm, 2),
            "n": [[int(n[t * GRID_N_MARGIN + j]) for j in range(GRID_N_MARGIN)] for t in range(GRID_N_TIME)]}


def _int_edges(e) -> list[int]:
    """Totals are integers, so `total >= e` equals `total >= ceil(e)`; integer edges cannot be moved by float formatting."""
    return [int(np.ceil(np.round(float(x), 9))) for x in e]


def _cells(pmf2d, thin, pool_row: bool, f):
    return [(None if (pool_row and thin[k]) else f(pmf2d[k])) for k in range(len(pmf2d))]


def build_remaining(tab: LG.Tables) -> dict:
    arm = tab.arm
    row_mode = arm.pool == "row"
    rows = tab.rows
    ex = rows[0]
    rem = {
        "time_bin_s": LG.COARSE_TB, "n_time": len(rows), "margin_edges": [int(x) for x in ex.edges], "n_margin": len(ex.edges) + 1,
        "min_n": int(tab.min_n),
        "rule": ("state = score at the midpoint of time bin t (last possession-open checkpoint at or before it); margin bin by "
                 "np.searchsorted(margin_edges, home-away, side=right); total bin by total_edges[t] the same way; curve {k0, v}: "
                 "v[j] = per-mille P(remaining >= k0 + j), 1000 below k0, 0 past the end; null cell (< min_n) -> "
                 + ("use the time row" if row_mode else "never null here: thin cells are neighbour-pooled") + f"; arm {arm.name}"),
        "n": [[int(x) for x in r.n_m] for r in rows],
        "widened": [[int(x) for x in r.widened_m] for r in rows],
        "margin_sf": [_cells(r.rm, r.thin_m, row_mode, _mcurve) for r in rows],
        "time_margin_sf": [_mcurve(r.rm_row) for r in rows],
        "time_total_sf": [_tcurve(r.rt_row) for r in rows],
    }
    if arm.total_axis:
        rem["total_edges"] = [_int_edges(r.t_edges) for r in rows]
        rem["total_n"] = [[int(x) for x in r.n_t] for r in rows]
        rem["total_sf"] = [_cells(r.rt_t, r.thin_t, row_mode, _tcurve) for r in rows]
    else:
        rem["total_sf"] = [_cells(r.rt_m, r.thin_m, row_mode, _tcurve) for r in rows]
    if tab.late:
        L = tab.late
        rem["late"] = {
            "start_s": LG.LATE_START, "time_bin_s": LG.LATE_TB, "n_time": len(L), "margin_edges": [int(x) for x in L[0].edges],
            "n": [[int(x) for x in r.n_m] for r in L], "widened": [[int(x) for x in r.widened_m] for r in L],
            "margin_sf": [_cells(r.rm, r.thin_m, row_mode, _mcurve) for r in L],
            "total_sf": [_cells(r.rt_m, r.thin_m, row_mode, _tcurve) for r in L],
        }
    if tab.end is not None:
        e = tab.end
        rem["end"] = {"n_tied": int(e.n_ot), "tied_margin_sf": None if e.ot_rm is None else _mcurve(e.ot_rm),
                      "tied_total_sf": None if e.ot_rt is None else _tcurve(e.ot_rt)}
    return rem


def sanity_gates(tab: LG.Tables, gs: LG.GameSeeds) -> dict:
    """Cheap exporter gates; raise ValueError on failure. Late-game win probability ordered +12 > 0 > -12; kickoff total within 1.0 pt."""
    out = {}
    t_probe = 2200.0 if tab.late else 2100.0
    z = np.zeros(3, dtype=np.int64)
    m = np.array([12, 0, -12])
    s = np.full(3, 100)
    rm, rt, _, _ = LG.price(tab, t_probe, m, s)
    pw = LG.win_prob(rm, m)
    out["p_win_late_+12_0_-12"] = [round(float(x), 4) for x in pw]
    # a thin-cell game (tiny seed counts) can have equal values at the clip; strict order is required only when n is not tiny
    if not (pw[0] >= pw[1] >= pw[2]) or (gs.S >= 100 and not (pw[0] > pw[1] > pw[2])):
        raise ValueError(f"late-game win probability not ordered: {pw}")
    m60, s60 = gs.state_at(60.0)             # in-sample: every seed's own first-row state priced; mean priced total ~ ensemble total
    rm0, rt0, _, _ = LG.price(tab, 60.0, m60, s60)
    priced = float((s60 + (rt0 * np.arange(LG.RT_N)[None, :]).sum(axis=1)).mean())
    out["kickoff_total_gap"] = round(priced - float(gs.ft.mean()), 3)
    if abs(priced - float(gs.ft.mean())) > max(1.0, 10.0 / np.sqrt(gs.S)):
        raise ValueError(f"kickoff total {priced:.2f} vs ensemble {gs.ft.mean():.2f}")
    return out


def verify_vs_games(gs: LG.GameSeeds, games_g: pd.DataFrame) -> None:
    """Second leg (CFB `verify_traj_vs_games`): the trajectory's terminal scores equal games.parquet for every kept seed."""
    g = games_g.set_index("seed")
    miss = [int(x) for x in gs.seeds if int(x) not in g.index]
    if miss:
        raise ValueError(f"trajectory seeds absent from games.parquet: {miss[:5]}")
    hp = g.loc[gs.seeds, "home_pts"].to_numpy()
    ap = g.loc[gs.seeds, "away_pts"].to_numpy()
    th = (gs.fm + gs.ft) // 2
    ta = gs.ft - th
    bad = int(((th != hp) | (ta != ap)).sum())
    if bad:
        raise ValueError(f"{bad} of {len(gs.seeds)} seeds: trajectory terminal score != games.parquet (orientation or run mismatch)")


def build_live(gs: LG.GameSeeds, game: dict, arm_name: str = DEFAULT_ARM, source_tag: str = "", updated: str = "",
               games_g: pd.DataFrame | None = None, min_n: int | None = None) -> dict:
    """gs = ALL seeds of the game in the trajectory. game = {game_id, slug, teamA(home), teamB(away)}."""
    if games_g is not None:
        verify_vs_games(gs, games_g)
    arm = LG.ARMS[arm_name]
    mn = int(min_n) if min_n else LG.scaled_min_n(gs.S)
    tab = LG.build_tables(gs, arm, mn)
    gates = sanity_gates(tab, gs)
    return {
        "convention": CONVENTION, "game_id": int(game["game_id"]), "slug": game["slug"], "teamA": game["teamA"], "teamB": game["teamB"],
        "n_sims": int(gs.S), "updated": updated, "source_tag": source_tag,
        "orientation": {"engine_home": game["teamA"], "schedule_home": game["teamA"], "flipped": False, "traj_vs_games": "direct"},
        "clock": dict(CLOCK),
        "grid": build_grid(gs, LG.scaled_min_n(gs.S, 100)),
        "uncond": {"p_home_win": round(float((gs.fm > 0).mean()), 4), "mean_total": round(float(gs.ft.mean()), 2),
                   "mean_margin": round(float(gs.fm.mean()), 2)},
        "remaining": build_remaining(tab),
        "live_arm": arm_name,
        "gates": gates,
        "note": "pregame ensemble sliced by (clock, margin, total so far); not updated for in-game news; unvalidated until the shadow logger has two weeks",
    }


def dumps(doc: dict) -> str:
    return json.dumps(doc, separators=(",", ":"))


def trajectory_path(daily_root: Path, day: str, run_id: str) -> Path:
    """results/trajectories/daily_<date>_<run_id>/trajectory.parquet, a sibling of results/daily (docs/ops/trajectory_cbb_design_2026-10-09.md)."""
    return Path(daily_root).parent / "trajectories" / f"daily_{day}_{run_id}" / "trajectory.parquet"


def read_game_traj(path: Path, game_id: int) -> pd.DataFrame:
    cols = ["game_id", "seed", "poss_idx", "period", "clock_start", "home_score", "away_score"]
    return pd.read_parquet(path, columns=cols, filters=[("game_id", "==", int(game_id))])


def write_game_live(traj_path: Path, game: dict, games_g: pd.DataFrame, gdir: Path, source_tag: str, updated: str,
                    arm_name: str = DEFAULT_ARM) -> int:
    """Read one game's rows, build and write `<gdir>/live.json`; returns bytes written. Raises on any gate failure."""
    df = read_game_traj(traj_path, game["game_id"])
    if df.empty:
        raise ValueError(f"game {game['game_id']} absent from {traj_path}")
    gs = LG.prepare_game(df)
    doc = build_live(gs, game, arm_name, source_tag, updated, games_g)
    txt = dumps(doc)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "live.json").write_text(txt, encoding="utf-8")
    return len(txt.encode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--games", required=True, help="the run's games.parquet")
    ap.add_argument("--slate", required=True, help="the run's slate.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", default=DEFAULT_ARM, choices=sorted(LG.ARMS))
    ap.add_argument("--source-tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "scripts"))
    import time
    from export_site_day_v1 import make_slug
    games = pd.read_parquet(a.games)
    slate = pd.read_parquet(a.slate)
    t0 = time.time()
    sizes = {}
    for i, r in enumerate(slate.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        slug = make_slug(str(r.home_name), str(r.away_name))
        g = {"game_id": int(r.game_id), "slug": slug, "teamA": str(r.home_name), "teamB": str(r.away_name)}
        try:
            sizes[slug] = write_game_live(Path(a.traj), g, games[games.game_id == r.game_id], Path(a.out_dir) / slug, a.source_tag,
                                          pd.Timestamp.utcnow().isoformat())
        except Exception as e:  # noqa: BLE001
            print(f"[live] {slug}: FAILED {e}")
    print(f"[live] {len(sizes)} games, {sum(sizes.values()) / 1e6:.2f} MB, {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
