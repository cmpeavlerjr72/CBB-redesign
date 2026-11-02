"""
grade_live_grid_v1.py -- ONE blind grader for the live-grid binning bake-off (docs/models/live_grid/experiments.md).

Per game the seeds are split into TRAIN / TEST (seed-level holdout). Every arm builds its tables from the TRAIN seeds only and is
scored on the SAME test checkpoints. Metrics: M1 win log loss, M2 CRPS remaining total, M3 CRPS remaining margin (+ Brier, segments,
calibration, per-game paired differences). Noise floor = same arm across independent seed splits.

  python scripts/grade_live_grid_v1.py --tags clk10_ctrl_o0_s25 clk10_ctrl_o1000_s25 ... --n-train 100 --n-test 25 --splits 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim import live_grid as LG  # noqa: E402

CHECKPOINTS = [240, 600, 960, 1200, 1500, 1800, 2040, 2160, 2220, 2280, 2340, 2395]
COLS = ["game_id", "seed", "poss_idx", "period", "clock_start", "home_score", "away_score"]
ALL_ARMS = list(LG.ARMS)
EPS = 1e-12


def load_games(tags: list[str]) -> list[LG.GameSeeds]:
    parts = [pd.read_parquet(ROOT / "results" / "trajectories" / t / "trajectory.parquet", columns=COLS) for t in tags]
    df = pd.concat(parts, ignore_index=True)
    games = []
    for gid, g in df.groupby("game_id", sort=True):
        games.append(LG.prepare_game(g))
    return games


def split_idx(game_id: int, split: int, S: int, n_train: int, n_test: int):
    rng = np.random.default_rng([split, int(game_id)])
    perm = rng.permutation(S)
    test = np.sort(perm[:n_test])
    train = np.sort(perm[n_test:n_test + n_train])
    assert not set(test) & set(train)
    return train, test


def eval_game(args):
    gs, split, n_train, n_test, league_ot, arms = args
    tr, te = split_idx(gs.game_id, split, gs.S, n_train, n_test)
    gtr, gte = gs.take(tr), gs.take(te)
    min_n = LG.scaled_min_n(len(tr))
    p_g = float((gtr.fm > 0).mean())
    meta = {"game": [], "t": [], "m": [], "s": [], "win": [], "ot": [], "pg": []}
    ymarg, ytot = [], []
    for t in CHECKPOINTS:
        m, s = gte.state_at(t)
        n = len(m)
        meta["game"].append(np.full(n, gs.game_id))
        meta["t"].append(np.full(n, t))
        meta["m"].append(m)
        meta["s"].append(s)
        meta["win"].append((gte.fm > 0).astype(np.int8))
        meta["ot"].append(gte.ot.astype(np.int8))
        meta["pg"].append(np.full(n, p_g))
        ymarg.append(np.clip(gte.fm - m + LG.RM_OFF, 0, LG.RM_N - 1))
        ytot.append(np.clip(gte.ft - s, 0, LG.RT_N - 1))
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    ymarg = np.concatenate(ymarg)
    ytot = np.concatenate(ytot)
    cache = {}
    out = {}
    for name in arms:
        tab = LG.build_tables(gtr, LG.ARMS[name], min_n, league_ot, cache)
        P, TH, WI, CT, CM = [], [], [], [], []
        for t in CHECKPOINTS:
            m, s = gte.state_at(t)
            rm, rt, thin, wid = LG.price(tab, float(t), m, s)
            P.append(LG.win_prob(rm, m))
            TH.append(thin)
            WI.append(wid)
            CT.append(None)
            CM.append(None)
            CT[-1] = rt
            CM[-1] = rm
        p = np.concatenate(P)
        crps_t = np.concatenate([LG.crps_discrete(CT[i], ytot[i * n_test:(i + 1) * n_test]) for i in range(len(CHECKPOINTS))])
        crps_m = np.concatenate([LG.crps_discrete(CM[i], ymarg[i * n_test:(i + 1) * n_test]) for i in range(len(CHECKPOINTS))])
        w = meta["win"]
        ll = -(w * np.log(p) + (1 - w) * np.log(1 - p))
        out[name] = dict(p=p.astype(np.float32), ll=ll.astype(np.float32), br=((p - w) ** 2).astype(np.float32),
                         crps_t=crps_t.astype(np.float32), crps_m=crps_m.astype(np.float32),
                         thin=np.concatenate(TH).astype(np.float32), wid=np.concatenate(WI).astype(np.float32))
    return meta, out


def run_split(games, split, n_train, n_test, arms, workers):
    prior_in = []
    for gs in games:
        tr, _ = split_idx(gs.game_id, split, gs.S, n_train, n_test)
        prior_in.append(gs.take(tr))
    league_ot = LG.league_ot_prior(prior_in)
    tasks = [(gs, split, n_train, n_test, league_ot, arms) for gs in games]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(eval_game, tasks, chunksize=8))
    else:
        res = [eval_game(t) for t in tasks]
    meta = {k: np.concatenate([r[0][k] for r in res]) for k in res[0][0]}
    df = pd.DataFrame(meta)
    for a in arms:
        for k in res[0][1][a]:
            df[f"{a}__{k}"] = np.concatenate([r[1][a][k] for r in res])
    df["split"] = split
    return df


def _md(d: pd.DataFrame) -> str:
    cols = [d.index.name or "idx"] + list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for i, r in d.iterrows():
        out.append("| " + " | ".join([str(i)] + [str(v) for v in r.to_numpy()]) + " |")
    return chr(10).join(out)


def summarise(df: pd.DataFrame, arms: list[str]) -> str:
    L = []
    splits = sorted(df.split.unique())
    metrics = [("M1_logloss", "ll"), ("M2_crps_total", "crps_t"), ("M3_crps_margin", "crps_m"), ("Brier", "br")]
    late = df.t >= 2160

    def msplit(a, k, mask=None):
        d = df if mask is None else df[mask]
        return np.array([d[d.split == sp][f"{a}__{k}"].mean() for sp in splits])

    tab = {}
    for a in arms:
        for mn, k in metrics:
            v = msplit(a, k)
            tab[(a, mn)] = v
        tab[(a, "M1_late")] = msplit(a, "ll", late)
        tab[(a, "thin_share")] = msplit(a, "thin")
    L.append("### Overall (mean over splits; floor = max-min of the same arm across splits)\n")
    L.append("| arm | M1 logloss | floor | M2 CRPS total | floor | M3 CRPS margin | floor | Brier | M1 late (t>=2160) | floor | thin share |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    fl = {}
    for a in arms:
        row = [a]
        for mn in ("M1_logloss", "M2_crps_total", "M3_crps_margin"):
            v = tab[(a, mn)]
            fl[(a, mn)] = v.max() - v.min()
            row += [f"{v.mean():.5f}", f"{fl[(a, mn)]:.5f}"]
        row += [f"{tab[(a, 'Brier')].mean():.5f}"]
        v = tab[(a, "M1_late")]
        fl[(a, "M1_late")] = v.max() - v.min()
        row += [f"{v.mean():.5f}", f"{fl[(a, 'M1_late')]:.5f}", f"{tab[(a, 'thin_share')].mean():.3f}"]
        L.append("| " + " | ".join(row) + " |")

    def beats(x, y, mn):
        d = tab[(y, mn)] - tab[(x, mn)]
        f = max(fl[(x, mn)], fl[(y, mn)])
        return bool(d.mean() > f and (d > 0).all())

    # selection
    elig = [a for a in LG.ELIGIBLE if a in arms]
    verdict = {}
    if elig:
        best = min(elig, key=lambda a: tab[(a, "M1_logloss")].mean())
        ok = []      # literal reading: one-sided distance to the best-M1 arm (defective: a better arm always passes)
        ok2 = []     # per-metric reading (8c): within one floor of the best arm on EACH of M1, M2, M3, M1-late
        mets = ("M1_logloss", "M2_crps_total", "M3_crps_margin", "M1_late")
        bestm = {mn: min(elig, key=lambda a: tab[(a, mn)].mean()) for mn in mets}
        for a in elig:
            within = all(tab[(a, mn)].mean() - tab[(best, mn)].mean() <= max(fl[(a, mn)], fl[(best, mn)]) for mn in mets[:3])
            late_veto = (tab[(a, "M1_late")].mean() - tab[(best, "M1_late")].mean()) > max(fl[(a, "M1_late")], fl[(best, "M1_late")])
            if within and not late_veto:
                ok.append(a)
            if all(tab[(a, mn)].mean() - tab[(bestm[mn], mn)].mean() <= max(fl[(a, mn)], fl[(bestm[mn], mn)]) for mn in mets):
                ok2.append(a)
        winner_literal = ok[0] if ok else best
        winner = ok2[0] if ok2 else best
        verdict = {"best_M1": best, "winner_literal_one_sided": winner_literal, "winner": winner,
                   "within_floor_per_metric": ok2, "best_per_metric": bestm}
        L.append(f"\nBest M1 among eligible: `{best}`. Literal one-sided reading passes {ok} (winner `{winner_literal}`). "
                 f"Per-metric reading (within one floor of the best arm on EACH of M1, M2, M3, M1-late) passes {ok2}. "
                 f"**WINNER (simplest of those): `{winner}`**; best per metric {bestm}.\n")
        L.append("### Pairwise BEATS (X beats Y: mean gain > larger floor and better in every split)\n")
        L.append("| X | Y | M1 | M2 | M3 | mean dM1 | mean dM2 | mean dM3 |")
        L.append("|---|---|---|---|---|---|---|---|")
        pairs = [("A_pool", "A_row"), ("B_pool", "B_row"), ("C_pool", "C_row"), ("B_row", "A_row"), ("B_pool", "A_pool"), ("D_row", "A_row"), ("D_pool", "A_pool"), ("C_row", "D_row"), ("C_pool", "D_pool"), ("D_pool", "D_row"),
                 ("C_row", "B_row"), ("C_pool", "B_pool"), ("A_row", "R0"), ("A_row", "A0")]
        for x, y in pairs:
            if x in arms and y in arms:
                L.append(f"| {x} | {y} | " + " | ".join(str(beats(x, y, mn)) for mn in ("M1_logloss", "M2_crps_total", "M3_crps_margin")) + " | " +
                         " | ".join(f"{(tab[(y, mn)] - tab[(x, mn)]).mean():+.5f}" for mn in ("M1_logloss", "M2_crps_total", "M3_crps_margin")) + " |")
        if "R0" in arms and "A_row" in arms:
            verdict["R0_loses_to_A_row_on_M1"] = beats("A_row", "R0", "M1_logloss")
        if "B_row" in arms:
            verdict["total_axis_beats_margin_only_M2"] = {s: beats(f"B_{s}", f"A_{s}", "M2_crps_total") for s in ("row", "pool") if f"B_{s}" in arms}
    # segments
    df = df.copy()
    df["seg_period"] = np.where(df.t < 1200, "period1", "period2")
    df["seg_absm"] = pd.cut(df.m.abs(), [-1, 5, 12, 999], labels=["0-5", "6-12", "13+"]).astype(str)
    df["seg_late"] = np.where(df.t >= 2160, "late(t>=2160)", "early")
    df["seg_ot"] = np.where(df.ot == 1, "OT games", "regulation games")
    df["seg_fav"] = "q" + (pd.qcut(df.pg.rank(method="first"), 5, labels=False) + 1).astype(str)
    df["seg_t"] = df.t.astype(str).str.zfill(4)
    for seg in ("seg_period", "seg_absm", "seg_late", "seg_ot", "seg_fav", "seg_t"):
        L.append(f"\n### Segment: {seg} (M1 logloss | M2 CRPS total | M3 CRPS margin; n rows per split in first column; <30 rows UNDERPOWERED)\n")
        L.append("| segment | n/split | " + " | ".join(arms) + " |")
        L.append("|---|---|" + "---|" * len(arms))
        for sv, g in df.groupby(seg):
            n = len(g) / len(splits)
            cells = []
            for a in arms:
                cells.append(f"{g[f'{a}__ll'].mean():.4f} / {g[f'{a}__crps_t'].mean():.3f} / {g[f'{a}__crps_m'].mean():.3f}")
            flag = " UNDERPOWERED" if n < 30 else ""
            L.append(f"| {sv}{flag} | {n:.0f} | " + " | ".join(cells) + " |")
    # per-game paired vs A_row on M1
    if "A_row" in arms:
        L.append("\n### Per-game paired M1 difference vs A_row (positive = arm better); game-clustered bootstrap SE (1,000 draws)\n")
        L.append("| arm | mean gain | SE | share of games arm better | games |")
        L.append("|---|---|---|---|---|")
        gm = df.groupby("game")[[f"{a}__ll" for a in arms]].mean()
        rng = np.random.default_rng(0)
        for a in arms:
            if a == "A_row":
                continue
            d = (gm["A_row__ll"] - gm[f"{a}__ll"]).to_numpy()
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            L.append(f"| {a} | {d.mean():+.5f} | {np.std(bs):.5f} | {(d > 0).mean():.3f} | {len(d)} |")
    # calibration of the leading arms (deciles), responsiveness by favourite quintile at t=240
    show = [a for a in ("R0", verdict.get("winner")) if a in arms]
    for a in show:
        L.append(f"\n### Calibration of `{a}` (predicted-probability decile, all rows)\n")
        d = pd.DataFrame({"p": df[f"{a}__p"], "w": df.win})
        d["dec"] = pd.qcut(d.p.rank(method="first"), 10, labels=False)
        c = d.groupby("dec").agg(pred=("p", "mean"), actual=("w", "mean"), n=("w", "size")).round(4)
        L.append(_md(c))
        e = df[df.t == 240]
        r = e.groupby("seg_fav").apply(lambda g: pd.Series({"mean_pred": g[f"{a}__p"].mean(), "actual": g.win.mean(), "n": len(g)})).round(4)
        L.append(f"\nResponsiveness of `{a}` at t=240 by pregame favourite-strength quintile (predictions must slope with actuals):\n")
        L.append(_md(r))
    # thin cells
    L.append("\n### Thin-cell share of looked-up cells (blend-weighted) by segment\n")
    L.append("| segment | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for seg in ("seg_late", "seg_absm"):
        for sv, g in df.groupby(seg):
            L.append(f"| {sv} | " + " | ".join(f"{g[f'{a}__thin'].mean():.3f} (w {g[f'{a}__wid'].mean():.2f})" for a in arms) + " |")
    return "\n".join(L), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=25)
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/diag/live_grid_bakeoff_v1")
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    games = load_games(a.tags)
    if a.max_games:
        games = games[:a.max_games]
    print(f"loaded {len(games)} games, seeds/game {games[0].S}, {time.time() - t0:.0f}s", flush=True)
    dfs = []
    for sp in a.splits:
        dfs.append(run_split(games, sp, a.n_train, a.n_test, a.arms, a.workers))
        print(f"split {sp} done {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    md, verdict = summarise(df, a.arms)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps({**verdict, "args": vars(a)}, indent=1, default=str), encoding="utf-8")
    print(md)
    print(json.dumps(verdict, indent=1, default=str))


if __name__ == "__main__":
    main()
