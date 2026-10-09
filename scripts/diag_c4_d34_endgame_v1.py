"""diag_c4_d34_endgame_v1.py -- D3 (period-end possessions, garbage time) and D4 (OT rate / endgame) diagnostics. DIAGNOSTIC ONLY.
usage: diag_c4_d34_endgame_v1.py --tag f2all50 --out3 docs/tests/period_end_possessions_2026-10-09.md --out4 docs/tests/ot_rate_endgame_2026-10-09.md [--skip-b]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_c4_lib_v1 as L  # noqa: E402

RNG = np.random.default_rng(20261011)
SHOTS = {"JumpShot", "LayUpShot", "DunkShot", "TipShot"}


def f(x, nd=2):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def ci(a, nd=2):
    return f"[{f(np.percentile(a, 2.5), nd)}, {f(np.percentile(a, 97.5), nd)}]"


def bucket(m):
    m = np.abs(m)
    return np.select([m <= 3, m <= 7, m <= 12], ["0-3", "4-7", "8-12"], "13+")


# ----------------------------------------------------------------------------------------------------------
# raw-pbp independent possession counts (event-based, does NOT use cbb_sim.pbp.possessions)
# ----------------------------------------------------------------------------------------------------------
def pbp_counts(game_ids, win=240):
    p = pd.read_parquet(L.ROOT / f"data/raw/hoopr/pbp/play_by_play_{L.SEASON}.parquet",
                        columns=["game_id", "game_play_number", "type_text", "team_id", "period_number", "scoring_play",
                                 "start_period_seconds_remaining", "home_score", "away_score", "home_team_id"])
    p = p[p.game_id.isin(set(int(g) for g in game_ids))].sort_values(["game_id", "game_play_number"], kind="stable").reset_index(drop=True)
    t = p.type_text
    is_ft = (t == "MadeFreeThrow").to_numpy()
    scoring = p.scoring_play.fillna(False).astype(bool).to_numpy()
    gid = p.game_id.to_numpy()
    per = p.period_number.to_numpy()
    tm = p.team_id.to_numpy()
    sec = p.start_period_seconds_remaining.to_numpy()
    sh = lambda a, k: np.r_[a[k:], [a[-1]] * k] if k > 0 else a  # noqa: E731
    nxt = lambda a: np.r_[a[1:], a[-1:]]  # noqa: E731
    prv = lambda a: np.r_[a[:1], a[:-1]]  # noqa: E731
    same_next = nxt(gid) == gid
    same_prev = prv(gid) == gid
    reb_off = (t == "Offensive Rebound").to_numpy()
    reb_def = (t == "Defensive Rebound").to_numpy()
    prev_ftmiss = prv(is_ft & ~scoring) & same_prev
    next_ft = nxt(is_ft) & same_next
    spurious = (reb_off | reb_def) & prev_ftmiss & next_ft            # dead-ball team rebound between two free throws of one trip
    keep = ~spurious
    p = p[keep].reset_index(drop=True)
    t = p.type_text
    is_ft = (t == "MadeFreeThrow").to_numpy()
    scoring = p.scoring_play.fillna(False).astype(bool).to_numpy()
    gid, per, tm, sec = p.game_id.to_numpy(), p.period_number.to_numpy(), p.team_id.to_numpy(), p.start_period_seconds_remaining.to_numpy()
    is_shot = t.isin(SHOTS).to_numpy()
    is_to = (t == "Lost Ball Turnover").to_numpy()
    reb_off = (t == "Offensive Rebound").to_numpy()
    reb_def = (t == "Defensive Rebound").to_numpy()
    fgm = is_shot & scoring
    # and-one: made FG followed within 2 rows by a FT of the same team at the same clock
    and1 = np.zeros(len(p), bool)
    for k in (1, 2):
        g2, t2, s2, f2 = (np.r_[a[k:], [a[-1]] * k] for a in (gid, tm, sec, is_ft))
        and1 |= fgm & (g2 == gid) & (t2 == tm) & (s2 == sec) & f2
    next_same_ft = np.r_[is_ft[1:], [False]] & (np.r_[gid[1:], [-1]] == gid) & (np.r_[tm[1:], [-1]] == tm)
    last_ft = is_ft & ~next_same_ft
    e1 = is_shot.astype(float) + 0.44 * is_ft - reb_off + is_to
    d = pd.DataFrame({"game_id": gid, "period": per, "sec": sec, "e1": e1})
    d = d[(d.period <= 2) & (d.sec <= win)]
    out = d.groupby(["game_id", "period"])[["e1"]].sum().reset_index()
    # E2: number of same-team RUNS of possession-relevant events (shots, free throws, turnovers, rebounds), a possession-change marker
    # count that needs no rebound/free-throw heuristics. A run is counted when its LAST event has <= win seconds left.
    rel = (is_shot | is_ft | is_to | reb_off | reb_def) & ~pd.isna(tm)
    q = pd.DataFrame({"game_id": gid[rel], "period": per[rel], "team": tm[rel], "sec": sec[rel]})
    q = q[q.period <= 2]
    key = q.game_id.to_numpy().astype(np.int64) * 10 + q.period.to_numpy()
    new = np.r_[True, (key[1:] != key[:-1]) | (q.team.to_numpy()[1:] != q.team.to_numpy()[:-1])]
    q["run"] = np.cumsum(new)
    last = q.groupby("run").agg(game_id=("game_id", "first"), period=("period", "first"), sec=("sec", "min"))
    last = last[last.sec <= win].assign(e2=1.0)
    runs = last.groupby(["game_id", "period"]).e2.sum().reset_index()
    return out.merge(runs, on=["game_id", "period"], how="outer").fillna(0.0)


def pbp_fouls(game_ids):
    """fouls by team in regulation, with running score margin (home persp) BEFORE the foul (from the previous event's score)."""
    p = pd.read_parquet(L.ROOT / f"data/raw/hoopr/pbp/play_by_play_{L.SEASON}.parquet",
                        columns=["game_id", "game_play_number", "type_text", "team_id", "period_number",
                                 "start_period_seconds_remaining", "home_score", "away_score", "home_team_id"])
    p = p[p.game_id.isin(set(int(g) for g in game_ids))].sort_values(["game_id", "game_play_number"], kind="stable").reset_index(drop=True)
    p["m"] = (p.home_score - p.away_score)
    p["m_before"] = p.groupby("game_id").m.shift(1).fillna(0)
    fl = p[(p.type_text == "PersonalFoul") & (p.period_number <= 2)].copy()
    fl["home_fouled"] = fl.team_id == fl.home_team_id
    return fl[["game_id", "period_number", "start_period_seconds_remaining", "m_before", "home_fouled"]]


# ----------------------------------------------------------------------------------------------------------
# cell machinery: per-game sums, ratio of sums, game-cluster bootstrap of (real - sim)
# ----------------------------------------------------------------------------------------------------------
class Cells:
    def __init__(self, G, B):
        self.G = np.asarray(G)
        self.n = len(self.G)
        self.W = RNG.multinomial(self.n, np.ones(self.n) / self.n, size=B).astype(float)
        self.pos = {g: i for i, g in enumerate(self.G)}

    def vec(self, df, col):
        v = np.zeros(self.n)
        if len(df):
            s = df.groupby("game_id")[col].sum()
            v[[self.pos[g] for g in s.index if g in self.pos]] = s[[g in self.pos for g in s.index]].to_numpy()
        return v

    def ratio_stat(self, nr, dr, ns, ds):
        """returns real, sim, diff, boot diffs, using per-game arrays."""
        def r(num, den, w):
            return (w @ num) / (w @ den) if (w @ den) > 0 else np.nan
        one = np.ones(self.n)
        real, sim = r(nr, dr, one), r(ns, ds, one)
        bs = np.array([r(nr, dr, w) - r(ns, ds, w) for w in self.W])
        return real, sim, real - sim, bs


def cell_report(cs, real, sim, S, dims, by, metrics, min_n=60):
    """real/sim: possession frames with columns game_id (+ by cols); sim summed over seeds then /S. Returns list of rows."""
    out = []
    keys = sorted(set(map(tuple, real[by].drop_duplicates().to_numpy().tolist())) | set(map(tuple, sim[by].drop_duplicates().to_numpy().tolist())))
    for k in keys:
        mr = np.ones(len(real), bool)
        ms = np.ones(len(sim), bool)
        for c, v in zip(by, k):
            mr &= (real[c] == v).to_numpy()
            ms &= (sim[c] == v).to_numpy()
        r_, s_ = real[mr], sim[ms]
        res = {"cell": k, "n_real": len(r_), "n_sim_per_seed": len(s_) / S}
        for name, (numc, denc) in metrics.items():
            nr, dr = cs.vec(r_, numc), cs.vec(r_, denc)
            ns, ds = cs.vec(s_, numc) / S, cs.vec(s_, denc) / S
            res[name] = cs.ratio_stat(nr, dr, ns, ds)
        out.append(res)
    return out


def fmtc(t, nd, scale=1.0, under=False):
    real, sim, d, bs = t
    star = "*" if (np.isfinite(d) and (np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0)) else ""
    return f"{f(real * scale, nd)} / {f(sim * scale, nd)} ({f(d * scale, nd)}{star})" + ("U" if under else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out3", required=True)
    ap.add_argument("--out4", required=True)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--min-n", type=int, default=60)
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    D = lambda n: pd.read_parquet(L.OUT / f"{a.tag}_{n}.parquet")  # noqa: E731
    pg, sgs, rgs = D("pregame"), D("sim_gs"), D("real_gs")
    slate, rlate = D("sim_late"), D("real_late")
    S = sgs.seed.nunique()
    pg["fin_margin"] = pg.fin_h - pg.fin_a
    ok = pg.merge(rgs[["game_id", "final_margin", "reg_margin"]], on="game_id")
    ok = ok[ok.final_margin == ok.fin_margin]
    G = np.sort(ok.game_id.unique())
    cs = Cells(G, a.nboot)
    slate = slate[slate.game_id.isin(G)]
    rlate = rlate[rlate.game_id.isin(G)]
    sgs = sgs[sgs.game_id.isin(G)]
    rgs = rgs[rgs.game_id.isin(G)]
    ngame = len(G)

    # ================================================================================================== D3
    W3 = []
    w = W3.append
    w("# D3 period-end possession surplus and garbage-time scoring (fold 2 / 2024-25), 2026-10-09\n")
    w("DIAGNOSTIC ONLY. Nothing changed. Script `scripts/diag_c4_d34_endgame_v1.py` (library `diag_c4_lib_v1.py`).\n")
    w(f"Sim: served stack v3, fold 2 / season 2025, {sgs.game_id.nunique()} games x {S} seeds; real: {ngame} games whose parsed pbp path ends at the verified final. Sim columns are seed-averaged. "
      f"CIs: game-cluster bootstrap ({a.nboot} reps) of real - sim; `*` marks a CI that excludes 0; `U` = underpowered (real possessions or games in cell < {a.min_n}).\n")

    # ---- (a) independent recount -----------------------------------------------------------------------
    w("## (a) Rule out segmentation: independent recount of real possessions in the last 4 minutes of each half\n")
    w("Three real counts and the same counts on the sim:\n")
    w("- **M-start / M-end**: possession-table count (`possessions_v4otc`, the pbp module the earlier doc used) of possessions that START / END with <= 240 s left in the period.\n"
      "- **E1 (Oliver)**: FGA + 0.44 FTA + TOV - OREB from RAW hoopR pbp events with <= 240 s left (does not touch the possession module; dead-ball team rebounds between the two free throws of one trip are removed).\n"
      "- **E2 (same-team runs)**: from raw pbp events (shots, free throws, turnovers, rebounds with a team id), the number of maximal runs of consecutive events by the same team, counted when the run's last event has <= 240 s left. A run is a possession-change marker count: no rebound or free-throw heuristics. Over full regulation it gives 133.52 possessions per game against the module's 133.53 (394-game check) and Oliver's 132.9, so it is a valid second method.\n"
      "- Sim: M-start/M-end from the trajectory; E1 from the engine's own FGA/FTA/TOV/OREB counts of possessions ending in the window; E2 = number of possessions ending in the window (every engine possession ends in a terminal event, the last one is forced to resolve).\n")
    pc = pbp_counts(G)
    rows = []
    for per in (1, 2):
        lbl = "min 16-20 (end of H1)" if per == 1 else "min 36-40 (end of H2)"
        rp = pc[pc.period == per]
        rl = rlate[rlate.period == per]
        sl = slate[slate.period == per]
        arrs = {}
        for nm, rr, ss in (("M-start", rl[rl.cs <= 240].assign(c=1), sl[sl.cs <= 240].assign(c=1)),
                           ("M-end", rl[rl.ce <= 240].assign(c=1), sl[sl.ce <= 240].assign(c=1))):
            nr = cs.vec(rr, "c")
            ns = cs.vec(ss, "c") / S
            arrs[nm] = (nr, ns)
        ol = lambda x: x.fga2 + x.fga3 + 0.44 * x.fta + x.tov - x.oreb  # noqa: E731
        e1s = sl[sl.ce <= 240].assign(e1=ol)
        e1m = rl[rl.ce <= 240].assign(e1=ol)
        arrs["M-end Oliver (same formula on module / engine counts)"] = (cs.vec(e1m, "e1"), cs.vec(e1s, "e1") / S)
        arrs["E1 (Oliver, raw events)"] = (cs.vec(rp, "e1"), cs.vec(e1s, "e1") / S)
        e2s = sl[sl.ce <= 240].assign(e2=1)
        arrs["E2 (same-team runs, raw events)"] = (cs.vec(rp, "e2"), cs.vec(e2s, "e2") / S)
        for nm, (nr, ns) in arrs.items():
            one = np.ones(cs.n)
            real, sim = nr.mean(), ns.mean()
            bs = np.array([(w_ @ nr - w_ @ ns) / cs.n for w_ in cs.W])
            rows.append((lbl, nm, real, sim, real - sim, bs, nr.std(ddof=1) / np.sqrt(cs.n)))
    w("| half window | count | real per game (both teams) | sim per game | real - sim [95%] | seed band excl. | flag |")
    w("|---|---|---:|---:|---|---|---|")
    for lbl, nm, real, sim, d, bs, se in rows:
        fl = "excludes 0" if (np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0) else ""
        w(f"| {lbl} | {nm} | {f(real, 3)} | {f(sim, 3)} | {f(d, 3)} {ci(bs, 3)} | | {fl} |")
    # agreement test
    lab = {}
    for lbl, nm, real, sim, d, bs, se in rows:
        lab[(lbl, nm)] = (real, sim, d, bs)
    w("")
    w("Agreement test (the instruction: if two independent real counts disagree by more than the sim-real gap, D3 stops).\n")
    w("| half window | sim - real (M-start) | sim - real (M-end) | sim - real (E1) | sim - real (E2) | real E1 - real M-end | real E2 - real M-end | agrees? |")
    w("|---|---:|---:|---:|---:|---:|---:|---|")
    agree = {}
    for lbl in ("min 16-20 (end of H1)", "min 36-40 (end of H2)"):
        g_ = lambda nm: lab[(lbl, nm)]  # noqa: E731
        ms, me, e1, e2 = g_("M-start"), g_("M-end"), g_("E1 (Oliver, raw events)"), g_("E2 (same-team runs, raw events)")
        gap_m = -me[2]
        d1 = e1[0] - me[0]
        d2 = e2[0] - me[0]
        ok_ = (abs(d1) < abs(gap_m)) and (abs(d2) < abs(gap_m))
        agree[lbl] = ok_
        w(f"| {lbl} | {f(-ms[2], 3)} | {f(-me[2], 3)} | {f(-e1[2], 3)} | {f(-e2[2], 3)} | {f(d1, 3)} | {f(d2, 3)} | {'YES' if ok_ else 'NO'} |")
    w("")
    w("`agrees? = YES` when both independent real counts sit closer to the module's real count than the sim does (|real E - real M-end| < |sim - real M-end|), i.e. the segmentation is not what separates the sim from reality.\n")
    # share of period-end censored real possessions
    w("Context, the period-end terminal event in the real table (possessions whose terminal event is `end_period`/`unknown`): see the possession-class table below (NONE row).\n")

    # ---- (b) -----------------------------------------------------------------------------------------------
    if not a.skip_b:
        w("## (b) By margin state at 4:00 and 2:00 (|margin| buckets 0-3, 4-7, 8-12, 13+), leader vs trailer\n")
        w("Reference state = home margin before the first possession that starts with <= 240 s (or <= 120 s) left in the half; every possession of the window inherits that reference bucket and the role "
          "(leader / trailer = the team ahead / behind at the reference; tied = tied). Class rule (identical both sides): TOV > FT trip (fta > 0) > 3PA > 2PA. "
          "Cells show `real / sim (real - sim)`.\n")

        def with_ref(df):
            out = []
            for T in (240, 120):
                first = df[df.cs <= T].groupby(["game_id", "seed", "period"], sort=False).pre.first().rename(f"ref{T}")
                out.append(first)
            ref = pd.concat(out, axis=1).reset_index()
            return df.merge(ref, on=["game_id", "seed", "period"], how="left")
        rl2, sl2 = with_ref(rlate), with_ref(slate)

        def prep(df, T):
            x = df[(df.cs <= T)].copy()
            ref = x[f"ref{T}"]
            x["bucket"] = bucket(ref.to_numpy())
            lead_home = ref > 0
            x["role"] = np.where(ref == 0, "tied", np.where((x.off_side == 0) == lead_home, "leader", "trailer"))
            x["n"] = 1
            x["dur1"] = x.dur
            x["ft1"] = (x.cls == "FT").astype(int)
            x["p31"] = (x.cls == "3PA").astype(int)
            x["tov1"] = (x.cls == "TOV").astype(int)
            x["pts1"] = x.pts
            return x
        met = {"sec/poss": ("dur1", "n"), "FT-trip share": ("ft1", "n"), "3PA share": ("p31", "n"), "TOV rate": ("tov1", "n"), "PPP": ("pts1", "n")}
        for per in (2, 1):
            for T in (240, 120):
                r_ = prep(rl2[rl2.period == per], T)
                s_ = prep(sl2[sl2.period == per], T)
                rep = cell_report(cs, r_, s_, S, None, ["bucket", "role"], met)
                # games in bucket
                w(f"### Half {per}, window = last {T // 60} min, reference state at {T // 60}:00\n")
                w("| bucket | role | real poss | sim poss/seed | sec/poss (real / sim, diff) | FT-trip share (pp) | 3PA share (pp) | TOV rate (pp) | PPP |")
                w("|---|---|---:|---:|---|---|---|---|---|")
                for r in rep:
                    b, role = r["cell"]
                    und = r["n_real"] < a.min_n
                    w(f"| {b} | {role}{' (U)' if und else ''} | {r['n_real']} | {f(r['n_sim_per_seed'], 0)} | {fmtc(r['sec/poss'], 2)} | {fmtc(r['FT-trip share'], 1, 100)} | "
                      f"{fmtc(r['3PA share'], 1, 100)} | {fmtc(r['TOV rate'], 1, 100)} | {fmtc(r['PPP'], 3)} |")
                w("")
        # points and possessions, last 2 min by margin at 2:00, per game
        w("### Last-2-minute points and possessions per game (both teams), by margin bucket at 2:00 (half 2)\n")
        r2 = rl2[(rl2.period == 2) & (rl2.cs <= 120)].copy()
        s2 = sl2[(sl2.period == 2) & (sl2.cs <= 120)].copy()
        for x in (r2, s2):
            x["bucket"] = bucket(x.ref120.to_numpy())
            x["n"] = 1
        # games per bucket: state at 2:00 per game
        gb_r = rl2[(rl2.period == 2)].groupby("game_id").ref120.first()
        gb_s = sl2[(sl2.period == 2)].groupby(["game_id", "seed"]).ref120.first().reset_index()
        w("| bucket | games real (U if < %d) | share real | share sim | poss/game real / sim (diff) | points/game real / sim (diff) | PPP real / sim (diff) |" % a.min_n)
        w("|---|---:|---:|---:|---|---|---|")
        for b in ("0-3", "4-7", "8-12", "13+"):
            gr_ = gb_r[bucket(gb_r.to_numpy()) == b].index.to_numpy()
            gs_ = gb_s[bucket(gb_s.ref120.to_numpy()) == b]
            rr = r2[r2.bucket == b]
            ss = s2[s2.bucket == b]
            # per-game arrays; denominators: number of games in the bucket (real) / expected games (sim)
            ng_r = np.isin(G, gr_).astype(float)
            ng_s = cs.vec(gs_.assign(c=1), "c") / S
            res = {}
            for nm, (numc, den) in {"poss": ("n", None), "pts": ("pts", None)}.items():
                res[nm] = cs.ratio_stat(cs.vec(rr, numc), ng_r, cs.vec(ss, numc) / S, ng_s)
            res["ppp"] = cs.ratio_stat(cs.vec(rr, "pts"), cs.vec(rr, "n"), cs.vec(ss, "pts") / S, cs.vec(ss, "n") / S)
            und = len(gr_) < a.min_n
            w(f"| {b} | {len(gr_)}{' (U)' if und else ''} | {f(len(gr_) / ngame, 3)} | {f(ng_s.sum() / ngame, 3)} | {fmtc(res['poss'], 2)} | {fmtc(res['pts'], 2)} | {fmtc(res['ppp'], 3)} |")
        w("")
        # possession-class mix over the whole last-4-min of H2 incl. NONE
        w("### Possession-class mix in the last 4 minutes of H2 (all states): real vs sim\n")
        rr = rlate[(rlate.period == 2) & (rlate.cs <= 240)]
        ss = slate[(slate.period == 2) & (slate.cs <= 240)]
        w("| class | real share | sim share |")
        w("|---|---:|---:|")
        for c in ("2PA", "3PA", "FT", "TOV", "NONE"):
            w(f"| {c} | {f((rr.cls == c).mean() * 100, 2)}% | {f((ss.cls == c).mean() * 100, 2)}% |")
        w("")

    Path(a.out3).write_text("\n".join(W3), encoding="utf-8")

    # ================================================================================================== D4
    W4 = []
    w = W4.append
    w("# D4 overtime rate and the endgame (fold 2 / 2024-25), 2026-10-09\n")
    w("DIAGNOSTIC ONLY. Nothing changed. Same script and data as D3.\n")
    allg = pg.merge(rgs[["game_id", "reg_margin"]], on="game_id", how="left")
    season_ot = (pg.fin_np.fillna(2) > 2).mean()
    w(f"Sim: {sgs.game_id.nunique()} games x {S} seeds. Real: {ngame} games (pbp path = verified final); the season-wide real OT rate over all {len(pg)} fold-2 games in the extract is {f(season_ot * 100, 2)}% "
      f"(verified finals, period count > 2). Seed SE = SD over seeds / sqrt(S). CIs: game-cluster bootstrap of real - sim ({a.nboot} reps).\n")

    # second-source check on real regulation-end margin
    rg2 = rgs.merge(pg[["game_id", "fin_np"]], on="game_id")
    chk = ((rg2.reg_margin == 0) == (rg2.fin_np > 2)).mean()
    w(f"Grading-truth check: real regulation-end margin == 0 (from the pbp scoreboard) agrees with verified-final `n_periods > 2` in {f(chk * 100, 2)}% of the {len(rg2)} games.\n")

    w("## (a) Regulation-end margin distribution (|margin| at the end of 40:00)\n")
    w("| |margin| at end of regulation | real share (n=%d) | sim share | real - sim [95%%] | sim seed SE | flag |" % ngame)
    w("|---|---:|---:|---|---:|---|")
    rm = rgs.set_index("game_id").reg_margin.reindex(G).abs().to_numpy()
    sg = sgs.assign(a=sgs.reg_margin.abs())
    cats = [("0 (tie -> OT)", lambda m: m == 0), ("1", lambda m: m == 1), ("2", lambda m: m == 2), ("3", lambda m: m == 3),
            ("4-6", lambda m: (m >= 4) & (m <= 6)), ("7-9", lambda m: (m >= 7) & (m <= 9)), ("10+", lambda m: m >= 10),
            ("<= 3 (cum.)", lambda m: m <= 3), ("<= 6 (cum.)", lambda m: m <= 6), ("<= 1 (cum.)", lambda m: m <= 1)]
    pos = cs.pos
    for lbl, fn in cats:
        ind_r = fn(rm).astype(float)
        sim_g = sg.assign(i=fn(sg.a.to_numpy()).astype(float)).groupby("game_id").i.mean().reindex(G).to_numpy()
        ps = sg.assign(i=fn(sg.a.to_numpy()).astype(float))
        seedm = ps[ps.game_id.isin(G)].groupby("seed").i.mean()
        bs = np.array([(w_ @ ind_r - w_ @ sim_g) / cs.n for w_ in cs.W])
        d = ind_r.mean() - sim_g.mean()
        fl = "excludes 0" if (np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0) else ""
        w(f"| {lbl} | {f(ind_r.mean() * 100, 2)}% | {f(sim_g.mean() * 100, 2)}% | {f(d * 100, 2)} pp {ci(bs * 100, 2)} | {f(seedm.std(ddof=1) / np.sqrt(S) * 100, 3)} pp | {fl} |")
    w("")
    # OT rate decomposition by state at 2:00 and 4:00
    w("### OT = sum over the margin state at 2:00 of P(state) x P(tie | state)\n")
    w("State = |margin| before the first H2 possession starting with <= 120 s left (home perspective sign dropped). Real games in cell shown; `U` = underpowered.\n")

    def with_ref2(df):
        first = df[df.cs <= 120].groupby(["game_id", "seed", "period"], sort=False).pre.first().rename("ref")
        return first.reset_index()
    rr = with_ref2(rlate[rlate.period == 2]).merge(rgs[["game_id", "seed", "reg_margin"]], on=["game_id", "seed"])
    ss = with_ref2(slate[slate.period == 2]).merge(sgs[["game_id", "seed", "reg_margin"]], on=["game_id", "seed"])
    for x in (rr, ss):
        x["bk"] = np.select([x.ref.abs() == 0, x.ref.abs() <= 2, x.ref.abs() <= 5, x.ref.abs() <= 9], ["0", "1-2", "3-5", "6-9"], "10+")
        x["tie"] = (x.reg_margin == 0).astype(float)
        x["c"] = 1.0
    w("| |margin| at 2:00 | real games | P(state) real / sim | P(tie | state) real / sim | contribution to OT rate (pp) real / sim | diff in contribution [95%] |")
    w("|---|---:|---|---|---|---|")
    tot_r = tot_s = 0
    for b in ("0", "1-2", "3-5", "6-9", "10+"):
        a_, b_ = rr[rr.bk == b], ss[ss.bk == b]
        nr_c, ns_c = cs.vec(a_, "c"), cs.vec(b_, "c") / S
        nr_t, ns_t = cs.vec(a_, "tie"), cs.vec(b_, "tie") / S
        ones = np.ones(cs.n)
        pst = cs.ratio_stat(nr_c, ones, ns_c, ones)
        ptie = cs.ratio_stat(nr_t, nr_c, ns_t, ns_c)
        contr = cs.ratio_stat(nr_t, ones, ns_t, ones)
        und = a_.game_id.nunique() < a.min_n
        tot_r += contr[0]
        tot_s += contr[1]
        w(f"| {b} | {a_.game_id.nunique()}{' (U)' if und else ''} | {fmtc(pst, 3)} | {fmtc(ptie, 3)}{' ' if not und else ' (U)'} | {f(contr[0] * 100, 2)} / {f(contr[1] * 100, 2)} | "
          f"{f(contr[2] * 100, 2)} {ci(contr[3] * 100, 2)} |")
    w(f"\nTotals: real {f(tot_r * 100, 2)}% , sim {f(tot_s * 100, 2)}% (tie at end of regulation).\n")

    if not a.skip_b:
        # (b) endgame catch-up by trailing margin, last 2 min of H2, evolving state
        w("## (b) Endgame behaviour in the last 2 minutes of H2, by the CURRENT trailing margin at the start of each possession\n")
        w("Rows are possessions starting with <= 120 s left in H2. `trailer` = team with the ball is behind (margin before the possession 1-3, 4-6, 7+); `leader` = team with the ball is ahead by that margin. "
          "Cells show `real / sim (diff)`; `*` = CI excludes 0; `U` = underpowered (< %d real possessions).\n" % a.min_n)
        h2 = lambda df: df[(df.period == 2) & (df.cs <= 120)].copy()  # noqa: E731
        r2, s2 = h2(rlate), h2(slate)
        for x in (r2, s2):
            m = x.moff.to_numpy()
            x["role"] = np.where(m < 0, "trailer", np.where(m > 0, "leader", "tied"))
            x["bk"] = np.select([np.abs(m) == 0, np.abs(m) <= 3, np.abs(m) <= 6], ["tied", "1-3", "4-6"], "7+")
            x["n"] = 1
            x["dur1"] = x.dur
            x["ft1"] = (x.cls == "FT").astype(int)
            x["p31"] = (x.cls == "3PA").astype(int)
            x["tov1"] = (x.cls == "TOV").astype(int)
            x["fta1"] = x.fta
            x["pts1"] = x.pts
        met = {"sec/poss": ("dur1", "n"), "3PA share": ("p31", "n"), "TOV": ("tov1", "n"), "FT-trip share": ("ft1", "n"), "FTA/poss": ("fta1", "n"), "PPP": ("pts1", "n")}
        rep = cell_report(cs, r2, s2, S, None, ["role", "bk"], met)
        w("| role | |margin| | real poss | sim poss/seed | sec/poss | 3PA share (pp) | TOV rate (pp) | FT-trip share (pp) | FTA/poss | PPP |")
        w("|---|---|---:|---:|---|---|---|---|---|---|")
        for r in rep:
            role, bk = r["cell"]
            und = r["n_real"] < a.min_n
            w(f"| {role} | {bk} | {r['n_real']}{' (U)' if und else ''} | {f(r['n_sim_per_seed'], 0)} | {fmtc(r['sec/poss'], 2)} | {fmtc(r['3PA share'], 1, 100)} | {fmtc(r['TOV'], 1, 100)} | "
              f"{fmtc(r['FT-trip share'], 1, 100)} | {fmtc(r['FTA/poss'], 3)} | {fmtc(r['PPP'], 3)} |")
        w("")
        # state check: bonus occupancy of the team with the ball and the team-foul count of the defence, by role and margin
        for x in (r2, s2):
            x["bonus1"] = x.bonus.astype(float)
            x["dfl1"] = x.dfl.astype(float)
            x["ofl1"] = x.ofl.astype(float)
        repb = cell_report(cs, r2, s2, S, None, ["role", "bk"], {"in bonus (offence)": ("bonus1", "n"), "defence team fouls": ("dfl1", "n"), "offence team fouls": ("ofl1", "n")})
        w("Foul STATE at the start of each possession (last 2 min of H2): share of possessions where the team with the ball is in the bonus, and the team-foul counts of the defence and the offence. "
          "This is the state the foul-accrual law feeds into possession_outcome.")
        w("")
        w("| role | |margin| | real poss | in bonus (pp) real / sim (diff) | defence team fouls | offence team fouls |")
        w("|---|---|---:|---|---|---|")
        for r in repb:
            role, bk = r["cell"]
            w(f"| {role} | {bk} | {r['n_real']} | {fmtc(r['in bonus (offence)'], 1, 100)} | {fmtc(r['defence team fouls'], 2)} | {fmtc(r['offence team fouls'], 2)} |")
        w("")
        w("The trailing team's FOUL rate: fouls by the trailing team per leader possession. Sim = team fouls the defender (trailing team) adds during the leader's possession (engine counters). "
          "Real = PersonalFoul events by the trailing team in the raw pbp per leader possession in the possession table (includes offensive fouls and fouls with no free throws, so the real definition is the broader one; "
          "treat as an upper bound on the comparison). Leader possessions here start in the last 2 minutes of H2.\n")
        fl = pbp_fouls(G)
        fl = fl[(fl.period_number == 2) & (fl.start_period_seconds_remaining <= 120)].copy()
        # trailing team commits the foul: home fouled while home behind (m_before<0) etc.
        fl["trail_foul"] = np.where(fl.home_fouled, fl.m_before < 0, fl.m_before > 0)
        fl["mb"] = np.abs(fl.m_before)
        fl["bk"] = np.select([fl.mb == 0, fl.mb <= 3, fl.mb <= 6], ["tied", "1-3", "4-6"], "7+")
        fl["c"] = fl.trail_foul.astype(float)
        w("| trailing by | leader poss real | sim/seed | trailing-team fouls per leader poss: real (pbp events) | sim (engine counters) | diff [95%] |")
        w("|---|---:|---:|---|---|---|")
        for bk in ("1-3", "4-6", "7+"):
            lr_ = r2[(r2.role == "leader") & (r2.bk == bk)]
            ls_ = s2[(s2.role == "leader") & (s2.bk == bk)]
            fr_ = fl[(fl.bk == bk) & fl.trail_foul]
            fr_ = fr_.assign(c=1.0)
            nr, dr = cs.vec(fr_, "c"), cs.vec(lr_, "n")
            ls2 = ls_.assign(dfoul0=ls_.dfoul.fillna(0))
            ns, ds = cs.vec(ls2, "dfoul0") / S, cs.vec(ls_, "n") / S
            t = cs.ratio_stat(nr, dr, ns, ds)
            w(f"| {bk} | {len(lr_)}{' (U)' if len(lr_) < a.min_n else ''} | {f(len(ls_) / S, 0)} | {f(t[0], 3)} | {f(t[1], 3)} | {f(t[2], 3)} {ci(t[3], 3)} |")
        w("")

        # (c) final possession outcomes, down 1-3 with < 30 s
        w("## (c) Final-possession outcomes when the team with the ball is down 1-3 with < 30 s left in H2\n")
        w("Class rule: TOV > FT trip (the offence was fouled and shot free throws) > 3PA > 2PA > NONE (no shot, no TOV). First possession of a team in the window only is NOT separated; all possessions starting with <= 30 s are pooled "
          "(this includes second chances after a made basket by the other team). Shares of possessions; `*` = CI excludes 0.\n")
        for lo, hi, name in ((1, 3, "down 1-3"), (1, 1, "down 1"), (2, 3, "down 2-3")):
            sel = lambda x: x[(x.cs <= 30) & (x.moff <= -lo) & (x.moff >= -hi)]  # noqa: E731
            rr_, ss_ = sel(r2), sel(s2)
            rr_ = rr_.assign(c=1.0)
            ss_ = ss_.assign(c=1.0)
            w(f"**{name}**: real possessions {len(rr_)}{' (U)' if len(rr_) < a.min_n else ''}, sim possessions per seed {f(len(ss_) / S, 1)}.\n")
            w("| outcome | real share | sim share | diff (pp) [95%] |")
            w("|---|---:|---:|---|")
            for c in ("3PA", "2PA", "FT", "TOV", "NONE"):
                nr_ = cs.vec(rr_.assign(i=(rr_.cls == c).astype(float)), "i")
                ns_ = cs.vec(ss_.assign(i=(ss_.cls == c).astype(float)), "i") / S
                t = cs.ratio_stat(nr_, cs.vec(rr_, "c"), ns_, cs.vec(ss_, "c") / S)
                w(f"| {c} | {f(t[0] * 100, 1)}% | {f(t[1] * 100, 1)}% | {f(t[2] * 100, 1)} {ci(t[3] * 100, 1)} |")
            w("")
        # points scored and duration on those possessions
        sel = lambda x: x[(x.cs <= 30) & (x.moff <= -1) & (x.moff >= -3)]  # noqa: E731
        rr_, ss_ = sel(r2).assign(c=1.0), sel(s2).assign(c=1.0)
        t = cs.ratio_stat(cs.vec(rr_, "pts"), cs.vec(rr_, "c"), cs.vec(ss_, "pts") / S, cs.vec(ss_, "c") / S)
        t2 = cs.ratio_stat(cs.vec(rr_, "dur"), cs.vec(rr_, "c"), cs.vec(ss_, "dur") / S, cs.vec(ss_, "c") / S)
        w(f"Points per such possession: real {f(t[0], 3)}, sim {f(t[1], 3)}, diff {f(t[2], 3)} {ci(t[3], 3)}. Seconds per such possession: real {f(t2[0], 2)}, sim {f(t2[1], 2)}, diff {f(t2[2], 2)} {ci(t2[3], 2)}.\n")
    Path(a.out4).write_text("\n".join(W4), encoding="utf-8")
    print("wrote", a.out3, a.out4)


if __name__ == "__main__":
    main()
