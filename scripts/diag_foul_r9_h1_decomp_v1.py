"""diag_foul_r9_h1_decomp_v1.py -- foul round 9 (lane C, 2026-09-30), step 1: a CLOSED
decomposition of the first-half FTA/FGA overshoot of `ENGINE_FOUL_JOINT=R8b`.

Engine side: per-possession taps written by `scripts/run_foul_joint_tap_v2.py` (v3 inputs,
the verified 500-game sample). Actual side: the 2025 event layer (`possessions_v2` chances
and start types) joined to the CORRECTED (engine-definition) foul state of
`round6/foul_accrual_poss_v2.parquet` (`def_team_fouls_true` = counter before the
possession's own pre-open fouls). Actual and-one fouls are NOT in the replay's
`def_trip` (round-8 amendment, s23.1), so the actual foul count here adds them back:
    actual fouls charged in a possession = def_silent + def_trip + and_ones (defence)
                                          + off_silent + off_trip (offence).

Sections written to the json/txt:
  A  per half x minute bucket, per possession (both sides, plus the sample-only actual)
  B  team-half ACCRUAL identity: mean fouls per team-half = sum of components, and the
     split of the sim - actual gap into PACE (possessions per team-half) and
     PER-POSSESSION rate, per component (closes exactly)
  C  time to the bonus in H1 (first open with defence count >= bonus threshold)
  D  shift-share of the H1 FTA/FGA gap over cells (state x H1 minute bucket):
     occupancy, then and-one FTA, shooting-trip FTA, bonus-trip FTA, other FTA, FGA
     (sequential, both orders; closes exactly by construction; the only approximation
     is the split of a possession's FTA between its trip types, reported as `mixed`)
  E  H1 segments: state, foul type, possession start type, offence site, defence
     prior-season foul-rate quintile; per game and per team summaries
  F  the half reset check (open count at the first H2 possession)

    diag_foul_r9_h1_decomp_v1.py OUT_STEM REF_TAG ARM_TAG [ARM_TAG ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.engine import state as S  # noqa: E402

R = ROOT / "results/engine_v0"
MB = [0, 5, 10, 15, 20, 25, 30, 35, 38, 40.001]
MLAB = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-37", "38-40"]
H1LAB = MLAB[:4]
PREV_NAMES = {v: k for k, v in S.PREV_END_CODE.items()}
MIN_N = 2000      # cells with fewer possessions (actual side) are labelled underpowered


def state_lab(df):
    """pre-bonus / one-and-one / double bonus on the GameState rule-era thresholds."""
    return np.where(df["def_fouls"] >= df["dthr"], "dbl (>=dthr)",
                    np.where(df["def_fouls"] >= df["thr"], "1-and-1", "pre-bonus"))


def actual(sample_ids=None) -> pd.DataFrame:
    c = pd.read_parquet(ROOT / "data/processed/possessions_v2/chances_2025.parquet")
    c["fga"] = c["fga_rim"] + c["fga_jump2"] + c["fga_3"]
    c["sh"] = (c["terminal_event"] == "FT_trip_shooting").astype(int)
    c["bo"] = (c["terminal_event"] == "FT_trip_bonus").astype(int)
    c["ao"] = c["and_one"].astype(int)
    c["fta_sh"] = c["fta"] * c["sh"]
    c["fta_bo"] = c["fta"] * c["bo"]
    c["fta_ao"] = c["fta"] * c["ao"] * (1 - c["sh"]) * (1 - c["bo"])
    k = ["game_id", "period", "poss_index"]
    p = c.groupby(k).agg(fta=("fta", "sum"), fga=("fga", "sum"), sh=("sh", "sum"), bo=("bo", "sum"),
                         ao=("ao", "sum"), fta_sh=("fta_sh", "sum"), fta_bo=("fta_bo", "sum"),
                         fta_ao=("fta_ao", "sum")).reset_index()
    ps = pd.read_parquet(ROOT / "data/processed/possessions_v2/possessions_2025.parquet",
                         columns=k + ["start_reason"])
    d = pd.read_parquet(ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet")
    d = d[d["season"] == 2025]
    m = d.merge(p, on=k, how="inner", validate="one_to_one").merge(ps, on=k, how="left",
                                                                    validate="one_to_one")
    if sample_ids is not None:
        m = m[m["game_id"].isin(sample_ids)]
    out = pd.DataFrame({
        "game_id": m["game_id"].to_numpy(), "rep": 0, "period": m["period"].to_numpy(),
        "sec": m["start_clock"].to_numpy(), "poss_index": m["poss_index"].to_numpy(),
        "def_fouls": m["def_team_fouls_true"].to_numpy(), "off_fouls": m["off_team_fouls_true"].to_numpy(),
        "thr": 6, "dthr": 9,
        "sh": m["sh"].to_numpy(), "bo": m["bo"].to_numpy(), "ao": m["ao"].to_numpy(),
        "def_nt": (m["def_silent"] + m["def_trip"] - m["sh"] - m["bo"]).to_numpy(),
        "off_f": (m["off_silent"] + m["off_trip"]).to_numpy(),
        "fta": m["fta"].to_numpy(), "fga": m["fga"].to_numpy(),
        "fta_sh": m["fta_sh"].to_numpy(), "fta_bo": m["fta_bo"].to_numpy(),
        "fta_ao": m["fta_ao"].to_numpy(),
        "off_team": m["offense_team_id"].to_numpy(), "def_team": m["defense_team_id"].to_numpy(),
        "site": np.where(m["neutral_site"], "neutral", np.where(m["offense_is_home"], "home", "away")),
        "start": m["start_reason"].fillna("other").to_numpy(),
    })
    out["fta_oth"] = out["fta"] - out["fta_sh"] - out["fta_bo"] - out["fta_ao"]
    out["mixed"] = ((out["sh"] > 0) & (out["bo"] > 0)).astype(int)
    return out


def sim(tag: str) -> pd.DataFrame:
    t = pd.read_parquet(R / tag / "poss_tap.parquet")
    g = pd.read_parquet(R / tag / "games_v2.parquet",
                        columns=["game_id", "home_team_id", "away_team_id", "neutral"]).drop_duplicates("game_id")
    t = t.merge(g, on="game_id", how="left", validate="many_to_one")
    ao = t["trip_fouls"] - t["n_shoot_trip"] - t["n_bonus_trip"]
    off_team = np.where(t["off_side"] == 0, t["home_team_id"], t["away_team_id"])
    def_team = np.where(t["off_side"] == 0, t["away_team_id"], t["home_team_id"])
    out = pd.DataFrame({
        "game_id": t["game_id"].to_numpy(), "rep": t["seed"].to_numpy(), "period": t["period"].to_numpy(),
        "sec": t["sec"].to_numpy(), "def_fouls": t["def_fouls"].to_numpy(),
        "off_fouls": t["off_fouls"].to_numpy(), "thr": t["bonus_thr"].to_numpy(),
        "dthr": t["dbonus_thr"].to_numpy(),
        "sh": t["n_shoot_trip"].to_numpy(), "bo": t["n_bonus_trip"].to_numpy(), "ao": ao.to_numpy(),
        "def_nt": t["silent"].to_numpy(), "off_f": t["off_foul"].to_numpy(),
        "fta": t["fta"].to_numpy(), "fga": t["fga"].to_numpy(),
        "off_team": off_team, "def_team": def_team,
        "site": np.where(t["neutral"] > 0, "neutral", np.where(t["off_side"] == 0, "home", "away")),
        "start": pd.Series(t["prev_end"].to_numpy()).map(PREV_NAMES).to_numpy(),
    })
    # split each possession's trip FTA between its trip types. Single-type possessions are
    # exact; a possession holding both a shooting and a bonus trip (`mixed`) is split by the
    # half's mean FTA per trip of each type measured on single-type possessions.
    out["fta_ao"] = out["ao"]
    rest = out["fta"] - out["ao"]
    only_sh = (out["sh"] > 0) & (out["bo"] == 0)
    only_bo = (out["bo"] > 0) & (out["sh"] == 0)
    both = (out["sh"] > 0) & (out["bo"] > 0)
    h = (out["period"] >= 2).astype(int)
    fsh = np.zeros(len(out)); fbo = np.zeros(len(out))
    fsh[only_sh] = rest[only_sh]; fbo[only_bo] = rest[only_bo]
    for hh in (0, 1):
        s1 = only_sh & (h == hh); b1 = only_bo & (h == hh)
        a = rest[s1].sum() / max(out.loc[s1, "sh"].sum(), 1)
        b = rest[b1].sum() / max(out.loc[b1, "bo"].sum(), 1)
        sel = both & (h == hh)
        w = out.loc[sel, "sh"] * a / (out.loc[sel, "sh"] * a + out.loc[sel, "bo"] * b)
        fsh[sel] = rest[sel] * w
        fbo[sel] = rest[sel] * (1 - w)
    out["fta_sh"] = fsh; out["fta_bo"] = fbo
    out["fta_oth"] = out["fta"] - out["fta_sh"] - out["fta_bo"] - out["fta_ao"]
    out["mixed"] = both.astype(int)
    return out


def keys(d: pd.DataFrame) -> pd.DataFrame:
    reg = d["period"] <= 2
    gm = np.where(reg, (d["period"] - 1) * 20 + (1200 - d["sec"]) / 60.0, np.nan)
    d["minute_b"] = pd.cut(gm, MB, right=False, labels=MLAB).astype(str)
    d["half"] = np.where(d["period"] >= 3, "OT", np.where(d["period"] == 1, "H1", "H2"))
    d["state"] = state_lab(d)
    d["in_bonus"] = (d["def_fouls"] >= d["thr"]).astype(int)
    d["fouls_def"] = d["sh"] + d["bo"] + d["ao"] + d["def_nt"]
    d["fouls_all"] = d["fouls_def"] + d["off_f"]
    return d


RATE = ["in_bonus", "sh", "bo", "ao", "def_nt", "off_f", "fouls_all", "fta", "fga"]


def rate_table(d, by):
    g = d.groupby(by, observed=True)
    t = g[RATE].mean()
    t["fta_fga"] = g["fta"].sum() / g["fga"].sum()
    t["n"] = g.size()
    return t


def accrual_identity(d: pd.DataFrame, half: str) -> pd.DataFrame:
    """Per team-half (one row per (rep, game, team)): fouls charged = sum of components."""
    x = d[d["half"] == half]
    dd = x.groupby(["rep", "game_id", "def_team"])[["sh", "bo", "ao", "def_nt"]].sum()
    dd["n_def_poss"] = x.groupby(["rep", "game_id", "def_team"]).size()
    oo = x.groupby(["rep", "game_id", "off_team"])[["off_f"]].sum()
    oo.index.names = ["rep", "game_id", "def_team"]
    t = dd.join(oo, how="outer").fillna(0)
    t["total"] = t[["sh", "bo", "ao", "def_nt", "off_f"]].sum(axis=1)
    return t


def bonus_time(d: pd.DataFrame) -> pd.DataFrame:
    """H1: game minute of the first possession opened with the defence in the bonus."""
    x = d[(d["half"] == "H1")].copy()
    x["minute"] = (1200 - x["sec"]) / 60.0
    x = x.sort_values(["rep", "game_id", "def_team", "minute"])
    inb = x[x["def_fouls"] >= x["thr"]]
    first = inb.groupby(["rep", "game_id", "def_team"])["minute"].min()
    allk = x.groupby(["rep", "game_id", "def_team"]).size().index
    return first.reindex(allk)


def shift_share(A: pd.DataFrame, B: pd.DataFrame, order: list[str]) -> dict:
    """H1 FTA/FGA: A = actual, B = sim. Cells = state x H1 minute bucket."""
    cell = ["state", "minute_b"]
    a = A[A["half"] == "H1"]; b = B[B["half"] == "H1"]
    comp = ["fta_ao", "fta_sh", "fta_bo", "fta_oth", "fga"]
    ra = a.groupby(cell)[comp].mean()
    rb = b.groupby(cell)[comp].mean()
    wa = a.groupby(cell).size() / len(a)
    wb = b.groupby(cell).size() / len(b)
    idx = ra.index.union(rb.index)
    ra, rb = ra.reindex(idx).fillna(0), rb.reindex(idx).fillna(0)
    wa, wb = wa.reindex(idx).fillna(0), wb.reindex(idx).fillna(0)

    def ratio(w, r):
        fta = (w * (r["fta_ao"] + r["fta_sh"] + r["fta_bo"] + r["fta_oth"])).sum()
        return float(fta / (w * r["fga"]).sum())

    steps = []
    w, r = wa.copy(), ra.copy()
    cur = ratio(w, r)
    start = cur
    for s in order:
        if s == "occupancy":
            w = wb.copy()
        else:
            r[s] = rb[s]
        nxt = ratio(w, r)
        steps.append({"step": s, "delta": nxt - cur})
        cur = nxt
    return {"order": order, "actual": start, "sim": cur, "gap": cur - start, "steps": steps,
            "check_sim_direct": ratio(wb, rb)}


def prior_quintile_def() -> pd.Series:
    """Defence prior-season (2024) fouls committed per defensive possession, quintile 1..5."""
    d = pd.read_parquet(ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet",
                        columns=["season", "defense_team_id", "offense_team_id", "def_silent", "def_trip",
                                 "off_silent", "off_trip"])
    d = d[d["season"] == 2024]
    dfn = d.groupby("defense_team_id")[["def_silent", "def_trip"]].sum().sum(axis=1)
    n = d.groupby("defense_team_id").size()
    r = (dfn / n)[n >= 1000]
    return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def seg(d, by, half="H1"):
    x = d[d["half"] == half]
    g = x.groupby(by, observed=True)
    t = g[["in_bonus", "sh", "bo", "ao", "def_nt", "off_f", "fouls_all"]].mean()
    t["fta_fga"] = g["fta"].sum() / g["fga"].sum()
    t["n"] = g.size()
    return t


def main() -> None:
    stem = Path(sys.argv[1])
    tags = sys.argv[2:]
    sims = {t: keys(sim(t)) for t in tags}
    ids = sims[tags[0]]["game_id"].unique()
    A_all = keys(actual())
    A = keys(actual(ids))
    n_reps = {t: int(s["rep"].nunique()) for t, s in sims.items()}
    out: dict = {"tags": tags, "n_reps": n_reps, "n_games_sample": int(len(ids)),
                 "actual_sample_poss": int(len(A)), "actual_all_poss": int(len(A_all))}
    lines: list[str] = []
    pd.set_option("display.width", 250)

    def show(title, tb):
        lines.append(f"\n=== {title} ===\n{tb.round(4).to_string()}")
        print(lines[-1])

    frames = {"ACT_sample": A, "ACT_all2025": A_all, **sims}
    # A: half and minute tables
    for by in ("half", "minute_b"):
        out[f"A_{by}"] = {}
        for k, d in frames.items():
            tb = rate_table(d, by)
            out[f"A_{by}"][k] = tb.reset_index().to_dict(orient="records")
            show(f"A {by} :: {k}", tb)
    # B: accrual identity, H1 and H2
    out["B"] = {}
    for half in ("H1", "H2"):
        rows = {}
        for k, d in frames.items():
            t = accrual_identity(d, half)
            rows[k] = t.mean()
        tb = pd.DataFrame(rows).T
        for c in ("sh", "bo", "ao", "def_nt", "off_f", "total"):
            tb[f"{c}_pp"] = tb[c] / tb["n_def_poss"]
        show(f"B accrual identity per team-half {half} (mean counts; *_pp per defensive possession)", tb)
        out["B"][half] = tb.reset_index().to_dict(orient="records")
        # pace vs per-possession split for each sim tag vs the sample actual
        a = tb.loc["ACT_sample"]
        for k in tags:
            b = tb.loc[k]
            sp = {}
            for c in ("sh", "bo", "ao", "def_nt", "off_f", "total"):
                pace = (b["n_def_poss"] - a["n_def_poss"]) * a[f"{c}_pp"]
                rate = b["n_def_poss"] * (b[f"{c}_pp"] - a[f"{c}_pp"])
                sp[c] = {"gap": float(b[c] - a[c]), "pace": float(pace), "per_poss": float(rate),
                         "check": float(pace + rate - (b[c] - a[c]))}
            out["B"][f"{half}_split_{k}"] = sp
            lines.append(f"  {half} {k} gap split (pace / per-poss): " + "; ".join(
                f"{c} {v['gap']:+.3f} = {v['pace']:+.3f} + {v['per_poss']:+.3f}" for c, v in sp.items()))
            print(lines[-1])
    # C: time to bonus
    out["C"] = {}
    for k, d in frames.items():
        bt = bonus_time(d)
        q = {"p_reach_bonus_H1": float(bt.notna().mean()),
             "mean_minute_if_reached": float(bt.mean()),
             "q25": float(bt.quantile(0.25)), "q50": float(bt.quantile(0.5)), "q75": float(bt.quantile(0.75))}
        for m in (10, 12, 14, 16, 18):
            q[f"p_in_bonus_by_min{m}"] = float((bt <= m).mean())
        out["C"][k] = q
    show("C H1 time to bonus (minute of first possession opened in the bonus, per team-half)",
         pd.DataFrame(out["C"]).T)
    # D: shift-share
    out["D"] = {}
    base_order = ["occupancy", "fta_ao", "fta_sh", "fta_bo", "fta_oth", "fga"]
    for k in tags:
        f = shift_share(A, sims[k], base_order)
        r = shift_share(A, sims[k], list(reversed(base_order)))
        out["D"][k] = {"forward": f, "reverse": r}
        tb = pd.DataFrame({"forward": {s["step"]: s["delta"] for s in f["steps"]},
                           "reverse": {s["step"]: s["delta"] for s in r["steps"]}})
        tb["mean"] = tb.mean(axis=1)
        tb.loc["TOTAL"] = tb.sum()
        show(f"D H1 FTA/FGA shift-share {k}: actual {f['actual']:.4f} -> sim {f['sim']:.4f} "
             f"(direct {f['check_sim_direct']:.4f})", tb)
    # mixed-trip share (the only approximation in D)
    out["mixed_share"] = {k: float(d.loc[d["half"] == "H1", "mixed"].mean()) for k, d in frames.items()}
    # E: segments
    q = prior_quintile_def()
    for k, d in frames.items():
        d["def_q"] = d["def_team"].map(q).fillna(0).astype(int)
    for by in (["state"], ["state", "minute_b"], ["start"], ["site"], ["def_q"]):
        name = "x".join(by)
        out[f"E_{name}"] = {}
        for k, d in frames.items():
            tb = seg(d, by)
            out[f"E_{name}"][k] = tb.reset_index().to_dict(orient="records")
            show(f"E H1 by {name} :: {k}", tb)
    # E: per game and per team (H1 FTA/FGA, sim mean over reps vs actual on the sample)
    def per(d, key):
        x = d[d["half"] == "H1"]
        g = x.groupby(["rep", "game_id", key])[["fta", "fga"]].sum().reset_index()
        g = g.groupby(["game_id", key])[["fta", "fga"]].mean()
        return g["fta"] / g["fga"].clip(lower=1)
    for key, lab in (("off_team", "per game x offence team (H1 FTA/FGA)"),):
        a = per(A, key)
        out[f"E_pergame"] = {}
        for k in tags:
            b = per(sims[k], key).reindex(a.index)
            ok = b.notna()
            out["E_pergame"][k] = {"bias": float((b - a)[ok].mean()), "mae": float((b - a)[ok].abs().mean()),
                                   "corr": float(np.corrcoef(a[ok], b[ok])[0, 1]), "n": int(ok.sum())}
        show(lab, pd.DataFrame(out["E_pergame"]).T)
    # F: reset check, open count of each side at the first H2 possession
    out["F"] = {}
    for k, d in frames.items():
        x = d[d["half"] == "H2"].sort_values(["rep", "game_id", "sec"], ascending=[True, True, False])
        f = x.groupby(["rep", "game_id"]).head(1)
        out["F"][k] = {"mean_def_fouls_first_H2_poss": float(f["def_fouls"].mean()),
                       "mean_off_fouls_first_H2_poss": float(f["off_fouls"].mean()),
                       "share_nonzero": float(((f["def_fouls"] > 0) | (f["off_fouls"] > 0)).mean())}
    show("F reset at half (open counts at the first H2 possession)", pd.DataFrame(out["F"]).T)
    stem.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{stem}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    Path(f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {stem}.json/.txt")


if __name__ == "__main__":
    main()
