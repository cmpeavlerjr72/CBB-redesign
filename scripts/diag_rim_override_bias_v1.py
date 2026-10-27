#!/usr/bin/env python
"""diag_rim_override_bias_v1.py -- rim-override make-bias diagnostic (PM ruling 17c, 2026-10-27). DESCRIPTIVE.

The possessions v2 override reclassifies a two-point JumpShot whose recorded release point is within 2.27 ft of a
basket as FGA_rim. Round 1 found it fires more on makes than misses within game. Measured here, from the
coordinate-audit master tables (`cls0` = v1 label, no override; `cls` = v2 label, override applied):

  1. rim and jump2 make rates per season under v1 vs v2 labels (ALL attempts; unlocated rows are untouched by the
     override) and the shift the override causes in each class (game-bootstrap SE).
  2. share of overridden rows that are makes vs the share of all v1-jump2 rows that are makes, by season and by
     distance band (0-1, 1-2, 2-2.27 ft); reference rows: native located rim attempts in the same bands, and
     located v1-jump2 rows just outside (2.27-3.5, 3.5-5 ft). The 0.39 ft spike pixel is split out.
  3. overridden make share and the make-vs-miss override rates by located-share-of-game quartile (per season).
  4. the SERVED fg_make recipe (round4_B1 + g9 wrapper, `ENGINE_FG_MAKE` default; fold-2 predictions of the PAE
     round's reference run R_s0_F2, no retrain) decile-calibrated separately on overridden rows vs native rim rows
     (rim class) and on jump2 rows by distance band (jump2 class).
  5. which possessions version the engine consumes (static trace, printed in the report).

    .venv/Scripts/python.exe scripts/diag_rim_override_bias_v1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

AUDIT = ROOT / "results/coordinate_audit"
DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
PREDS = ROOT / "data/processed/models/fg_make/round_pae/R_s0_F2/preds_F2_s0.parquet"
SEASONS = [2022, 2023, 2024, 2025]
N_BOOT = 400
SPIKE = 0.39
REP: dict = {}
BANDS = [(0.0, 1.0, "0-1"), (1.0, 2.0, "1-2"), (2.0, 2.27001, "2-2.27")]
REF_BANDS = [(2.27001, 3.5, "2.27-3.5"), (3.5, 5.0, "3.5-5")]


def log(m, t0):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def boot_ratio(game, num, den, rng, n=N_BOOT):
    """Game-block bootstrap SE of sum(num)/sum(den)."""
    g = pd.DataFrame({"g": game, "n": num, "d": den}).groupby("g").sum()
    nn, dd = g["n"].to_numpy(float), g["d"].to_numpy(float)
    if len(nn) < 2 or dd.sum() == 0:
        return float("nan")
    est = []
    for _ in range(n):
        k = rng.integers(0, len(nn), len(nn))
        est.append(nn[k].sum() / max(dd[k].sum(), 1e-12))
    return float(np.std(est, ddof=1))


def load_master(s):
    m = pd.read_parquet(AUDIT / f"master_{s}.parquet", columns=[
        "gameId", "period", "secondsRemaining", "teamId", "shot_shooter_id", "cls", "cls0", "made", "located", "dist",
        "game_date"])
    m["season"] = s
    m["ovr"] = (m["cls"] == "rim") & (m["cls0"] != "rim")
    m["made"] = m["made"].astype(float)
    return m


# -------------------------------------------------------------------------------------------- parts 1-3
def parts123(t0):
    rng = np.random.default_rng(20261027)
    p1, p2, p3 = [], [], []
    for s in SEASONS:
        m = load_master(s)
        # 1. make rates under v1 vs v2 labels
        for lab in ("rim", "jump2"):
            v1 = m["cls0"] == lab
            v2 = m["cls"] == lab
            r1, r2 = m.loc[v1, "made"].mean(), m.loc[v2, "made"].mean()
            # game-bootstrap SE of (rate_v2 - rate_v1): per game num/den sums
            g = m.assign(a1=v1 * m["made"], n1=v1.astype(float), a2=v2 * m["made"], n2=v2.astype(float)) \
                .groupby("gameId")[["a1", "n1", "a2", "n2"]].sum()
            ks = []
            for _ in range(N_BOOT):
                k = rng.integers(0, len(g), len(g))
                x = g.to_numpy()[k].sum(0)
                ks.append(x[2] / x[3] - x[0] / x[1])
            p1.append({"season": s, "class": lab, "n_v1": int(v1.sum()), "n_v2": int(v2.sum()),
                       "make_v1": float(r1), "make_v2": float(r2), "shift_pp": float(100 * (r2 - r1)),
                       "boot_se_pp": float(100 * np.std(ks, ddof=1))})
        # 2. overridden vs all v1-jump2 makes, by distance band
        j = m[(m["cls0"] == "jump2") & m["located"]]
        ov = j[j["ovr"]]
        rows = [{"season": s, "band": "all located v1-jump2", "n": int(len(j)), "make_share": float(j["made"].mean()),
                 "kind": "jump2_all"},
                {"season": s, "band": "overridden (0-2.27)", "n": int(len(ov)), "make_share": float(ov["made"].mean()),
                 "boot_se": boot_ratio(ov["gameId"], ov["made"], np.ones(len(ov)), rng), "kind": "override"}]
        nat = m[(m["cls0"] == "rim") & m["located"]]
        sp = ov[np.isclose(ov["dist"], SPIKE, atol=0.02)]
        rows.append({"season": s, "band": "overridden at spike pixel (0.39 ft)", "n": int(len(sp)),
                     "share_of_overridden": float(len(sp) / max(len(ov), 1)), "make_share": float(sp["made"].mean()),
                     "kind": "override_spike"})
        ns = ov[~np.isclose(ov["dist"], SPIKE, atol=0.02)]
        rows.append({"season": s, "band": "overridden not at spike", "n": int(len(ns)),
                     "make_share": float(ns["made"].mean()), "kind": "override_nonspike"})
        for lo, hi, nm in BANDS:
            a = ov[(ov["dist"] >= lo) & (ov["dist"] < hi)]
            b = nat[(nat["dist"] >= lo) & (nat["dist"] < hi)]
            rows.append({"season": s, "band": nm, "n": int(len(a)), "make_share": float(a["made"].mean()) if len(a) else None,
                         "boot_se": boot_ratio(a["gameId"], a["made"], np.ones(len(a)), rng) if len(a) > 5 else None,
                         "kind": "override_band",
                         "native_rim_n": int(len(b)), "native_rim_make_share": float(b["made"].mean()) if len(b) else None})
        for lo, hi, nm in REF_BANDS:
            a = j[(j["dist"] >= lo) & (j["dist"] < hi)]
            rows.append({"season": s, "band": nm + " (native v1/v2 jump2, not overridden)", "n": int(len(a)),
                         "make_share": float(a["made"].mean()), "kind": "ref_jump2"})
        p2 += rows
        # 3. by located-share-of-game quartile (per season)
        gl = m.groupby("gameId")["located"].mean().rename("loc_share")
        mm = m.merge(gl, on="gameId")
        q = pd.qcut(mm["loc_share"].rank(method="first"), 4, labels=False)
        mm["q"] = q
        for qq in range(-1, 4):              # -1 = all games of the season (two-point JumpShot rows only)
            x = mm if qq < 0 else mm[mm["q"] == qq]
            jj = x[(x["cls0"] == "jump2") & x["located"]]
            ovq = jj[jj["ovr"]]
            mk, ms = jj[jj["made"] == 1], jj[jj["made"] == 0]
            # within-game makes-minus-misses override rate (games with both)
            gg = jj.groupby(["gameId", "made"])["ovr"].agg(["sum", "size"]).unstack("made").dropna()
            if len(gg):
                dd = (gg[("sum", 1.0)] / gg[("size", 1.0)] - gg[("sum", 0.0)] / gg[("size", 0.0)]).to_numpy()
                w = (gg[("size", 1.0)] + gg[("size", 0.0)]).to_numpy()
                est = float(np.average(dd, weights=w))
                bs = [np.average(dd[k], weights=w[k]) for k in (rng.integers(0, len(dd), len(dd)) for _ in range(N_BOOT))]
                diff, se = 100 * est, 100 * float(np.std(bs, ddof=1))
            else:
                diff = se = None
            p3.append({"season": s, "quartile": qq + 1, "games": int(x["gameId"].nunique()),
                       "mean_loc_share": float(x["located"].mean()), "located_jumpshots": int(len(jj)),
                       "overridden": int(len(ovq)), "override_make_share": float(ovq["made"].mean()) if len(ovq) else None,
                       "override_pct_of_makes": float(100 * mk["ovr"].mean()) if len(mk) else None,
                       "override_pct_of_misses": float(100 * ms["ovr"].mean()) if len(ms) else None,
                       "within_game_diff_pp": diff, "within_game_boot_se_pp": se})
        log(f"{s} parts 1-3 done", t0)
    REP["part1_make_rates_v1_vs_v2"] = p1
    REP["part2_override_make_share"] = p2
    REP["part3_by_located_share_quartile"] = p3


# -------------------------------------------------------------------------------------------- part 4
def decile(y, p, n=10):
    o = np.argsort(p, kind="stable")
    rows = []
    for ix in np.array_split(o, n):
        rows.append((float(p[ix].mean()), float(y[ix].mean()), int(len(ix))))
    pm = np.array([r[0] for r in rows])
    am = np.array([r[1] for r in rows])
    gap = 100 * (am - pm)
    lvl = float(100 * (y.mean() - p.mean()))
    return {"n": int(len(y)), "pred": float(p.mean()), "actual": float(y.mean()), "level_pp": lvl,
            "worst_decile_gap_pp": float(gap[np.argmax(np.abs(gap))]), "shape_pp": float(np.abs(gap - lvl).max()),
            "deciles": [{"pred": r[0], "actual": r[1], "n": r[2]} for r in rows]}


def part4(t0):
    des = pd.read_parquet(DESIGN)
    _, te = FG.fold_slices(des, "F2")
    pr = pd.read_parquet(PREDS)
    out = {}
    m = load_master(2025)
    m["shooter"] = m["shot_shooter_id"]
    for cname, key in (("FGA_rim", "rim"), ("FGA_jump2", "jump2")):
        t = FG.class_slice(te, cname).reset_index(drop=True)
        pc = pr[pr["shot_class"] == cname].reset_index(drop=True)
        assert len(pc) == len(t), (len(pc), len(t))
        for k in ("game_id", "shooter_id", "y"):
            assert (pc[k].to_numpy() == t[k].to_numpy()).all(), k
        t["p"] = pc["p"].to_numpy()
        # attach master rows: key (cbbd game, period, seconds, shooter, v2 class, made) (team ids differ between sources) + ordinal
        t["k"] = list(zip(t["cbbd_game_id"].astype("int64"), t["period"].astype("int64"),
                          t["seconds_remaining"].astype("int64"), t["shooter_id"].astype("int64"),
                          t["y"].astype("int64")))
        t["ord"] = t.groupby("k").cumcount()
        mm = m[m["cls"] == key].copy()
        mm["k"] = list(zip(mm["gameId"].astype("int64"), mm["period"].astype("int64"),
                           mm["secondsRemaining"].astype("int64"), mm["shot_shooter_id"].fillna(-1).astype("int64"),
                           mm["made"].astype("int64")))
        mm["ord"] = mm.groupby("k").cumcount()
        J = t.reset_index().merge(mm[["k", "ord", "ovr", "located", "dist", "cls0"]], on=["k", "ord"], how="left")
        matched = J["located"].notna()
        rep = {"design_rows": int(len(t)), "matched": int(matched.sum()), "matched_pct": float(100 * matched.mean()),
               "master_rows_this_class": int(len(mm))}
        J["ovr"] = J["ovr"].fillna(False).astype(bool)
        J["located"] = J["located"].fillna(False).astype(bool)
        y, p = J["y"].to_numpy(float), J["p"].to_numpy(float)
        groups = {"all": np.ones(len(J), bool)}
        if key == "rim":
            groups["overridden (v1 jump2)"] = J["ovr"].to_numpy()
            groups["native rim, located"] = (~J["ovr"] & J["located"]).to_numpy()
            groups["native rim, unlocated/unmatched"] = (~J["located"]).to_numpy()
            groups["native rim, located, dist<2.27"] = (~J["ovr"] & J["located"] & (J["dist"] < 2.27001)).to_numpy()
            groups["overridden at spike pixel"] = (J["ovr"] & np.isclose(J["dist"], SPIKE, atol=0.02)).to_numpy()
            groups["overridden not at spike"] = (J["ovr"] & ~np.isclose(J["dist"], SPIKE, atol=0.02)).to_numpy()
        else:
            d = J["dist"].to_numpy(float)
            lo = J["located"].to_numpy()
            groups["located, 2.27-3.5 ft"] = lo & (d >= 2.27001) & (d < 3.5)
            groups["located, 3.5-5 ft"] = lo & (d >= 3.5) & (d < 5)
            groups["located, >5 ft"] = lo & (d >= 5)
            groups["unlocated/unmatched"] = ~lo
        rep["groups"] = {}
        gid = J["game_id"].to_numpy()
        for nm, mk in groups.items():
            if mk.sum() < 200:
                rep["groups"][nm] = {"n": int(mk.sum()), "note": "UNDERPOWERED (< 200 rows)"}
                continue
            r = decile(y[mk], p[mk])
            # game-block bootstrap of the level shift
            diff = (y - p)
            r["level_boot_se_pp"] = 100 * boot_ratio(gid[mk], diff[mk], np.ones(int(mk.sum())), np.random.default_rng(3))
            r["log_loss"] = float(-np.mean(np.where(y[mk] == 1, np.log(np.clip(p[mk], 1e-12, 1)), np.log(np.clip(1 - p[mk], 1e-12, 1)))))
            rep["groups"][nm] = r
        out[key] = rep
        log(f"4 {key}: matched {rep['matched_pct']:.2f}%", t0)
    REP["part4_served_fg_make_calibration_F2"] = out


# -------------------------------------------------------------------------------------------- part 5
def part5():
    from cbb_sim.pbp import possessions as P
    from cbb_sim.pbp import events as E
    d = pd.read_parquet(DESIGN, columns=["season"]).shape
    import pyarrow.parquet as pq
    md = pq.read_metadata(DESIGN).metadata
    attrs = json.loads(md[b"PANDAS_ATTRS"].decode()) if b"PANDAS_ATTRS" in md else {}
    REP["part5_engine_version"] = {
        "DEFAULT_POSSESSION_VERSION": P.DEFAULT_POSSESSION_VERSION,
        "events.RIM_OVERRIDE_MAX_FT (classify_frame default)": E.RIM_OVERRIDE_MAX_FT,
        "fg_make.DEFAULT_VERSION": FG.DEFAULT_VERSION,
        "fg_make design_v2_shotshooter.parquet attrs": attrs,
        "note": "see report text: the served engine names v2 explicitly in fg_make (design + engine inputs) and in the "
                "possession_outcome round-2 team block (build_engine_event_round2.py); DEFAULT_POSSESSION_VERSION only "
                "binds callers that name no version."}


def main() -> int:
    t0 = time.time()
    assert_not_sealed(SEASONS, context="rim override diag")
    REP["created_at"] = pd.Timestamp.now("UTC").isoformat()
    parts123(t0)
    part4(t0)
    part5()
    out = ROOT / "docs/tests/rim_override_make_bias_2026-10-27.json"
    out.write_text(json.dumps(REP, indent=2, default=float), encoding="utf-8")
    log(f"wrote {out}", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
