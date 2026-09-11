#!/usr/bin/env python
"""
diag_fg_make_state_confound.py -- what `score_diff` really does to field-goal
make probability, and how much of it the engine is allowed to see.

    .venv/Scripts/python.exe scripts/diag_fg_make_state_confound.py \
        --design <cached design parquet>

This is the EVIDENCE step that `docs/LEARNINGS.md` L23 and
`ARCHITECTURE_DECISIONS.md` Decision 10 ask for before fg_make round 2 is
pre-registered. It fits nothing that ships and adopts nothing.

It runs in two parts.

PART 0 -- THE LEAK. `fg_make`'s `score_diff` is read off `homeScore`/`awayScore`
on the attempt's OWN row, and in the CBBD/ESPN feed that column is the score
AFTER the play: a made three already carries its own three points. The feature
is therefore post-outcome, in the same family as the `blocked` and `and_one`
flags the model bans by name. Part 0 proves it from the raw feed, independently
of the model, and measures how much of the apparent effect it manufactures.

PART 1 -- THE REAL EFFECT, decomposed. With the attempt's own points removed
(`score_diff_pre`), the surviving relationship is decomposed into
  (a) a PREGAME TEAM-STRENGTH confound (the leading team is the better team),
  (b) a GARBAGE-TIME effect (big margin, little time: bench lineups), and
  (c) an END-GAME TACTICAL effect (trailing late: threes and rushed attempts),
each per shot class, per margin bucket, per minute bucket, per team quintile
and per chance type.

Method for (a): g-computation on nested logistic models. Every row is
counterfactually assigned to each margin bucket in turn and the mean predicted
make probability is reported; the SPAN of that curve is the model-implied
`score_diff` effect. Nesting pregame blocks in and watching the span collapse
IS the confound measurement.

    M0  margin bucket only                      (the marginal effect)
    M1  + pregame own ratings of BOTH teams, site, season index
    M2  + the two teams' as-of form on this shot class
    M3  + the shooter block (identity, prior season, position, exposure)
    M4  + the defensive five's as-of allowed rates   (2024+ rows only, part b)

Output: `data/processed/models/fg_make/state_confound.json`; the markdown
report built from it is `docs/tests/fg_make_state_confound_2026-09-10.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

OUT_JSON = Path("data/processed/models/fg_make/state_confound.json")
PBP_DIR = Path("data/raw/cbbd/pbp")
SEASONS = [2022, 2023, 2024, 2025]

MARGIN_BINS = [-np.inf, -19.5, -9.5, -3.5, 3.5, 9.5, 19.5, np.inf]
MARGIN_LABELS = ["<= -20", "-19..-10", "-9..-4", "-3..+3", "+4..+9",
                 "+10..+19", ">= +20"]
REF_MARGIN = "-3..+3"

MINUTE_LABELS = ["H1 20-10", "H1 10-0", "H2 20-10", "H2 10-5",
                 "H2 5-2", "H2 2-0", "OT"]

RATING_BLOCK = ["off_rating_off_c", "off_rating_def_c",
                "def_rating_off_c", "def_rating_def_c",
                "site_home", "site_away", "season_idx"]
FORM_BLOCK = ["off_make_c", "def_allow_c"]
SHOOTER_BLOCK = ["shooter_make_c", "shooter_att_c", "prior_season_make_c",
                 "has_prior_season", "pos_G", "pos_F", "pos_C",
                 "shooter_games_asof", "shooter_fga_asof"]
LINEUP_BLOCK = ["def5_rim_allow_c", "def5_three_allow_c"]

MIN_CELL = 2000          # cells below this are LABELLED UNDERPOWERED


# ---------------------------------------------------------------------------
# part 0: the leak, proved off the raw feed
# ---------------------------------------------------------------------------
def leak_check(seasons: list[int]) -> dict:
    """Is the feed's running score POST-play on the attempt's own row?

    Independent of the model and of `event_stream`: it reads the raw CBBD plays
    table and measures, on the attempt's own row, how much the shooting team's
    own score moved relative to the row before it. If the column were pre-play
    the move would be zero on makes as it is on misses."""
    out = {}
    for s in seasons:
        d = pd.read_parquet(PBP_DIR / f"plays_{s}.parquet",
                            columns=["gameId", "playType", "homeScore", "awayScore",
                                     "scoringPlay", "scoreValue", "isHomeTeam"])
        hs = pd.to_numeric(d["homeScore"], errors="coerce").ffill().fillna(0).to_numpy()
        as_ = pd.to_numeric(d["awayScore"], errors="coerce").ffill().fillna(0).to_numpy()
        g = d["gameId"].to_numpy()
        same = np.concatenate([[False], g[1:] == g[:-1]])
        dh = np.where(same, np.concatenate([[0], np.diff(hs)]), 0)
        da = np.where(same, np.concatenate([[0], np.diff(as_)]), 0)
        ih = d["isHomeTeam"].map({True: True, False: False}).fillna(False).to_numpy().astype(bool)
        own = np.where(ih, dh, da)
        sp = d["scoringPlay"].fillna(False).astype(bool).to_numpy()
        sv = pd.to_numeric(d["scoreValue"], errors="coerce").fillna(0).to_numpy()
        pt = d["playType"].astype(str)
        fg = pt.str.contains("Shot").to_numpy() & ~pt.str.contains("Free").to_numpy() \
            & ~pt.str.contains("Block").to_numpy()
        row = {}
        for lab, mask in (("made_fg", fg & sp), ("missed_fg", fg & ~sp)):
            o, v = own[mask], sv[mask]
            row[lab] = {
                "n": int(mask.sum()),
                "mean_own_score_delta_on_own_row": round(float(o.mean()), 4),
                "pct_delta_equals_shot_value": round(float((o == v).mean() * 100), 3),
                "pct_delta_zero": round(float((o == 0).mean() * 100), 3),
            }
        out[str(s)] = row
    return out


def add_pre_shot_margin(ev: pd.DataFrame) -> pd.DataFrame:
    """`score_diff_pre`: the shooting team's margin BEFORE its own attempt.

    The attempt's own points are already in the feed's score on its own row
    (`leak_check`), so removing them is the correction. A miss is unchanged by
    construction, which is why the correction cannot itself leak: it is a
    function of the row's own outcome ONLY in the direction that removes the
    outcome."""
    ev = ev.copy()
    pts = np.where(ev["shot_class"].to_numpy() == "FGA_3", 3, 2)
    own = np.where(ev["made"].to_numpy().astype(bool), pts, 0)
    ev["score_diff_post"] = ev["score_diff"].astype("float32")
    ev["score_diff_pre"] = (ev["score_diff"].to_numpy() - own).astype("float32")
    return ev


# ---------------------------------------------------------------------------
# buckets
# ---------------------------------------------------------------------------
def game_seconds_remaining(period: np.ndarray, sec: np.ndarray) -> np.ndarray:
    g = np.where(period <= 1, 1200.0 + sec, sec)
    return np.where(period >= 3, sec, g)


def minute_bucket(period: np.ndarray, gsr: np.ndarray) -> np.ndarray:
    out = np.full(len(gsr), "OT", dtype=object)
    reg = period <= 2
    out[reg & (gsr > 1800)] = "H1 20-10"
    out[reg & (gsr > 1200) & (gsr <= 1800)] = "H1 10-0"
    out[reg & (gsr > 600) & (gsr <= 1200)] = "H2 20-10"
    out[reg & (gsr > 300) & (gsr <= 600)] = "H2 10-5"
    out[reg & (gsr > 120) & (gsr <= 300)] = "H2 5-2"
    out[reg & (gsr <= 120)] = "H2 2-0"
    return out


def prep(d: pd.DataFrame, margin_col: str = "score_diff_pre") -> pd.DataFrame:
    d = d.copy()
    per = d["period"].to_numpy(float)
    sec = d["seconds_remaining"].to_numpy(float)
    d["gsr"] = game_seconds_remaining(per, sec)
    d["minute_bucket"] = minute_bucket(per, d["gsr"].to_numpy())
    d["margin_bucket"] = pd.cut(d[margin_col].to_numpy(float), MARGIN_BINS,
                                labels=MARGIN_LABELS).astype(object)
    d["abs_margin"] = np.abs(d[margin_col].to_numpy(float))
    d["net_rating_c"] = (d["off_rating_off_c"].to_numpy(float)
                         - d["off_rating_def_c"].to_numpy(float))
    return d


# ---------------------------------------------------------------------------
# nested g-computation
# ---------------------------------------------------------------------------
def _fit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-6 * np.maximum(np.abs(mu), 1.0), sd, 1.0)
    clf = LogisticRegression(C=1e6, max_iter=400, solver="lbfgs")
    clf.fit((X - mu) / sd, y)
    return clf, mu, sd


def _dummies(bucket: np.ndarray, labels: list[str]) -> np.ndarray:
    return np.column_stack([(bucket == b).astype(float)
                            for b in labels if b != REF_MARGIN])


def gcomp_curve(d: pd.DataFrame, blocks: list[str],
                labels: list[str]) -> tuple[dict, float]:
    y = d["y"].to_numpy(np.int8)
    D = _dummies(d["margin_bucket"].to_numpy(), labels)
    Z = d[blocks].to_numpy(np.float64) if blocks else np.empty((len(d), 0))
    clf, mu, sd = _fit(np.hstack([D, Z]), y)
    free = [b for b in labels if b != REF_MARGIN]
    curve = {}
    for b in labels:
        Dc = np.zeros_like(D)
        if b != REF_MARGIN:
            Dc[:, free.index(b)] = 1.0
        p = clf.predict_proba((np.hstack([Dc, Z]) - mu) / sd)[:, 1]
        curve[b] = float(p.mean() * 100.0)
    return curve, max(curve.values()) - min(curve.values())


def block_table(d: pd.DataFrame, nested: list[tuple[str, list[str]]],
                labels: list[str]) -> dict:
    out = {"n": int(len(d)), "actual_make_pct": float(d["y"].mean() * 100.0),
           "curves": {}, "spans": {}}
    for name, blocks in nested:
        c, s = gcomp_curve(d, blocks, labels)
        out["curves"][name] = {k: round(v, 3) for k, v in c.items()}
        out["spans"][name] = round(s, 4)
    return out


def indicator_residual(d: pd.DataFrame, flag: np.ndarray,
                       nested: list[tuple[str, list[str]]]) -> dict:
    """Average marginal effect of a 0/1 indicator under each nesting, in pp."""
    y = d["y"].to_numpy(np.int8)
    g = flag.astype(float)[:, None]
    row = {"n_flag": int(g.sum()), "underpowered": bool(g.sum() < MIN_CELL)}
    for name, blocks in nested:
        Z = d[blocks].to_numpy(np.float64) if blocks else np.empty((len(d), 0))
        clf, mu, sd = _fit(np.hstack([g, Z]), y)
        p1 = clf.predict_proba((np.hstack([np.ones_like(g), Z]) - mu) / sd)[:, 1].mean()
        p0 = clf.predict_proba((np.hstack([np.zeros_like(g), Z]) - mu) / sd)[:, 1].mean()
        row[name] = round(float((p1 - p0) * 100.0), 4)
    return row


def raw_by_bucket(d: pd.DataFrame, col: str, labels: list[str]) -> dict:
    g = d.groupby(col, observed=True)["y"]
    sz, mn = g.size(), g.mean()
    return {b: {"n": int(sz.get(b, 0)),
                "make_pct": round(float(mn.get(b, np.nan) * 100.0), 3),
                "underpowered": bool(int(sz.get(b, 0)) < MIN_CELL)}
            for b in labels}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="")
    ap.add_argument("--skip-leak", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    if a.design and Path(a.design).exists():
        design = pd.read_parquet(a.design)
        print(f"loaded cached design {design.shape}", flush=True)
    else:
        uni = ES.load_universe(FG.DEFAULT_UNIVERSE, require_pbp_complete=True)
        ev = pd.read_parquet("data/processed/models/fg_make/events_v2.parquet")
        ev.attrs["rim_override_max_ft"] = ES.rim_override_for_version("v2")
        ev.attrs["possessions_version"] = "v2"
        design = FG.build_design(SEASONS, universe=uni, version="v2", events=ev)
        print(f"built design {design.shape} in {time.time() - t0:.1f}s", flush=True)

    design = add_pre_shot_margin(design)
    rep: dict = {"created_at": pd.Timestamp.utcnow().isoformat(), "seasons": SEASONS,
                 "n_attempts": int(len(design)), "margin_labels": MARGIN_LABELS,
                 "minute_labels": MINUTE_LABELS, "ref_margin": REF_MARGIN,
                 "min_cell": MIN_CELL}

    # ---- PART 0: the leak -------------------------------------------------
    if not a.skip_leak:
        print("\nPART 0: leak check against the raw feed", flush=True)
        rep["leak_raw_feed"] = leak_check(SEASONS)
        for s, v in rep["leak_raw_feed"].items():
            print(f"  {s}: made delta=value {v['made_fg']['pct_delta_equals_shot_value']}%  "
                  f"missed delta=0 {v['missed_fg']['pct_delta_zero']}%", flush=True)

    print("\nPART 0b: make rate by POST-shot vs PRE-shot margin bucket", flush=True)
    dpost = prep(design, "score_diff_post")
    dpre = prep(design, "score_diff_pre")
    rep["leak_bucket_tables"] = {}
    for c in FG.SHOT_CLASSES:
        rep["leak_bucket_tables"][c] = {
            "post_shot": raw_by_bucket(dpost[dpost["shot_class"] == c],
                                       "margin_bucket", MARGIN_LABELS),
            "pre_shot": raw_by_bucket(dpre[dpre["shot_class"] == c],
                                      "margin_bucket", MARGIN_LABELS),
        }
        post = rep["leak_bucket_tables"][c]["post_shot"]
        pre = rep["leak_bucket_tables"][c]["pre_shot"]
        sp_post = max(v["make_pct"] for v in post.values()) - min(v["make_pct"] for v in post.values())
        sp_pre = max(v["make_pct"] for v in pre.values()) - min(v["make_pct"] for v in pre.values())
        rep["leak_bucket_tables"][c]["span_post_pp"] = round(sp_post, 3)
        rep["leak_bucket_tables"][c]["span_pre_pp"] = round(sp_pre, 3)
        rep["leak_bucket_tables"][c]["manufactured_pp"] = round(sp_post - sp_pre, 3)
        print(f"  {c}: raw span post {sp_post:.2f} pp -> pre {sp_pre:.2f} pp "
              f"(manufactured {sp_post - sp_pre:.2f} pp)", flush=True)

    # ---- PART 1: decomposition on the CORRECTED margin ---------------------
    d = dpre
    nested = [("M0_margin_only", []),
              ("M1_plus_pregame_ratings", RATING_BLOCK),
              ("M2_plus_team_form", RATING_BLOCK + FORM_BLOCK),
              ("M3_plus_shooter", RATING_BLOCK + FORM_BLOCK + SHOOTER_BLOCK)]

    print("\n(a) confound decomposition by shot class [pre-shot margin]", flush=True)
    rep["a_by_class"] = {}
    for c in FG.SHOT_CLASSES:
        sub = d[d["shot_class"] == c]
        t = block_table(sub, nested, MARGIN_LABELS)
        t["raw"] = raw_by_bucket(sub, "margin_bucket", MARGIN_LABELS)
        rep["a_by_class"][c] = t
        print(f"  {c}: spans " + json.dumps(t["spans"]), flush=True)

    print("\n(a2) by minute bucket (classes pooled)", flush=True)
    rep["a_by_minute"] = {}
    for mb in MINUTE_LABELS:
        sub = d[d["minute_bucket"] == mb]
        if len(sub) < MIN_CELL:
            rep["a_by_minute"][mb] = {"n": int(len(sub)), "underpowered": True}
            continue
        t = block_table(sub, nested, MARGIN_LABELS)
        t["underpowered"] = False
        rep["a_by_minute"][mb] = t
        print(f"  {mb}: n={len(sub):,} spans " + json.dumps(t["spans"]), flush=True)

    print("\n(a2b) by minute bucket x shot class (M0 and M3 spans only)", flush=True)
    rep["a_by_minute_class"] = {}
    for mb in MINUTE_LABELS:
        for c in FG.SHOT_CLASSES:
            sub = d[(d["minute_bucket"] == mb) & (d["shot_class"] == c)]
            k = f"{mb}|{c}"
            if len(sub) < MIN_CELL:
                rep["a_by_minute_class"][k] = {"n": int(len(sub)), "underpowered": True}
                continue
            _, s0 = gcomp_curve(sub, [], MARGIN_LABELS)
            _, s3 = gcomp_curve(sub, RATING_BLOCK + FORM_BLOCK + SHOOTER_BLOCK,
                                MARGIN_LABELS)
            rep["a_by_minute_class"][k] = {"n": int(len(sub)), "underpowered": False,
                                           "M0_span_pp": round(s0, 3),
                                           "M3_span_pp": round(s3, 3)}
    print("   " + json.dumps({k: v.get("M3_span_pp", "UP")
                             for k, v in rep["a_by_minute_class"].items()}), flush=True)

    print("\n(a3) by offensive-team pregame rating quintile", flush=True)
    d["team_q"] = pd.qcut(d["net_rating_c"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rep["a_by_team_quintile"] = {}
    for q in (1, 2, 3, 4, 5):
        sub = d[d["team_q"] == q]
        t = block_table(sub, nested, MARGIN_LABELS)
        t["mean_net_rating_c"] = round(float(sub["net_rating_c"].mean()), 4)
        rep["a_by_team_quintile"][str(q)] = t
        print(f"  Q{q}: n={len(sub):,} spans " + json.dumps(t["spans"]), flush=True)

    print("\n(a4) by chance type", flush=True)
    ct = np.where(d["chance_number"].to_numpy() > 1, "continuation", "first")
    ct = np.where(d["is_transition_f"].to_numpy() > 0, "transition_first", ct)
    d["chance_type"] = ct
    rep["a_by_chance_type"] = {}
    for k in ("first", "transition_first", "continuation"):
        sub = d[d["chance_type"] == k]
        if len(sub) < MIN_CELL:
            rep["a_by_chance_type"][k] = {"n": int(len(sub)), "underpowered": True}
            continue
        t = block_table(sub, nested, MARGIN_LABELS)
        t["underpowered"] = False
        rep["a_by_chance_type"][k] = t
        print(f"  {k}: n={len(sub):,} spans " + json.dumps(t["spans"]), flush=True)

    # ---- (b) garbage time --------------------------------------------------
    print("\n(b) garbage time -- residual after the defensive five's as-of rates",
          flush=True)
    ev = pd.read_parquet("data/processed/models/fg_make/events_v2.parquet")
    ev24 = ev[ev["season"].isin([2024, 2025])]
    rates = FG.defender_rates(ev24, prior_att=100)
    d24 = d[d["season"].isin([2024, 2025])].copy()
    d24 = FG.attach_lineup_features(d24, rates)
    d24 = d24[d24["lineup_on_floor_ok"]].copy()
    print(f"  lineup-complete rows 2024-2025: {len(d24):,}", flush=True)
    nested_b = nested + [("M4_plus_def_five",
                          RATING_BLOCK + FORM_BLOCK + SHOOTER_BLOCK + LINEUP_BLOCK)]
    rep["b_lineup"] = {"n": int(len(d24)), "prior_att": 100, "by_class": {}}
    for c in FG.SHOT_CLASSES:
        t = block_table(d24[d24["shot_class"] == c], nested_b, MARGIN_LABELS)
        rep["b_lineup"]["by_class"][c] = t
        print(f"  {c}: spans " + json.dumps(t["spans"]), flush=True)

    # the garbage-time cell itself, at several candidate thresholds
    print("\n(b2) garbage-time indicator residual, candidate thresholds", flush=True)
    rep["b_garbage_grid"] = {}
    for mgn in (10, 15, 20):
        for secs in (300, 480, 600):
            key = f"abs_margin>={mgn} & gsr<={secs}"
            flag_all = ((d24["abs_margin"].to_numpy() >= mgn)
                        & (d24["gsr"].to_numpy() <= secs)
                        & (d24["period"].to_numpy() <= 2))
            cell = {}
            for c in FG.SHOT_CLASSES:
                sub = d24[d24["shot_class"] == c]
                f = flag_all[(d24["shot_class"] == c).to_numpy()]
                r = indicator_residual(sub, f, nested_b)
                r["flag_make_pct"] = round(float(sub["y"].to_numpy()[f].mean() * 100), 3) \
                    if f.sum() else None
                r["rest_make_pct"] = round(float(sub["y"].to_numpy()[~f].mean() * 100), 3)
                r["mean_shooter_make_c_pp_flag"] = round(
                    float(sub["shooter_make_c"].to_numpy()[f].mean() * 100), 3) if f.sum() else None
                r["mean_def5_rim_allow_c_pp_flag"] = round(
                    float(sub["def5_rim_allow_c"].to_numpy()[f].mean() * 100), 3) if f.sum() else None
                cell[c] = r
            rep["b_garbage_grid"][key] = cell
            print(f"  {key}: " + json.dumps(
                {c: {"n": v["n_flag"], "M3": v["M3_plus_shooter"], "M4": v["M4_plus_def_five"]}
                 for c, v in cell.items()}), flush=True)

    # ---- (c) end-game tactics ---------------------------------------------
    print("\n(c) end-game mix (3PA share, rushed attempts) by minute x margin", flush=True)
    tb_bins = [-np.inf, -9.5, -3.5, -0.5, 0.5, 3.5, 9.5, np.inf]
    tb_labels = ["trail 10+", "trail 4-9", "trail 1-3", "tied",
                 "lead 1-3", "lead 4-9", "lead 10+"]
    d["trail_bucket"] = pd.cut(d["score_diff_pre"].to_numpy(float), tb_bins,
                               labels=tb_labels).astype(object)
    mix = []
    for mb in MINUTE_LABELS:
        for tb in tb_labels:
            s = d[(d["minute_bucket"] == mb) & (d["trail_bucket"] == tb)]
            if not len(s):
                continue
            mix.append({"minute_bucket": mb, "margin": tb, "n": int(len(s)),
                        "three_share_pct": round(float((s["shot_class"] == "FGA_3").mean() * 100), 3),
                        "rim_share_pct": round(float((s["shot_class"] == "FGA_rim").mean() * 100), 3),
                        "mean_chance_elapsed_s": round(float(s["chance_elapsed_s"].mean()), 3),
                        "make_pct": round(float(s["y"].mean() * 100), 3),
                        "underpowered": bool(len(s) < MIN_CELL)})
    rep["c_mix"] = mix

    print("\n(c2) end-game indicator residual, candidate thresholds", flush=True)
    rep["c_endgame_grid"] = {}
    for secs, lo, hi in ((120, 1, 9), (120, 1, 6), (300, 1, 9), (60, 1, 6)):
        key = f"gsr<={secs} & trail {lo}-{hi}"
        flag_all = ((d["gsr"].to_numpy() <= secs) & (d["period"].to_numpy() <= 2)
                    & (d["score_diff_pre"].to_numpy() <= -lo)
                    & (d["score_diff_pre"].to_numpy() >= -hi))
        cell = {}
        for c in FG.SHOT_CLASSES:
            sub = d[d["shot_class"] == c]
            f = flag_all[(d["shot_class"] == c).to_numpy()]
            r = indicator_residual(sub, f, nested)
            r["flag_make_pct"] = round(float(sub["y"].to_numpy()[f].mean() * 100), 3) \
                if f.sum() else None
            cell[c] = r
        rep["c_endgame_grid"][key] = cell
        print(f"  {key}: " + json.dumps(
            {c: {"n": v["n_flag"], "M0": v["M0_margin_only"], "M3": v["M3_plus_shooter"]}
             for c, v in cell.items()}), flush=True)

    # ---- (d) per-season stability -----------------------------------------
    print("\n(d) per-season stability of the M0 / M3 spans", flush=True)
    seas = {}
    for s in SEASONS:
        sub = d[d["season"] == s]
        row = {}
        for c in FG.SHOT_CLASSES:
            s2 = sub[sub["shot_class"] == c]
            _, sp0 = gcomp_curve(s2, [], MARGIN_LABELS)
            _, sp3 = gcomp_curve(s2, RATING_BLOCK + FORM_BLOCK + SHOOTER_BLOCK,
                                 MARGIN_LABELS)
            row[c] = {"n": int(len(s2)), "M0_span_pp": round(sp0, 3),
                      "M3_span_pp": round(sp3, 3)}
        seas[str(s)] = row
        print(f"  {s}: " + json.dumps(row), flush=True)
    rep["d_by_season"] = seas

    rep["runtime_s"] = round(time.time() - t0, 1)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}  ({rep['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
