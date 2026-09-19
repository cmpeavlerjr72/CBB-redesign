#!/usr/bin/env python
"""
build_rebound_round3_design_v1.py -- the round-3 design cache for L3 REBOUND.

Pre-registration: `docs/models/rebound/experiments.md` sections 9 and 10,
committed BEFORE this file was written to run anything.

WHAT THIS DOES. It reads the SERVED round-1 event layer
(`data/processed/models/rebound/events_v1.parquet`, read-only -- other workers
read it) and `cbb_sim.models.rebound.build_design`, reproduces the served
`C_plus_state` columns exactly, and ADDS one column per round-3 arm:

  Block A (season drift / level)
    season_idx          already built by `build_design` (A1)
    lg_oreb_asof        the league's EXPANDING as-of live OREB% at the row's
                        date, within season, strictly before it (A4). Centred
                        per fold by the trainer on the TRAIN slice only, which
                        is a constant shift and changes no tree split.
    (A5's `trend_level_c` is fold-dependent -- an OLS on the fold's TRAIN
     seasons extrapolated forward -- and is therefore built by the trainer, not
     here, so a fold can never see another fold's fit.)

  Block B (the blocked_f feed)
    blk_asof_cell       the as-of measured block rate for this (miss type,
                        DEFENCE) cell: expanding within season over that
                        defence's prior games, falling back to the league's own
                        as-of rate for the miss type when the cell is empty.
                        This is arm B2's feed and nothing else uses it.

  Block C (prior-season carry)
    off_oreb_g1/g2/g3, opp_def_dreb_g1/g2/g3
                        the empirical-Bayes ladder, transcribed from
                        `scripts/train_possession_outcome_v4.py` (C3 = G1,
                        C1 = G2, C2 = G3 in this round's names), with k fitted
                        by method of moments on COMPLETED PRIOR SEASONS ONLY,
                        per season.
    C4 (roster-continuity weighting) is gated on a usability check that is run
    and REPORTED here rather than assumed.

  Block D (Decision 9's mandatory arms)
    off_oreb_oa1/oa2_c, opp_def_dreb_oa1/oa2_c
                        opponent-adjusted as-of rates, `one_pass` and
                        `iterative`, via `cbb_sim.features.opponent_adjust`.
    is_conf_game        the conference-game flag.

LEAK SAFETY. Every added column is either (i) an expanding mean over games
strictly before the row's own game within its season, (ii) a completed PRIOR
season's total shifted forward one season, or (iii) a parameter fitted on
completed prior seasons only. No column reads the test season's completed
totals, and the builder asserts that the reproduced raw columns match the
served design bit-for-bit (to 1e-6) before writing anything.

    .venv/Scripts/python.exe scripts/build_rebound_round3_design_v1.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("CBB_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.features import opponent_adjust as OA  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
IN_EVENTS = _ROOT / "data/processed/models/rebound/events_v1.parquet"
OUT_DIR = _ROOT / "data/processed/models/rebound/round3"
OUT_DESIGN = OUT_DIR / "design_round3.parquet"
OUT_META = OUT_DIR / "design_round3.meta.json"
CONTINUITY = _ROOT / "data/processed/roster_continuity_2027.parquet"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rate(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def live_boxes(events: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offence team): live first-chance opportunities and
    offensive rebounds. The SAME source `RB.team_rebound_form` uses, so the
    round-3 columns and the served columns are built on identical mass."""
    ev = events[events["outcome"].isin(["OREB", "DREB"])]
    ev = ev[ev["chance_index"] == 0]
    ev = ev.assign(_oreb=(ev["outcome"] == "OREB").astype("int32"), _opp=1)
    box = ev.groupby(["season", "game_id", "off_team_id", "def_team_id"],
                     as_index=False).agg(opps=("_opp", "sum"), orebs=("_oreb", "sum"))
    box = box.rename(columns={"off_team_id": "team_id", "def_team_id": "opp_id"})
    dates = universe[["game_id", "game_date"]].copy()
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    box = box.merge(dates, on="game_id", how="left")
    return box.sort_values(["season", "game_date", "game_id"], kind="stable").reset_index(drop=True)


def asof_panel(box: pd.DataFrame) -> pd.DataFrame:
    """As-of accumulated (opps, orebs) mass for every (season, game, team), on
    the offence side and on the defence-allowed side, plus the league's own
    accumulation on the same date. Transcribed from `RB.team_rebound_form`."""
    cols = ["opps", "orebs"]
    off = box.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    oa = PM.expanding_asof(off, ["season", "team_id"], cols)
    oa.columns = [f"off_{c}" for c in oa.columns]
    off = pd.concat([off[["season", "game_id", "team_id", "opp_id", "game_date"]], oa], axis=1)

    dfd = box.rename(columns={"team_id": "_off", "opp_id": "team_id"})
    dfd = dfd.sort_values(["season", "team_id", "game_date", "game_id"], kind="stable")
    da = PM.expanding_asof(dfd, ["season", "team_id"], cols)
    da.columns = [f"def_{c}" for c in da.columns]
    dfd = pd.concat([dfd[["season", "game_id", "team_id"]], da], axis=1)

    form = off.merge(dfd, on=["season", "game_id", "team_id"], how="left")
    day = box.groupby(["season", "game_date"], as_index=False)[cols].sum()
    day = day.sort_values(["season", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season"], cols)
    lg.columns = [f"lg_{c}" for c in lg.columns]
    day = pd.concat([day[["season", "game_date"]], lg], axis=1)
    return form.merge(day, on=["season", "game_date"], how="left")


def fit_k(box: pd.DataFrame, seasons_pool: list[int]) -> dict:
    """Method-of-moments empirical-Bayes `k = s2 / tau2`, in units of live
    opportunities, from COMPLETED seasons in `seasons_pool` only. The formula is
    `scripts/train_possession_outcome_v4.fit_k`'s, transcribed for this model's
    single rate so the two sub-models shrink by the same estimator."""
    out = {}
    b = box[box["season"].isin(seasons_pool)]
    for side, key_team in (("off", "team_id"), ("def", "opp_id")):
        g = b[[key_team, "season", "orebs", "opps"]].rename(columns={key_team: "tid"})
        g = g[g["opps"] > 0]
        if not len(g):
            out[side] = {"k": float("nan"), "n_team_seasons": 0, "seasons_pool": seasons_pool}
            continue
        ts = g.groupby(["season", "tid"], as_index=False)[["orebs", "opps"]].sum()
        ts["n_games"] = g.groupby(["season", "tid"]).size().to_numpy()
        ts["rate"] = _rate(ts["orebs"].to_numpy(), ts["opps"].to_numpy())
        lgs = g.groupby("season", as_index=False)[["orebs", "opps"]].sum()
        lgs["lg_rate"] = _rate(lgs["orebs"].to_numpy(), lgs["opps"].to_numpy())
        ts = ts.merge(lgs[["season", "lg_rate"]], on="season", how="left")

        gg = g.merge(ts[["season", "tid", "rate", "n_games"]], on=["season", "tid"], how="left")
        gg["rate_g"] = _rate(gg["orebs"].to_numpy(), gg["opps"].to_numpy())
        gg["sq"] = gg["opps"].to_numpy() * (gg["rate_g"].to_numpy() - gg["rate"].to_numpy()) ** 2
        multi = gg[gg["n_games"] >= 2]
        per_team = multi.groupby(["season", "tid"]).agg(ss=("sq", "sum"), n=("sq", "size")).reset_index()
        dfree = (per_team["n"] - 1).to_numpy()
        s2 = float(per_team["ss"].to_numpy().sum() / max(dfree.sum(), 1))

        w = ts["opps"].to_numpy().astype("float64")
        dev = ts["rate"].to_numpy() - ts["lg_rate"].to_numpy()
        mu = float(np.average(dev, weights=w))
        var_obs = float(np.average((dev - mu) ** 2, weights=w))
        samp = float(np.average(s2 / np.maximum(w, 1e-9), weights=w))
        tau2 = var_obs - samp
        k = float(min(max(s2 / max(tau2, 1e-9), 0.0), 1e6))
        out[side] = {"k": round(k, 3), "s2": round(s2, 6), "tau2_raw": round(var_obs, 6),
                     "tau2_net": round(tau2, 6), "n_team_seasons": int(len(ts)),
                     "seasons_pool": seasons_pool}
    return out


def prior_season_table(box: pd.DataFrame) -> dict:
    """Per (season, team): the team's COMPLETED prior-season centred rate and
    the denominator mass behind it, shifted forward one season so it is
    available to the next season only."""
    out = {}
    for side, key_team in (("off", "team_id"), ("def", "opp_id")):
        g = box[[key_team, "season", "orebs", "opps"]].rename(columns={key_team: "tid"})
        ts = g.groupby(["season", "tid"], as_index=False)[["orebs", "opps"]].sum()
        lgs = g.groupby("season", as_index=False)[["orebs", "opps"]].sum()
        lgs["lg_rate"] = _rate(lgs["orebs"].to_numpy(), lgs["opps"].to_numpy())
        ts = ts.merge(lgs[["season", "lg_rate"]], on="season", how="left")
        ts["rate"] = _rate(ts["orebs"].to_numpy(), ts["opps"].to_numpy())
        ts[f"{side}_priorc"] = (ts["rate"] - ts["lg_rate"]).astype("float32")
        ts[f"{side}_Dprev"] = ts["opps"].astype("float64")
        ts["season"] = ts["season"] + 1
        out[side] = ts[["season", "tid", f"{side}_priorc", f"{side}_Dprev"]]
    return out


def block_asof_cell(events: pd.DataFrame) -> pd.DataFrame:
    """As-of measured block rate for the (miss type, DEFENCE) cell -- arm B2's
    engine feed. Expanding over that defence's prior games within the season,
    with the league's own as-of rate for the miss type as the fallback on an
    empty cell (never a fabricated level, never the season total)."""
    fg = events[events["miss_type"].isin(["rim", "jump2", "three"])].copy()
    fg["game_date"] = pd.to_datetime(fg["game_date"])
    fg["_blk"] = fg["blocked"].astype("int32")
    fg["_n"] = 1
    tg = fg.groupby(["season", "def_team_id", "miss_type", "game_id", "game_date"],
                    as_index=False)[["_blk", "_n"]].sum()
    tg = tg.sort_values(["season", "def_team_id", "miss_type", "game_date", "game_id"],
                        kind="stable").reset_index(drop=True)
    a = PM.expanding_asof(tg, ["season", "def_team_id", "miss_type"], ["_blk", "_n"])
    tg = pd.concat([tg[["season", "def_team_id", "miss_type", "game_id", "game_date"]], a], axis=1)

    day = fg.groupby(["season", "miss_type", "game_date"], as_index=False)[["_blk", "_n"]].sum()
    day = day.sort_values(["season", "miss_type", "game_date"], kind="stable")
    lg = PM.expanding_asof(day, ["season", "miss_type"], ["_blk", "_n"])
    lg.columns = [f"lg{c}" for c in lg.columns]
    day = pd.concat([day[["season", "miss_type", "game_date"]], lg], axis=1)
    tg = tg.merge(day, on=["season", "miss_type", "game_date"], how="left")

    own = _rate(tg["_blk"].to_numpy(), tg["_n"].to_numpy())
    lgr = _rate(tg["lg_blk"].to_numpy(), tg["lg_n"].to_numpy())
    tg["blk_asof_cell"] = np.where(np.isnan(own), np.where(np.isnan(lgr), 0.0, lgr),
                                   own).astype("float32")
    tg["blk_cell_n_prior"] = tg["_n"].astype("float32")
    return tg[["season", "game_id", "def_team_id", "miss_type",
               "blk_asof_cell", "blk_cell_n_prior"]]


def continuity_usable() -> dict:
    """Condition (c)'s gate on arm C4, MEASURED rather than assumed."""
    if not CONTINUITY.exists():
        return {"usable": False, "reason": f"{CONTINUITY.name} does not exist"}
    d = pd.read_parquet(CONTINUITY)
    seasons = sorted(int(s) for s in d["season"].unique())
    need = [s for s in SEASONS if s > min(SEASONS)]      # 2023, 2024, 2025
    have = [s for s in need if s in seasons]
    return {"usable": bool(have) and len(have) == len(need),
            "reason": (f"roster-continuity table covers seasons {seasons} only; arm C4 needs "
                       f"a returning-minutes share for test seasons {need}, of which {have} "
                       f"are present. Coverage {len(have)}/{len(need)}."),
            "seasons_present": seasons, "seasons_needed": need, "n_rows": int(len(d))}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(SEASONS, context="rebound round-3 design")

    universe = ES.load_universe()
    ev = pd.read_parquet(IN_EVENTS)
    ev.attrs["rim_override_max_ft"] = ES.rim_override_for_version("v1")
    ev.attrs["possessions_version"] = "v1"
    d = RB.build_design(SEASONS, universe=universe, version="v1", events=ev)
    print(f"[{time.time()-t0:.0f}s] served design {d.shape}", flush=True)

    box = live_boxes(ev, universe)
    panel = asof_panel(box)
    meta: dict = {"n_rows_design": int(len(d)), "n_team_games_box": int(len(box))}

    # ---- reproduce the served raw centred columns (the pre-registered check) --
    own_o = _rate(panel["off_orebs"].to_numpy(), panel["off_opps"].to_numpy())
    lg_o = _rate(panel["lg_orebs"].to_numpy(), panel["lg_opps"].to_numpy())
    allowed = _rate(panel["def_orebs"].to_numpy(), panel["def_opps"].to_numpy())
    panel["rawc_off"] = np.where(np.isnan(own_o) | np.isnan(lg_o), 0.0, own_o - lg_o).astype("float32")
    # the defence's DREB deviation is the NEGATIVE of its allowed-OREB deviation
    panel["rawc_def"] = np.where(np.isnan(allowed) | np.isnan(lg_o), 0.0,
                                 (1.0 - allowed) - (1.0 - lg_o)).astype("float32")
    panel["D_off"] = panel["off_opps"].fillna(0.0).astype("float64")
    panel["D_def"] = panel["def_opps"].fillna(0.0).astype("float64")
    panel["lg_oreb_asof"] = np.where(np.isnan(lg_o), 0.0, lg_o).astype("float32")

    # ---- prior-season targets and the EB weights -----------------------------
    prior = prior_season_table(box)
    ks = {}
    pooled = fit_k(box, [s for s in SEASONS if s < max(SEASONS)])
    for s in SEASONS:
        pool = [x for x in SEASONS if x < s]
        ks[s] = (fit_k(box, pool) if pool
                 else {kk: dict(v, fallback="pooled -- 2022 has no prior season in the panel "
                                            "and is train-only in both folds")
                       for kk, v in pooled.items()})
    meta["k_by_season"] = ks

    p_off = prior["off"].rename(columns={"tid": "team_id"})
    panel = panel.merge(p_off, on=["season", "team_id"], how="left")
    p_def = prior["def"].rename(columns={"tid": "team_id"})
    panel = panel.merge(p_def, on=["season", "team_id"], how="left")
    meta["n_team_games_no_prior_season_off"] = int(panel["off_priorc"].isna().sum())
    meta["n_team_games_no_prior_season_def"] = int(panel["def_priorc"].isna().sum())
    for c, f in (("off_priorc", 0.0), ("def_priorc", 0.0), ("off_Dprev", 0.0), ("def_Dprev", 0.0)):
        panel[c] = panel[c].fillna(f)
    # the defence side's prior is an ALLOWED-oreb deviation; flip it to a DREB one
    panel["def_priorc"] = (-panel["def_priorc"]).astype("float32")

    sa = panel["season"].to_numpy()
    for side, pfx in (("off", "off_oreb"), ("def", "opp_def_dreb")):
        kk = pd.Series(sa).map({int(s): float(ks[int(s)][side]["k"]) for s in SEASONS}).to_numpy()
        D = panel[f"D_{side}"].to_numpy()
        Dp = panel[f"{side}_Dprev"].to_numpy()
        w = D / np.maximum(D + kk, 1e-9)
        w2 = Dp / np.maximum(Dp + kk, 1e-9)
        raw = panel[f"rawc_{side}"].to_numpy().astype("float64")
        pri = panel[f"{side}_priorc"].to_numpy().astype("float64")
        panel[f"{pfx}_g1"] = (w * raw).astype("float32")                        # round-3 C3
        panel[f"{pfx}_g2"] = (w * raw + (1 - w) * pri).astype("float32")        # round-3 C1
        panel[f"{pfx}_g3"] = (w * raw + (1 - w) * w2 * pri).astype("float32")   # round-3 C2
        panel[f"{side}_w"] = w.astype("float32")

    # ---- Block D: opponent adjustment ---------------------------------------
    for tag, method in (("oa1", "one_pass"), ("oa2", "iterative")):
        t1 = time.time()
        res = OA.adjust_team_form(box, {"oreb": ("orebs", "opps")}, {"oreb": 1.0}, method)
        b = box[["season", "game_id", "team_id"]].copy()
        b[f"corr_off_{tag}"] = res.off_adj["oreb"].to_numpy()
        b[f"corr_def_{tag}"] = res.def_adj["oreb"].to_numpy()
        off_c = b[["season", "game_id", "team_id", f"corr_off_{tag}"]]
        panel = panel.merge(off_c, on=["season", "game_id", "team_id"], how="left")
        dfc = box[["season", "game_id", "opp_id"]].copy()
        dfc["team_id"] = box["opp_id"]
        dfc[f"corr_def_{tag}"] = res.def_adj["oreb"].to_numpy()
        # `def_adj` is indexed by the OFFENCE row; the defence of that row is `opp_id`
        dfc = dfc.groupby(["season", "game_id", "team_id"], as_index=False)[f"corr_def_{tag}"].mean()
        panel = panel.merge(dfc, on=["season", "game_id", "team_id"], how="left")
        panel[f"corr_off_{tag}"] = panel[f"corr_off_{tag}"].fillna(0.0)
        panel[f"corr_def_{tag}"] = panel[f"corr_def_{tag}"].fillna(0.0)
        panel[f"off_oreb_{tag}_c"] = (panel["rawc_off"].to_numpy()
                                      - panel[f"corr_off_{tag}"].to_numpy()).astype("float32")
        # DREB deviation = -(allowed deviation); adjusting allowed by -corr flips the sign
        panel[f"opp_def_dreb_{tag}_c"] = (panel["rawc_def"].to_numpy()
                                          + panel[f"corr_def_{tag}"].to_numpy()).astype("float32")
        meta[f"opponent_adjust_{tag}"] = {"method": method,
                                          "seconds": round(time.time() - t1, 1),
                                          "meta": res.meta["oreb"]}
        print(f"[{time.time()-t0:.0f}s] opponent adjust {method} done", flush=True)

    # ---- join the panel onto the design --------------------------------------
    add_off = ["rawc_off", "lg_oreb_asof", "off_oreb_g1", "off_oreb_g2", "off_oreb_g3",
               "off_w", "off_priorc", "off_Dprev", "off_oreb_oa1_c", "off_oreb_oa2_c"]
    add_def = ["rawc_def", "opp_def_dreb_g1", "opp_def_dreb_g2", "opp_def_dreb_g3",
               "def_w", "def_priorc", "opp_def_dreb_oa1_c", "opp_def_dreb_oa2_c"]
    o = panel[["season", "game_id", "team_id"] + add_off].rename(columns={"team_id": "off_team_id"})
    f = panel[["season", "game_id", "team_id"] + add_def].rename(columns={"team_id": "def_team_id"})
    d = d.merge(o, on=["season", "game_id", "off_team_id"], how="left")
    d = d.merge(f, on=["season", "game_id", "def_team_id"], how="left")
    for c in add_off + add_def:
        d[c] = d[c].fillna(0.0)

    checks = {}
    for design_col, rebuilt in (("off_oreb_c", "rawc_off"), ("opp_def_dreb_c", "rawc_def")):
        diff = float(np.nanmax(np.abs(d[design_col].to_numpy().astype("float64")
                                      - d[rebuilt].to_numpy().astype("float64"))))
        checks[design_col] = diff
        assert diff < 1e-6, (f"round-3 rebuild of {design_col} does not reproduce the served "
                             f"column (max abs diff {diff}); the arms would then differ by more "
                             "than the arm. ABORT.")
    meta["reproduction_max_abs_diff"] = checks
    print(f"[{time.time()-t0:.0f}s] reproduction check PASS {checks}", flush=True)

    # ---- Block B feed and the conference flag --------------------------------
    blk = block_asof_cell(ev)
    d = d.merge(blk, on=["season", "game_id", "def_team_id", "miss_type"], how="left")
    d["blk_asof_cell"] = d["blk_asof_cell"].fillna(0.0).astype("float32")
    d["blk_cell_n_prior"] = d["blk_cell_n_prior"].fillna(0.0).astype("float32")
    meta["blk_feed_ft_rows"] = int((d["miss_type"] == "ft").sum())

    conf = CF.build_conference_flags(SEASONS)
    cc = conf[["season", "game_id", "is_conference_game"]].drop_duplicates() \
        if "is_conference_game" in conf.columns else None
    if cc is None:
        cand = [c for c in conf.columns if "conf" in c.lower() and conf[c].dtype == bool]
        cc = conf[["season", "game_id", cand[0]]].drop_duplicates().rename(
            columns={cand[0]: "is_conference_game"})
    d = d.merge(cc, on=["season", "game_id"], how="left")
    d["is_conf_game"] = d["is_conference_game"].fillna(False).astype("float32")
    d = d.drop(columns=["is_conference_game"])
    meta["conf_flag_share"] = float(d["is_conf_game"].mean())

    meta["continuity_gate_C4"] = continuity_usable()

    keep = [c for c in d.columns if c not in tuple(ES.ON_FLOOR_COLS)]
    d = d[keep]
    d["game_date"] = pd.to_datetime(d["game_date"])
    d.to_parquet(OUT_DESIGN, index=False)
    meta["runtime_s"] = round(time.time() - t0, 1)
    meta["out"] = str(OUT_DESIGN)
    meta["n_cols"] = int(d.shape[1])
    OUT_META.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"[{time.time()-t0:.0f}s] wrote {OUT_DESIGN} {d.shape}")
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("k_by_season", "opponent_adjust_oa1", "opponent_adjust_oa2")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
