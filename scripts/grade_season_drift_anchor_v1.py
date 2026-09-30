#!/usr/bin/env python
"""
grade_season_drift_anchor_v1.py -- the ONE blind grader for the season-drift
anchor round (`docs/models/season_drift/experiments.md` section 1).

Reads every prediction written by `scripts/exp_season_drift_anchor_v1.py` and
scores it with the same functions: a per-model REDUCER turns a prediction
matrix into (row loss, predicted rate, realised rate, exposure) and every
metric after that is model- and arm-agnostic. Then applies the pre-registered
decision rule (section 1.7) mechanically.

    .venv/Scripts/python.exe scripts/grade_season_drift_anchor_v1.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402

OUT = _ROOT / "results/season_drift/round1"
PREDS, FRAMES = OUT / "preds", OUT / "frames"
MODELS = ["rebound", "shot_block", "ft_tech", "po"]
SUBMODELS = ["rebound", "shot_block", "ft_tech"]
ARMS = ["R", "C0", "O", "P", "W", "F", "T"]
SIMPLICITY = {a: i for i, a in enumerate(ARMS)}
FOLDS = {"F1": {"train": [2022, 2023], "test": [2024]},
         "F2": {"train": [2022, 2023, 2024], "test": [2025]}}
PUBLISHED_FLOOR = {"rebound": 6.7e-05, "po": 8.04e-04}
MIN_TEAM_ROWS = 50
N_BOOT = 200
BOOT_SEED = 12345
TOL = {"rebound": 0.10, "shot_block": 0.10, "ft_tech": 2.0, "po": 0.10}  # pp, pp, rel %, pp
DRIFT_STOPS_TOL = 0.005
SLOPE_TOL = 0.05
PO_CLASSES = ("TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus")
RB_CLASSES = ("OREB", "DREB", "DEAD")


# ---------------------------------------------------------------------------
# reducers: the only model-specific code, identical for every arm
# ---------------------------------------------------------------------------
def reduce(model: str, p: np.ndarray, fr: pd.DataFrame) -> dict:
    if model == "rebound":
        cls = fr["cls"].to_numpy()
        loss = -np.log(np.clip(p[np.arange(len(p)), cls], 1e-15, None))
        rp = (p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-12))[:, None]
        return {"loss": loss, "rp": rp, "y": fr[["y0"]].to_numpy(), "w": fr["den"].to_numpy()}
    if model == "po":
        cls = fr["cls"].to_numpy()
        loss = -np.log(np.clip(p[np.arange(len(p)), cls], 1e-15, None))
        return {"loss": loss, "rp": p, "y": fr[[f"y{k}" for k in range(6)]].to_numpy(),
                "w": fr["den"].to_numpy()}
    if model == "shot_block":
        y = fr["y0"].to_numpy()
        q = np.clip(p[:, 0], 1e-15, 1 - 1e-15)
        loss = -(y * np.log(q) + (1 - y) * np.log(1 - q))
        return {"loss": loss, "rp": p, "y": fr[["y0"]].to_numpy(), "w": fr["den"].to_numpy()}
    # ft_tech: Poisson deviance per team-game; p is a rate per exposure
    y = fr["y0"].to_numpy()
    mu = np.maximum(p[:, 0] * fr["den"].to_numpy(), 1e-12)
    loss = 2.0 * np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu) - (y - mu), mu)
    return {"loss": loss, "rp": p, "y": fr[["y0"]].to_numpy(), "w": fr["den"].to_numpy()}


def is_binary_like(model: str) -> bool:
    return model != "ft_tech"


# ---------------------------------------------------------------------------
# generic metrics
# ---------------------------------------------------------------------------
def level(r: dict, k: int, mask=None) -> tuple[float, float, float]:
    """(predicted, actual, n_w) rate on rows in mask for class k."""
    w = r["w"] if mask is None else r["w"] * mask
    sw = w.sum()
    if sw <= 0:
        return np.nan, np.nan, 0.0
    return float((w * r["rp"][:, k]).sum() / sw), float((r["y"][:, k] * (w > 0)).sum() / sw), float(sw)


def gap_units(model: str, pred: float, act: float) -> float:
    """pp for binary / class-share models; relative % for ft_tech."""
    if model == "ft_tech":
        return (pred / act - 1.0) * 100.0 if act > 0 else np.nan
    return (pred - act) * 100.0


def team_table(r: dict, fr: pd.DataFrame, k: int, mask=None) -> pd.DataFrame:
    w = r["w"] if mask is None else r["w"] * mask
    df = pd.DataFrame({"team": fr["team"].to_numpy(), "wp": w * r["rp"][:, k],
                       "wy": r["y"][:, k] * (w > 0), "w": w, "prior": fr[f"prior{k}"].to_numpy()})
    g = df.groupby("team").agg(wp=("wp", "sum"), wy=("wy", "sum"), w=("w", "sum"),
                               prior=("prior", "first"))
    g = g[g["w"] > 0]
    g["p"] = g["wp"] / g["w"]
    g["a"] = g["wy"] / g["w"]
    return g


def slope_and_spread(model: str, r: dict, fr: pd.DataFrame, k: int, mask=None,
                     min_rows: float | None = None) -> dict:
    g = team_table(r, fr, k, mask)
    if model != "ft_tech":
        g = g[g["w"] >= (MIN_TEAM_ROWS if min_rows is None else min_rows)]
    out = {"n_teams": int(len(g))}
    gp = g[g["prior"].notna()]
    if len(gp) >= 50:
        gp = gp.assign(q=pd.qcut(gp["prior"].rank(method="first"), 5, labels=False))
        q = gp.groupby("q").agg(p=("p", "mean"), a=("a", "mean"))
        span_a = float(q["a"].iloc[-1] - q["a"].iloc[0])
        span_p = float(q["p"].iloc[-1] - q["p"].iloc[0])
        sign = 1.0 if span_a >= 0 else -1.0
        out.update({"slope_ratio": round(span_p / span_a, 4) if abs(span_a) > 1e-12 else None,
                    "monotone_steps": int((np.diff(q["p"].to_numpy()) * sign > 0).sum()),
                    "pred_by_q": [round(float(v), 6) for v in q["p"]],
                    "act_by_q": [round(float(v), 6) for v in q["a"]]})
    else:
        out.update({"slope_ratio": None, "UNDERPOWERED": True})
    if len(g) >= 20:
        sd_p = float(g["p"].std())
        sd_a = float(g["a"].std())
        if model == "ft_tech":
            samp = float((g["a"] / g["w"]).mean())
        else:
            samp = float((g["a"] * (1 - g["a"]) / g["w"]).mean())
        var_true = sd_a ** 2 - samp
        lvl = float(g["p"].mean() / g["a"].mean()) if g["a"].mean() > 0 else np.nan
        out.update({"sd_pred": sd_p, "sd_act": sd_a, "team_level_ratio": round(lvl, 4),
                    "sd_ratio_raw": round(sd_p / sd_a, 4) if sd_a > 0 else None,
                    "sd_ratio_nc": round(sd_p / np.sqrt(var_true), 4) if var_true > 0 else None})
        # SUPPLEMENTARY (not in the pre-registered rule): spread and slope with the
        # level ratio divided out, so a multiplicative level error does not read as spread
        if out["sd_ratio_nc"] is not None and np.isfinite(lvl) and lvl > 0:
            out["sd_ratio_nc_levelnorm"] = round(out["sd_ratio_nc"] / lvl, 4)
        if out.get("slope_ratio") is not None and np.isfinite(lvl) and lvl > 0:
            out["slope_levelnorm"] = round(out["slope_ratio"] / lvl, 4)
    return out


def calib(model: str, r: dict, p: np.ndarray, fr: pd.DataFrame) -> dict:
    if model in ("rebound", "po"):
        classes = RB_CLASSES if model == "rebound" else PO_CLASSES
        c = PM.decile_calibration(fr["cls"].to_numpy(), p, classes)
        ok, worst, who = PM.calibration_verdict(c)
        return {"pass": bool(ok), "worst_pp": worst, "class": who}
    if model == "shot_block":
        y, q = r["y"][:, 0], r["rp"][:, 0]
        b = pd.qcut(pd.Series(q).rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"y": y, "q": q, "b": b}).groupby("b").agg(a=("y", "mean"), e=("q", "mean"))
        worst = float((g["e"] - g["a"]).abs().max() * 100)
        return {"pass": worst <= 2.0, "worst_pp": round(worst, 3), "class": "blocked"}
    pr, ac, _ = level(r, 0)
    rel = gap_units(model, pr, ac)
    return {"pass": abs(rel) <= 10.0, "worst_pp": round(rel, 3), "class": "level_rel_pct"}


def per_game_mae(r: dict, fr: pd.DataFrame, k: int = 0) -> dict:
    w = r["w"]
    df = pd.DataFrame({"g": fr["game_id"].to_numpy(), "wp": w * r["rp"][:, k],
                       "wy": r["y"][:, k] * (w > 0), "w": w})
    g = df.groupby("g").sum()
    g = g[g["w"] >= 10]
    d = (g["wp"] - g["wy"]) / g["w"] * 100
    return {"n_games": int(len(g)), "mean_pp": round(float(d.mean()), 4),
            "mae_pp": round(float(d.abs().mean()), 4)}


def game_loss(r: dict, fr: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"g": fr["game_id"].to_numpy(), "l": r["loss"], "n": 1.0}).groupby("g").sum()


def paired_boot_se(ga: pd.DataFrame, gr: pd.DataFrame) -> float:
    j = ga.join(gr, lsuffix="_a", rsuffix="_r", how="inner")
    d = (j["l_a"] - j["l_r"]).to_numpy()
    n = j["n_a"].to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    stats = d[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(stats.std(ddof=1))


# ---------------------------------------------------------------------------
# per-cell scorecard
# ---------------------------------------------------------------------------
def score_cell(model: str, fold: str, fr: pd.DataFrame, p: np.ndarray) -> tuple[dict, dict]:
    r = reduce(model, p, fr)
    K = r["rp"].shape[1]
    month = pd.to_datetime(fr["game_date"]).dt.month.to_numpy()
    nd = np.isin(month, [11, 12]).astype("float64")
    out = {"primary": round(float(r["loss"].mean()), 7), "n": int(len(fr)), "classes": []}
    for k in range(K):
        pr, ac, _ = level(r, k)
        c = {"k": k, "pred": pr, "act": ac, "level": round(gap_units(model, pr, ac), 4),
             "level_rel": round(pr / ac - 1.0, 5) if ac > 0 else None}
        pn, an, _ = level(r, k, nd)
        c["novdec"] = round(gap_units(model, pn, an), 4)
        c["by_month"] = {}
        for mth in sorted(set(month.tolist())):
            pm_, am_, nw = level(r, k, (month == mth).astype("float64"))
            c["by_month"][int(mth)] = round(gap_units(model, pm_, am_), 4) if nw > 0 else None
        c["by_sub"] = {}
        for s in sorted(fr["sub"].unique().tolist()):
            ps, as_, nw = level(r, k, (fr["sub"].to_numpy() == s).astype("float64"))
            c["by_sub"][s] = {"level": round(gap_units(model, ps, as_), 4),
                              "rel": round(ps / as_ - 1.0, 5) if as_ > 0 else None,
                              "act": as_, "n_w": nw}
        c["by_site"] = {}
        for s in ("home", "away", "neutral"):
            ps, as_, nw = level(r, k, (fr["site"].to_numpy() == s).astype("float64"))
            c["by_site"][s] = round(gap_units(model, ps, as_), 4) if nw > 0 else None
        c["team"] = slope_and_spread(model, r, fr, k)
        c["team_novdec"] = slope_and_spread(model, r, fr, k, nd,
                                            min_rows=MIN_TEAM_ROWS / 3 if model != "ft_tech" else None)
        out["classes"].append(c)
    out["calib"] = calib(model, r, p, fr)
    if model in ("rebound", "shot_block"):
        out["per_game"] = per_game_mae(r, fr)
    return out, {"game_loss": game_loss(r, fr)}


def drift_cells(model: str, fold: str, meta: dict) -> list[dict]:
    """Section 1.5 item 5: classify rate x season cells from realised levels."""
    lv = meta["season_levels"]
    train, test = FOLDS[fold]["train"], FOLDS[fold]["test"][0]
    out = []
    for sub, by_s in lv.items():
        if sub == "__all__" and model in ("rebound", "shot_block"):
            continue
        if sub != "__all__" and model in ("ft_tech", "po"):
            continue
        K = len(next(iter(by_s.values())))
        for k in range(K):
            ys = np.array([by_s[str(s)][k] if str(s) in by_s else by_s[s][k] for s in train])
            beta = float(np.polyfit(np.array(train, dtype=float), ys, 1)[0])
            tv = by_s[str(test)][k] if str(test) in by_s else by_s[test][k]
            delta = float(tv - ys[-1])
            cont = (np.sign(delta) == np.sign(beta)) and abs(delta) >= 0.5 * abs(beta)
            out.append({"model": model, "fold": fold, "sub": "all" if sub == "__all__" else sub,
                        "k": k, "beta": beta, "delta": delta, "class": "TREND" if cont else "FLAT_OR_REV",
                        "levels": [round(float(v), 6) for v in ys] + [round(float(tv), 6)]})
    return out


def cell_rel_gap(card: dict, model: str, sub: str, k: int) -> float | None:
    c = card["classes"][k]
    if sub == "all" and model in ("ft_tech", "po"):
        return c["level_rel"]
    return (c["by_sub"].get(sub) or {}).get("rel")


# ---------------------------------------------------------------------------
# rendering (tables only; every number comes from the cards above)
# ---------------------------------------------------------------------------
def _t(rows: list[list], head: list[str]) -> str:
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    for r in rows:
        s += "| " + " | ".join("" if v is None else str(v) for v in r) + " |\n"
    return s


def _f(v, n=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), n) if isinstance(v, (int, float, np.floating)) else v


def render(cards, decision, dcells, ds_score, reseed, metas, votes, winner, po_check) -> str:
    L = ["# Season-drift anchor round 1: graded tables (generated by "
         "`scripts/grade_season_drift_anchor_v1.py`)\n",
         "Units: rebound / shot_block / po level gaps in pp; ft_tech in RELATIVE % of the "
         "realised rate. `sd_nc` = SD of team predictions over the noise-corrected SD of "
         "realised team rates. Slope = prior-season-quintile span ratio.\n"]
    L.append(f"Reseed floors (R seed 1 vs 0, F2): `{json.dumps(reseed)}`\n")
    for model in MODELS:
        L.append(f"\n## {model}\n")
        for fold in ("F2", "F1"):
            rows = []
            R = cards.get((model, fold, "R", 0))
            for arm in ARMS:
                c = cards.get((model, fold, arm, 0))
                if c is None:
                    rows.append([arm, "NOT RUN"] + [None] * 14)
                    continue
                c0 = c["classes"][0]
                lv = (max(abs(x["level"]) for x in c["classes"]) if model == "po" else c0["level"])
                nd = (max(abs(x["novdec"]) for x in c["classes"]) if model == "po" else c0["novdec"])
                gain = None if R is None else R["primary"] - c["primary"]
                rows.append([arm, c["primary"], _f(gain, 7), _f(c.get("boot_se"), 7), lv, nd,
                             c0["team"].get("slope_ratio"), c0["team_novdec"].get("slope_ratio"),
                             c0["team"].get("sd_ratio_nc"), c0["team"].get("sd_ratio_raw"),
                             c0["team"].get("slope_levelnorm"), c0["team"].get("sd_ratio_nc_levelnorm"),
                             f"{'P' if c['calib']['pass'] else 'F'} {c['calib']['worst_pp']}",
                             (c.get("per_game") or {}).get("mae_pp"),
                             ds_score[arm]["per_model"].get(model)])
            L.append(f"\n### {model} {fold}\n")
            head = ["arm", "primary", "gain vs R", "paired boot SE", "level" + (" max|cls|" if model == "po" else ""),
                    "Nov-Dec" + (" max|cls|" if model == "po" else ""), "team slope", "slope Nov-Dec",
                    "sd_nc", "sd_raw", "slope/lvl (supp)", "sd_nc/lvl (supp)", "calib", "per-game MAE pp", "drift-stops (model)"]
            L.append(_t(rows, head))
            # sub-type / class levels
            if R is not None:
                if model == "po":
                    rows = [[arm] + [c["level"] for c in cards[(model, fold, arm, 0)]["classes"]]
                            + [c["team"].get("slope_ratio") for c in cards[(model, fold, arm, 0)]["classes"]]
                            for arm in ARMS if (model, fold, arm, 0) in cards]
                    L.append("\nper class: level pp, then team slope\n\n")
                    L.append(_t(rows, ["arm"] + [f"lvl {c}" for c in PO_CLASSES]
                                + [f"slope {c}" for c in PO_CLASSES]))
                else:
                    subs = sorted(R["classes"][0]["by_sub"])
                    rows = [[arm] + [cards[(model, fold, arm, 0)]["classes"][0]["by_sub"][s]["level"]
                                     for s in subs]
                            + [cards[(model, fold, arm, 0)]["classes"][0]["by_site"].get(s)
                               for s in ("home", "away", "neutral")]
                            for arm in ARMS if (model, fold, arm, 0) in cards]
                    L.append("\nlevel by sub-type and by site\n\n")
                    L.append(_t(rows, ["arm"] + subs + ["home", "away", "neutral"]))
                months = sorted(R["classes"][0]["by_month"])
                kk = 0
                rows = [[arm] + [cards[(model, fold, arm, 0)]["classes"][kk]["by_month"].get(m) for m in months]
                        for arm in ARMS if (model, fold, arm, 0) in cards]
                L.append("\nlevel by month" + (" (class TOV)" if model == "po" else "") + "\n\n")
                L.append(_t(rows, ["arm"] + [str(m) for m in months]))
            meta = metas.get((model, fold))
            if meta:
                a = meta["anchor"]
                L.append(f"\nanchor: n0 fitted = {a['n0_fitted']} (fit seasons {a['n0_fit_seasons']}); "
                         f"prior-season level for test = {a['prior_for_test']}; trend to test = "
                         f"{a['trend_test']}; Lbar = {a['Lbar']}\n")
    L.append("\n## Drift-stops / reverses cells (relative gap per arm; FLAT_OR_REV cells score E4)\n\n")
    rows = []
    for dc in dcells:
        rows.append([dc["model"], dc["fold"], dc["sub"], dc["k"], dc["class"], dc["levels"],
                     _f(dc["beta"], 5), _f(dc["delta"], 5)]
                    + [(dc.get("rel_gap") or {}).get(a) for a in ARMS])
    L.append(_t(rows, ["model", "fold", "sub", "k", "class", "levels (train..., test)", "train slope",
                       "test - last"] + ARMS))
    L.append("\npooled drift-stops score (mean |rel gap arm| - |rel gap R| over FLAT_OR_REV cells):\n\n")
    L.append(_t([[a, ds_score[a]["pooled"], ds_score[a]["n_cells"], json.dumps(ds_score[a]["per_model"])]
                 for a in ARMS], ["arm", "pooled", "n cells", "per model"]))
    L.append("\n## Decision rule (section 1.7), applied mechanically\n")
    for m in SUBMODELS:
        rows = []
        for a, r in decision[m]["rows"].items():
            if "checks" not in r:
                rows.append([a, r.get("status")] + [None] * 12)
                continue
            ch = r["checks"]
            rows.append([a, r["gain_F2"], _f(r["floor"], 7), r["x_floor"], r["gain_F1"]]
                        + ["PASS" if ch[k] else "fail" for k in ch] + [r["eligible"],
                                                                         r["E5_sym_supplementary"],
                                                                         r["eligible_if_E5_sym"]])
        L.append(f"\n### {m}: selected **{decision[m]['selected']}** (eligible {decision[m]['eligible']})\n\n")
        L.append(_t(rows, ["arm", "F2 gain", "floor", "x floor", "F1 gain", "E1", "E2", "E3", "E4", "E5",
                           "E6", "E7", "eligible", "E5-sym (supp)", "eligible if E5-sym (supp)"]))
    L.append(f"\nVotes: `{json.dumps(votes)}`; po control: `{json.dumps(po_check)}`; "
             f"**winning design: {winner}**\n")
    return "".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    assert_not_sealed([2022, 2023, 2024, 2025], context="season_drift grader")
    cards: dict = {}
    extras: dict = {}
    metas: dict = {}
    for model in MODELS:
        for fold in FOLDS:
            fp = FRAMES / f"{model}_{fold}.parquet"
            if not fp.exists():
                continue
            fr = pd.read_parquet(fp)
            assert_not_sealed(fr, context=f"{model}_{fold} frame")
            metas[(model, fold)] = json.loads((FRAMES / f"{model}_{fold}.meta.json").read_text())
            for arm in ARMS:
                for seed in (0, 1):
                    pp = PREDS / f"{model}_{fold}_{arm}_s{seed}.npy"
                    if not pp.exists():
                        continue
                    p = np.load(pp)
                    card, ex = score_cell(model, fold, fr, p)
                    cards[(model, fold, arm, seed)] = card
                    extras[(model, fold, arm, seed)] = ex
                    print(f"graded {model} {fold} {arm} s{seed}: primary={card['primary']} "
                          f"level={[c['level'] for c in card['classes']]}", flush=True)

    # paired bootstrap SE vs R and operative floors
    for (model, fold, arm, seed), card in cards.items():
        ref = extras.get((model, fold, "R", 0))
        if ref is None or (arm == "R" and seed == 0):
            card["boot_se"] = 0.0
            continue
        card["boot_se"] = paired_boot_se(extras[(model, fold, arm, seed)]["game_loss"], ref["game_loss"])
    reseed = {}
    for model in MODELS:
        a, b = cards.get((model, "F2", "R", 0)), cards.get((model, "F2", "R", 1))
        if a and b:
            reseed[model] = {"primary": abs(a["primary"] - b["primary"]),
                             "level": max(abs(x["level"] - y["level"])
                                          for x, y in zip(a["classes"], b["classes"]))}

    def floor(model, fold, arm):
        c = cards[(model, fold, arm, 0)]
        return max(reseed.get(model, {}).get("primary", 0.0), PUBLISHED_FLOOR.get(model, 0.0),
                   2.0 * c.get("boot_se", 0.0))

    # drift-stops robustness
    dcells = []
    for (model, fold), meta in metas.items():
        dcells += drift_cells(model, fold, meta)
    ds_score: dict = {}
    for arm in ARMS:
        vals, per_model = [], {}
        for dc in dcells:
            if dc["class"] != "FLAT_OR_REV":
                continue
            ca = cards.get((dc["model"], dc["fold"], arm, 0))
            cr = cards.get((dc["model"], dc["fold"], "R", 0))
            if ca is None or cr is None:
                continue
            ga = cell_rel_gap(ca, dc["model"], dc["sub"], dc["k"])
            gr = cell_rel_gap(cr, dc["model"], dc["sub"], dc["k"])
            if ga is None or gr is None:
                continue
            v = abs(ga) - abs(gr)
            dc.setdefault("rel_gap", {})[arm] = round(ga, 5)
            vals.append(v)
            per_model.setdefault(dc["model"], []).append(v)
        ds_score[arm] = {"pooled": round(float(np.mean(vals)), 5) if vals else None,
                         "n_cells": len(vals),
                         "per_model": {m: round(float(np.mean(v)), 5) for m, v in per_model.items()}}
    for dc in dcells:  # also record trend cells' gaps for the report
        for arm in ARMS:
            ca = cards.get((dc["model"], dc["fold"], arm, 0))
            if ca is not None:
                g = cell_rel_gap(ca, dc["model"], dc["sub"], dc["k"])
                if g is not None:
                    dc.setdefault("rel_gap", {})[arm] = round(g, 5)

    # decision rule, section 1.7
    def headline_level(card, model):
        if model == "po":
            return max(abs(c["level"]) for c in card["classes"])
        return abs(card["classes"][0]["level"])

    def headline_novdec(card, model):
        if model == "po":
            return max(abs(c["novdec"]) for c in card["classes"])
        return abs(card["classes"][0]["novdec"])

    decision: dict = {}
    for model in SUBMODELS:
        R2, R1 = cards.get((model, "F2", "R", 0)), cards.get((model, "F1", "R", 0))
        rows = {}
        for arm in ARMS[1:]:
            c2, c1 = cards.get((model, "F2", arm, 0)), cards.get((model, "F1", arm, 0))
            if c2 is None or R2 is None:
                rows[arm] = {"status": "NOT RUN"}
                continue
            fl = floor(model, "F2", arm)
            gain = R2["primary"] - c2["primary"]
            t2, tR = c2["classes"][0]["team"], R2["classes"][0]["team"]
            tn2, tnR = c2["classes"][0]["team_novdec"], R2["classes"][0]["team_novdec"]

            def ge(a, b):
                return (a is not None) and (b is not None) and a >= b - SLOPE_TOL
            e = {
                "E1_primary": gain > fl,
                "E2_level": headline_level(c2, model) < headline_level(R2, model),
                "E3_novdec": headline_novdec(c2, model) <= headline_novdec(R2, model) + TOL[model],
                "E4_drift_stops": (ds_score[arm]["pooled"] is not None
                                   and ds_score[arm]["pooled"] <= DRIFT_STOPS_TOL),
                "E5_slope_spread": (ge(t2.get("slope_ratio"), tR.get("slope_ratio"))
                                    and ge(tn2.get("slope_ratio"), tnR.get("slope_ratio"))
                                    and ge(t2.get("sd_ratio_nc"), tR.get("sd_ratio_nc"))),
                "E6_calib": not (R2["calib"]["pass"] and not c2["calib"]["pass"]),
            }
            if c1 is None or R1 is None:
                e["E7_fold1"] = False
                f1 = "NOT RUN"
            else:
                e["E7_fold1"] = ((c1["primary"] < R1["primary"])
                                 and headline_level(c1, model) <= headline_level(R1, model) + TOL[model])
                f1 = round(R1["primary"] - c1["primary"], 7)
            def closer(a, b):
                return (a is not None) and (b is not None) and abs(a - 1.0) <= abs(b - 1.0) + SLOPE_TOL
            e5_sym = (closer(t2.get("slope_levelnorm"), tR.get("slope_levelnorm"))
                      and closer(t2.get("sd_ratio_nc_levelnorm"), tR.get("sd_ratio_nc_levelnorm")))
            rows[arm] = {"E5_sym_supplementary": bool(e5_sym),
                         "eligible_if_E5_sym": bool(all(v for k, v in e.items() if k != "E5_slope_spread") and e5_sym),
                         "gain_F2": round(gain, 7), "floor": fl, "x_floor": round(gain / fl, 2) if fl > 0 else None,
                         "gain_F1": f1, "checks": e, "eligible": all(e.values())}
        elig = [a for a in rows if rows[a].get("eligible")]
        sel = "R"
        if elig:
            best = max(elig, key=lambda a: rows[a]["gain_F2"])
            within = [a for a in elig if rows[best]["gain_F2"] - rows[a]["gain_F2"] <= rows[best]["floor"]]
            sel = min(within, key=lambda a: SIMPLICITY[a])
        decision[model] = {"rows": rows, "eligible": elig, "selected": sel}

    votes = {}
    for m in SUBMODELS:
        s = decision[m]["selected"]
        if s != "R":
            votes.setdefault(s, []).append(m)
    winner, po_check = None, {}
    cand = [a for a, ms in votes.items() if len(ms) >= 2]
    for a in cand:
        Rp, Ap = cards.get(("po", "F2", "R", 0)), cards.get(("po", "F2", a, 0))
        if Rp is None or Ap is None:
            po_check[a] = {"status": "NOT RUN"}
            continue
        pf = max(PUBLISHED_FLOOR["po"], reseed.get("po", {}).get("primary", 0.0))
        ok1 = Ap["primary"] - Rp["primary"] <= pf
        ok2 = headline_level(Ap, "po") <= headline_level(Rp, "po") + TOL["po"]
        ok3 = all((ca["team"].get("slope_ratio") is None) or (cr["team"].get("slope_ratio") is None)
                  or ca["team"]["slope_ratio"] >= cr["team"]["slope_ratio"] - SLOPE_TOL
                  for ca, cr in zip(Ap["classes"], Rp["classes"]))
        po_check[a] = {"primary_ok": ok1, "level_ok": ok2, "slope_ok": ok3}
        if ok1 and ok2 and ok3:
            winner = a
    result = {"cards": {"|".join(map(str, k)): v for k, v in cards.items()},
              "reseed": reseed, "drift_cells": dcells, "drift_stops_score": ds_score,
              "decision": decision, "votes": votes, "po_check": po_check, "winner": winner,
              "anchor_meta": {"|".join(k): v["anchor"] for k, v in metas.items()},
              "created_at": pd.Timestamp.now("UTC").isoformat()}
    (OUT / "grade_v1.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    (OUT / "tables_v1.md").write_text(render(cards, decision, dcells, ds_score, reseed, metas,
                                             votes, winner, po_check), encoding="utf-8")
    print(json.dumps({"votes": votes, "winner": winner, "po_check": po_check,
                      "selected": {m: decision[m]["selected"] for m in SUBMODELS},
                      "eligible": {m: decision[m]["eligible"] for m in SUBMODELS},
                      "drift_stops": ds_score, "reseed": reseed}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
