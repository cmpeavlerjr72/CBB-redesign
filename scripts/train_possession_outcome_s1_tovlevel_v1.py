"""train_possession_outcome_s1_tovlevel_v1.py -- PO section 30 (TOV level round), arms T1 / T2. PO worker, 2026-10-05.

Versioned sibling; nothing existing is edited. Runs the served `first` lgbm S1 monthly schedule through lane M's
anchored artifacts trainer (`train_possession_outcome_s1_par_anchor_artifacts_v1`, which reuses lane C's
`build_tasks_anchor` / `_dispatch` and the unmodified round-2 builder) with ONE change: the offset columns.

  offset[:, TOV] = logit L(s, t) - logit Lbar ; the other five classes 0
  L(s, t) = (N_tov,before + n0 * prior(s)) / (N_before + n0)    `first` rows of season s dated strictly before t
  T1: prior(s) = L_end(s-1)                                     (Lbar when s-1 is not a train season of the fold)
  T2: prior(s) = expit(logit L_end(s-1) + b(s)),  b(s) = OLS slope of logit league box TOV/poss over s-5..s-1
  n0 chosen on the fold's train seasons with a previous season by the anchor-only binomial likelihood.

The `first` joblibs are MARKED anchored (arm T1/T2, cols [0]); serve with ENGINE_SEASON_ANCHOR=<npz> from
`scripts/build_engine_tovlevel_offsets_v1.py`. Spec: `docs/models/possession_outcome/experiments.md` section 30.

    .venv/Scripts/python.exe scripts/train_possession_outcome_s1_tovlevel_v1.py --arm T1 --fold F2 --season 2025 \
        --engine-dir data/processed/models/engine --out-root data/processed/models/possession_outcome/tovlevel/T1_F2_s0 --n-jobs 6
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
import train_par_common_v1 as C  # noqa: E402

C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_engine_event_round2 as B  # noqa: E402
import train_possession_outcome_s1_par_anchor_artifacts_v1 as AA  # noqa: E402
import train_possession_outcome_s1_par_anchor_v1 as A  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

TOV = 0
N0_GRID = [0.0] + [float(10 ** (k / 2)) for k in range(2, 17)] + [np.inf]   # 0, 1e1, 3.16e1, ..., 1e8, inf
BOX_DIRS = [_ROOT / "data/raw/hoopr_hist/team_box", _ROOT / "data/raw/hoopr/team_box"]
TREND_K = 5


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="float64")))


def box_tov_rate(season: int) -> float:
    """League TOV per possession from hoopR team box (regular season, every team row)."""
    for d in BOX_DIRS:
        p = d / f"team_box_{season}.parquet"
        if p.exists():
            t = pd.read_parquet(p)
            t = t[t["season_type"] == 2]
            num = {c: pd.to_numeric(t[c], errors="coerce") for c in
                   ("total_turnovers", "turnovers", "field_goals_attempted", "offensive_rebounds",
                    "free_throws_attempted")}
            tv = num["total_turnovers"].fillna(num["turnovers"])
            P = num["field_goals_attempted"] - num["offensive_rebounds"] + tv + 0.44 * num["free_throws_attempted"]
            ok = P.notna() & tv.notna() & (P > 40)
            return float(tv[ok].sum() / P[ok].sum())
    raise FileNotFoundError(f"team_box_{season}.parquet not in {BOX_DIRS}")


def trend_slope(season: int) -> float:
    """b(s): OLS slope of logit league box TOV/poss on season over s-K..s-1 (completed seasons only)."""
    if season >= 2026:
        raise SystemExit("2026 SEALED")
    xs = np.arange(season - TREND_K, season)
    ys = logit([box_tov_rate(int(x)) for x in xs])
    return float(np.polyfit(xs, ys, 1)[0])


class TovLevel:
    """Daily `first` TOV aggregates of the fold's train seasons + test season; levels at any (season, date)."""

    def __init__(self, season, date, y, train_seasons, arm: str):
        df = pd.DataFrame({"season": np.asarray(season).astype("int64"),
                           "date": pd.to_datetime(date).values.astype("datetime64[D]"),
                           "tov": (np.asarray(y).astype(int) == TOV).astype("float64")})
        g = df.groupby(["season", "date"])["tov"].agg(["sum", "size"]).reset_index()
        self.train_seasons = [int(s) for s in train_seasons]
        tr = g[g["season"].isin(self.train_seasons)]
        self.Lbar = float(tr["sum"].sum() / tr["size"].sum())
        self.by = {}
        for s, gs in g.groupby("season"):
            gs = gs.sort_values("date")
            self.by[int(s)] = (gs["date"].to_numpy(), np.concatenate([[0.0], np.cumsum(gs["sum"].to_numpy())]),
                               np.concatenate([[0.0], np.cumsum(gs["size"].to_numpy(dtype="float64"))]))
        self.Lend = {s: float(v[1][-1] / v[2][-1]) for s, v in self.by.items() if s in self.train_seasons}
        self.arm = arm
        self.slopes = {}
        self.n0 = None

    def prior(self, s: int) -> float:
        if (s - 1) not in self.Lend:
            return self.Lbar
        if self.arm == "T1":
            return self.Lend[s - 1]
        if s not in self.slopes:
            self.slopes[s] = trend_slope(s)
        return float(expit(logit(self.Lend[s - 1]) + self.slopes[s]))

    def before(self, s: int, dates) -> tuple[np.ndarray, np.ndarray]:
        """(N_tov, N) over season s strictly before each date."""
        dates = pd.to_datetime(dates).values.astype("datetime64[D]")
        if s not in self.by:
            return np.zeros(len(dates)), np.zeros(len(dates))
        d, cn, cd = self.by[s]
        k = np.searchsorted(d, dates, side="left")
        return cn[k], cd[k]

    def level(self, s: int, dates, n0: float | None = None) -> np.ndarray:
        n0 = self.n0 if n0 is None else n0
        nt, nd = self.before(s, dates)
        pr = self.prior(s)
        if np.isinf(n0):
            return np.full(len(nt), pr)
        with np.errstate(invalid="ignore", divide="ignore"):
            L = (nt + n0 * pr) / (nd + n0)
        return np.where(nd + n0 > 0, L, pr)

    def fit_n0(self) -> dict:
        fit_seasons = [s for s in self.train_seasons if (s - 1) in self.Lend]
        out = {}
        for n0 in N0_GRID:
            ll = 0.0
            for s in fit_seasons:
                d, cn, cd = self.by[s]
                L = np.clip(self.level(s, d, n0), 1e-9, 1 - 1e-9)
                nt, nn = np.diff(cn), np.diff(cd)
                ll += float(np.sum(nt * np.log(L) + (nn - nt) * np.log(1 - L)))
            out[n0] = ll
        self.n0 = max(out, key=out.get)
        return {"fit_seasons": fit_seasons, "loglik": {str(k): round(v, 3) for k, v in out.items()},
                "n0": str(self.n0)}

    def offset(self, season, dates) -> np.ndarray:
        season = np.asarray(season).astype("int64")
        out = np.zeros(len(season))
        dates = pd.to_datetime(pd.Series(dates)).to_numpy()
        for s in np.unique(season):
            m = season == s
            out[m] = logit(self.level(int(s), dates[m])) - logit(self.Lbar)
        return out

    def meta(self) -> dict:
        ss = sorted(self.by)
        return {"arm": self.arm, "Lbar": self.Lbar, "n0": str(self.n0),
                "prior_by_season": {str(s): self.prior(s) for s in ss},
                "Lend": {str(k): v for k, v in self.Lend.items()},
                "trend_slope_by_season": {str(k): v for k, v in self.slopes.items()}}


def build_level(design: pd.DataFrame, fold: str, season: int, arm: str) -> tuple[TovLevel, dict]:
    m = ((design["population"] == "first") & design["season"].isin([*B.FOLD_TRAIN[fold], season])).to_numpy()
    sub = design.loc[m]
    lv = TovLevel(sub["season"].to_numpy(), sub["game_date"].to_numpy(), sub["y"].to_numpy(),
                  B.FOLD_TRAIN[fold], arm)
    fit = lv.fit_n0()
    return lv, fit


def make_add_columns(arm: str, store: dict):
    def add_anchor_columns(design: pd.DataFrame, fold: str, season: int):
        d = design.copy()
        d[A.OFF_COLS] = 0.0
        lv, fit = build_level(d, fold, season, arm)
        m = ((d["population"] == "first") & d["season"].isin([*B.FOLD_TRAIN[fold], season])).to_numpy()
        d.loc[m, A.OFF_COLS[TOV]] = lv.offset(d.loc[m, "season"].to_numpy(), d.loc[m, "game_date"].to_numpy())
        meta = dict(lv.meta(), n0_fit=fit)
        store["meta"] = meta
        print(json.dumps({"arm": arm, "n0": meta["n0"], "Lbar": round(lv.Lbar, 5),
                          "prior_by_season": meta["prior_by_season"]}), flush=True)
        return d, meta
    return add_anchor_columns


def make_mark_first(arm: str):
    def mark_first(built: Path, anc_meta: dict) -> dict:
        mark = {"arm": arm, "kind": "binary_tov_only", "cols": [TOV], "classes": list(PO.CLASSES),
                "definition": "train_possession_outcome_s1_tovlevel_v1.TovLevel; offset[:,TOV] = logit L - logit Lbar",
                "Lbar": anc_meta["Lbar"], "n0": anc_meta["n0"], "prior_by_season": anc_meta["prior_by_season"],
                "serve": "ENGINE_SEASON_ANCHOR=<npz from build_engine_tovlevel_offsets_v1.py>",
                "writer": "train_possession_outcome_s1_tovlevel_v1"}
        idx = json.loads((built / "index.json").read_text(encoding="utf-8"))
        for sg in idx["populations"]["first"]["segments"]:
            p = built / sg["file"]
            w = joblib.load(p)
            w["anchor"] = dict(mark)
            joblib.dump(w, p, compress=3)
        idx["populations"]["first"]["anchor"] = dict(mark)
        (built / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")
        return mark
    return mark_first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["T1", "T2"], required=True)
    ap.add_argument("--fold", default="F2", choices=sorted(B.FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--design", default=str(B.R2_DESIGN))
    ap.add_argument("--verdict", default=str(B.R2_VERDICT))
    ap.add_argument("--engine-dir", default=str(B.ENGINE_DIR))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--level-only", action="store_true", help="print the level / n0 fit and exit")
    a = ap.parse_args()
    store: dict = {}
    if a.level_only:
        d = pd.read_parquet(a.design, columns=["season", "population", "y", "game_date"])
        lv, fit = build_level(d, a.fold, a.season, a.arm)
        print(json.dumps(dict(lv.meta(), n0_fit=fit), indent=1, default=str))
        return 0
    A.add_anchor_columns = make_add_columns(a.arm, store)
    AA.mark_first = make_mark_first(a.arm)
    ns = argparse.Namespace(anchor=a.arm, fold=a.fold, season=a.season, seed=a.seed, design=a.design,
                            verdict=a.verdict, engine_dir=a.engine_dir, team_rate_table="",
                            team_rate_missing="raise", out_root=a.out_root, n_jobs=a.n_jobs, stop_at="",
                            max_cuts=0, test_n_estimators=0, sample_games=0)
    r = AA.run(ns)
    out = Path(a.out_root)
    rp = out / "par_anchor_artifacts_v1_report.json"
    if rp.exists():
        rep = json.loads(rp.read_text(encoding="utf-8"))
        rep["anchor"] = a.arm
        rep["tovlevel_meta"] = store.get("meta")
        (out / "tovlevel_v1_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return 0 if r.get("complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
