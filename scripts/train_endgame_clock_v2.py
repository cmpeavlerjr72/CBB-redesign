"""train_endgame_clock_v2.py -- clock experiments.md section 46, blocks A and B: offline fits and test-season
predictions for the ONE blind grader `grade_endgame_clock_v2.py` (committed bf83e6e before any fit).

Opus engine worker, 2026-10-14. NOT ADOPTED, NOT A DEFAULT. Laws: `cbb_sim.models.endgame_v2` (shared with the
engine). Folds: F1 trains 2022-23 (2022 + 2023 seasons) and tests 2023-24; F2 trains through 2023-24 and tests 2024-25.
2025-26 is never read (window frames stop at 2025, `build_endgame_window_v1._seal`).

Block A (`--block A`): rows = late-game window (`build_endgame_window_v1.clock_window`).
  CA0    served K2 (`r8_K2/<fold>`, S1 monthly; test rows scored by the refit at or before the game date with
         max_train_date < game date asserted, `train_endgame_clock_v1.served_pmfs`). Control.
  CA0s1  K2 refitted under seed 1 by its own trainer (`r8_K2/<fold>_seed1`): noise-floor arm, never selectable.
  CA1    ratio form, m in {20, 50, 100} (`--m`); CA2 cell-AFT, m in {20, 50, 100}; CA3 smooth AFT (no m).
  Training rows: the fold's training-season window rows (periods 1-2); the served law on them is the fold's FIRST S1
  refit, whose training data are exactly those seasons (as round 7). CA2/CA3 fixed effects: K2 conditional finest cell
  (`clock_env.cond_cell_key`; tempo tercile edges from the fold's training window rows), uncensored rows only.
  The fitted law is applied to the served test-row pmfs (marginal and end-class conditional).
Block B (`--block B`): rows = FPS (`grade_endgame_clock_v2.fps_rows`) of the event window (first chances).
  FP0    served possession_outcome (`train_endgame_event_v1.served`, S1) and served fg_make round4_B1 (S1 schedule:
         F2 `fg_make/round4/B1`, F1 `fold1_v1/fg/B1`; per refit `max_train_date < game date` asserted).
  FP0s1  noise-floor arm: the served predictions under the seed-1 label (the served artifacts are fixed objects; the
         block's own fits are deterministic and their `--seed 1` refits are identical: reseed proof). Never selectable.
  FP1    class ratio on trailing 1-3 FPS rows, m = 50 (s46 2.B); FP2 = FP1 + FPS make-logit intercept, m = 50.
  FP3    offline = FP2 (LF2 changes the foul-accrual law only); written as its own file so the grader reads it.
  Shot rows: an FPS first chance that ended in a FGA is matched to its fg_make design row (same game, period, offence,
  shot class; shot clock in [chance end clock, possession start clock], the latest such shot = the terminal one);
  unmatched shots get NaN (the same rows for every arm).

Writes `results/endgame_clock_v2/offline/<fold>/<A|B>/<arm>.npz` and engine artifacts
`data/processed/models/late_game/endgame_clock_v2/<fold>[_seed1]/` (never overwriting without EGC_OVERWRITE=1).

    .venv/Scripts/python.exe scripts/train_endgame_clock_v2.py --block A --fold F1 --arm CA1 --m 50
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_endgame_window_v1 as W  # noqa: E402
import grade_endgame_clock_v2 as GR  # noqa: E402

from cbb_sim.models import endgame_v2 as E2  # noqa: E402

OFF = ROOT / "results/endgame_clock_v2/offline"
CACHE = ROOT / "results/endgame_clock_v2/cache"
ART = ROOT / "data/processed/models/late_game/endgame_clock_v2"
FG_SCHED = {"F2": ROOT / "data/processed/models/fg_make/round4/B1", "F1": ROOT / "data/processed/models/fold1_v1/fg/B1"}
SHOTS = {1: "FGA_rim", 2: "FGA_jump2", 3: "FGA_3"}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _guard(f: Path) -> None:
    if f.exists() and os.environ.get("EGC_OVERWRITE") != "1":
        raise SystemExit(f"{f} exists; refusing to overwrite (EGC_OVERWRITE=1 for an own rerun)")


# =====================================================================================================================
# block A
# =====================================================================================================================
def served_A(fold: str, seed: int, which: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    import train_endgame_clock_v1 as T1
    p = CACHE / f"A_{fold}_s{seed}_{which}.npz"
    if p.exists():
        z = np.load(p)
        if len(z["pm"]) == len(df):
            return z["pm"].astype(np.float64), z["pe"].astype(np.float64)
    sched = T1.load_schedule(fold, seed)
    pm, pe = T1.served_pmfs(df, sched, first_only=(which == "train"))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(p, pm=pm.astype(np.float32), pe=pe.astype(np.float32))
    return pm, pe


def fit_A(a) -> int:
    from cbb_sim.models import clock_env as CE
    tr, te = W.split(W.clock_window(), a.fold)
    seed = 1 if a.arm == "CA0s1" else a.seed
    np.random.seed(seed)
    log(f"A {a.fold} {a.arm} m={a.m} seed {seed}: train {len(tr):,} {sorted(tr.season.unique())}, test {len(te):,}")
    p0t, p0te = served_A(a.fold, 1 if a.arm == "CA0s1" else 0, "test", te)
    meta = {"arm": a.arm, "m": a.m, "fold": a.fold, "seed": seed, "created_at": datetime.now(UTC).isoformat(),
            "train_seasons": sorted(int(s) for s in tr["season"].unique()), "test_season": int(te["season"].iloc[0]),
            "spec": "clock/experiments.md s46 block A"}
    name = a.arm if a.arm in ("CA0", "CA0s1", "CA3") or a.final else f"{a.arm}_m{a.m}"
    if a.arm in ("CA0", "CA0s1"):
        pm, pe = p0t, p0te
    else:
        per, sec, mg = (tr["period"].to_numpy(), tr["seconds_remaining"].to_numpy(), tr["score_diff"].to_numpy())
        if a.arm == "CA1":
            p0, p0e = served_A(a.fold, 0, "train", tr)
            law = E2.fit_ca1(per, sec, mg, tr["duration_s"].to_numpy(), tr["censored"].to_numpy(bool), p0, p0e, a.m)
            fn = "ca1_law.npz"
        else:
            unc = ~tr["censored"].to_numpy(bool)
            t = tr.loc[unc]
            edges = CE.tempo_edges(t)
            key = CE.cond_cell_key(t, edges)
            args = (t["period"].to_numpy(), t["seconds_remaining"].to_numpy(), t["score_diff"].to_numpy(),
                    t["duration_s"].to_numpy(np.float64), key)
            law = E2.fit_ca2(*args, a.m) if a.arm == "CA2" else E2.fit_ca3(*args)
            law.info["tempo_edges"] = list(edges)
            fn = f"{a.arm.lower()}_law.json"
        meta["law_info"] = {k: v for k, v in law.info.items() if k in ("m", "n_rows", "fit_bucket", "fit_cell",
                                                                        "newton_iter", "n_fit_rows", "tempo_edges")}
        if a.arm == "CA3":
            meta["coef"] = np.asarray(law.beta).round(5).tolist()
            log("CA3 coef (P2 by bucket: b_L, b_T): " + " ".join(f"{x:+.3f}" for x in law.beta[10:]))
        if a.arm == "CA2":
            log("CA2 P2 beta (30,60] by margin bucket: " + " ".join(f"{x:+.3f}" for x in law.beta[1, :, 3]))
        args = (te["period"].to_numpy(), te["seconds_remaining"].to_numpy(), te["score_diff"].to_numpy())
        pm = law.apply(p0t, *args)
        pe = law.apply_end(p0te, *args, te["end1_code"].to_numpy())
        if a.final:
            d = ART / (a.fold if seed == 0 else f"{a.fold}_seed{seed}")
            d.mkdir(parents=True, exist_ok=True)
            f = d / fn
            _guard(f)
            law.save(f)
            meta["artifact"] = str(f.relative_to(ROOT)).replace("\\", "/")
    out = OFF / a.fold / "A"
    out.mkdir(parents=True, exist_ok=True)
    if seed != 0 and a.arm != "CA0s1":
        name = f"{name}_seed{seed}"
        out = OFF.parent / "offline_reseed" / a.fold / "A"
        out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", game_id=te["game_id"].to_numpy(np.int64),
                        period=te["period"].to_numpy(np.int64), poss_index=te["poss_index"].to_numpy(np.int64),
                        pmf=pm.astype(np.float32), pmf_end=pe.astype(np.float32), meta=np.array(json.dumps(meta)))
    log(f"wrote {out / name}.npz")
    return 0


# =====================================================================================================================
# block B
# =====================================================================================================================
def fg_p0(fold: str, rows: pd.DataFrame, first_only: bool) -> np.ndarray:
    """Served fg_make (round4_B1, S1) make probability for each FPS row's terminal first-chance shot; NaN when the
    chance did not end in a FGA or no design row matches."""
    import joblib

    from cbb_sim.models import fg_make as FG
    d = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet")
    ex = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v4_extra_v2.parquet", columns=["shooter_shrunk_dev_c"])
    d["shooter_shrunk_dev_c"] = ex["shooter_shrunk_dev_c"].to_numpy()
    del ex
    d = d[d["game_id"].isin(set(rows["game_id"].tolist())) & (d["period"] <= 2) & (d["seconds_remaining"] <= 31)]
    d = d.reset_index(drop=True)
    d["_d"] = np.arange(len(d))
    out = np.full(len(rows), np.nan)
    y = rows["y"].to_numpy()
    shot = np.flatnonzero(np.isin(y, list(SHOTS)))
    s = rows.iloc[shot][["game_id", "period", "offense_team_id", "start_clock", "end_clock"]].copy()
    s["shot_class"] = [SHOTS[int(c)] for c in y[shot]]
    s["_r"] = shot
    j = s.merge(d[["game_id", "period", "off_team_id", "shot_class", "seconds_remaining", "_d"]],
                left_on=["game_id", "period", "offense_team_id", "shot_class"],
                right_on=["game_id", "period", "off_team_id", "shot_class"], how="inner")
    j = j[(j["seconds_remaining"] >= j["end_clock"]) & (j["seconds_remaining"] <= j["start_clock"])]
    j = j.sort_values(["_r", "seconds_remaining"]).drop_duplicates("_r", keep="first")
    j = j.drop_duplicates("_d", keep="first")
    dd = d.iloc[j["_d"].to_numpy()].reset_index(drop=True)
    rr = j["_r"].to_numpy()
    p = np.full(len(dd), np.nan)
    for cls in FG.SHOT_CLASSES:
        man = json.loads((FG_SCHED[fold] / f"manifest_{cls}.json").read_text(encoding="utf-8"))
        feats = man["features"]
        arts = man["artifacts"]
        cuts = np.array([np.datetime64(x["refit_date"], "ns") for x in arts])
        mtd = np.array([np.datetime64(x["max_train_date"], "ns") for x in arts])
        ci = np.flatnonzero(dd["shot_class"].to_numpy() == cls)
        if not len(ci):
            continue
        gd = pd.to_datetime(dd["game_date"].to_numpy()[ci]).to_numpy()
        seg = np.zeros(len(ci), dtype=np.int64) if first_only else np.searchsorted(cuts, gd, side="right") - 1
        if not first_only and ((seg < 0).any() or not (mtd[seg] < gd).all()):
            raise SystemExit("created_at < tipoff violated for an fg_make test row")
        for k in np.unique(seg):
            r = ci[seg == k]
            w = joblib.load(FG_SCHED[fold] / arts[int(k)]["path"])
            p[r] = w["model"].predict_proba(FG.design_matrix(dd.iloc[r], feats))[:, 1]
    out[rr] = p
    return out


def fit_B(a) -> int:
    import train_endgame_event_v1 as TE
    seed = 1 if a.arm == "FP0s1" else a.seed
    np.random.seed(seed)
    tr, te = W.split(W.event_window(), a.fold)
    tr, te = GR.fps_rows(tr), GR.fps_rows(te)
    log(f"B {a.fold} {a.arm}: train FPS {len(tr):,}, test FPS {len(te):,}")
    cp = CACHE / f"B_{a.fold}_served.npz"
    if cp.exists():
        z = np.load(cp)
        p0t, m0t, p0, m0 = z["p0t"], z["m0t"], z["p0"], z["m0"]
    else:
        p0t, p0 = TE.served(te, a.fold, first_only=False), TE.served(tr, a.fold, first_only=True)
        m0t, m0 = fg_p0(a.fold, te, first_only=False), fg_p0(a.fold, tr, first_only=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cp, p0t=p0t, m0t=m0t, p0=p0, m0=m0)
    log(f"make rows matched: test {np.isfinite(m0t).sum()} of {np.isin(te['y'], list(SHOTS)).sum()} FGA, "
        f"train {np.isfinite(m0).sum()} of {np.isin(tr['y'], list(SHOTS)).sum()}")
    meta = {"arm": a.arm, "fold": a.fold, "seed": seed, "created_at": datetime.now(UTC).isoformat(),
            "train_seasons": sorted(int(s) for s in tr["season"].unique()), "test_season": int(te["season"].iloc[0]),
            "spec": "clock/experiments.md s46 block B", "make_rows_test": int(np.isfinite(m0t).sum())}
    args_te = (te["period"].to_numpy(), te["start_clock"].to_numpy(), te["start_score_diff"].to_numpy())
    if a.arm in ("FP0", "FP0s1"):
        # FP0s1: the served possession_outcome / fg_make artifacts are fixed objects; the block's own fits (FP1, FP2)
        # are deterministic, so their seed-1 refits equal the seed-0 fits (reseed proof, `--seed 1`). The reseed
        # difference that enters the floor is therefore 0, carried by FP0s1 = the served predictions under the seed-1
        # label (as CA0s1 = the K2 seed-1 refit, identical to CA0).
        p, pm = p0t, m0t
    else:
        f = E2.in_fps(tr["period"].to_numpy(), tr["start_clock"].to_numpy(), tr["start_score_diff"].to_numpy())
        r, ic = E2.fit_fp_classes(tr["period"].to_numpy()[f], tr["y"].to_numpy()[f], p0[f])
        delta, im = np.zeros(2), {}
        if a.arm in ("FP2", "FP3"):
            made = np.full(len(tr), np.nan)
            yy = tr["y"].to_numpy()
            for c, col in ((1, "fgm_rim"), (2, "fgm_jump2"), (3, "fgm_3")):
                made[yy == c] = (tr[col].to_numpy()[yy == c] > 0)
            ok = f & np.isfinite(m0) & np.isfinite(made)
            delta, im = E2.fit_fp_make(tr["period"].to_numpy()[ok], made[ok], m0[ok])
        law = E2.FPLaw(r=r, delta=delta, arm=a.arm, info={"classes": ic, "make": im})
        log(f"{a.arm}: P2 class ratio " + " ".join(f"{x:.3f}" for x in r[1]) + f" | make delta {delta.round(4).tolist()}"
            + (f" {im}" if im else ""))
        p = law.apply(p0t, *args_te)
        pm = law.apply_make(m0t, *args_te)
        meta["law"] = {"r": r.tolist(), "delta": delta.tolist(), "make": im}
        if a.arm in ("FP1", "FP2") and seed == 0:
            d = ART / (a.fold if seed == 0 else f"{a.fold}_seed{seed}")
            d.mkdir(parents=True, exist_ok=True)
            fpath = d / f"{a.arm.lower()}_law.json"
            _guard(fpath)
            law.save(fpath)
            meta["artifact"] = str(fpath.relative_to(ROOT)).replace("\\", "/")
    out = OFF / a.fold / "B"
    name = a.arm
    if seed != 0 and a.arm != "FP0s1":
        out, name = OFF.parent / "offline_reseed" / a.fold / "B", f"{a.arm}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", game_id=te["game_id"].to_numpy(np.int64),
                        period=te["period"].to_numpy(np.int64), poss_index=te["poss_index"].to_numpy(np.int64),
                        prob=p, pmake=pm, meta=np.array(json.dumps(meta)))
    log(f"wrote {out / name}.npz")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["A", "B"], required=True)
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--m", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="A: the frozen-m fit (writes <arm>.npz and the artifact)")
    a = ap.parse_args()
    if a.m == int(a.m):
        a.m = int(a.m)
    return fit_A(a) if a.block == "A" else fit_B(a)


if __name__ == "__main__":
    raise SystemExit(main())
