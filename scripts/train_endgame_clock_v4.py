"""train_endgame_clock_v4.py -- clock experiments.md section 51 (PRE-REGISTERED by PM Fable 5.1, 2026-10-14
afternoon): offline fits of the cell term on the served K2 window law, round 3, for the ONE blind grader
`grade_endgame_clock_v2.py` (s51 extension committed a129828 BEFORE any s51 fit output existed).

Opus engine worker, 2026-10-14. NOT ADOPTED, NOT A DEFAULT. New versioned trainer; `train_endgame_clock_v3.py` (s49)
is untouched. Folds, served-pmf scoring, the seal and walk-forward checks are v3's unchanged.

Arms (`--arm`):
  CA0, CA0s1   served K2 / its seed-1 refit (noise floor), as s49.
  CA1E         s49 law re-fitted under this grader (disclosed): z_off within-cell scale (+ defence-z iff the s49
               fold-1 rule supports it; the rule is re-applied here on the fold-1 training rows and frozen to
               `offline_s51/ca1e_defz_decision.json`), then the CA1 ratio on the de-scaled baseline.
  CA1R         step 1: within-cell env scale log k = c_role,off z_off + c_role,def z_def, role in {leader, trailer,
               tied} by the offence's margin (`endgame_v2.env_design_role`, 6 columns), round-8 M2 Poisson
               pseudo-likelihood (`endgame_v2.poisson_fe`) with fixed effects = window cell (half x 45) x K2
               conditional finest cell, uncensored training window rows (periods 1-2). Both terms entered for every
               role, NO inclusion screen; the walk-forward fit alone sets their size. step 2: the CA1 ratio on
               durations and censoring times divided by k, against the served row law, m in {20, 50, 100} chosen
               blind on fold 1 (`grade_endgame_clock_v2.py s51-select-m`). Served: scale_pmf(CA1(p0), k_row) for
               the marginal and the end-class-conditional law. No coefficient is shrunk, capped or blended by hand.
Writes `results/endgame_clock_v2/offline_s51/<fold>/A/<arm>[_m<m>].npz` and, with `--final`, the engine artifacts
`data/processed/models/late_game/endgame_clock_s51/<fold>/` (never overwriting without EGC_OVERWRITE=1).

    .venv/Scripts/python.exe scripts/train_endgame_clock_v4.py --fold F1 --arm CA1R --m 50
    .venv/Scripts/python.exe scripts/train_endgame_clock_v4.py --fold F2 --arm CA1R --m <chosen> --final
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_endgame_window_v1 as W  # noqa: E402
import train_endgame_clock_v2 as T2  # noqa: E402

from cbb_sim.models import endgame_v2 as E2  # noqa: E402

OFF = ROOT / "results/endgame_clock_v2/offline_s51"
ART = ROOT / "data/processed/models/late_game/endgame_clock_s51"
DEFZ = OFF / "ca1e_defz_decision.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _guard(f: Path) -> None:
    if f.exists() and os.environ.get("EGC_OVERWRITE") != "1":
        raise SystemExit(f"{f} exists; refusing to overwrite (EGC_OVERWRITE=1 for an own rerun)")


def fe_key(t, edges) -> np.ndarray:
    from cbb_sim.models import clock_env as CE
    c = E2.cell_index(t["period"].to_numpy(), t["seconds_remaining"].to_numpy(), t["score_diff"].to_numpy())
    k = CE.cond_cell_key(t, edges).astype(np.int64)
    return c.astype(np.int64) * (int(k.max()) + 1) + k


def env_fit(tr, cols: list[str], design=None) -> tuple[np.ndarray, np.ndarray, dict]:
    from cbb_sim.models import clock_env as CE
    unc = ~tr["censored"].to_numpy(bool)
    t = tr.loc[unc]
    edges = CE.tempo_edges(t)
    key = fe_key(t, edges)
    X = design(t) if design is not None else E2.env_design(t, cols)
    y = t["duration_s"].to_numpy(np.float64)
    c, info = E2.fit_env_within(y, X, key)
    se = E2.poisson_fe_cluster_se(y, X, key, t["game_id"].to_numpy(), c)
    info.update({"cols": cols, "coef": c.tolist(), "cluster_se": se.tolist(), "tempo_edges": list(edges),
                 "n_fe_cells": int(len(np.unique(key))), "nan_rating_rows": int(np.isnan(
                     tr[["off_rating_off_c", "def_rating_def_c"]].to_numpy(np.float64)).any(1).sum())})
    return c, se, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--arm", choices=["CA0", "CA0s1", "CA1E", "CA1R"], required=True)
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--final", action="store_true", help="writes <arm>.npz and the engine artifact")
    a = ap.parse_args()
    seed = 1 if a.arm == "CA0s1" else 0
    np.random.seed(seed)
    tr, te = W.split(W.clock_window(), a.fold)
    log(f"{a.fold} {a.arm} m={a.m}: train {len(tr):,} {sorted(tr.season.unique())}, test {len(te):,} "
        f"{sorted(te.season.unique())}")
    if int(te["season"].max()) >= 2026 or int(tr["season"].max()) >= int(te["season"].min()):
        raise SystemExit("fold seasons violate walk-forward / the 2025-26 seal")
    p0t, p0te = T2.served_A(a.fold, seed, "test", te)
    meta = {"arm": a.arm, "m": a.m if a.arm in ("CA1E", "CA1R") else None, "fold": a.fold, "seed": seed,
            "created_at": datetime.now(UTC).isoformat(), "train_seasons": sorted(int(s) for s in tr.season.unique()),
            "test_season": int(te["season"].iloc[0]), "spec": "clock/experiments.md s51"}
    args = (te["period"].to_numpy(), te["seconds_remaining"].to_numpy(), te["score_diff"].to_numpy())
    if a.arm in ("CA0", "CA0s1"):
        pm, pe = p0t, p0te
        name = a.arm
    else:
        per, sec, mg = (tr["period"].to_numpy(), tr["seconds_remaining"].to_numpy(), tr["score_diff"].to_numpy())
        p0, p0e = T2.served_A(a.fold, 0, "train", tr)
        y, cen = tr["duration_s"].to_numpy(), tr["censored"].to_numpy(bool)
        if a.arm == "CA1R":
            dsg = lambda d: E2.env_design_role(d, d["score_diff"].to_numpy())  # noqa: E731
            c, se, info = env_fit(tr, list(E2.CA1R_COLS), dsg)
            log(f"CA1R env scale {E2.CA1R_COLS}: c = {c.round(5).tolist()} (clustered se {se.round(5).tolist()}); "
                f"{info['n_fit_rows']:,} uncensored rows, {info['n_fe_cells']:,} FE cells, iter {info['newton_iter']}")
            k = np.exp(dsg(tr) @ c)
            ca1 = E2.fit_ca1(per, sec, mg, E2.descale(y, k), cen, p0, p0e, a.m)
            law = E2.CA1RLaw(ca1=ca1, coef=c, info=info)
            pm, pe = law.apply(p0t, *args, df=te), law.apply_end(p0te, *args, df=te)
            meta["env"] = {"cols": list(E2.CA1R_COLS), "coef": c.tolist(), "cluster_se": se.tolist(),
                           "k_train_quantiles": np.quantile(k, [0.01, 0.5, 0.99]).round(4).tolist(),
                           "n_train_by_role": {r: int((np.sign(tr["score_diff"].to_numpy()) == v).sum())
                                               for r, v in E2.ROLE_ORDER}}
        else:
            if a.fold == "F1" and not DEFZ.exists():
                c2, se2, i2 = env_fit(tr, ["zoff", "defz"])
                inc = bool(abs(c2[1]) > 2 * se2[1])
                OFF.mkdir(parents=True, exist_ok=True)
                DEFZ.write_text(json.dumps({"rule": "include def_c iff |c_def| > 2 x game-clustered SE on fold-1 "
                                                    "training rows (X = [z_off, def_c])", "include_defz": inc,
                                            "fit": i2, "written_at": datetime.now(UTC).isoformat()}, indent=1),
                                encoding="utf-8")
                log(f"defence-z decision (fold 1): c = {c2.round(5).tolist()} se = {se2.round(5).tolist()} -> "
                    f"include_defz = {inc}")
            if not DEFZ.exists():
                raise SystemExit("run --fold F1 --arm CA1E first (frozen defence-z decision)")
            cols = ["zoff", "defz"] if json.loads(DEFZ.read_text(encoding="utf-8"))["include_defz"] else ["zoff"]
            c, se, info = env_fit(tr, cols)
            log(f"CA1E env scale {cols}: c = {c.round(5).tolist()} (clustered se {se.round(5).tolist()}); "
                f"{info['n_fit_rows']:,} uncensored rows, {info['n_fe_cells']:,} FE cells, iter {info['newton_iter']}")
            k = np.exp(E2.env_design(tr, cols) @ c)
            ca1 = E2.fit_ca1(per, sec, mg, E2.descale(y, k), cen, p0, p0e, a.m)
            law = E2.CA1ELaw(ca1=ca1, cols=cols, coef=c, info=info)
            pm, pe = law.apply(p0t, *args, df=te), law.apply_end(p0te, *args, df=te)
            meta["env"] = {"cols": cols, "coef": c.tolist(), "cluster_se": se.tolist(),
                           "k_train_quantiles": np.quantile(k, [0.01, 0.5, 0.99]).round(4).tolist()}
        name = a.arm if a.final else f"{a.arm}_m{a.m}"
        if a.final:
            d = ART / a.fold
            d.mkdir(parents=True, exist_ok=True)
            if a.arm == "CA1R":
                _guard(d / "ca1r_env.json")
                law.save(d)
            else:
                _guard(d / "ca1e_env.json")
                law.save(d)
            meta["artifact_dir"] = str(d.relative_to(ROOT)).replace("\\", "/")
    out = OFF / a.fold / "A"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{name}.npz"
    _guard(f)
    pm32, pe32 = pm.astype(np.float32), pe.astype(np.float32)
    meta["pmf_sha16"] = hashlib.sha256(pm32.tobytes()).hexdigest()[:16]
    np.savez_compressed(f, game_id=te["game_id"].to_numpy(np.int64), period=te["period"].to_numpy(np.int64),
                        poss_index=te["poss_index"].to_numpy(np.int64), pmf=pm32, pmf_end=pe32,
                        meta=np.array(json.dumps(meta)))
    log(f"wrote {f} (pmf sha16 {meta['pmf_sha16']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
