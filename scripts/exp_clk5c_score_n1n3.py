"""
exp_clk5c_score_n1n3.py -- round 5c: the two offline no-regression lines the
main grid ran out of clock for, N1 `CRPS_trunc` and N3 the PIT cell table.

Pre-registration: `docs/models/clock/experiments.md` section 24 (committed
`98f8db5`), metrics carried verbatim from sections 16.5 and 21.5. Scored by the
SAME blind path rounds 3, 3b, 3c, 4, 5 and 5b used -- `clock_v3.score_arm_v3`
and `clock_v3.pit_by_cell_v3`, both UNEDITED -- over
`clock_v5.LatentArm(..., loc_kind="plus_half")`, the B1 location, with the
per-row sigma each arm's fitted dispersion function produces. Fitted parameters
are re-read from `v5c_bakeoff/v5c_params.json`; NOTHING IS REFITTED here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, fn: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r5 = _load("exp_clk5", "exp_clk5_dispersion_bakeoff.py")
r5c = _load("exp_clk5c", "exp_clk5c_dispersion_function.py")
log = r5.log

from cbb_sim.models import clock_v3 as c3   # noqa: E402
from cbb_sim.models import clock_v5 as c5   # noqa: E402


def main() -> None:
    out = ROOT / "data" / "processed" / "models" / "clock" / "v5c_bakeoff"
    univ = pd.read_parquet(r5.UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    design = pd.read_parquet(r5.DESIGN)
    design = design[(design["period"] <= 2.0) & (design["game_id"].isin(cc))].copy()
    sched = r5.load_schedule("v3c_srfloor_P3_s1")
    te = r5c.annotate(design[design["season"] == 2025].copy(), sched, True)
    seg = te["_seg"].to_numpy()
    gt, _ = r5c.game_table(te)
    p2 = json.loads((out / "v5c_params.json").read_text())["F2"]
    p2["C1"]["table"] = {int(k): float(v) for k, v in p2["C1"]["table"].items()}

    rows = []
    for arm in ("R", "B1", "C1", "C2", "C4"):
        t0 = time.time()
        if arm == "R":
            def inner_of(k):
                return sched[int(k)]["arm"]
        else:
            s2g = np.clip(r5c.sigma2_rows(arm, gt, p2), 1e-8, None)
            mp = dict(zip(gt["game_id"].astype(int), s2g))

            def sig(df, mp=mp):
                return np.sqrt(np.array([mp[int(g)] for g in df["game_id"]]))

            def inner_of(k, sig=sig):
                return c5.LatentArm(sched[int(k)]["arm"], sig,
                                    loc_kind="plus_half")
        crps = np.empty(len(te)); pit = np.empty(len(te)); ll = np.empty(len(te))
        for k in np.unique(seg):
            r = np.flatnonzero(seg == k)
            s = c3.score_arm_v3(inner_of(k), te.iloc[r].reset_index(drop=True))
            crps[r] = s["_crps_trunc_rows"]; pit[r] = s["_pit_rows"]
            ll[r] = s["_loglik_rows"]
        unc = ~te["censored"].to_numpy(dtype=bool)
        cells = c3.pit_by_cell_v3(te, pit)
        pw = cells[cells["powered"]]
        rows.append({"arm": arm,
                     "crps_trunc": float(np.nanmean(crps[unc])),
                     "censored_loglik": float(np.mean(ll)),
                     "pit_worst_D": float(pw["ks_D"].max()) if len(pw) else np.nan,
                     "pit_leak_cells": int(pw["leak_sized"].sum()) if len(pw) else 0,
                     "n_powered_cells": int(len(pw)),
                     "secs": round(time.time() - t0, 1)})
        log(f"{arm}: CRPS {rows[-1]['crps_trunc']:.6f} loglik "
            f"{rows[-1]['censored_loglik']:.5f} PIT D {rows[-1]['pit_worst_D']:.5f} "
            f"leak {rows[-1]['pit_leak_cells']} ({rows[-1]['secs']}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "v5c_n1n3_F2.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
