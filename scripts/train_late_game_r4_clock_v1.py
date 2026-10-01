"""train_late_game_r4_clock_v1.py -- late-game ROUND 4 (b): leading-team window clock laws (experiments.md 9).

Round 2's `clk_D` (role3 x bucket x bonus x prev_end) puts leading-by-1-3 and leading-by-4-6 in one role
cell: the 1-3 team runs clock (18 s actual in (30,60]) and the 4-6 team is fouled early (7.5 s). Mixed, the
sim gets both wrong and adds possessions. Arm LGL refines role3 to round 1's five bands (`eg_role6`), the
same Kaplan-Meier family, window rows only, no floor. Fitted per fold on round 1's training slice and
predicted on round 1's held-out window rows in round 1's row order, so the PMFs drop into
`grade_late_game_r3_offline_v1`-style composites. Computes no metric.

Output: data/processed/models/late_game/round4/clk_LGL_F{1,2}.npy (test PMFs), clk_LGL.pkl (F2, served)
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from cbb_sim.models import clock as ck                               # noqa: E402
from cbb_sim.models import clock_v3 as c3                            # noqa: E402
from cbb_sim.models import late_game as LGM                          # noqa: E402

DESIGN_V2 = ROOT / "data/processed/models/clock/design_v2.parquet"
R1 = ROOT / "data/processed/models/late_game"
OUT = R1 / "round4"
ALL_SEASONS = [2022, 2023, 2024, 2025]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = pd.read_parquet(DESIGN_V2)
    d, _ = c3.attach_horn_censoring(d, ALL_SEASONS)
    c3.set_flag_inplace(d, "horn")
    d = c3.add_p_state(d)
    per2 = d["period"].to_numpy() == 2
    sec = d["seconds_remaining"].to_numpy()
    asd = np.abs(d["score_diff"].to_numpy())
    d["in_window"] = per2 & (sec <= LGM.GATE_SEC) & (asd <= LGM.GATE_MARGIN)
    order = ["game_id", "period", "poss_index"]
    for fold in ("F1", "F2"):
        tr_all, te_all = ck.fold_slices(d, fold)
        te = te_all[te_all["in_window"].to_numpy()].sort_values(order, kind="stable")
        tr = tr_all[tr_all["in_window"].to_numpy()]
        ref = pd.read_parquet(R1 / "round1_clock" / f"window_test_{fold}.parquet")
        if len(ref) != len(te) or not np.array_equal(ref["poss_index"].to_numpy(), te["poss_index"].to_numpy()):
            raise SystemExit(f"{fold}: test rows do not match round 1's window_test order")
        # reproduce D first: the same code path must give round 1's D PMF exactly
        dref = LGM.fit_cell_arm(tr, "LGD_dummy", 0, "D_clk|LGD_dummy|floor0").pmf(te.copy()).astype("float32")
        if not np.array_equal(dref, np.load(R1 / "round1_clock" / "pmf" / f"D_clk_{fold}.npy")):
            raise SystemExit(f"{fold}: D does not reproduce round 1; stop")
        arm = LGM.fit_cell_arm(tr, "LGL_dummy", 0, "LGL_clk|LGL_dummy|floor0")
        np.save(OUT / f"clk_LGL_{fold}.npy", arm.pmf(te.copy()).astype("float32"))
        print(f"{fold}: train {len(tr):,} test {len(te):,}; D reproduced exactly; LGL fitted "
              f"({time.time() - t0:.0f}s)", flush=True)
        if fold == "F2":
            with open(OUT / "clk_LGL.pkl", "wb") as f:
                pickle.dump(arm, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
