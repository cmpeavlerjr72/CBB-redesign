"""exp_chance_time_f1_parity_v1.py -- chance_time round 1, F1 confirmation (model-free; experiments.md s1.3).

Tables built on 2022-2023 (F1 LUT) applied to the 2024 real states; the fed chance_elapsed_s quantiles
(10/25/50/75/90) by class x chance bucket and the fed chance-1 transition share vs the real ones.
Score per arm: mean absolute quantile gap (s) over the 3 classes x 2 buckets x 5 quantiles, and the
mean absolute transition-share gap over the 3 classes. R's chance-2+ medians are the 2022-2023 design
medians by chance number (the engine builder's rule).

Usage: exp_chance_time_f1_parity_v1.py <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exp_chance_time_offline_v1 import real_rows, feed  # noqa: E402
from build_chance_time_lut_v1 import CLASSES  # noqa: E402

OUT = Path(sys.argv[1])
d = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet",
                    columns=["season", "chance_number", "chance_elapsed_s"])
d = d[d["season"].isin([2022, 2023])]
ce_med = {int(k): float(v) for k, v in d.groupby(d["chance_number"].clip(1, 3))["chance_elapsed_s"].median().items()}
lut = dict(np.load(ROOT / "data/processed/models/chance_time/F1/lut_v1.npz"))
m = real_rows(2024)
m = m[m["duration_s"].notna()].reset_index(drop=True)
Q = [.1, .25, .5, .75, .9]
first = (m["chance_number"] == 1).to_numpy()
res = {"n_rows": int(len(m)), "ce_med_F1": ce_med, "arms": {}}
for arm in ("R", "C2", "C12"):
    el, tr = feed(arm, m, lut, ce_med, np.random.default_rng(1000))
    gaps, tg = [], []
    det = {}
    for c in CLASSES:
        s = (m["shot_class"] == c).to_numpy()
        for nm, b in (("c1", first), ("c2p", ~first)):
            fq = np.quantile(el[s & b], Q)
            rq = np.quantile(m.loc[s & b, "chance_elapsed_s"], Q)
            gaps.extend(np.abs(fq - rq).tolist())
            det[f"{c}_{nm}"] = {"fed": fq.tolist(), "real": rq.tolist()}
        ft, rt = float(tr[s & first].mean()), float(m.loc[s & first, "is_transition_f"].mean())
        tg.append(abs(ft - rt))
        det[f"{c}_trans"] = {"fed": ft, "real": rt}
    res["arms"][arm] = {"mean_abs_quantile_gap_s": float(np.mean(gaps)),
                        "mean_abs_transition_gap": float(np.mean(tg)), "detail": det}
    print(arm, round(float(np.mean(gaps)), 3), round(float(np.mean(tg)), 4), flush=True)
OUT.write_text(json.dumps(res, indent=1))
