import sys, json, pickle
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
import grade_clock_r8_offline_v1 as G
from cbb_sim.models import clock_r8
CKD = G.CKD
d = pd.read_parquet('data/processed/models/clock/r6_L2/design_v2.parquet', columns=[c for c in G.COLS if c != 'censored'] + ['censored'])
d['days_since_start'] = clock_r8.season_day(d)
d = d[(d.season == 2025) & (d.period <= 2)].reset_index(drop=True)
d = d.iloc[::10].reset_index(drop=True)
out = {}
a = G.load_schedule(CKD / 'r8_M2D/F2/manifest.json'); b = G.load_schedule(CKD / 'r8_M2D/F2_seed1/manifest.json')
mx = 0.0
for (ta, aa), (tb, bb) in zip(a, b):
    pa = aa.pmf(d.iloc[:30000].reset_index(drop=True)); pb = bb.pmf(d.iloc[:30000].reset_index(drop=True))
    mx = max(mx, float(np.abs(pa - pb).max()))
    assert np.array_equal(aa.coef, bb.coef)
print(json.dumps({"max_abs_pmf_diff_all_6_refits": mx, "coef_identical": True, "rows": 30000}))
