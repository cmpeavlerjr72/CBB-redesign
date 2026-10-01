#!/usr/bin/env python
"""Event-stream OT foul carry check (lane F, 2026-10-01): build_stream with CBB_OT_FOUL_CARRY unset vs "1", per season.
Regulation rows (period <= 2) must be DataFrame.equals; reports per-column OT row diffs. Writes results/ot_carry_event_stream.json."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src")]
from cbb_sim.models import event_stream as ES
seasons = [int(x) for x in sys.argv[1:]] or [2022, 2023, 2024, 2025]
assert 2026 not in seasons
u = ES.load_universe()
out = {}
for s in seasons:
    os.environ.pop(ES.OT_FOUL_CARRY_ENV, None); a = ES.build_stream(s, u)
    os.environ[ES.OT_FOUL_CARRY_ENV] = "1";    b = ES.build_stream(s, u)
    os.environ.pop(ES.OT_FOUL_CARRY_ENV, None)
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    reg = (a["period"] <= 2).to_numpy(); ot = ~reg
    d = {}
    for c in a.columns:
        x, y = a.loc[ot, c], b.loc[ot, c]
        ne = ~((x == y) | (x.isna() & y.isna())).to_numpy()
        if ne.any(): d[c] = int(ne.sum())
    anyd = np.zeros(len(a), bool)
    for c in d:
        x, y = a[c], b[c]; anyd |= (~((x == y) | (x.isna() & y.isna())).to_numpy()) & ot
    out[s] = {"rows": int(len(a)), "regulation_rows": int(reg.sum()),
              "regulation_bit_identical": bool(a[reg].reset_index(drop=True).equals(b[reg].reset_index(drop=True))),
              "ot_rows": int(ot.sum()), "ot_rows_any_difference": int(anyd.sum()), "ot_diff_by_column": d,
              "ot_mean_fouls_opp_prior_reset": float(a.loc[ot & (a["fouls_opp_prior"] >= 0), "fouls_opp_prior"].mean()),
              "ot_mean_fouls_opp_prior_carry": float(b.loc[ot & (b["fouls_opp_prior"] >= 0), "fouls_opp_prior"].mean())}
    print(s, json.dumps(out[s]), flush=True)
(R / "results/ot_carry_event_stream.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
