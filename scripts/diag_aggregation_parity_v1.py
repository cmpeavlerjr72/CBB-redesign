"""diag_aggregation_parity_v1.py -- lane A 2026-09-30, docs/models/aggregation/experiments.md section 2.3
(addendum F): train/serve parity of fg_make's team-rate and shooter inputs. For every fold-2
(game, offence team, shot class) the TRAINING design's value (as `train_fg_make_v4_par_v1.py` builds it)
is compared with the ENGINE inputs' value (team_static / slot_static of the tagged stack).
`T` design (E3 v4 overlay + locally rebuilt E3 extra cache) vs `S1_laneA`; `R` design (served) vs `S0_laneA`.
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
import train_fg_make_v4_shooter_block as R4M  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.team_rate_adapter import apply as trt_apply  # noqa: E402

KEY = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}
PAIRS = {
    "T_vs_S1": (ROOT / "data/processed/team_rate_features_E3_v4.parquet",
                ROOT / "data/processed/models/fg_make/round_aggfix/design_v4_extra_E3_local_g0.parquet",
                ROOT / "data/processed/models/engine_v3_S1_laneA"),
    "R_vs_S0": (None, R4M.EXTRA_CACHE, ROOT / "data/processed/models/engine_v3_S0_laneA"),
}


def cmp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return {"n": int(ok.sum()), "corr": float(np.corrcoef(a, b)[0, 1]), "sd_design": float(a.std()),
            "sd_engine": float(b.std()), "sd_ratio_engine_over_design": float(b.std() / a.std()),
            "mean_diff_engine_minus_design": float((b - a).mean()),
            "share_absdiff_gt_0.01": float((np.abs(b - a) > 0.01).mean()),
            "slope_engine_on_design": float(np.cov(a, b)[0, 1] / a.var())}


def main() -> int:
    base = pd.read_parquet(R4M.DESIGN)
    out = {}
    for name, (table, cache, sdir) in PAIRS.items():
        d = base.copy()
        if table is not None:
            d = trt_apply(d, table, "fg_make", fold="F2", missing="raise")
        extra = pd.read_parquet(cache)
        for c in extra.columns:
            d[c] = extra[c].to_numpy()
        d = d[d["season"] == 2025]
        z = np.load(sdir / "arrays_F2_2025.npz")
        names = json.load(open(sdir / "names_F2_2025.json"))
        g = pd.read_parquet(sdir / "games_F2_2025.parquet")
        ts, ss, ros = z["team_static"], z["slot_static"], z["roster_cbbd"]
        tn, sn = names["team_names"], names["slot_names"]
        rows = []
        for side, col in ((0, "home_team_id"), (1, "away_team_id")):
            e = pd.DataFrame({"game_id": g["game_id"].to_numpy(), "off_team_id": g[col].to_numpy()})
            for cls, k in KEY.items():
                e2 = e.copy()
                e2["shot_class"] = cls
                e2["eng_off_make_c"] = ts[:, side, tn[f"off_make_c__{k}"]]
                e2["eng_def_allow_c"] = ts[:, side, tn[f"def_allow_c__{k}"]]
                rows.append(e2)
        eng = pd.concat(rows, ignore_index=True)
        tm = d.groupby(["game_id", "off_team_id", "shot_class"], as_index=False)[["off_make_c", "def_allow_c"]].first()
        m = tm.merge(eng, on=["game_id", "off_team_id", "shot_class"], how="inner")
        res = {"n_design_team_class_rows": int(len(tm)), "n_matched": int(len(m))}
        for cls in KEY:
            mm = m[m["shot_class"] == cls]
            res[f"off_make_c_{cls}"] = cmp(mm["off_make_c"], mm["eng_off_make_c"])
            res[f"def_allow_c_{cls}"] = cmp(mm["def_allow_c"], mm["eng_def_allow_c"])
        # shooter dev: design per (game, shooter, class) vs engine slot value for that player id
        srows = []
        for side, col in ((0, "home_team_id"), (1, "away_team_id")):
            for s in range(ros.shape[2]):
                for cls, k in KEY.items():
                    srows.append(pd.DataFrame({"game_id": g["game_id"].to_numpy(), "shooter_id": ros[:, side, s],
                                               "shot_class": cls,
                                               "eng_dev": ss[:, side, s, sn[f"shooter_shrunk_dev_c__{k}"]]}))
        se = pd.concat(srows, ignore_index=True)
        se = se[se["shooter_id"] >= 0]
        dsh = d.groupby(["game_id", "shooter_id", "shot_class"], as_index=False)["shooter_shrunk_dev_c"].first()
        ms = dsh.merge(se.drop_duplicates(["game_id", "shooter_id", "shot_class"]), on=["game_id", "shooter_id", "shot_class"], how="inner")
        res["n_shooter_rows_design"] = int(len(dsh)); res["n_shooter_matched"] = int(len(ms))
        for cls in KEY:
            mm = ms[ms["shot_class"] == cls]
            if len(mm) > 100:
                res[f"shooter_dev_{cls}"] = cmp(mm["shooter_shrunk_dev_c"], mm["eng_dev"])
        out[name] = res
        print(name, json.dumps({k: ({q: round(v, 4) for q, v in x.items()} if isinstance(x, dict) else x) for k, x in res.items()}, indent=0), flush=True)
    (ROOT / "results/aggregation_v1/analysis_parity_v1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
