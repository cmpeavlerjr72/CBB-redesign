"""grade_team_rate_estimator_v4.py -- the Stage C S3 pre-check (experiments.md section 7).

Is O1a's variance calibrated around the V0 POINT estimates? In S3 the means stay V0, and the draw uses
O1a's estimation variance v and overdispersion phi. The means come from the V0 rows and (v, phi) from
the O1a rows of the same team-game-rate; both are in results/team_rate_estimator/estimates_q1_v3.parquet.
The metrics and bands are the v3 grader's `band_table` (variance ratio, z SD) with no other change.
The V0 line with V0's own v is printed alongside.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_team_rate_estimator_v3 as G3  # noqa: E402

OUT = Path("results/team_rate_estimator")


def main():
    e = G3.prep(OUT / "estimates_q1_v3.parquet")
    key = ["fold", "rs", "game_id", "team_id"]
    v0 = e[e["arm"] == "V0"]
    o1 = e[e["arm"] == "O1a"][key + ["v", "phi"]].rename(columns={"v": "v_o1", "phi": "phi_o1"})
    s3 = v0.merge(o1, on=key, how="inner")
    assert len(s3) == len(v0)
    s3 = s3.assign(v=s3["v_o1"], phi=s3["phi_o1"], arm="S3: V0 mean + O1a (v, phi)")
    both = pd.concat([v0.assign(arm="S2: V0 mean + V0 v"), s3.drop(columns=["v_o1", "phi_o1"])], ignore_index=True)
    tv = G3.true_var(both)
    B = G3.band_table(both, tv)
    B.to_json(OUT / "grade_s3_check_v4.json", orient="records", indent=1)
    pd.set_option("display.width", 250)
    for fold in ("F2", "F1"):
        g = B[B["fold"] == fold].groupby(["arm", "band"]).agg(var_ratio=("var_ratio", "median"), z_sd=("z_sd", "median"),
                                                              v_in=("var_ratio", lambda s: int(G3.inband(s).sum())),
                                                              z_in=("z_sd", lambda s: int(G3.inband(s).sum())))
        print(fold); print(g.round(3).to_string())


if __name__ == "__main__":
    main()
