"""diag_c4_d2_supp_v1.py -- supporting regressions quoted in the D2 verdict (game possessions on scoring environment and mismatch;
team points on own / opponent scoring strength). DIAGNOSTIC ONLY.  usage: diag_c4_d2_supp_v1.py --tag f2all50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_c4_lib_v1 as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    pg = pd.read_parquet(L.OUT / f"{a.tag}_pregame.parquet")
    rtg = pd.read_parquet(L.OUT / f"{a.tag}_real_tg.parquet")
    stg = pd.read_parquet(L.OUT / f"{a.tag}_sim_tg.parquet")
    S = stg.seed.nunique()
    s = int(pg._rating_sign.iloc[0])

    def side(tg, div):
        t = tg.groupby(["game_id", "off_side"])[["poss", "pts"]].sum() / div
        return t.xs(0, level=1).add_prefix("h_").join(t.xs(1, level=1).add_prefix("a_"))
    d = pg.set_index("game_id").join(side(rtg, 1).add_prefix("r_")).join(side(stg, S).add_prefix("s_"), how="inner")
    d = d[d.home_net.notna() & d.away_net.notna() & (d.r_h_poss + d.r_a_poss >= 100)].copy()
    d["env"] = d.home_off_c + d.away_off_c - s * (d.home_def_c + d.away_def_c)
    d["gap"] = (d.home_net - d.away_net).abs()
    d["xh"] = d.home_off_c - s * d.away_def_c
    d["xa"] = d.away_off_c - s * d.home_def_c
    print(f"games {len(d)}; SD env {d.env.std():.2f}, SD gap {d.gap.std():.2f}")

    def ols(cols, y):
        X = np.c_[np.ones(len(d)), d[cols].to_numpy()]
        return np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0][1:].round(4)
    for pre in ("r", "s"):
        poss = d[f"{pre}_h_poss"] + d[f"{pre}_a_poss"]
        print(pre, "game poss on env", ols(["env"], poss), "on gap", ols(["gap"], poss), "on env,gap", ols(["env", "gap"], poss),
              "on xh,xa", ols(["xh", "xa"], poss))
        print(pre, "home pts on xh,xa", ols(["xh", "xa"], d[f"{pre}_h_pts"]), "away pts on xh,xa", ols(["xh", "xa"], d[f"{pre}_a_pts"]),
              "total pts on xh,xa", ols(["xh", "xa"], d[f"{pre}_h_pts"] + d[f"{pre}_a_pts"]))


if __name__ == "__main__":
    main()
