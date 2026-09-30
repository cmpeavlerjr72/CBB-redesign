"""diag_team_rate_stageb_smoke_v1.py -- local Stage B SMOKE TEST (experiments.md section 7, step 4).

This is NOT a Stage B result. It checks that each served sub-model's own fit function runs end to end
on features substituted by `cbb_sim.team_rate_adapter`.

Setup: one refit date (2025-01-01), fold F2. The training rows are a 25% random sample of rows
strictly before the cut, to keep it local. LightGBM runs single-threaded, monkeypatched in THIS
process only; no file under src/ is edited. The served model modules' `fit_arm`/`predict_arm` or
`LgbmArm` are called unmodified. Lane J's box-parallel trainers were not committed at the time, so
the existing fit functions are called instead.

For each sub-model it prints the rows replaced, the rows kept at their served value (missing keys),
wall time, and the test-month log loss on served vs substituted features. The log losses are
plumbing checks only: a 25% sample and one cut decide nothing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim import team_rate_adapter as TRA  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

TABLE = Path("data/processed/team_rate_features_E3_v2.parquet")
CUT = pd.Timestamp("2025-01-01")
END = CUT + pd.Timedelta(days=31)
OUT = Path("results/team_rate_estimator")


def pin(cls):
    for attr in ("PARAMS", "params", "DEFAULT_PARAMS"):
        if hasattr(cls, attr) and isinstance(getattr(cls, attr), dict):
            getattr(cls, attr)["n_jobs"] = 1


for m in (PO, FG, RB):
    if hasattr(m, "LgbmArm"):
        pin(m.LgbmArm)


def split(d):
    d = d.assign(game_date=pd.to_datetime(d["game_date"]))
    tr = d[(d["season"] < 2025) | ((d["season"] == 2025) & (d["game_date"] < CUT))]
    te = d[(d["season"] == 2025) & (d["game_date"] >= CUT) & (d["game_date"] < END)]
    return tr.sample(frac=0.25, random_state=0), te


def run(name, d, sub, fit, pred):
    res = {}
    t0 = time.time()
    ds = TRA.apply(d, TABLE, sub, fold="F2", missing="keep_served")
    info = ds.attrs["team_rate_adapter"]
    for lab, frame in (("served", d), ("E3_v2", ds)):
        tr, te = split(frame)
        m = fit(tr)
        p = pred(m, te)
        res[lab] = {"n_train": len(tr), "n_test": len(te), "logloss": float(log_loss(te["y"], p, labels=np.arange(p.shape[1])))}
    res["adapter"] = {k: v for k, v in info.items() if k != "table"}
    res["seconds"] = round(time.time() - t0, 1)
    print(name, json.dumps(res), flush=True)
    return res


def main():
    out = {}
    idx = json.loads(Path("data/processed/models/engine/event_round2_s1_F2_2025/index.json").read_text(encoding="utf-8"))
    feats = list(idx["populations"]["first"]["features"])
    po = pd.read_parquet("data/processed/models/possession_outcome/round2/design.parquet")
    po = po[po["population"] == "first"].reset_index(drop=True)
    out["possession_outcome_first"] = run("possession_outcome/first", po, "possession_outcome",
                                          lambda tr: PO.fit_arm("lgbm", tr, feats, seed=0),
                                          lambda m, te: PO.predict_arm("lgbm", m, te, feats))
    del po
    fgd = pd.read_parquet("data/processed/models/fg_make/design_v2_shotshooter.parquet")
    ext = pd.read_parquet("data/processed/models/fg_make/design_v4_extra_v2.parquet", columns=["shooter_shrunk_dev_c"])
    fgd["shooter_shrunk_dev_c"] = ext["shooter_shrunk_dev_c"].to_numpy()
    fgd = fgd[fgd["shot_class"] == "FGA_rim"].reset_index(drop=True)
    ff = ["off_make_c", "def_allow_c", "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
          "site_home", "site_away", "season_idx", "period", "seconds_remaining", "in_bonus", "chance_number",
          "chance_elapsed_s", "is_transition_f", "shooter_shrunk_dev_c"]
    X = lambda df: np.ascontiguousarray(df[ff].to_numpy(dtype="float32"))  # noqa: E731
    out["fg_make_rim"] = run("fg_make/FGA_rim", fgd, "fg_make",
                             lambda tr: FG.LgbmArm(seed=0).fit(X(tr), tr["y"].to_numpy()),
                             lambda m, te: m.predict_proba(X(te)))
    del fgd
    rb = pd.read_parquet("data/processed/models/rebound/round3/design_round3.parquet")
    rf = ["off_oreb_c", "opp_def_dreb_c", "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
          "site_home", "site_away", "miss_rim", "miss_jump2", "miss_three", "blocked_f", "period", "seconds_remaining",
          "score_diff", "in_bonus"]
    out["rebound"] = run("rebound", rb, "rebound",
                         lambda tr: RB.fit_arm("lgbm", tr, rf, seed=0),
                         lambda m, te: RB.predict_arm("lgbm", m, te, rf))
    json.dump(out, open(OUT / "stageb_smoke_v1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
