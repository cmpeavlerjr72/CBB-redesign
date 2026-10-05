"""diag_g5_ftp_coupling_v1.py -- why the engine carries a negative cross-team Cov(FT%, FG value).

Variance worker 2026-10-05, DIAGNOSTIC ONLY. Within-game (sim, w = row - seed mean) vs
realised residual (r = actual - seed mean) correlations of each side's FT% with the
other side's FG points per FGA, with the side's own final margin, and FT% conditioned
on the side leading or trailing at the final horn (the late-game intentional-foul window
is when the leader shoots most of its free throws). Writes results/g5_vdecomp/<tag>_ftp.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402
run, season, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
g = pd.read_parquet(Path(run) / "games.parquet")
summary, raw = G.build_grading_frame(g, season)
summary = summary[summary["home_score"].fillna(0) + summary["away_score"].fillna(0) > 0]
raw = raw[raw["game_id"].isin(summary["game_id"])].reset_index(drop=True)
d = pd.DataFrame({"game_id": raw["game_id"]})
for s, o in (("h", "home"), ("a", "away")):
    fga = raw[f"{o}_fga3"] + raw[f"{o}_fga2_rim"] + raw[f"{o}_fga2_jump"]
    d[f"val_{s}"] = (2 * raw[f"{o}_fgm2_rim"] + 2 * raw[f"{o}_fgm2_jump"] + 3 * raw[f"{o}_fgm3"]) / fga
    d[f"fta_{s}"] = raw[f"{o}_fta"].astype(float)
    d[f"ftp_{s}"] = raw[f"{o}_ftm"] / raw[f"{o}_fta"].where(raw[f"{o}_fta"] > 0)
d["mar_h"] = raw["home_pts"].astype(float) - raw["away_pts"]
d["mar_a"] = -d["mar_h"]
tb = R.load_actual_team_box(season)
h = tb[tb["team_id"] == tb["home_team_id"]].drop_duplicates("game_id").set_index("game_id")
a = tb[tb["team_id"] == tb["away_team_id"]].drop_duplicates("game_id").set_index("game_id")
fin = summary.set_index("game_id")
c = fin.index.intersection(h.index).intersection(a.index)
A = pd.DataFrame(index=c)
for s, t in (("h", h.loc[c]), ("a", a.loc[c])):
    A[f"val_{s}"] = (2 * (t["fgm"] - t["tpm"]) + 3 * t["tpm"]) / t["fga"]
    A[f"fta_{s}"] = t["fta"].astype(float)
    A[f"ftp_{s}"] = t["ftm"] / t["fta"].where(t["fta"] > 0)
A["mar_h"] = (fin.loc[c, "home_score"] - fin.loc[c, "away_score"]).astype(float)
A["mar_a"] = -A["mar_h"]
cols = [k for k in A.columns]
M = d[d["game_id"].isin(c)].groupby("game_id")[cols].mean()
S = d[d["game_id"].isin(c)].reset_index(drop=True)
W = S[cols] - M.loc[S["game_id"], cols].to_numpy()
Rr = A[cols] - M.loc[c, cols]
def corr(D, x, y):
    ok = D[x].notna() & D[y].notna()
    return float(np.corrcoef(D.loc[ok, x], D.loc[ok, y])[0, 1])
out = {}
for lab, D in (("act", Rr), ("sim", W)):
    out[lab] = {
        "ftp_h~val_a": corr(D, "ftp_h", "val_a"), "ftp_a~val_h": corr(D, "ftp_a", "val_h"),
        "ftp_h~mar_h": corr(D, "ftp_h", "mar_h"), "ftp_a~mar_a": corr(D, "ftp_a", "mar_a"),
        "fta_h~mar_h": corr(D, "fta_h", "mar_h"), "fta_a~mar_a": corr(D, "fta_a", "mar_a"),
        "ftp_h~fta_h": corr(D, "ftp_h", "fta_h"), "ftp_h~val_h": corr(D, "ftp_h", "val_h")}
# level: FT% of the side that finished ahead vs behind (sim rows, actual games), relative to seed mean
for lab, raw_, D in (("act", A, Rr), ("sim", S, W)):
    lead = np.asarray(raw_["mar_h"] > 0)
    out[lab]["dftp_home_when_leading"] = float(np.nanmean(np.asarray(D["ftp_h"])[lead]))
    out[lab]["dftp_home_when_trailing"] = float(np.nanmean(np.asarray(D["ftp_h"])[~lead]))
Path(ROOT / "results/g5_vdecomp").mkdir(parents=True, exist_ok=True)
(ROOT / f"results/g5_vdecomp/{tag}_ftp.json").write_text(json.dumps(out, indent=1))
for k in out["act"]:
    print(f"{k:28s} act {out['act'][k]:+.4f}  sim {out['sim'][k]:+.4f}")
