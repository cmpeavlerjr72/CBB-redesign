"""Post-hoc reclassification of the cells the per-date parity report left UNEXPLAINED (v3 replay).
Two further backtest-side causes, both verified per cell here:
  BT_REB_FILL_CONSTANT   reb_rate slot where the backtest value is the whole-season median constant (blank slot)
  BT_FG_TEAM_FALLBACK    fg team-form cell of a game that is not in the fg design (pbp_complete False): the backtest
                         builder backward-fills it from the team's earlier game, live computes it exactly
Anything left is reported as RESIDUAL."""
import json, glob, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ED = ROOT / "data/processed/models/engine"; LD = ROOT / "data/processed/models/engine_live"
gb = pd.read_parquet(ED / "games_F2_2025_v2.parquet"); zb = dict(np.load(ED / "arrays_F2_2025_v2.npz"))
nb = json.loads((ED / "names_F2_2025_v2.json").read_text(encoding="utf-8")); tn = nb["team_names"]
u = pd.read_parquet(ROOT / "data/processed/games_universe.parquet").set_index("game_id")
pos = {int(g): i for i, g in enumerate(gb["game_id"])}
real = zb["roster_cbbd"] > 0
consts = [float(pd.Series(zb["reb_rate"][..., k][real]).mode().iloc[0]) for k in range(2)]
tot = collections.Counter(); per = {}
for f in sorted(glob.glob(str(ROOT / "results/engine_v3_replay/dates/*.json"))):
    D = json.loads(Path(f).read_text(encoding="utf-8"))["date"]; tag = f"LIVE_F2_2025_{D}"
    g = pd.read_parquet(LD / f"games_{tag}.parquet"); z = np.load(LD / f"arrays_{tag}.npz")
    bi = np.array([pos[int(x)] for x in g["game_id"]]); bt = zb["team_static"][bi]
    same = (z["roster_cbbd"] == zb["roster_cbbd"][bi]) & (z["roster_cbbd"] > 0)
    c = collections.Counter()
    for k in range(2):
        d = (z["reb_rate"][..., k] != zb["reb_rate"][bi][..., k]) & same
        atc = np.isclose(zb["reb_rate"][bi][..., k], consts[k], atol=1e-7)
        c["BT_REB_FILL_CONSTANT"] += int((d & atc).sum()); c["residual_reb"] += int((d & ~atc).sum())
    no_po = (bt[:, :, :16] == 0).all(axis=2)
    inc = np.array([not bool(u.loc[int(x), "pbp_complete"]) for x in g["game_id"]])[:, None] & np.ones((1, 2), bool)
    for name, j in tn.items():
        if name.startswith(("off_make", "def_allow")):
            d = (z["team_static"][:, :, j] != bt[:, :, j]) & ~no_po
            c["BT_FG_TEAM_FALLBACK"] += int((d & inc).sum()); c["residual_fg_team"] += int((d & ~inc).sum())
    per[D] = dict(c); tot.update(c)
print(dict(tot)); Path(ROOT / "results/engine_v3_replay/posthoc.json").write_text(json.dumps({"total": dict(tot), "per_date": per}), encoding="utf-8")
