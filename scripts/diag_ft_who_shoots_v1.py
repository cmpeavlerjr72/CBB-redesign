"""diag_ft_who_shoots_v1.py -- FT% channel, part 1: WHO goes to the line in the served sim vs reality (lane I, 2026-10-01)

DIAGNOSTIC ONLY. Nothing in src/ is changed. No new simulation: reads a finished full-size run's players.parquet.

Per (game, side, roster slot) of the v3 engine inputs:
  sim_fta   = mean FTA per simulated game (players.parquet, all seeds), sim_min = mean minutes;
  real_fta  = the player's real FTA in that game (hoopR player box, joined on the ESPN id of the slot), real_min.
Every slot is valued at the SERVED free-throw model's own make probability for its served shooter block, averaged over
a fixed sample of real 2024-25 attempt STATES (the same states for every slot), so the FT% implied by the sim's FTA
weights minus the FT% implied by the real FTA weights is the "who shoots" channel in probability points with the
state held fixed. Real FTA by players NOT on the engine roster is reported separately (off-roster).

Groups: has_prior_season (served slot value), as-of FTA bins, usage FT_trip-rate quintile (within side), minutes.

Usage: diag_ft_who_shoots_v1.py <run_dir> <input_dir> <out_json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from cbb_sim.engine.adapters import Adapters  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402

RUN, IN_DIR, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
FEATS = ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season", "season_idx",
         "seconds_remaining", "period", "score_diff", "in_bonus"]
SLOT_COL = {"shooter_ft_asof": "shooter_ft_asof", "shooter_fta_asof": "shooter_fta_asof",
            "prior_season_ft": "prior_season_ft", "has_prior_season": "has_prior_season_ft"}
N_STATES = 40

inp = EngineInputs.load(IN_DIR, "F2_2025")
ad = Adapters.load(inp, "F2", 2025)
games = inp.games.reset_index(drop=True)
G, _, S = inp.roster_cbbd.shape
pos = {int(g): i for i, g in enumerate(games["game_id"].to_numpy())}

# ---------------- slot frame ------------------------------------------------
fr = pd.DataFrame({"gidx": np.repeat(np.arange(G), 2 * S), "side": np.tile(np.repeat([0, 1], S), G),
                   "slot": np.tile(np.arange(S), 2 * G), "pid": inp.roster_cbbd.reshape(-1),
                   "espn": inp.roster_espn.reshape(-1), "valid": inp.roster_valid.reshape(-1)})
for c, sc in SLOT_COL.items():
    fr[c] = inp.slot_static[..., inp.slot_names[sc]].reshape(-1)
ft_idx = list(inp.usage_classes).index("FT_trip") if "FT_trip" in inp.usage_classes else None
fr["u_ft"] = inp.usage_rate[..., ft_idx].reshape(-1) if ft_idx is not None else np.nan
fr["rot_share"] = inp.rot_share.reshape(-1)
fr["game_id"] = games["game_id"].to_numpy()[fr["gidx"]]
fr["team_id"] = np.where(fr["side"] == 0, games["home_team_id"].to_numpy()[fr["gidx"]],
                         games["away_team_id"].to_numpy()[fr["gidx"]])

# ---------------- sim -------------------------------------------------------
pf = pq.ParquetFile(RUN / "players.parquet")
parts = []
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=["game_id", "seed", "cbbd_id", "team_id", "fta", "minutes"]).to_pandas()
    t = pf.read_row_group(i, columns=["game_id", "seed", "cbbd_id", "team_id", "fta", "minutes", "pts",
                                      "fgm2_rim", "fgm2_jump", "fgm3"]).to_pandas()
    t["ftm"] = t["pts"] - 2 * (t["fgm2_rim"] + t["fgm2_jump"]) - 3 * t["fgm3"]
    parts.append(t.groupby(["game_id", "team_id", "cbbd_id"], as_index=False)[["fta", "ftm", "minutes"]].sum())
sp = pd.concat(parts).groupby(["game_id", "team_id", "cbbd_id"], as_index=False)[["fta", "ftm", "minutes"]].sum()
gm = pd.read_parquet(RUN / "games.parquet", columns=["game_id", "seed", "home_fta", "away_fta", "home_ftm", "away_ftm"])
nseed = gm.groupby("game_id")["seed"].nunique()
sp["ns"] = sp["game_id"].map(nseed)
sp["sim_fta"] = sp["fta"] / sp["ns"]
sp["sim_ftm"] = sp["ftm"] / sp["ns"]
sp["sim_min"] = sp["minutes"] / sp["ns"]
g_fta, g_ftm = float((gm["home_fta"] + gm["away_fta"]).sum()), float((gm["home_ftm"] + gm["away_ftm"]).sum())
n_fta, n_ftm = float(sp["fta"].sum()), float(sp["ftm"].sum())
res: dict = {"run": str(RUN), "n_seeds_median": float(nseed.median()),
             "sim_fta_games_total_per_game": g_fta / nseed.sum(),
             "sim_fta_players_total_per_game": float(sp["sim_fta"].sum() / len(nseed)),
             "sim_ft_pct_all": g_ftm / g_fta, "sim_ft_pct_named_players": n_ftm / n_fta,
             "sim_unattributed_fta_share": 1 - n_fta / g_fta,
             "sim_ft_pct_unattributed": (g_ftm - n_ftm) / max(g_fta - n_fta, 1)}
fr = fr.merge(sp[["game_id", "team_id", "cbbd_id", "sim_fta", "sim_ftm", "sim_min"]].rename(columns={"cbbd_id": "pid"}),
              on=["game_id", "team_id", "pid"], how="left")
fr[["sim_fta", "sim_ftm", "sim_min"]] = fr[["sim_fta", "sim_ftm", "sim_min"]].fillna(0.0)

# ---------------- real ------------------------------------------------------
pb = pd.read_parquet("data/raw/hoopr/player_box/player_box_2025.parquet",
                     columns=["game_id", "athlete_id", "team_id", "minutes", "free_throws_attempted",
                              "free_throws_made", "did_not_play"])
pb = pb[pb["game_id"].isin(pos)]
pb["game_id"] = pb["game_id"].astype("int64")
pb["athlete_id"] = pb["athlete_id"].astype("int64")
pb["team_id"] = pb["team_id"].astype("int64")
real = pb.groupby(["game_id", "team_id", "athlete_id"], as_index=False)[
    ["free_throws_attempted", "free_throws_made", "minutes"]].sum()
real = real.rename(columns={"athlete_id": "espn", "free_throws_attempted": "real_fta",
                            "free_throws_made": "real_ftm", "minutes": "real_min"})
fr = fr.merge(real, on=["game_id", "team_id", "espn"], how="left", indicator=True)
fr["on_box"] = fr["_merge"] == "both"
fr = fr.drop(columns="_merge")
fr[["real_fta", "real_ftm", "real_min"]] = fr[["real_fta", "real_ftm", "real_min"]].fillna(0.0)
matched = real.merge(fr.loc[fr["espn"] > 0, ["game_id", "team_id", "espn"]].drop_duplicates(),
                     on=["game_id", "team_id", "espn"], how="left", indicator=True)
off = matched["_merge"] == "left_only"
games_with_box = real["game_id"].nunique()
res["real"] = {"games_with_player_box": int(games_with_box),
               "fta_per_game": float(real["real_fta"].sum() / games_with_box),
               "fta_off_roster_share": float(real.loc[off.to_numpy(), "real_fta"].sum() / real["real_fta"].sum()),
               "ftm_over_fta_off_roster": float(real.loc[off.to_numpy(), "real_ftm"].sum()
                                                / max(real.loc[off.to_numpy(), "real_fta"].sum(), 1)),
               "ftm_over_fta_on_roster": float(real.loc[~off.to_numpy(), "real_ftm"].sum()
                                               / real.loc[~off.to_numpy(), "real_fta"].sum())}
# restrict both sides to games that have a real player box (like for like)
fr = fr[fr["game_id"].isin(set(real["game_id"]))].reset_index(drop=True)

# ---------------- value every slot at the served model ----------------------
ES.load_universe()
att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
att = att[att["season"].isin([2022, 2023, 2024, 2025])]
d = FT.build_ft_design(att)
d = d[(d["season"] == 2025) & d["game_id"].isin(pos)]
st = d.sample(N_STATES, random_state=20261001)[["seconds_remaining", "period", "score_diff", "in_bonus"]]
segs = ad.ft.manifest.segments(fr["gidx"].to_numpy().astype(np.int64))
X0 = np.zeros((len(fr), len(FEATS)), dtype=np.float32)
for c in SLOT_COL:
    X0[:, FEATS.index(c)] = fr[c].to_numpy()
X0[:, FEATS.index("season_idx")] = 3.0
p = np.zeros(len(fr))
for _, row in st.iterrows():
    X = X0.copy()
    for c in ("seconds_remaining", "period", "score_diff", "in_bonus"):
        X[:, FEATS.index(c)] = row[c]
    for k in np.unique(segs):
        r = np.flatnonzero(segs == k)
        p[r] += ad.ft.models_by_seg[k].predict_proba(X[r])[:, 1]
fr["p"] = p / N_STATES


def implied(w):
    w = fr[w].to_numpy()
    return float((w * fr["p"]).sum() / w.sum())


res["implied_ft_pct"] = {"sim_weights": implied("sim_fta"), "real_weights_on_roster": implied("real_fta")}
res["implied_ft_pct"]["who_shoots_pp"] = 100 * (res["implied_ft_pct"]["sim_weights"]
                                                - res["implied_ft_pct"]["real_weights_on_roster"])
res["real_ftm_over_fta_on_roster_slots"] = float(fr["real_ftm"].sum() / fr["real_fta"].sum())


def group_table(key, labels=None):
    g = fr.groupby(key)
    out = []
    tot_s, tot_r = fr["sim_fta"].sum(), fr["real_fta"].sum()
    ps, pr = res["implied_ft_pct"]["sim_weights"], res["implied_ft_pct"]["real_weights_on_roster"]
    for k, x in g:
        ss, rr = x["sim_fta"].sum() / tot_s, x["real_fta"].sum() / tot_r
        pk = float((x["p"] * (x["sim_fta"] + x["real_fta"])).sum() / max((x["sim_fta"] + x["real_fta"]).sum(), 1e-9))
        out.append({"group": str(k), "slots": int(len(x)), "sim_fta_share": float(ss), "real_fta_share": float(rr),
                    "sim_min_share": float(x["sim_min"].sum() / fr["sim_min"].sum()),
                    "real_min_share": float(x["real_min"].sum() / fr["real_min"].sum()),
                    "sim_fta_per_40": float(40 * x["sim_fta"].sum() / max(x["sim_min"].sum(), 1e-9)),
                    "real_fta_per_40": float(40 * x["real_fta"].sum() / max(x["real_min"].sum(), 1e-9)),
                    "model_p": pk, "real_ft_pct": float(x["real_ftm"].sum() / max(x["real_fta"].sum(), 1)),
                    "sim_ft_pct": float(x["sim_ftm"].sum() / max(x["sim_fta"].sum(), 1e-9)),
                    # EXACT additive contribution to who_shoots_pp: sum over the group's slots of
                    # (sim weight - real weight) x (p - c); the groups of a partition sum to the total
                    "contrib_pp": float(100 * ((x["sim_fta"] / tot_s - x["real_fta"] / tot_r)
                                               * (x["p"] - (ps + pr) / 2)).sum())})
    return out


fr["has_prior_g"] = fr["has_prior_season"].round().astype(int)
fr["fta_asof_bin"] = pd.cut(fr["shooter_fta_asof"], [-1, 0.5, 10, 30, 1e9], labels=["0", "1-10", "11-30", "31+"])
fr["prior_x_asof"] = fr["has_prior_g"].astype(str) + "_" + fr["fta_asof_bin"].astype(str)
fr["u_ft_q"] = pd.qcut(fr["u_ft"].rank(method="first"), 5, labels=False)
fr["anon"] = fr["pid"] <= 0
fr["espn_missing"] = fr["espn"] <= 0
res["by_espn_missing"] = group_table("espn_missing")
res["by_has_prior"] = group_table("has_prior_g")
res["by_prior_x_asof"] = group_table("prior_x_asof")
res["by_usage_ft_quintile"] = group_table("u_ft_q")
res["by_anonymous_slot"] = group_table("anon")
# per player-game responsiveness: sim vs real FTA by served usage FT-trip rate quintile (slots that played in reality)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
