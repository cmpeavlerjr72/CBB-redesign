#!/usr/bin/env python
"""
diag_shot_block_game_bootstrap_v1.py -- PM follow-up on the drawn block flag:
paired bootstrap over GAMES (2,000 draws; each game keeps its 25 paired seeds
in both arms) of the arm-minus-served difference for G5 total SD ratio, G4
TOV% pooled, G4 OREB% pooled and G9 total bias, with the gate's own formulas
(`cbb_sim.eval.gates`: G5 ratio = mean_g(sim SD_g) / SD_g(actual_g - sim mean_g);
TOV% = TOV / (FGA - OREB + TOV + 0.44 FTA), pooled). Also the two components
of the G5 ratio and TOV per team-game as a count.

    .venv/Scripts/python.exe scripts/diag_shot_block_game_bootstrap_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402

R = ROOT / "results/engine_v0"
REF, ARMS, FLOOR = "po4b_R_s25", ["sb_K2O_s25", "sb_K2_s25"], "po4b_R_s25_floor"
N_BOOT, SEED = 2000, 20260930


def per_game(tag: str) -> pd.DataFrame:
    eng = C.load_engine_results(str(R / tag))
    summ, _ = G.build_grading_frame(eng.games, 2025)
    g = eng.games.copy()
    for s in ("home", "away"):
        g[f"{s}_fga"] = g[f"{s}_fga3"] + g[f"{s}_fga2_rim"] + g[f"{s}_fga2_jump"]
    agg = pd.DataFrame({
        "game_id": g["game_id"],
        "tov": g["home_tov"] + g["away_tov"],
        "pest": (g["home_fga"] + g["away_fga"] - g["home_oreb"] - g["away_oreb"]
                 + g["home_tov"] + g["away_tov"] + 0.44 * (g["home_fta"] + g["away_fta"])),
        "oreb": g["home_oreb"] + g["away_oreb"],
        "orebd": g["home_oreb"] + g["away_oreb"] + g["home_dreb"] + g["away_dreb"],
        "poss": g["possessions"]}).groupby("game_id").sum()
    s = summ.set_index("game_id")[["total", "sim_total_mean", "sim_total_sd"]]
    return s.join(agg, how="inner").sort_index()


def stats(d: pd.DataFrame, idx: np.ndarray) -> dict:
    x = d.iloc[idx]
    res = x["total"] - x["sim_total_mean"]
    return {"g5_total_ratio": x["sim_total_sd"].mean() / res.std(),
            "tov_pct": x["tov"].sum() / x["pest"].sum(),
            "oreb_pct": x["oreb"].sum() / x["orebd"].sum(),
            "total_bias": -res.mean(),
            "g5_within_sd": x["sim_total_sd"].mean(), "g5_resid_sd": res.std()}


def main() -> int:
    ref = per_game(REF)
    box = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2025.parquet",
                          columns=["game_id", "turnovers", "total_turnovers"])
    act_tov = box[box["game_id"].isin(ref.index)]
    out = {"n_games": int(len(ref)), "n_boot": N_BOOT, "arms": {}}
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, len(ref), size=(N_BOOT, len(ref)))
    full = np.arange(len(ref))
    fl = per_game(FLOOR).reindex(ref.index)
    s_ref, s_fl = stats(ref, full), stats(fl, full)
    out["served"] = s_ref
    out["seed_offset_floor"] = {k: abs(s_fl[k] - s_ref[k]) for k in s_ref}
    for arm in ARMS:
        a = per_game(arm).reindex(ref.index)
        pt = {k: stats(a, full)[k] - s_ref[k] for k in s_ref}
        bs = {k: [] for k in s_ref}
        for b in boots:
            sa, sr = stats(a, b), stats(ref, b)
            for k in bs:
                bs[k].append(sa[k] - sr[k])
        out["arms"][arm] = {k: {"diff": float(pt[k]), "ci95": [float(np.quantile(v, 0.025)),
                                                               float(np.quantile(v, 0.975))],
                                "excludes_zero": bool(np.quantile(v, 0.025) > 0 or np.quantile(v, 0.975) < 0)}
                            for k, v in bs.items()}
        out["arms"][arm]["value"] = stats(a, full)
        # TOV per team-game (count)
        out["arms"][arm]["tov_per_team_game"] = float(a["tov"].sum() / (2 * len(a)) / 25)
    out["served"]["tov_per_team_game"] = float(ref["tov"].sum() / (2 * len(ref)) / 25)
    out["actual_tov_per_team_game"] = float(act_tov["total_turnovers"].mean())
    out["actual_tov_per_team_game_box_turnovers_col"] = float(act_tov["turnovers"].mean())
    out["poss_per_game"] = {"served": float(ref["poss"].sum() / len(ref) / 25),
                            **{a: float(per_game(a)["poss"].sum() / len(ref) / 25) for a in ARMS}}
    p = ROOT / "results/shot_block_round2/game_bootstrap_v1.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
