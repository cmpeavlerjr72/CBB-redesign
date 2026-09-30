"""diag_team_rate_draw_proofs_v1.py -- proofs (b)-(e) for the Stage C draw
(experiments.md section 7c). Proof (a) and the digest halves of (b) and (d) are `digest_engine_run.py --compare`
runs, which are logged in results/team_rate_estimator/trd_*.log.

  (b) K = 1: `k_from_book` makes no RNG call (the team_rate counter stays 0). The runs are bit-identical (digest).
  (c) For the S2 (E3 v) and S3 (O1a v) K = 64 files, per mapped rate-side over the 5,710 F2 games:
      * the mean over K minus the point estimate, standardised by sqrt(v/K), should be ~N(0, 1);
      * SD over K / sqrt(v) should be ~1.
  (d) The draw index consumes only the team_rate family. A K = 4 zero-variance file reproduces the digest
      (logged). Here: that k actually varies across seeds, and every other family's key and counter are
      untouched.
  (e) A small end-to-end run (40 games x 8 seeds) for S0 / S1 / S2 / S3. It reports that each runs, and the
      between-seed spread of a game's drawn inputs. It gives NO gate reading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.engine import team_rate_draw as TRD  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.engine.rng import FAMILIES, StreamBook  # noqa: E402
import build_engine_inputs_trdraw_v1 as BLD  # noqa: E402

D = ROOT / "results/trdraw"
OUT = ROOT / "results/team_rate_estimator"


def main():
    res = {}
    inp = EngineInputs.load(ROOT / "data/processed/models/engine", "F2_2025", "v2")
    games = inp.games
    # (b) no RNG call at K = 1
    d1 = TRD.load(D / "v2base_S1_K1.npz")
    book = StreamBook(np.arange(8), np.repeat(games["game_id"].to_numpy()[:1], 8))
    k = d1.k_from_book(book)
    res["b_K1_team_rate_counter_after"] = int(book.counter["team_rate"].max())
    res["b_K1_k_all_zero"] = bool((k == 0).all())
    assert res["b_K1_team_rate_counter_after"] == 0 and res["b_K1_k_all_zero"]
    # (c) moments of the perturbed sets
    tab = pd.read_parquet(ROOT / "data/processed/team_rate_features_E3_v4.parquet"); tab = tab[tab["fold"] == "F2"]
    o1a = pd.read_parquet(ROOT / "data/processed/team_rate_variance_O1a_v3.parquet"); o1a = o1a[o1a["fold"] == "F2"]
    point = d1.team_static_k[0]
    tn = inp.team_names
    for name, fn, var, suf in (("S2_e3", "v2base_S2_K64.npz", None, "_v"), ("S3_o1a", "v2base_S3_K64.npz", o1a, "_v_o1a")):
        dk = TRD.load(D / fn)
        rates, c, v, L = BLD.team_values(games, tab, var, suf)
        rix = {r: i for i, r in enumerate(rates)}
        rows = []
        for col, (r, owner, scale) in BLD.TEAM_MAP.items():
            if col not in tn:
                continue
            for side in (0, 1):
                ti = side if owner == "T" else 1 - side
                si = 0 if owner == "T" else 1
                sd_exp = abs(scale) * np.sqrt(v[:, ti, rix[r], si])
                x = dk.team_static_k[:, :, side, tn[col]].astype(np.float64)
                mean_err = (x.mean(axis=0) - point[:, side, tn[col]]) / (sd_exp / np.sqrt(dk.K))
                sd_ratio = x.std(axis=0, ddof=1) / sd_exp
                rows.append({"col": col, "side": side, "z_mean": float(mean_err.mean()), "z_sd": float(mean_err.std()),
                             "sd_ratio_mean": float(sd_ratio.mean()), "sd_ratio_sd": float(sd_ratio.std())})
        R = pd.DataFrame(rows)
        res[f"c_{name}"] = {"z_mean_range": [float(R.z_mean.min()), float(R.z_mean.max())],
                            "z_sd_range": [float(R.z_sd.min()), float(R.z_sd.max())],
                            "sd_ratio_mean_range": [float(R.sd_ratio_mean.min()), float(R.sd_ratio_mean.max())],
                            "clip": json.loads((D / fn.replace(".npz", ".json")).read_text())["clip"]}
        R.to_csv(OUT / f"trd_proof_c_{name}.csv", index=False)
        print(name); print(R.round(4).to_string())
    # (d) k varies; other families untouched by the team_rate draw
    d4 = TRD.load(D / "v2base_zero_K4.npz")
    gids = np.repeat(games["game_id"].to_numpy()[:60], 5); seeds = np.tile(np.arange(5), 60)
    b1 = StreamBook(seeds, gids); b2 = StreamBook(seeds, gids)
    k4 = d4.k_from_book(b2)
    res["d_K4_k_counts"] = np.bincount(k4, minlength=4).tolist()
    res["d_other_families_identical"] = all(np.array_equal(b1.keys[f], b2.keys[f]) and np.array_equal(b1.counter[f], b2.counter[f])
                                            for f in FAMILIES if f != "team_rate")
    res["d_k_index_equals_k_from_book"] = bool(np.array_equal(d4.k_index(seeds, gids), k4))
    # (e) end-to-end: runs exist; between-seed spread of a game's drawn inputs; margin SD across seeds (descriptive)
    e = {}
    for arm in ("off", "S1_K1", "S2_K64", "S3_K64"):
        g = pd.read_parquet(D / f"TRD_e2e_{arm}" / "games.parquet")
        g["margin"] = g["home_pts"].astype(int) - g["away_pts"].astype(int)
        e[arm] = {"rows": len(g), "games": int(g.game_id.nunique()), "seeds": int(g.seed.nunique()),
                  "mean_margin": float(g.margin.mean()), "median_per_game_margin_sd": float(g.groupby("game_id").margin.std().median())}
    for arm, fn in (("S2_K64", "v2base_S2_K64.npz"), ("S3_K64", "v2base_S3_K64.npz")):
        dk = TRD.load(D / fn)
        g40 = games["game_id"].to_numpy()[:40]; gi = np.arange(40)
        sd = []
        for col in ("off_tov_c", "off_make_c__rim", "off_oreb_c", "off_3pa_c"):
            vals = np.stack([dk.team_static_k[dk.k_index(np.full(40, s), g40), gi, 0, tn[col]] for s in range(8)])
            sd.append({"col": col, "median_between_seed_sd": float(np.median(vals.std(axis=0, ddof=1))),
                       "median_draw_sd_over_K": float(np.median(dk.team_static_k[:, gi, 0, tn[col]].std(axis=0, ddof=1)))})
        e[arm]["between_seed_input_spread"] = sd
    s0 = pd.read_parquet(D / "TRD_e2e_off" / "games.parquet").sort_values(["game_id", "seed"]).reset_index(drop=True)
    s1 = pd.read_parquet(D / "TRD_e2e_S1_K1" / "games.parquet").sort_values(["game_id", "seed"]).reset_index(drop=True)
    e["S1_vs_S0_rows_differing"] = int((s0[["home_pts", "away_pts"]].to_numpy() != s1[["home_pts", "away_pts"]].to_numpy()).any(axis=1).sum())
    res["e"] = e
    json.dump(res, open(OUT / "trd_proofs_v1.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
