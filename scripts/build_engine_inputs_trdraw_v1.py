"""build_engine_inputs_trdraw_v1.py -- Stage C draw files for ENGINE_TEAM_RATE_DRAW (team_rate_draw.py).

This is the pre-registration in docs/models/team_rate_estimator/experiments.md sections 7, 7a and 7c.
The file is new; no existing builder is edited.

Base inputs. By default the builder reads the v3 inputs that the live replay path writes
(`scripts/build_engine_inputs_v3_replay.py assemble`): `data/processed/models/engine_v3/{games,arrays,names}_F2_2025.*`
and `event_block_F2_2025.npz`. The replay module's own OUT path is imported when the module is present.
Any other inputs directory (e.g. the served v2, for tests) is a path argument.

Output: `<out>.npz` holds
  * team_static_k (K, G, 2, Ft) and team_block_k (K, G, 2, 16), both float32;
  * game_ids (G).
`<out>.json` holds the meta: sources, K, seed, the variance source, clip counts per rate-side, and the
`--artifact` paths.

For every game g, the side-s row is the offence view, with T = the side-s team and O = the opponent.
  * Offence-owned columns take T's `<rate>_off` value; defence-owned columns take O's `<rate>_def` value.
  * The mapping is `team_rate_adapter`'s, name for name, applied to the engine arrays:
    - possession_outcome: x100;
    - rebound `opp_def_dreb_c` = -oreb_def;
    - fg_make columns per shot class.
  * The same drawn team rate feeds every column that consumes it, so a team's offence rate and its
    opponent-view defence rate are each drawn once per set.

Draw on the estimator's own scale. For each set k, team and rate-side:

    c_k = c + sqrt(v) * e,   e ~ N(0, 1), independent across rates, sides and teams (sec. 7.2)

Here v is either the E3 `<rate>_<side>_v` column (S2) or `team_rate_variance_O1a_*.parquet`
`<rate>_<side>_v_o1a` (S3). The same e is used for S2 and S3 at the same `--seed`, so the arms are
paired. The rate L + c_k is clipped only to its support: [1e-4, 1 - 1e-4] for binomial rates and
[1e-4, inf) for the Poisson FT rate. Every clip is counted and reported.

Modes:
  * `--variance none --K 1`: S1. The single set is the point estimate, substituted.
  * `--no-substitute`: keep the base values. That gives an identity file (proof b) and, with
    `--variance zero --K 4`, a zero-variance multi-set file (proof d).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402

try:                                   # the live replay builder's own output location (not edited)
    import build_engine_inputs_v3_replay as V3R  # noqa: E402
    V3_DIR = Path(V3R.OUT)
except Exception:                      # noqa: BLE001  (module absent or not importable here)
    V3_DIR = ROOT / "data/processed/models/engine_v3"

FAMILY = {"tov": "binom", "ftr": "pois", "share_rim": "binom", "pa3": "binom", "make_rim": "binom",
          "make_jump": "binom", "make3": "binom", "oreb": "binom"}
#: engine column -> (rate, owner {"T": offence team's _off, "O": opponent's _def}, scale)
TEAM_MAP: dict[str, tuple[str, str, float]] = {
    "off_tov_c": ("tov", "T", 100.0), "opp_def_tov_c": ("tov", "O", 100.0),
    "off_ftr_c": ("ftr", "T", 100.0), "opp_def_ftr_c": ("ftr", "O", 100.0),
    "off_rim_c": ("share_rim", "T", 100.0), "opp_def_rim_c": ("share_rim", "O", 100.0),
    "off_3pa_c": ("pa3", "T", 100.0), "opp_def_3pa_c": ("pa3", "O", 100.0),
    "off_oreb_c": ("oreb", "T", 1.0), "opp_def_dreb_c": ("oreb", "O", -1.0),
    "off_make_c__rim": ("make_rim", "T", 1.0), "def_allow_c__rim": ("make_rim", "O", 1.0),
    "off_make_c__jump2": ("make_jump", "T", 1.0), "def_allow_c__jump2": ("make_jump", "O", 1.0),
    "off_make_c__three": ("make3", "T", 1.0), "def_allow_c__three": ("make3", "O", 1.0),
}
EPS = 1e-4


def load_base(inputs_dir: Path, tag: str, version: str | None, event_block: Path):
    inp = EngineInputs.load(inputs_dir, tag, version)
    eb = np.load(event_block)["team_block"]
    idx_path = ROOT / "data/processed/models/engine/event_round2_s1_F2_2025/index.json"
    ev_cols = json.loads(idx_path.read_text(encoding="utf-8"))["team_cols"]
    assert eb.shape == (inp.n_games, 2, len(ev_cols)), (eb.shape, inp.n_games)
    return inp, eb, {c: i for i, c in enumerate(ev_cols)}


def team_values(games: pd.DataFrame, table: pd.DataFrame, var: pd.DataFrame | None, var_suffix: str):
    """(c, v, L) arrays of shape (G, 2 teams [home, away], rate, side) for the mapped rates."""
    rates = list(FAMILY)
    t = table.set_index(["game_id", "team_id"])
    vv = var.set_index(["game_id", "team_id"]) if var is not None else None
    G = len(games)
    c = np.zeros((G, 2, len(rates), 2)); v = np.zeros_like(c); L = np.zeros_like(c)
    for ti, col in enumerate(("home_team_id", "away_team_id")):
        key = pd.MultiIndex.from_arrays([games["game_id"].to_numpy(), games[col].to_numpy()])
        rows = t.reindex(key)
        if rows.isna().any(axis=None):
            raise AssertionError(f"team-rate table lacks keys for {int(rows.iloc[:, 0].isna().sum())} {col} rows")
        vr = vv.reindex(key) if vv is not None else None
        for ri, r in enumerate(rates):
            for si, sd in enumerate(("off", "def")):
                c[:, ti, ri, si] = rows[f"{r}_{sd}_c"].to_numpy()
                L[:, ti, ri, si] = rows[f"{r}_{sd}_L"].to_numpy()
                if vr is not None:
                    v[:, ti, ri, si] = vr[f"{r}_{sd}{var_suffix}"].to_numpy()
                elif var_suffix == "_v":
                    v[:, ti, ri, si] = rows[f"{r}_{sd}_v"].to_numpy()
    return rates, c, v, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", type=Path, default=V3_DIR)
    ap.add_argument("--inputs-tag", default="F2_2025")
    ap.add_argument("--inputs-version", default=None, help="EngineInputs version (v3 dir: leave unset; served: v2)")
    ap.add_argument("--event-block", type=Path, default=None,
                    help="npz with team_block (G,2,16); default <inputs-dir>/event_block_<tag>.npz")
    ap.add_argument("--table", type=Path, default=ROOT / "data/processed/team_rate_features_E3_v4.parquet")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--variance", choices=["none", "zero", "e3", "o1a"], required=True)
    ap.add_argument("--variance-table", type=Path, default=ROOT / "data/processed/team_rate_variance_O1a_v3.parquet")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--no-substitute", action="store_true", help="keep base values (identity / zero-variance proofs)")
    ap.add_argument("--artifact", action="append", default=[], help="KEY=PATH, recorded in meta for the sim")
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    a = ap.parse_args()
    if a.variance == "none" and a.K != 1:
        raise SystemExit("--variance none is the point-estimate set; use --K 1")
    eb_path = a.event_block or (a.inputs_dir / f"event_block_{a.inputs_tag}.npz")
    inp, eb, ev_idx = load_base(a.inputs_dir, a.inputs_tag, a.inputs_version, eb_path)
    G = inp.n_games
    ts_k = np.repeat(inp.team_static[None].astype(np.float32), a.K, axis=0)
    tb_k = np.repeat(eb[None].astype(np.float32), a.K, axis=0)
    meta = {"K": a.K, "seed": a.seed, "variance": a.variance, "inputs_dir": str(a.inputs_dir), "inputs_tag": a.inputs_tag,
            "inputs_version_loaded": inp.meta.get("inputs_version_loaded"), "event_block": str(eb_path),
            "table": str(a.table), "fold": a.fold, "substituted": not a.no_substitute,
            "artifacts": dict(kv.split("=", 1) for kv in a.artifact), "clip": {}}
    if not a.no_substitute:
        tab = pd.read_parquet(a.table); tab = tab[tab["fold"] == a.fold]
        var = None; suffix = "_v"
        if a.variance == "o1a":
            var = pd.read_parquet(a.variance_table); var = var[var["fold"] == a.fold]; suffix = "_v_o1a"
            meta["variance_table"] = str(a.variance_table)
        rates, c, v, L = team_values(inp.games, tab, var, suffix)
        if a.variance in ("none", "zero"):
            v[:] = 0.0
        if a.K == 1 and a.variance == "none":
            draws = c[None]
        else:
            rng = np.random.default_rng(a.seed)
            e = rng.standard_normal((a.K,) + c.shape)
            draws = c[None] + np.sqrt(np.maximum(v, 0.0))[None] * e
        # support clip on the rate scale, counted
        for ri, r in enumerate(rates):
            p = L[None, :, :, ri, :] + draws[..., ri, :]
            lo, hi = EPS, (1 - EPS if FAMILY[r] == "binom" else np.inf)
            n_clip = int(((p < lo) | (p > hi)).sum())
            meta["clip"][r] = {"n": n_clip, "share": n_clip / p.size}
            draws[..., ri, :] = np.clip(p, lo, hi) - L[None, :, :, ri, :]
        rix = {r: i for i, r in enumerate(rates)}
        tn = inp.team_names
        for side in (0, 1):
            T, O = side, 1 - side
            for col, (r, owner, scale) in TEAM_MAP.items():
                src = draws[:, :, T, rix[r], 0] if owner == "T" else draws[:, :, O, rix[r], 1]
                val = (scale * src).astype(np.float32)
                if col in tn:
                    ts_k[:, :, side, tn[col]] = val
                if col in ev_idx:
                    tb_k[:, :, side, ev_idx[col]] = val
        meta["mapped_team_static"] = [c_ for c_ in TEAM_MAP if c_ in tn]
        meta["mapped_event_block"] = [c_ for c_ in TEAM_MAP if c_ in ev_idx]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(a.out) + ".npz", team_static_k=ts_k, team_block_k=tb_k,
             game_ids=inp.games["game_id"].to_numpy().astype(np.int64))
    Path(str(a.out) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote", str(a.out) + ".npz", ts_k.shape, tb_k.shape, json.dumps(meta["clip"]))


if __name__ == "__main__":
    main()
