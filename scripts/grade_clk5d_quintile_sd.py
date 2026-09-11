"""grade_clk5d_quintile_sd.py -- line L12 of clock round 5d (experiments.md s26).

The RESPONSIVENESS line, read CLOSED-LOOP rather than offline: the per-game
possession SD ratio -- the engine's within-game (across-seed) SD over
`SD(actual - sim per-game mean)` -- cut by PREGAME-TEMPO quintile.

Both SDs are `grade_clk5_dispersion_loop.score`'s, unchanged: the numerator is
the MEAN over games of the per-game across-seed SD, the denominator is the SD of
the per-game residual `actual - sim mean`. That module is imported, and its
`REF` actuals loader and `UNIVERSE_V2` are reused, so this script cannot drift
from the grader the round-5b table was read on. Nothing here adjusts anything.

The tempo feature is the one `grade_clk3c_closed_loop.tempo_prior` uses --
`tempo_prior_game` from the engine's own input arrays, averaged over the game's
two team rows (they are equal to 0.0, verified) -- so the quintile cut here and
the quintile cut in the round-3c/4 responsiveness table are the same cut.

    .venv/Scripts/python.exe scripts/grade_clk5d_quintile_sd.py \
        --dirs clk5b_B1_s25 clk5d_C4_s25 clock4_R_s25 clock4_R_s25_floor
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, fn: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


g5 = _load("grade_clk5_dispersion_loop", "grade_clk5_dispersion_loop.py")
REF = g5.REF
RESULTS = g5.RESULTS
INPUT_DIR = Path("data/processed/models/engine")


def tempo_prior(season: int, fold: str = "F2", suffix: str = "_v2") -> pd.DataFrame:
    """`grade_clk3c_closed_loop.tempo_prior`, with the inputs-version suffix the
    run actually used. The two team rows of a game carry the same value, so the
    mean is a decode, not an aggregation."""
    g = pd.read_parquet(INPUT_DIR / f"games_{fold}_{season}{suffix}.parquet")
    z = np.load(INPUT_DIR / f"arrays_{fold}_{season}{suffix}.npz")
    names = json.loads(
        (INPUT_DIR / f"names_{fold}_{season}{suffix}.json").read_text(encoding="utf-8"))
    j = names["team_names"]["tempo_prior_game"]
    v = z["team_static"][:, :, j]
    assert float(np.nanmax(np.abs(v[:, 0] - v[:, 1]))) == 0.0, \
        "tempo_prior_game is not game-level in the engine inputs"
    return pd.DataFrame({"game_id": g["game_id"].to_numpy(),
                         "tempo_prior": v.mean(axis=1)})


def per_game(d: Path, act: pd.DataFrame) -> pd.DataFrame:
    """The per-game table of `grade_clk5_dispersion_loop.score`, verbatim."""
    raw = pd.read_parquet(d / "games.parquet")
    raw["poss"] = raw["possessions"].astype(float)
    g = raw.groupby("game_id")
    per = pd.DataFrame({
        "n_seeds": g.size(),
        "poss_mean": g["poss"].mean(), "poss_sd": g["poss"].std(ddof=1),
    }).reset_index().merge(act, on="game_id", how="inner")
    return per


def ratios(per: pd.DataFrame, tempo: pd.DataFrame, nq: int = 5) -> dict:
    p = per.merge(tempo, on="game_id", how="left").dropna(
        subset=["tempo_prior", "act_poss"]).copy()
    p["q"] = pd.qcut(p["tempo_prior"], nq,
                     labels=[f"Q{i}" for i in range(1, nq + 1)]).astype(str)
    out = {}
    for q, sub in [("ALL", p)] + sorted(p.groupby("q"), key=lambda kv: kv[0]):
        within = float(sub["poss_sd"].mean())
        resid = float(np.std(sub["act_poss"].to_numpy()
                             - sub["poss_mean"].to_numpy(), ddof=1))
        out[q] = {"n": int(len(sub)), "within_sd": within, "resid_sd": resid,
                  "ratio": (within / resid if resid else float("nan")),
                  "tempo_mean": float(sub["tempo_prior"].mean()),
                  "underpowered": bool(len(sub) < 50)}
    qs = [out[f"Q{i}"]["ratio"] for i in range(1, nq + 1)]
    out["_spread"] = float(max(qs) - min(qs))
    out["_worst_dev"] = float(max(abs(r - 1.0) for r in qs))
    out["_outside_band"] = [f"Q{i + 1}" for i, r in enumerate(qs)
                            if not (0.85 <= r <= 1.15)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--out",
                    default="data/processed/models/clock/v5d_quintile_sd.json")
    a = ap.parse_args()

    dirs = [Path(a.results_dir) / d for d in a.dirs]
    for d in dirs:
        if not (d / "games.parquet").exists():
            raise SystemExit(f"missing {d}/games.parquet")
    ids = pd.read_parquet(dirs[0] / "games.parquet")["game_id"].unique()
    act = REF.load_actual_games(a.season)[["game_id"]]
    poss = REF.load_actual_possessions(a.season).rename(
        columns={"game_poss": "act_poss"})
    act = act.merge(poss, on="game_id", how="left")
    act = act[act["game_id"].isin(ids)].reset_index(drop=True)
    tempo = tempo_prior(a.season, suffix=a.suffix)

    rows = {}
    for d in dirs:
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        r = ratios(per_game(d, act), tempo)
        r["_arm"] = meta.get("arm")
        r["_n_seeds"] = int(meta.get("n_seeds", 0))
        rows[d.name] = r
    Path(a.out).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    tbl = pd.DataFrame({
        t: {q: round(r[q]["ratio"], 4) for q in
            ("Q1", "Q2", "Q3", "Q4", "Q5", "ALL")} | {
            "spread": round(r["_spread"], 4),
            "worst_dev": round(r["_worst_dev"], 4),
            "outside": ",".join(r["_outside_band"]) or "-",
            "n_seeds": r["_n_seeds"]}
        for t, r in rows.items()}).T
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\nL12 -- possession SD ratio by PREGAME-TEMPO quintile "
              "(produced within-game SD / SD(actual - sim mean))")
        print(tbl.to_string())
        print("\nper-quintile n (games):",
              {q: rows[dirs[0].name][q]["n"] for q in
               ("Q1", "Q2", "Q3", "Q4", "Q5")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
