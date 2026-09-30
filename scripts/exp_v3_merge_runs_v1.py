"""Merge the two seed-split v3 runs into one grader-format directory results/engine_v0/v3_replay_s25."""
import json, glob
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
src = sorted(glob.glob(str(ROOT / "results/engine_v3_replay/sim/F2_2025_s*_o*")))
assert len(src) == 2, src
g = pd.concat([pd.read_parquet(Path(s) / "games.parquet") for s in src], ignore_index=True)
p = pd.concat([pd.read_parquet(Path(s) / "players.parquet") for s in src], ignore_index=True)
ref = json.loads((ROOT / "results/engine_v0/po4b_R_s25/run_meta.json").read_text(encoding="utf-8"))
out = ROOT / "results/engine_v0/v3_replay_s25"; out.mkdir(parents=True, exist_ok=True)
drop = ["tipoff_utc", "created_at"]
g.drop(columns=drop).to_parquet(out / "games.parquet", index=False)
p.drop(columns=drop).to_parquet(out / "players.parquet", index=False)
g.to_parquet(out / "games_stamped.parquet", index=False)        # rows carry created_at and tipoff_utc
assert (pd.to_datetime(g["created_at"], utc=True) < pd.to_datetime(g["tipoff_utc"], utc=True)).all()
m = dict(ref)
m.update(engine_tag="engine_v0/v3_replay_s25", seeds=sorted(int(s) for s in g["seed"].unique()),
         n_seeds=int(g["seed"].nunique()), n_rows=int(len(g)), input_dir="data/processed/models/engine_v3",
         inputs_version="v3-replay", game_ids=sorted(int(x) for x in g["game_id"].unique()),
         note="v3 replay inputs via run_engine_live.py; event block = live round-2 block",
         source_runs=[Path(s).name for s in src])
(out / "run_meta.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
print(len(g), g["game_id"].nunique(), g["seed"].nunique())
