"""diag_aggregation_grade_prep_v1.py -- lane A 2026-09-30: give an aggregation runner output a contract run_meta
(copied keys from the runner's own meta plus the contract keys) so scripts/eval_gates.py can grade it.
games.parquet is hard-linked (no copy, nothing overwritten)."""
import json, os, sys
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
m = json.loads((src / "run_meta.json").read_text(encoding="utf-8"))
meta = {"engine_tag": f"aggregation_v1/{src.name}", "created_at": m.get("created_at"), "seeds": m["seeds"],
        "n_seeds": len(m["seeds"]), "fold": "F2", "backtest": True, "sealed_touched": False, "partial": False,
        "season": 2025, "n_games": m["n_games"], "ast_is_placeholder": True, "source_meta": m}
(dst / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
if not (dst / "games.parquet").exists():
    os.link(src / "games.parquet", dst / "games.parquet")
print("prepared", dst)
