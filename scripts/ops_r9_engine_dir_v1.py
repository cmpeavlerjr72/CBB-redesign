"""ops_r9_engine_dir_v1.py -- foul round 9 (lane C): build the scratch ENGINE_DIR a LOCAL run needs to
reproduce a box v3 tag (the box bind-mounts `engine_v3_<tag>/overlay/.../event_round2_s1_F2_2025` over the
served path; locally the served path is shared by other lanes and is never touched).

    ops_r9_engine_dir_v1.py TAG OUT_DIR

OUT_DIR gets: event_round2_s1_F2_2025/ (copied from the tag's overlay) and copies of every other file
directly under data/processed/models/engine/ that the adapters read (*.joblib). Nothing served is edited.
"""
import shutil
import sys
from pathlib import Path

tag, out = sys.argv[1], Path(sys.argv[2])
src_ev = Path(f"data/processed/models/engine_v3_{tag}/overlay/data/processed/models/engine/event_round2_s1_F2_2025")
assert src_ev.is_dir(), src_ev
out.mkdir(parents=True, exist_ok=False)
shutil.copytree(src_ev, out / "event_round2_s1_F2_2025")
for f in Path("data/processed/models/engine").glob("*.joblib"):
    shutil.copy2(f, out / f.name)
print("built", out, sorted(p.name for p in out.iterdir()))
