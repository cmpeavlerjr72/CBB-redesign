"""ops_overlay_check_v1.py -- run INSIDE the sim container (cwd /app) before a sweep on a tagged v3 input dir.
    python scripts/ops_overlay_check_v1.py --input-dir data/processed/models/engine_v3_<tag> [--root /app]
Fails (exit 1) unless the files the ENGINE will read equal this tag's own: (1) the event adapter's team_block.npz under
<root>/data/processed/models/engine/event_round2_s1_F2_2025/ equals <input-dir>/event_block_F2_2025.npz (otherwise the
sim silently uses the SERVED v2 round-2 block on v3 team_static); (2) when builder_report.json lists fg / rebound artifact
overlays, the manifests under <root> equal the overlay's. Lane G.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--input-dir", required=True); ap.add_argument("--root", default=".")
a = ap.parse_args(); root = Path(a.root); d = Path(a.input_dir); M = "data/processed/models"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ok = True
eng = np.load(root / M / "engine/event_round2_s1_F2_2025/team_block.npz")["team_block"]
want = np.load(d / "event_block_F2_2025.npz")["team_block"]
same = eng.shape == want.shape and bool(np.array_equal(eng, want)); ok &= same
print(f"event team_block equals the input dir's event block: {same}")
ov = d / "overlay" / M
for rel in ("engine/event_round2_s1_F2_2025/index.json", "fg_make/round4/B1/manifest_FGA_rim.json",
            "rebound/s1_confirm/S1_weekly/F2/manifest.json"):
    if (ov / rel).exists():
        s = (root / M / rel).exists() and sha(root / M / rel) == sha(ov / rel); ok &= s
        print(f"{rel}: mounted == overlay: {s}")
print("OVERLAY CHECK:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
