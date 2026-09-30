"""Proofs for build_engine_inputs_v3_tag_v1.py (a: defaults bit-identical to engine_v3; b: table substitution only
changes team-rate-derived cells and equals the adapter's values read independently from the table)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from cbb_sim import team_rate_adapter as TRA
M = ROOT / "data/processed/models"; TAG = "F2_2025"
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base = M / "engine_v3"; out = {}
# ---- (a)
d = M / "engine_v3_DEFAULTS_PROOF"
out["a_files_sha_equal"] = {f: sha(base / f) == sha(d / f) for f in (f"arrays_{TAG}.npz", f"games_{TAG}.parquet", f"names_{TAG}.json", f"event_block_{TAG}.npz")}
za, zb = np.load(base / f"arrays_{TAG}.npz"), np.load(d / f"arrays_{TAG}.npz")
out["a_arrays_equal"] = all(np.array_equal(za[k], zb[k]) for k in za.files)
ev = M / "engine_v3_DEFAULTS_PROOF/overlay/data/processed/models/engine/event_round2_s1_F2_2025"
served = M / "engine/event_round2_s1_F2_2025"
out["a_overlay_event_block_equals_base"] = bool(np.array_equal(np.load(ev / "team_block.npz")["team_block"], np.load(base / f"event_block_{TAG}.npz")["team_block"]))
out["a_overlay_model_files_identical_to_served"] = all(sha(ev / f.name) == sha(f) for f in served.glob("*.joblib"))
out["a_overlay_index_equals_served"] = json.loads((ev / "index.json").read_text()) == json.loads((served / "index.json").read_text())
# ---- (b)
e = M / "engine_v3_E3_served_proof"; tab = ROOT / "data/processed/team_rate_features_E3_v4.parquet"
names = json.loads((base / f"names_{TAG}.json").read_text()); tn, sn = names["team_names"], names["slot_names"]
g = pd.read_parquet(base / f"games_{TAG}.parquet"); n = len(g)
z0 = dict(np.load(base / f"arrays_{TAG}.npz")); z1 = dict(np.load(e / f"arrays_{TAG}.npz"))
e0 = np.load(base / f"event_block_{TAG}.npz")["team_block"]; e1 = np.load(e / f"event_block_{TAG}.npz")["team_block"]
t = TRA.load_table(tab, "F2").set_index(["season", "game_id", "team_id"])
def tv(col, team, scale=1.0):
    idx = pd.MultiIndex.from_arrays([np.repeat(2025, n), g["game_id"].to_numpy(), team])
    return t[col].reindex(idx).to_numpy(dtype="float64") * scale
H, A = g["home_team_id"].to_numpy(), g["away_team_id"].to_numpy()
def expect(col, who, scale, sign=1.0):
    # side 0 = home offence, side 1 = away offence; 'def' rows belong to the opponent
    own = [tv(col, H, scale), tv(col, A, scale)]
    if who == "off": return np.stack(own, axis=1) * sign
    return np.stack([tv(col, A, scale), tv(col, H, scale)], axis=1) * sign
exp = {}
for c, (tc, who, sc) in TRA.PO_MAP.items(): exp[c] = expect(tc, who, sc)
for c, (tc, who, sg) in TRA.RB_MAP.items(): exp[c] = expect(tc, who, 1.0, sg)
for cls, r in TRA.FG_CLASS_RATE.items():
    key = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}[cls]
    exp[f"off_make_c__{key}"] = expect(f"{r}_off_c", "off", 1.0); exp[f"def_allow_c__{key}"] = expect(f"{r}_def_c", "def", 1.0)
worst = {c: float(np.abs(z1["team_static"][:, :, tn[c]].astype("f8") - v).max()) for c, v in exp.items()}
out["b_team_static_max_abs_vs_table_float32_tolerance"] = max(worst.values())
out["b_event_block_PO_cols_max_abs_vs_table"] = max(float(np.abs(e1[:, :, tn[c]].astype("f8") - exp[c]).max()) for c in TRA.PO_MAP)
changed_ts = [c for c, j in tn.items() if not np.array_equal(z0["team_static"][:, :, j], z1["team_static"][:, :, j])]
out["b_changed_team_static"] = changed_ts
out["b_team_static_changed_exactly_the_16_mapped"] = sorted(changed_ts) == sorted(exp)
out["b_team_static_unmapped_columns_bit_equal"] = all(np.array_equal(z0["team_static"][:, :, j], z1["team_static"][:, :, j]) for c, j in tn.items() if c not in exp)
out["b_event_block_changed_cols"] = [i for i in range(16) if not np.array_equal(e0[:, :, i], e1[:, :, i])]
out["b_event_block_other_cols_bit_equal"] = all(np.array_equal(e0[:, :, i], e1[:, :, i]) for i in range(8, 16))
chs = [c for c, j in sn.items() if not np.array_equal(z0["slot_static"][..., j], z1["slot_static"][..., j])]
out["b_changed_slot_columns"] = chs
out["b_other_arrays_bit_equal"] = all(np.array_equal(z0[k], z1[k]) for k in z0 if k not in ("team_static", "slot_static"))
# validate the shrunk-dev identity on the SERVED v3 values (league rate cancels)
m = json.loads((M / "fg_make/round4/m_fitted.json").read_text()); res = {}
for cls, key in (("FGA_rim", "rim"), ("FGA_jump2", "jump2"), ("FGA_3", "three")):
    att = z0["slot_static"][..., sn[f"shooter_att_c__{key}"]].astype("f8"); smc = z0["slot_static"][..., sn[f"shooter_make_c__{key}"]].astype("f8")
    c = z0["team_static"][:, :, tn[f"off_make_c__{key}"]].astype("f8")[:, :, None]; mm = float(m[cls]["m"])
    dev = np.where(att > 0, att * (smc - c) / (mm + att), 0.0); real = z0["roster_cbbd"] > 0
    res[key] = float(np.abs(dev - z0["slot_static"][..., sn[f"shooter_shrunk_dev_c__{key}"]].astype("f8"))[real].max())
out["b_shrunk_dev_identity_max_abs_on_served_v3_real_slots"] = res
print(json.dumps(out, indent=1)); (ROOT / "results").mkdir(exist_ok=True); (ROOT / "results/v3_tag_proofs_ab.json").write_text(json.dumps(out, indent=1))
