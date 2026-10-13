"""diag_clock_env10_artifact_hash_v1.py -- clock round 10 (experiments.md section 43): assert the E1 schedules
used by the closed loop are the round-9 fit unchanged. Local sha256 of every file under
data/processed/models/clock/r9_env_E1/{F1,F2} is compared with the LFS sha256 of the copy round 9 pushed to the HF
dataset (model_artifacts/clock/r9_env_E1/...), and every pickled refit's coefficients with its manifest row.
Read-only. Writes results/clock_env10/artifact_hash.json; exits non-zero on any mismatch.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import hf_sync_data as H  # noqa: E402

CK = ROOT / "data/processed/models/clock"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    from huggingface_hub import HfApi
    api = HfApi(token=H.resolve_token())
    out, bad = {"files": []}, 0
    for fold in ("F2", "F1"):
        d = CK / "r9_env_E1" / fold
        files = sorted(d.glob("*"))
        rp = [f"model_artifacts/clock/r9_env_E1/{fold}/{f.name}" for f in files]
        info = {i.path: i for i in api.get_paths_info(H.REPO_ID, rp, repo_type="dataset")}
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        coef = {Path(m["model_file"]).name: m["coef"] for m in man["months"]}
        for f, r in zip(files, rp):
            loc = sha(f)
            i = info.get(r)
            remote = (i.lfs.sha256 if getattr(i, "lfs", None) else None) if i is not None else None
            if remote is None and i is not None:
                remote = "git-blob:" + str(i.blob_id)
                loc_cmp = "git-blob:" + hashlib.sha1(b"blob %d\0" % f.stat().st_size + f.read_bytes()).hexdigest()
            else:
                loc_cmp = loc
            row = {"file": f"{fold}/{f.name}", "sha256_local": loc, "remote": remote, "match": loc_cmp == remote}
            if f.suffix == ".pkl":
                arm = pickle.load(open(f, "rb"))
                c = [round(float(x), 6) for x in arm.marginal.coef]
                row["coef_pickle"] = c
                row["coef_manifest"] = coef.get(f.name)
                row["coef_match"] = c == coef.get(f.name)
                bad += not row["coef_match"]
            bad += not row["match"]
            out["files"].append(row)
            print(row["file"], row["match"], row.get("coef_pickle"), flush=True)
    out["all_match"] = bad == 0
    p = ROOT / "results/clock_env10/artifact_hash.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("ALL MATCH" if bad == 0 else f"{bad} MISMATCHES")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
