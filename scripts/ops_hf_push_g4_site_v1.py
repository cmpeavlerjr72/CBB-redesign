"""ops_hf_push_g4_site_v1.py -- Lane G 2026-09-30: upload ONLY the gitignored fg_make
G4 site-offset artifacts to the private HF dataset under the existing
`model_artifacts` bulk key path (`model_artifacts/fg_make/round4_site/G4/`), so a
box `hf_sync_data.py pull --dirs model_artifacts --only 'fg_make/round4_site/**'`
lands them at data/processed/models/fg_make/round4_site/G4/. Then verifies the
remote file list.

    .venv/Scripts/python.exe scripts/ops_hf_push_g4_site_v1.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hf_sync_data as H  # noqa: E402

LOCAL = ROOT / "data" / "processed" / "models" / "fg_make" / "round4_site" / "G4"
REMOTE = "model_artifacts/fg_make/round4_site/G4"


def main() -> int:
    from huggingface_hub import HfApi
    tok = H.resolve_token()
    api = HfApi(token=tok)
    files = sorted(p for p in LOCAL.iterdir() if p.is_file())
    print(f"uploading {len(files)} files from {LOCAL} -> {H.REPO_ID}:{REMOTE}")
    api.upload_folder(repo_id=H.REPO_ID, repo_type="dataset", folder_path=str(LOCAL),
                      path_in_repo=REMOTE, commit_message="Lane G: fg_make round4_site/G4 artifacts")
    remote = [f for f in api.list_repo_files(H.REPO_ID, repo_type="dataset") if f.startswith(REMOTE + "/")]
    missing = {p.name for p in files} - {Path(f).name for f in remote}
    print(f"remote has {len(remote)} files under {REMOTE}; missing {sorted(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
