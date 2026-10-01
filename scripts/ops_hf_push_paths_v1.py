"""ops_hf_push_paths_v1.py -- push NAMED local dirs under data/processed/models/ to the HF dataset's
`model_artifacts/` prefix, and nothing else (lane D, 2026-10-01).

`hf_sync_data.py push --dirs model_artifacts` uploads every gitignored file that differs, i.e. every lane's
in-progress artifacts too. This pushes only the dirs named, at the same repo paths `hf_sync_data.py pull --dirs
model_artifacts --only '<dir>/**'` reads back. The token is read the way hf_sync_data reads it (env, else .env).

    .venv/Scripts/python.exe scripts/ops_hf_push_paths_v1.py fold1_v1 engine_f1 engine_v3_f1
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hf_sync_data as H  # noqa: E402


def main() -> int:
    from huggingface_hub import HfApi
    names = sys.argv[1:]
    if not names:
        raise SystemExit("name at least one dir under data/processed/models/")
    tok = H.resolve_token()
    api = HfApi(token=tok)
    for n in names:
        local = ROOT / "data/processed/models" / n
        if not local.is_dir():
            raise SystemExit(f"{local} is not a directory")
        print(f"push {local} -> {H.REPO_ID}:model_artifacts/{n}", flush=True)
        api.upload_folder(repo_id=H.REPO_ID, repo_type="dataset", folder_path=str(local),
                          path_in_repo=f"model_artifacts/{n}", commit_message=f"lane D {n}",
                          ignore_patterns=["**/cuts/**", "**/__pycache__/**"])
        print(f"  done {n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
