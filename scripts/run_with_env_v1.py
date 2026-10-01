"""run_with_env_v1.py -- run a repo script with ENGINE_* (or any) environment variables set INSIDE the process
(lane D, 2026-10-01). `scripts/box_run.sh` forwards only a fixed list of ENGINE_* names into the container (not
ENGINE_SHOT_BLOCK / ENGINE_FOUL_JOINT / ENGINE_SHARED_SHOOTING / ENGINE_CHANCE_TIME / ENGINE_EVENT_TEAM_BLOCK), so a
box read that needs one of those set cannot pass it from the host. This sets each `--env K=V` in os.environ, then runs
the named script as a child process with the remaining arguments and that environment.

    scripts/box_run.sh scripts/run_with_env_v1.py --env ENGINE_FOUL_JOINT=reference --env ENGINE_CLOCK=v5b_glat_pmean \
        -- scripts/run_engine_overlay_v1.py --overrides ... --runner full -- ...
"""
import os
import subprocess
import sys


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: run_with_env_v1.py --env K=V [--env K=V ...] -- <script.py> [args]")
    k = argv.index("--")
    own, rest = argv[:k], argv[k + 1:]
    it = iter(own)
    for a in it:
        if a != "--env":
            raise SystemExit(f"unexpected argument {a!r}")
        kv = next(it)
        key, _, val = kv.partition("=")
        os.environ[key] = val
        print(f"[run_with_env_v1] {key}={val}", flush=True)
    # a child process (not runpy): multiprocessing workers then re-import the real script under any start method
    raise SystemExit(subprocess.call([sys.executable, *rest], env=dict(os.environ)))


if __name__ == "__main__":
    main()
