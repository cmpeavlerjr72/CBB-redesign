#!/bin/bash
# clock round 10 full-size read, post-processing of ONE finished run: chunked diag_c4 extract of its 50-seed trajectory and
# diag_margin_dispersion_v2 (--seeds 50) on that extract. usage: chain_clk10_env_full_post_v1.sh <tag>...
cd /c/Users/devuser/CBB-clean-sheet
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for TAG in "$@"; do
  date
  .venv/Scripts/python.exe scripts/diag_c4_extract_chunked_v1.py --tag $TAG --traj-file results/trajectories/$TAG/trajectory.parquet || exit 2
  .venv/Scripts/python.exe scripts/diag_margin_dispersion_v2.py --prefix results/diag_c4/${TAG}_ --seeds 50 \
    --out results/clock_env10/full_dispersion/$TAG > results/clock_env10/logs/disp_$TAG.log 2>&1 || exit 3
  echo "$TAG post done"
done
date
echo POST_DONE
