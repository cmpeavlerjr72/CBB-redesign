"""One-minute LightGBM thread probe for the box (lane J 2026-09-30). Run INSIDE the image:

    scripts/box_run.sh scripts/ops_lgbm_thread_probe_v1.py
    docker run --rm -e OMP_THREAD_LIMIT=192 -e OMP_NUM_THREADS=16 -v "$PWD:/app" -w /app \
        --entrypoint python cbb-sweep scripts/ops_lgbm_thread_probe_v1.py     # the hypothesis test

Times a 300k x 50, 6-class fit at n_jobs 1, 4, 16. 09-18 saw n_jobs=1 == n_jobs=150 in the image;
the suspect is the baked OMP_THREAD_LIMIT=1 (a hard OpenMP ceiling), not the wheel. This is
informational only: the par_v1 trainers do not depend on the answer (n_jobs=1 per fit, processes
across refit dates). Prints env, timings, and a verdict line.
"""
import os
import time

import numpy as np

import lightgbm as lgb

print({k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "LIGHTGBM_NUM_THREADS")},
      "cpus", os.cpu_count(), "lightgbm", lgb.__version__)
rng = np.random.RandomState(0)
X = rng.rand(300_000, 50).astype("float32")
y = rng.randint(0, 6, 300_000)
t = {}
for nj in (1, 4, 16):
    s = time.time()
    lgb.LGBMClassifier(objective="multiclass", n_estimators=30, num_leaves=63, verbose=-1,
                       n_jobs=nj, random_state=0).fit(X, y)
    t[nj] = round(time.time() - s, 1)
    print(f"n_jobs={nj}: {t[nj]} s", flush=True)
print("VERDICT:", "threads scale" if t[16] < 0.7 * t[1] else "no useful scaling (capped or overhead-bound)")
