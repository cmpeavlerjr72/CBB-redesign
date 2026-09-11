# AWS launch chain -- CBB engine seed sweeps (PREPARED, NOT LAUNCHED)

Status 2026-09-10: every file below is written and the parity gate has been
proven on the home box. **No instance has been launched and no cost has been
incurred.** This doc, `Dockerfile.cbb`, `requirements-cloud.txt`,
`scripts/run_aws_sweep.sh` and `scripts/digest_engine_run.py` are the launch
chain; nothing here executes anything on AWS.

Pattern reused from `C:\Users\devuser\cfb-props-sim`
(`docs/ops/cloud_sweep_runbook.md`, `Dockerfile`,
`scripts/cloud/{entrypoint.sh,parity_digest.py,push_results.py}`): same base
image, same thread-pin policy, same "parity gate is a hard stop before any
sweep" rule, same exit-trap evidence discipline. One structural difference is
called out in section 5.

Why this run needs it at all: `docs/tests/engine_seed_count_2026-09-10.md`
fixes 200 seeds as the minimum for a gate report and ~2,000 before any ROI
number is read. At the measured ~817-880 poss/s/core, a 200-seed x 5,710-game
run is ~2x10^5 core-seconds (~2.8 h on this 20-core box, matching the figure
already in `docs/models/engine/RESUME.md` section 3); 2,000 seeds is ~2x10^6
core-seconds, ~a day locally. Section 6 has the cloud numbers.

---

## 1. What the engine actually reads (traced, not assumed)

Traced by reading `scripts/run_engine.py`'s imports and
`src/cbb_sim/engine/{inputs.py,adapters.py}`'s file-open calls for the
`ENGINE_EVENT=round2_s1` (adopted) config -- the same discipline as cfb's
`parity_digest.py --trace-opens`, done here by code reading since this repo
has no audit-hook harness yet.

| File | Size | Git-tracked? |
|---|---:|---|
| `data/processed/models/engine/games_F2_2025.parquet` | 72 KB | yes |
| `data/processed/models/engine/arrays_F2_2025.npz` | 9.2 MB | yes |
| `data/processed/models/engine/names_F2_2025.json` | 4 KB | yes |
| `data/processed/models/engine/rebound_F2.joblib` | 8.1 MB | yes |
| `data/processed/models/engine/free_throw_F2.joblib` | 2.7 MB | yes |
| `data/processed/models/engine/fg_make_FGA_3_decision8_F2.joblib` | 1.4 MB | yes |
| `data/processed/models/engine/event_round2_s1_F2_2025/` (14 files: 6 monthly `first_*.joblib`, 6 `cont_*.joblib`, `index.json`, `team_block.npz`) | **45 MB** | **NO -- gitignored** |
| `data/processed/models/free_throw/bonus_era.json` | 4 KB | yes |
| `data/processed/models/fg_make/winner_FGA_rim.joblib` | 1.4 MB | yes |
| `data/processed/models/fg_make/winner_FGA_jump2.joblib` | 1.4 MB | yes |
| `data/processed/models/fg_make/winner_FGA_3.joblib` | 1.4 MB | yes (loaded even under `ENGINE_FG3=decision8` -- see note) |
| `data/processed/models/clock/reference_not_adopted_lgbm_quantile.pkl` | 16 MB | yes (default `ENGINE_CLOCK=reference`) |
| `data/processed/models/rotation/rotation_fit.json` | 20 KB | yes |

Fallback-mode-only (not needed for the recommended `ENGINE_EVENT=round2_s1`
config, but present if someone runs `ENGINE_EVENT=reference`):
`data/processed/models/possession_outcome/reference_not_adopted_{first,cont}.pkl`
(17 MB + 4 KB, both tracked). `reference_empirical` clock mode needs
`data/processed/models/clock/reference_not_adopted_empirical.pkl` (tracked).

**`winner_FGA_3.joblib` note**: `FgMakeAdapter.load()`
(`src/cbb_sim/engine/adapters.py`) unconditionally `joblib.load()`s
`winner_FGA_3.joblib` for every shot class in its loop, then overwrites the
`FGA_3` arm/model/features from `fg_make_FGA_3_decision8_F2.joblib` only if
`ENGINE_FG3=decision8`. Both files are required in the image regardless of
the flag.

**No `data/raw` dependency.** Nothing above is under `data/raw`; the engine's
`usage` adapter reads its priors out of `arrays_F2_2025.npz`/`names_F2_2025.json`
(already loaded), not from `data/processed/models/usage/usage_params_v1.json`
(that path is recorded in `run_meta.json` for provenance only, never opened).
`--bootstrap raw` in `run_aws_sweep.sh` exists only for parity with the cfb
pattern and is not needed for a sim run.

**Total: ~68 MB of engine inputs, of which ~45 MB (the S1 event artifacts) is
the one gap -- section 4.**

---

## 2. Getting the data onto the box

1. `git clone https://github.com/cmpeavlerjr72/CBB-redesign.git` (branch
   `main`) brings every row of the table above **except**
   `event_round2_s1_F2_2025/`, because `data/processed/` is git-tracked by
   policy (`.gitignore`: "data/processed/ and data/reference/ ARE tracked in
   git ... unrecoverable state this repo exists to protect").
2. `event_round2_s1_F2_2025/` (45 MB, 14 files) must be staged into
   `data/processed/models/engine/` **before** `docker build` runs --
   `Dockerfile.cbb` hard-fails the build (`test -d` / `test -f` checks) if
   it's missing, exactly like cfb's image-resident-parquet guard. See
   section 4 for why this is a manual step today and what to do about it.
3. Nothing else needs pulling. `--bootstrap raw` (`hf_sync_data.py pull --dirs raw`,
   ~1.6 GB) is available in `run_aws_sweep.sh` but not needed for a sim run.

---

## 3. Container build

- `requirements-cloud.txt` (repo root): full `pip freeze` of the Windows
  `.venv` on 2026-09-10 (git `db72e0d4b2`), minus the editable
  `-e git+...#egg=cbb_sim` line -- `run_engine.py` puts `src/` on `sys.path`
  itself, so the package need not be installed (same reasoning cfb used to
  drop its own editable line). Load-bearing: numpy 2.5.3, pandas 3.0.5,
  lightgbm 4.7.0, scikit-learn 1.9.0, scipy 1.18.1, joblib 1.6.0, pyarrow
  25.0.1. **scikit-learn and statsmodels are load-bearing even though the
  engine never calls them directly**: `cbb_sim.models.pace` imports
  `sklearn.linear_model` and `statsmodels.api` at module level, and
  `cbb_sim.models.clock` imports `pace`, so `ClockAdapter.load()` fails to
  import without both. The rest of the freeze (duckdb, numba, matplotlib,
  ruff, ...) ships as-is rather than hand-trimmed, same choice cfb made for
  `pyreadr`/`patsy`: a hand-trimmed list risks a transitive-import failure
  discovered only on the cloud box.
- `Dockerfile.cbb` (repo root): `python:3.12-slim` + `libgomp1`/`git`/
  `ca-certificates` + the pins + `COPY . /app` + a build-time existence check
  over every file in section 1's table + `chmod +x scripts/run_aws_sweep.sh`.
  Thread env vars pinned: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `LIGHTGBM_NUM_THREADS` all
  `=1` (mirrors `scripts/run_engine.py`'s own `_PIN` tuple), plus
  `OMP_THREAD_LIMIT=1`/`POLARS_MAX_THREADS=1`/`RAYON_NUM_THREADS=1` as the
  same insurance cfb's Dockerfile carries against the frozen-`num_threads`
  LightGBM footgun. **That footgun is already guarded in this repo's own
  code**: every adapter that loads a fitted booster
  (`EventAdapter`/`ClockAdapter`/`FgMakeAdapter`/`FreeThrowAdapter`/
  `ReboundAdapter` in `src/cbb_sim/engine/adapters.py`) calls
  `.set_params(n_jobs=1)` on it at load time, so the env pins are
  belt-and-suspenders here, not the load-bearing fix cfb needed.
- `.dockerignore` (repo root): excludes `.env`, `.venv/`, `data/raw/`,
  `results/**`, rebuildable intermediates. Does **not** exclude
  `event_round2_s1_F2_2025/` -- it must be present on disk and is not
  filtered out.

Build (on the box, not run here):
```
docker build -f Dockerfile.cbb -t cbb-sweep .
```

---

## 4. BLOCKER: the S1 event artifacts have no HF sync path today

`docs/models/engine/RESUME.md` section 2 states: "Per PM decision
2026-09-10, `data/processed/models/engine/event_round2_s1_*/` is now in
`.gitignore` and syncs to the private HF dataset via `scripts/hf_sync_data.py`."

**That is not currently true of the script.** `scripts/hf_sync_data.py`
defines `BULK_DIRS = ["raw", "results"]` and its `--dirs` argparse `choices`
are locked to exactly those two; `_root_for()` only maps `"raw"` ->
`data/raw` and `"results"` -> `results/`. There is no `"processed"` option,
so `event_round2_s1_F2_2025/` (or anything else under `data/processed/`)
cannot be pushed or pulled through this script as it stands.

**Effect:** a fresh clone + `hf_sync_data.py pull` (any `--dirs`) does NOT
reproduce a working `ENGINE_EVENT=round2_s1` input set. `docker build` will
fail its own existence check on `event_round2_s1_F2_2025/index.json` rather
than silently ship a broken image (that check does its job), but this stops
the whole chain cold.

**This was not fixed as part of this task.** Editing `hf_sync_data.py` is a
shared data-pipeline script six other workers may be using; per the worker
discipline in `CLAUDE.md` ("never overwrite a data file another worker may
be reading") this is flagged for the PM rather than changed unilaterally.

**Two ways to unblock, for the PM to choose between:**
1. Extend `hf_sync_data.py` to add `"processed"` (or a narrower
   `event_round2_s1`) to `BULK_DIRS`/the `--dirs` choices, mapped to
   `data/processed`, then `push --dirs processed` once from this box.
2. Stage the 45 MB directory manually per launch: `scp -r` or `rsync` it
   straight into the clone on the box before `docker build`, e.g.
   `scp -r data/processed/models/engine/event_round2_s1_F2_2025 \
   user@box:/opt/CBB-redesign/data/processed/models/engine/`. No script
   change, works today, must be repeated (or re-verified) every time the
   directory is rebuilt (`scripts/build_engine_event_round2.py`).

---

## 5. Run command, block size, and the honest resumability gap

```bash
docker run --rm -d --name sweep \
    -e HF_TOKEN="hf_xxx" \
    -v /root/out:/out \
    cbb-sweep \
    --tag F2_2025_s200_r2event --seeds 200 --chunk-seeds 50 \
    --workers <vCPU COUNT> \
    --games-per-block 60 --seeds-per-block 25 \
    --engine-event round2_s1 --engine-clock reference \
    --engine-rotation reference --engine-fg3 decision8
```

`--workers` should equal the vCPU count (`nproc` on the box) --
`run_engine.py` parallelises with a `ProcessPoolExecutor`, one process per
worker, each pinned to `OMP_NUM_THREADS=1` etc. internally, so workers are
the whole parallelism budget. `--games-per-block 60 --seeds-per-block 25`
matches `RESUME.md` section 3's own recommended block size: **small blocks
so a `--time-budget-s` cutoff (or a chunk boundary) still returns usable,
gradeable output** rather than one giant unit of work that either finishes
whole or contributes nothing.

**Structural gap vs. cfb, stated plainly:** `scripts/run_engine.py` writes
its `games.parquet`/`players.parquet` **exactly once, at process end** -- it
has no incremental checkpoint file the way cfb's `build_sweep_dist_v10.py`
resumes from. A spot kill mid-invocation loses that invocation's entire
seed range, not just the in-flight block. This was not changed (an engine
change needs its own parity re-emit, out of scope here). The workaround built
into `scripts/run_aws_sweep.sh` instead: it chunks the sweep **between**
`run_engine.py` invocations (`--chunk-seeds`, default 50), each with its own
`--seed-offset` and its own `results/engine_v0/<tag>_off<N>_n<K>/` output,
pushed to HF after every chunk. A kill costs at most one chunk's seeds, same
guarantee cfb's `--chunk` gives, without touching the engine. The tradeoff:
grading scripts (`eval_gates.py` etc.) expect one results directory, so
whoever runs the real sweep needs a manual concat step over the chunk
parquets afterward -- not built here, not requested by this task.

For a run without spot risk (on-demand, or a box you will babysit),
`--chunk-seeds 0` runs the whole `--seeds` count in one invocation.

---

## 6. Results back, and the push caveat

`run_aws_sweep.sh` mirrors `results/engine_v0/` into the bind-mounted `/out`
after every chunk, then calls `scripts/hf_sync_data.py push --dirs results`.
**That push uploads the ENTIRE local `results/` tree as a single
`HfApi.upload_folder` commit** -- unlike cfb's `push_results.py`, which
uploads only the files matching one tag. It is not incremental in the
per-file sense; every chunk's push re-sends everything not yet confirmed on
the remote. For a short sweep this is fine; for a long multi-chunk one it
gets slower as `results/` grows, and is worth the PM's attention if it
becomes the bottleneck.

Pull them back on Windows the same way as any other bulk dir:
```
.venv\Scripts\python.exe scripts\hf_sync_data.py pull --dirs results
```

**Output size, measured and extrapolated** (5 seeds x 5,710 games measured
at `results/engine_v0/F2_2025_s5_r2event/`: games.parquet 436 KB + players.parquet
4.0 MB = 4.5 MB total, ~337 bytes/player-row after parquet compression):

| seeds | full 5,710-game slate, extrapolated |
|---|---:|
| 5 (measured) | 4.5 MB |
| 200 | ~180 MB |
| 2,000 | **~1.8 GB** |

**A 2,000-seed full push crosses the task's 1 GB stop-and-report line.**
Nothing in this prep triggers that (no sweep has run), but whoever launches
the real 2,000-seed sweep should plan to report before that push, not
discover it mid-run.

---

## 7. The parity gate (proven on this box today)

`scripts/digest_engine_run.py` hashes a completed `results/engine_v0/<tag>/`
directory: every `games.parquet` row (all 20 columns) and every
`players.parquet` row (all 13 columns), floats rounded to 6 dp, sorted by
`(game_id, seed[, athlete_id])`, plus the `adapter_flags`/
`engine_rules_from_data` block of `run_meta.json` (performance fields like
`runtime_s`/`workers`/`created_at` are excluded from the hash on purpose --
a faster or slower box must not fail parity). `--emit` writes a reference
JSON with the sha256 and version/platform metadata; `--compare` recomputes
and diffs, exiting 0 only on a byte-identical hash and printing every
differing field otherwise. This is the CBB-adapted equivalent of cfb's
`parity_digest.py`, restructured because this engine's unit of work is the
whole multiprocess `run_engine.py` invocation and its contract IS the output
files, not a single pure `run_one(game_id, seed)` call.

**Gate procedure (what `run_aws_sweep.sh --parity gate` runs first, always):**
run `run_engine.py --seeds 5 --max-games 60` (RNG seeded on `(seed, game_id,
family)`, so the specific 60 games chosen doesn't matter, only that both
boxes run the identical set), then `digest_engine_run.py --compare` against
`docs/ops/parity_reference_windows.json`. **On any mismatch: stop, no sweep,
no upload** -- tolerance is a PM adjudication per `CLAUDE.md`'s "no hand
tuning" rule, never this script's.

**Done today, on the home box (`ENGINE_EVENT=round2_s1`,
`ENGINE_CLOCK=reference`, `ENGINE_ROTATION=reference`, `ENGINE_FG3=decision8`,
git `0e55ca6199`):**

| run | workers | blocking | wall | result |
|---|---|---|---|---|
| `smoke60x5` (reference) | 4 | 1 block of 60 games x 5 seeds | 44 s, 990 poss/s | sha256 `30ce31d846fc0590c320628946dd4c71eb14c0949580f310530d7e8ed493de2e` |
| `smoke60x5_verify` | 2 | 2 blocks of 30 games x 5 seeds | 35 s, 1,239 poss/s | **identical sha256** -- PASS |

The second run used a different worker count AND a different game/seed
block split and still produced a bit-identical digest -- direct evidence (on
top of `tests/test_engine.py`'s own unit test) that the RNG really is
independent of parallelism, which is the property the whole cross-platform
parity strategy depends on. Reference digest is committed at
`docs/ops/parity_reference_windows.json`.

**Caveat, same one cfb's runbook names for its own reference:** this digest
was emitted from a working tree with uncommitted changes elsewhere in the
repo (unrelated files -- rotation/usage docs and training scripts; nothing
under the engine's read path per section 1). Re-emit on Windows whenever the
engine, an adapter, or any file in section 1's table changes, and commit the
new reference in the same commit as that change, exactly as cfb's rule
requires.

---

## 8. Cost estimate

Using the cfb project's own recorded spot economics for the ~196-vCPU class
box (`cfb-props-sim/docs/ops/cloud_sweep_runbook.md` section 5: no real
176-vCPU `c7a` size exists, `c7a.48xlarge` at 192 vCPU is the nearest, spot
~$3.5-5.0/h vs. on-demand ~$9.85/h; spot quota `L-34B43A08` in `us-east-2`
was raised to 196 to fit it) and this project's own core-second figures
(`docs/tests/engine_seed_count_2026-09-10.md`, `docs/models/engine/RESUME.md`
section 3):

| seeds | core-seconds | wall @ 20-core home box | wall @ 192 vCPU (+ ~0.4h clone/build/gate) | spot cost | on-demand cost |
|---|---:|---:|---:|---:|---:|
| **200** | ~2x10^5 | ~2.8 h (RESUME.md's own direct measurement: 2.6 h) | ~0.7 h | **~$2.5-3.5** | ~$6.9 |
| **2,000** | ~2x10^6 | ~1 day (extrapolated 10x from the 200-seed measurement) | ~3.3 h | **~$11.5-16.5** | ~$32.5 |

Caveat, verbatim from the cfb runbook: **verify at checkout, do not treat
these as quotes.** These also assume the section-4 blocker is resolved and
the parity gate passes on the cloud box's exact image -- unproven until a
Linux build actually runs (never attempted here, no Docker on this box
either, same as cfb's own "unproven until the first Linux run" list).

---

## 9. HF dataset state (`mvpeav/cbb-sim-data`, private)

`scripts/hf_sync_data.py status` (read-only; nothing was pushed or pulled by
this task):

| dir | local files | missing on remote |
|---|---:|---:|
| `raw` | 911 | 14 (roster parquets for 2024-2026, and the `preseason/2027/` batch: 9 files) |
| `results` | 44 | 10 (mostly `results/engine_v0/F2_2025/*` and `results/engine_v0/seed_count/*`) |

24 files outstanding total, all well under 1 GB combined (roster/preseason
parquets and a handful of small engine result files) -- **not pushed**, per
this task's "stop and report before pushing >1 GB" instruction and because
nothing here required it. `HF_TOKEN` is present in `.env` and confirmed
working (the `status` call above authenticates and lists the remote repo).

**`data/processed/models/engine/` (67 MB total) is not part of either bulk
dir** and, per section 4, cannot be pushed through this script at all today.

Local sizes for reference: `data/raw` 1.6 GB, `data/processed` 780 MB,
`results` 132 MB (+ 96 KB from this session's smoke runs), `data/processed/models/engine`
67 MB (9.2 MB arrays + 45 MB `event_round2_s1_F2_2025/` + 1.4+2.7+8.1 MB
joblibs + 76 KB parquet/json).

---

## 10. Everything blocking an actual launch

1. **Section 4's blocker**: `event_round2_s1_F2_2025/` has no HF sync path;
   a fresh-clone box cannot build the image without a manual copy step or a
   PM-approved `hf_sync_data.py` change.
2. **No AMI/instance has been chosen or launched.** `c7a.48xlarge` (spot) is
   the size implied by the existing 196-vCPU quota and the cost table above;
   nobody has picked an AMI, security group, or key pair for THIS project.
3. **`docker build -f Dockerfile.cbb` has never been run.** Untested, no
   Docker on this box, same caveat cfb's own Dockerfile carried before its
   first Linux run -- the 56 pins in `requirements-cloud.txt` installing
   cleanly on manylinux is unverified.
4. **The parity gate has only been proven Windows-vs-Windows** (two runs on
   this box, different worker counts, identical digest). Cross-platform
   (glibc vs. MSVC float paths) is the real unproven variable, exactly as
   cfb's own runbook section 9 flags for its libm risk.
5. **No consolidation step exists** for the chunked sweep output described
   in section 5 -- grading scripts read one `results/<tag>/` dir, and a real
   multi-chunk run leaves several.
