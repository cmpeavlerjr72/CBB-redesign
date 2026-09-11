# AWS launch chain -- CBB engine seed sweeps (PREPARED, NOT LAUNCHED)

Status 2026-09-10 (updated, PM decisions applied same day): every file below
is written and the parity gate has been proven on the home box. **No
instance has been launched and no cost has been incurred.** This doc,
`Dockerfile.cbb`, `requirements-cloud.txt`, `scripts/run_aws_sweep.sh`,
`scripts/digest_engine_run.py` and `scripts/concat_engine_runs.py` are the
launch chain; nothing here executes anything on AWS.

**PM decisions applied in this update**: (1) `scripts/hf_sync_data.py` now
has a third bulk dir, `engine_inputs` -> `data/processed/models/engine/`,
and `engine_inputs` plus the previously-outstanding `raw`/`results` files
have been pushed -- section 4's blocker is resolved. (2)
`scripts/concat_engine_runs.py` (new) merges chunked `run_aws_sweep.sh`
output into one gradeable `results/engine_v0/<tag>/` -- section 5's
consolidation gap is resolved. (3) The AMI, instance type and security group
are not yet chosen for this project; per PM instruction they are to be
copied from cfb-props-sim's last launch when the time comes (section 10).

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
2. `event_round2_s1_F2_2025/` (45 MB, 14 files) must be present in
   `data/processed/models/engine/` **before** `docker build` runs --
   `Dockerfile.cbb` hard-fails the build (`test -d` / `test -f` checks) if
   it's missing, exactly like cfb's image-resident-parquet guard. As of
   section 4's fix, get it with:
   `.venv/Scripts/python.exe scripts/hf_sync_data.py pull --dirs engine_inputs`
   (or plain `pull` for all three bulk dirs) -- no manual `scp`/`rsync`
   needed anymore.
3. Nothing else needs pulling for a sim run. `--bootstrap raw`
   (`hf_sync_data.py pull --dirs raw`, ~1.6 GB) is available in
   `run_aws_sweep.sh` but not needed unless `SIM_USAGE_DIRICHLET`-style raw
   rebuilds are ever added to this engine (they are not today).

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

## 4. RESOLVED: the S1 event artifacts now have an HF sync path

`docs/models/engine/RESUME.md` section 2 states: "Per PM decision
2026-09-10, `data/processed/models/engine/event_round2_s1_*/` is now in
`.gitignore` and syncs to the private HF dataset via `scripts/hf_sync_data.py`."

**That was not true of the script when this doc was first written** --
`BULK_DIRS` was locked to `["raw", "results"]` with no `"processed"` option,
so `event_round2_s1_F2_2025/` could not be pushed or pulled through it.

**PM decision 2026-09-10 (same day): fixed.** `scripts/hf_sync_data.py` now
has a third bulk key, `engine_inputs`, mapped to
`data/processed/models/engine/` (the whole ~67 MB directory, not just the
gitignored subdirectory -- the duplication of already-git-tracked files on
HF costs little and keeps `_root_for()` a plain one-directory-in,
one-prefix-out mapping like `raw` and `results`; `raw`/`results` behaviour is
unchanged). `engine_inputs` plus the 27 files then outstanding under
`raw`/`results` (24 originally reported, +3 from this session's own smoke
runs) were pushed. Remote state after the push (`hf_sync_data.py status`):

| dir | local files | missing on remote |
|---|---:|---:|
| `raw` | 911 | 0 |
| `results` | 47 | 0 |
| `engine_inputs` | 20 | 0 |

`mvpeav/cbb-sim-data` now carries 978 files total, fully mirrored. A fresh
clone + `hf_sync_data.py pull --dirs engine_inputs` now reproduces a working
`ENGINE_EVENT=round2_s1` input set without any manual `scp`/`rsync` step.
`Dockerfile.cbb`'s build-time existence checks (section 3) are unchanged and
still the thing that would catch a future regression here.

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
grading scripts (`eval_gates.py`, `grade_market_games.py`,
`grade_market_props.py`, `diag_engine_multilevel.py`) expect one results
directory, so the chunk outputs need consolidating afterward.

**PM decision 2026-09-10: `scripts/concat_engine_runs.py` (new) does that
consolidation.** It finds every `results/engine_v0/<tag>_off*_n*/` chunk,
validates that `fold`/`season`/`n_games`/`adapter_flags`/
`engine_rules_from_data` are identical across all of them (refusing to merge
chunks from different configs), **refuses overlapping seeds** (checked
against each chunk's own `run_meta.json` `seeds` list, and independently
re-checked as a duplicate-`(game_id,seed)`-row assertion on the concatenated
frame), and writes `results/engine_v0/<tag>/{games.parquet,players.parquet,
run_meta.json}` -- validated through the SAME `cbb_sim.eval.contract` module
`eval_gates.py` uses, so "gradeable" means the same thing here as for a
single run. The merged `run_meta.json` additionally records `"chunks"` (each
chunk's tag, seed offset, seed list, runtime) and `"merged_from"`
(tool/timestamp/chunk count), so a later reader can see the run was
assembled rather than produced by one invocation. Refuses to overwrite an
existing output directory without `--overwrite`
(`.venv/Scripts/python.exe scripts/concat_engine_runs.py --tag <tag>
--results-dir results/engine_v0`). Tested end-to-end today (2-chunk clean
merge -> passes `cbb_sim.eval.contract.load_engine_results` with zero
warnings; a deliberately overlapping 3rd chunk -> refused with the specific
seed and chunk names named in the error).

For a run without spot risk (on-demand, or a box you will babysit),
`--chunk-seeds 0` runs the whole `--seeds` count in one invocation and no
concat step is needed.

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

**Updated after the PM decision.** `hf_sync_data.py status` originally found
24 files outstanding (14 `raw`, 10 `results`), all well under 1 GB combined
(roster/preseason parquets and a handful of small engine result files); this
session's own smoke/concat-test runs added 3 more `results` files before the
push. Per PM decision, `engine_inputs` plus all outstanding `raw`/`results`
files were pushed (`hf_sync_data.py push --dirs raw results engine_inputs`):

| dir | local files | missing on remote (before) | missing on remote (after push) |
|---|---:|---:|---:|
| `raw` | 911 | 14 | **0** |
| `results` | 47 | 13 | **0** |
| `engine_inputs` | 20 | 20 | **0** |

`mvpeav/cbb-sim-data` now carries **978 files, fully mirrored** (0 missing in
any bulk dir). `HF_TOKEN` is present in `.env` and confirmed working (both
the `status` calls and the push itself authenticated successfully). Total
pushed this session: ~67 MB (`engine_inputs`) + a few MB of small `raw`/
`results` files -- nowhere near the 1 GB stop-and-report threshold.

Local sizes for reference: `data/raw` 1.6 GB, `data/processed` 780 MB,
`results` 132 MB, `data/processed/models/engine` 67 MB (9.2 MB arrays + 45 MB
`event_round2_s1_F2_2025/` + 1.4+2.7+8.1 MB joblibs + 76 KB parquet/json).

---

## 10. Everything blocking an actual launch

1. ~~Section 4's blocker~~ **RESOLVED 2026-09-10**: `engine_inputs` now syncs
   via `hf_sync_data.py`; pushed and confirmed fully mirrored (section 9).
2. ~~No consolidation step~~ **RESOLVED 2026-09-10**: `scripts/concat_engine_runs.py`
   merges chunked sweep output into one gradeable dir (section 5).
3. **No AMI, instance type or security group has been chosen for this
   project.** Per PM decision, these are to be **copied from
   cfb-props-sim's last launch** when the time comes, rather than picked
   fresh -- that project's own runbook
   (`cfb-props-sim/docs/ops/cloud_sweep_runbook.md` section 6b) used a
   spot `c7a` family instance (Amazon Linux 2023 AMI resolved via the
   `resolve:ssm:/aws/service/ami-amazon-linux-latest/...` alias, not a
   pinned AMI id), an SSH-only-from-your-IP security group, and the
   `c7a.48xlarge` size implied by its 196-vCPU spot quota (`L-34B43A08`,
   `us-east-2`). Nobody has re-verified that quota, AMI alias, or security
   group still resolve the same way today, or that this project's key pair
   is the same one -- copy the commands, re-verify each value at launch
   time, do not assume they are still current.
4. **`docker build -f Dockerfile.cbb` has never been run.** Untested, no
   Docker on this box, same caveat cfb's own Dockerfile carried before its
   first Linux run -- the 56 pins in `requirements-cloud.txt` installing
   cleanly on manylinux is unverified.
5. **Linux parity is unproven until the first smoke run on the box.** Per PM
   instruction, this is recorded explicitly rather than assumed: the parity
   gate has only been proven Windows-vs-Windows so far (two runs on this
   box, different worker counts, identical digest -- section 7). Whatever
   box gets launched, its FIRST action must be `--parity only` against
   `docs/ops/parity_reference_windows.json`, and a mismatch (most likely
   cause: glibc vs. MSVC float paths in a sigmoid/exp/log call, exactly the
   risk cfb's own runbook section 9 names for its libm concern) stops
   everything -- no sweep, no upload, back to the PM, never a loosened
   tolerance.

---

## 11. fg_make round-2b artifacts: synced via the new `model_artifacts` HF key (2026-09-10)

`scripts/hf_sync_data.py` gained a fourth bulk dir, `model_artifacts` ->
`data/processed/models/` (the whole tree). Unlike `engine_inputs` (which
mirrors the full `engine/` directory, tracked files included, as a
deliberate simplification -- section 4), `model_artifacts` pushes **only**
the gitignored subset of `data/processed/models/`, computed fresh each run
via `git check-ignore --stdin`: most of that tree is git-tracked and must
never be duplicated onto HF. The `engine/` subtree is explicitly excluded
from this key's scan (it already has its own dedicated `engine_inputs` path;
scanning it again here would just re-upload the same S1 artifacts under a
second prefix).

This is the sync path for `data/processed/models/fg_make/round2/` (~21 MB)
and `data/processed/models/fg_make/round2b/` (~25 MB) -- the fg_make
bake-off scratch (grid-search checkpoints and, for round2b, monthly S1
fitted boosters) added to `.gitignore` this session
(`data/processed/models/fg_make/round2/`,
`data/processed/models/fg_make/round2b/`, plus a general
`data/processed/models/*/round*/` / `data/processed/models/*/*_s1/` pattern
for future round-N / S1 sets in any model directory, narrowed with `!`
negations for the two dirs that were already git-tracked at the time --
`possession_outcome/round2/` and `clock/v3b_s1/`, both of which are
unaffected and stay tracked as before). If a future `ENGINE_FG3` config
adopts a round2b fitted model into the engine's read path, pull it with:

```
.venv\Scripts\python.exe scripts\hf_sync_data.py pull --dirs model_artifacts
```

`Dockerfile.cbb`'s section-1/section-3 build-time existence checks would
need a matching entry added if/when that happens -- not done yet, since no
adopted engine config reads round2b today.

---

## 12. FIRST LAUNCH, 2026-09-11 (parity proven, box terminated)

**Credentials.** `aws sts get-caller-identity` OK, account ending `9871`.

**Parity reference regenerated as v2, NOT from the dirty working tree.**
The working tree had uncommitted engine-path changes from a concurrent
worker mid-session (`src/cbb_sim/engine/{adapters,loop}.py`,
`rotation_adapter.py` -- a real rotation-lineup-timing fix, unreviewed/not
yet bake-off-adopted). Emitting straight from that tree would have produced
a reference the box's clean `git clone` could never match (not a real
cross-platform defect, just an uncommitted local diff). Instead: a local
`git clone --local` of this repo pinned to commit
`718a3ebc4b55f1d8f79b444000fd12b11987037e` was made in a scratch dir, the
two gitignored engine-input trees
(`data/processed/models/engine/event_round2_s1_F2_2025/`,
`data/processed/models/fg_make/round2b/S_C_s1/`) were copied in, and the
60x5 smoke ran there with `ENGINE_EVENT=round2_s1 ENGINE_FG_MAKE=round2b_S_C_s1
ENGINE_CLOCK=reference ENGINE_ROTATION=reference ENGINE_FG3=decision8
--workers 4`. Reproducibility re-proven the same way as v1 (4 workers/1
block vs. 2 workers/2 blocks of 30 games -- bit-identical). Reference:
`docs/ops/parity_reference_windows_v2.json`, sha256
`89275a1dc6dbbddcc8be1dfd8f9a4ca924ad0b5cdc502109ba105eda4f4279d9`, pinned to
git `718a3ebc4b`. `docs/ops/parity_reference_windows.json` (v1) untouched.
**Commit `718a3ebc4b` is the exact ref the box was told to clone and check
out**, not "whatever `main` is" -- `main` advanced at least twice more
during this session from other concurrent workers.

`scripts/digest_engine_run.py` extended (`digest_version` 1 -> 2): every
`ENGINE_*` key is pulled out of `run_meta.json`'s `adapter_flags` into its
own top-level `flags` block in the digest doc, and `--compare` prints both
sides' flags and warns explicitly on a flags mismatch, so a wrong-config run
prints as one obvious line instead of being buried in a full metadata diff.
This changed file (like the v2 reference) is NOT committed; it was `scp`'d
onto the box into the cloned tree before `docker build` so the image baked
in the same script version used to emit v2.

**Launch.** Spot capacity was available on the first try -- no fallback to a
smaller instance was needed.

| | |
|---|---|
| Instance | `i-02092cafa1d72fdde`, spot, `c7a.48xlarge` (192 vCPU, 369 GiB RAM) |
| Region / AZ | `us-east-2` / `us-east-2c` |
| AMI | `ami-02b1d32fdf87a0437` (AL2023, `al2023-ami-2023.12.20260909.0-kernel-6.1-x86_64`) |
| Key / SG | `cfb-sweep` / `sg-05aacf67a5a55fbf7` (`cfb-sweep-ssh`, copied from cfb-props-sim) |
| Launch (real clock) | `2026-09-11T02:04:24Z` |
| Terminate call | `2026-09-11T04:22:06Z`; `describe-instances` confirmed `terminated` ~15s later |
| Wall duration | ~2h18m |
| Cost (spot, measured rate $2.26-2.80/h in this session) | **~$5.2-6.4** |

**On-box setup.** `dnf install docker git` (AL2023), repo cloned via HTTPS
(public read, no token needed) and checked out to `718a3ebc4b`. `hf_sync_data.py`
needs `huggingface_hub` on the bare-metal host (not just in the container) to
stage `event_round2_s1_F2_2025/` and `fg_make/round2b/` before `docker build`;
the pinned `huggingface_hub==1.31.0` from `requirements-cloud.txt` needs Python
>=3.10 and AL2023's system `python3` is 3.9, so a throwaway venv on
`huggingface_hub==0.34.6` was used for this staging step only (host-side
utility, not inside the parity-sensitive image; irrelevant to the digest).

**BUG FOUND: `hf_sync_data.py pull` for `engine_inputs`/`model_artifacts` lands
one directory too deep.** `push()` uploads with `path_in_repo=<dirname>`
(e.g. `engine_inputs/...`), but `pull()`'s `snapshot_download(local_dir=root,
allow_patterns=[f"{d}/**"])` does not strip that prefix, so files land at
`root/<d>/...` instead of `root/...` -- e.g.
`data/processed/models/engine/engine_inputs/arrays_F2_2025.npz` instead of
`data/processed/models/engine/arrays_F2_2025.npz`. Confirmed on first pull
(engine_inputs: 67 files, model_artifacts: 773 files, both nested one level
under a spurious `engine_inputs/`/`model_artifacts/` subdirectory). Worked
around by hand on the box (`cp -a .../model_artifacts/. ...models/; rm -rf
.../model_artifacts`, similarly for `engine_inputs`) rather than patched, given
the session's time-box; `raw`/`results` likely have the same defect (untested
here -- `raw` was skipped entirely per section 1, "not needed for a sim run").
~~This needs a real fix in `hf_sync_data.py`'s `pull()` before the next cloud
launch~~ **FIXED (Sonnet worker, 2026-09-11).** `pull()` now downloads each
bulk dir into a scratch staging directory first (`snapshot_download(local_dir=
<staging>, ...)`, which still mirrors the `<d>/`-prefixed repo path under
`local_dir`), then moves each file from `<staging>/<d>/<rel_path>` to
`root/<rel_path>` via the new `_remote_to_local_rel(d, remote_path)` -- the
exact inverse of `_local_to_remote(d, rel_path)`, which `local_files()` (and
now `pull()`) both use as the single source of truth for the prefix mapping.
Confirmed **all four** `BULK_DIRS` keys had the identical defect (not just
`engine_inputs`/`model_artifacts` as first suspected here), since all four
share the same `_root_for`/`path_in_repo=d` shape. Verified by: (1) a dry-run
`snapshot_download(..., dry_run=True)` reproducing the pre-fix
`.../engine_inputs/arrays_F2_2025.npz` nesting; (2) `tests/test_hf_sync_paths.py`
(44 cases, no network) proving `_local_to_remote`/`_remote_to_local_rel` are
exact inverses for all four keys, including under an arbitrary `--dest-root`
override; (3) a real `pull --dirs engine_inputs --dest-root <scratch dir>`
(new CLI flag, defaults to the repo) landing all 20 files at the correct
unnested relative paths, byte-identical (`cmp`) to the corresponding files
already on disk under `data/processed/models/engine/`. Push layout now
provably equals pull layout for every key.
Separately, an early pull attempt against the pinned `huggingface_hub==1.31.0`
(before the py3.9 incompatibility was caught) and a stray inline
`HF_TOKEN='...'` shell-variable prefix together caused the token to appear
briefly in this session's own `pgrep -af` tool output (not printed
deliberately, not included in this doc) -- flagging per this repo's own
"revoke/rotate whenever a token is exposed" posture; **the HF token used
tonight should be rotated.**

**Parity result: PASS on every simulated value; one cosmetic metadata diff.**
`digest_engine_run.py --compare` against v2 from inside the built container
(same flags, `OMP_NUM_THREADS=1`, 60 games x 5 seeds) reported **zero**
differing `games.*` or `players.*` fields -- every possession-level box score
and player row is bit-identical between the Windows reference and this Linux
box. The ONE reported diff was `meta.adapter_flags` as a whole, and inspecting
it shows the entire difference is `sources.*.path` strings rendered with
Windows backslashes (`data\processed\models\...`) on the reference vs. POSIX
forward slashes (`data/processed/models/...`) on Linux -- `str(Path(...))`
being OS-native, embedded in provenance metadata that happens to be hashed.
**No computed value differs.** Per CLAUDE.md/this script's own rule, this is
not this worker's tolerance call to make -- reported to the PM rather than
patched; the fix, when decided, is almost certainly normalizing `sources.*.path`
to posix in `adapters.py` (or excluding raw path strings from the digest hash
the way `runtime_s`/`workers` already are) rather than anything about the
simulation itself.

**Throughput, measured (200/2,000-seed estimates are now real numbers).**
First attempt at 192 workers used `--games-per-block 60 --seeds-per-block 25`
(copied from the 4-worker smoke) -- with only 60 games and 25 seeds total that
forms exactly ONE block, so only 1 of 192 workers ever ran; measured 3,260
poss/s was a single-core number mislabelled, not a finding about the box.
Corrected with `--games-per-block 1 --seeds-per-block 5` (300 blocks, enough to
spread across 192 workers):

| run | workers | blocking | wall | possessions | throughput |
|---|---|---|---|---:|---|
| box_throughput60x25_v2 | 192 | 300 blocks of 1 game x 5 seeds | 13.85s | 210,605 | **15,963 poss/s (83 poss/s/core)** |

**83 poss/s/core is well below the home box's measured ~817-880 poss/s/core**,
and this run's wall time (13.85s) is short enough that 192-process pool
spin-up/model-load is a real, un-amortized fraction of it -- this number should
be read as a conservative floor, not a clean per-core rate. A longer follow-up
(more games/seeds, several hundred blocks per worker instead of ~1.5) is needed
before trusting a revised cost table; that follow-up was not done here
(time-boxed). Using this measured number as-is (honest, not extrapolated
past what was seen):

| seeds | possessions (5,710-game slate) | wall @ 192 vCPU (measured rate) | + ~15min clone/build/gate | spot cost (~$2.3-2.8/h) |
|---|---:|---:|---:|---:|
| 200 | ~160M | ~2.8h | ~3.05h | ~$7-8.5 |
| 2,000 | ~1.6B | ~28h | ~28.25h | ~$65-79 |

These are **worse than the prior doc's estimate** (section 8: ~0.7h / ~3.3h at
192 vCPU) because the measured per-core rate here (83 poss/s/core) is ~10x
lower than the ~850 poss/s/core the prior estimate assumed by extrapolating
from the home box. Given the pool-startup caveat above, this may be
pessimistic rather than a real 10x regression -- **do not size a real sweep off
this number without a second, longer measurement** (e.g. 60 games x 200 seeds,
or the full 5,710-game slate x a handful of seeds, so startup is <5% of wall
time).

**Files written this session:** `docs/ops/parity_reference_windows_v2.json`
(new), `scripts/digest_engine_run.py` (flags field, `digest_version` 2),
this section. `docs/ops/parity_reference_windows.json` (v1) untouched.
`main` moved at least twice from other workers while this ran (`259d42da42`
-> `718a3ebc4b` -> `2a7f6242d5` observed); `718a3ebc4b` is the commit
everything above describes and the one the box actually ran.

**Termination confirmed:** `describe-instances` returned `terminated` for
`i-02092cafa1d72fdde` at `2026-09-11T04:22:2x Z`, ~2h18m after launch.

---

## 13. SECOND LAUNCH, 2026-09-11 morning: full 200-seed A/B, parity gate fixed for real, a grading-scale bug found and fixed

**Instance.** `i-091914a0a5285b681`, spot `c7a.48xlarge`, `us-east-2b`. Launched
`2026-09-11T14:00:45Z` (10:00:45 ET), terminated (verified via `describe-instances`)
`2026-09-11T14:47:51Z` (10:47:51 ET), ~47m6s wall, spot rate $2.2976/h, **~$1.80**. Engine
commit `c824711b67e9bd82b2fc32ef1bd4d506bbb592d2`. Full account, gate table and market
scorecard: `docs/tests/engine_v1_gates_F2_2025_s200_aws_2026-09-11.md`. This section is the
ops/timeline record; that doc is the evidence record.

**Setup was fast**: 6m14s to host-setup-done (docker/git install, clone, `hf_sync_data.py pull
--dirs engine_inputs model_artifacts` -- both already fully mirrored on HF, 0 missing, so no
push was needed this time), 1m43s to `docker build`. The section 12 pull-path bug (fixed
2026-09-11 by a Sonnet worker, `230e302`) held up cleanly on both bulk keys.

**Parity gate failed once, on a KNOWN defect, and was fixed for real this time.** The v3
reference (`493a818eeb`) was stale (four engine-path files had moved since, all from
"NO ARM ADOPTED" clock/rotation experiments). A fresh local re-emit at `c824711b` (v4) hashed
byte-identical to v3, confirming those commits changed nothing about the served config's
output. The box's first gate attempt against v4 still FAILED, on the exact same cosmetic
`meta.adapter_flags` mismatch section 12 already diagnosed and left unfixed
(`sources`/`manifest` provenance paths rendered with Windows backslashes vs POSIX) --
recurring now on the v1 served stack's dated S1 manifests (clock/fg_make/free_throw/rebound/
rotation all carry this). **Fixed this time**: `scripts/digest_engine_run.py` `digest_version`
2 -> 3, recursively posix-normalizes string leaves of the hashed metadata block before hashing.
Zero engine files touched, zero simulated values affected. Re-emitted as
`docs/ops/parity_reference_windows_v5.json` (sha `e4d4e3a76b...`, pinned `c824711b`), mounted
over the already-built image (no rebuild needed) for a second gate attempt: **PASS,
bit-identical.** `docs/ops/parity_reference_windows.json` (v1) and `..._v2.json`/`..._v3.json`
are untouched; v5 is the one to use for any future box on this served config until it changes
again.

**Throughput: the instructed 200-game x 4-seed test at 192 workers measured 25.4 and 15.7
poss/s/core** (two block shapes tried), both severely startup/IPC-overhead-dominated by
construction (200 games x 4 seeds cannot spread past ~1 block/worker at 192 workers no matter
the block size) -- the same failure mode section 12 already flagged for this exact test
shape, now reproduced and shown to be ~45-75x off. **Cross-checked against the real sweep's own
block shape** (`--games-per-block 30 --seeds-per-block 4`): a 12-seed pilot pair measured
1,033.6 / 1,022.7 poss/s/core at 96 workers; the real 200-seed A/B runs sustained 1,196.1 /
1,197.3 poss/s/core at 96 workers -- consistent with (better than) the ~817-880 poss/s/core home
box figure this project has used for cost tables. **Anyone costing a future sweep off a small
game x seed-count throughput probe should use the production block shape for that probe, not
`--games-per-block 1`** -- this is now the second time the small-probe number has undershot the
real number by more than an order of magnitude.

**Seed count: 200/200 for both A and B, not cut down.** The cross-checked throughput put a
paired 200/200 run at ~23 minutes wall (both ran concurrently, 96 workers each), well inside the
12:00 ET checkpoint, so no reduction was needed. Both launched 10:21:07/09 ET, both completed
naturally (not via the `--time-budget-s 3000` safety net, armed but never triggered) at
10:44:17/18 ET, `partial=False`, `seeds_dropped_incomplete=0` on both. Run A: seeds 0-199,
`F2_2025_s200_v1_clockv3c_A`. Run B (noise floor): seeds 1000-1199,
`F2_2025_s200_v1_clockv3c_B`. Pulled to the local machine via `scp` (206 MB each), pushed to
`mvpeav/cbb-sim-data` under `results` (`hf_sync_data.py push --dirs results`: `push-results: OK`)
before termination.

**A grading-scale defect was found and fixed while producing the gate table, not before.**
`src/cbb_sim/eval/gates.py`'s G8 used `.groupby([...]).transform(lambda x: x.rank(...) <= 5)`
twice; that lambda form forces a per-group Python callback in pandas, invisible at 5-50 seeds
but O(n_groups) at 200 (2.28M `(game, seed, team)` groups over ~29M player rows) -- it grew past
10 GB resident memory and ran 20+ minutes without finishing on the shared local grading box,
risking an OOM that would have cost every concurrently-running lane's work, not just this one.
Killed (a process this worker started) before that happened, fixed with the vectorized
`.groupby([...]).rank(...) <= 5` (verified bit-for-bit identical output on a tie-including
synthetic case and on an existing small local results dir, before and after), after which
`eval_gates.py` on the full 200-seed run completed in under a minute. Report-tool performance
fix only; no engine default, adapter, or simulated value changed.

**Instance termination:** `aws ec2 terminate-instances --instance-ids i-091914a0a5285b681`,
confirmed via `aws ec2 describe-instances ... --query State.Name` -> `terminated` at
`2026-09-11T14:47:51Z`, ~9s after the terminate call (`shutting-down` at :27 and :39, `terminated`
at :51 on three successive 10s-spaced polls).

**Files this session**: `scripts/digest_engine_run.py` (digest_version 3, posix fix),
`docs/ops/parity_reference_windows_v5.json` (new), `src/cbb_sim/eval/gates.py` (G8 vectorized
rank), `scripts/diag_gate_noise_band.py` (new -- paired-run G1-G9 noise-band table), this
section. `docs/ops/parity_reference_windows.json`/`..._v2.json`/`..._v3.json`/`..._v4.json`
untouched (v4 was a same-session intermediate, superseded by v5's posix fix; not separately
committed).
