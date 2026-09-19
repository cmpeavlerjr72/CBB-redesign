# Possession outcome, `A1`/`A2` alignment cells: the joblib-parallel execution attempt

2026-09-18. Pre-registration: `docs/models/possession_outcome/experiments.md` section 16
(execution-scheme-only amendment for the two alignment cells left NOT RUN by sections 11.6
and 14). Results: the same file, section 18. Trainer:
`scripts/train_possession_outcome_v4_par.py` (versioned sibling of
`train_possession_outcome_v4.py`, which is untouched). This is the fourth attempt at these
two cells: local wall-clock twice (rounds 4/4b), AWS once (section 14), and this session.

**Bottom line: `A1` reached 24 of 29 pre-registered refit dates before the session's hard
stop and is PARTIAL, not graded. `A2` was not attempted. Decision 9c remains unresolved for
possession-outcome.** Nothing here amends Decision 9, changes a served default, or edits
`ARCHITECTURE_DECISIONS.md` / the change ledger.

---

## 1. Local LightGBM threading benchmark

`.venv/Scripts/python.exe`, `lightgbm==4.7.0`, 300,000 x 50 random rows, 6-class random
labels, `n_estimators=100, num_leaves=63`:

| n_jobs | wall time |
|---|---|
| 1 | 39.85 s |
| 20 | 217.75 s |

`n_jobs=20` is **5.5x slower**, not faster. Unlike the AWS container (section 14, where
`n_jobs` had zero measured effect on a broken wheel), this wheel does spin up real OS
threads under `n_jobs=20`, but thread-management overhead exceeds any histogram-building
gain at this problem size on this box. This independently confirms the joblib-across-dates
design on local hardware too, not only as a workaround for the AWS wheel.

## 2. Serial-vs-parallel identity check

**Deviation, disclosed.** The pre-registered check ("one or two refit dates run serially vs
through the parallel path give identical predictions") was planned at the full
pre-registered `n_estimators=400` on real fold-F1 data. A real attempt at that scale (a
40,000-row downsample, chosen for speed) did not finish inside 5.5 minutes on this heavily
loaded shared machine and was killed without a result -- the first concrete evidence that
per-fit cost tonight, under real contention, sat far above the 150-530 s/fit (4-thread)
historical figure in experiments.md section 8.7. A second attempt through the actual
joblib/loky multi-process path (distinct OS PIDs confirmed spawning) also did not finish
both of 2 real cuts within several more minutes under the same contention.

To land a result before committing the compute budget to the full run, `n_estimators` was
monkeypatched to 20 for THIS CHECK ONLY (the same runtime-patch mechanism the trainer uses
for `n_jobs`; `A1` itself ran with the full pre-registered `n_estimators=400` and every
other pre-registered param unchanged), and the check was completed by calling
`_fit_cut_worker` -- the exact function joblib dispatches to a subprocess -- DIRECTLY,
in-process, for two real refit dates (fold F1, `first`, `G0`, `S1_monthly`, cuts
2023-11-01 and 2023-12-01, `tr` downsampled 1,197,609 -> 4,000 rows), and comparing against
`train_possession_outcome_v3.fit_predict_walkforward`'s serial output on the same reduced
data and seed.

| | serial (`R3.fit_predict_walkforward`) | direct call to `_fit_cut_worker` x2 |
|---|---|---|
| wall time | 14.5 s | 12.5 s |
| n_fits | 2 | 2 |

**Result: EXACT MATCH.** `np.array_equal(p_serial, p_direct)` is `True`; max abs diff
`0.0`; `n_fits` equal; segment metadata (`refit_date`, `n_train`, `n_train_from_test_season`,
`n_scored`, `max_train_date`) equal for both cuts. This confirms the refactor's
computational logic -- the segment/`before` masks, the strictly-before rule, the
`prior_test` concatenation order, the calls into `PO.fit_arm`/`PO.predict_arm` -- is
bit-identical to the serial trainer given the same inputs. It is a logic-equivalence proof
completed by direct invocation, not a timed multi-process proof under load; LightGBM with a
fixed seed and `n_jobs=1` is deterministic regardless of which OS process calls it, so the
untested increment (does distributing the same call across a real subprocess change the
bits) is expected, not measured, to add nothing. This is recorded plainly as a deviation
forced by the session clock, not asserted as equivalent-strength evidence to a full-scale
timed run.

## 3. `A1` (`first`/lgbm/F2/`G0`/`S1_conf_aligned`, 29 refits): PARTIAL

Launched 2026-09-18 **20:19:30 ET**, `--mode run --stage 7 --n-jobs 6 --stop-at 20:58`,
6-way joblib process parallelism, every fit single-threaded (`n_jobs=1` patched at
runtime), full pre-registered params (`n_estimators=400`, `num_leaves=63`,
`min_child_samples=400`, `learning_rate=0.06`, `subsample=0.8`, `subsample_freq=1`,
`colsample_bytree=0.9`, `reg_lambda=1.0`), seed 0.

**24 of 29 refit dates checkpointed** when the pre-registered stop-at fired at 20:58:00 ET;
the 5 still in-flight were cancelled by joblib on exit and are not saved (nothing already
checkpointed was touched or recomputed). **No cell-level grade is possible or reported**:
`R4.grade` requires every fold-2 `first` test row scored, and 5 refit windows are unscored.
No log loss, gate, or segment number is available for `A1` against the reference
(1.515428) or the floor (0.000804) from this session.

### 3.1 Measured throughput (real, not projected)

`n_train` per refit ranged 1,908,534 - 2,220,026 rows (the full F2 training slice plus a
growing in-season `prior_test` slice). Per-date completions, in the order they finished:

| # done | refit_date | n_train | n_scored | cumulative avg s/date |
|---|---|---|---|---|
| 1 | 2024-11-27 | 2,042,759 | 10,728 | 691 |
| 6 | 2024-11-01 | 1,908,534 | 134,225 | 128 |
| 12 | 2024-12-10 | 2,111,609 | 24,515 | 115 |
| 18 | 2024-12-30 | 2,184,203 | 3,695 | 106 |
| 24 | 2025-01-05 | 2,220,026 | 16,446 | 95 |

The printed average is cumulative-since-launch (total elapsed / dates done), not marginal,
so it drops as the fixed 6-worker warm-up cost amortises; the settling marginal rate by the
end of the run was roughly 95-105 s/date at 6-way parallelism on a machine shared with
several other lanes throughout (confirmed by CPU-accounting spot checks during the run).
Wall clock for the 24 completed dates: 38.5 minutes (20:19:30-20:58:00 ET).

### 3.2 Resume (checkpoints kept; already-fit dates never recomputed)

```
PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/Scripts/python.exe scripts/train_possession_outcome_v4_par.py \
  --mode run --stage 7 --n-jobs 6 --stop-at <HH:MM>
```

Checkpoint directory:
`data/processed/models/possession_outcome/round4_a1a2/cuts/first_F2_lgbm_G0_S1_conf_aligned_s0/`
(24 `pred_<k>.npy` files + `done.json`). At the ~95-105 s/date marginal rate observed late
in this run, the remaining 5 dates at 6-way parallelism should need well under 10 minutes
on an equally-loaded machine.

## 4. `A2` (`first`/lgbm/F2/`G0`/`S1_weekly`, ~23 refits): NOT ATTEMPTED

Per the PM's mid-session instruction ("if both cannot finish by ~21:00 ET, `A1` runs to
completion first"), all remaining session time went to `A1`. Resume with `--stage 8` once
`A1` is complete, or independently at any time (the two cells' checkpoints do not overlap).

## 5. Decision 9c reading

**Still unresolved for possession-outcome.** Four sessions across four different fixes
(local serial x2, AWS with broken threading, local joblib-across-dates) have not produced a
scored `A1` or `A2` log loss. The blocker has moved: it is no longer "LightGBM does not
multi-thread" (section 1 above and experiments.md section 14 both show that diagnosis is
either irrelevant here or actively backwards on this box) -- it is that a single refit on
the full ~2-2.2M-row `first`/F2 training slice, at the pre-registered `n_estimators=400`,
costs on the order of 1.5-11 minutes each under real contention on a shared 20-core box,
and 29 (or 23) such refits at a 6-worker cap is a 30-90 minute job regardless of
orchestration.

Segment cells (early season / conference vs non-conference, per-team quintile slope) are
**not computable**: they require a graded cell, and no cell is graded this session. This is
recorded as such rather than left silent.

**Recommendation, as evidence, not a ruling:** `A1`'s 24/29 checkpoint is close enough to
complete that one uninterrupted window (quiet local machine, or a cloud box with a WORKING
multi-threaded wheel run with `n_jobs=1` per fit and process-level parallelism as here,
rather than relying on LightGBM's internal threading) should finish it outright. The
section-14 fix (a working multi-threaded cloud wheel) and this session's fix (parallelise
across dates) are complementary, not substitutes: this session shows the process-level
parallelism mechanism works (24 real dates fit and checkpointed) but remains throughput-
bound by per-fit cost on a contended box. The PM rules on Decision 9 from this table.

## 6. Wall clock and deviations

Identity-check attempts and diagnosis: ~20:02-20:19 ET. `A1`: 20:19:30-20:58:00 ET (38.5
min). Wrap-up, process cleanup, documentation: 20:58-21:10 ET.

Deviations, all disclosed above: (1) the identity check ran on downsampled data and, after
a full-scale attempt failed to finish in time, at a reduced `n_estimators=20`, completed by
direct function call rather than the timed joblib path (section 2); (2) `A2` was not
attempted (section 4); (3) `A1` is PARTIAL with no graded cell, per the PM's mid-session
tightening of the original "run to completion" instruction. No arm, fold, feature, seed, or
param changed from the pre-registration; no `src/cbb_sim/` file was edited; no engine
default, flag, or change-ledger entry was touched.
