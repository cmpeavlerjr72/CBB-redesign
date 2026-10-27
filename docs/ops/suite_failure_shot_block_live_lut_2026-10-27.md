# Suite failure: tests/test_shot_block_live_lut.py::test_attach_serves_live_table_and_missing_fails_loudly (2026-10-27)

Diagnostic only. Nothing in `src/`, `tests/`, the live arrays or the scheduler was modified. Worker: Sonnet, 2026-10-27.

## Answer

The cause is a latent defect in the shot-block cache key, not a scheduler job and not a changed file. `cbb_sim.engine.shot_block.load` caches the built table in the module-level dict `_CACHE` under the key `(arm, id(inp))`. `id()` is the object's memory address, and it is reused once the original object is garbage collected. Another test (`tests/test_shot_block_engine.py`, and any test that loads the full 5,710-game F2_2025 inputs) leaves a 5,710-game table in `_CACHE`. If the later `sliced` fixture's 6-game `EngineInputs` object lands on the same address, `SBK.load(SBK.DEFAULT, s)` returns the stale 5,710-game table instead of building the 6-game one. The assertion `sb.team.shape == (len(s.games), 2, 2)` then fails with (5710, 2, 2) vs (6, 2, 2), which is the observed "5710 vs 6". Whether the address is reused depends on allocator and GC timing, so the failure is intermittent and depends on what ran before the test in the same process.

## Evidence

1. The failure did not reproduce today: the file alone (2 passed), the three shot_block test files together twice (9 passed each), and the full suite at 12:0x ET (`pytest tests -q`: 880 passed, 1 skipped, 298 s). The 10-27 run that failed (879 passed / 1 failed) is not available to inspect, so the exact trigger in that run is not observed; the mechanism below is.
2. Mechanism confirmed by construction (`scripts/diag_shot_block_cache_repro_v1.py`): load the full F2_2025 inputs, `SBK.load("K2_Ocell", inp)` caches a table of shape (5710, 2, 2) under `(arm, id(inp))`; place that cached object under the key of a 6-game sliced inputs object; `SBL.attach(s, tmp)` then `SBK.load(SBK.DEFAULT, s)` returns shape (5710, 2, 2) while `len(s.games)` is 6. This is the test's exact failing comparison. The script could not force a natural address collision in 7 tries; the forced equivalent demonstrates the consequence, not the probability.
3. The test reads and writes nothing the scheduler touches. `REF = data/processed/models/engine/shot_block_K2_Ocell_F2_2025.npz` is dated 2026-09-30 12:44:31 and unchanged since (no `git status` entry under `data/processed/models/`). `SBL.attach(s, tmp_path)` writes the live table into pytest's `tmp_path`. `data/processed/models/engine_live/` holds no shot-block file at all (610 files: `arrays_*`, `event_block_*`, `names_*`, `build_diag_*`; newest 2026-10-08 08:51 for D1P_A3_F2_2027).
4. Scheduler: tasks `DailyChain_Morning` and `DailyChain_Evening` run `scripts/run_daily_chain_pass_v1.ps1`; the ingest manifests dated 10-15 to 10-27 (`data/processed/ingest/manifests/ingest_2027_*`, twice daily at about 00:00 and 13:00 ET) and `chain_v3/2026-10-14..27.json`, `tip_times_2027.parquet`, `own_ratings_2027.parquet` are the only files changed. `run_daily_chain_pass_v1.ps1` does not reference shot_block or `engine_live`. None of these is read by the failing test.
5. Git: no change to `tests/test_shot_block_live_lut.py` since its creation (32ec026), `src/cbb_sim/engine/shot_block.py` last changed 2026-10-01 (e59dc56), `scripts/build_shot_block_lut_live_v1.py` last changed 2026-10-09 (d727cbb). The only commit touching `src` or `tests` since 10-14 adds `tests/test_pull_bigquery_ncaa.py`.
6. Dependency change, ruled out as the cause but noted: `.venv/Lib/site-packages/pyreadr*` was installed 2026-10-27 10:12 (for the GameZone `.rds` read). numpy 2.5.3 and pandas 3.0.5 are unchanged by it. The new test file adds one more module-level import to the suite, which can shift allocation order in a single-process run; that is a plausible trigger for the address reuse, not a cause.

## Proposed fix (NOT applied)

Test-side (smallest; makes the test independent of run order). In `tests/test_shot_block_live_lut.py`, after the `pytestmark` line:

```diff
+@pytest.fixture(autouse=True)
+def _fresh_shot_block_cache():
+    SBK._CACHE.clear()
+    yield
+    SBK._CACHE.clear()
+
+
 @pytest.fixture(scope="module")
 def sliced():
```

Source-side (removes the defect for any caller; changes only the cache key, so served numbers are unchanged). In `src/cbb_sim/engine/shot_block.py`, `load`:

```diff
-    key = (arm, id(inp))
+    gid = inp.games["game_id"].to_numpy()
+    key = (arm, id(inp), len(gid), hash(gid.tobytes()))
```

Either is enough to stop the failure; the source-side one also protects the daily chain and any multi-slate process from serving a previous slate's table to a later object at the same address. The PM decides which; the source-side change touches a served module and should get a paired run before it ships, although the key change cannot alter a table that was built for the right object.
