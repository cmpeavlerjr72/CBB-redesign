# ENGINE v0 -- resume state (2026-09-10, 16:35 ET)

Written at a hard session stop. This file says exactly what exists, what is
provisional, what is unfinished, and the command to continue. Read
`docs/models/engine/model.md` first for the architecture;
`docs/tests/engine_v0_F2_2026-09-10.md` for the gate read.

---

## 1. What exists and is complete

| file | state |
|---|---|
| `src/cbb_sim/engine/__init__.py` | complete |
| `src/cbb_sim/engine/state.py` | complete. GameState as a struct of arrays; the rule era (bonus thresholds) read from `bonus_era.json` into state. |
| `src/cbb_sim/engine/rng.py` | complete. `uniforms_at` (per-row index) + `StreamBook` (one counter per (family, simulation)). Verified bit-identical to `control.rng.uniforms` for a constant index. |
| `src/cbb_sim/engine/inputs.py` | complete. `EngineInputs` container + `FeaturePlan` static/state splice + save/load. |
| `src/cbb_sim/engine/adapters.py` | complete. One batched adapter per sub-model + the flag/provenance bundle. |
| `src/cbb_sim/engine/rotation_adapter.py` | complete. `RotationSampler.next_lineup`'s decision rule vectorised over (2N, 15). |
| `src/cbb_sim/engine/loop.py` | complete. The possession loop (a)-(g). |
| `scripts/build_engine_inputs.py` | complete and RUN. Output in `data/processed/models/engine/`. |
| `scripts/run_engine.py` | complete. Sealed guard, multiprocessing, partial-run handling. |
| `tests/test_engine.py` | complete. 5 of 10 tests were RUN and PASSED before the stop (`uniforms_at` == `control.rng.uniforms` bit for bit; three consecutive draws differ in every family; team score == sum of player points and team FGA/FG3A/FTA == the player sums; minutes == 5 x game seconds per side; OT length + team-foul carry-over; contract validation of both frames). The other 5 (determinism, batch-independence, five-on-floor, period/OT transitions, allocation spread) were written but NOT executed -- run `pytest tests/test_engine.py -q`. |
| `docs/models/engine/model.md` | complete. |

Prep artifacts on disk (`data/processed/models/engine/`, built 16:10, 153 s):

    games_F2_2025.parquet      5,710 games (the eval harness's own universe)
    arrays_F2_2025.npz         team_static (5710, 2, 28), slot_static (5710, 2, 15, 19),
                               roster/rotation/usage/rebound blocks
    names_F2_2025.json         column maps, the data-derived rule constants, provenance
    rebound_F2.joblib          lgbm / C_plus_state refit on 2022-2024
    free_throw_F2.joblib       lgbm refit on 2022-2024
    fg_make_FGA_3_decision8_F2.joblib   the Decision-8 LightGBM for FGA_3

Measured at prep time: rotation fallback 1,144 of 11,420 team-games (10.02%);
ESPN athlete-id coverage 99.98%; 134,707 of 171,300 roster slots are named
players.

---

## 2. What is PROVISIONAL

Four adapters stand on sub-models whose own bake-offs adopted nothing. Every flag
below is written into `results/engine_v0/<tag>/run_meta.json` under
`adapter_flags`.

| flag | why |
|---|---|
| `provisional_event=True` | `possession_outcome` round 1 adopted nothing (0 of 11); round 2 was still running and has written no winner. Engine uses `reference_not_adopted_{first,cont}.pkl` behind `ENGINE_EVENT=reference`. |
| `provisional_clock=True` | `clock` adopted nothing (0 of 20 eligible). Engine uses the best-CRPS `reference_not_adopted_lgbm_quantile.pkl` behind `ENGINE_CLOCK=reference`. |
| `provisional_rotation=True` | `rotation` adopted nothing in either round. Engine uses R2 hierarchical Dirichlet + the fitted scheduler from `rotation_fit.json` behind `ENGINE_ROTATION=reference`, vectorised (two stated RNG divergences, model.md section 4.5). |
| `provisional_usage=True` | the adopted LightGBM choice arm persists no booster anywhere; engine runs `draw_player`'s U1 proportional path with the fitted (prior, m). |
| `provisional_foul_accrual=True` | the L3 vocabulary has no class for a non-shooting foul that awards no attempt; the engine draws one per possession at the measured rate 0.12335 and bumps TEAM fouls only. Team fouls and personal fouls are NOT reconciled. |
| `provisional_and_one=True` | and-ones come from a measured rate per made shot class, not a model. |
| `ENGINE_FG3=decision8` | Decision 8 selects LightGBM for FGA_3; `winner_FGA_3.joblib` predates it and still holds the flat `team_baseline` (shooter slope 0.0086). The engine fits the Decision-8 model into its own directory and never touches `fg_make`'s artifact. `ENGINE_FG3=artifact` runs the stale one. |

Not provisional: `fg_make` rim/jump2 (adopted winners loaded from their
joblibs), `rebound` and `free_throw` (adopted winners, refit by the engine
because neither trainer persists a model -- see model.md section 4).

---

## 3. What is UNFINISHED

1. **The F2 run produced NOTHING, and the gate scripts were never run on the
   engine.** A run over all 5,710 games x 50 seeds was launched at 16:22 ET with
   `--games-per-block 250 --seeds-per-block 25` (6,250 simulations, ~900k
   possessions per block) on a machine shared with other model workers. No block
   returned inside the session, and `run_engine.py` only writes after a block
   returns, so `results/engine_v0/F2_2025/` is EMPTY. A background process may
   still be running; **do not kill it if another worker owns it, and check
   whether it has since written before re-launching.**

   The fix is block size, not the engine: blocks must be small enough that
   several return early. Start with a gradeable smoke pass, confirm
   `eval_gates.py` reads it, then scale:

       # ~300 simulations per block, ~1 min each -- gradeable in minutes
       .venv/Scripts/python.exe scripts/run_engine.py --fold F2 --season 2025 \
           --seeds 5 --workers 16 --games-per-block 60 --seeds-per-block 5 \
           --tag F2_2025_s5

   Then resume with the full count:

       .venv/Scripts/python.exe scripts/run_engine.py --fold F2 --season 2025 \
           --seeds 200 --workers 16 --games-per-block 250 --seeds-per-block 25 \
           --tag F2_2025_s200

   **Expect this to take longer than the 2-hour target.** The MEASURED
   end-to-end rate at a 6,000-simulation batch on a contended machine is **862
   possessions/s/core**; 5,710 x 200 x ~145 = 1.66e8 possessions is then 1.93e5
   core-seconds = **2.7 hours on 20 cores**. The per-model benchmarks (clock
   4,900 rows/s/core and ~85% of model cost) would project ~3,500 poss/s/core on
   a free core, i.e. ~40 minutes, but no core was free during the build session
   so that number is unverified. Re-measure on a quiet box before quoting either.

2. **A seed-count study.** CLAUDE.md requires the minimum seed count be fixed by
   study before any ROI or gate number is read. Not done. Every gate number in
   `docs/tests/engine_v0_F2_2026-09-10.md` is therefore a provisional read at
   whatever seed count the run holds.

3. **The noise-floor / paired-seed run** (`--seed-offset 1000`) that the
   bake-off rule requires before any adoption. Not done.

4. **The lookup-table export -- now REQUIRED, not optional.** The measured 862
   poss/s/core misses the throughput target by ~35%, so the deliverable's escape
   clause applies. The engine runs batched predicts, not binned lookup tables.
   `ENGINE_CLOCK=reference_empirical` is wired and the empirical arm IS a lookup
   table (a (n_cells, 91) pmf over 7 binned dims of sizes 6 x 5 x 2 x 3 x 3 x 5 x
   2), which is the cheapest path; the quantile arm's own binned export, and the
   binning error the deliverable asks be reported, are **not measured**. Do not
   assume the error is small. The clock arm is ~85% of model cost, so binning it
   alone captures nearly all of the available speedup.

5. **`grade_market_props.py`** has not been run. `players.parquet` exists but the
   prop scorecard was out of time.

6. **`ast` has no model.** `players.parquet` writes 0; `run_meta.json` carries
   `ast_is_placeholder: true`.

7. **The change ledger** (`docs/models/change_ledger.md`) has no engine-v0 row
   yet; the PM owns that entry.

---

## 4. Commands to resume, in order

    # 1. (only if the prep inputs are stale or the season changes)
    .venv/Scripts/python.exe scripts/build_engine_inputs.py --fold F2 --season 2025

    # 2. the full run
    .venv/Scripts/python.exe scripts/run_engine.py --fold F2 --season 2025 --seeds 200 \
        --workers 16 --games-per-block 250 --seeds-per-block 25 --tag F2_2025_s200

    # 3. grade it
    .venv/Scripts/python.exe scripts/eval_gates.py        --results results/engine_v0/F2_2025_s200 --season 2025
    .venv/Scripts/python.exe scripts/grade_market_games.py --results results/engine_v0/F2_2025_s200 --season 2025
    .venv/Scripts/python.exe scripts/grade_market_props.py --results results/engine_v0/F2_2025_s200 --season 2025

    # 4. the noise floor the bake-off rule requires
    .venv/Scripts/python.exe scripts/run_engine.py --fold F2 --season 2025 --seeds 200 \
        --seed-offset 1000 --tag F2_2025_s200_seedoff1000

    # 5. tests
    .venv/Scripts/python.exe -m pytest tests/test_engine.py -q

Environment flags, all defaulting to the values recorded above:
`ENGINE_EVENT`, `ENGINE_CLOCK` (`reference` | `reference_empirical`),
`ENGINE_ROTATION`, `ENGINE_FG3` (`decision8` | `artifact`).
Pin `OMP_NUM_THREADS=1` (the runner does this in every worker).
`CBB_UNSEAL=1` is required to touch season 2026 and should stay unset.

---

## 5. The first thing to fix

The clock model. The engine's possession count runs about 4.5 too high and its
points per possession about 6% too low, and the two nearly cancel in the game
total -- which is exactly the offsetting-error pattern CLAUDE.md's multi-level
evidence rule exists to catch. The clock bake-off adopted nothing *because* no
arm passed the emergent G1 gate, and Decision 7 records this as the acknowledged
risk of making pace emergent. Every other gate reads downstream of it. Round 2 of
the clock bake-off was still running when this engine was built; re-wire it the
moment a winner exists, before reading any other gate as a model result.
