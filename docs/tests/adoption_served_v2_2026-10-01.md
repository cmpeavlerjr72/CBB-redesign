# Adoption: served stack v2 (the Decision 11 set), 2026-10-01

Adoption executor, overnight session, 2026-10-01 02:06-03:15 EDT (wall clock). This doc records the code change, the two parity proofs, how to reproduce the old stack, and what the adopted stack still fails.

## Decision

The PM adopted the Decision 11 set as ONE served stack, under the user's delegation of 2026-09-30.

Evidence: `docs/tests/engine_gates_F2_2025_s200_v3_COMB9GKD_full_2026-10-01.md`. The run was 5,710 games x 200 seeds, graded on verified truth with Decision 12 floors and paired against S0. What moved:

- G1 possessions mean: FAIL -> PASS.
- OREB%: FAIL -> PASS.
- Total bias and margin bias: PASS, unchanged.
- Calibration slope: 0.917 -> 0.948.
- Home/away corr: 0.117 -> 0.126.
- G2 cells inside: 3/9 -> 6/9.
- No verdict regresses.

| flag | new default | SERVED_V1 value | default defined at |
|---|---|---|---|
| `ENGINE_CLOCK` | `v5b_r6L2_glat_pmean` | `v5b_glat_pmean` | `src/cbb_sim/engine/adapters.py` (`Adapters.load`) |
| `ENGINE_SHOT_BLOCK` | `K2_Ocell` | `reference` | `shot_block.DEFAULT`, read in `loop.py` |
| `ENGINE_FOUL_JOINT` | `R9ao3` | `reference` | `foul_joint.DEFAULT`, read in `loop.py` |
| `ENGINE_SHARED_SHOOTING` | `G3` | `reference` | `shared_shooting.DEFAULT` |
| `ENGINE_CHANCE_TIME` | `KD` | `reference` | `chance_time.DEFAULT` |

Three other flags of the box run are unchanged: `ENGINE_EVENT=round2_s1`, `ENGINE_ROTATION=reference` and `ENGINE_FG3=decision8` were already the defaults.

Two smaller changes go with this:

- The four loop-level switches are now written into `run_meta.adapter_flags` (and therefore into the digest `flags` block), but only when they are not `reference`. A SERVED_V1 run's run_meta is unchanged.
- `clock_adapter_v3.ADOPTED_MODES` gains L2, which sets `provisional_clock` to False. This is a label only.

`scripts/run_aws_sweep.sh` changes:

- Its always-set `ENGINE_CLOCK` default is now the adopted L2. Without this, it would have forced the old clock under the four new switches, producing a hybrid stack.
- It gains `--parity-input-dir`.

## Reproduce S0 (SERVED_V1)

```
ENGINE_CLOCK=v5b_glat_pmean ENGINE_SHOT_BLOCK=reference ENGINE_FOUL_JOINT=reference ENGINE_SHARED_SHOOTING=reference ENGINE_CHANCE_TIME=reference
```

The same values are available in Python as `cbb_sim.engine.adapters.SERVED_V1` (a dict to merge into `os.environ` before the engine loads).

## Proof (a): new default == the box's adopted run. PASS

Reference: `results/engine_v0/v3full_COMB9GCTKD_s200_o0`, the box run named in `docs/ops/box_queue/pm_1.done.md`. The comparison used 60 games x seeds 0-4 on `data/processed/models/engine_v3`, with exact equality on every column of `games` and `players`:

| run | vs box rows | vs pre-change explicit |
|---|---|---|
| pre-change code, five flags explicit, S0 event block | games 300/300, players 4,956/4,956 BIT-IDENTICAL | n/a |
| post-change code, NO flags set, S0 event block | games 300/300, players 4,956/4,956 BIT-IDENTICAL | BIT-IDENTICAL |
| post-change `run_engine.py`, no flags (`adopt_post_default_60x5`) | n/a (local event block, see below) | BIT-IDENTICAL to `adopt_pre_explicit5_60x5` (games and players) |

**Finding, input-level and pre-existing (not caused by this adoption).** The box S0 stack serves the event-model team block from the inputs-v3 replay (`engine_v3/event_block_F2_2025.npz`, sha `228794fc`), mounted by the S0 overlay over `data/processed/models/engine/event_round2_s1_F2_2025/team_block.npz`. The local file at that path is the 2026-09-10 block (sha `1fe68771`).

- The two blocks differ on 265 of 5,710 games, which the engine reads from `adapters.ENGINE_DIR` and not from `--input-dir`.
- A plain local `run_engine.py --input-dir data/processed/models/engine_v3` therefore differs from the box on those games: 4 of the first 600 (401756997, 401751915, 401743949, 401743885), with or without the five flags.
- The same holds for the v7 reference, which was generated with the local block.
- The S0-block rows above were produced in-process by redirecting that single `np.load` path. No shared file was touched.
- Whether the served backtest default should carry the v3 event block is an inputs decision for the PM. It is not part of this adoption.

## Proof (b): SERVED_V1 reproduces v7 and v6. PASS

| reference | inputs | result |
|---|---|---|
| `docs/ops/parity_reference_windows_v7.json` | `engine_v3`, 60 x 5 | PASS, bit-identical digest `34cd58dd...1736` |
| `docs/ops/parity_reference_windows_v6.json` | default inputs (v2), 60 x 5 | PASS, bit-identical digest `0d4ddccc...029f` |

## New parity reference: v8

`docs/ops/parity_reference_windows_v8.json` uses the same generator as v7: `run_engine.py --seeds 5 --max-games 60 --input-dir data/processed/models/engine_v3`, with no ENGINE_* variables set, and `scripts/digest_engine_run.py --emit`. Its sha256 is `0192a3a8426070d73acda954d4a2c1b62feb5980ea2bbbe6690738aae622523b`.

The box gate for the adopted stack:

```
run_aws_sweep.sh --parity only --parity-ref docs/ops/parity_reference_windows_v8.json --parity-input-dir data/processed/models/engine_v3
```

This needs `engine_inputs_v3` pulled, and no docker `-e ENGINE_*` overrides. The v6 gate now needs the SERVED_V1 values passed into the container.

## Artifacts on a fresh clone

| flag | files | tracked? |
|---|---|---|
| clock L2 | `data/processed/models/clock/r6_L2/` (14 files, 48 MB) | gitignored; on HF `model_artifacts` (14/14 files listed on HF, checked ~02:21 EDT) |
| K2_Ocell | `data/processed/models/engine/shot_block_K2_Ocell_F2_2025.npz` | tracked |
| R9ao3 | `possession_outcome/round9/lut_ao_AO3_F2.npz` and `ao_team_prior_v1.parquet` | tracked |
| R9ao3 | R8b base `possession_outcome/round7/lut_acc_A2_F2.npz` and `lut_trip_T2c_F2.npz` | gitignored; on HF `model_artifacts` (verified) |
| G3 | `shared_shooting/params_v1.json` | tracked |
| KD | `chance_time/F2/lut_v3.npz` | tracked |

```
python scripts/hf_sync_data.py pull --dirs model_artifacts --only 'clock/r6_L2/**' 'possession_outcome/round7/lut_*'
python scripts/hf_sync_data.py pull --dirs engine_inputs engine_inputs_v3
```

## Ops path

- `run_engine.py`, `run_engine_live.py` and `chain_daily_v3.py` read the engine defaults, so they now serve the adopted stack.
- **The lane F fold-2 replay smoke FAILS on the new defaults.** The command was `chain_daily_v3.py --replay-season 2025 --slate-date 2025-02-11 --seeds 4 --root <scratch>`. The sim stage stops with `FileNotFoundError: data/processed/models/engine/shot_block_K2_Ocell_.npz`.
- Cause: the shot-block LUT exists only for the `F2_2025` backtest slate, and live inputs carry no slate tag. With `ENGINE_SHOT_BLOCK=reference`, all four stages (sim, publish, grade, bias_clv) complete. So the other four adopted switches do run on the live path.
- Needed: a per-slate live shot-block LUT build in the daily inputs stage. Until it exists, the daily chain cannot serve the adopted stack. Running it with `ENGINE_SHOT_BLOCK=reference` gives an UNGATED hybrid, and must be labelled as such if used.

## Tests

- Full suite after the code change (`pytest tests`): 670 passed, 1 skipped. No test asserted the old defaults, so no existing test needed changing.
- New `tests/test_served_defaults_v2.py` pins the five defaults and checks that SERVED_V1 switches every loop-level sub-model off.
- `tests/test_daily_chain_v3.py` (lane F) passes. Its fixtures do not reach the shot-block LUT lookup, so it does not catch the replay failure above.

## What the adopted stack still FAILS (from the gate doc)

- G5 total SD ratio 0.925.
- G5 home/away corr 0.126 vs actual 0.228.
- G9 calibration slope 0.948.
- OT rate 0.0305 vs 0.0557.
- G6 neutral.
- G8 lines.
- G1 by month: 4/5 months inside.
- G2: 6/9 cells inside.
- PIT p 0.002.
