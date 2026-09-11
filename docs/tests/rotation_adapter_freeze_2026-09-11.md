# Rotation round-9 engine adapter + the Decision-10 freeze on K1 and Z1

**2026-09-11, 17:00-17:20Z (13:00-13:20 ET).** Lane: rotation engine adapter for
rounds 7/8/9 (three rounds overdue, `docs/models/rotation/experiments.md` 21.15)
and Decision 10 on K1 and Z1. Full write-up and every disclosure:
experiments.md **section 22**. This file is the test record.

**Outcome in one line.** The adapter is written and default-off, both
bit-identity checks PASS, it is identical to the offline sampler on 21,506 real
waves, and the freeze PASSES for both K1 and Z1 -- and **nothing was adopted and
no default moved**, because Z1 still fails three of round 9's six conditions
offline by 2 to 58 floors.

## 1. Bit-identity: the served engine must not have moved

| # | check | how | result |
|---|---|---|---|
| 1 | served R2 path (`ENGINE_ROTATION=reference`) | 60 games x 5 seeds, `ENGINE_CLOCK=v5b_glat_pmean` pinned explicitly (the served clock, read out of `results/engine_v0/smoke60x5_default_v5b/run_meta.json`, not assumed), all other flags copied from that run_meta; `scripts/digest_engine_run.py --compare` | **PASS** -- sha256 `492300a7fd1e6dc388a3c47b822f61e04da6501720ef762bde0afe7ba15451a1`, equal to the stored smoke |
| 2 | existing round-6 K1 path | the same 30-game x 3-seed `round6`/`K1` chunk simulated under a copy of the tree at `HEAD` and under the working tree, both `games`+`players` frames rounded to 6 dp, sorted and hashed | **PASS** -- `fd3ec1575dbf840d18da3728a05fe91e665b1ada79616e24043a65b36329f0ee` on both, 12,896 possessions each |
| 3 | round-6 K1 path, full scale (incidental) | `K1_live` / `K1_frozen` below vs the published 15.13 table | **PASS** -- 16.5337 / 16.5367, 0.0361 / 0.0399, 71.7092 / 71.6818, 8.7390 / 8.6491, every digit reproduced |

Check 1 is the one 21.15's scope owed the served default; check 2 is the one it
owed the only non-default rotation arm anyone runs. Check 3 was not planned and
is the strongest of the three: a 500-game x 5-seed round-6 run is byte-for-byte
what it was before the round-9 code existed.

## 2. Parity against the offline sampler

`ENGINE_ROT9_AUDIT=1` (default-off) records every wave the round-9 exit block
resolves. On **150 games x 3 seeds** (450 team-game sims, 64,896 possessions)
that is **21,506 waves**; `k_out` was then recomputed from the same states, the
same uniforms and the same fitted tables by the OFFLINE scalar block in
`rotation_v8.run_wave8` that `rotation_v9.run_wave9` delegates to.

**Floor A for an identity check is 0.**

| n_starters on floor | waves | adapter starter share | offline starter share | \|delta\| |
|---:|---:|---:|---:|---:|
| 0 | 201 | 0.000000 | 0.000000 | 0.000000 (UNDERPOWERED) |
| 1 | 1,457 | 0.224374 | 0.224374 | 0.000000 |
| 2 | 4,588 | 0.350178 | 0.350178 | 0.000000 |
| 3 | 7,267 | 0.502056 | 0.502056 | 0.000000 |
| 4 | 5,728 | 0.728523 | 0.728523 | 0.000000 |
| 5 | 2,265 | 1.000000 | 1.000000 | 0.000000 |

**0 of 21,506 `k_out` draws differ; max |share delta| 0.000000.** Artifact:
`data/processed/models/rotation/round9_adapter_parity_2026-09-11.json`.

The `n_starters = 0` cell is labelled underpowered (201 waves) and carries no
information either way -- with no starter on the floor `k_out` is 0 by
construction on both sides.

## 3. Unit tests

`pytest tests/test_rotation_round9.py -q` -> **8 passed** (1.4 s):

1. the served default is still `reference` and the freeze/audit hooks are off;
2. `ENGINE_ROTATION_ARM` selection, including a round-6 arm name falling back
   to `Z1` rather than crashing;
3. **`Round9Batch` is dispatched BEFORE `Round6Batch`** -- it subclasses it, so
   the reverse order would silently turn every round-9 run into a round-6 run;
4. the stated RNG divergence: `2S + 5` in round 9, `2S + 4` still in round 6,
   `u_x` at `2S + 4`;
5. round 6's K1 entry arithmetic appears verbatim in both;
6. **20,000 randomised states**: the vectorised exit block and the offline
   scalar block agree exactly, including the all-zero-row and degenerate
   (`hi < lo`) support branches;
7. the audit hook is default-off;
8. the freeze zeroes margin and BOTH foul counts, for the rotation only.

`pytest tests/test_engine.py -q` -> **20 passed** (241 s), unchanged.

## 4. The Decision-10 freeze

14.9's design, unchanged: the fixed 500-game subset of F2 2025 (sorted by
`game_id`, every 11th row, first 500), 5 seeds, paired streams, 3 workers per
arm, `ENGINE_EVENT=round2_s1` / `ENGINE_FG_MAKE=round3_shooter_S_C_s1` /
`ENGINE_CLOCK=reference` pinned and recorded in every `run_meta.json`.
`ENGINE_ROTATION_FREEZE=1` holds margin and the personal/team foul counts at 0
for the rotation model only.

| run | seeds | margin SD | per-game margin SD | home/away corr | possessions | total | minutes MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 live | 5 | 16.5337 | 12.1434 | 0.0361 | 71.7092 | 149.367 | 8.7390 |
| K1 `FREEZE=1` | 5 | 16.5367 | 12.2718 | 0.0399 | 71.6818 | 149.516 | 8.6491 |
| Z1 live | 5 | 16.3696 | 11.9869 | 0.0404 | 71.7064 | 149.088 | 9.2922 |
| Z1 `FREEZE=1` | 5 | 16.2635 | 12.0815 | 0.0567 | 71.7038 | 149.162 | 9.3511 |

| arm | margin SD ratio | corr delta | possessions delta | floor | verdict |
|---|---:|---:|---:|---|---|
| **K1** live/frozen | **0.9998** | -0.0038 | +0.027 | ratio in [0.95, 1.05]; possessions within G1 mean +/- 1.0 | **PASS** |
| **Z1** live/frozen | **1.0065** | -0.0163 | +0.003 | same | **PASS** |
| 15.13 K1 5 seeds | 0.9998 | -0.0038 | +0.027 | -- | reproduced exactly |
| 15.13 K1 25 seeds | 0.9934 | +0.0099 | -0.016 | -- | quoted, not re-run |
| 13.12 W4 5 seeds | 0.9948 | -0.0132 | -0.087 | -- | quoted |

14.9 pins the margin-SD and possession floors numerically and states the
correlation floor only as "inside the G1/G2 tolerances". No numeric correlation
band is pre-registered, so Z1's -0.0163 is reported against the -0.0132 /
-0.0038 / +0.0099 the three prior passing pairs produced, **not** against a
threshold invented after seeing it.

**25 seeds were not run.** Four 500-game x 25-seed runs cost about 90 minutes on
the 3 workers per arm this lane was capped at, against a 13:35 ET stop the
5-seed set cleared at 13:16.

## 5. Verdict

* **Condition 6 is now RUN and PASSED for K1 and for Z1.** It remains unrun for
  X1, Y1 and Z2.
* **Nothing is adopted.** Z1 fails conditions 1, 4 and 5 by 2 to 58 floors
  (21.12). A closed-loop pass removes an obstacle and supplies no evidence for
  the arm.
* `ENGINE_ROTATION=reference` (R2) remains the served default. `round9` ships
  default-off. **No default, no `DEFAULTS` entry and no served artifact was
  touched.**
* Independent cross-check: on the same 500 games the engine's own player minutes
  put K1 at 8.7390 and Z1 at 9.2922 -- a 0.553-minute gap in the same direction
  and about the same size as the offline 8.8622 vs 9.3132 (0.451). The engine
  and the offline sampler agree that round 9's exit rule is worse than round 6's.

## 6. Process

Six engine workers (3 per arm, two chains), plus one extra single-worker process
for about 100 seconds during the bit-identity smoke. No process started by
another lane was killed, signalled or restarted; no existing results directory or
data file was overwritten (all four runs and the smoke wrote new directories).
`scripts/run_rot6_closed_loop.py` was copied, not edited.
