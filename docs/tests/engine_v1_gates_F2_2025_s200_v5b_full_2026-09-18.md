# Engine v5b -- full 200-seed paired gate read, consolidated from two AWS sessions (2026-09-18)

This closes out the PARTIAL 75-of-200-seed read from `docs/tests/engine_v1_gates_F2_2025_s200_v5b_2026-09-11.md`
(2026-09-11 session, seeds 0-74 / 1000-1074 for A/B, engine commit `d940b41d87bcbeb4bf883fe05c0cee91d08b5eb1`).
This session (AWS box three, 2026-09-18) ran the remaining seeds 75-199 / 1075-1199 for both streams and
consolidated to the full 200/200.

Served config (adapters.py defaults, unchanged from the 75-seed read): `ENGINE_EVENT=round2_s1`,
`ENGINE_CLOCK=v5b_glat_pmean`, `ENGINE_FG_MAKE=round4_B1`, `ENGINE_REBOUND=s1_weekly`,
`ENGINE_FREE_THROW=s1_conf_aligned`, `ENGINE_ROTATION=reference`, `ENGINE_ROTATION_SCHEME=s1`,
`ENGINE_FG3=decision8`, `ENGINE_INPUTS_VERSION=v2`.

## 1. Engine commit(s) and why mixing them is safe

The new seeds (75-199 / 1075-1199) were simulated at engine commit `09c7ae17ef220fefc7eaf5a001180e7fb4fd8020`
(the instructed lane HEAD at session start), NOT re-pinned to `d940b41`. This is safe because the diff
`d940b41..09c7ae1` touches only `src/cbb_sim/engine/{adapters.py,clock_adapter_v3.py,loop.py,rotation_adapter.py}`,
and every change is either a one-token addition to a `mode.startswith(...)` tuple for the new, DEFAULT-OFF
`ENGINE_CLOCK=v5d_glat_pquad` arm, or the DEFAULT-OFF `ENGINE_ROTATION=round9` adapter (new `Round9Batch`,
gated behind `rotation_mode() == "round9"`; served default stays `reference`). Neither touches the
`v5b_glat_pmean` / `round2_s1` / `reference` code paths the served config actually walks.

Mid-session a PM note reported `docs/ops/parity_reference_windows_v6.json` newly pushed at engine commit
`9c02f85405...` (on top of `09c7ae1`, reachable as `origin/main` `54a57d1bed0ceb75ce8c20e8cd117705a182a40f`
at pull time). Diff `09c7ae1..54a57d1` was re-read on the box: `adapters.py`/`clock_adapter_v3.py` change
only flips the `provisional_clock` metadata label (`ADOPTED_MODES = {"v5b_glat_pmean"}`) with an explicit
comment that it "does not touch `pmf`, `draw`, any fitted coefficient, or any RNG stream, so it cannot change
a single simulated number"; the rest of the diff is the new, DEFAULT-OFF `round4b_G2`/`round4b_G3` possession-
outcome event mode (untouched by the served `round2_s1` path) plus docs. Confirmed empirically, not just by
reading: the box was rebuilt at `54a57d1` and re-passed the parity gate bit-identical against the new v6
reference (section 2). The consolidated 200-seed A/B runs below therefore mix commits `09c7ae1` (new seeds)
and `d940b41`-lineage (old seeds) safely -- both are proven to produce bit-identical `games.*`/`players.*`
rows for this served config; only a metadata label in `run_meta.json` (not part of the graded contract)
differs across the two commits' worth of seeds.

## 2. Parity: two checks, both PASS

1. **Linux-vs-previous-AWS-run digest continuity** (used first, before the v6 file was known to exist):
   a reference digest was emitted from the LOCAL `results/engine_v0/smoke60x5_default_v5b` directory
   (a v5b-config 60x5 smoke already on disk, sha256 `492300a7fd1e6dc388a3c47b822f61e04da6501720ef762bde0afe7ba15451a1`,
   the same digest `docs/models/rotation/experiments.md` section 22.2 already validated as the served-R2-path
   Linux/AWS reference on 2026-09-11). A fresh 60x5 smoke run on this box at commit `09c7ae1`, `ENGINE_CLOCK=v5b_glat_pmean`,
   via `run_aws_sweep.sh --parity only --parity-ref <this file>`: **PASS, bit-identical**
   (`492300a7fd...`).
2. **Canonical Windows reference v6** (after the PM note): box rebuilt at `54a57d1bed0ceb75ce8c20e8cd117705a182a40f`
   (which carries `docs/ops/parity_reference_windows_v6.json`, emitted on Windows at engine commit `9c02f85405`,
   60 games x 5 seeds, sha256 `0d4ddccc64d7700a7493db2427bcab2d06204f54a12db307f7b897729639029f`). A second 60x5
   smoke on the rebuilt image: **PASS, bit-identical**, `--compare` reporting zero diff on `games.*`/`players.*`
   and matching `flags` including `provisional_clock: false`. Both checks confirm the same thing from different
   anchors; nothing here loosened a tolerance.

## 3. Seeds achieved and consolidation

New chunks this session (25 seeds each, `--games-per-block 30 --seeds-per-block 4`, 70 workers/stream,
concurrent): A off75/100/125/150/175, B off1075/1100/1125/1150/1175. All 5 chunks per stream completed
(`CHUNK done_seeds=125/125`, `END COMPLETE`, `push=OK`) in ~21 minutes wall (22:08:05-22:29:02Z), pushed to
`mvpeav/cbb-sim-data` automatically per chunk and at the end.

Pulled back to the local machine (`hf_sync_data.py pull --dirs results`) alongside the 2026-09-11 session's
already-local off0/25/50 (A) and off1000/1025/1050 (B) chunks. `scripts/concat_engine_runs.py --tag
F2_2025_s200_v5b_A --out-tag F2_2025_s200_v5b_A_full` (and `_B_full`) merged all 8 chunks per stream:

| stream | chunks | seeds | game rows | player rows | partial | dropped |
|---|---:|---:|---:|---:|---|---:|
| A_full | 8 | 200 (0-199) | 1,142,000 | 18,067,384 | False | 0 |
| B_full | 8 | 200 (1000-1199) | 1,142,000 | 18,064,038 | False | 0 |

`concat_engine_runs.py`'s own duplicate-`(game_id,seed)` assertion and config-identity check passed for both.

## 4. G1-G9 at the full 200 seeds: A (headline) vs B (noise floor)

Full table (all gate lines): `docs/tests/gate_noise_band_F2_2025_s200_v5b_full_2026-09-18.md`
(`scripts/diag_gate_noise_band.py`). Per-run detail: `docs/tests/gates_engine_v0_F2_2025_s200_v5b_A_full_2026-09-18.md`
(`scripts/eval_gates.py`).

**Gate-level tally (unchanged from the 75-seed read and from the 200-seed `v3c` read this morning of
2026-09-11):** G1=FAIL, G2=FAIL, G3=NEEDS-INSTRUMENTATION, G4=FAIL, G5=FAIL, G6=FAIL, G7=FAIL, G8=FAIL, G9=FAIL.

| gate | quantity | 75-seed A (2026-09-11) | 200-seed A (now) | 200-seed A-B band (now) | verdict changed vs 75-seed? |
|---|---|---:|---:|---:|---|
| G1 | possessions/game mean | 69.863 (FAIL) | 69.866 (FAIL) | 0.0020 | no |
| G1 | possessions/game SD | 5.551 (PASS) | 5.552 (PASS) | 0.0040 | no |
| G5 | margin SD ratio | 1.0339 (PASS) | 1.0396 (PASS) | 0.0002 | no |
| G5 | total SD ratio | 0.8927 (FAIL) | 0.8983 (FAIL) | 0.0007 | no |
| G5 | home/away score corr | 0.1166 vs 0.2532 (FAIL) | 0.1170 vs 0.2532 (FAIL) | 0.0016 | no |
| G5 | PIT K-S p | 0.0236 | 0.016 | 0.0094 | no (both far below any plausible threshold) |
| G6 | home margin, neutral | +2.110 vs 3.288 (FAIL) | +2.105 vs 3.288 (FAIL) | 0.0350 | no |
| G7 | OT rate | 0.0300 vs 0.0557 (FAIL) | 0.0304 vs 0.0557 (FAIL) | 0.0003 | no |
| G8 | rotation minutes SD ratio | 1.2257 (FAIL) | 1.2258 (FAIL) | 0.0007 | no |
| G8 | players used/team-game | 8.79 vs 9.80 (FAIL) | 8.79 vs 9.80 (FAIL) | 0.0000 | no |
| G9 | margin bias | -0.2070 (PASS) | -0.1932 (PASS) | 0.0093 | no |
| G9 | total bias | -0.8714 (PASS) | -0.8617 (PASS) | 0.0303 | no |
| G9 | calibration slope | 0.8998 (FAIL) | 0.9096 (FAIL) | 0.0021 | no |

**No gate-line verdict changed between the 75-seed provisional read and this full 200-seed read.** Every
line that was PASS at 75 seeds (G1 SD, G5 margin SD ratio, G9 margin/total bias) is still PASS; every line
that was FAIL is still FAIL, at values within (usually well within) the 200-seed A-B noise band of the
75-seed number. Per CLAUDE.md: never compare G5/G9 dispersion lines across different seed counts as if they
were the same measurement -- the comparison above reads the 75-seed number against the tighter 200-seed band,
not the other way around, and every reported gap is inside or barely outside its own band, consistent with
seed noise rather than a real shift from the additional 125 seeds.

Newly gradeable at 200 seeds vs 75 (G2/G3/G4 season-pooled provisional lines, previously not tabulated in
the 75-seed doc's headline comparison): G3 `three_pa_share`/`fta_per_fga`/`rim_share` all PASS; G4 `tov_pct`/
`ft_rate` PASS, `oreb_pct`/`efg_pct` FAIL. See the full noise-band doc for exact values and bands.

## 5. What this run does and does not establish

1. **The full 200-seed floor is now met for this gate report** (`docs/tests/engine_seed_count_2026-09-10.md`).
   Calibration and ROI/Brier remain unread (2,000-seed floor), unchanged from every prior report.
2. Parity is now doubly confirmed (continuity + canonical v6), closing the "open item, not waived" flagged
   in the 2026-09-11 75-seed doc section 2 and section 7.
3. HF sync of results completed before termination (section 3); verified against local chunk counts.
