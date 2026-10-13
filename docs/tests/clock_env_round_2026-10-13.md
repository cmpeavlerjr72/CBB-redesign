# Clock round 9: scoring-environment and mismatch response of possession duration, results 2026-10-13

Pre-registration: `docs/models/clock/experiments.md` section 41 (PROMOTED 2026-10-13, PM Fable 5.1; 41.0 rulings 1-6;
41.A the draft verbatim; 41.B pre-fit implementation notes and control-only findings, committed with the grader before
any fit). Results section: `experiments.md` section 42. **REFUTED by the literal rule. NO ARM ADOPTED. No served default
changed. Model freeze ON. Parity v10 bit-identical with every new flag off. No closed loop run (it is for an offline
winner only).**

## 1. Outcome

| step | result |
|---|---|
| Q1 wiring pre-check | **PASS**: the design, the live day-1 path and the F2 engine inputs give identical rating columns (max abs diff 0.0); no rebuild (section 2) |
| offline, F2 selects / F1 confirms | every arm improves the full-set deviance (-0.023 to -0.025, about 6 floors) and moves the responsiveness lines on F2: env ratio 0.16 -> 1.30, z_off-quintile slope 0.20 -> 0.89, team-game x_off ratio 0.20 -> 0.90. **But the deviance win does not survive on E0's positive-mass rows** (+0.0006 / +0.0015 / -0.0001, floor 0.0017-0.0018) |
| decision (literal, 41.A s5 + 41.0 item 2) | **REFUTED: no arm wins.** Every arm fails the dual-deviance gate, rule (b) on the gap line, and the two absolute ranges of rule (c) that E0 itself fails (41.B). E2 and E3 also land outside [0.7, 1.3] on env (1.317 / 1.303) |
| reported-only readings (select nothing) | R1 (the ranges read relative to E0): still no winner (dual-deviance gate, gap line). R2 (slope differences): the env model slope moves toward the actual by 186 floors (E1); the gap slope moves by 11 floors (E2, E3) |
| closed loop, dispersion gates, full-size read | **not run**: section 6 is for an offline winner only |
| parity v10 (flags off) | PASS, bit-identical digest, 60 x 5 |

## 2. Q1 wiring pre-check (41.0 item 3)

`scripts/diag_clock_env_q1_wiring_v1.py` -> `results/clock_env/q1_wiring.json`.

| check | rows | result |
|---|---:|---|
| design columns vs a fresh `join_as_of` on the batch files (`as_of_date == game date`), 2022-2025 | 35,920 team-games | max abs diff 0.0 on all four columns; 0 NaN fills |
| live code path `rating_site_block` vs design, batch ratings, slates 2024-11-04 / 2025-01-16 / 2025-03-04 | 142 / 114 / 82 team-games | max abs diff 0.0 |
| same, fed the live as-of builder's own output (`build_own_ratings_asof_v1`) | same | 1.6e-16 / 0.0 / 0.0 |
| F2 engine inputs team_static (what the closed loop serves) vs design | 10,220 team-games | max abs diff 0.0 |
| 2027 day-1 snapshot (`ratings_asof/2026-11-02`) | 365 teams | batch schema plus created_at; as_of 2026-11-02 only; created_at 2026-10-13 13:00 UTC < first tip 2026-11-02 05:00 UTC; team mean of off_c and def_c is 0 (1e-16), as in the batch day-1 snapshot; hyperparameters (lambda 5/5, prior weight 0.8/0.8) and prior carry identical |

Only difference: the live ridge re-centres over the CBBD 2027 team list (365 teams); historical runs re-centre over the
teams with played games. The team set is part of the definition (`docs/ops/own_ratings_daily_2026-09-30.md`). The 2027
day-1 spread is wider (off_c SD 6.44 vs 5.80 on 2025 day 1). z uses the fold's fixed training SD, so the wider spread
passes through unchanged.

## 3. What ran

- `55d669c` promotion (section 41, ledger row). `cd6f941` grader `scripts/grade_clock_env_v1.py`, Q1 script, law
  module `src/cbb_sim/models/clock_env.py`, trainer `scripts/train_clock_env_v1.py`, default-off keys
  `ENGINE_CLOCK=v5b_r8K2_env{E1,E2,E3}_glat_pmean` (+ `v3c_r8K2_env*_P3_s1`), tests, and 41.B. All of this was committed BEFORE
  any fit; the grader had only been smoke-run on E0, E0s1 and K2M.
- Fits ran 13:20-13:23 EDT: E1/E2/E3 x F1/F2, plus the three F2 `--seed 1` refits (25-35 s each). Coefficients per SD of the
  feature (F2 first / last refit): E1 c = -0.0186 / -0.0193; E2 c = -0.0188 / -0.0195, d = -0.0062 / -0.0060; F1 E1
  -0.0171 / -0.0178. k SD 0.019-0.023 (a ~2% duration scale per SD). Fold norms F2: sd_off 6.83, gap mean 8.58, SD 6.97.
- Grade 13:23-13:24 (`results/clock_env/offline_grade.json`, `offline_tables.md`).
- Honesty asserted in code: no 2025-26 rows; every refit's rows dated before its cut; 17,960 of 17,960 games have the
  ratings' as-of cut (00:00 ET of the game date) before tipoff_utc.
- Reseed proof: E0 vs E0s1 and E1 / E2 / E3 vs their seed-1 refits, max abs pmf diff **0.0** (30,000 rows x every refit,
  marginal and conditional laws). Reseed floor 0; the binding floor is 2 x SD of the paired game bootstrap (200 draws, seed 12345).
- Artifacts `data/processed/models/clock/r9_env_{E1,E2,E3}/{F1,F2,F2_seed1}` (49 MB each; gitignored), pushed to HF
  `model_artifacts/clock/r9_env_*` (`scripts/ops_hf_push_paths_v1.py`).

## 4. Offline blind table (written under codes; key at the end of `results/clock_env/offline_tables.md`)

Rows: test-season regulation possessions of clock-complete games (F2 257,897 rows / 1,916 games; F1 226,677 / 1,672).
Cells: value (delta vs E0; floor). For the ratio lines the delta is in |ratio - 1|.

**Fold 2 (selects)**

| arm | deviance | deviance, E0 > 0 rows (0.121% excluded) | d1 dev cond. | env ratio | gap ratio | offence-tempo q slope (a) | z_off q (b) | gap q (c) | def q (d) | game-prior q | elasticity ratio | count gap /team-game |
|---|---|---|---:|---|---|---|---|---|---|---:|---|---|
| E0 | 6.7436 | 6.6851 | 6.7079 | 0.163 | -0.635 | 0.405 | 0.197 | 0.619 | 0.342 | 0.814 | 0.724 | +0.452 |
| E0s1 | 6.7436 (0; 0) | 6.6851 (0; 0) | 6.7079 | 0.163 (0; 0) | -0.635 (0; 0) | 0.405 (0; 0) | 0.197 | 0.619 | 0.342 | 0.814 | 0.724 | +0.452 |
| E1 | 6.7207 (-0.0229; 0.0039) | 6.6856 (**+0.0006**; 0.0017) | 6.6816 | 1.299 (-0.538; 0.473) | -0.736 (+0.101; 1.735) | 0.405 (-0.000; 0.011) | 0.894 (-0.698; 0.086) | 0.536 (+0.083; 0.188) | 1.043 (-0.615; 0.215) | 0.826 | 0.742 (+0.018; 0.012) | +0.475 (+0.023) |
| E2 | 6.7201 (-0.0236; 0.0043) | 6.6865 (**+0.0015**; 0.0018) | 6.6807 | 1.317 (-0.520; 0.479) | 1.785 (-0.850; 11.84) | 0.405 (+0.000; 0.012) | 0.894 (-0.697; 0.087) | 1.202 (-0.179; 1.211) | 1.022 (-0.636; 0.200) | 0.826 | 0.740 (+0.017; 0.013) | +0.348 (-0.104) |
| E3 | 6.7191 (-0.0245; 0.0042) | 6.6850 (**-0.0001**; 0.0017) | 6.6796 | 1.303 (-0.534; 0.474) | 1.752 (-0.883; 11.33) | 0.406 (+0.001; 0.012) | 0.889 (-0.692; 0.088) | 1.194 (-0.187; 1.200) | 1.019 (-0.639; 0.198) | 0.827 | 0.741 (+0.018; 0.013) | +0.350 (-0.102) |
| K2M (ref) | 6.6971 (-0.0465; 0.0054) | 6.6757 (-0.0094; 0.0021) | 6.6556 | 0.332 | -0.732 | 0.888 | 0.207 | 0.859 | 0.232 | 0.986 | 0.980 | +0.518 |

**Fold 1 (confirms)**

| arm | deviance | deviance, E0 > 0 rows (0.133% excluded) | env ratio | gap ratio | offence-tempo q (a) | z_off q (b) | gap q (c) | def q (d) | elasticity ratio | count gap |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| E0 | 6.7525 | 6.6880 | 0.004 | -0.841 | 0.472 | 0.193 | -0.753 | 0.421 | 0.808 | +0.219 |
| E1 | 6.7217 (-0.0307; 0.0053) | 6.6880 (-0.0000; 0.0016) | 1.779 (-0.217; 1.475) | -0.898 (+0.057; 1.181) | 0.481 | 1.014 | -0.811 | 1.129 | 0.822 | +0.236 (+0.017) |
| E2 | 6.7230 (-0.0295; 0.0053) | 6.6882 (+0.0002; 0.0017) | 1.814 (-0.182; 1.505) | 0.822 (-1.663; 7.92) | 0.480 | 1.031 | 0.415 | 1.154 | 0.820 | +0.134 (-0.085) |
| E3 | 6.7246 (-0.0278; 0.0051) | 6.6887 (+0.0007; 0.0018) | 1.796 (-0.199; 1.490) | 0.803 (-1.644; 8.06) | 0.481 | 1.026 | 0.401 | 1.150 | 0.821 | +0.136 (-0.084) |
| K2M (ref) | 6.6962 (-0.0563; 0.0066) | 6.6757 (-0.0123; 0.0026) | 0.252 | -0.879 | 1.011 | 0.194 | -0.684 | 0.282 | 1.090 | -0.145 |

Underlying slopes, in implied possessions per team per unit (F2 actual: env +0.0795, gap +0.0228; F1: +0.0498, +0.0312).
F2: E0 env +0.0130 / gap -0.0145; E1 +0.1033 / -0.0168; E2 +0.1047 / +0.0407; E3 +0.1036 / +0.0399. F1: E0 +0.0002 /
-0.0263; E1 +0.0886 / -0.0280; E2 +0.0903 / +0.0257. Joint (env, gap) ratios, F2: E0 0.165 / -0.672; E1 1.304 / -0.830;
E2 1.316 / 1.807. Team-game line (implied possessions on the offence's own x_off = off_c + opponent def_c): actual slope
+0.255 (F2) / +0.222 (F1); ratio E0 0.195 / 0.183, E1 0.903 / 0.977, E2 0.917 / 0.994, E3 0.913 / 0.991.

## 5. Decision rule applied literally (F2)

Rule line (41.A section 5 with 41.0 item 2): "Offline, an arm WINS if on fold 2 (a) deviance beats `E0` by more than its
floor, (b) the `env` and `gap` possession-slope ratios each move toward 1 by more than their floors and land in
[0.7, 1.3], (c) the quintile slopes (a)-(d) above hold, (d) fold 1 does not reverse the sign of (a) or (b)." + "An arm
whose deviance win on the full set does not survive on the positive-mass subset (beyond that line's own floor) does NOT
win."

| check | E1 | E2 | E3 |
|---|---|---|---|
| (a) deviance beats E0 beyond floor | PASS (-0.0229 vs 0.0039) | PASS | PASS |
| (a') dual deviance: win survives on E0 > 0 rows | **FAIL** (+0.0006; 0.0017) | **FAIL** (+0.0015; 0.0018) | **FAIL** (-0.0001; 0.0017) |
| (b) env ratio toward 1 beyond floor | PASS (-0.538; 0.473) | PASS (-0.520; 0.479) | PASS (-0.534; 0.474) |
| (b) env ratio in [0.7, 1.3] | PASS (1.299) | **FAIL** (1.317) | **FAIL** (1.303) |
| (b) gap ratio toward 1 beyond floor | **FAIL** (+0.101; 1.735) | **FAIL** (-0.850; 11.84) | **FAIL** (-0.883; 11.33) |
| (b) gap ratio in [0.7, 1.3] | **FAIL** (-0.736) | **FAIL** (1.785) | **FAIL** (1.752) |
| (c)(a) offence-tempo slope in [0.85, 1.15] | **FAIL** (0.405; E0 0.405) | **FAIL** | **FAIL** |
| (c)(a) not below E0 beyond floor | PASS | PASS | PASS |
| (c)(b)-(d) not further from 1 than E0 beyond floor | PASS (z 0.89, gap q 0.54, def q 1.04) | PASS | PASS |
| (c) elasticity ratio in [0.9, 1.1] | **FAIL** (0.742; E0 0.724) | **FAIL** | **FAIL** |
| (d) F1 sign: deviance / env / gap | yes / yes / **no** | yes / yes / yes | yes / yes / yes |
| veto count gap (> +0.10) | +0.023 pass | -0.104 pass | -0.102 pass |
| veto start type (> 0.20 s) | +0.014 pass | +0.033 pass | +0.044 pass |
| veto month Nov-Mar (> 0.25 s) | +0.022 pass | +0.010 pass | +0.009 pass |
| **WINS** | **no** | **no** | **no** |

**Outcome: REFUTED (no arm wins).** No closed loop (section 6 is "offline winner only"), so the dispersion gates of
41.0 item 1, the seed-count check and the full-size read were not run.

Reported-only readings (41.B; they select nothing): R1 (the two absolute ranges read relative to E0) still gives no
winner, because the dual-deviance gate and the gap line fail. R2 (model slope differences vs E0, direction toward the
actual): env +0.0903 (E1, floor 0.0005), +0.0917 (E2, 0.0021), +0.0906 (E3, 0.0021), all toward the actual. Gap: -0.0023
(E1, floor 0.0047, no move); +0.0552 (E2, 0.0049) and +0.0544 (E3, 0.0048), toward the actual and past it (actual +0.023).

## 6. Why the deviance win is not real (the 41.0 item 2 gate did its job)

E0 (the served K2) assigns zero mass to 311 F2 rows (0.121%; F1 301, 0.133%), and these hit the 1e-12 clip. A continuous-CDF AFT
scale with k != 1 spreads every integer bin over its neighbours, so it fills those zero bins, and that alone is the full-set
gain. On the 99.88% of rows where E0 has positive mass, E1 / E2 / E3 change deviance by +0.0006 / +0.0015 / -0.0001
(F2) and -0.0000 / +0.0002 / +0.0007 (F1), against floors of 0.0016-0.0018. That is no change. The env term moves the per-row law by
about 2% per SD of z_off, which the per-possession likelihood cannot detect. The aggregate responsiveness lines can (env
slope +0.090 implied possessions per team per unit, 186 floors on F2). The endgame round found the same mechanism
(GT2, `docs/tests/endgame_round_2026-10-13.md` section 4).

## 7. Multi-level evidence (reported)

**Per team (offence teams with >= 300 test rows).** Share whose arm-minus-E0 mean implied duration has the sign of
actual-minus-E0: F2 E1 0.599, E2 0.588, E3 0.585 (359 teams; K2M 0.682); F1 0.545 / 0.548 / 0.553 (347; K2M 0.666).
Share whose |model - actual| team mean shrinks: F2 0.549 / 0.535 / 0.529; F1 0.507 / 0.496 / 0.504.

**Per game.** Count calibration slope (actual on implied possessions): F2 E0 0.952, E1 0.947, E2 0.949, E3 0.950
(floor 0.017-0.019); F1 0.936 / 0.933 / 0.941 / 0.941.

**Per possession type (start type).** See the segment table. The worst worsening vs E0 is +0.014 s (E1), +0.033 (E2), +0.044 (E3).

**First half vs second half.** The env response appears in the first half (H1 env ratio E0 0.13 -> E1 1.20, z_off-q
slope 0.18 -> 0.76), so it is not an end-of-game state effect.

#### F2 segments (reported; U = UNDERPOWERED: < 150 games for ratios, < 3,000 rows for slopes)

| segment | rows | games | actual s/poss | gap s E0 / E1 / E2 / E3 | env ratio E0 / E1 / E2 / E3 | gap ratio E0 / E1 / E2 / E3 | z_off-q slope E0 / E1 / E2 / E3 |
|---|---:|---:|---:|---|---|---|---|
| month=1 | 66,224 | 496 | 17.96 | -0.212 / -0.220 / -0.176 / -0.176 | 0.29 / 1.26 / 1.26 / 1.25 | -2.35 / -2.70 / 6.22 / 5.95 | 0.17 / 0.90 / 0.90 / 0.89 |
| month=2 | 55,845 | 419 | 17.99 | -0.172 / -0.194 / -0.153 / -0.153 | 0.29 / 1.34 / 1.33 / 1.32 | -0.74 / -1.08 / 1.59 / 1.53 | 0.15 / 0.75 / 0.75 / 0.75 |
| month=3 | 40,263 | 301 | 17.94 | -0.122 / -0.125 / -0.081 / -0.083 | 0.33 / 1.57 / 1.54 / 1.52 | 0.01 / 0.16 / -0.55 / -0.54 | 0.40 / 1.34 / 1.35 / 1.34 |
| month=4 | 973 | 7 | 17.26 | 0.460 / 0.348 / 0.409 / 0.410 | U | U | U |
| month=11 | 52,473 | 384 | 17.54 | -0.077 / -0.068 / -0.047 / -0.047 | -0.07 / 1.08 / 1.17 / 1.16 | -1.94 / -1.78 / 0.92 / 0.91 | 0.21 / 0.77 / 0.77 / 0.77 |
| month=12 | 42,119 | 309 | 17.60 | 0.035 / 0.034 / 0.045 / 0.044 | -0.23 / 1.03 / 1.12 / 1.10 | -0.36 / -0.30 / 0.88 / 0.88 | 0.21 / 0.85 / 0.85 / 0.84 |
| site=neutral | 39,117 | 292 | 17.90 | -0.166 / -0.187 / -0.134 / -0.135 | 0.12 / 1.52 / 1.60 / 1.58 | 0.61 / 0.46 / -0.16 / -0.15 | 0.27 / 0.82 / 0.79 / 0.79 |
| site=off_away | 109,374 | 1,624 | 17.88 | -0.136 / -0.112 / -0.082 / -0.083 | 0.21 / 1.25 / 1.26 / 1.25 | 0.53 / 1.25 / 0.01 / 0.05 | 0.16 / 1.00 / 0.96 / 0.95 |
| site=off_home | 109,406 | 1,624 | 17.73 | -0.085 / -0.116 / -0.086 / -0.086 | 0.11 / 1.14 / 1.15 / 1.15 | 0.03 / 0.32 / 0.78 / 0.78 | 0.21 / 0.80 / 0.83 / 0.83 |
| prev_end=DREB | 91,001 | 1,916 | 14.54 | -0.004 / -0.008 / 0.022 / 0.031 | 0.44 / 1.16 / 1.17 / 1.34 | -0.20 / -0.05 / 1.24 / 1.96 | 0.11 / 0.67 / 0.68 / 0.81 |
| prev_end=TOV | 44,011 | 1,916 | 14.95 | 0.026 / 0.029 / 0.059 / 0.070 | 0.40 / 0.99 / 1.00 / 0.98 | -0.04 / 0.04 / 0.83 / 1.19 | 0.13 / 0.77 / 0.79 / 0.77 |
| prev_end=made_FG | 90,577 | 1,916 | 22.21 | -0.200 / -0.214 / -0.176 / -0.189 | 0.32 / 1.24 / 1.26 / 1.17 | -0.13 / -0.77 / 3.49 / 1.66 | 0.17 / 1.01 / 0.99 / 0.93 |
| prev_end=made_FT | 27,513 | 1,916 | 18.94 | -0.417 / -0.420 / -0.385 / -0.393 | 0.42 / 1.27 / 1.28 / 0.99 | 1.73 / 2.17 / 0.45 / 1.15 | 0.17 / 1.16 / 1.19 / 0.84 |
| prev_end=other | 964 | 645 | 1.68 | -0.301 / -0.297 / -0.296 / -0.285 | 0.84 / 0.05 / -0.11 / -1.64 | 2.76 / 2.82 / 1.99 / -1.54 | U |
| prev_end=period_start | 3,831 | 1,916 | 20.87 | -0.408 / -0.407 / -0.367 / -0.380 | 0.85 / 3.58 / 3.62 / 3.69 | -1.75 / -1.33 / 7.01 / 5.12 | 0.25 / 1.00 / 0.89 / 0.94 |
| tier=both_power | 47,934 | 357 | 17.87 | -0.135 / -0.184 / -0.142 / -0.142 | 0.45 / 1.98 / 1.98 / 1.97 | 0.46 / 0.50 / -0.15 / -0.14 | 0.25 / 0.85 / 0.80 / 0.80 |
| tier=neither_power | 188,504 | 1,402 | 17.84 | -0.144 / -0.140 / -0.096 / -0.096 | 0.13 / 1.36 / 1.37 / 1.35 | -3.36 / -4.67 / 14.44 / 14.09 | 0.17 / 0.90 / 0.91 / 0.90 |
| tier=one_power | 21,459 | 157 | 17.55 | 0.139 / 0.137 / 0.057 / 0.056 | 0.14 / 0.57 / 0.63 / 0.62 | -0.19 / -0.15 / 0.44 / 0.44 | 0.26 / 0.77 / 0.77 / 0.77 |
| half=H1 | 128,028 | 1,916 | 17.95 | -0.046 / -0.056 / -0.021 / -0.021 | 0.13 / 1.20 / 1.21 / 1.21 | 0.02 / -0.05 / 1.57 / 1.58 | 0.18 / 0.76 / 0.76 / 0.76 |
| half=H2 | 129,869 | 1,916 | 17.69 | -0.191 / -0.193 / -0.161 / -0.162 | 0.20 / 1.42 / 1.44 / 1.41 | -2.93 / -3.16 / 2.60 / 2.41 | 0.23 / 1.10 / 1.09 / 1.08 |
| teamq=0 | 45,415 | 554 | 18.21 | -0.519 / -0.281 / -0.263 / -0.265 | 1.08 / 3.62 / 3.82 / 3.77 | 0.60 / 1.64 / 0.91 / 0.93 | 0.38 / 1.17 / 1.12 / 1.12 |
| teamq=1 | 52,753 | 655 | 17.70 | -0.026 / 0.054 / 0.105 / 0.104 | -0.22 / 0.47 / 0.50 / 0.49 | 0.96 / 1.90 / 1.12 / 1.13 | 0.09 / 0.50 / 0.49 / 0.48 |
| teamq=2 | 53,236 | 671 | 17.89 | -0.130 / -0.152 / -0.100 / -0.100 | 0.41 / 1.66 / 1.65 / 1.63 | 0.50 / 0.69 / -0.12 / -0.10 | 0.17 / 0.74 / 0.73 / 0.72 |
| teamq=3 | 52,913 | 658 | 17.77 | -0.066 / -0.170 / -0.137 / -0.137 | 0.34 / 2.45 / 2.39 / 2.36 | 0.49 / 0.91 / 1.57 / 1.56 | 0.26 / 0.88 / 0.88 / 0.87 |
| teamq=4 | 53,580 | 541 | 17.58 | 0.087 / -0.096 / -0.086 / -0.086 | 0.39 / 1.22 / 1.20 / 1.19 | 0.10 / 0.45 / 0.68 / 0.68 | 0.39 / 0.99 / 1.03 / 1.02 |

#### F1 segments (reported; U = UNDERPOWERED: < 150 games for ratios, < 3,000 rows for slopes)

| segment | rows | games | actual s/poss | gap s E0 / E1 / E2 / E3 | env ratio E0 / E1 / E2 / E3 | gap ratio E0 / E1 / E2 / E3 | z_off-q slope E0 / E1 / E2 / E3 |
|---|---:|---:|---:|---|---|---|---|
| month=1 | 55,204 | 409 | 17.77 | -0.092 / -0.091 / -0.056 / -0.055 | -0.45 / 1.58 / 1.73 / 1.71 | -0.45 / -0.19 / 0.95 / 0.90 | 0.04 / 1.03 / 1.07 / 1.06 |
| month=2 | 57,954 | 432 | 17.88 | -0.151 / -0.162 / -0.132 / -0.132 | 0.25 / 5.80 / 5.84 / 5.79 | -2.15 / -2.40 / 2.21 / 2.13 | 0.27 / 1.29 / 1.29 / 1.29 |
| month=3 | 34,459 | 257 | 17.88 | -0.144 / -0.153 / -0.110 / -0.110 | 0.31 / 1.08 / 1.10 / 1.09 | -12.89 / -12.75 / -1.79 / -2.02 | 0.33 / 1.02 / 1.05 / 1.05 |
| month=11 | 40,778 | 294 | 17.28 | 0.124 / 0.130 / 0.143 / 0.141 | -0.38 / 0.98 / 0.99 / 0.97 | -0.62 / -0.76 / 0.31 / 0.31 | 0.23 / 0.79 / 0.80 / 0.80 |
| month=12 | 38,282 | 280 | 17.54 | 0.020 / 0.012 / 0.019 / 0.019 | 0.32 / 2.56 / 2.49 / 2.46 | -2.32 / -3.38 / 1.65 / 1.71 | 0.19 / 0.92 / 0.91 / 0.91 |
| site=neutral | 30,622 | 227 | 17.78 | -0.157 / -0.199 / -0.162 / -0.162 | 0.59 / 1.21 / 1.22 / 1.21 | 4.48 / 4.85 / 0.23 / 0.30 | 0.36 / 0.93 / 0.93 / 0.93 |
| site=off_away | 98,025 | 1,445 | 17.84 | -0.138 / -0.113 / -0.088 / -0.088 | -0.66 / 2.88 / 2.97 / 2.93 | 4.48 / 7.00 / 0.69 / 0.84 | 0.13 / 1.26 / 1.25 / 1.23 |
| site=off_home | 98,030 | 1,445 | 17.51 | 0.055 / 0.033 / 0.058 / 0.057 | -0.14 / 1.44 / 1.48 / 1.47 | -0.06 / 0.21 / 0.69 / 0.69 | 0.15 / 0.79 / 0.82 / 0.82 |
| prev_end=DREB | 80,862 | 1,672 | 14.48 | -0.026 / -0.029 / -0.006 / -0.002 | 0.37 / 1.26 / 1.28 / 1.52 | -0.34 / -0.25 / 0.45 / 0.70 | 0.10 / 0.72 / 0.74 / 0.91 |
| prev_end=TOV | 38,336 | 1,672 | 14.93 | 0.123 / 0.121 / 0.145 / 0.157 | 0.30 / 2.69 / 2.74 / 2.60 | -0.14 / 0.04 / 1.07 / 1.74 | 0.06 / 0.93 / 1.00 / 0.99 |
| prev_end=made_FG | 79,421 | 1,672 | 22.01 | -0.092 / -0.102 / -0.072 / -0.081 | 0.36 / 1.50 / 1.52 / 1.41 | -0.24 / -0.47 / 1.08 / 0.41 | 0.18 / 1.21 / 1.22 / 1.13 |
| prev_end=made_FT | 23,975 | 1,672 | 18.69 | -0.312 / -0.309 / -0.281 / -0.284 | 0.52 / 1.56 / 1.58 / 1.17 | 1.02 / 1.22 / 0.50 / 0.66 | 0.03 / 0.91 / 0.88 / 0.55 |
| prev_end=other | 739 | 523 | 1.09 | -0.041 / -0.037 / -0.037 / -0.028 | U | U | U |
| prev_end=period_start | 3,344 | 1,672 | 20.66 | -0.204 / -0.207 / -0.174 / -0.171 | 0.07 / 0.60 / 0.61 / 0.63 | 0.06 / 0.12 / 0.46 / 0.53 | -0.26 / 0.92 / 1.06 / 1.12 |
| tier=both_power | 33,860 | 251 | 17.78 | -0.007 / -0.046 / -0.011 / -0.011 | 0.52 / 1.66 / 1.72 / 1.71 | -0.74 / -0.64 / 0.34 / 0.32 | 0.25 / 0.79 / 0.78 / 0.78 |
| tier=neither_power | 172,398 | 1,273 | 17.71 | -0.097 / -0.093 / -0.061 / -0.061 | -0.12 / 1.98 / 2.02 / 2.00 | 8.84 / 9.00 / -2.98 / -2.78 | 0.17 / 1.17 / 1.20 / 1.19 |
| tier=one_power | 20,419 | 148 | 17.38 | 0.200 / 0.176 / 0.142 / 0.139 | U | U | 0.31 / 0.76 / 0.74 / 0.74 |
| half=H1 | 112,272 | 1,672 | 17.86 | 0.025 / 0.017 / 0.045 / 0.045 | 0.06 / 1.39 / 1.41 / 1.41 | -0.61 / -0.71 / 1.95 / 1.97 | 0.18 / 0.80 / 0.81 / 0.81 |
| half=H2 | 114,405 | 1,672 | 17.52 | -0.137 / -0.138 / -0.113 / -0.114 | -0.07 / 2.57 / 2.62 / 2.57 | -0.95 / -0.98 / 0.29 / 0.25 | 0.24 / 1.37 / 1.40 / 1.38 |
| teamq=0 | 37,251 | 459 | 17.99 | -0.328 / -0.084 / -0.085 / -0.086 | 0.10 / 1.04 / 1.14 / 1.12 | 1.29 / 3.60 / 2.11 / 2.14 | 0.35 / 1.08 / 0.99 / 0.98 |
| teamq=1 | 44,730 | 574 | 17.83 | -0.198 / -0.141 / -0.098 / -0.099 | 1.38 / -3.22 / -3.27 / -3.21 | 0.54 / 1.39 / 0.61 / 0.62 | 0.47 / 1.61 / 1.54 / 1.53 |
| teamq=2 | 46,799 | 594 | 17.56 | 0.039 / 0.051 / 0.094 / 0.093 | 0.45 / -1.40 / -1.41 / -1.39 | 0.49 / 0.74 / 0.36 / 0.37 | 0.30 / 1.02 / 0.98 / 0.98 |
| teamq=3 | 50,948 | 636 | 17.66 | -0.051 / -0.135 / -0.105 / -0.105 | 0.05 / 1.30 / 1.34 / 1.33 | -0.49 / 0.45 / 1.38 / 1.37 | -0.00 / 0.97 / 1.02 / 1.01 |
| teamq=4 | 46,949 | 512 | 17.48 | 0.191 / 0.001 / 0.013 / 0.013 | 0.28 / 1.11 / 1.11 / 1.10 | 0.16 / 0.43 / 0.59 / 0.59 | 0.17 / 0.63 / 0.72 / 0.72 |

Segment ratio cells are single-season, overlapping and noisy. The gap ratio is the noisiest because the offline actual gap slope is
small (+0.02-0.03). These cells are reported, not read as signal.

## 8. Findings for the PM (not proposals)

1. **The scoring-environment response is real, and an env AFT term reproduces it offline.** The env ratio goes 0.16 -> 1.30 (F2)
   and 0.00 -> 1.78 (F1, floor 1.5); the team-game x_off ratio 0.20 -> 0.90 and 0.18 -> 0.98; the z_off-quintile slope 0.20 -> 0.89.
   No count-gap, start-type or month veto fires, and the tempo lines lose nothing. The game-level env ratio overshoots
   (1.3-1.8) while the team-game line lands near 1. The gap line is unresolvable offline at 1,916 / 1,672
   clock-complete games (|ratio - 1| floors 1.7-11.8).
2. **The registered rule cannot select such an arm.** Three reasons were identified before the fit (41.B) and one by the
   fit: (i) the offence-tempo slope range [0.85, 1.15] (the served K2 is at 0.405); (ii) the elasticity range [0.9, 1.1]
   (K2 is at 0.724); (iii) the offline gap floor; (iv) the dual-deviance gate, because a 2% per-SD scale has no
   per-possession likelihood signal once zero-mass smoothing is removed. A deviance primary is the wrong instrument for a
   responsiveness term of this size; the responsiveness lines are the ones with power.
3. The served K2's offline tempo responsiveness is 0.405 (offence quintile) and 0.724 (elasticity). K2M (the tempo law
   on K2) is at 0.888 / 0.980 offline. Any re-registration that keeps those ranges needs a K2M-based control.
4. Q2 (41.0 item 4) was not reached (no winning pace arm, no loop), so the total-on-env remainder is unmeasured here.

## 9. Not done, and why

- Closed loop (500 x 25 paired, four floor draws), the 41.0 item 1 dispersion gates, the seed-count check, the latent sigma
  refit, parity on a clean src for a candidate and the full-size 5,710 x 200 read were **not run**, because there is no offline winner (41.A
  section 6 is "offline winner only").
- Engine smoke of the new keys was not run. A K-family key runs only through its game-latent wrapper, whose sigma is
  refitted for a closed-loop candidate only (the bare served `v3c_r8K2_P3_s1` key fails the same way in the engine).
  `tests/test_clock_env.py` covers the adapter contract (rating columns in TEAM_COLS, keys default-off).
- Nothing was adopted and no served default changed; the freeze is untouched. Not touched: data/processed/ingest, injuries, results/daily, the
  chain scripts, the served inputs, PROJECT_STATUS.md and HANDOFF.md.
