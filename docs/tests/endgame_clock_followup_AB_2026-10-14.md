# Clock/endgame follow-up round, blocks A and B (clock experiments.md s46): offline results, 2026-10-14

Pre-registration: `docs/models/clock/experiments.md` section 46 (PM Fable 5.1, 2026-10-14, rulings 1-10 binding).
Results section: 47. Opus engine worker.

**Both blocks are REFUTED offline by the literal s46 4.1 rule.**
- No offline winner, so no closed-loop screen, no full-size box read and no AWS spend.
- NOTHING ADOPTED. No served default changed and no engine flag added. Nothing the engine imports changed under `src/`.
- Freeze ON.

## 1. Outcome

| block | arms | offline verdict (F2 period 2 selects, F1 confirms) | screen | full size |
|---|---|---|---|---|
| A: cell term on the served K2 window law | CA0 (control), CA0s1 (floor), CA1 ratio (m = 100), CA2 cell-AFT (m = 100), CA3 smooth AFT | **REFUTED, no winner.** CA1 beats CA0 on both primaries: dev_pos -0.1479 (floor 0.0049, 30.2 floors); cal 2.294 -> 0.403 s (floor 0.037, 50.6 floors). It holds the tempo hard gate (0.365 -> 0.403, floor 0.048) and keeps both F1 signs. It FAILS the team-prior trailer slope gate: 0.025 -> -0.120 (floor 0.084); F1 0.107 -> -0.035 (floor 0.072). CA2 and CA3 fail dev_pos (+0.080 and +0.120, i.e. worse) and the trailer gate. CA3 also fails slope_team_lead and slope_lead | NOT RUN (no offline winner) | NOT RUN |
| B: final-possession conversion | FP0 (control), FP0s1 (floor), FP1 class mix, FP2 = FP1 + make intercept, FP3 = FP2 on LF2 (offline identical to FP2) | **REFUTED, no winner.** Cell: trailing 1-3, <= 30 s, period 2 (1,521 rows, 989 shot rows; slopes UNDERPOWERED). FP1 mll -0.0007 (floor 0.0010, 0.65 floors). FP2 make_dev -0.0089 (floor 0.0117, 0.76 floors). No arm clears either primary's floor. The reported-only "changed-law" reading also has no winner | NOT RUN | NOT RUN |

Literal rule (s46 4.1): "An arm WINS its block if it beats its control by more than 1 floor on the block primary (A:
positive-mass-row deviance and cell calibration; B: `FPS` log loss and make-logit deviance), every responsiveness gate
holds, and the dual-deviance gate holds. ... No winner: block REFUTED." Step 2 (the screen) is for offline winners only.

## 2. What ran (all local; no box)

| step | what | commit |
|---|---|---|
| grader | `scripts/grade_endgame_clock_v2.py` (offline A/B, select-m, loop screen, full read) + 6 unit tests. Committed BEFORE any s46 fit output existed | `bf83e6e` |
| fits | law module `src/cbb_sim/models/endgame_v2.py` (not imported by the engine; no flag), trainer `scripts/train_endgame_clock_v2.py`, tables script `scripts/grade_endgame_clock_v2_tables_v1.py`, 3 law tests | `8eddd85` |
| results | this doc, clock experiments.md s47, ledger row | results commit |

Commands (PYTHONIOENCODING=utf-8, `.venv/Scripts/python.exe`):

```
# block A: fold-1 grid (ruling 7), then frozen-m fits on both folds
train_endgame_clock_v2.py --block A --fold F1 --arm CA0 | --arm CA0s1 | --arm CA1 --m {20,50,100} | --arm CA2 --m {20,50,100} | --arm CA3 --final
train_endgame_clock_v2.py --block A --fold F2 --arm CA0 | --arm CA0s1 | --arm CA3 --final
grade_endgame_clock_v2.py select-m                      # -> results/endgame_clock_v2/grade/m_selection.json (frozen)
train_endgame_clock_v2.py --block A --fold {F1,F2} --arm {CA1,CA2} --m 100 --final
train_endgame_clock_v2.py --block A --fold F2 --arm CA1 --m 100 --seed 1        # reseed proof
grade_endgame_clock_v2.py offline --block A --arms CA0 CA0s1 CA1 CA2 CA3
# block B
train_endgame_clock_v2.py --block B --fold {F1,F2} --arm {FP0,FP1,FP2,FP3,FP0s1}
train_endgame_clock_v2.py --block B --fold {F1,F2} --arm {FP1,FP2} --seed 1     # reseed proof
grade_endgame_clock_v2.py offline --block B --arms FP0 FP0s1 FP1 FP2 FP3
grade_endgame_clock_v2_tables_v1.py > results/endgame_clock_v2/tables_offline.md
```

Outputs:
- `results/endgame_clock_v2/{offline,offline_reseed,grade}/` and `tables_offline.md` (gitignored; HF `results/`).
- Fit artifacts `data/processed/models/late_game/endgame_clock_v2/{F1,F2}/` (gitignored; HF `model_artifacts`).

| artifact | sha256 |
|---|---|
| F2/ca1_law.npz | 1d911d78489d3a26590f65a5558744a440782c1011d0a5a7a030a4f34fe6f472 |
| F2/ca2_law.json | 8d3d7e383a3e4404631866938caa84a4796d9b8e4da49be899b0411518fae99c |
| F2/ca3_law.json | cc64f97b5abffdb404ff10aa2e2de84df04242cb5a0cfa90da43bd10470a6df5 |
| F2/fp1_law.json | ea79d4c09a335042476295149dd419bb36ca5624c04a2b532b804b8de21036bd |
| F2/fp2_law.json | 0d481833ec69abbb72ff8ee7c37bf247aa94bbc27b4435f830300711b4a85384 |
| F1/ca1_law.npz | bd7aa715642a3c1a099c5dae78710df84a85071d537b713aa88d994d46fb97a2 |
| F1/ca2_law.json | e22f204f738307372d584e6ed0bea2c383a13e543f72dd16dccd221e975cb6f8 |
| F1/ca3_law.json | a284d61b49d91d986383a13a70cbadce6e7342777161d2711a0675289ee11189 |
| F1/fp1_law.json | 73d2e40fceeaebf119387a86e77aff09627e9155667029dbcdfb6af352814b04 |
| F1/fp2_law.json | 5c9552d779cbd11724ef28b593038ca48aa45acc025c26e60668fe9bbc4bdf78 |

Compute and checks:
- Local only; each fit takes seconds.
- AWS not used: cost $0, no instance launched, nothing to terminate.
- Parity v10 (`run_parity_smoke_v1.py`, served defaults, 60 x 5): PASS, bit-identical.
- Suite: 856 passed, 1 skipped (847 + 9 new).

## 3. Implementation choices (fixed in the grader / trainer before the fits; disclosed)

1. **Deviance.** The late-game window's first-chance duration deviance (`grade_endgame_v1` block C `dev`). This is the quantity the s46 finding (2) zero-mass share (0.19% / 0.28%) was measured on; it is not round 9's possession-level deviance. `dev_pos` is the same deviance on rows where CA0 gives the observed outcome positive mass (> 1e-12).
   - F2 period 2: 142 rows (0.19%) excluded, carrying 0.106 of CA0's `dev`.
   - F1: 196 rows (0.28%), carrying 0.156.
2. **cal.** Mean over cells with >= 200 test rows (the s46 3 cell power line). All 45 cells are powered in F2 and F1 period 2. `cal45` (all populated cells) is reported and is equal there.
3. **Gates.**
   - Tempo (hard), z_off, defence-z and the team-prior lead/trail slopes: "not below the control's by more than its floor" (s46 3, "not below control beyond floor").
   - Band slopes lead/trail: |slope - 1| not worse beyond floor (late_game s15 (a)).
   - The game-level tempo elasticity is a loop line with no offline counterpart.
4. **Floor** = max(reseed difference, 2 x paired game-block bootstrap SE); 200 draws, seed 12345. The binding floor is the bootstrap term everywhere. Reseed differences:
   - CA0 vs CA0s1 (K2 refit under seed 1): 0.0 on both folds.
   - CA1 seed 0 vs seed 1: 6e-8. This is float32 rounding of the cached served training pmfs between the two runs, not a seed effect.
   - FP1 and FP2 seed 0 vs seed 1: 0.0. FP2 vs FP3: 0.0.
5. **Shrink grid (ruling 7).** Chosen on F1 period-2 `dev_pos`: among the m values within one F1 floor of the best, the largest m. Frozen in `m_selection.json` before any F2 CA1/CA2 fit.
   - CA1: 6.46306 / 6.46318 / 6.46376 for m = 20 / 50 / 100 (floor 0.0055) -> **m = 100**.
   - CA2: 6.68069 / 6.68039 / 6.68029 (floor 0.0123) -> **m = 100**.
6. **CA1.**
   - 8 duration bins: [0,3), [3,6), [6,9), [9,13), [13,18), [18,24), [24,31), [31,90].
   - Observed/expected per bin. A horn-censored row is spread over the bins at or above its censoring time, in proportion to the served law.
   - Separate ratio tables for the marginal and the end-class-conditional law, each fitted against the law it multiplies.
   - Shrink hierarchy: cell -> time bucket -> 1.
7. **CA2 / CA3.**
   - Estimator: round-8 M2 Poisson pseudo-likelihood on uncensored training window rows, with the K2 conditional finest-cell fixed effects (`clock_env.cond_cell_key`).
   - Tempo tercile edges come from the fold's training window rows; K2 uses its full training design. The fixed effects are only a grouping, so this changes only which rows share an intercept.
   - The Newton step is minimum-norm, so a column the fixed effects absorb gets 0.
   - CA2's bucket and cell betas are shrunk as registered (m = 100).
   - The served row law is scaled, not refit (s46: "served pmf scaled by k_cell").
8. **Block B.**
   - Rows: event-window FIRST chances with start_clock <= 30 s and offence margin in [-6, 0]. The law is fitted on trailing 1-3 only (ruling 9).
   - Served make probability: the fg_make round4_B1 S1 schedule (F2 `fg_make/round4/B1`; F1 `fold1_v1/fg/B1`, the fold-1 served-equivalent). `max_train_date < game date` is asserted per refit.
   - Each FPS shot is matched to its fg_make design row: same game, period, offence and shot class, shot clock within [chance end, start], latest such shot. Matched: 3,703 of 3,711 F2 test shots and 3,326 of 3,333 F1.
   - FP2's intercept is pooled over the three shot classes, one per half, shrunk by n / (n + 50).
9. **Correction before the block-B grade (disclosed).**
   - The trainer first wrote FP0s1 as FP1's law refitted under seed 1. Read against FP0, that would have put FP1's own effect into the floor.
   - Before `offline --block B` first ran, FP0s1 was redefined as the served predictions under the seed-1 label. The served artifacts are fixed objects. This is the role CA0s1 plays in block A.
   - The block's own fits were refitted under `--seed 1` separately and are identical (item 4).
   - No block-B grade existed before the correction.
10. **FP3.** LF2 changes only the foul-accrual law, so FP3's class and make predictions equal FP2's offline (max abs diff 0.0). Ruling 3 makes FP3 a loop composition read beside FP2. With no block-B winner it has no loop read.
11. **Loop-mode R2/R4 (not exercised).** For blocks A and B the grader reads R2 and R4 as no-regression vetoes; round 10's "H1 must move toward" reading belongs to env arms. This is stated in the grader docstring, written before any fit.

## 4. Block A tables

### Shrink grid (ruling 7, fold 1 only)

Rule: lowest F1 period-2 dev_pos; among m within one F1 floor of the best, the largest m.

| arm | m = 20 dev_pos | m = 50 | m = 100 | within one F1 floor | chosen m |
|---|---:|---:|---:|---|---:|
| CA1 | 6.46306 | 6.46318 | 6.46376 | CA1_m20, CA1_m50, CA1_m100 | 100 |
| CA2 | 6.68069 | 6.68039 | 6.68029 | CA2_m20, CA2_m50, CA2_m100 | 100 |

### Reseed proof (max abs difference)

| pair | max abs diff |
|---|---:|
| A_F2_CA0_vs_CA0s1 | 0 |
| A_F1_CA0_vs_CA0s1 | 0 |
| A_F2_CA1_seed0_vs_seed1 | 5.96e-08 |
| B_F1_FP1_seed0_vs_seed1 | 0 |
| B_F1_FP2_seed0_vs_seed1 | 0 |
| B_F1_FP2_vs_FP3 | 0 |
| B_F2_FP1_seed0_vs_seed1 | 0 |
| B_F2_FP2_seed0_vs_seed1 | 0 |
| B_F2_FP2_vs_FP3 | 0 |

### Block A (offline)

#### Block A, F2 period 2 (selection): 74,164 rows, 5,110 games; control zero-mass rows excluded from dev_pos: 142 (0.19%, carrying 0.1058 of the control's `dev`); powered cells 45 of 45

| line | role | actual / target | CA0 (control) | CA0s1 (floor arm) | CA1 | CA2 | CA3 |
|---|---|---|---:|---:|---:|---:|---:|
| dev_pos | PRIMARY | lower | 6.5967 | 6.5967 (+0.0000; fl 0.0000) | 6.4488 (-0.1479; fl 0.0049) PASS (beats by > floor) | 6.6763 (+0.0796; fl 0.0110) FAIL (inside floor or worse) | 6.7163 (+0.1196; fl 0.0202) FAIL (inside floor or worse) |
| cal | PRIMARY | lower | 2.2936 | 2.2936 (+0.0000; fl 0.0000) | 0.4030 (-1.8907; fl 0.0374) PASS (beats by > floor) | 1.9686 (-0.3250; fl 0.0730) PASS (beats by > floor) | 2.1801 (-0.1135; fl 0.0364) PASS (beats by > floor) |
| dev | reported (i) | lower | 6.6899 | 6.6899 (+0.0000; fl 0.0000) | 6.5423 (-0.1476; fl 0.0049) REPORTED | 6.6985 (+0.0086; fl 0.0166) REPORTED | 6.7633 (+0.0734; fl 0.0225) REPORTED |
| cal45 | reported | lower | 2.2936 | 2.2936 (+0.0000; fl 0.0000) | 0.4030 (-1.8907; fl 0.0374) REPORTED | 1.9686 (-0.3250; fl 0.0730) REPORTED | 2.1801 (-0.1135; fl 0.0364) REPORTED |
| slope_tempo | HARD GATE (not below) | 1 | 0.3649 | 0.3649 (+0.0000; fl 0.0000) | 0.4025 (+0.0377; fl 0.0480) hold | 0.4339 (+0.0690; fl 0.0816) hold | 0.3771 (+0.0123; fl 0.0243) hold |
| slope_zoff | gate (not below) | 1 | 0.6879 | 0.6879 (+0.0000; fl 0.0000) | 2.0952 (+1.4073; fl 0.4402) hold | 2.9372 (+2.2493; fl 0.7324) hold | 1.0897 (+0.4018; fl 0.1857) hold |
| slope_defz | gate (not below) | 1 | 0.7858 | 0.7858 (+0.0000; fl 0.0000) | 3.0368 (+2.2510; fl 2.0202) hold | 4.3871 (+3.6012; fl 3.2310) hold | 1.5427 (+0.7569; fl 0.9089) hold |
| slope_team_lead | gate (not below) | 1 | 0.5700 | 0.5700 (+0.0000; fl 0.0000) | 0.9238 (+0.3538; fl 0.3062) hold | 0.7711 (+0.2010; fl 0.1996) hold | 0.4609 (-0.1091; fl 0.0869) FAIL |
| slope_team_trail | gate (not below) | 1 | 0.0248 | 0.0248 (+0.0000; fl 0.0000) | -0.1201 (-0.1449; fl 0.0842) FAIL | -0.1151 (-0.1399; fl 0.0863) FAIL | -0.2568 (-0.2816; fl 0.1378) FAIL |
| slope_lead | gate (|s-1| not worse) | 1 | 0.4796 | 0.4796 (+0.0000; fl 0.0000) | 1.0380 (+0.5584; fl 0.0556) hold | 0.4496 (-0.0300; fl 0.0343) hold | 0.4424 (-0.0372; fl 0.0139) FAIL |
| slope_trail | gate (|s-1| not worse) | 1 | 0.5349 | 0.5349 (+0.0000; fl 0.0000) | 0.9029 (+0.3680; fl 0.0327) hold | 0.9710 (+0.4361; fl 0.0368) hold | 0.5795 (+0.0446; fl 0.0210) hold |
| dev_end | reported | lower | 6.6277 | 6.6277 (+0.0000; fl 0.0000) | 6.5076 (-0.1201; fl 0.0043) REPORTED | 6.6358 (+0.0080; fl 0.0260) REPORTED | 6.6818 (+0.0541; fl 0.0293) REPORTED |
| pergame_abs_err | reported (per game) | lower | 2.0264 | 2.0264 (+0.0000; fl 0.0000) | 1.9042 (-0.1222; fl 0.0166) REPORTED | 2.0206 (-0.0058; fl 0.0286) REPORTED | 2.1528 (+0.1263; fl 0.0164) REPORTED |

#### Block A, F2 period 1 (period 1, REPORTED only): 66,424 rows, 5,110 games; control zero-mass rows excluded from dev_pos: 73 (0.11%, carrying 0.0607 of the control's `dev`); powered cells 44 of 45

| line | role | actual / target | CA0 (control) | CA0s1 (floor arm) | CA1 | CA2 | CA3 |
|---|---|---|---:|---:|---:|---:|---:|
| dev_pos | PRIMARY | lower | 6.6957 | 6.6957 (+0.0000; fl 0.0000) | 6.6908 (-0.0049; fl 0.0013) | 6.7440 (+0.0483; fl 0.0050) | 6.7197 (+0.0240; fl 0.0062) |
| cal | PRIMARY | lower | 0.6168 | 0.6168 (+0.0000; fl 0.0000) | 0.5821 (-0.0346; fl 0.0499) | 1.3743 (+0.7575; fl 0.0836) | 0.7937 (+0.1770; fl 0.0427) |
| dev | reported (i) | lower | 6.7491 | 6.7491 (+0.0000; fl 0.0000) | 6.7442 (-0.0048; fl 0.0013) | 6.7682 (+0.0192; fl 0.0096) | 6.7569 (+0.0078; fl 0.0086) |
| cal45 | reported | lower | 0.6468 | 0.6468 (+0.0000; fl 0.0000) | 0.6156 (-0.0312; fl 0.0488) | 1.4119 (+0.7651; fl 0.0821) | 0.8198 (+0.1730; fl 0.0417) |
| slope_tempo | HARD GATE (not below) | 1 | 0.4558 | 0.4558 (+0.0000; fl 0.0000) | 0.4624 (+0.0066; fl 0.0052) | 0.4896 (+0.0338; fl 0.0082) | 0.4579 (+0.0021; fl 0.0066) |
| slope_zoff | gate (not below) | 1 | 0.2708 | 0.2708 (+0.0000; fl 0.0000) | 0.2113 (-0.0596; fl 0.0097) | 0.2068 (-0.0640; fl 0.0140) | 0.1838 (-0.0871; fl 0.0134) |
| slope_defz | gate (not below) | 1 | 0.2806 | 0.2806 (+0.0000; fl 0.0000) | 0.1938 (-0.0868; fl 0.0296) | 0.2029 (-0.0777; fl 0.0376) | 0.1659 (-0.1147; fl 0.0370) |
| slope_team_lead | gate (not below) | 1 | 0.0652 | 0.0652 (+0.0000; fl 0.0000) | 0.1005 (+0.0353; fl 0.0111) | 0.1063 (+0.0411; fl 0.0182) | 0.0751 (+0.0099; fl 0.0114) |
| slope_team_trail | gate (not below) | 1 | 0.1625 | 0.1625 (+0.0000; fl 0.0000) | 0.0701 (-0.0924; fl 0.0433) | 0.0366 (-0.1259; fl 0.0660) | 0.0184 (-0.1441; fl 0.0667) |
| slope_lead | gate (|s-1| not worse) | 1 | 0.5626 | 0.5626 (+0.0000; fl 0.0000) | 0.8372 (+0.2745; fl 0.0656) | 0.9308 (+0.3682; fl 0.0860) | 0.6778 (+0.1151; fl 0.0332) |
| slope_trail | gate (|s-1| not worse) | 1 | -3.5277 | -3.5277 (+0.0000; fl 0.0000) | -1.3934 (+2.1343; fl 6.4655) | -0.8237 (+2.7040; fl 7.1358) | -0.2106 (+3.3172; fl 7.9758) |
| dev_end | reported | lower | 6.6996 | 6.6996 (+0.0000; fl 0.0000) | 6.6937 (-0.0059; fl 0.0013) | 6.6803 (-0.0193; fl 0.0179) | 6.6799 (-0.0197; fl 0.0153) |
| pergame_abs_err | reported (per game) | lower | 1.8707 | 1.8707 (+0.0000; fl 0.0000) | 1.8545 (-0.0162; fl 0.0051) | 1.9120 (+0.0413; fl 0.0253) | 1.8991 (+0.0285; fl 0.0082) |

#### Block A, F1 period 2 (confirm): 69,410 rows, 4,777 games; control zero-mass rows excluded from dev_pos: 196 (0.28%, carrying 0.1560 of the control's `dev`); powered cells 45 of 45

| line | role | actual / target | CA0 (control) | CA0s1 (floor arm) | CA1 | CA2 | CA3 |
|---|---|---|---:|---:|---:|---:|---:|
| dev_pos | PRIMARY | lower | 6.6109 | 6.6109 (+0.0000; fl 0.0000) | 6.4638 (-0.1471; fl 0.0053) | 6.6803 (+0.0694; fl 0.0123) | 6.7236 (+0.1127; fl 0.0182) |
| cal | PRIMARY | lower | 2.3193 | 2.3193 (+0.0000; fl 0.0000) | 0.4866 (-1.8327; fl 0.0467) | 1.8573 (-0.4620; fl 0.0735) | 2.1864 (-0.1329; fl 0.0415) |
| dev | reported (i) | lower | 6.7483 | 6.7483 (+0.0000; fl 0.0000) | 6.6016 (-0.1467; fl 0.0053) | 6.7094 (-0.0389; fl 0.0198) | 6.7891 (+0.0408; fl 0.0231) |
| cal45 | reported | lower | 2.3193 | 2.3193 (+0.0000; fl 0.0000) | 0.4866 (-1.8327; fl 0.0467) | 1.8573 (-0.4620; fl 0.0735) | 2.1864 (-0.1329; fl 0.0415) |
| slope_tempo | HARD GATE (not below) | 1 | 0.3804 | 0.3804 (+0.0000; fl 0.0000) | 0.4243 (+0.0439; fl 0.0450) | 0.4787 (+0.0983; fl 0.0716) | 0.3802 (-0.0002; fl 0.0244) |
| slope_zoff | gate (not below) | 1 | 0.4509 | 0.4509 (+0.0000; fl 0.0000) | 1.3235 (+0.8726; fl 0.1553) | 1.7413 (+1.2904; fl 0.2381) | 0.7122 (+0.2613; fl 0.0747) |
| slope_defz | gate (not below) | 1 | 0.3656 | 0.3656 (+0.0000; fl 0.0000) | 1.4937 (+1.1281; fl 0.4279) | 2.0037 (+1.6381; fl 0.6444) | 0.7593 (+0.3937; fl 0.1991) |
| slope_team_lead | gate (not below) | 1 | 0.4461 | 0.4461 (+0.0000; fl 0.0000) | 0.6627 (+0.2166; fl 0.1731) | 0.5882 (+0.1422; fl 0.1301) | 0.3937 (-0.0524; fl 0.0503) |
| slope_team_trail | gate (not below) | 1 | 0.1066 | 0.1066 (+0.0000; fl 0.0000) | -0.0345 (-0.1411; fl 0.0719) | 0.0203 (-0.0863; fl 0.0711) | -0.1401 (-0.2467; fl 0.0886) |
| slope_lead | gate (|s-1| not worse) | 1 | 0.4432 | 0.4432 (+0.0000; fl 0.0000) | 0.9517 (+0.5085; fl 0.0427) | 0.4475 (+0.0043; fl 0.0275) | 0.4173 (-0.0259; fl 0.0108) |
| slope_trail | gate (|s-1| not worse) | 1 | 0.5054 | 0.5054 (+0.0000; fl 0.0000) | 0.8531 (+0.3476; fl 0.0299) | 0.9036 (+0.3982; fl 0.0323) | 0.5620 (+0.0565; fl 0.0178) |
| dev_end | reported | lower | 6.7065 | 6.7065 (+0.0000; fl 0.0000) | 6.5701 (-0.1365; fl 0.0049) | 6.6598 (-0.0467; fl 0.0262) | 6.7423 (+0.0358; fl 0.0296) |
| pergame_abs_err | reported (per game) | lower | 2.0721 | 2.0721 (+0.0000; fl 0.0000) | 1.9589 (-0.1132; fl 0.0175) | 2.0410 (-0.0311; fl 0.0290) | 2.2114 (+0.1393; fl 0.0172) |

#### Block A, F1 period 1 (period 1, REPORTED only): 62,307 rows, 4,777 games; control zero-mass rows excluded from dev_pos: 98 (0.16%, carrying 0.0869 of the control's `dev`); powered cells 44 of 45

| line | role | actual / target | CA0 (control) | CA0s1 (floor arm) | CA1 | CA2 | CA3 |
|---|---|---|---:|---:|---:|---:|---:|
| dev_pos | PRIMARY | lower | 6.7060 | 6.7060 (+0.0000; fl 0.0000) | 6.7014 (-0.0046; fl 0.0014) | 6.7483 (+0.0422; fl 0.0060) | 6.7294 (+0.0234; fl 0.0079) |
| cal | PRIMARY | lower | 0.6033 | 0.6033 (+0.0000; fl 0.0000) | 0.5487 (-0.0546; fl 0.0588) | 1.3161 (+0.7129; fl 0.1002) | 0.7581 (+0.1548; fl 0.0466) |
| dev | reported (i) | lower | 6.7824 | 6.7824 (+0.0000; fl 0.0000) | 6.7777 (-0.0046; fl 0.0014) | 6.7797 (-0.0027; fl 0.0128) | 6.7725 (-0.0099; fl 0.0124) |
| cal45 | reported | lower | 0.6423 | 0.6423 (+0.0000; fl 0.0000) | 0.5941 (-0.0482; fl 0.0575) | 1.3680 (+0.7257; fl 0.0982) | 0.7936 (+0.1514; fl 0.0456) |
| slope_tempo | HARD GATE (not below) | 1 | 0.5324 | 0.5324 (+0.0000; fl 0.0000) | 0.5353 (+0.0029; fl 0.0058) | 0.5649 (+0.0325; fl 0.0099) | 0.5329 (+0.0005; fl 0.0082) |
| slope_zoff | gate (not below) | 1 | 0.2988 | 0.2988 (+0.0000; fl 0.0000) | 0.2347 (-0.0641; fl 0.0111) | 0.2272 (-0.0716; fl 0.0144) | 0.1983 (-0.1005; fl 0.0158) |
| slope_defz | gate (not below) | 1 | 0.4399 | 0.4399 (+0.0000; fl 0.0000) | 0.3650 (-0.0750; fl 0.0223) | 0.3751 (-0.0648; fl 0.0261) | 0.3353 (-0.1046; fl 0.0279) |
| slope_team_lead | gate (not below) | 1 | 0.1169 | 0.1169 (+0.0000; fl 0.0000) | 0.1409 (+0.0241; fl 0.0093) | 0.1484 (+0.0315; fl 0.0164) | 0.1358 (+0.0189; fl 0.0133) |
| slope_team_trail | gate (not below) | 1 | -0.0752 | -0.0752 (+0.0000; fl 0.0000) | -0.1626 (-0.0874; fl 0.0486) | -0.1876 (-0.1124; fl 0.0743) | -0.2129 (-0.1377; fl 0.0726) |
| slope_lead | gate (|s-1| not worse) | 1 | 0.5558 | 0.5558 (+0.0000; fl 0.0000) | 0.7439 (+0.1882; fl 0.0446) | 0.7970 (+0.2412; fl 0.0594) | 0.6655 (+0.1097; fl 0.0322) |
| slope_trail | gate (|s-1| not worse) | 1 | 0.8862 | 0.8862 (+0.0000; fl 0.0000) | -1.0771 (-1.9633; fl 4.3719) | -0.4469 (-1.3331; fl 4.9669) | -0.1228 (-1.0090; fl 4.7707) |
| dev_end | reported | lower | 6.7777 | 6.7777 (+0.0000; fl 0.0000) | 6.7709 (-0.0067; fl 0.0016) | 6.7241 (-0.0535; fl 0.0203) | 6.7311 (-0.0465; fl 0.0188) |
| pergame_abs_err | reported (per game) | lower | 1.8090 | 1.8090 (+0.0000; fl 0.0000) | 1.7921 (-0.0169; fl 0.0052) | 1.8367 (+0.0277; fl 0.0241) | 1.8415 (+0.0325; fl 0.0078) |

#### Block A per-cell calibration, F2 period 2 (mean first-chance seconds, E[min(T, R)]; U = UNDERPOWERED, < 200 rows)

| cell | n | actual | CA0 | CA1 | CA2 | CA3 |
|---|---:|---:|---:|---:|---:|---:|
| leader 0-3 x (180,240] | 1348 | 20.00 | 17.41 | 19.36 | 19.22 | 17.42 |
| leader 0-3 x (120,180] | 1371 | 20.75 | 17.41 | 20.29 | 20.14 | 17.42 |
| leader 0-3 x (60,120] | 1502 | 21.76 | 19.86 | 21.41 | 27.34 | 19.58 |
| leader 0-3 x (30,60] | 873 | 17.71 | 11.32 | 16.76 | 16.21 | 11.60 |
| leader 0-3 x (0,30] | 1648 | 2.76 | 6.82 | 2.97 | 3.26 | 6.57 |
| leader 4-7 x (180,240] | 1639 | 20.40 | 17.30 | 19.91 | 19.86 | 17.32 |
| leader 4-7 x (120,180] | 1658 | 20.78 | 17.27 | 20.31 | 20.55 | 17.29 |
| leader 4-7 x (60,120] | 1831 | 18.43 | 17.11 | 18.17 | 27.61 | 16.46 |
| leader 4-7 x (30,60] | 1257 | 6.97 | 8.99 | 6.99 | 8.47 | 9.60 |
| leader 4-7 x (0,30] | 1634 | 2.47 | 6.26 | 2.77 | 3.35 | 5.59 |
| leader 8-12 x (180,240] | 1663 | 19.71 | 17.35 | 19.62 | 19.96 | 17.39 |
| leader 8-12 x (120,180] | 1685 | 19.30 | 17.17 | 18.95 | 19.48 | 17.21 |
| leader 8-12 x (60,120] | 1859 | 15.06 | 16.95 | 14.94 | 24.22 | 15.78 |
| leader 8-12 x (30,60] | 1319 | 8.98 | 8.92 | 9.21 | 10.11 | 10.00 |
| leader 8-12 x (0,30] | 680 | 3.94 | 7.20 | 4.02 | 4.57 | 5.72 |
| leader 13+ x (180,240] | 3146 | 18.07 | 17.05 | 17.97 | 17.86 | 17.14 |
| leader 13+ x (120,180] | 3233 | 17.80 | 16.97 | 17.98 | 17.82 | 17.05 |
| leader 13+ x (60,120] | 3423 | 17.36 | 16.92 | 17.48 | 23.94 | 14.47 |
| leader 13+ x (30,60] | 1902 | 17.08 | 9.02 | 16.96 | 12.55 | 11.59 |
| leader 13+ x (0,30] | 322 | 9.73 | 7.70 | 8.74 | 6.74 | 4.19 |
| trailer 0-3 x (180,240] | 1363 | 18.59 | 17.56 | 18.01 | 17.61 | 17.16 |
| trailer 0-3 x (120,180] | 1406 | 17.87 | 17.32 | 17.80 | 17.44 | 16.83 |
| trailer 0-3 x (60,120] | 1432 | 17.36 | 19.82 | 17.93 | 22.21 | 19.95 |
| trailer 0-3 x (30,60] | 800 | 15.06 | 11.20 | 14.46 | 12.51 | 11.57 |
| trailer 0-3 x (0,30] | 1479 | 7.96 | 7.19 | 7.17 | 5.63 | 7.20 |
| trailer 4-7 x (180,240] | 1727 | 16.83 | 17.67 | 16.95 | 16.42 | 16.57 |
| trailer 4-7 x (120,180] | 1704 | 15.76 | 17.36 | 15.91 | 15.28 | 16.05 |
| trailer 4-7 x (60,120] | 1885 | 13.68 | 12.62 | 13.28 | 16.09 | 12.86 |
| trailer 4-7 x (30,60] | 1361 | 9.54 | 9.03 | 9.39 | 9.22 | 9.86 |
| trailer 4-7 x (0,30] | 2158 | 6.84 | 7.52 | 6.53 | 5.73 | 7.57 |
| trailer 8-12 x (180,240] | 1689 | 15.22 | 17.64 | 15.58 | 14.81 | 15.71 |
| trailer 8-12 x (120,180] | 1794 | 13.71 | 17.49 | 14.11 | 13.19 | 15.15 |
| trailer 8-12 x (60,120] | 2058 | 11.50 | 12.62 | 11.59 | 13.54 | 13.04 |
| trailer 8-12 x (30,60] | 1575 | 9.00 | 8.98 | 8.96 | 8.47 | 10.48 |
| trailer 8-12 x (0,30] | 1373 | 7.41 | 8.14 | 7.36 | 6.49 | 8.23 |
| trailer 13+ x (180,240] | 3532 | 15.39 | 17.92 | 15.55 | 14.70 | 13.87 |
| trailer 13+ x (120,180] | 3533 | 14.51 | 17.53 | 14.56 | 13.72 | 12.77 |
| trailer 13+ x (60,120] | 3817 | 13.29 | 12.97 | 13.20 | 15.76 | 13.96 |
| trailer 13+ x (30,60] | 2097 | 11.69 | 9.36 | 10.97 | 11.16 | 13.24 |
| trailer 13+ x (0,30] | 1250 | 9.21 | 8.47 | 8.63 | 8.12 | 8.68 |
| tied x (180,240] | 486 | 19.47 | 17.38 | 18.99 | 18.63 | 17.38 |
| tied x (120,180] | 494 | 19.70 | 17.29 | 19.10 | 18.95 | 17.29 |
| tied x (60,120] | 498 | 20.53 | 19.84 | 20.39 | 25.66 | 19.84 |
| tied x (30,60] | 281 | 20.89 | 11.42 | 19.78 | 16.66 | 11.42 |
| tied x (0,30] | 379 | 10.91 | 6.79 | 7.91 | 6.11 | 6.79 |

(30,60] bucket, the s46 finding (1) cells:

| cell | n | actual | CA0 | CA1 | CA2 | CA3 |
|---|---:|---:|---:|---:|---:|---:|
| leader 0-3 x (30,60] | 873 | 17.71 | 11.32 | 16.76 | 16.21 | 11.60 |
| leader 4-7 x (30,60] | 1257 | 6.97 | 8.99 | 6.99 | 8.47 | 9.60 |
| leader 8-12 x (30,60] | 1319 | 8.98 | 8.92 | 9.21 | 10.11 | 10.00 |
| leader 13+ x (30,60] | 1902 | 17.08 | 9.02 | 16.96 | 12.55 | 11.59 |
| trailer 0-3 x (30,60] | 800 | 15.06 | 11.20 | 14.46 | 12.51 | 11.57 |
| trailer 4-7 x (30,60] | 1361 | 9.54 | 9.03 | 9.39 | 9.22 | 9.86 |
| trailer 8-12 x (30,60] | 1575 | 9.00 | 8.98 | 8.96 | 8.47 | 10.48 |
| trailer 13+ x (30,60] | 2097 | 11.69 | 9.36 | 10.97 | 11.16 | 13.24 |
| tied x (30,60] | 281 | 20.89 | 11.42 | 19.78 | 16.66 | 11.42 |

#### Block A per possession type (start type), F2 period 2

| start | n | actual | CA0 | CA1 | CA2 | CA3 |
|---|---:|---:|---:|---:|---:|---:|
| DREB | 24655 | 13.03 | 12.91 | 13.27 | 14.13 | 12.53 |
| TOV | 10922 | 12.09 | 11.92 | 12.16 | 12.73 | 11.55 |
| made_FG | 22921 | 18.05 | 17.73 | 17.66 | 18.95 | 17.01 |
| made_FT | 15207 | 13.89 | 13.41 | 13.40 | 14.09 | 13.14 |
| other | 459 | 0.52 | 0.40 | 0.40 | 0.43 | 0.40 |

#### Block A per team, F2 period 2 (206 offence teams with >= 200 window rows)

| arm | share of teams moved in the actual's direction | share with |error| reduced |
|---|---:|---:|
| CA0s1 | 0.000 | 0.000 |
| CA1 | 0.665 | 0.583 |
| CA2 | 0.583 | 0.296 |
| CA3 | 0.427 | 0.325 |


## 5. Block B tables


Rows: F2 5,065 FPS first chances (3703 of 3711 shots matched to an fg_make design row); F1 4,580 (3326 of 3333).

#### Block B, F2 period 2, trail 1-3 (PRIMARY cell, selection): 1521 rows, 1054 games, 989 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.4956 | 1.4956 (+0.0000; fl 0.0000) | 1.4949 (-0.0007; fl 0.0010) FAIL (inside floor or worse) | 1.4949 (-0.0007; fl 0.0010) FAIL (inside floor or worse) | 1.4949 (-0.0007; fl 0.0010) FAIL (inside floor or worse) |
| make_dev | PRIMARY |  | 1.1807 | 1.1807 (+0.0000; fl 0.0000) | 1.1807 (+0.0000; fl 0.0000) FAIL (inside floor or worse) | 1.1718 (-0.0089; fl 0.0117) FAIL (inside floor or worse) | 1.1718 (-0.0089; fl 0.0117) FAIL (inside floor or worse) |
| share_3pa | reported | 0.3498 | 0.3576 | 0.3576 (+0.0000; fl 0.0000) | 0.3606 (+0.0030; fl 0.0001) REPORTED | 0.3606 (+0.0030; fl 0.0001) REPORTED | 0.3606 (+0.0030; fl 0.0001) REPORTED |
| share_2pa | reported | 0.3011 | 0.3176 | 0.3176 (+0.0000; fl 0.0000) | 0.3105 (-0.0071; fl 0.0001) REPORTED | 0.3105 (-0.0071; fl 0.0001) REPORTED | 0.3105 (-0.0071; fl 0.0001) REPORTED |
| share_tov | reported | 0.1302 | 0.1214 | 0.1214 (+0.0000; fl 0.0000) | 0.1204 (-0.0011; fl 0.0000) REPORTED | 0.1204 (-0.0011; fl 0.0000) REPORTED | 0.1204 (-0.0011; fl 0.0000) REPORTED |
| share_ft | reported | 0.2189 | 0.2034 | 0.2034 (+0.0000; fl 0.0000) | 0.2086 (+0.0052; fl 0.0001) REPORTED | 0.2086 (+0.0052; fl 0.0001) REPORTED | 0.2086 (+0.0052; fl 0.0001) REPORTED |
| make_rate | reported | 0.3296 | 0.3715 | 0.3715 (+0.0000; fl 0.0000) | 0.3715 (+0.0000; fl 0.0000) REPORTED | 0.3313 (-0.0402; fl 0.0004) REPORTED | 0.3313 (-0.0402; fl 0.0004) REPORTED |
| shot_pts_per_shot | reported | 0.7826 | 0.8667 | 0.8667 (+0.0000; fl 0.0000) | 0.8667 (+0.0000; fl 0.0000) REPORTED | 0.7678 (-0.0989; fl 0.0007) REPORTED | 0.7678 (-0.0989; fl 0.0007) REPORTED |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.1440 | 0.1440 (+0.0000; fl 0.0000) | 0.1450 (+0.0010; fl 0.0029) hold | 0.1450 (+0.0010; fl 0.0029) hold | 0.1450 (+0.0010; fl 0.0029) hold |
| slope_team_make | gate (not below; UNDERPOWERED) |  | 0.0725 | 0.0725 (+0.0000; fl 0.0000) | 0.0725 (+0.0000; fl 0.0000) hold | 0.0775 (+0.0050; fl 0.0140) hold | 0.0775 (+0.0050; fl 0.0140) hold |

#### Block B, F2 period 2, trail 4-6 (REPORTED): 1763 rows, 1120 games, 1417 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.3186 | 1.3186 (+0.0000; fl 0.0000) | 1.3186 (+0.0000; fl 0.0000) | 1.3186 (+0.0000; fl 0.0000) | 1.3186 (+0.0000; fl 0.0000) |
| make_dev | PRIMARY |  | 1.1925 | 1.1925 (+0.0000; fl 0.0000) | 1.1925 (+0.0000; fl 0.0000) | 1.1925 (+0.0000; fl 0.0000) | 1.1925 (+0.0000; fl 0.0000) |
| share_3pa | reported | 0.5156 | 0.4997 | 0.4997 (+0.0000; fl 0.0000) | 0.4997 (+0.0000; fl 0.0000) | 0.4997 (+0.0000; fl 0.0000) | 0.4997 (+0.0000; fl 0.0000) |
| share_2pa | reported | 0.2893 | 0.2758 | 0.2758 (+0.0000; fl 0.0000) | 0.2758 (+0.0000; fl 0.0000) | 0.2758 (+0.0000; fl 0.0000) | 0.2758 (+0.0000; fl 0.0000) |
| share_tov | reported | 0.0788 | 0.0986 | 0.0986 (+0.0000; fl 0.0000) | 0.0986 (+0.0000; fl 0.0000) | 0.0986 (+0.0000; fl 0.0000) | 0.0986 (+0.0000; fl 0.0000) |
| share_ft | reported | 0.1163 | 0.1259 | 0.1259 (+0.0000; fl 0.0000) | 0.1259 (+0.0000; fl 0.0000) | 0.1259 (+0.0000; fl 0.0000) | 0.1259 (+0.0000; fl 0.0000) |
| make_rate | reported | 0.3691 | 0.3580 | 0.3580 (+0.0000; fl 0.0000) | 0.3580 (+0.0000; fl 0.0000) | 0.3580 (+0.0000; fl 0.0000) | 0.3580 (+0.0000; fl 0.0000) |
| shot_pts_per_shot | reported | 0.8970 | 0.8644 | 0.8644 (+0.0000; fl 0.0000) | 0.8644 (+0.0000; fl 0.0000) | 0.8644 (+0.0000; fl 0.0000) | 0.8644 (+0.0000; fl 0.0000) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.1641 | 0.1641 (+0.0000; fl 0.0000) | 0.1641 (+0.0000; fl 0.0000) | 0.1641 (+0.0000; fl 0.0000) | 0.1641 (+0.0000; fl 0.0000) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | -0.0552 | -0.0552 (+0.0000; fl 0.0000) | -0.0552 (+0.0000; fl 0.0000) | -0.0552 (+0.0000; fl 0.0000) | -0.0552 (+0.0000; fl 0.0000) |

#### Block B, F2 period 2, tied (REPORTED): 372 rows, 330 games, 268 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.6109 | 1.6109 (+0.0000; fl 0.0000) | 1.6109 (+0.0000; fl 0.0000) | 1.6109 (+0.0000; fl 0.0000) | 1.6109 (+0.0000; fl 0.0000) |
| make_dev | PRIMARY |  | 1.2610 | 1.2610 (+0.0000; fl 0.0000) | 1.2610 (+0.0000; fl 0.0000) | 1.2610 (+0.0000; fl 0.0000) | 1.2610 (+0.0000; fl 0.0000) |
| share_3pa | reported | 0.2581 | 0.2895 | 0.2895 (+0.0000; fl 0.0000) | 0.2895 (+0.0000; fl 0.0000) | 0.2895 (+0.0000; fl 0.0000) | 0.2895 (+0.0000; fl 0.0000) |
| share_2pa | reported | 0.4651 | 0.4098 | 0.4098 (+0.0000; fl 0.0000) | 0.4098 (+0.0000; fl 0.0000) | 0.4098 (+0.0000; fl 0.0000) | 0.4098 (+0.0000; fl 0.0000) |
| share_tov | reported | 0.1425 | 0.1496 | 0.1496 (+0.0000; fl 0.0000) | 0.1496 (+0.0000; fl 0.0000) | 0.1496 (+0.0000; fl 0.0000) | 0.1496 (+0.0000; fl 0.0000) |
| share_ft | reported | 0.1344 | 0.1511 | 0.1511 (+0.0000; fl 0.0000) | 0.1511 (+0.0000; fl 0.0000) | 0.1511 (+0.0000; fl 0.0000) | 0.1511 (+0.0000; fl 0.0000) |
| make_rate | reported | 0.3022 | 0.3475 | 0.3475 (+0.0000; fl 0.0000) | 0.3475 (+0.0000; fl 0.0000) | 0.3475 (+0.0000; fl 0.0000) | 0.3475 (+0.0000; fl 0.0000) |
| shot_pts_per_shot | reported | 0.6940 | 0.7719 | 0.7719 (+0.0000; fl 0.0000) | 0.7719 (+0.0000; fl 0.0000) | 0.7719 (+0.0000; fl 0.0000) | 0.7719 (+0.0000; fl 0.0000) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | -0.0879 | -0.0879 (+0.0000; fl 0.0000) | -0.0879 (+0.0000; fl 0.0000) | -0.0879 (+0.0000; fl 0.0000) | -0.0879 (+0.0000; fl 0.0000) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | 0.0640 | 0.0640 (+0.0000; fl 0.0000) | 0.0640 (+0.0000; fl 0.0000) | 0.0640 (+0.0000; fl 0.0000) | 0.0640 (+0.0000; fl 0.0000) |

#### Block B, F2 period 1, trail 1-3 (REPORTED): 629 rows, 595 games, 452 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.5495 | 1.5495 (+0.0000; fl 0.0000) | 1.5521 (+0.0026; fl 0.0026) | 1.5521 (+0.0026; fl 0.0026) | 1.5521 (+0.0026; fl 0.0026) |
| make_dev | PRIMARY |  | 1.2581 | 1.2581 (+0.0000; fl 0.0000) | 1.2581 (+0.0000; fl 0.0000) | 1.2529 (-0.0052; fl 0.0064) | 1.2529 (-0.0052; fl 0.0064) |
| share_3pa | reported | 0.3482 | 0.3693 | 0.3693 (+0.0000; fl 0.0000) | 0.3727 (+0.0034; fl 0.0001) | 0.3727 (+0.0034; fl 0.0001) | 0.3727 (+0.0034; fl 0.0001) |
| share_2pa | reported | 0.3736 | 0.3669 | 0.3669 (+0.0000; fl 0.0000) | 0.3658 (-0.0011; fl 0.0002) | 0.3658 (-0.0011; fl 0.0002) | 0.3658 (-0.0011; fl 0.0002) |
| share_tov | reported | 0.1574 | 0.1475 | 0.1475 (+0.0000; fl 0.0000) | 0.1475 (-0.0000; fl 0.0001) | 0.1475 (-0.0000; fl 0.0001) | 0.1475 (-0.0000; fl 0.0001) |
| share_ft | reported | 0.1208 | 0.1163 | 0.1163 (+0.0000; fl 0.0000) | 0.1140 (-0.0023; fl 0.0001) | 0.1140 (-0.0023; fl 0.0001) | 0.1140 (-0.0023; fl 0.0001) |
| make_rate | reported | 0.3827 | 0.3415 | 0.3415 (+0.0000; fl 0.0000) | 0.3415 (+0.0000; fl 0.0000) | 0.3577 (+0.0163; fl 0.0002) | 0.3577 (+0.0163; fl 0.0002) |
| shot_pts_per_shot | reported | 0.9093 | 0.7996 | 0.7996 (+0.0000; fl 0.0000) | 0.7996 (+0.0000; fl 0.0000) | 0.8392 (+0.0396; fl 0.0005) | 0.8392 (+0.0396; fl 0.0005) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.0785 | 0.0785 (+0.0000; fl 0.0000) | 0.0734 (-0.0051; fl 0.0061) | 0.0734 (-0.0051; fl 0.0061) | 0.0734 (-0.0051; fl 0.0061) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | 0.0797 | 0.0797 (+0.0000; fl 0.0000) | 0.0797 (+0.0000; fl 0.0000) | 0.0808 (+0.0011; fl 0.0069) | 0.0808 (+0.0011; fl 0.0069) |

#### Block B, F2 period 1, trail 4-6 (REPORTED): 581 rows, 566 games, 433 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.5322 | 1.5322 (+0.0000; fl 0.0000) | 1.5322 (+0.0000; fl 0.0000) | 1.5322 (+0.0000; fl 0.0000) | 1.5322 (+0.0000; fl 0.0000) |
| make_dev | PRIMARY |  | 1.3087 | 1.3087 (+0.0000; fl 0.0000) | 1.3087 (+0.0000; fl 0.0000) | 1.3087 (+0.0000; fl 0.0000) | 1.3087 (+0.0000; fl 0.0000) |
| share_3pa | reported | 0.3460 | 0.3814 | 0.3814 (+0.0000; fl 0.0000) | 0.3814 (+0.0000; fl 0.0000) | 0.3814 (+0.0000; fl 0.0000) | 0.3814 (+0.0000; fl 0.0000) |
| share_2pa | reported | 0.4028 | 0.3568 | 0.3568 (+0.0000; fl 0.0000) | 0.3568 (+0.0000; fl 0.0000) | 0.3568 (+0.0000; fl 0.0000) | 0.3568 (+0.0000; fl 0.0000) |
| share_tov | reported | 0.1497 | 0.1477 | 0.1477 (+0.0000; fl 0.0000) | 0.1477 (+0.0000; fl 0.0000) | 0.1477 (+0.0000; fl 0.0000) | 0.1477 (+0.0000; fl 0.0000) |
| share_ft | reported | 0.1015 | 0.1141 | 0.1141 (+0.0000; fl 0.0000) | 0.1141 (+0.0000; fl 0.0000) | 0.1141 (+0.0000; fl 0.0000) | 0.1141 (+0.0000; fl 0.0000) |
| make_rate | reported | 0.3811 | 0.3420 | 0.3420 (+0.0000; fl 0.0000) | 0.3420 (+0.0000; fl 0.0000) | 0.3420 (+0.0000; fl 0.0000) | 0.3420 (+0.0000; fl 0.0000) |
| shot_pts_per_shot | reported | 0.9122 | 0.7945 | 0.7945 (+0.0000; fl 0.0000) | 0.7945 (+0.0000; fl 0.0000) | 0.7945 (+0.0000; fl 0.0000) | 0.7945 (+0.0000; fl 0.0000) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.0924 | 0.0924 (+0.0000; fl 0.0000) | 0.0924 (+0.0000; fl 0.0000) | 0.0924 (+0.0000; fl 0.0000) | 0.0924 (+0.0000; fl 0.0000) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | -0.1022 | -0.1022 (+0.0000; fl 0.0000) | -0.1022 (+0.0000; fl 0.0000) | -0.1022 (+0.0000; fl 0.0000) | -0.1022 (+0.0000; fl 0.0000) |

#### Block B, F2 period 1, tied (REPORTED): 199 rows, 172 games, 144 shot rows; slopes UNDERPOWERED (< 3,000 rows); cell UNDERPOWERED

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.5601 | 1.5601 (+0.0000; fl 0.0000) | 1.5601 (+0.0000; fl 0.0000) | 1.5601 (+0.0000; fl 0.0000) | 1.5601 (+0.0000; fl 0.0000) |
| make_dev | PRIMARY |  | 1.3227 | 1.3227 (+0.0000; fl 0.0000) | 1.3227 (+0.0000; fl 0.0000) | 1.3227 (+0.0000; fl 0.0000) | 1.3227 (+0.0000; fl 0.0000) |
| share_3pa | reported | 0.3417 | 0.3665 | 0.3665 (+0.0000; fl 0.0000) | 0.3665 (+0.0000; fl 0.0000) | 0.3665 (+0.0000; fl 0.0000) | 0.3665 (+0.0000; fl 0.0000) |
| share_2pa | reported | 0.3819 | 0.3753 | 0.3753 (+0.0000; fl 0.0000) | 0.3753 (+0.0000; fl 0.0000) | 0.3753 (+0.0000; fl 0.0000) | 0.3753 (+0.0000; fl 0.0000) |
| share_tov | reported | 0.1508 | 0.1525 | 0.1525 (+0.0000; fl 0.0000) | 0.1525 (+0.0000; fl 0.0000) | 0.1525 (+0.0000; fl 0.0000) | 0.1525 (+0.0000; fl 0.0000) |
| share_ft | reported | 0.1256 | 0.1058 | 0.1058 (+0.0000; fl 0.0000) | 0.1058 (+0.0000; fl 0.0000) | 0.1058 (+0.0000; fl 0.0000) | 0.1058 (+0.0000; fl 0.0000) |
| make_rate | reported | 0.4028 | 0.3412 | 0.3412 (+0.0000; fl 0.0000) | 0.3412 (+0.0000; fl 0.0000) | 0.3412 (+0.0000; fl 0.0000) | 0.3412 (+0.0000; fl 0.0000) |
| shot_pts_per_shot | reported | 0.9583 | 0.7964 | 0.7964 (+0.0000; fl 0.0000) | 0.7964 (+0.0000; fl 0.0000) | 0.7964 (+0.0000; fl 0.0000) | 0.7964 (+0.0000; fl 0.0000) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.0219 | 0.0219 (+0.0000; fl 0.0000) | 0.0219 (+0.0000; fl 0.0000) | 0.0219 (+0.0000; fl 0.0000) | 0.0219 (+0.0000; fl 0.0000) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | -0.2761 | -0.2761 (+0.0000; fl 0.0000) | -0.2761 (+0.0000; fl 0.0000) | -0.2761 (+0.0000; fl 0.0000) | -0.2761 (+0.0000; fl 0.0000) |

#### Block B, F1 period 2, trail 1-3 (confirm): 1357 rows, 934 games, 917 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.4798 | 1.4798 (+0.0000; fl 0.0000) | 1.4795 (-0.0002; fl 0.0013) | 1.4795 (-0.0002; fl 0.0013) | 1.4795 (-0.0002; fl 0.0013) |
| make_dev | PRIMARY |  | 1.1622 | 1.1622 (+0.0000; fl 0.0000) | 1.1622 (+0.0000; fl 0.0000) | 1.1527 (-0.0095; fl 0.0113) | 1.1527 (-0.0095; fl 0.0113) |
| share_3pa | reported | 0.3559 | 0.3653 | 0.3653 (+0.0000; fl 0.0000) | 0.3666 (+0.0012; fl 0.0001) | 0.3666 (+0.0012; fl 0.0001) | 0.3666 (+0.0012; fl 0.0001) |
| share_2pa | reported | 0.3220 | 0.3244 | 0.3244 (+0.0000; fl 0.0000) | 0.3140 (-0.0104; fl 0.0002) | 0.3140 (-0.0104; fl 0.0002) | 0.3140 (-0.0104; fl 0.0002) |
| share_tov | reported | 0.1179 | 0.1243 | 0.1243 (+0.0000; fl 0.0000) | 0.1261 (+0.0017; fl 0.0001) | 0.1261 (+0.0017; fl 0.0001) | 0.1261 (+0.0017; fl 0.0001) |
| share_ft | reported | 0.2041 | 0.1859 | 0.1859 (+0.0000; fl 0.0000) | 0.1933 (+0.0074; fl 0.0001) | 0.1933 (+0.0074; fl 0.0001) | 0.1933 (+0.0074; fl 0.0001) |
| make_rate | reported | 0.3217 | 0.3650 | 0.3650 (+0.0000; fl 0.0000) | 0.3650 (+0.0000; fl 0.0000) | 0.3238 (-0.0412; fl 0.0005) | 0.3238 (-0.0412; fl 0.0005) |
| shot_pts_per_shot | reported | 0.7394 | 0.8505 | 0.8505 (+0.0000; fl 0.0000) | 0.8505 (+0.0000; fl 0.0000) | 0.7498 (-0.1007; fl 0.0007) | 0.7498 (-0.1007; fl 0.0007) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.2463 | 0.2463 (+0.0000; fl 0.0000) | 0.2537 (+0.0074; fl 0.0085) | 0.2537 (+0.0074; fl 0.0085) | 0.2537 (+0.0074; fl 0.0085) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | 0.1698 | 0.1698 (+0.0000; fl 0.0000) | 0.1698 (+0.0000; fl 0.0000) | 0.1565 (-0.0134; fl 0.0241) | 0.1565 (-0.0134; fl 0.0241) |

#### Block B, F1 period 2, trail 4-6 (REPORTED): 1520 rows, 936 games, 1160 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.3968 | 1.3968 (+0.0000; fl 0.0000) | 1.3968 (+0.0000; fl 0.0000) | 1.3968 (+0.0000; fl 0.0000) | 1.3968 (+0.0000; fl 0.0000) |
| make_dev | PRIMARY |  | 1.1507 | 1.1507 (+0.0000; fl 0.0000) | 1.1507 (+0.0000; fl 0.0000) | 1.1507 (+0.0000; fl 0.0000) | 1.1507 (+0.0000; fl 0.0000) |
| share_3pa | reported | 0.4954 | 0.5135 | 0.5135 (+0.0000; fl 0.0000) | 0.5135 (+0.0000; fl 0.0000) | 0.5135 (+0.0000; fl 0.0000) | 0.5135 (+0.0000; fl 0.0000) |
| share_2pa | reported | 0.2684 | 0.2712 | 0.2712 (+0.0000; fl 0.0000) | 0.2712 (+0.0000; fl 0.0000) | 0.2712 (+0.0000; fl 0.0000) | 0.2712 (+0.0000; fl 0.0000) |
| share_tov | reported | 0.1079 | 0.0987 | 0.0987 (+0.0000; fl 0.0000) | 0.0987 (+0.0000; fl 0.0000) | 0.0987 (+0.0000; fl 0.0000) | 0.0987 (+0.0000; fl 0.0000) |
| share_ft | reported | 0.1283 | 0.1166 | 0.1166 (+0.0000; fl 0.0000) | 0.1166 (+0.0000; fl 0.0000) | 0.1166 (+0.0000; fl 0.0000) | 0.1166 (+0.0000; fl 0.0000) |
| make_rate | reported | 0.3448 | 0.3543 | 0.3543 (+0.0000; fl 0.0000) | 0.3543 (+0.0000; fl 0.0000) | 0.3543 (+0.0000; fl 0.0000) | 0.3543 (+0.0000; fl 0.0000) |
| shot_pts_per_shot | reported | 0.8371 | 0.8619 | 0.8619 (+0.0000; fl 0.0000) | 0.8619 (+0.0000; fl 0.0000) | 0.8619 (+0.0000; fl 0.0000) | 0.8619 (+0.0000; fl 0.0000) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.2224 | 0.2224 (+0.0000; fl 0.0000) | 0.2224 (+0.0000; fl 0.0000) | 0.2224 (+0.0000; fl 0.0000) | 0.2224 (+0.0000; fl 0.0000) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | 0.1428 | 0.1428 (+0.0000; fl 0.0000) | 0.1428 (+0.0000; fl 0.0000) | 0.1428 (+0.0000; fl 0.0000) | 0.1428 (+0.0000; fl 0.0000) |

#### Block B, F1 period 2, tied (REPORTED): 368 rows, 324 games, 276 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.5609 | 1.5609 (+0.0000; fl 0.0000) | 1.5609 (+0.0000; fl 0.0000) | 1.5609 (+0.0000; fl 0.0000) | 1.5609 (+0.0000; fl 0.0000) |
| make_dev | PRIMARY |  | 1.2368 | 1.2368 (+0.0000; fl 0.0000) | 1.2368 (+0.0000; fl 0.0000) | 1.2368 (+0.0000; fl 0.0000) | 1.2368 (+0.0000; fl 0.0000) |
| share_3pa | reported | 0.2962 | 0.2755 | 0.2755 (+0.0000; fl 0.0000) | 0.2755 (+0.0000; fl 0.0000) | 0.2755 (+0.0000; fl 0.0000) | 0.2755 (+0.0000; fl 0.0000) |
| share_2pa | reported | 0.4620 | 0.4169 | 0.4169 (+0.0000; fl 0.0000) | 0.4169 (+0.0000; fl 0.0000) | 0.4169 (+0.0000; fl 0.0000) | 0.4169 (+0.0000; fl 0.0000) |
| share_tov | reported | 0.1332 | 0.1593 | 0.1593 (+0.0000; fl 0.0000) | 0.1593 (+0.0000; fl 0.0000) | 0.1593 (+0.0000; fl 0.0000) | 0.1593 (+0.0000; fl 0.0000) |
| share_ft | reported | 0.1087 | 0.1482 | 0.1482 (+0.0000; fl 0.0000) | 0.1482 (+0.0000; fl 0.0000) | 0.1482 (+0.0000; fl 0.0000) | 0.1482 (+0.0000; fl 0.0000) |
| make_rate | reported | 0.3297 | 0.3328 | 0.3328 (+0.0000; fl 0.0000) | 0.3328 (+0.0000; fl 0.0000) | 0.3328 (+0.0000; fl 0.0000) | 0.3328 (+0.0000; fl 0.0000) |
| shot_pts_per_shot | reported | 0.7717 | 0.7521 | 0.7521 (+0.0000; fl 0.0000) | 0.7521 (+0.0000; fl 0.0000) | 0.7521 (+0.0000; fl 0.0000) | 0.7521 (+0.0000; fl 0.0000) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.1228 | 0.1228 (+0.0000; fl 0.0000) | 0.1228 (+0.0000; fl 0.0000) | 0.1228 (+0.0000; fl 0.0000) | 0.1228 (+0.0000; fl 0.0000) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | 1.3936 | 1.3936 (+0.0000; fl 0.0000) | 1.3936 (+0.0000; fl 0.0000) | 1.3936 (+0.0000; fl 0.0000) | 1.3936 (+0.0000; fl 0.0000) |

#### Block B, F1 period 1, trail 1-3 (REPORTED): 613 rows, 571 games, 442 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.5780 | 1.5780 (+0.0000; fl 0.0000) | 1.5824 (+0.0043; fl 0.0054) | 1.5824 (+0.0043; fl 0.0054) | 1.5824 (+0.0043; fl 0.0054) |
| make_dev | PRIMARY |  | 1.1851 | 1.1851 (+0.0000; fl 0.0000) | 1.1851 (+0.0000; fl 0.0000) | 1.1859 (+0.0007; fl 0.0074) | 1.1859 (+0.0007; fl 0.0074) |
| share_3pa | reported | 0.3491 | 0.3647 | 0.3647 (+0.0000; fl 0.0000) | 0.3723 (+0.0076; fl 0.0002) | 0.3723 (+0.0076; fl 0.0002) | 0.3723 (+0.0076; fl 0.0002) |
| share_2pa | reported | 0.3719 | 0.3661 | 0.3661 (+0.0000; fl 0.0000) | 0.3641 (-0.0020; fl 0.0003) | 0.3641 (-0.0020; fl 0.0003) | 0.3641 (-0.0020; fl 0.0003) |
| share_tov | reported | 0.1468 | 0.1528 | 0.1528 (+0.0000; fl 0.0000) | 0.1534 (+0.0006; fl 0.0001) | 0.1534 (+0.0006; fl 0.0001) | 0.1534 (+0.0006; fl 0.0001) |
| share_ft | reported | 0.1321 | 0.1164 | 0.1164 (+0.0000; fl 0.0000) | 0.1102 (-0.0063; fl 0.0002) | 0.1102 (-0.0063; fl 0.0002) | 0.1102 (-0.0063; fl 0.0002) |
| make_rate | reported | 0.3394 | 0.3345 | 0.3345 (+0.0000; fl 0.0000) | 0.3345 (+0.0000; fl 0.0000) | 0.3528 (+0.0184; fl 0.0003) | 0.3528 (+0.0184; fl 0.0003) |
| shot_pts_per_shot | reported | 0.7805 | 0.7865 | 0.7865 (+0.0000; fl 0.0000) | 0.7865 (+0.0000; fl 0.0000) | 0.8312 (+0.0447; fl 0.0006) | 0.8312 (+0.0447; fl 0.0006) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | -0.0053 | -0.0053 (+0.0000; fl 0.0000) | -0.0039 (+0.0014; fl 0.0041) | -0.0039 (+0.0014; fl 0.0041) | -0.0039 (+0.0014; fl 0.0041) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | 0.1298 | 0.1298 (+0.0000; fl 0.0000) | 0.1298 (+0.0000; fl 0.0000) | 0.1317 (+0.0019; fl 0.0076) | 0.1317 (+0.0019; fl 0.0076) |

#### Block B, F1 period 1, trail 4-6 (REPORTED): 519 rows, 506 games, 389 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.5198 | 1.5198 (+0.0000; fl 0.0000) | 1.5198 (+0.0000; fl 0.0000) | 1.5198 (+0.0000; fl 0.0000) | 1.5198 (+0.0000; fl 0.0000) |
| make_dev | PRIMARY |  | 1.2487 | 1.2487 (+0.0000; fl 0.0000) | 1.2487 (+0.0000; fl 0.0000) | 1.2487 (+0.0000; fl 0.0000) | 1.2487 (+0.0000; fl 0.0000) |
| share_3pa | reported | 0.3565 | 0.3775 | 0.3775 (+0.0000; fl 0.0000) | 0.3775 (+0.0000; fl 0.0000) | 0.3775 (+0.0000; fl 0.0000) | 0.3775 (+0.0000; fl 0.0000) |
| share_2pa | reported | 0.3931 | 0.3612 | 0.3612 (+0.0000; fl 0.0000) | 0.3612 (+0.0000; fl 0.0000) | 0.3612 (+0.0000; fl 0.0000) | 0.3612 (+0.0000; fl 0.0000) |
| share_tov | reported | 0.1368 | 0.1496 | 0.1496 (+0.0000; fl 0.0000) | 0.1496 (+0.0000; fl 0.0000) | 0.1496 (+0.0000; fl 0.0000) | 0.1496 (+0.0000; fl 0.0000) |
| share_ft | reported | 0.1137 | 0.1116 | 0.1116 (+0.0000; fl 0.0000) | 0.1116 (+0.0000; fl 0.0000) | 0.1116 (+0.0000; fl 0.0000) | 0.1116 (+0.0000; fl 0.0000) |
| make_rate | reported | 0.3522 | 0.3349 | 0.3349 (+0.0000; fl 0.0000) | 0.3349 (+0.0000; fl 0.0000) | 0.3349 (+0.0000; fl 0.0000) | 0.3349 (+0.0000; fl 0.0000) |
| shot_pts_per_shot | reported | 0.8278 | 0.7846 | 0.7846 (+0.0000; fl 0.0000) | 0.7846 (+0.0000; fl 0.0000) | 0.7846 (+0.0000; fl 0.0000) | 0.7846 (+0.0000; fl 0.0000) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.0288 | 0.0288 (+0.0000; fl 0.0000) | 0.0288 (+0.0000; fl 0.0000) | 0.0288 (+0.0000; fl 0.0000) | 0.0288 (+0.0000; fl 0.0000) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | 0.0804 | 0.0804 (+0.0000; fl 0.0000) | 0.0804 (+0.0000; fl 0.0000) | 0.0804 (+0.0000; fl 0.0000) | 0.0804 (+0.0000; fl 0.0000) |

#### Block B, F1 period 1, tied (REPORTED): 203 rows, 169 games, 142 shot rows; slopes UNDERPOWERED (< 3,000 rows)

| line | role | actual | FP0 (control) | FP0s1 | FP1 | FP2 | FP3 (= FP2 offline) |
|---|---|---:|---:|---:|---:|---:|---:|
| mll | PRIMARY |  | 1.5654 | 1.5654 (+0.0000; fl 0.0000) | 1.5654 (+0.0000; fl 0.0000) | 1.5654 (+0.0000; fl 0.0000) | 1.5654 (+0.0000; fl 0.0000) |
| make_dev | PRIMARY |  | 1.3250 | 1.3250 (+0.0000; fl 0.0000) | 1.3250 (+0.0000; fl 0.0000) | 1.3250 (+0.0000; fl 0.0000) | 1.3250 (+0.0000; fl 0.0000) |
| share_3pa | reported | 0.3498 | 0.3557 | 0.3557 (+0.0000; fl 0.0000) | 0.3557 (+0.0000; fl 0.0000) | 0.3557 (+0.0000; fl 0.0000) | 0.3557 (+0.0000; fl 0.0000) |
| share_2pa | reported | 0.3498 | 0.3757 | 0.3757 (+0.0000; fl 0.0000) | 0.3757 (+0.0000; fl 0.0000) | 0.3757 (+0.0000; fl 0.0000) | 0.3757 (+0.0000; fl 0.0000) |
| share_tov | reported | 0.1675 | 0.1629 | 0.1629 (+0.0000; fl 0.0000) | 0.1629 (+0.0000; fl 0.0000) | 0.1629 (+0.0000; fl 0.0000) | 0.1629 (+0.0000; fl 0.0000) |
| share_ft | reported | 0.1330 | 0.1058 | 0.1058 (+0.0000; fl 0.0000) | 0.1058 (+0.0000; fl 0.0000) | 0.1058 (+0.0000; fl 0.0000) | 0.1058 (+0.0000; fl 0.0000) |
| make_rate | reported | 0.4155 | 0.3378 | 0.3378 (+0.0000; fl 0.0000) | 0.3378 (+0.0000; fl 0.0000) | 0.3378 (+0.0000; fl 0.0000) | 0.3378 (+0.0000; fl 0.0000) |
| shot_pts_per_shot | reported | 0.9859 | 0.7981 | 0.7981 (+0.0000; fl 0.0000) | 0.7981 (+0.0000; fl 0.0000) | 0.7981 (+0.0000; fl 0.0000) | 0.7981 (+0.0000; fl 0.0000) |
| slope_team_3pa | gate (not below; UNDERPOWERED) |  | 0.0722 | 0.0722 (+0.0000; fl 0.0000) | 0.0722 (+0.0000; fl 0.0000) | 0.0722 (+0.0000; fl 0.0000) | 0.0722 (+0.0000; fl 0.0000) |
| slope_team_make | gate (not below; UNDERPOWERED) |  | -0.1499 | -0.1499 (+0.0000; fl 0.0000) | -0.1499 (+0.0000; fl 0.0000) | -0.1499 (+0.0000; fl 0.0000) | -0.1499 (+0.0000; fl 0.0000) |


## 6. Findings (evidence only; no arm is proposed)

**Block A, CA1.** CA1 (a ratio cell term on the served row law) fixes the (30,60] cells named in s46 finding (1):

| (30,60] cell | control | CA1 | actual |
|---|---:|---:|---:|
| leader 13+ | 9.02 s | 16.96 s | 17.08 s |
| tied | 11.42 s | 19.78 s | 20.89 s |
| leader 0-3 | 11.32 s | 16.76 s | 17.71 s |
| trailer 0-3 | 11.20 s | 14.46 s | 15.06 s |

- Cell calibration goes from 2.29 s to 0.40 s.
- The positive-mass deviance improves as much as the full-set deviance (-0.148 vs -0.148), so the gain is not zero-mass smoothing.
- The within-window tempo response is kept: 0.365 -> 0.403. GT1 failed this gate.
- The band slopes move to about 1: lead 0.48 -> 1.04, trail 0.53 -> 0.90.
- Per-game |error| falls from 2.03 s to 1.90 s, and 67% of teams move in the actual's direction.

CA1 fails one registered gate, the team-prior-quintile slope for TRAILING offences:
- F2: +0.025 -> -0.120 (floor 0.084). F1 has the same sign: +0.107 -> -0.035 (floor 0.072).
- The control is already flat on this line; the target is 1.

The z_off and defence-z quintile slopes overshoot their target of 1:
- z_off: 0.69 -> 2.10. Defence-z: 0.79 -> 3.04.
- The registered gate reads only "not below control", so both pass, but both end far from 1.

**Block A, CA2 and CA3.** The AFT forms cannot reproduce the cell shape.
- Both are WORSE than the control on dev_pos: +0.080 (floor 0.011) and +0.120 (floor 0.020).
- A pure time-scale moves the whole pmf, but the miss is in the pmf's shape: the (30,60] cells mix clock-run-out possessions with fast ones.

**Block A, period 1 (reported only).** CA1 changes little: dev_pos -0.005, and cal 0.62 -> 0.58 is inside its floor.

**Block B, class mix.** Offline, the served possession_outcome class law is close to calibrated on trailing-1-3 final possessions (F2):

| share | model | actual |
|---|---:|---:|
| 3PA | 0.358 | 0.350 |
| 2PA | 0.318 | 0.301 |
| TOV | 0.121 | 0.130 |
| FT | 0.203 | 0.219 |

- The s46 finding (3) gap (3PA 36.1 vs 31.1%, 2PA 26.0 vs 29.7%) was measured sim vs real in the closed loop.
- The class model does not show that gap offline. So the loop's mix gap comes from the sim's state at these possessions (start clock, margin and bonus distribution), not from the class law given the state.
- FP1 moves each share by less than 1 pp.

**Block B, make rate.** The served fg_make law over-predicts makes on these shots:

| slice | served | actual |
|---|---:|---:|
| trailing 1-3, F2 | 0.372 | 0.330 |
| trailing 1-3, F1 | 0.365 | 0.322 |
| tied, F2 | 0.348 | 0.302 |
| trailing 4-6, F2 (opposite sign) | 0.358 | 0.369 |
| period 1 (opposite sign) | 0.342 | 0.383 |

- FP2's intercept for F2 period 2 is -0.206 after shrink (MLE -0.210 on 2,536 training shots).
- It brings the F2 make rate to 0.331, and predicted points per shot from 0.867 to 0.768 (actual 0.783).
- On 989 test shots the make-deviance gain (-0.0089) is 0.76 of its floor (0.0117). This is UNDERPOWERED, not a win.

**Bonus occupancy.** Report-only, and a loop line; not read because no loop ran.

## 7. NOT RUN, and why

- **Closed-loop screens (500 x 25):** NOT RUN for any arm, so the seed-count check, MD1-MD6 and every s45 veto line were not read either. s46 4.2 screens offline winners only, and neither block has one.
- **Full-size box read (5,710 x 200) and the control-draw reuse check (ruling 1):** NOT RUN. No box was launched; cost $0.
- **Blocks C (C-pre, ED/EG/EM/EH) and D:** not scheduled for this day (ruling 2).
