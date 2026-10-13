# Late-game round 7 (endgame margin/clock/foul): results, 2026-10-13

Pre-registration: `docs/models/late_game/experiments.md` section 15 (PROMOTED 2026-10-09) with possession_outcome
`experiments.md` section 29 as block F; PM rulings 2026-10-13 (Q1 F first in the loop, offline C/E/F; Q2 E enters
the loop only if EV1 wins offline; Q3 period 1 reported only; local compute only). Results section:
`late_game/experiments.md` section 16. **NO ARM ADOPTED. No served default changed. Model freeze ON. Parity v10
bit-identical with every new flag off.**

## 1. Outcome

| block | offline (F2 selects, F1 confirms) | closed loop (500 x 25 paired, served v3) | decision (literal rule) |
|---|---|---|---|
| C clock (GT0 / GT0s1 / GT2 / GT1) | no winner. GT1 beats GT0 on `dev` (+4.8 floors) and `cal` (+29 floors) and on slopes (a), but FAILS mandatory slope (b) (tempo slope 0.365 -> -0.009, floor 0.100). GT2 beats `dev` (+3.8) but not `cal` (+0.07); F1 sign lost (its F1 fit is b = 0); team-trailer slope falls beyond its floor | not eligible (no block winner); GT1 run as a labelled DIAGNOSTIC read only | **NO ARM** (s15.A.4: "an arm WINS a block if on fold 2 it beats its control by more than 1 floor on the block primary, all mandatory slopes hold, and fold 1 keeps the sign") |
| E event (EV0 / EV0s1 / EV1) | **EV1 wins**: mll -0.0011 (2.8 floors), F1 -0.0013 same sign, slope (c) holds | primary (leader TOV 13+) moves AWAY: 4:00 ref 0.222 -> 0.228 (-1.7 floors), 2:00 ref 0.254 -> 0.261 (-1.0); fails MD1-MD5 | **NO ARM** ("a block winner or composition is a candidate if its closed-loop primary moves toward the actual by more than one floor and every hard veto passes"; MD1-MD6 of s15.1 also fail) |
| F foul state (LF0 / LF0s1 / LF1 / LF2) | **LF2 selected** (s29.2): LF1 +13.3 floors, LF2 +9.7 floors vs LF0; LF2 beats LF1 by 5.9 floors (F1 5.0); (d) holds | tie rate 3.62% -> 3.65% (+0.09 floors toward 6.80%); leader bonus occupancy up 0.9-1.2 pp (1.6-2.2 floors toward real); every hard veto passes; fails MD1-MD5 | **NO ARM** (primary does not move by a floor; MD gates fail) |
| composition EV1+LF2 (block winners) | -- | tie rate +0.27 floors; fails MD1-MD5 | **NO ARM** |
| clear best | -- | veto-passing eligible arms LF2, EV1, EV1+LF2: none beats every other by more than one floor on its primary (LF2 vs EV1+LF2 tie rate 0.0365 vs 0.0371, floor 0.0036) | **NO ARM ADOPTED** ("otherwise NO ARM ADOPTED") |

No candidate, so no full-size 5,710 x 200 read and no fold-1 closed-loop confirmation were run (both are for
candidates only). The draft's listed compositions (GTw+EVw, GTw+LFw, GTw+EVw+LFw) all need a block-C winner and do not
exist; EV1+LF2 was run as the only composition of block winners (PM Q1 ruling).

## 2. What ran (all local; commits)

- `fe9ee15` ONE blind grader `scripts/grade_endgame_v1.py` (offline + loop modes), window builder
  `scripts/build_endgame_window_v1.py`, law module `src/cbb_sim/models/endgame.py`, trainers
  `scripts/train_endgame_{clock,event,foul}_v1.py`: committed BEFORE any arm was fitted (only the served-law
  control predictions and the served control loop runs existed; they were used to smoke-test the grader).
- `f520692` offline fits + artifacts `data/processed/models/late_game/endgame_v1/{F1,F2}[_seed1]/` (< 0.2 MB per
  fold) + default-off wiring: `ENGINE_ENDGAME=GT1|GT2|EV1` (`src/cbb_sim/engine/endgame_adapter.py`),
  `ENGINE_LATE_FOUL=eg_LF1|eg_LF2` (`late_foul.py`); parity v10 PASS bit-identical flags off; 7 unit tests.
- `5ae0302` grader clarification BEFORE any closed-loop arm ran: block E's loop primary = leader TOV in the 13+
  state at the 4:00 and 2:00 references (the only E targets s15.A.3 quotes); the 8-12 lines (already on target,
  D3 17.9 vs 17.9) became move-away vetoes.
- Closed loop: `scripts/run_engine_window_v2.py`, `stride500_verified_minswap_v1_F2_2025` (500 games), served v3,
  seeds 0-24 (arms and control paired), control floor draws at seed offsets 1000/2000/3000/4000, trajectories on
  every seed; tags `eg_ctrl_o{0,1000,2000,3000,4000}_s25`, `eg_LF2_o0_s25`, `eg_EV1_o0_s25`, `eg_EV1LF2_o0_s25`,
  `eg_GT1diag_o0_s25` (results/endgame/loop, results/trajectories; gitignored). About 2.2 min per 12,500 game-seeds
  on 12 workers.
- D3/D4 tables rerun on every run's taps: `diag_c4_extract_v1.py --traj-file` + `diag_c4_d34_endgame_v1.py`
  (results/endgame/d34/<tag>_d3.md, _d4.md); multi-level table `scripts/diag_endgame_multilevel_v1.py`.

## 3. Implementation choices fixed before any fit (disclosed)

1. **GT1** = Kaplan-Meier first-chance pmf per role-band x time bucket (per half), round-3 tail rule, shrunk with m =
   50 pseudo-rows as a mixture cell -> time bucket -> the served row law (`p = a_c p_c + (1-a_c)(a_b p_b +
   (1-a_b) p0_row)`, a = n/(n+m)); K2's end-class-conditional re-draw law gets the same hierarchy with the end
   class added. Literal cell law: once a cell is populated it carries none of the served law's tempo dimension.
2. **GT2** = AFT scale on the served row law (continuous-CDF `clock_r7.scale_pmf`), maximum likelihood per half
   (Nelder-Mead), applied identically to the marginal and the conditional law.
3. **EV1** = observed/expected class ratio per role-band x time bucket shrunk to the served row law with m = 50
   (`(n O/E + m)/(n + m)`), so the served model's row effects incl. the site main effect are carried (s15.A.2).
   The served law on TRAINING rows is the fold's first S1 refit (in-sample; stated).
4. **Block F** = section 29 unchanged (`train_late_foul_v1.fit` imported); F1's LF0 uses the fold-1 A2 table
   (`fold1_v1/loop/lut_acc_A2_F1.npz`); section 29's own script read the F2 table on both folds, which F1 cannot use.
5. **Engine scope**: period 2, possession start <= 240 s only (selection scope; Q3), same uniforms, same game latent,
   both the K2 marginal draw and the K2 re-draw; first halves are bit-identical to the control (verified).
6. **Noise-floor arms**: GT0s1 = the served K2 law refitted under seed 1 by its own trainer
   (`r8_K2/{F1,F2}_seed1`, gitignored); EV0s1 = EV1's law under seed 1; LF0s1 = the F winner (LF2) under seed 1.
7. **Closed-loop reading**: real trailing-team fouls per leader possession have no like-for-like value in the
   possession table (its open-time team-foul counters miss fouls on the possession-change second: 0.06 real vs 0.65
   pbp events), so that line is sim-only and the D4 pbp count (an upper bound) is in the D4 tables;
   window-possessions-by-k cells with < 60 real games are labelled UNDERPOWERED and do not veto (every k cell is
   < 60 at 500 games).

Reseed proof (`results/endgame/grade/reseed_proof.json`): max abs difference 0.0 for every law and every
prediction (GT0 vs GT0s1, GT1/GT2 seed 0 vs 1, EV1 vs EV0s1, LF2 vs LF0s1; both folds). Reseed floor = 0; the
binding floor is 2 x the paired game-block bootstrap SE (200 draws, seed 12345).

## 4. Offline (blind table, fold 2 then fold 1; period 2 selects, period 1 reported)

Window rows: C 140,588 (F2 test) / 131,717 (F1); E 149,475 / 142,646 window first chances; F 45,353 / 45,448.
Training rows: training seasons only; every test row scored by refits with max_train_date < game date (asserted).
Codes (C-A, ...) are the blind labels the table was written under before the key.

**Block C, F2, period 2 (selection)** (74,164 rows, 5,110 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | dev | cal | slope_lead | slope_trail | slope_tempo | slope_team_lead | slope_team_trail | dev_end |
|---|---|---|---|---|---|---|---|---|
| control | 6.6899 | 2.2936 | 0.4796 | 0.5349 | 0.3649 | 0.5700 | 0.0248 | 6.6277 |
| GT1 (C-A) | 6.6178 (-0.0720; 0.0149) | 0.3355 (-1.9582; 0.0668) | 1.0291 (+0.5495; 0.0654) | 0.9439 (+0.4090; 0.0459) | -0.0092 (-0.3740; 0.0998) | 0.9026 (+0.3325; 0.4996) | -0.0587 (-0.0835; 0.1182) | 6.4615 (-0.1663; 0.0228) |
| GT0s1 (C-B) | 6.6899 (+0.0000; 0.0000) | 2.2936 (+0.0000; 0.0000) | 0.4796 (+0.0000; 0.0000) | 0.5349 (+0.0000; 0.0000) | 0.3649 (+0.0000; 0.0000) | 0.5700 (+0.0000; 0.0000) | 0.0248 (+0.0000; 0.0000) | 6.6277 (+0.0000; 0.0000) |
| GT2 (C-C) | 6.6680 (-0.0219; 0.0058) | 2.2936 (-0.0000; 0.0004) | 0.4796 (+0.0000; 0.0000) | 0.5358 (+0.0009; 0.0003) | 0.3651 (+0.0003; 0.0004) | 0.5700 (+0.0000; 0.0000) | 0.0207 (-0.0041; 0.0018) | 6.5727 (-0.0550; 0.0096) |

**Block C, F2, period 1 (reported only)** (66,424 rows, 5,110 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | dev | cal | slope_lead | slope_trail | slope_tempo | slope_team_lead | slope_team_trail | dev_end |
|---|---|---|---|---|---|---|---|---|
| control | 6.7491 | 0.6468 | 0.5626 | -3.5277 | 0.4558 | 0.0652 | 0.1625 | 6.6996 |
| GT1 (C-A) | 6.9604 (+0.2113; 0.0116) | 0.6760 (+0.0292; 0.0676) | 0.8532 (+0.2905; 0.1405) | -0.7695 (+2.7582; 9.1678) | 0.0125 (-0.4433; 0.0401) | 0.0853 (+0.0201; 0.0900) | -0.0390 (-0.2015; 0.1920) | 6.8623 (+0.1627; 0.0199) |
| GT0s1 (C-B) | 6.7491 (+0.0000; 0.0000) | 0.6468 (+0.0000; 0.0000) | 0.5626 (+0.0000; 0.0000) | -3.5277 (+0.0000; 0.0000) | 0.4558 (+0.0000; 0.0000) | 0.0652 (+0.0000; 0.0000) | 0.1625 (+0.0000; 0.0000) | 6.6996 (+0.0000; 0.0000) |
| GT2 (C-C) | 6.7376 (-0.0115; 0.0043) | 0.6423 (-0.0045; 0.0034) | 0.5626 (-0.0000; 0.0000) | -3.2346 (+0.2931; 0.7207) | 0.4559 (+0.0001; 0.0005) | 0.0652 (-0.0000; 0.0000) | 0.1518 (-0.0107; 0.0048) | 6.6668 (-0.0328; 0.0073) |

**Block C, F1, period 2 (selection)** (69,410 rows, 4,777 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | dev | cal | slope_lead | slope_trail | slope_tempo | slope_team_lead | slope_team_trail | dev_end |
|---|---|---|---|---|---|---|---|---|
| control | 6.7483 | 2.3193 | 0.4432 | 0.5054 | 0.3804 | 0.4461 | 0.1066 | 6.7065 |
| GT1 (C-A) | 6.6248 (-0.1234; 0.0185) | 0.3643 (-1.9551; 0.0694) | 0.9457 (+0.5025; 0.0524) | 0.8931 (+0.3877; 0.0449) | 0.0474 (-0.3331; 0.1082) | 0.5486 (+0.1025; 0.2375) | -0.0218 (-0.1284; 0.1037) | 6.4788 (-0.2277; 0.0238) |
| GT0s1 (C-B) | 6.7483 (+0.0000; 0.0000) | 2.3193 (+0.0000; 0.0000) | 0.4432 (+0.0000; 0.0000) | 0.5054 (+0.0000; 0.0000) | 0.3804 (+0.0000; 0.0000) | 0.4461 (+0.0000; 0.0000) | 0.1066 (+0.0000; 0.0000) | 6.7065 (+0.0000; 0.0000) |
| GT2 (C-C) | 6.7483 (+0.0000; 0.0000) | 2.3193 (-0.0000; 0.0000) | 0.4432 (+0.0000; 0.0000) | 0.5054 (-0.0000; 0.0000) | 0.3804 (-0.0000; 0.0000) | 0.4461 (+0.0000; 0.0000) | 0.1066 (+0.0000; 0.0000) | 6.7065 (+0.0000; 0.0000) |

**Block C, F1, period 1 (reported only)** (62,307 rows, 4,777 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | dev | cal | slope_lead | slope_trail | slope_tempo | slope_team_lead | slope_team_trail | dev_end |
|---|---|---|---|---|---|---|---|---|
| control | 6.7824 | 0.6423 | 0.5558 | 0.8862 | 0.5324 | 0.1169 | -0.0752 | 6.7777 |
| GT1 (C-A) | 6.9701 (+0.1877; 0.0143) | 0.7099 (+0.0676; 0.0787) | 0.7183 (+0.1625; 0.1172) | -2.5213 (-3.4076; 6.2828) | -0.0126 (-0.5450; 0.0540) | 0.0775 (-0.0393; 0.0897) | -0.0655 (+0.0097; 0.1956) | 6.9014 (+0.1238; 0.0224) |
| GT0s1 (C-B) | 6.7824 (+0.0000; 0.0000) | 0.6423 (+0.0000; 0.0000) | 0.5558 (+0.0000; 0.0000) | 0.8862 (+0.0000; 0.0000) | 0.5324 (+0.0000; 0.0000) | 0.1169 (+0.0000; 0.0000) | -0.0752 (+0.0000; 0.0000) | 6.7777 (+0.0000; 0.0000) |
| GT2 (C-C) | 6.7636 (-0.0188; 0.0062) | 0.6415 (-0.0008; 0.0034) | 0.5558 (+0.0000; 0.0000) | 0.8315 (-0.0547; 0.2476) | 0.5325 (+0.0002; 0.0004) | 0.1169 (+0.0000; 0.0000) | -0.0820 (-0.0068; 0.0034) | 6.7327 (-0.0450; 0.0091) |

Decision block C: winners [], selected **None**. Rule: s15.A.4: beats control by > 1 floor on the F2 primary, all mandatory slopes hold, F1 keeps the sign; simplest winner whose primary is within one floor of the best.
- GT0s1: beats control by > 1 floor {'dev': False, 'cal': False}; slopes hold {'slope_lead': True, 'slope_trail': True, 'slope_tempo': True, 'slope_team_lead': True, 'slope_team_trail': True}; F1 keeps sign {'dev': True, 'cal': True}; floors {'dev': None, 'cal': None}; WINS False (noise-floor arm, never selectable)
- GT1: beats control by > 1 floor {'dev': True, 'cal': True}; slopes hold {'slope_lead': True, 'slope_trail': True, 'slope_tempo': False, 'slope_team_lead': True, 'slope_team_trail': True}; F1 keeps sign {'dev': True, 'cal': True}; floors {'dev': 4.831056914629298, 'cal': 29.295775850163075}; WINS False
- GT2: beats control by > 1 floor {'dev': True, 'cal': False}; slopes hold {'slope_lead': True, 'slope_trail': True, 'slope_tempo': True, 'slope_team_lead': True, 'slope_team_trail': False}; F1 keeps sign {'dev': False, 'cal': True}; floors {'dev': 3.757691810302721, 'cal': 0.07083559550342443}; WINS False

**Block E, F2, period 2 (selection)** (78,933 rows, 5,445 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | mll | slope_tov_lead | slope_team_tov_lead |
|---|---|---|---|
| control | 1.4865 | 1.1772 | 0.6437 |
| EV0s1 (E-A) | 1.4854 (-0.0011; 0.0004) | 1.1620 (-0.0152; 0.0326) | 0.6818 (+0.0381; 0.0186) |
| EV1 (E-B) | 1.4854 (-0.0011; 0.0004) | 1.1620 (-0.0152; 0.0326) | 0.6818 (+0.0381; 0.0186) |

**Block E, F2, period 1 (reported only)** (70,542 rows, 5,445 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | mll | slope_tov_lead | slope_team_tov_lead |
|---|---|---|---|
| control | 1.5655 | -0.1332 | 1.0275 |
| EV0s1 (E-A) | 1.5658 (+0.0002; 0.0002) | -0.2857 (-0.1525; 0.4723) | 1.0140 (-0.0136; 0.0106) |
| EV1 (E-B) | 1.5658 (+0.0002; 0.0002) | -0.2857 (-0.1525; 0.4723) | 1.0140 (-0.0136; 0.0106) |

**Block E, F1, period 2 (selection)** (75,204 rows, 5,178 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | mll | slope_tov_lead | slope_team_tov_lead |
|---|---|---|---|
| control | 1.4975 | 0.9193 | 0.5068 |
| EV0s1 (E-A) | 1.4962 (-0.0013; 0.0004) | 0.8918 (-0.0276; 0.0274) | 0.5352 (+0.0284; 0.0125) |
| EV1 (E-B) | 1.4962 (-0.0013; 0.0004) | 0.8918 (-0.0276; 0.0274) | 0.5352 (+0.0284; 0.0125) |

**Block E, F1, period 1 (reported only)** (67,442 rows, 5,178 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | mll | slope_tov_lead | slope_team_tov_lead |
|---|---|---|---|
| control | 1.5655 | 0.0407 | 0.8693 |
| EV0s1 (E-A) | 1.5661 (+0.0006; 0.0003) | 0.6681 (+0.6273; 0.6817) | 0.8580 (-0.0113; 0.0072) |
| EV1 (E-B) | 1.5661 (+0.0006; 0.0003) | 0.6681 (+0.6273; 0.6817) | 0.8580 (-0.0113; 0.0072) |

Decision block E: winners ['EV1'], selected **EV1**. Rule: s15.A.4: beats control by > 1 floor on the F2 primary, all mandatory slopes hold, F1 keeps the sign; simplest winner whose primary is within one floor of the best.
- EV0s1: beats control by > 1 floor {'mll': True}; slopes hold {'slope_tov_lead': True}; F1 keeps sign {'mll': True}; floors {'mll': 2.764288017988222}; WINS False (noise-floor arm, never selectable)
- EV1: beats control by > 1 floor {'mll': True}; slopes hold {'slope_tov_lead': True}; F1 keeps sign {'mll': True}; floors {'mll': 2.764288017988222}; WINS True

**Block F, F2, period 2 (selection)** (45,353 rows, 5,593 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | mll | bll | ek_f45 | ek_f68 | ek_f9 |
|---|---|---|---|---|---|
| control | 0.0948 | 0.0680 | 0.1482 | 0.0217 | 0.0227 |
| LF0s1 (F-A) | 0.0599 (-0.0349; 0.0036) | 0.0519 (-0.0161; 0.0012) | 0.2028 (+0.0546; 0.0054) | 0.0029 (-0.0188; 0.0002) | 0.0028 (-0.0199; 0.0002) |
| LF2 (F-B) | 0.0599 (-0.0349; 0.0036) | 0.0519 (-0.0161; 0.0012) | 0.2028 (+0.0546; 0.0054) | 0.0029 (-0.0188; 0.0002) | 0.0028 (-0.0199; 0.0002) |
| LF1 (F-C) | 0.0796 (-0.0152; 0.0011) | 0.0519 (-0.0161; 0.0012) | 0.1786 (+0.0304; 0.0042) | 0.0029 (-0.0188; 0.0002) | 0.0028 (-0.0199; 0.0002) |

LF2 vs LF1: delta mll -0.0197, floor 0.0034 (5.9 floors).

**Block F, F1, period 2 (selection)** (45,448 rows, 5,553 games). Cells: value (delta vs control; floor = 2 x paired game-block bootstrap SE).

| arm (code) | mll | bll | ek_f45 | ek_f68 | ek_f9 |
|---|---|---|---|---|---|
| control | 0.0906 | 0.0681 | 0.1565 | 0.0247 | 0.0211 |
| LF0s1 (F-A) | 0.0605 (-0.0301; 0.0033) | 0.0531 (-0.0150; 0.0010) | 0.2024 (+0.0459; 0.0047) | 0.0029 (-0.0218; 0.0003) | 0.0031 (-0.0181; 0.0002) |
| LF2 (F-B) | 0.0605 (-0.0301; 0.0033) | 0.0531 (-0.0150; 0.0010) | 0.2024 (+0.0459; 0.0047) | 0.0029 (-0.0218; 0.0003) | 0.0031 (-0.0181; 0.0002) |
| LF1 (F-C) | 0.0760 (-0.0146; 0.0010) | 0.0531 (-0.0150; 0.0010) | 0.1786 (+0.0222; 0.0037) | 0.0029 (-0.0218; 0.0003) | 0.0031 (-0.0181; 0.0002) |

LF2 vs LF1: delta mll -0.0155, floor 0.0031 (5.0 floors).

Decision block F: winners ['LF1', 'LF2'], selected **LF2**. Rule: s29.2: simplest arm beating LF0 by > 1 floor on F2, same sign on F1; LF2 over LF1 only if LF2 beats LF1 by > 1 floor.
- LF0s1: beats control by > 1 floor {'mll': True}; slopes hold {'d_ek_falls_4-5_to_6-8': True}; F1 keeps sign {'mll': True}; floors {'mll': 9.709384929355679}; WINS False (noise-floor arm, never selectable)
- LF1: beats control by > 1 floor {'mll': True}; slopes hold {'d_ek_falls_4-5_to_6-8': True}; F1 keeps sign {'mll': True}; floors {'mll': 13.265076330293024}; WINS True
- LF2: beats control by > 1 floor {'mll': True}; slopes hold {'d_ek_falls_4-5_to_6-8': True}; F1 keeps sign {'mll': True}; floors {'mll': 9.709384929355679}; WINS True

### Block C per-cell calibration, F2 period 2 (actual vs model mean first-chance seconds, E[min(T, R)])

| cell | n | actual | GT0 (served) | GT2 | GT1 |
|---|---:|---:|---:|---:|---:|
| leader 8-12 (180,240] | 1663 | 19.71 | 17.35 | 17.35 | 20.00 |
| leader 8-12 (60,120] | 1859 | 15.06 | 16.95 | 16.95 | 14.65 |
| leader 8-12 (0,30] | 680 | 3.94 | 7.20 | 7.20 | 3.84 |
| leader 13+ (180,240] | 3146 | 18.07 | 17.05 | 17.05 | 18.06 |
| leader 13+ (60,120] | 3423 | 17.36 | 16.92 | 16.92 | 17.41 |
| leader 13+ (30,60] | 1902 | 17.08 | 9.02 | 9.02 | 17.19 |
| leader 13+ (0,30] | 322 | 9.73 | 7.70 | 7.70 | 8.99 |
| trailer 13+ (180,240] | 3532 | 15.39 | 17.92 | 17.91 | 14.93 |
| trailer 13+ (30,60] | 2097 | 11.69 | 9.36 | 9.34 | 11.16 |
| tied (30,60] | 281 | 20.89 | 11.42 | 11.42 | 20.12 |
| tied (0,30] | 379 | 10.91 | 6.79 | 6.79 | 8.21 |

The served K2 law is far off in the (30,60] bucket for leaders and tied offences (9.0 s vs 17.1 s for 13+ leaders)
and too slow for 13+ trailers early in the window; GT1 fixes the cells (cal 2.29 -> 0.34 s) but, being a pooled
cell law, flattens the within-window tempo response (gate (b) slope 0.36 -> -0.01).

**Finding (the deviance primary is fragile).** 142 of 74,164 F2 period-2 window rows (0.19%; F1 0.28%) get zero
probability from the served K2 law and hit the 1e-12 clip; they contribute 0.106 (F1 0.156) of GT0's 6.690
deviance. Excluding them the served law has the BEST deviance (GT0 6.597, GT1 6.611, GT2 6.618 on F2). So the
`dev` wins of GT2 (+3.8 floors) and part of GT1's come from smoothing the served law's zero-mass bins, not from
state content. No decision changes (neither arm wins), but a later clock round scoring deviance should report the
clipped-row share.

### Block E calibration by role-band, F2 period 2 (actual / EV0 / EV1)

| role-band | n | TOV | 3PA | FT trip |
|---|---:|---|---|---|
| leader 0-3 | 7155 | 0.134 / 0.129 / 0.131 | 0.164 / 0.163 / 0.161 | 0.370 / 0.374 / 0.375 |
| leader 4-7 | 8552 | 0.136 / 0.127 / 0.135 | 0.133 / 0.140 / 0.133 | 0.445 / 0.444 / 0.450 |
| leader 8-12 | 7672 | 0.165 / 0.154 / 0.161 | 0.144 / 0.150 / 0.141 | 0.397 / 0.399 / 0.406 |
| leader 13+ | 12901 | 0.176 / 0.181 / 0.183 | 0.271 / 0.259 / 0.259 | 0.178 / 0.176 / 0.174 |
| trailer 13+ | 15253 | 0.116 / 0.112 / 0.111 | 0.326 / 0.319 / 0.321 | 0.168 / 0.172 / 0.168 |
| tied | 2222 | 0.121 / 0.138 / 0.143 | 0.234 / 0.244 / 0.246 | 0.162 / 0.168 / 0.153 |

EV1's offline gain comes from leader 4-7 / 8-12 TOV and the tied FT mix; on the targeted cell (leader 13+ TOV) it
moves AWAY offline too (0.181 -> 0.183 vs 0.176 actual). Its training-season ratio for that cell is about 1.0 (the
served law is calibrated in-sample), so EV1 cannot address the sim's 13+ leader TOV overshoot (D3: 22.1% vs 19.0%),
which is a state-feed defect, not an offline possession_outcome error.

### Block F expected K by cell, F2 (actual / LF0 / LF1 / LF2; U = < 200 test rows, UNDERPOWERED)

| defence cell | n | actual | LF0 | LF1 | LF2 |
|---|---:|---:|---:|---:|---:|
| trailing 1-3, (0,30], 4-5 fouls | 64 U | 1.000 | 0.183 | 0.716 | 0.886 |
| trailing 1-3, (30,60], 4-5 | 88 U | 0.545 | 0.189 | 0.363 | 0.460 |
| trailing 4-6, (30,60], 4-5 | 51 U | 0.824 | 0.205 | 0.545 | 0.662 |
| trailing 1-3, (0,30], 6-8 | 604 | 0.003 | 0.029 | 0.010 | 0.010 |
| trailing 1-3, (0,30], 9+ | 1148 | 0.006 | 0.032 | 0.005 | 0.005 |

E[K] by defence foul count, all window rows (actual): 0-3 0.381, 4-5 0.211, 6-8 0.002, 9+ 0.0015; LF2 0.203 /
0.003 / 0.003 at 4-5 / 6-8 / 9+ (falls as in the data; (d) holds).

## 5. Closed loop (500 verified fold-2 games x 25 paired seeds; served v3 control; four floor draws)

Floor per line = max(2 x SD of the four control draws, 2 x paired game-bootstrap SE, 200 draws seed 12345). Cells:
arm value (floors moved toward the target) for primaries; arm value (improvement / floor, pass) for vetoes.
Real targets are the sample's (483 of 500 games have a pbp path ending at the verified final; the sample's OT rate
is 6.80% vs 5.57% season-wide, so the tie target here is the sample's).

| line | target (real, sample) | control | LF2 (F) | EV1 (E) | EV1+LF2 (comp.) | GT1 DIAG (not eligible) |
|---|---:|---:|---:|---:|---:|---:|
| PRIMARY MD4_h2_last4_poss (C) | 15.4286 | 16.7499 | 16.7651 (-0.20 fl toward) | 16.7705 (-0.27 fl toward) | 16.7840 (-0.45 fl toward) | 16.7487 (+0.02 fl toward) |
| PRIMARY MD5_pts13_last2 (C) | 7.1162 | 9.6902 | 9.7121 (-0.11 fl toward) | 9.6694 (+0.10 fl toward) | 9.6917 (-0.01 fl toward) | 9.0172 (+3.34 fl toward) |
| PRIMARY tie_rate (F) | 0.0680 | 0.0362 | 0.0365 (+0.09 fl toward) | 0.0372 (+0.29 fl toward) | 0.0371 (+0.27 fl toward) | 0.0423 (+1.63 fl toward) |
| PRIMARY ltov_r4_13p (E) | 0.2015 | 0.2224 | 0.2225 (-0.05 fl toward) | 0.2282 (-1.72 fl toward) | 0.2278 (-1.61 fl toward) | 0.2149 (+2.21 fl toward) |
| PRIMARY ltov_r2_13p (E) | 0.2062 | 0.2535 | 0.2540 (-0.06 fl toward) | 0.2607 (-0.96 fl toward) | 0.2604 (-0.93 fl toward) | 0.2481 (+0.74 fl toward) |
| veto G1_poss_mean | 68.3108 | 68.8826 | 68.9058 (-0.0232/0.0843, pass) | 68.9020 (-0.0194/0.0843, pass) | 68.9125 (-0.0299/0.0843, pass) | 68.9428 (-0.0602/0.0843, pass) |
| veto G1_poss_sd | 5.1951 | 5.6583 | 5.6893 (-0.0310/0.1144, pass) | 5.6568 (+0.0015/0.1144, pass) | 5.6695 (-0.0112/0.1144, pass) | 5.6097 (+0.0486/0.1144, pass) |
| veto half1_share | 0.4696 | 0.4693 | 0.4691 (-0.0002/0.0014, pass) | 0.4692 (-0.0002/0.0014, pass) | 0.4691 (-0.0003/0.0014, pass) | 0.4690 (-0.0004/0.0014, pass) |
| veto half2_share | 0.5179 | 0.5247 | 0.5247 (+0.0000/0.0009, pass) | 0.5248 (-0.0000/0.0009, pass) | 0.5248 (-0.0001/0.0009, pass) | 0.5242 (+0.0006/0.0009, pass) |
| veto G5_margin_sd_ratio | 1.0000 | 0.9778 | 0.9792 (+0.0015/0.0173, pass) | 0.9756 (-0.0022/0.0173, pass) | 0.9759 (-0.0018/0.0173, pass) | 0.9792 (+0.0014/0.0173, pass) |
| veto G5_total_sd_ratio | 1.0000 | 0.9248 | 0.9294 (+0.0046/0.0116, pass) | 0.9220 (-0.0028/0.0116, pass) | 0.9234 (-0.0014/0.0116, pass) | 0.9350 (+0.0102/0.0116, pass) |
| veto G9_margin_bias | 0.0000 | 0.0196 | 0.0108 (+0.0088/0.1832, pass) | 0.0129 (+0.0067/0.1832, pass) | 0.0022 (+0.0174/0.1832, pass) | -0.0059 (+0.0137/0.1832, pass) |
| veto G9_margin_slope | 1.0000 | 1.0015 | 1.0036 (-0.0021/0.0121, pass) | 1.0034 (-0.0019/0.0121, pass) | 1.0040 (-0.0025/0.0121, pass) | 1.0063 (-0.0048/0.0121, pass) |
| veto h1buz_ppp | 0.8333 | 0.9258 | 0.9258 (+0.0000/0.0131, pass) | 0.9258 (+0.0000/0.0131, pass) | 0.9258 (+0.0000/0.0131, pass) | 0.9258 (+0.0000/0.0131, pass) |
| veto D1_home_margin_bias | 0.0000 | 0.2290 | 0.2152 (+0.0138/0.2003, pass) | 0.2168 (+0.0122/0.2003, pass) | 0.2022 (+0.0268/0.2003, pass) | 0.1886 (+0.0404/0.2003, pass) |
| veto D1_efg_gap | 0.0305 | 0.0312 | 0.0312 (+0.0001/0.0014, pass) | 0.0313 (-0.0001/0.0014, pass) | 0.0313 (-0.0000/0.0014, pass) | 0.0310 (+0.0003/0.0014, pass) |
| veto D2_poss_on_env | 0.1240 | -0.0124 | -0.0131 (-0.0007/0.0062, pass) | -0.0131 (-0.0007/0.0062, pass) | -0.0136 (-0.0012/0.0062, pass) | -0.0079 (+0.0045/0.0062, pass) |
| veto D2_poss_on_gap | 0.0852 | 0.0076 | 0.0062 (-0.0014/0.0240, pass) | 0.0074 (-0.0002/0.0240, pass) | 0.0066 (-0.0010/0.0240, pass) | 0.0022 (-0.0055/0.0240, pass) |
| veto D2_total_on_env | 0.6676 | 0.4066 | 0.4077 (+0.0012/0.0207, pass) | 0.4094 (+0.0028/0.0207, pass) | 0.4075 (+0.0009/0.0207, pass) | 0.4147 (+0.0081/0.0207, pass) |
| veto D2_leadchg_on_spread | -0.0646 | -0.0979 | -0.0978 (+0.0001/0.0119, pass) | -0.0977 (+0.0002/0.0119, pass) | -0.0977 (+0.0002/0.0119, pass) | -0.0975 (+0.0004/0.0119, pass) |
| veto tie_rate | 0.0680 | 0.0362 | 0.0365 (+0.0003/0.0036, pass) | 0.0372 (+0.0010/0.0036, pass) | 0.0371 (+0.0010/0.0036, pass) | 0.0423 (+0.0062/0.0038, pass) |
| veto ltov_r4_812 | 0.1852 | 0.1827 | 0.1821 (-0.0006/0.0042, pass) | 0.1872 (+0.0005/0.0042, pass) | 0.1869 (+0.0009/0.0042, pass) | 0.1784 (-0.0043/0.0042, FAIL) |
| veto ltov_r2_812 | 0.1791 | 0.1744 | 0.1744 (-0.0001/0.0072, pass) | 0.1819 (+0.0018/0.0072, pass) | 0.1822 (+0.0016/0.0072, pass) | 0.1678 (-0.0067/0.0072, pass) |
| veto window_poss_by_k (k=0..6) | all k cells < 60 real games: UNDERPOWERED | -- | pass (underpowered) | pass (underpowered) | pass (underpowered) | pass (underpowered) |
| **MD1** | gap -> 0 | gap 15.49 (sim -12.86 vs real -28.35) | gap 15.81 vs ctrl 15.49, fl 2.49 **FAIL** | gap 14.45 vs ctrl 15.49, fl 2.49 **FAIL** | gap 14.56 vs ctrl 15.49, fl 2.49 **FAIL** | gap 14.80 vs ctrl 15.49, fl 2.49 **FAIL** |
| **MD2** | 1.0 | Q1-Q5 1.071 1.181 0.958 1.082 0.941 | Q1-Q5 1.069 1.181 0.960 1.084 0.944; Q1 falls False, any rise False **FAIL** | Q1-Q5 1.063 1.188 0.954 1.077 0.939; Q1 falls False, any rise False **FAIL** | Q1-Q5 1.064 1.187 0.953 1.080 0.938; Q1 falls False, any rise False **FAIL** | Q1-Q5 1.093 1.178 0.959 1.088 0.932; Q1 falls False, any rise False **FAIL** |
| **MD3** | 0.0680 | 0.0362 | 0.0365 (+0.09 fl) **FAIL** | 0.0372 (+0.29 fl) **FAIL** | 0.0371 (+0.27 fl) **FAIL** | 0.0423 (+1.63 fl) **PASS** |
| **MD4** | 15.429 | 16.750 | 16.7651 (-0.20 fl) **FAIL** | 16.7705 (-0.27 fl) **FAIL** | 16.7840 (-0.45 fl) **FAIL** | 16.7487 (+0.02 fl) **FAIL** |
| **MD5** | 7.116 | 9.690 | 9.7121 (-0.11 fl) **FAIL** | 9.6694 (+0.10 fl) **FAIL** | 9.6917 (-0.01 fl) **FAIL** | 9.0172 (+3.34 fl) **PASS** |
| **MD6** | not rise | 1.009 | ratio 1.010 (ctrl 1.009, fl 0.019); Var036 140.1 (ctrl 140.1); Var last4 19.80 (ctrl 19.73) **PASS** | ratio 1.006 (ctrl 1.009, fl 0.019); Var036 140.1 (ctrl 140.1); Var last4 19.97 (ctrl 19.73) **PASS** | ratio 1.006 (ctrl 1.009, fl 0.019); Var036 140.1 (ctrl 140.1); Var last4 19.96 (ctrl 19.73) **PASS** | ratio 1.010 (ctrl 1.009, fl 0.019); Var036 140.1 (ctrl 140.1); Var last4 20.03 (ctrl 19.73) **PASS** |
| candidate? | | | no (fails: MD1, MD2, MD3, MD4, MD5; primary moves False) | no (fails: MD1, MD2, MD3, MD4, MD5; primary moves False) | no (fails: MD1, MD2, MD3, MD4, MD5; primary moves False) | no (fails: ltov_r4_812, MD1, MD2, MD4; primary moves False) |

Reported (floors):

| line | control | LF2 (F) | EV1 (E) | EV1+LF2 (comp.) | GT1 DIAG (not eligible) |
|---|---:|---:|---:|---:|---:|
| P1 | 0.0572 | 0.0560 (d -0.0012, fl 0.0060) | 0.0580 (d +0.0008, fl 0.0060) | 0.0575 (d +0.0003, fl 0.0060) | 0.0598 (d +0.0026, fl 0.0060) |
| P0_over_P1 | 0.6322 | 0.6514 (d +0.0193, fl 0.1189) | 0.6414 (d +0.0092, fl 0.1189) | 0.6453 (d +0.0132, fl 0.1189) | 0.7072 (d +0.0751, fl 0.1189) |
| lead13_bonus | 0.8838 | 0.8926 (d +0.0088, fl 0.0054) | 0.8865 (d +0.0028, fl 0.0052) | 0.8944 (d +0.0106, fl 0.0073) | 0.8853 (d +0.0015, fl 0.0052) |
| lead46_bonus | 0.8957 | 0.9077 (d +0.0121, fl 0.0055) | 0.8964 (d +0.0007, fl 0.0042) | 0.9092 (d +0.0135, fl 0.0065) | 0.8942 (d -0.0015, fl 0.0064) |
| lead7p_bonus | 0.8908 | 0.9003 (d +0.0095, fl 0.0073) | 0.8901 (d -0.0007, fl 0.0073) | 0.9004 (d +0.0096, fl 0.0073) | 0.8913 (d +0.0005, fl 0.0073) |
| trail_fouls_13 | 0.5160 | 0.5178 (d +0.0018, fl 0.0082) | 0.5183 (d +0.0023, fl 0.0076) | 0.5208 (d +0.0048, fl 0.0103) | 0.5381 (d +0.0221, fl 0.0113) |
| trail_fouls_46 | 0.6567 | 0.6677 (d +0.0110, fl 0.0144) | 0.6686 (d +0.0119, fl 0.0144) | 0.6817 (d +0.0250, fl 0.0144) | 0.6744 (d +0.0176, fl 0.0144) |
| trail_fouls_7p | 0.4410 | 0.4338 (d -0.0072, fl 0.0040) | 0.4429 (d +0.0019, fl 0.0032) | 0.4360 (d -0.0050, fl 0.0049) | 0.4443 (d +0.0033, fl 0.0048) |
| lsec_r4_812 | 13.7424 | 13.7225 (d -0.0199, fl 0.1942) | 13.6938 (d -0.0486, fl 0.1942) | 13.6814 (d -0.0610, fl 0.1942) | 14.7974 (d +1.0550, fl 0.1942) |
| lsec_r4_13p | 14.9719 | 14.9494 (d -0.0226, fl 0.1460) | 14.9686 (d -0.0034, fl 0.1460) | 14.9472 (d -0.0248, fl 0.1460) | 16.4426 (d +1.4707, fl 0.1460) |
| lsec_r2_812 | 10.7537 | 10.7282 (d -0.0255, fl 0.1457) | 10.6476 (d -0.1060, fl 0.1457) | 10.6228 (d -0.1309, fl 0.1457) | 10.6611 (d -0.0925, fl 0.1518) |
| lsec_r2_13p | 12.7510 | 12.7216 (d -0.0293, fl 0.0955) | 12.7663 (d +0.0154, fl 0.0955) | 12.7370 (d -0.0140, fl 0.0955) | 14.2669 (d +1.5159, fl 0.1109) |
| share_st13 | 0.4125 | 0.4125 (d +0.0000, fl 0.0064) | 0.4123 (d -0.0002, fl 0.0064) | 0.4123 (d -0.0002, fl 0.0064) | 0.4151 (d +0.0026, fl 0.0064) |
| G9_total_bias | -0.6930 | -0.6217 (d +0.0714, fl 0.2234) | -0.6417 (d +0.0514, fl 0.2234) | -0.6091 (d +0.0839, fl 0.2234) | -0.5768 (d +0.1162, fl 0.2234) |
| G9_total_bias_H1 | -0.3624 | -0.3624 (d +0.0000, fl 0.2873) | -0.3624 (d +0.0000, fl 0.2873) | -0.3624 (d +0.0000, fl 0.2873) | -0.3624 (d +0.0000, fl 0.2873) |
| G9_total_bias_H2OT | -0.3468 | -0.2761 (d +0.0707, fl 0.1968) | -0.2981 (d +0.0488, fl 0.1968) | -0.2669 (d +0.0799, fl 0.1968) | -0.2178 (d +0.1290, fl 0.1968) |
| d13_ppp | 1.1281 | 1.1345 (d +0.0064, fl 0.0434) | 1.1305 (d +0.0024, fl 0.0434) | 1.1293 (d +0.0012, fl 0.0434) | 1.1088 (d -0.0193, fl 0.0442) |
| d13_3pa | 0.3131 | 0.3160 (d +0.0029, fl 0.0164) | 0.3067 (d -0.0064, fl 0.0164) | 0.3120 (d -0.0011, fl 0.0164) | 0.3141 (d +0.0010, fl 0.0200) |
| MD6_varlast4_sim | 19.7336 | 19.7971 (d +0.0634, fl 0.4743) | 19.9678 (d +0.2341, fl 0.4743) | 19.9568 (d +0.2232, fl 0.4743) | 20.0286 (d +0.2949, fl 0.4743) |
| MD1_cov_sim | -12.8580 | -12.7539 (d +0.1041, fl 2.1316) | -13.5713 (d -0.7133, fl 2.1316) | -13.5124 (d -0.6544, fl 2.1316) | -13.7318 (d -0.8737, fl 2.1316) |
| MD1_cov_real | -28.3504 | -28.5682 (d -0.2178, fl 1.4636) | -28.0164 (d +0.3340, fl 1.4636) | -28.0697 (d +0.2807, fl 1.4636) | -28.5319 (d -0.1815, fl 1.4636) |

Seed-count check (2 x SD of the four control draws vs |control - target|): MD4_h2_last4_poss 0.0756 vs 1.3213 resolved; MD5_pts13_last2 0.2014 vs 2.5741 resolved; ltov_r4_13p 0.0034 vs 0.0208 resolved; ltov_r2_13p 0.0074 vs 0.0473 resolved; tie_rate 0.0036 vs 0.0318 resolved; MD1_gap 2.4886 vs 15.4924 resolved; MD2_q1 0.0593 vs 0.0711 resolved. All resolved: True.

**Seed-count check (s15.0): every deciding line is resolved at 25 seeds; no 50-seed rerun was needed.**

## 6. D3 / D4 rerun on each run's taps (per state, per possession type)

Half 2, last 2 minutes, reference at 2:00 (real / sim, diff; `*` CI excludes 0):

| run | 13+ leader sec/poss | 13+ leader TOV % | 13+ last-2-min pts/game | M-end H2 36-40 (real 15.418) | tie (real 6.83%) | P(1) (real 3.31%) | leader 1-3 in bonus % | trailing fouls / leader poss (1-3, 4-6, 7+; real pbp 0.642, 0.722, 0.432) | down 1-3 < 30 s PPP (real 0.984) |
|---|---|---|---|---|---|---|---|---|---|
| control | 16.14 / 12.75 (3.39*) | 20.6 / 25.4 | 7.12 / 9.69 | 16.749 | 3.61% | 5.68% | 92.9 / 88.4 | 0.475, 0.610, 0.396 | 1.128 |
| LF2 | 16.14 / 12.72 | 20.6 / 25.4 | 7.12 / 9.71 | 16.764 | 3.64% | 5.57% | 92.9 / 89.3 | 0.477, 0.622, 0.390 | 1.135 |
| EV1 | 16.14 / 12.77 | 20.6 / 26.1 | 7.12 / 9.67 | 16.770 | 3.71% | 5.76% | 92.9 / 88.7 | 0.479, 0.623, 0.398 | 1.125 |
| EV1+LF2 | 16.14 / 12.74 | 20.6 / 26.0 | 7.12 / 9.69 | 16.783 | 3.70% | 5.72% | 92.9 / 89.4 | 0.481, 0.637, 0.392 | 1.123 |
| GT1 (diag) | 16.14 / 14.27 (1.87*) | 20.6 / 24.8 | 7.12 / 9.02 | 16.748 | 4.26% | 5.95% | 92.9 / 88.5 | 0.510, 0.642, 0.402 | 1.108 |

Possession-class mix, H2 last 4:00 (real 2PA 30.8 / 3PA 25.3 / FT 28.3 / TOV 15.5%): control 31.1 / 25.4 / 27.6 /
15.9; LF2 31.0 / 25.4 / 27.8 / 15.9; EV1 31.2 / 25.1 / 27.7 / 16.0; GT1 31.0 / 26.2 / 27.3 / 15.5.
OT conversion P(tie | |margin| at 2:00), real / control / LF2 / GT1: 1-2 0.190 / 0.161 / 0.160 / 0.163; 3-5 0.147 /
0.076 / 0.077 / 0.097; 6-9 0.048 / 0.013 / 0.015 / 0.026; state 0 underpowered (14 real games).
Full tables: `results/endgame/d34/<tag>_d3.md`, `_d4.md` (gitignored; regenerate with the commands in section 2).

**Reading.** LF2 moves the bonus state the right way (leader in bonus +0.9 to +1.2 pp, about 2 floors) but the tie
rate does not follow: the conversion gap (P(tie | 3-5) 0.077 vs 0.147) is not owned by the bonus state alone. GT1
(diagnostic, not eligible) is the only arm that moves the decided-game lines: 13+ last-2-min points 9.69 -> 9.02
(+3.3 floors), 13+ leader seconds 12.75 -> 14.27, tie rate 3.6% -> 4.3% (+1.6 floors); it leaves the H2 36-40
possession count unchanged (16.75; slower leaders are offset by faster trailers, 8-12 trailers 10.91 -> 10.09 s)
and fails MD1, MD2 and the 8-12 TOV veto.

## 7. Multi-level evidence (per team quintile, per game)

## Per offence team-strength quintile, H2 last 4:00 (leaders / trailers by 8+), last 2:00 bonus (leading 1-6)

| run | Q | lead poss | sec/poss lead (real / sim) | TOV lead (real / sim) | trail poss | sec/poss trail (real / sim) | in-bonus lead 1-6 (real / sim) |
|---|---|---:|---|---|---:|---|---|
| eg_ctrl_o0_s25 | Q1 | 198 | 15.25 / 14.01 | 0.278 / 0.232 | 697 | 14.34 / 14.23 | 0.947 / 0.890 |
| eg_ctrl_o0_s25 | Q2 | 283 | 17.61 / 14.23 | 0.230 / 0.221 | 507 | 13.32 / 14.15 | 0.983 / 0.904 |
| eg_ctrl_o0_s25 | Q3 | 374 | 17.32 / 14.17 | 0.193 / 0.224 | 385 | 14.04 / 14.16 | 0.915 / 0.902 |
| eg_ctrl_o0_s25 | Q4 | 493 | 16.29 / 14.29 | 0.176 / 0.211 | 306 | 14.03 / 14.39 | 0.920 / 0.885 |
| eg_ctrl_o0_s25 | Q5 | 492 | 14.95 / 14.55 | 0.161 / 0.195 | 261 | 13.75 / 14.11 | 0.920 / 0.870 |
| eg_LF2_o0_s25 | Q1 | 198 | 15.25 / 13.95 | 0.278 / 0.231 | 697 | 14.34 / 14.23 | 0.947 / 0.904 |
| eg_LF2_o0_s25 | Q2 | 283 | 17.61 / 14.22 | 0.230 / 0.222 | 507 | 13.32 / 14.13 | 0.983 / 0.917 |
| eg_LF2_o0_s25 | Q3 | 374 | 17.32 / 14.17 | 0.193 / 0.224 | 385 | 14.04 / 14.13 | 0.915 / 0.912 |
| eg_LF2_o0_s25 | Q4 | 493 | 16.29 / 14.27 | 0.176 / 0.212 | 306 | 14.03 / 14.40 | 0.920 / 0.891 |
| eg_LF2_o0_s25 | Q5 | 492 | 14.95 / 14.55 | 0.161 / 0.195 | 261 | 13.75 / 14.08 | 0.920 / 0.879 |
| eg_EV1_o0_s25 | Q1 | 198 | 15.25 / 14.04 | 0.278 / 0.240 | 697 | 14.34 / 14.25 | 0.947 / 0.893 |
| eg_EV1_o0_s25 | Q2 | 283 | 17.61 / 14.18 | 0.230 / 0.228 | 507 | 13.32 / 14.15 | 0.983 / 0.908 |
| eg_EV1_o0_s25 | Q3 | 374 | 17.32 / 14.19 | 0.193 / 0.228 | 385 | 14.04 / 14.14 | 0.915 / 0.899 |
| eg_EV1_o0_s25 | Q4 | 493 | 16.29 / 14.31 | 0.176 / 0.216 | 306 | 14.03 / 14.42 | 0.920 / 0.884 |
| eg_EV1_o0_s25 | Q5 | 492 | 14.95 / 14.53 | 0.161 / 0.200 | 261 | 13.75 / 14.13 | 0.920 / 0.875 |
| eg_EV1LF2_o0_s25 | Q1 | 198 | 15.25 / 13.98 | 0.278 / 0.239 | 697 | 14.34 / 14.25 | 0.947 / 0.907 |
| eg_EV1LF2_o0_s25 | Q2 | 283 | 17.61 / 14.17 | 0.230 / 0.228 | 507 | 13.32 / 14.12 | 0.983 / 0.916 |
| eg_EV1LF2_o0_s25 | Q3 | 374 | 17.32 / 14.18 | 0.193 / 0.227 | 385 | 14.04 / 14.12 | 0.915 / 0.910 |
| eg_EV1LF2_o0_s25 | Q4 | 493 | 16.29 / 14.29 | 0.176 / 0.216 | 306 | 14.03 / 14.42 | 0.920 / 0.893 |
| eg_EV1LF2_o0_s25 | Q5 | 492 | 14.95 / 14.52 | 0.161 / 0.201 | 261 | 13.75 / 14.11 | 0.920 / 0.885 |
| eg_GT1diag_o0_s25 | Q1 | 198 | 15.25 / 15.45 | 0.278 / 0.226 | 697 | 14.34 / 12.83 | 0.947 / 0.890 |
| eg_GT1diag_o0_s25 | Q2 | 283 | 17.61 / 15.78 | 0.230 / 0.217 | 507 | 13.32 / 12.74 | 0.983 / 0.902 |
| eg_GT1diag_o0_s25 | Q3 | 374 | 17.32 / 15.65 | 0.193 / 0.215 | 385 | 14.04 / 12.69 | 0.915 / 0.901 |
| eg_GT1diag_o0_s25 | Q4 | 493 | 16.29 / 15.72 | 0.176 / 0.206 | 306 | 14.03 / 12.85 | 0.920 / 0.886 |
| eg_GT1diag_o0_s25 | Q5 | 492 | 14.95 / 15.96 | 0.161 / 0.188 | 261 | 13.75 / 12.66 | 0.920 / 0.868 |

## Responsiveness across team quintiles and per-game errors

| run | slope sim on real, lead sec/poss (5 quintiles) | same, trail | per-game H2 poss ending <= 240 s: mean (sim - real) | MAE | corr | per-game last-2:00 points: mean (sim - real) | MAE | corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eg_ctrl_o0_s25 | -0.05 | 0.13 | +1.331 | 2.854 | 0.087 | +1.442 | 4.125 | -0.040 |
| eg_LF2_o0_s25 | -0.04 | 0.15 | +1.346 | 2.866 | 0.080 | +1.475 | 4.129 | -0.029 |
| eg_EV1_o0_s25 | -0.05 | 0.14 | +1.352 | 2.863 | 0.082 | +1.474 | 4.135 | -0.034 |
| eg_EV1LF2_o0_s25 | -0.05 | 0.17 | +1.365 | 2.874 | 0.075 | +1.496 | 4.136 | -0.028 |
| eg_GT1diag_o0_s25 | -0.01 | 0.10 | +1.330 | 2.860 | 0.069 | +1.388 | 4.106 | 0.006 |

Quintile slopes rest on 5 points each and are noisy at this sample size; read as direction.


The leader TOV rate falls with offence strength in reality (0.278 -> 0.161 across quintiles) and much less in the
sim (0.232 -> 0.195 control; no arm changes it): a team-responsiveness shortfall in the late leader event mix that
none of the registered arms targets.

## 8. Not done, and why

- Full-size 5,710 x 200 paired read: not run (no candidate; s15.A.4 reserves it for candidates).
- Fold-1 closed loop (served-equivalent v2 + K2 overlay): not run (confirmatory only, for candidates).
- Parity v10 on a clean `src/`: run anyway for the wiring commit (flags off): PASS bit-identical, 60 x 5.
- Nothing adopted, no default changed, freeze untouched; the scheduled daily passes, results/daily,
  data/processed/ingest and the served inputs were not touched.

## 9. Pointers for the PM (findings, not proposals)

1. The served K2 window law is miscalibrated by cell, worst in the (30,60] bucket (13+ leaders 9.0 vs 17.1 s, tied
   11.4 vs 20.9 s): this is where the D3 surplus lives offline. A pooled cell law fixes the cells but loses the
   tempo response (gate (b)); a cell term that keeps the served row heterogeneity (the EV1-style ratio form, or a
   cell AFT on the served law) was not registered in this round.
2. The deviance primary is dominated by about 0.2-0.3% zero-mass rows of the served clock law (section 4 finding).
3. The tie-rate gap is conversion inside the 3-5 and 6-9 point states and is not closed by the bonus state (LF2).
4. Leader TOV team-strength responsiveness is compressed in the sim (section 7).
