# Full retrain x served stack v2 (fold 2) and fold-1 closed-loop confirmation: gate report (2026-10-05)

AWS operator worker. Nothing adopted. Pre-registrations: `docs/models/engine/experiments.md` section 1 (+1.4) and section 2
(committed a85707f / e840299 before any run). Decision 12 floor = max(2 x draw SD, paired game-bootstrap 95% half-width).

## Provenance (read this first)

Every sim read below was ALREADY RUN on the 2026-10-01 day box (d1001_D_1, d1001_D_2, d1001_D_3) and mirrored to HF
(`results`), but the fold-1 and R8a Decision 12 tables and the report were never produced. Today's box
(spot c7a.48xlarge, i-0f736aabd7746efc8, us-east-2a, 14:17:56Z -> terminate 14:46:28Z, parity v9 PASS bit-identical on
d06f23f) pulled those reads, verified them (`run_meta`: 200 seeds, partial False, `overlay_v2_source_check` / `full_retrain_source_check`
PASS, flags as registered) and ran only the paired-bootstrap tables. No re-simulation, so no new seeds were drawn: re-running F_R / F_T
from scratch would have reproduced FR_box_v1 / FT_box_v2 byte for byte (resumable chain, all stages `.done`).
Clock double-apply: registered check (section 1.3) holds, the chain's clock stage is the L2 refit byte for byte (six S1
pickles, sigma 0.04682263314741274), so L2 is applied once. Fold-1 V1 and V2 were simulated at engine commits 8eca67b and
102c6c6; the commits between them add only default-off flags with parity v9 PASS, and each arm sets its switches explicitly.
Fold-1 arms use `ENGINE_ROTATION_SCHEME=static` (no fold-1 rotation fit; stated deviation in section 2).

Files: `results/d1001D/boot_vs_S2.md`, `boot_R8a_vs_S2.md`, `boot_f1_V2_vs_V1.md` (+ .json), graders in `results/engine_v0/v3full_grade/`
(gitignored; HF `results`). Local table copies are in this report.

## A. Fold 2 (2024-25, 5,705 common games x 200 seeds): retrained stacks vs served v2 (S2)

Targets: slope 1.0, biases 0, G1 67.88 (actual), OREB% .2984, eFG% .5086, FTA/FGA .3295, TOV% .1739, total SD ratio 1.0, h/a corr .228.

| gate line | S2 | FRa (adopted switches) | FTa | verdict FRa / FTa | veto fired? |
|---|---|---|---|---|---|
| G9 slope (primary) | 0.9479 | 0.9398 (-0.9 fl, inside) | 0.9503 (+0.2 fl, inside) | NEUTRAL / NEUTRAL | no |
| G9 total bias | -0.337 | +0.425 (+21 fl, away) | +0.403 (+14 fl, away) | FAIL / FAIL | YES |
| G9 margin bias | -0.261 | -0.144 (toward) | -0.215 (inside) | PASS / PASS | no |
| G4 FTA/FGA | 0.3271 | 0.3657 (+99 fl, away) | 0.3650 (+58 fl, away) | FAIL / FAIL | YES |
| G4 OREB% | 0.2887 | 0.2874 (-7 fl, away) | 0.2874 (-3 fl, away) | FAIL / FAIL | YES |
| G5 total SD ratio | 0.9254 | 0.9152 (-4.4 fl, away) | 0.9164 (-3.0 fl, away) | FAIL / FAIL | YES |
| G5 h/a corr | 0.1260 | 0.1091 (-5.5 fl, away) | 0.1170 (-2.5 fl, away) | FAIL / FAIL | YES |
| G1 possessions mean | 68.823 | 68.820 inside | 68.816 inside | PASS / PASS | no |
| G5 margin SD ratio, G7 OT, G6 home margin | | inside | inside | PASS | no |

L-R (FRa vs S2): LOSE (vetoes fire). L-E3 (FTa vs FRa): FTa beyond floor on h/a corr (+), TOV (away), margin bias (away), inside on slope;
no WIN. Both retrained stacks lose to the incumbent S2.

Labelled diagnostic (section 1.4, `ENGINE_FOUL_JOINT=R8a`, removes the double-applied trip table):

| line | FRa_R8a | FTa_R8a |
|---|---|---|
| FTA/FGA 0.3271 ref | 0.3272 inside | 0.3266 inside |
| G9 total bias (ref -0.337) | -0.179 (toward, +4.4 fl) | -0.202 (toward, +2.5 fl) |
| G9 margin bias (ref -0.261) | -0.086 (toward) | -0.150 (toward) |
| G9 slope (ref 0.9479) | 0.9306 (-1.9 fl, AWAY) | 0.9403 (-0.7 fl, inside) |
| G5 total SD ratio | -2.2 fl away | -1.6 fl away |
| G5 h/a corr | -5.4 fl away | -2.3 fl away |
| G1 possessions | +0.025 (+3.1 fl, immaterial) | +0.022 (+2.7 fl, immaterial) |

Without the double-applied table the FTA/FGA and total-bias vetoes clear, but no arm moves the primary slope toward 1.0
beyond the floor, and G5 total SD ratio and h/a corr stay worse than S2. FTa_R8a is NEUTRAL at best; ties go to the incumbent.

## B. Fold 1 (2023-24, 5,635 games x 200 seeds): served v2 (V2) vs SERVED_V1 (V1)

V1 floors: offsets 1000-4000. Fold-1 targets: G1 68.391, OREB .2899, eFG .5047, FTA/FGA .3281, TOV% .1724.

| gate line | V1 | V2 | move (x floor) | toward target? |
|---|---|---|---|---|
| G1 possessions mean | 70.228 | 69.271 | -0.957 (-124) | yes (registered line, confirmed) |
| G4 OREB% | 0.2789 | 0.2849 | +0.0060 (+74) | yes (registered line, confirmed) |
| G9 slope | 0.9081 | 0.9371 | +0.029 (+5.9) | yes (registered line, confirmed) |
| G4 eFG% | 0.4964 | 0.5024 | +0.0061 (+72) | yes |
| G4 FTA/FGA | 0.3112 | 0.3198 | +0.0086 (+34) | yes |
| G4 TOV% | 0.1800 | 0.1797 | -0.0003 (-9) | yes |
| G9 margin bias | +0.245 | +0.052 | -0.194 (-6.2) | yes |
| **G9 total bias** | -1.524 | **-1.713** | -0.189 (-6.5) | **NO, away, veto line** |
| G5 margin SD ratio | 1.052 | 1.043 | -0.009 (-2.8) | yes |
| G5 total SD ratio | 0.9177 | 0.9179 | inside | n/a |
| G5 h/a corr (target .252) | 0.133 | 0.147 | +0.014 (+6.7) | yes |
| G6 home margin non-neutral (5.37) | 5.738 | 5.540 | n/a | yes, both PASS |
| G7 OT rate | 0.0304 | 0.0309 | inside | n/a |

Verdict by the registered rule: **NOT CONFIRMED**, on one veto line (G9 total bias, 6.5 floors away). All three
registered adoption lines confirm beyond their fold-1 floors and every other veto line moves toward target or sits inside.
Reading (not a rule): the possession fix (-0.96) removes a compensating over-count, exposing the fold-1 scoring deficit
(total bias already -1.52 under V1; FTA/FGA and eFG still below target). Not confirming does not un-adopt anything;
PM decides. Caveats: static rotation in both arms; V1 and V2 differ in engine commit (default-off flags only).

## Recommendation

1. Do not adopt F_R or F_T (ratings C, E3 retrain): no slope gain, vetoes fire under the adopted switches; keep served stack v2.
2. Fold-1 set confirmation is NOT CONFIRMED only on total bias (away); the adoption lines replicate. PM to rule whether the
   fold-1 total-bias under-shoot (-1.7 vs -1.5 at V1, both outside +/-1.0) is a pre-existing scoring deficit rather than an adoption effect.
3. If retrained-PO is pursued, the T2c trip table needs a refit against the retrained PO (needs its own registration).

AWS: ~29 min spot c7a.48xlarge, about $1.4 (spot ~$2.3-2.8/h incl. setup). Instance terminated (see ops note in final reply).
