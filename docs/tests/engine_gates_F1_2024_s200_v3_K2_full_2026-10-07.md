# Clock round 8 arm K2, fold-1 full-size read vs served v2 F1 (box d1007_K2F1, 2026-10-07)

Run: `results/engine_v0/f1c_K2_full_s200_o0` (5,635 games x 200 seeds 0-199, verified truth, fold 1 = 2023-24). Reference: `results/engine_v0/f1c_V2_full_s200_o0` (served v2 F1, same seeds, same overlay `fold1_v1/overrides_V2.json` except the K2 clock re-point in `docs/ops/box_queue/d1007_K2F1_overrides.json`). Only difference: `ENGINE_CLOCK=v5b_r8K2_glat_pmean` on the fold-1 K2 artifacts (`clock/r8_K2/F1`). Spec and decision rule: `docs/ops/box_queue/d1007_K2F1.md` (committed 506b4f3 before the run). Full grade files: `results/engine_v0/v3full_grade/f1c_{V2,K2}_full_s200_o0__verified.md`. Box: spot c7a.48xlarge, 94 workers, 34 min wall including setup, about $2. Parity v9 PASS on the box. `run_meta`: ENGINE_CLOCK = v5b_r8K2_glat_pmean, source check PASS, all other flags equal to the reference.

## Headline (fold 1; raw gate tolerances, the PM rules)

| gate | served v2 F1 | K2 F1 | real |
|---|---|---|---|
| G1 possessions/game mean / SD | 69.271 / 5.584 PASS | 69.229 / 5.681 PASS | 68.391 / 5.480 |
| G1 by month (5 powered months, mean status) | 3/5 inside (months 1, 2 FAIL) | 3/5 inside (months 1, 2 FAIL) | |
| G2 PPP cells inside +/-0.02 | 3/9 FAIL | 4/9 FAIL | |
| G4 pooled tov / oreb / ft_rate / eFG | 0.1797 / 0.2849 / 0.3198 / 0.5024 | 0.1800 / 0.2848 / 0.3234 / 0.5042 | 0.1724 / 0.2899 / 0.3281 / 0.5047 |
| G5 margin SD ratio | 1.0425 PASS | 1.0395 PASS | 1.0 |
| G5 total SD ratio | 0.9179 FAIL | 0.9608 PASS | 1.0 |
| G5 home/away score corr | 0.1466 FAIL | 0.1875 FAIL | 0.2522 |
| G5 PIT K-S p | 0.0079 | 0.044 | > 0.1 |
| G6 home margin non-neutral / neutral | +5.540 / +2.157 PASS | +5.538 / +2.168 PASS | +5.370 / +2.897 |
| G7 OT rate | 0.0309 FAIL | 0.0344 FAIL | 0.0600 |
| G8 rotation min SD ratio / players used | 1.2317 / 8.73 FAIL | 1.2340 / 8.73 FAIL | 1.0 / 9.77 |
| G9 total bias | -1.713 FAIL | -1.457 FAIL | 0 |
| G9 calibration slope (total) | 0.9371 FAIL | 0.9411 FAIL | 1.0 |
| G9 margin bias / margin MAE / total MAE | +0.0515 / 9.163 / 13.592 | +0.0509 / 9.153 / 13.603 | 0 |

Gate-level status is identical on both runs for every gate (G1 FAIL, G2 FAIL, G3 and G4 NEEDS-INSTRUMENTATION, G5 FAIL, G6 PASS, G7 FAIL, G8 FAIL, G9 FAIL); no line moved PASS to FAIL except the single G9 by-month cell below.

## Against the decision rule

- (a) G5 total SD ratio 0.918 -> 0.961 (F2: 0.925 -> 0.973) and score corr 0.147 -> 0.188 (F2: 0.126 -> 0.170): both move toward real, by about the same size as F2 (+0.043 / +0.041 here vs +0.048 / +0.044). The corr is still 0.065 below real and still FAILs; the total SD ratio now passes. CONFIRMED.
- (b) No gate regressed. G1 possession mean is 0.04 lower (closer to real), the G1 SD is 0.10 wider (inside the +/-0.75 tolerance, still above real 5.480; same direction as F2 G1 SD 5.504 -> 5.629). G2 improves by one cell. G4 pooled ft_rate and eFG move slightly toward real. G7 OT rate rises 0.0309 -> 0.0344 (real 0.060), toward real. G8 unchanged. One by-month status flip: G9 month 12 margin bias -0.496 -> -0.502 (threshold 0.5), a 0.006 move that sits on the tolerance edge, not a movement away from real beyond noise.
- (c) G9: total slope 0.937 -> 0.941 (unchanged within noise, as on F2), total bias -1.71 -> -1.46 (toward 0), margin bias, margin MAE and total MAE flat. Not worse. Total MAE +0.012 and the by-month total-bias cells improve in months 1, 2, 12 (month 11 and the pre-existing -5.0 bias unchanged), so the residual total bias is not a clock effect.

Verdict: K2 CONFIRMED on fold 1 by the pre-registered rule. Caveats, labelled: one seed set (0-199), no fold-1 noise-floor retrain of K2 under another seed was run here; the fold-1 floor of the served stack (`f1c_V1` offsets) was not re-derived for this arm; fold 1 runs `ENGINE_ROTATION_SCHEME=static` for both arms (stated deviation of `chain_fold1_v1.py`). The per-team G3/G4 tables are underpowered at 30-40 games per team and carry no signal in either direction. Adoption is the PM's call; nothing was changed here.
