# CA1 trailer-slope diagnostic (DIAGNOSTIC, not a selection read), 2026-10-14

Ordered by the PM ruling of 2026-10-14 on clock experiments.md section 47. Block A was REFUTED offline because CA1 (the ratio cell law on the served K2 window law) failed the team-prior trailer-slope gate. This doc asks why. It changes nothing: no arm is selected, adopted or re-registered here. Every table below is labelled DIAGNOSTIC, not a selection read.

Opus engine worker. Linked from clock experiments.md section 48.

## Inputs and commands

- Offline (a)-(c): `scripts/diag_ca1_trailer_slope_v1.py`.
  - It uses the same truth frame, predictions and quintile definitions as `grade_endgame_clock_v2.py` offline block A (`truth_A`, `load_pred`, `rows_A`, imported unchanged).
  - Rows: the fold-2 test season (2024-25), period-2 window rows (start <= 240 s), 74,164 rows. Fold 1 (2023-24, 69,410 rows) is reported as a check.
  - Floors: 2 x SD of the game-block bootstrap (200 draws, numpy seed 12345).
  - Outputs: `results/endgame_clock_v2/diag_ca1/offline_diag.{json,md}` (`offline_diag_tables.md`).
- Loop (d): CA1 wired DEFAULT-OFF as `ENGINE_ENDGAME=CA1`.
  - Wiring: `src/cbb_sim/engine/endgame_adapter.py`, commit `9b3f5f3`. Parity v10 PASS bit-identical with flags off.
  - Run: 500 verified fold-2 games x 25 paired seeds, local, 132 s; tag `egc2_CA1diag_o0_s25`.
  - Graded with `grade_endgame_clock_v2.py loop --block A`, so every line definition is unchanged. The grader's "screen" field is IGNORED: this is not a screen.
  - Control and floors: the round-10 served-v3 runs `clk10_ctrl_o0_s25` (control) and `clk10_ctrl_o{1000..4000}_s25` (four draws). Reuse was verified by re-running the control at HEAD (`egc2_ctrlcheck_o0_s25`): games.parquet is bit-identical on every numeric column (hash 8bc1e8f318f4bc54 both).
  - Floor = max(2 x SD of the four draws, 2 x paired bootstrap SE).

```
set CBB_TRUTH=verified_v1
.venv/Scripts/python.exe scripts/diag_ca1_trailer_slope_v1.py
bash results/endgame_clock_v2/logs/chain_ca1_diag.sh
.venv/Scripts/python.exe scripts/grade_endgame_clock_v2.py loop --block A --control clk10_ctrl_o0_s25 \
  --draws clk10_ctrl_o1000_s25 clk10_ctrl_o2000_s25 clk10_ctrl_o3000_s25 clk10_ctrl_o4000_s25 \
  --arm egc2_CA1diag_o0_s25 --loop-dir results/endgame_clock_v2/loop --disp-dir results/endgame_clock_v2/dispersion \
  --out results/endgame_clock_v2/diag_ca1/loop_CA1_diag.json
```

## (a) What is the ACTUAL trailer slope on team prior? (DIAGNOSTIC, not a selection read)

The table shows trailer first-chance seconds by the offence's prior-strength quintile (`off_rating_off_c`, quintiles over all period-2 window rows, as the gate). Fold 2:

| quintile | strength mean | rows | actual | control CA0 | CA1 |
|---|---:|---:|---:|---:|---:|
| Q1 | -9.45 | 9,593 | 13.491 | 13.854 | 13.017 |
| Q2 | -4.12 | 8,630 | 13.293 | 13.809 | 12.989 |
| Q3 | -0.23 | 7,601 | 12.981 | 13.859 | 13.049 |
| Q4 | +3.73 | 6,836 | 12.788 | 13.791 | 13.036 |
| Q5 | +10.83 | 5,373 | 12.861 | 13.852 | 13.144 |

| line (fold 2) | actual | control CA0 | CA1 |
|---|---|---|---|
| seconds per rating point (floor) | **-0.0345 (0.0131)** | -0.0002 (0.0074) | +0.0064 (0.0073) |
| model minus actual, s/pt (floor) | | +0.034 (0.011) | +0.041 (0.011) |
| Q5 - Q1 seconds (floor) | **-0.630 (0.282)** | -0.002 (0.153) | +0.127 (0.155) |
| gate form: model quintile means on actual quintile means (target 1) | 1 | +0.025 (0.182) | -0.120 (0.198); CA1 - CA0 floor 0.084 |
| fold 1: s/pt / Q5 - Q1 | -0.0555 (0.0164) / -1.113 (0.327) | -0.0061 / -0.123 | +0.0027 / +0.039 |

What the table shows:
- The actual trailer response is NEGATIVE and resolved on both folds: strong offences trail with shorter possessions, about 0.6 s (F2) to 1.1 s (F1) faster from Q1 to Q5, roughly 2.6 and 3.4 floors.
- Neither law has it. The control is flat (-0.0002 s/pt) and CA1 is slightly positive (+0.0064 s/pt).
- Both laws miss the actual: the control by 3.0 floors, CA1 by 3.8. Their distances from the actual differ by 0.007 s/pt, which is under either floor. So the control did not meet the actual response; it sat about as far from it as CA1 does.
- Leaders for comparison (F2): actual +0.019 s/pt (floor 0.020, unresolved), control +0.013, CA1 +0.026.

## (b) Which (margin x time) cells drive the sign flip? (DIAGNOSTIC, not a selection read)

Each trailer cell's contribution to the trailer Q5 - Q1 difference is share(Q5, c) x mean(Q5, c) - share(Q1, c) x mean(Q1, c). The full 20-cell table for both folds is in `offline_diag_tables.md`. Most cells have fewer than 3,000 rows and are UNDERPOWERED for slopes; all are reported.

Cells where CA1's contribution differs from the control's beyond the paired floor:

| cell | rows (Q1 / Q5) | CA1 - CA0 contribution F2 (floor) | F1 (floor) |
|---|---|---|---|
| trailer 13+ \| (180,240] | 3,532 (1,063 / 359) | **+0.101 (0.029)** | +0.121 (0.028) |
| trailer 13+ \| (120,180] | 3,533 (1,079 / 357) | **+0.137 (0.037)** | +0.152 (0.036) |
| trailer 13+ \| (30,60] | 2,097 (605 / 215), underpowered | -0.036 (0.012) | -0.037 (0.013) |
| trailer 0-3 \| (60,120] | 1,432, underpowered | -0.032 (0.015) | -0.025 (0.013) |
| all 20 trailer cells | | +0.129 | +0.162 |

The flip comes from the two early trailer-13+ cells. The table shows their quintile-1 and quintile-5 means (seconds), fold 2:

| cell | Q1 actual / CA0 / CA1 | Q5 actual / CA0 / CA1 | within-cell Q5 - Q1 actual / CA0 / CA1 |
|---|---|---|---|
| trailer 13+ \| (180,240] | 15.90 / 17.80 / 15.44 | 15.04 / 18.14 / 15.75 | -0.86 / +0.34 / +0.30 |
| trailer 13+ \| (120,180] | 14.78 / 17.43 / 14.47 | 13.68 / 17.50 / 14.54 | -1.10 / +0.06 / +0.07 |

How the flip happens:
- The control over-predicts these cells by 2-3 s.
- These cells are 3:1 weak-team rows (Q1 1,063 vs Q5 359), so the over-prediction inflates the weak-team quintile mean.
- That inflation partly imitated the actual negative slope through composition.
- CA1 corrects the cells (17.8 -> 15.4 s vs actual 15.9), which removes the spurious composition term.
- Inside every trailer-13+ cell the actual has a negative response (-0.7 to -1.4 s Q5 - Q1). Neither law has one (control and CA1 between -0.2 and +0.3).

## (c) Are the z_off / defence-z overshoots the same defect, and is a baseline-only cell arm well defined? (DIAGNOSTIC, not a selection read)

The table splits each Q5 - Q1 spread into two parts:
- between-cell (composition) = sum over cells of (share(Q5, c) - share(Q1, c)) x the actual cell mean. This uses the same weights for every source.
- within-cell = the rest. For the control it also carries the control's cell miscalibration x composition.

| quintile line (fold 2) | source | total (floor) | between | within |
|---|---|---|---|---|
| z_off quintile, all period-2 rows | actual | +0.737 (0.255) | +1.747 | **-1.010 (0.212)** |
| | control CA0 | +0.517 (0.099) | +1.747 | -1.230 (0.135) |
| | CA1 | +1.737 (0.148) | +1.747 | **-0.010 (0.117)** |
| defence-z quintile | actual | +0.143 (0.251) | +0.855 | **-0.712 (0.206)** |
| | control CA0 | +0.168 (0.112) | +0.855 | -0.687 (0.109) |
| | CA1 | +0.832 (0.153) | +0.855 | **-0.023 (0.079)** |
| offence strength quintile, trailers | actual | -0.630 (0.282) | +0.005 | **-0.635 (0.265)** |
| | control CA0 | -0.002 (0.153) | +0.005 | -0.007 (0.105) |
| | CA1 | +0.127 (0.155) | +0.005 | **+0.122 (0.091)** |
| offence strength quintile, leaders | actual | +0.262 (0.430) | +0.476 | -0.214 (0.346) |
| | CA1 | +0.470 (0.241) | +0.476 | -0.005 (0.113) |

Fold 1 has the same pattern. Within-cell for actual / control / CA1: z_off -0.849 / -1.449 / -0.219; defence-z -0.520 / -0.962 / -0.242; trailers -1.129 / -0.139 / +0.023.

**Reading: one defect, not three.** The served K2 window law has no within-cell scoring-environment response.
- CA1 calibrates the cells. Its quintile spread is then pure composition (within about 0), so it overshoots wherever the actual within-cell response is negative:
  - z_off: 1.74 vs 0.74 (gate form 2.10).
  - defence-z: 0.83 vs 0.14 (gate form 3.04).
  - trailers: +0.13 vs -0.63 (gate form -0.120).
- The control's "within" term is the cell miscalibration x composition. It happens to sit near the actual within-cell response for z_off and defence-z (-1.23 vs -1.01, -0.69 vs -0.71). So the control's z_off and defence-z slopes looked partly right for the wrong reason, and its trailer slope looked flat for the same reason.
- So the ratio did not multiply a served response that was already correct. There was no served response: the control's apparent slopes were a cancellation that CA1 removed.

**Proposed arm for the next spec (described, NOT fitted): `CA1E`, a cell term on the duration baseline under a within-cell env AFT.**
1. Fit an env AFT scale `log k = c * z_off` (plus the defence split if the spec wants it). Use the M2 Poisson estimator with cell fixed effects = the 45 window cells x the K2 conditional cell, so `c` is identified only from within-cell variation (the actual's -1.0 s z_off within-term).
2. De-scale the window durations by k and fit the CA1 ratio table on the de-scaled durations against the served law, so the cell term carries only the baseline.
3. Serve it as `p = scale_pmf(CA1(p0), k_row)`. The cell term then cannot carry the rating response, and the response cannot carry the cell miscalibration.

This is well defined: the two factors are identified from orthogonal variation (between-cell vs within-cell). The next spec should grade it on the "not above actual beyond floor" form for every responsiveness line (the PM's 10-14 ruling). It should also check whether such a within-cell env AFT is the same term block C's ED/EG try to fit at full-game scope, so the two rounds do not double count.

## (d) Does CA1's offline gain show up in the sim's endgame lines? (DIAGNOSTIC, not a selection read)

500 verified fold-2 games x 25 paired seeds. CA1 only (period 2, start <= 240 s). 483 games have complete pbp. Floor = max(2 x SD of the four control draws, 2 x paired bootstrap SE).

| line | actual | control | CA1 | delta | floor | toward actual? |
|---|---:|---:|---:|---:|---:|---|
| H2 36-40 possessions (MD4) | 15.43 | 16.750 | 16.753 | +0.003 | 0.076 | no movement (inside floor) |
| 13+ last-2-min points (MD5) | 7.12 | 9.690 | 9.097 | -0.594 | 0.201 | yes, 2.9 floors (closes 23% of the gap) |
| tie rate (regulation tie = OT) | 0.0680 | 0.0362 | 0.0420 | +0.0058 | 0.0037 | yes, 1.6 floors |
| leader seconds/possession, 4:00 ref, 13+ | 17.48 | 14.972 | 16.316 | +1.344 | 0.146 | yes, 9.2 floors |
| leader seconds/possession, 2:00 ref, 13+ | 16.14 | 12.751 | 14.178 | +1.427 | 0.096 | yes, 15 floors |
| leader seconds/possession, 4:00 ref, 8-12 | 15.60 | 13.742 | 14.573 | +0.831 | 0.194 | yes, 4.3 floors |
| leader seconds/possession, 2:00 ref, 8-12 | 8.25 | 10.754 | 10.446 | -0.308 | 0.146 | yes, 2.1 floors |
| leader TOV rate 13+, 4:00 ref | 0.2015 | 0.2224 | 0.2182 | -0.0042 | 0.0034 | yes, 1.2 floors |
| G1 possession SD | 5.195 | 5.658 | 5.659 | +0.001 | 0.114 | no movement |
| G1 possession mean | 68.31 | 68.88 | 68.94 | +0.05 | 0.08 | inside floor |
| game possessions on env (D2) | 0.124 | -0.012 | -0.016 | -0.004 | 0.006 | inside floor |

Reported by the unchanged grader:
- Section-45 hard vetoes: none fail.
- MD flags: MD3, MD5 and MD6 pass; MD1, MD2 and MD4 do not move beyond floor.
- These are the grader's block-A screen semantics, recorded for completeness and not read as a verdict.

**Reading:**
- CA1's offline cell-calibration gain does reach the sim's endgame. Leader possession lengths move 1.3-1.4 s toward the actual in the 13+ state, 13+ last-2-min points fall 0.6 toward the actual, and the tie rate rises 0.006 toward the actual.
- The possession COUNT in the last four minutes (MD4) does not move. Leaders take longer, but CA1 shortens trailer possessions in the same cells (13+ trailer (180,240] 17.8 -> 15.4 s offline), and the two offset in the count.
- The CA1 change is confined to the window, and the env response is untouched (D2 -0.012 -> -0.016, inside floor). This is consistent with (c): CA1 fixes the cell level, not the within-cell response.

## What this changes

Nothing. Block A stays REFUTED by the literal s46 rule. CA1 stays not adopted, default off, and is wired only for this diagnostic. Freeze ON.
