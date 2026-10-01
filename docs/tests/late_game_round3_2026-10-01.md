# Late-game round 3: the hold without the extra possessions, on served stack v2 (2026-10-01)

Lane L, day session 2026-10-01, 06:14-09:10 EDT (wall clock, run log `results/late_game/round3/run_log.txt`).
**NOTHING IS ADOPTED. No served default changes.** The new flag values are default-off.

- Pre-registration: `docs/models/late_game/experiments.md` section 6 (lane C, 2026-09-30).
- Amendment: section 7, commit `61cb7ef`. It was pushed at 06:18, before any arm was wired or run.
- Gate G7 (OT rate): served v2 reads 0.0305 at full size, against 0.0557 actual.

## 0. Verdict

Primary: P(0)/P(1) (target 1.546). Floor (Decision 12) = max(SD over R9 + four seed-offset draws, 2 x paired game-bootstrap SE).
Screen: 500 verified games x 25 paired seeds, fold 2.

| line (pre-registered) | R9 (served v2) | Dt9 (`clk_Dt`) | Dtt9 (`clk_Dtt`) | D9 (`clk_D`) | verdict |
|---|---:|---:|---:|---:|---|
| P(0)/P(1) | 0.552 | 0.672 | 0.709 | 0.841 | all three move it by > 1 floor; all three stay < 1.0 |
| floors vs R9 (floor) | -- | **+2.13** (0.056) | +1.76 (0.089) | +2.83 (0.102) | |
| OT rate (band 0.046-0.055) | 0.0290 | 0.0341 (paired dP0 +0.0051, SE 0.0012) | 0.0319 | 0.0402 | none in band |
| vetoes (G1, G5, G9, half share, first half, window possessions) | -- | PASS | PASS | FAIL: G1 mean +6.2 fl, G1 SD +2.6, half share +4.5, window possessions +2.6 to +4.9 fl | |
| offline guard 7.5 (fold 2 CRPS on replaced rows) | -- | PASS (9.3 fl better) | PASS (9.0) | PASS (7.1) | fold 1 agrees |
| **candidate (6.5)** | -- | **no** (ratio < 1.0) | no | no | **NO ARM IS A CANDIDATE** |

Section 1.3's rule decides the round: an arm that raises the tie rate but leaves P(0)/P(1) below 1.0 is NOT a fix. Status: RUN, nothing adopted.

**Decision-rule attribution (6.5).** Dt9 and Dtt9 pass the window-possession veto; D9 fails it. So round 2's extra possessions come from D's LEADING-offence draws. They do not come from the hold or from the trailing draws.

**Foul-state attribution (B0 = v2 with `ENGINE_FOUL_JOINT=reference`).** The corrected foul state does nothing to the tie rate:
- R0 vs R9: +0.08 floors on the ratio (OT 0.0297 vs 0.0290).
- Dt0 vs R0: +1.94 floors (Dt9 vs R9: +2.13).
- D0 OT is 0.0425 (D9: 0.0402).

R9ao3 does raise late bonus occupancy for the leading offence, from 0.78-0.82 to 0.89-0.92 (actual 0.93-1.00), and its FT-only share from 0.65 to 0.74 (actual 0.83). Neither moves ties.

## 1. Spec check against served stack v2 (step 1)

Section 6 was written before served stack v2. Section 7 restates only what the new stack forces:
- B9 = the plain default stack, which is the candidate base.
- B8 (R8b) is superseded and NOT RUN.
- B0 = v2 with the foul state reverted, for attribution only.
- Inputs: engine_v3 (tag F2_2025), the verified minswap 500, `CBB_TRUTH=verified_v1`.

It adds two things:
- the offline composite guard on both folds (7.5);
- the tie-loss diagnostic (7.6).

It also fixes one reading of the window-possession veto: if R9 itself exceeds the actual by more than one floor in some k, that k is read against R9. This was fixed before any run.

`clk_D` is served unchanged: the round-2 pickle, sha `13f94be1...`. It is scaled by the served L2 clock's per-game latent.

**No round-3 arm trains on any rows.** `clk_D` was fitted in round 1 on period-2 window rows only (OT excluded, section 4.1). This answers the PM's OT foul-state audit note: no OT-period foul-state feature enters any arm, and every diagnostic below reads period-2 rows only.

## 2. Where the regulation ties are lost (step 2; reported lines)

Tools:
- Runner: `scripts/diag_late_game_r3_tieloss_v1.py`, output `results/late_game/round3/tieloss_b9.json` and `tieloss_r0.json`.
- Sim side: the R9 possession log (158,459 period-2 possessions starting at <= 180 s; 12,500 simulations).
- Actual side: `possessions_v4` 2025 with the same definitions, plus verified finals for OT (5,705 games).

`m(T)` is the home margin at the first possession that starts at <= T seconds left.

### 2.1 Arrival vs conversion

A shift-share of the R9 tie-rate gap (0.0290 vs 0.0560 season, gap -0.0270) splits it into arrival (the |m(T)| distribution) and conversion (the kernel):

| T | P(m=0) sim / act | arrival | conversion | arrival share |
|---|---|---:|---:|---:|
| 2:00 | 0.027 / 0.031 | -0.0044 | -0.0227 | 16% |
| 1:00 | 0.028 / 0.031 | -0.0058 | -0.0212 | 21% |
| 0:30 | 0.031 / 0.037 | -0.0071 | -0.0199 | 26% |
| 0:10 | 0.031 / **0.055** | **-0.0196** | -0.0074 | **73%** |

- Being close at 2:00 explains about a sixth of the gap. This agrees with round 1's 12.5%.
- The loss builds up between 1:00 and 0:10. Real close games converge to a tie and the sim's do not.
- The kernel from 1:00:

| |m(1:00)| | sim P(tie) | actual P(tie) | actual n |
|---:|---:|---:|---:|
| 0 | 0.21 | 0.37 | 172, UNDERPOWERED |
| 1 | 0.12 | 0.17 | -- |
| 2 | 0.12 | 0.25 | -- |
| 3 | 0.08 | 0.13 | -- |

- Once tied at 0:10, the sim breaks the tie 33% of the time against 15% actual (kernel 0.67 vs 0.85).

### 2.2 Behaviour cells

Final 2:00, by offence role and clock bucket, R9 vs season actual. Cells with n < 200 are marked.

| behaviour | cell | sim | actual |
|---|---|---:|---:|
| **tied offence does not hold** | tied (30,60], duration used (s) | 12.2 | 22.0 |
| | tied (10,30], duration used (s) | 10.1 | 16.7 |
| | tied final possession (<= 35 s): used s / reaches horn | 8.6 / 34% | 13.4 / 50% |
| **end-of-clock possessions score too well** | tied (10,30], PPP | 1.12 | 0.84 |
| | tied (0,10], PPP | 1.01 | 0.61 (n 172, UNDERPOWERED) |
| | trailing 1-3 (0,10], PPP | 1.04 | 0.58 |
| | tied final possession: share scoring 0 | 46% | 64% |
| **trailing team fouls too late** | leading 1-3 (10,30], duration used (s) | 10.2 | 4.0 |
| | leading 1-3 (0,10], duration used (s) | 4.1 | 1.4 |
| | leading 1-3 (0,10], FT-only share | 0.75 | 0.92 |
| | leading offence inside 30 s, 3PA share of FGA | 0.21-0.28 | 0.00-0.13 |
| **FT make, leading offence late (wrong-signed for ties)** | leading 1-3 (10,30] / (0,10] | 0.64 / 0.64 | 0.77 / 0.77 |
| | leading 4-6 | 0.72-0.79 | 0.77 |

The FT make row cuts the other way. More leader misses means more ties, so this defect hides part of the tie deficit. Fixing it lowers the tie rate (see section 7).

**Trailing-team shot selection.** Trailing 4-6 three-point share is close down to 10 s (0.41-0.55 vs 0.42-0.56); inside 10 s it is 0.69 vs 0.75. Trailing 1-3 inside 10 s takes too few threes: 0.57 vs 0.69.

**What Dt changes.**
- It fixes the hold: tied (30,60] used 21.9 s; the tied final possession uses 12.6 s and reaches the horn 54% of the time.
- It does not fix the conversion: tied final possessions still score 0 only 50% of the time (actual 64%), and the kernel at 0:10 is 0.70 (actual 0.85).

So the hold is half of the tied-game defect. The other half is end-of-clock shot quality.

### 2.3 Responsiveness

Script: `scripts/diag_late_game_r3_resp_v1.py`, output `resp.json`. Games are bucketed by the engine's own pregame closeness (|mean simulated margin|).

On the full-size served v2 run (`v3full_COMB9GCTKD_s200_o0`, 5,710 x 200), sim OT runs at roughly half the actual rate in every quintile, closest first:

| quintile | sim OT | actual OT | actual SE |
|---|---:|---:|---:|
| 1 (closest) | 0.039 | 0.082 | 0.008 |
| 2 | 0.038 | 0.061 | -- |
| 3 | 0.034 | 0.052 | -- |
| 4 | 0.028 | 0.056 | -- |
| 5 | 0.013 | 0.028 | -- |

- The span ratio is 0.48. The response has the right shape at half the level.
- This fits a conversion defect that scales every game's tie chance. It does not fit an upstream margin defect.
- Arm quintiles on the 500 sample are UNDERPOWERED on the actual side (2-11 OTs per cell).

### 2.4 Per game and per team

`scripts/diag_late_game_r2_levels_v1.py`, output `levels_b9.json`.

Per game, paired over 25 seeds against R9:

| run | games with more ties | games with fewer ties | mean extra ties per game |
|---|---:|---:|---:|
| reseed floor | 31% | 31% | -0.008 |
| Dt9 | 19.8% | 9.4% | +0.13 |
| D9 | 35.0% | 16.6% | +0.28 |

Per team: 1 of 347 teams reaches 200 simulations, so per-team cells are UNDERPOWERED and not read. Per player: not applicable.

## 3. Offline line (7.5; both folds; one grader)

`scripts/grade_late_game_r3_offline_v1.py`, output `results/late_game/round3/offline.json`. Round 1's held-out window rows; CRPS of the horn-truncated law. The floor is the block-bootstrap SE of the paired delta vs A (the Kaplan-Meier laws are deterministic, so the reseed floor is 0).

| arm | F2 all | F2 tied rows | F2 trailing | F2 leading | F1 all | F1 tied |
|---|---:|---:|---:|---:|---:|---:|
| A (served family) | 4.150 | 5.873 | 3.798 | 4.271 | 4.190 | 5.841 |
| D | -0.126 (+7.1 fl) | -1.107 (+9.3) | -0.088 (+4.1) | -0.042 (+1.8) | +6.5 fl | +7.9 |
| Dt | -0.066 (+8.2 fl) | -1.107 (+9.3) | 0 | 0 | +7.4 fl | +7.9 |
| Dtt | -0.106 (+8.2 fl) | -1.107 (+9.3) | -0.088 (+4.1) | 0 | +5.0 fl | +7.9 |

- Every guard passes on both folds.
- Tied rows: n = 1,208 on F2 and 1,237 on F1.
- The censored log-likelihood agrees in sign everywhere.

Offline, the hold law is right by a wide margin. In the closed loop it is not sufficient.

## 4. Wiring, parity, tree state

- **Flag.** `ENGINE_LATE_GAME=clk_Dt|clk_Dtt` in `src/cbb_sim/engine/late_game_adapter.py`. It is the same `clk_D` law, gated by the sign of the offence's live `score_diff`. Unset means the module is never imported.
- **Parity.** Off path vs parity v9: `run_engine.py --seeds 5 --max-games 60 --input-dir data/processed/models/engine_v3` digest **PASS, bit-identical** (sha `e0a42353...`).
- **`clk_D` unchanged by construction.** Its gate is `roles=None`, so the edited branch never runs for it.
- **First half.** It is bit-identical to the base in 12,500 of 12,500 simulations for every arm.
- **Dirty shared tree.** Other lanes' uncommitted engine edits were live when several of my runs started. Each START line in the run log records them; the files included:
  - `adapters.py`
  - `loop.py`
  - `clock_adapter_v3.py`
  - `foul_r9.py`
  - `shot_block.py`

  I re-ran a 20-game x 25-seed slice of all 11 runs: eight at 08:11-08:29 and three at 09:00 on a clean tree at `476ca10`. **33 of 33 files were bit-identical** (`games`, `tap_sims`, `tap_poss`). The dirty edits did not touch these runs.
- **Code commits.**
  - `61cb7ef`: the amendment.
  - `71225f8`: wiring, runner, offline grader, diagnostic.
  - `3d148f0`: closed-loop grader, box script.

## 5. Box request (PENDING)

`docs/ops/box_queue/d1001_L_1.md`: the registered 500 x 200 tapped re-read (6.4) of R9 (five draws) and the three arms. It is a re-read only; there is no candidate, so no full-size tier was requested. It was still unanswered when this was written. If it returns before 15:45, its numbers go in a short appended section 9.

## 6. What was NOT run

- **Fold-1 closed loop.** No 2023-24 v3 engine inputs or fold-1 serving artifacts exist (HANDOFF open item 10). Fold 1 enters through section 3 only.
- **The B8 (R8b) arms.** Superseded (7.2).
- **A local closed-loop tap for an OT read, and the full-size box read.** Step 4 is conditional on a winner and none exists. For scale: at 500 x 25 the paired SE of the OT-rate delta is 0.0012, so the screen already reads Dt's +0.0051 at about 4 SE.
- **Sensitivity windows (150 s / 8, 90 s / 5).**

## 7. Recommended next step

The hold is necessary but not sufficient. The remaining tie conversion is owned by two sub-models.

1. **End-of-clock shot quality (event + fg_make).** Possessions whose shot comes at the horn score about 1.0 PPP in the sim against 0.6 actual. fg_make is read at the possession-start state; it never sees that the shot is a last-second attempt. A round 4 should pre-register a fg_make / event term for time-left-at-the-shot inside the final period seconds (the engine knows `sec - used`), served together with `clk_Dt`. The decision rule should stay section 1.3's.
2. **The trailing team's foul timing.** Leading-offence durations inside 30 s are 10 s against 4 s actual. D's leading law fixes them but adds possessions, so the clock's leading-role law needs its own count-preserving design.

**For lane B (FT round 14, score_diff).** The leading-offence FT make in the final 30 s reads 0.64 against 0.77. Fixing it will LOWER the tie rate; it is a compensation to price in.

## 8. Resume commands

```
.venv/Scripts/python.exe scripts/run_late_game_r3_closed_loop.py --late-game clk_Dt --seeds 25 --workers 3 --tag lg3_Dt9_s25
.venv/Scripts/python.exe scripts/grade_late_game_r3_v1.py --base lg3_R9_s25 --draws lg3_R9_f1_s25 lg3_R9_f2_s25 lg3_R9_f3_s25 lg3_R9_f4_s25 \
    --arms lg3_Dt9_s25 lg3_Dtt9_s25 lg3_D9_s25 --b0 lg3_R0_s25 --b0-arms lg3_Dt0_s25 lg3_D0_s25 --out results/late_game/round3/grade.json
.venv/Scripts/python.exe scripts/diag_late_game_r3_tieloss_v1.py --runs lg3_R9_s25 lg3_Dt9_s25 lg3_D9_s25 --out results/late_game/round3/tieloss_b9.json
.venv/Scripts/python.exe scripts/grade_late_game_r3_offline_v1.py --out results/late_game/round3/offline.json
bash scripts/box_late_game_r3_v1.sh tierA ~/cbb 90 clk_Dt clk_Dtt clk_D      # box; then grade lg3box_* with the same grader
```


## 9. Box re-read, 500 x 200 (appended 10:30 EDT when `d1001_L_1.done.md` arrived)

Box clone `8eca67b`, plain-default parity v9 PASS. Graded locally with the same grader
(`results/late_game/round3/grade_box_s200.json`). The base is R9 at offset 0; the floor draws are offsets
1000-4000, all at 200 seeds.

| arm | P(0)/P(1) | floors vs R9 (floor) | OT (dP0, SE) | vetoes |
|---|---:|---:|---|---|
| R9 | 0.542 | -- | 0.0288 | -- |
| `clk_Dt` | 0.679 | +6.84 (0.020) | 0.0343 (+0.0055, 0.0004) | FAIL G1 SD +2.3 floors (5.510 -> 5.543 vs 5.20; floor = draw SD only, 0.014) |
| `clk_Dtt` | 0.750 | +6.26 (0.033) | 0.0327 | FAIL G1 SD +1.2 |
| `clk_D` | 0.884 | +9.12 (0.038) | 0.0420 | FAIL G1 mean, G1 SD, half share, window possessions (+5 to +13 floors) |

The screen verdict stands: no arm reaches 1.0, and none is a candidate. At 200 seeds `clk_Dt` also fails the G1 SD veto.
