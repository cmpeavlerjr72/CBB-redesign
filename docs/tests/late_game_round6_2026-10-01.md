# Late-game round 6: one-point finishes, and whether possession_outcome owns late fouling (2026-10-01)

Lane L, 10:40-11:05 EDT (wall clock; run log `results/late_game/round3/run_log.txt`).
**NOTHING IS ADOPTED. No served default changes.**

- Pre-registration: `docs/models/late_game/experiments.md` section 13, commit `8434d30`. It was pushed before any round-6 arm was scored or run.
- Base: served stack v2 (`lg4_R9_s25`, verified 500 games x 25 seeds). Lane I's FT-trip allocation (WHO is fouled) is not touched.

## 0. Verdict

| line (13.2 / 13.3) | result | floor | verdict |
|---|---|---|---|
| Offline: BL3 vs A, log loss on intentional-foul cells (leading offence, <= 60 s), F2 | -0.0148 | 0.0030 (block SE; BL3 reseed floor 0.00003) | **+5.0 floors: WIN** |
| same, F1 | -0.0117 | 0.0033 (block SE only; F1 seed-1 refit not run) | +3.6 floors, same sign |
| Guard: all window first chances | +8.7 floors F2, +7.5 F1 | -- | PASS |
| Responsiveness (FT trip by defence late-foul prior quintile) | data span +0.020 (F1) / -0.018 (F2); A and BL3 both flat (+0.001 / -0.009) | -- | no slope in the data to reproduce; BL3 is not flatter than A |
| Closed loop `clk_Dt+ev_BL3` vs R9 (500 x 25) | P(0)/P(1) 0.738 (+2.04 floors); vs `clk_Dt` alone (0.672): +0.066, about one floor | 0.091 | ratio < 1.0; **FAILS hard vetoes G1 mean and G1 SD** |
| One-point finishes P(1) | 0.0512, against R9 0.0525, Dt 0.0507, actual 0.0361 | -- | **not moved** |

The arm wins offline. In the closed loop it is not a candidate, and it cannot be the clear best because it fails hard vetoes. **No `d1001_L_2.md` was filed. Status: RUN, nothing adopted.**

## 1. One-point finishes: which final sequences produce them (part 1)

Scripts: `scripts/diag_late_game_r6_onepoint_v1.py` and `diag_late_game_r6_onepoint_seq_v1.py`. Outputs: `results/late_game/round6/onepoint_R9.json` and `onepoint_seq_R9.json`.

P(|regulation margin| = 1): sim 0.0525 against actual 0.0366, a gap of **+0.0158**. The gap is closed exactly by the last scoring sequence of regulation (offence margin before -> points scored):

| last scoring sequence | sim | actual | gap |
|---|---:|---:|---:|
| **trailer down 3 scores 2** (then no more scoring) | 0.0132 | 0.0046 | **+0.0086**, of which 10-30 s left +0.0054 |
| **trailer down 1 scores 2** (wins by 1) | 0.0210 | 0.0124 | **+0.0085** (> 30 s left +0.0038, 5-10 s +0.0018, 10-30 s +0.0018) |
| trailer down 2 scores 1 | 0.0028 | 0.0014 | +0.0014 |
| everything else | -- | -- | -0.0027 net |

By the scorer's role, the whole gap is the TRAILING team scoring last: +0.0177. Tied scorers are -0.0003 and leaders -0.0005.

By type, the gap is twos (+0.0113) and FT trips made 2+ (+0.0054). Threes are -0.0007; FT 1-of-2 is +0.0013.

Conditional on the trailer's two being the last score of regulation:
- that is the last score equally often in the sim and the data (0.202 vs 0.223 of games);
- but it leaves a one-point game twice as often in the sim (0.120 vs 0.057).

**The sim's game stops scoring after the trailer cuts the margin to 1 or takes a one-point lead.** In the data, the leading team is then fouled, and the margin keeps moving.

**|margin| distribution, P(0) to P(3):**

| time left | sim | actual |
|---|---|---|
| 0:30 | 0.031, 0.057, 0.054, 0.053 | 0.037, 0.060, 0.069, 0.067 |
| 0:10 | 0.031, 0.055, 0.051, 0.046 | 0.054, 0.059, 0.064, 0.060 |
| 0:05 | 0.029, 0.055, 0.053, 0.045 | 0.055, 0.051, 0.060, 0.060 |

The sim holds too much mass at |m| = 1 relative to 0 all the way to the horn.

**Behaviour, final 10 s:**

| state | measure | sim | actual |
|---|---|---:|---:|
| trailing by 3 | three-point share of FGA | 0.74 | 0.89 |
| trailing by 3 | fouled (FT-only possessions, "foul up three") | 0.20 | 0.29 |
| trailing by 3 | points per possession | 0.89 | 0.63 |
| trailing by 2 | three-point share | 0.54 | 0.63 |
| trailing by 2 | points per possession | 1.03 | 0.49 |
| trailing by 1 | points per possession | 1.16 | 0.54 |
| leading by 1-3 | FT-only | 0.75 | 0.91 |
| leading by 1-3 | three-point share of FGA | 0.24 | 0.17 (0.00 at 10-30 s) |
| leading by 1-3 | FT trips made 1 of 2 | 0.28-0.36 | 0.27-0.37 |

The FT 1-of-2 line is close; it is NOT a source.

**Leading-team turnovers are NOT the source.** In 1-6-point leads inside 30 s the sim's TOV rate is 0.08-0.11 against 0.05-0.13 in the data. The round-4 excess (0.21-0.29) sits in 7+ point leads (garbage time).

## 2. Who owns "the leading team is not fouled" (part 2)

`scripts/diag_late_game_r6_po_mix_v1.py` (output `po_mix.json`) and `grade_late_game_r6_bl3_v1.py` (output `bl3_grade.json`).

Leading offence, window first chances, inside 30 s, FT_trip_bonus (the intentional-foul trip), fold 2:

| layer | rate | gap |
|---|---:|---:|
| actual | 0.849 / 0.878 ((10,30] / (0,10]) | -- |
| possession_outcome offline prediction on ACTUAL states (round 1 arm A = served bundle, S0) | 0.825 / 0.839 | **model gap -0.02 to -0.04** (F1 lead 1-3: -0.05, -4 SE) |
| the served model on the SIM's own states (closed-loop event tap) | 0.753 / 0.763 | **state gap -0.07** |

**State-gap attribution (13.4).** The offline model gives P(trip | in bonus) = 0.835 and P(trip | not in bonus) = 0.0001. Leading offences inside 30 s are in the bonus 99.5% of the time in the data and **91.7% in the sim**. That alone moves the trip rate by **-0.066**, which is the whole state gap.

The trailing team does not foul often enough, early enough, to put the leader in the bonus. In the data a non-bonus foul does not end the possession; it is an accrual step on the way to the bonus. This is foul accrual (`ENGINE_FOUL_JOINT=R9ao3`, lane C's model), not possession_outcome.

**The possession_outcome part.** BL3 (round 1's role x time-left terms) wins offline (section 0), but in the FT-trip calibration cells it closes only about 0.005-0.016 of the 0.02-0.05 model gap.

In the closed loop with `clk_Dt` it adds:
- +0.0037 OT;
- nothing on P(1): 0.0512, against `clk_Dt`'s 0.0507;
- +0.06 possessions per game, which fails the G1 mean and SD vetoes.

The one-point sequences are unchanged: trailer down 3 scores 2 is still +0.0086.

## 3. Ownership of the one-point gap

Ranked by evidence size:
1. **Leading-team time use inside 30 s.** The served clock uses 10 s against 4 s actual, so after the trailer scores the clock runs out. The window laws fix the durations cell by cell (round 5) but are parked behind lane H's horn-count fix.
2. **Bonus occupancy of the leading offence late.** 0.917 vs 0.995 is foul accrual's job (lane C).
3. **End-of-clock trailing efficiency.** 1.0-1.2 PPP vs about 0.5 inside 10 s; the no-shot law BZ3 (validated-pending) owns part of it, and fg_make the rest (lane A note, round 5).
4. **possession_outcome's own mix.** About 0.02-0.04 of trip rate; BL3 captures a third of it offline and nothing in the loop.

## 4. Multi-level and parity

- **Per game.** The one-point decomposition is over all 12,500 simulations. The per-cell actual counts behind the top sequences are about 26 (down 3 scores 2) and about 70 (down 1 scores 2) one-point games over the season: UNDERPOWERED as cells, read only as the sum that closes the gap.
- **Per team.** UNDERPOWERED.
- **Site and half.** Not separately informative here: all one-point sequences are period 2.
- **Run state.** The closed-loop run started on a clean `src/` at `8434d30`. `ENGINE_LATE_GAME=clk_Dt+ev_BL3` uses round 2's existing artifact and wiring, so no new engine code was added this round; the parity v9 status of the default path is unchanged from `52a2d7a`.
- **Reseed refit.** The BL3 F2 seed-1 refit (`scripts/train_late_game_r6_bl3_seed1_v1.py`, 118 s, 3 threads) gives a seed floor of 0.00003. The block SE binds.

## 5. NOT run

- **The F1 seed-1 refit of BL3.** F1 was read on the block SE, as pre-registered. Command: `train_late_game_r6_bl3_seed1_v1.py` with fold F1 (would need a fold argument; about 2 min).
- **The full-size box read.** The closed-loop arm failed its hard vetoes. Resume, had it qualified: `bash scripts/box_late_game_r3_v1.sh tierB ~/cbb 90 clk_Dt+ev_BL3`, paired with lane D's S2 floors.
- **Any change to foul accrual, fg_make or FT-trip allocation.** Out of lane.
