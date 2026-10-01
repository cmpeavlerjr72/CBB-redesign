# Late-game round 5: BZ3 + a five-band window law on trailing and leading rows (2026-10-01)

Lane L, 10:24-10:50 EDT (wall clock; run log `results/late_game/round3/run_log.txt`).
**NOTHING IS ADOPTED. No served default changes.** The new flag value is default-off.

- Pre-registration: `docs/models/late_game/experiments.md` section 11, commit `3d8cc05`. It was pushed before any arm was wired or run.
- Wiring: commit `1c55224` (`ENGINE_LATE_GAME=clk_DtLL`).
- The make law (MK2) was held out, as the PM ruled.

## 0. Verdict (500 verified games x 25 paired seeds, fold 2; base R9 = served v2; floors = round 4's five draws)

| arm | P(0)/P(1) | floors vs R9 | OT | hard vetoes (11.3) | points/game vs R9: total (H1 / H2+OT), SE |
|---|---:|---:|---:|---|---|
| Dt (round 3) | 0.672 | +2.13 | 0.0341 | PASS | +0.02 (0.00 / +0.02), 0.03 |
| Dt+a1 (BZ3, round 4) | 0.796 | +2.93 | 0.0383 | FAIL G1 SD +1.6 fl, half share +1.9 | -0.37 (-0.29 / -0.09), 0.05 |
| Dt+LL (`clk_DtLL`) | 0.779 | +2.45 | 0.0422 | FAIL G1 mean +7.1, G1 SD +2.5, half share +5.2, window possessions | +0.64 (0.00 / +0.64), 0.05 |
| **Dt+LL+a1 (the set)** | 0.845 | +2.90 | 0.0450 | FAIL G1 mean +5.1, G1 SD +2.9, half share +6.0, window possessions | +0.12 (-0.29 / +0.41), 0.06 |

**The set is not a candidate.** Its P(0)/P(1) is below 1.0 and it fails four hard vetoes.

**The set is not the clear best.** It fails hard vetoes, so the rule does not apply to it. The only arm that passes every hard veto is round 3's `clk_Dt`, already ruled on.

**Outcome: NO ARM ADOPTED. No `d1001_L_2.md` was filed.** Status: RUN.

**G9 total bias, as a priced exposure.** This is not a disqualifier under 11.3.
- BZ3 removes 0.29 points per game in H1 and 0.09 in H2.
- The window law adds 0.64 in H2, through extra possessions.
- The set nets +0.12 per game (SE 0.06).

## 1. Does the trailing law remove the leading law's extra possessions? No: it adds more.

| run | G1 possessions / game | window possessions by k = 0..6 |
|---|---:|---|
| R9 | 68.92 | 7.66 7.80 7.86 7.62 7.42 6.59 5.85 |
| Dt+b (leading only, round 4) | 69.13 | -- |
| Dt+LL | **69.22** | 7.85 8.35 8.24 8.03 7.82 7.25 6.48 |
| actual (sample / season) | 68.31 | 7.50-8.45 / 7.46-7.77 |

**Durations are now right; the count is not.** In the closed loop, the five-band law puts the possession time right in EVERY role x bucket cell:

| cell | Dt+LL | actual | R9 |
|---|---:|---:|---:|
| trail 1-3, (30,60] | 15.3 s | 15.7 s | 12.1 s |
| lead 1-3, (30,60] | 17.3 s | 18.1 s | -- |
| lead 1-3, (10,30] | 4.3 s | 4.0 s | 10.2 s |
| lead 4-6, (10,30] | 3.0 s | 2.9 s | 8.0 s |
| tied, (30,60] | 22.2 s | 22.0 s | -- |

Yet:
- Final-2:00 possessions per simulation (all margins) are 9.65, against 9.36 for R9 and 7.98 actual.
- The served engine already over-produces final-2:00 possessions by about 17%.
- Window possessions exceed the actual by 0.2-0.6 per k.

So the excess count is NOT owned by the window durations. The evidence points at the leading offence's EVENT mix (round 4's owner table):
- Late leading possessions end in a turnover 0.21-0.29 of the time, against 0.07-0.18 actual.
- They end FT-only 0.75-0.79 of the time, against 0.83-0.92 actual.

A turnover hands the trailing team a live-ball (short) possession where the data hand it a dead ball after free throws. This is a hypothesis with that evidence; no counterfactual was run. The next object is the trailing team's late fouling in the possession-outcome model, not the clock.

**Offline (11.2, both folds; guards pass):**

| rows | F2 | F1 |
|---|---:|---:|
| trailing, CRPS | +10.0 floors | +7.2 floors |
| leading, CRPS | +13.1 | +13.2 |

On F1 the trailing rows' censored log-likelihood is slightly worse (-1.6 floors); that is reported, not a guard line. Source: `results/late_game/round5/clock_offline.json`.

## 2. The remaining gap, reported separately

`results/late_game/round5/decision.json`, `tieloss.json`.

| run | P(tied) at 1:00 | at 0:30 | at 0:10 | P(1) one-point finish |
|---|---:|---:|---:|---:|
| R9 | 0.028 | 0.031 | 0.031 | 0.052 |
| Dt+a1 | 0.028 | 0.035 | 0.039 | 0.048 |
| Dt+LL+a1 | 0.027 | 0.032 | 0.044 | 0.053 |
| actual | 0.031 | 0.037 | 0.055 | 0.036 |

**Arrival.** Games reaching a tie at 0:10 improve: 0.031 -> 0.044 against 0.055. Arrival at 1:00 does not move; it is already close (0.027-0.028 vs 0.031).

**One-point finishes.** P(1) is the unmoved half of the ratio: 0.053 against 0.036. No arm in rounds 3-5 reduces it beyond about 0.004. The ratio target cannot be met by tie conversion alone. The one-point excess is the next diagnostic object: which final possessions end one-point games (leading-team FT make, round 4 (c), already priced at -0.006 P(1)).

## 3. Multi-level

Source: `results/late_game/round5/levels.json`.

- **Per game, paired vs R9 over 25 seeds.** The reseed floor moves about 31% of games each way (round 3).

  | arm | games with more ties | games with fewer | mean extra ties per game |
  |---|---:|---:|---:|
  | set | 40.4% | 15.4% | +0.40 |
  | Dt+LL | 35.6% | 14.4% | +0.33 |

- **Site.** The set's OT rate is 0.046 neutral / 0.045 non-neutral, against actual 0.068 / 0.054. The rise is uniform across sites.
- **Half.** The H1 buzzer PPP is unchanged from round 4: 0.764 against 0.730, and it holds by site.
- **The count excess, by half.** Period-2 possessions starting in the final 10 s:

  | run | per game |
  |---|---:|
  | actual | 0.62 |
  | R9 | 1.02 |
  | Dt+LL | 1.31 |

  The served engine already has about 65% too many possessions starting in the last 10 s of regulation. This is the same horn-count defect as the first half's, which the PM has passed to lane H, and the window laws make it worse.
- **Per team.** UNDERPOWERED (1 of 347 teams reaches 200 simulations).
- **Per player.** Not applicable.

## 4. Parity and tree state

- **Runs.** Both round-5 runs started on a clean `src/` (run log START lines at HEAD `7b35024` and `0b6c3c8`).
- **Off path.** The `clk_DtLL` edit changes no default path: unset never imports the module. `clk_DtL` keeps `lead_roles=(1,)`, so it is unchanged.
- **Parity v9.** Last proven at `52a2d7a`, with the round-4 wiring. NOT re-run for round 5: no box request was needed, and the edit is confined to the `clk_DtLL` branch.

## 5. NOT run

- The full-size box read: no candidate and no clear best.
- The fold-1 closed loop: no fold-1 inputs.
- Any change to fg_make: as instructed.
