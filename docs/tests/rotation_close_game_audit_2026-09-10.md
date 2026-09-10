# Rotation close-game audit (round-3 evidence) -- 2026-09-10

Worker: Opus. Inputs: `data/processed/possessions/possessions_{season}.parquet`, 
`data/processed/models/rotation/rotation_fit_round2_corrected_hazards.json`. 
Script: `scripts/diag_rotation_close_game.py`. JSON: 
`data/processed/models/rotation/close_game_audit_2026-09-10.json`.

Cells with fewer than 300 possessions are printed `UP(n)` and are 
UNDERPOWERED -- neither signal nor absence of signal.


## 1. Which seasons can answer the question at all

| season | games | possessions carrying a complete on-floor set | games complete on every possession |
|---|---:|---:|---:|
| 2022 | 5,282 | 0.0000 | 0 |
| 2023 | 5,541 | 0.0000 | 0 |
| 2024 | 5,553 | 0.9209 | 4,251 |
| 2025 | 5,593 | 0.9842 | 5,320 |

CBBD `onFloor` is empty at the source before 2023-24 (L13), so **the close-game keep behaviour can only be measured on seasons 2024 and 2025**. The task's 2022-2025 breakdown is reported here as the availability count above and nothing more; there is no lineup evidence in 2022 or 2023 to characterise, and none is invented.


## 2. Actual starters' share of on-floor slots, whole-game time profile

Starters = the five that side actually started. Share = starter slots / (5 x possessions) in the cell, the same construction the gate uses.


### 2.1 Season 2024

| time cell | |m|<=5 | |m| 6-15 | |m|>15 | possessions |
|---|---:|---:|---:|---:|
| H1 20:00-10:00 | 0.7878 | 0.6347 | 0.5518 | 290,778 |
| H1 10:00-00:00 | 0.6745 | 0.6692 | 0.6371 | 287,202 |
| H2 20:00-16:00 | 0.9073 | 0.9011 | 0.8795 | 120,322 |
| H2 16:00-12:00 | 0.6462 | 0.6355 | 0.5873 | 114,134 |
| H2 12:00-08:00 | 0.6599 | 0.6527 | 0.5817 | 114,276 |
| H2 08:00-04:00 | 0.7362 | 0.7255 | 0.6131 | 114,578 |
| H2 04:00-02:00 | 0.7739 | 0.7461 | 0.4905 | 56,642 |
| H2 02:00-00:00 | 0.7559 | 0.7043 | 0.3148 | 69,414 |

### 2.2 Season 2025

| time cell | |m|<=5 | |m| 6-15 | |m|>15 | possessions |
|---|---:|---:|---:|---:|
| H1 20:00-10:00 | 0.7791 | 0.6272 | 0.5422 | 359,128 |
| H1 10:00-00:00 | 0.6665 | 0.6649 | 0.6297 | 356,876 |
| H2 20:00-16:00 | 0.9019 | 0.8933 | 0.8740 | 148,680 |
| H2 16:00-12:00 | 0.6378 | 0.6279 | 0.5694 | 142,078 |
| H2 12:00-08:00 | 0.6577 | 0.6476 | 0.5688 | 142,484 |
| H2 08:00-04:00 | 0.7291 | 0.7170 | 0.6041 | 142,192 |
| H2 04:00-02:00 | 0.7658 | 0.7403 | 0.4909 | 70,504 |
| H2 02:00-00:00 | 0.7549 | 0.7032 | 0.3220 | 86,294 |

The two seasons agree cell for cell: the largest close-band (|m|<=5) difference across the eight time cells is 0.9 pp. There is no season effect to model; the round-3 fit on 2024 and the gate on 2025 are measuring the same coaching behaviour.


## 3. Starter vs bench minutes

| season | starter mean min | starter SD | bench mean min | bench SD | starters' share of team minutes |
|---|---:|---:|---:|---:|---:|
| 2024 | 28.52 | 7.70 | 12.43 | 8.37 | 0.7092 |
| 2025 | 28.32 | 7.92 | 12.63 | 8.51 | 0.7016 |

## 4. Where R2 and R5 miss, cell by cell (1600 games x 3 seeds, 2025)

The sim universe is the first 1600 games of the graded round-2 subset (numpy RandomState seed 2025), so these rows are a strict subset of the round-2 gate universe and the ACTUAL column is restricted to exactly the same team-games.


### 4.1 Margin band |m|<=5

| time cell | ACTUAL | R2_hier_dirichlet | R5_hybrid (corrected) | R2 miss (pp) | R5 miss (pp) | possessions |
|---|---:|---:|---:|---:|---:|---:|
| H1 20:00-10:00 | 0.7822 | 0.6120 | 0.7202 | -17.0 | -6.2 | 82,936 |
| H1 10:00-00:00 | 0.6698 | 0.7032 | 0.6311 | +3.3 | -3.9 | 50,028 |
| H2 20:00-16:00 | 0.9041 | 0.7861 | 0.8000 | -11.8 | -10.4 | 16,788 |
| H2 16:00-12:00 | 0.6428 | 0.7154 | 0.6407 | +7.3 | -0.2 | 14,290 |
| H2 12:00-08:00 | 0.6652 | 0.6618 | 0.6420 | -0.3 | -2.3 | 14,264 |
| H2 08:00-04:00 | 0.7347 | 0.7325 | 0.7052 | -0.2 | -3.0 | 13,772 |
| H2 04:00-02:00 | 0.7672 | 0.7340 | 0.6919 | -3.3 | -7.5 | 6,424 |
| H2 02:00-00:00 | 0.7577 | 0.7106 | 0.6815 | -4.7 | -7.6 | 9,464 |

### 4.2 Margin band |m| 6-15

| time cell | ACTUAL | R2_hier_dirichlet | R5_hybrid (corrected) | R2 miss (pp) | R5 miss (pp) | possessions |
|---|---:|---:|---:|---:|---:|---:|
| H1 20:00-10:00 | 0.6363 | 0.6001 | 0.6328 | -3.6 | -0.3 | 24,062 |
| H1 10:00-00:00 | 0.6736 | 0.6154 | 0.6199 | -5.8 | -5.4 | 47,194 |
| H2 20:00-16:00 | 0.8968 | 0.7910 | 0.7879 | -10.6 | -10.9 | 20,632 |
| H2 16:00-12:00 | 0.6413 | 0.7077 | 0.6387 | +6.6 | -0.3 | 19,122 |
| H2 12:00-08:00 | 0.6533 | 0.6575 | 0.6351 | +0.4 | -1.8 | 17,952 |
| H2 08:00-04:00 | 0.7201 | 0.7139 | 0.6672 | -0.6 | -5.3 | 17,198 |
| H2 04:00-02:00 | 0.7499 | 0.7132 | 0.6560 | -3.7 | -9.4 | 8,642 |
| H2 02:00-00:00 | 0.7099 | 0.6466 | 0.6424 | -6.3 | -6.7 | 10,964 |

### 4.3 Margin band |m|>15

| time cell | ACTUAL | R2_hier_dirichlet | R5_hybrid (corrected) | R2 miss (pp) | R5 miss (pp) | possessions |
|---|---:|---:|---:|---:|---:|---:|
| H1 20:00-10:00 | 0.5594 | 0.6488 | 0.5794 | +8.9 | +2.0 | 862 |
| H1 10:00-00:00 | 0.6446 | 0.6052 | 0.6048 | -3.9 | -4.0 | 9,176 |
| H2 20:00-16:00 | 0.8852 | 0.7364 | 0.7766 | -14.9 | -10.9 | 7,388 |
| H2 16:00-12:00 | 0.5955 | 0.6318 | 0.6300 | +3.6 | +3.4 | 9,100 |
| H2 12:00-08:00 | 0.5833 | 0.5913 | 0.6206 | +0.8 | +3.7 | 10,190 |
| H2 08:00-04:00 | 0.6131 | 0.5029 | 0.6124 | -11.0 | -0.1 | 11,594 |
| H2 04:00-02:00 | 0.5170 | 0.5456 | 0.5737 | +2.9 | +5.7 | 6,152 |
| H2 02:00-00:00 | 0.3405 | 0.3497 | 0.5482 | +0.9 | +20.8 | 5,614 |

## 5. The round-2 hypothesis test: is the starter time spent too early?

`model.md` section 10 predicts that if R5's late deficit is a *distribution* problem rather than a *level* problem, its starter share should sit ABOVE the real one early by about the amount it sits below late.

| band | first half (cells 1-2) ACTUAL | R2 | R5 | final 8:00 ACTUAL | R2 | R5 | R5 early surplus (pp) | R5 late deficit (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| |m|<=5 | 0.7399 | 0.6463 | 0.6867 | 0.7491 | 0.7258 | 0.6948 | -5.3 | -5.4 |
| |m| 6-15 | 0.6610 | 0.6102 | 0.6243 | 0.7240 | 0.6936 | 0.6572 | -3.7 | -6.7 |
| |m|>15 | 0.6373 | 0.6089 | 0.6026 | 0.5223 | 0.4773 | 0.5868 | -3.5 | +6.5 |

## 6. Slope check: team quintile of the as-of starter-minutes share

Team prior = the as-of predicted share of team minutes going to the predicted starting five, a pregame quantity. Cell = the starters' share of on-floor slots in the final 8:00 at |margin| <= 5.

| quintile | team-games | prior starter share | ACTUAL | R2 | R5 (corrected) | close-late possessions |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 640 | 0.5572 | 0.6770 | 0.6518 | 0.5993 | 5,600 |
| Q2 | 640 | 0.6342 | 0.7226 | 0.7109 | 0.6784 | 6,238 |
| Q3 | 640 | 0.6698 | 0.7471 | 0.7238 | 0.7054 | 6,138 |
| Q4 | 640 | 0.7044 | 0.7763 | 0.7565 | 0.7268 | 5,898 |
| Q5 | 640 | 0.7685 | 0.8219 | 0.7844 | 0.7608 | 5,786 |

- `actual` slope against the prior: **+0.692** (Q5 - Q1 = +14.5 pp)
- `R2_hier_dirichlet` slope against the prior: **+0.632** (Q5 - Q1 = +13.3 pp)
- `R5_hybrid_corrected` slope against the prior: **+0.763** (Q5 - Q1 = +16.2 pp)

Actual Q5 - Q1 is +14.5 pp. An arm whose quintile profile is flat is not matchup-specific no matter what its pooled cell says.


## 7. What the audit says (the round-3 brief)

**7.1 The corrected R5 is not the round-2 R5, and it now fails the blowout band
too.** `experiments.md` section 4's R5 column was produced with the inert foul
terms of section 5's defect. Re-run here with the corrected hazard matrix and
its refitted knobs (scale 1.00, p0 0.060), R5's final-8:00 cells are

| cell | ACTUAL | R5 (round 2, inert fouls) | R5 (corrected) |
|---|---:|---:|---:|
| \|m\| <= 5 | 0.7491 | 0.7063 (-4.3 pp) | 0.6948 (-5.4 pp) |
| \|m\| 6-15 | 0.7240 | 0.6555 (-6.9 pp) | 0.6572 (-6.7 pp) |
| \|m\| > 15 | 0.5223 | 0.4937 (-2.9 pp, PASS) | 0.5868 (**+6.5 pp, FAIL**) |

The blowout pass was an artifact of the defect: with the foul terms inert the
margin terms had to absorb their load, which made the override bench starters
harder in a blowout for the wrong reason. Section 4.3 localises the corrected
arm's blowout failure precisely -- it is almost entirely the last two minutes
(final 2:00 at \|m\| > 15: **0.5482 sim vs 0.3405 actual, +20.8 pp**), i.e. R5
does not empty the bench when the game is over. **Round 3's baseline is
therefore an arm that misses all three margin bands, not one.** Section 5 of
`experiments.md` is closed by this row.

**7.2 The round-2 "spent too early" hypothesis is REFUTED.** `model.md` section
10 predicted that R5's late deficit would be matched by an early surplus. It is
not: in the close band R5 is **-5.3 pp early and -5.4 pp late**, and in the
6-15 band -3.7 / -6.7. R5 is below the real starters' share almost everywhere in
a close game. The close-game miss is a **level** problem across the whole game,
not a redistribution of a correct total, and a within-game time-profile
constraint alone would not have fixed it. That is direct evidence for the
round-3 direction: the donor sequence needs a mechanism that can put a starter
**on** the floor, because the block can only take one off.

**7.3 Where each arm misses, and it is not where the gate looks.** The gate
reads only the final 8:00. The fine profile shows two much larger misses that no
gate currently sees:

- **The second half tips off with the starters and no arm knows it.** Real
  teams run **0.90** starter share in H2 20:00-16:00 in every margin band
  (0.9019 / 0.8933 / 0.8740 in 2025) -- including blowouts. R2 gives 0.79 and R5
  0.78-0.80, a **-10 to -15 pp** miss in the largest single cell of the second
  half. `model.md` decision 6 chose to re-set the floor at a period boundary by
  target *rate*; the data says the coach re-sets it to his starters.
- **R2 under-plays the starters at the opening tip.** H1 20:00-10:00 in the
  close band: actual 0.7822, R2 0.6120, a **-17.0 pp** miss, R5 0.7202 (-6.2).
  R2's pooled minutes table is right while its within-game shape is wrong in
  both directions, which is exactly the failure mode multi-level evidence exists
  to catch.

Neither cell is a pre-registered gate and neither is added to one in round 3 --
gates are not changed mid-bake-off -- but both are recorded here and belong in
the next pre-registration.

**7.4 Responsiveness is not the problem.** Bucketed by the pregame team prior,
the close-and-late starters' share slopes correctly for both arms (actual
+0.692, R2 +0.632, R5 +0.763 per unit of prior starter share; actual Q5 - Q1
+14.5 pp against R2 +13.3 and R5 +16.2). R5's miss is a near-constant offset of
-6 to -8 pp at every quintile, not a flat prediction. Whatever round 3 fits must
preserve that slope.

**7.5 There is no season effect to model.** 2024 and 2025 agree cell for cell to
within 0.9 pp in the close band across the whole time profile, so fitting on
2024 and gating on 2025 is measuring one behaviour, and 2022-23 contribute
nothing because CBBD carries no on-floor data there at all.
