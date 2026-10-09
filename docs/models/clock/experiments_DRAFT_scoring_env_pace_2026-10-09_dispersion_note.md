# Sibling note to the scoring-environment pace DRAFT: margin-dispersion gate (2026-10-09)

Written by the margin-dispersion diagnostic worker. The DRAFT itself is untouched. Evidence: `docs/tests/margin_dispersion_2026-10-09.md`.

## What the diagnostic found that bears on this round

- The sim's per-game margin is about 5% too wide in SD (sim own SD 12.22 vs MC-corrected realised residual SD 11.65, ratio 1.049, 95% [1.028, 1.070], 5,710 games; about 10% in variance).
- The **pace latent does not own the width**. Covariance-share attribution of the sim's within-game margin variance: pace (possession-count latent) 0.2%, per-possession efficiency gap 99.4%, interaction -0.2%, overtime 0.5%. Pace SD ratio (sim within-game SD vs realised residual SD of possessions) is 1.009 [0.984, 1.033].
- What the pace round does touch is the **pace level**: the sim runs 68.39 regulation possessions per team vs 67.54 real (+1.3%). Margin is possessions x PPP gap, so this alone scales margin SD by about 1.3% (of the 4.7% excess). Fixing the level trims the ratio slightly; it is not the main source.
- The bulk of the excess sits in the late-game regime (covariance between the minute-36 margin and the last-4:00 increment is -12.0 in the sim vs -20.9 realised), owned by the D3/D4 late-game round, plus a flat efficiency variance function (sim own SD barely moves with |spread| while realised residual SD grows from 10.8 in the closest quintile to 12.8 in the widest).

## Gate to add to section 6 of the DRAFT (paired-seed closed loop, must not regress)

Not a new selection metric; a no-regression and reporting gate, because a pace slope on `env`/`gap` changes the possession count per game and therefore the scale of the margin in points.

1. Regulation-margin SD ratio (sim own SD / MC-corrected realised residual SD, fold 2, game-cluster bootstrap CI) overall and by |spread| quintile, computed with `scripts/diag_margin_dispersion_v1.py` on the arm's own 50-seed run. Baselines: 1.047 overall; Q1 1.126, Q2 1.045, Q3 1.051, Q4 1.065, Q5 0.985. A candidate fails if any quintile ratio moves further from 1.0 than the baseline by more than the seed-bootstrap band (about +/-0.009).
2. Pace SD ratio (possessions per team, sim within-game SD vs realised residual SD) must stay inside [0.95, 1.05] overall and must not move away from 1.0 in any |spread| quintile (baseline: Q1 1.009, Q2 1.026, Q3 1.022, Q4 1.044, Q5 0.933).
3. Regulation total-points SD ratio must hold at its baseline 0.994 [0.974, 1.014]; the pace slope must not buy a better mean total by shrinking or inflating total dispersion.
4. Report the pace share of margin variance (table 4a) for the winner. A winner whose pace term exceeds a few percent of margin variance has introduced width the sim did not have before.

These gates are checks, not adjustments: nothing is scaled or clipped on sim output.
