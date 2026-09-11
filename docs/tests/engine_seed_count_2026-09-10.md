# ENGINE v0 -- seed-count study (F2, season 2025)

Generated 2026-09-10 by `scripts/exp_engine_seed_count.py`. Raw output:
`results/engine_v0/seed_count/{seed_count.json,ladder.csv,games.parquet}`.

`CLAUDE.md`, standing rule "backtests must be honest": *"A seed-count study
fixes the minimum seeds before any ROI number is read."* This is that study.
It is the FIRST one run on this engine, so every engine number written before
today -- including every cell of
`docs/tests/gates_engine_v0_F2_2025_s5_r2event_2026-09-10.md` and
`docs/tests/market_games_engine_v0_F2_2025_s5_r2event_2026-09-10.md` -- was
read at 5 seeds with no statement of its own Monte-Carlo error. Section 4 says
what that means for those numbers, and it is not a small correction.

---

## 1. Method

300 games (a fixed random subset, drawn with seed 20260910 **before** anything
was simulated) x 200 seeds = 60,000 simulations, 8,637,043 possessions.
Adapters: `ENGINE_EVENT=round2_s1`, `ENGINE_CLOCK=reference`,
`ENGINE_ROTATION=reference`, `ENGINE_FG3=decision8`.

The ladder is read OUT of that one run rather than re-simulated. This is exact,
not an approximation, and it is legal only because of the engine's RNG rule: a
simulation's draws depend on `(seed, game_id, family)` and on nothing else, so
seed *s* of game *g* is the same realisation whether it was run in a block of 5
or of 200. `tests/test_engine.py::test_a_games_draws_do_not_depend_on_the_batch_it_is_in`
pins that property. The seeds are therefore **paired across every rung by
construction**, and a "k-seed run" is exactly a k-seed slice.

For each rung k the 200 seeds are cut into `200 / k` **disjoint** blocks. Each
block is one independent k-seed estimate of a game's margin mean, total mean and
win probability; the SD across blocks, per game, IS the standard error of a
k-seed run -- measured, not assumed. Every rung below has at least 2 blocks and
is measured directly; only the "seeds required" column extrapolates.

All 300 of 300 games carried all 200 seeds, so no rung is read off a partial
panel.

---

## 2. The measured ladder

`se_mean` is the mean over games, `se_p90` the 90th percentile over games (a
gate is read on the slate, an edge is bet on one game). `slate_se` is the SE of
the 300-game AVERAGE, which is what G1/G6/G9 actually read.

| seeds | blocks | margin SE (mean) | margin SE (p90) | margin slate SE | total SE (mean) | total SE (p90) | total slate SE | win-prob SE (mean) | win-prob SE (p90) |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 40 | 14.850 | 17.699 | 0.772 | 6.434 | 7.390 | 0.354 | **0.2164** | 0.2481 |
| 10 | 20 | 10.275 | 12.606 | 0.669 | 4.540 | 5.421 | 0.249 | 0.1503 | 0.1839 |
| 25 | 8 | 6.364 | 8.730 | 0.272 | 2.833 | 3.758 | 0.150 | 0.0938 | 0.1264 |
| 50 | 4 | 4.282 | 6.585 | 0.211 | 1.874 | 2.884 | 0.123 | 0.0630 | 0.0974 |
| 100 | 2 | 3.032 | 6.013 | 0.112 | 1.185 | 2.506 | 0.107 | 0.0421 | 0.0856 |

Every column falls as `1/sqrt(k)` to within the noise of the estimate, as it
must. The 100-seed row rests on only 2 blocks and its own SE is therefore wide;
the p90 columns at 50 and 100 in particular should not be read as precise.

### Seeds required, from the fitted `c/sqrt(k)` through the measured rungs

| target | seeds required |
|---|---|
| **game-level win-prob SE < 1.0 pp (mean over games)** | **2,102** |
| game-level win-prob SE < 1.0 pp (p90 game) | 4,284 |
| game-level win-prob SE < 2.0 pp (mean) | 526 |
| game-level margin-mean SE < 0.50 pt | 3,995 |
| game-level margin-mean SE < 0.25 pt | 15,978 |
| game-level total-mean SE < 0.50 pt | 737 |
| slate-mean margin SE < 0.05 pt | 935 |
| slate-mean total SE < 0.05 pt | 288 |

---

## 3. The answer the deliverable asks for

**The game-level win-probability SE falls below 1 percentage point at about
2,100 seeds** (4,300 for the 90th-percentile game).

That number is mostly NOT a statement about this engine. A win probability is
the mean of k Bernoulli draws, so its SE is `sqrt(p(1-p)/k)`, which is at most
`0.5/sqrt(k)`: **no** simulator of any quality reaches a 1 pp per-game win-prob
SE under 2,500 seeds, and one with realistic `p` around 0.7-0.75 still needs
~1,900. The 1 pp target is a counting requirement, not an engine requirement,
and it should be understood as such before it is used to size a run.

What IS a statement about this engine is the margin column. At 5 seeds the
per-game margin-mean SE is **14.85 points**. That is not a small correction to
a market number; it is larger than the quantity being predicted. It is this
large because the engine's within-game margin SD is ~32-34 points against a
realistic ~11-14 (see section 5), so the required seed count is inflated by
roughly the SQUARE of that ratio -- about 6x. Fixing the defect in section 5
would cut every "seeds required" figure above by around six.

### Recommended minimum seed counts, stated separately because they differ by an order of magnitude

| what is being read | minimum seeds | why |
|---|---|---|
| **G1-G9 slate-level gates** (possessions/game, PPP, totals, home margin, OT rate) | **200** | slate margin SE ~0.08 pt and slate total SE ~0.08 pt at 200, an order of magnitude below every G1/G6/G9 tolerance (+/-1.0 pt) |
| per-team and per-quintile breakdowns | **200** | each cell averages 70+ teams, so the slate argument applies within cell |
| **G5 dispersion and the PIT histogram** | **200 minimum, and the PIT needs more** | at 5 seeds a game's PIT can only take 6 values, so the decile histogram is a comb by construction and its K-S p is meaningless. See section 4 |
| **G10: per-game ATS / OU / ML, Brier, ROI by edge bucket** | **NOT READABLE below ~2,000** | these read a PER-GAME probability or margin, where the SE at 200 seeds is still 2.1 pt (margin) and 3.3 pp (win prob) |

**The operative rule for the PM: 200 seeds for the gate report, and no ROI or
Brier number is quoted from a run under ~2,000 seeds.** At the measured 817
possessions/s/core a 200-seed full-season run is about 5.6 core-hours;
a 2,000-seed one is 56, which is why the lookup-table export
(`docs/tests/engine_v0_F2_2026-09-10.md`) is a prerequisite for the market
scorecard rather than an optimisation.

---

## 4. What this invalidates in today's 5-seed reads

Both of today's engine reports were generated at 5 seeds, before this study
existed. Reading them with the table above:

- **G1 possessions/game (+4.24)**: real. Slate SE at 5 seeds is well under
  0.1 possessions; a +4.24 miss is 40x that. **The verdict stands.**
- **G2 PPP by tercile, G3/G4 mix, section 3 of the multi-level report**: real.
  All are slate- or cell-level means over 1,100+ games.
- **G5 margin SD ratio 1.598**: **contaminated and overstated.** The reported
  "sim SD" is the within-game SD over 5 seeds, which the engine genuinely
  produces -- so the ratio is a true statement about the engine's dispersion,
  and section 5 shows it is a real defect. But the ACROSS-game SD of the
  engine's margin prediction (17.43 in the multi-level report) is almost
  entirely Monte-Carlo noise at 5 seeds: `sqrt(17.43^2 - 14.85^2) = 9.1` is the
  actual signal, against an actual 14.63.
- **G5 PIT K-S p = 2.3e-183**: **not readable at 5 seeds.** With 5 seeds the
  PIT takes at most 6 distinct values and the observed decile shares
  (0.042, 0.002, 0.159, 0.006, 0.270, 0.279, 0.005, 0.005, 0.173, 0.060) are
  that comb, not a distributional finding. Re-read at 200+.
- **G10 margin MAE 16.03, Brier 0.2715, every ROI bucket**: **not readable.**
  The per-game margin SE alone is 14.85 at 5 seeds, which accounts for the bulk
  of a 16.03 MAE, and the calibration deciles show model probabilities of
  exactly 0.2/0.4/0.6/0.8/1.0 -- the 5-seed quantisation, not the engine's
  belief. These numbers must be struck, not interpreted.

---

## 5. The defect this study made visible, and its effect on the seed count

The per-game margin SE is ~2.4x larger than it should be because the engine's
within-game margin SD is ~32-34 points. An ablation
(`docs/tests/engine_v0_F2_2026-09-10.md`, the 2026-09-10 evening section) shows
that **freezing the `score_diff` state feature at zero collapses the within-game
margin SD from 33.8 to 14.1**, moves the home/away score correlation from
-0.636 to +0.059 and moves possessions/game from 71.6 to 68.4 against an actual
67.9. The score-difference state feature is driving a positive feedback loop in
simulation that it cannot drive in training.

Nothing has been changed to compensate (`CLAUDE.md`, "no hand tuning on engine
output"); the ablation is a DIAGNOSTIC that names the responsible feature. Until
it is resolved, the seed counts in section 2 are the honest requirement for the
engine as it stands, and they should be re-measured -- not rescaled -- after the
fix.

---

## 6. Performance recorded alongside

| quantity | value |
|---|---|
| simulations | 60,000 (300 games x 200 seeds) |
| possessions | 8,637,043 |
| wall | 1,337 s on 8 workers |
| summed worker CPU | 10,566 s |
| **possessions/s/core** | **817.4** |
| possessions/s (8 workers) | 6,460 |
| machine | 20 logical cores, shared with four other model workers |
