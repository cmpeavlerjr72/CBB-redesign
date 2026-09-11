# Attribution `score_diff` leak: quantification (2026-09-10)

Worker: attribution round 2 (Sonnet). Script: `scripts/diag_attribution_score_diff_leak.py`.
Raw output: `data/processed/models/attribution/score_diff_leak_2026-09-10.json`.
Fix applied in the same session: `src/cbb_sim/models/attribution.py` (`_state_block`,
`build_attr_events`), a `score_diff_mode` parameter defaulting to `"pre_play"`
with `"leaked"` selectable for reproduction. See `docs/models/change_ledger.md`
row "`cbb_sim.models.attribution` builds `score_diff` from `hs - as_` on the
candidate's own row" (2026-09-10, flagged, not fixed) — this doc closes that
flag.

## 0. The suspect and the test

`docs/LEARNINGS.md` L27 (fg_make): CBBD's `homeScore`/`awayScore` are the score
**after** the row's own play, so a state feature built from the candidate's own
row is post-outcome whenever that row is itself a scoring play. `attribution.py`
line ~534 (pre-fix) built `score_diff` the identical way, off the same two
columns, for **every** population it constructs.

The question this round has to answer is narrower than "is `score_diff`
leaked": attribution builds five populations (`reb_off`, `reb_def`, `made_fga`,
`tov`, `miss_fga`) feeding eight targets, and the leak can only reach a
population whose **own row is a scoring play**. A live rebound's own row is the
rebound, not the miss; a turnover's own row never changes the score; a missed
or blocked attempt's own row is a miss. Only `made_fga` — the population behind
`assist` and `assisted` — is a made field goal on its own row. This is a
testable claim, not an assumption, so it is tested on all five populations, not
assumed on four of them.

## 1. Part 0 — the own-row delta test (L27), all five populations, both seasons

Off the cleaned attribution stream (`attribution.build_attr_stream`), independent
of `_state_block`: for every row, does the row's own team's score move, on its
own record, relative to the row immediately before it in the same game? A
made field goal's own row should move by exactly its point value if the column
is post-play; every other event should move by exactly zero either way.

| population | season | n | mean own-row delta | % delta == expected value | % delta == 0 |
|---|---:|---:|---:|---:|---:|
| `reb_off` | 2024 | 107,768 | -0.0006 | 99.967 | 99.967 |
| `reb_off` | 2025 | 115,007 | -0.0002 | 99.921 | 99.921 |
| `reb_def` | 2024 | 259,639 | -0.0000 | 99.984 | 99.984 |
| `reb_def` | 2025 | 266,671 | 0.0000 | 99.974 | 99.974 |
| **`made_fga`** | 2024 | 267,004 | **2.2708** | **99.304** | **0.683** |
| **`made_fga`** | 2025 | 279,510 | **2.2825** | **99.311** | **0.677** |
| `tov` | 2024 | 121,637 | 0.0002 | 99.984 | 99.984 |
| `tov` | 2025 | 128,428 | -0.0030 | 99.974 | 99.974 |
| `miss_fga` | 2024 | 337,450 | 0.0002 | 99.984 | 99.984 |
| `miss_fga` | 2025 | 351,925 | -0.0007 | 99.979 | 99.979 |

`made_fga` is qualitatively different from the other four: its own-row delta
equals the shot's own point value on 99.3% of rows (mean 2.27-2.28, matching the
population's mix of 2s and 3s) and is zero on well under 1%. `reb_off`,
`reb_def`, `tov` and `miss_fga` are at or above 99.9% zero-delta in every
season, the residual ~0.02-0.08% being same-second adjacent scoring plays the
stream's ordering does not fully disambiguate (an and-1 free throw landing on
the same clock tick as the rebound that follows it), not a systematic leak.
**Only `made_fga` — the population behind `assist` and `assisted` — carries the
leak.** This matches L27's own numbers for fg_make almost exactly (93.6-94.5%
there vs 99.3% here — attribution's `made_fga` population is *already*
filtered to made shots only, so there is no missed-shot denominator diluting
the percentage the way fg_make's mixed make/miss population has).

## 2. Part 1 — how much of the apparent effect vanishes: the three binaries

Outcome-rate bucket span (max − min binary rate across score_diff deciles),
computed on the same rows under both constructions so the only thing that
changes is the feature. Pooled 2024+2025.

| target | base rate | rows with `score_diff` changed by the fix | leaked span (pp) | pre-play span (pp) | **manufactured (pp)** | manufactured share |
|---|---:|---:|---:|---:|---:|---:|
| **`assisted`** | 51.50% | **100.0%** | 12.312 | 10.017 | **2.295** | **18.6%** |
| `stolen` | 56.52% | 0.0% | 6.426 | 6.426 | 0.000 | 0.0% |
| `blocked` | 9.91% | 0.0% | 1.691 | 1.691 | 0.000 | 0.0% |

`assisted` is the only binary the fix touches at all — every row's `score_diff`
changes, and 18.6% of its apparent decile-span in the leaked construction is
manufactured by the post-outcome column (a made three, already 3 points ahead
of where the defence actually let the game get, reads as "further ahead" than
it was). `stolen` and `blocked` are provably untouched: 0% of rows change,
because their own event (a turnover, a miss) never carries points on its own
row in the first place. This is smaller, in relative terms, than fg_make's
62-84% manufactured share on made-vs-missed — plausible, since "was this make
assisted" is not driven by game state anywhere near as strongly as "was this
shot made," so there is less apparent effect for a leak to manufacture out of
to begin with.

## 3. Part 2 — how much of the apparent effect vanishes: the five choice targets

The P2 conditional logit's own `log_share_x_scorediff` / `is_C_x_scorediff`
interaction coefficients (state enters a conditional logit only as an
interaction — `model.md` section 3.1 / decision 11), fit on the real pipeline
(`usable` → `build_choice_design` → `cl_design` → `CondLogitArm`, `l2=1.0`,
`prior_kind='position'`, `m=25`, an effect-size read rather than a searched
arm) on the F1 training slice (season 2024), leaked vs pre-play:

| target | population | `log_share_x_scorediff` (leaked → pre-play) | `is_C_x_scorediff` (leaked → pre-play) |
|---|---|---|---|
| `REB_off` | `reb_off` | 0.01535 → 0.01535 (0.0%) | 0.01443 → 0.01443 (0.0%) |
| `REB_def` | `reb_def` | 0.00430 → 0.00430 (0.0%) | 0.00465 → 0.00465 (0.0%) |
| **`assist`** | `made_fga` | 0.00304 → 0.00308 (-1.3%) | -0.00243 → -0.00267 (-9.9%) |
| `steal` | `tov` | 0.01190 → 0.01190 (0.0%) | -0.04524 → -0.04524 (0.0%) |
| `block` | `miss_fga` | 0.02696 → 0.02696 (0.0%) | 0.00333 → 0.00333 (0.0%) |

`REB_off`, `REB_def`, `steal` and `block` are **bit-identical** between the two
constructions, because their populations' own row is never a scoring play, so
`score_diff` is unchanged by the fix and there is nothing for a coefficient to
move. `assist` is the one choice target the fix touches: its two interaction
terms move by -1.3% and -9.9%. Both are small relative to the binary's 18.6%,
consistent with the choice question ("which of four teammates assisted")
depending on score margin far more weakly than the binary question ("was it
assisted at all").

## 4. Verdict

The leak is real and reaches exactly two of the eight targets: **`assisted`**
(binary, 18.6% of its apparent decile-span manufactured) and **`assist`**
(choice, a real but small 1-10% coefficient shift on its two score_diff
interaction terms). The other six targets — `REB_off`, `REB_def`, `steal`,
`stolen`, `block`, `blocked` — are proved clean by the own-row delta test in
Part 0 (>= 99.9% zero-delta) and confirmed unaffected by the fix in Parts 1-2
(0% of rows changed, coefficients bit-identical). This is not a hand-wave: the
leak is present or absent by construction (whether the population's own row is
a scoring play), and Part 0 tests that construction directly against the raw
feed rather than assuming it from the target list.

Fix: `src/cbb_sim/models/attribution.py`'s `_state_block` now takes
`score_diff_mode` (`"pre_play"` default, `"leaked"` selectable for
reproduction) and `build_attr_events` threads it through; for `made_fga` rows
under `"pre_play"` the shot's own points (2 or 3, read off `shot_class`/`made`)
are subtracted from the candidate's post-play margin, exactly the repair
`fg_make.add_round2_state`'s `score_diff_pre` already shipped (L27). No other
population's construction changes under the fix (own points are 0 by
construction there), which Part 0-2 above confirm rather than assume.
`tests/test_attribution.py` (27 tests) passes unchanged under the new default.

Round 2's pre-registration (`docs/models/attribution/experiments.md` section 4)
carries `assisted` and `assist` as the two targets the fix is expected to move,
and the other six as a "no gate should move" control.
