# Shot-location stage: round 1 spec (DRAFT, PM Fable 5.1, 2026-10-27)

Status: DRAFT. Not promoted, nothing fitted, no fit output exists. Promotion = this text copied
verbatim into `docs/models/shot_location/experiments.md` section 1, with `model.md` and
`features.md` stubs per `DOCUMENTATION_STANDARD.md`, in the commit BEFORE the first fit.
Freeze is ON; nothing in this round is served. The round is one Opus worker day, local compute.

## 0. Origin and the question

Collaborator Heuermann's method (rfq-collab `CBB Model/Heuermann/README.md`): the shot LOCATION is
drawn first from a matchup grid (offence attempt distribution combined with what the defence
allows), and make, shooting foul and offensive rebound are each conditioned on it. Our cascade is
class-first: possession outcome draws one of six terminal classes (`TOV`, `FGA_rim`, `FGA_jump2`,
`FGA_3`, `FT_trip_shooting`, `FT_trip_bonus`); make is conditioned on class; rebound on miss type.
Foul and rebound never see a location.

The 2026-10-25 PAE round (fg_make `experiments.md` sections 31-32) tested only the make channel,
as a feature on a class-level model. The 2026-10-27 coordinate audit
(`docs/tests/coordinate_audit_2026-10-27.md`) found rim x/y is outcome-dependent in the ESPN feed
(misses mischarted twice as often as makes, makes dropped from the chart more often, a placeholder
pixel at the basket that makes less and responds less), while jumper and three placements are
clean. The source survey (`docs/research/shot_coordinate_sources_2026-10-27.md`) found no free
source that extends coordinates before 2021-22.

The engine never observes a historic coordinate at run time. It would SAMPLE a location. So the
value of a location stage to the sim is not "how much does known location explain" (a ceiling) but
"is the per-matchup location mix predictable before tipoff, and does carrying it change the
game-level aggregates the engine is graded on". This round reads both, and the decision rule is on
the second.

## 1. Data, frame, population

- Source: CBBD play-by-play on disk, seasons 2022-2025 (2021-22 through 2024-25). NO union with
  hoopR coordinates for 2022-24 (independent charting vintages, audit part 1). 2026 stays sealed.
- Population: the fg_make design population (D-I, non-truncated, `pbp_complete`, the possessions
  v2 event layer with the 2.27 ft rim override), de-duplicated on (gameId, id). Located = both
  `shot_location_x` and `_y` present.
- Frame: fold x to the half court, basket at (52.5, 250) in tenths of a foot, exactly as
  `scripts/build_pae_shot_rating_v1.py` does. Distance d in ft from the basket centre; angle
  theta in degrees in the same convention as that script's `da15` grid (state the convention in
  the evidence doc with a plot).
- Fold 1: train 2022-2023, test 2024 (87.5% located; missing games skew to power-conference,
  neutral-site and postseason games: labelled BIASED SUBSET, reported, not gated).
- Fold 2: train 2022-2024, test 2025 (98.6% located). SELECTION fold and the "vacuum" read.
- Coverage handling, fixed in advance: a location feature is NEVER imputed for an unlocated
  attempt. Attempt-level reads (section 3) use located rows only and report the located share
  per cell. Team-game reads (sections 4-5) use realised aggregates from ALL attempts, located or
  not, and report located-only games and all games side by side.

## 2. Bands and cells (fixed before any fit)

The rim class is ONE band (ruling 1, 2026-10-27): no rim location feature of any kind enters a
training table in this round, because rim placement is outcome-dependent. Within the two clean
classes:

| class | band | definition (folded frame) |
|---|---|---|
| `FGA_jump2` | `short2` | d <= 10 ft |
| `FGA_jump2` | `long2` | d > 10 ft |
| `FGA_3` | `corner3` | within 4.0 ft of either sideline, i.e. abs(y - 250) >= 210 tenths, AND d <= 24 ft |
| `FGA_3` | `wing3` | not corner, abs(theta) >= 30 deg from the centre line |
| `FGA_3` | `top3` | not corner, abs(theta) < 30 deg |

Five bands, two per jumper, three per three. Side (left/right) is not a band. The arm that uses
the fine grid uses `da15` (2 ft x 12 deg cells) as chosen on fold-1 training data in section 31.2
of the fg_make log; it is NOT re-chosen. Cell baselines are prior-season make rates exactly as in
31.2 (season 2022's baseline is 2023, training rows only).

Band shares per class are reported for the pooled training seasons in the evidence doc before any
arm runs, with the per-season drift, so a reader can see whether the bands are stable.

## 3. Read A: ceiling (attempt level, located rows, F2 selection, F1 reported)

Does known location explain each channel BEYOND shot class and the served features? Per channel,
the served trainer is wrapped (as `train_fg_make_v5_pae_v1.py` wrapped the fg_make trainer) and
the arms differ ONLY in added columns.

| channel | population | served reference `R` | `A1` | `A2` |
|---|---|---|---|---|
| make | `FGA_jump2` and `FGA_3` attempts (rim excluded) | lane-A B1 features, S1 monthly schedule, frozen ladder | R + band (one-hot) | R + `cell_p` (prior-season `da15` cell make rate) |
| foul draw | MADE `FGA_jump2` / `FGA_3` attempts; target = and-one (a shooting foul on a made shot) | the served fg_make feature set reused as the base | R + band | R + `cell_p` |
| rebound | rebound-opportunity rows whose miss is a `FGA_jump2` or `FGA_3` (rim misses and free-throw misses excluded) | served rebound `lgbm` / `C_plus_state` | R + band of the miss | R + d and theta of the miss |

- Floor per channel and class: `max(abs(LL(R) - LL(R_seed)), paired game-block bootstrap SE of
  LL(arm) - LL(R))`, 200 replicates, as in 31.5. `R_seed` is `R` under seed 1.
- Metrics: attempt-level log loss (multiclass for rebound); the `prob_metrics` decile battery
  (worst gap, level, shape); responsiveness lines: predicted vs realised rate by band, and by
  quintile of the offence's as-of band share, slope ratio in [0.8, 1.2].
- Segments: site; month; power / mid / low tier of the offence; located-share-of-game quartile.
  Cells under 5,000 attempts (make, foul) or 2,000 opportunities (rebound) are UNDERPOWERED.
- **The foul channel is PARTIAL by construction.** The feed logs no field-goal attempt for a missed
  shot on which a shooting foul was called (free_throw `model.md` section 2), so no coordinate
  exists for it. The and-one rate on made attempts is the only observable location-conditioned
  foul quantity. It is reported as such and never described as "foul rate by location".
- Read A verdict per channel: `SIGNAL` if A1 or A2 beats the floor with the decile gate held, else
  `NO SIGNAL`. Read A verdicts are FINDINGS. They never make the stage a candidate.

## 4. Read B: predictability (the matchup read, F2 selection, F1 reported)

Is the within-class band mix of a team-game predictable before tipoff, and does the opponent's
allowed profile add to the offence's own profile? This is the Heuermann claim in testable form.

Unit: located attempt, target = band within class (two-way for jumpers, three-way for threes).
As-of features, strictly before tipoff, built with the 31.2 recipe (shrunk to the snapshot league
mean, prior-season carry-over `rho`, separate `k` per side; `k` and `rho` chosen on fold-1
TRAINING seasons by located-attempt-weighted squared error against the game's realised shares,
before any arm is fitted):

- `off_band_c`: the offence's centred as-of log-odds of each band within the class.
- `def_band_c`: the defence's centred as-of log-odds of each band ALLOWED within the class.
- site, and the `C_plus_state` game-state columns.

| arm | model |
|---|---|
| `B0` | league-average band shares by class (the null; also what the engine implicitly assumes today) |
| `B1` | multinomial logit on `off_band_c` + site + state (team profile only) |
| `B2` | `B1` + `def_band_c` (offence x defence: the matchup form) |
| `B3` | LightGBM multiclass, frozen ladder params, on B2's columns (checks whether the GLM form is leaving signal) |

- Floor: these fits are deterministic, so the seed floor is zero; the floor is the paired
  game-block bootstrap SE of the log-loss difference (200 replicates), plus a sensitivity fit of
  `B2` under the fold-1 second-best `(k, rho)`, reported.
- Metrics: attempt-level multinomial log loss per class; team-game realised band shares vs
  predicted (MAE, and the SD of the residual against the SD implied by the multinomial, per class);
  responsiveness: predicted vs realised band share by quintile of `off_band_c` AND by quintile of
  `def_band_c` (slope ratio in [0.8, 1.2], monotone 4 of 4, the Decision-8 battery with the
  low-span exemption); the leak test on both rating families (|corr| <= 0.15, change form against
  the verified own-game margin) BEFORE any arm, REFUSED on failure.
- Segments as in section 3.
- Read B verdict per class: `MATCHUP` if B2 beats B1 beyond the floor and both responsiveness lines
  pass; `PROFILE` if B1 beats B0 beyond the floor with the offence line passing but B2 does not add;
  `FLAT` otherwise. B3 is reported; if B3 beats B2 beyond the floor the finding is recorded and the
  GLM is still the arm carried into section 5 (ties and form questions go to the simpler model).

## 5. Read C: implied game aggregates (the sim-relevant read, offline, no engine run)

Whatever the attempt-level results, the engine is graded on game aggregates. Read C asks whether
carrying a location stage would change them, computed offline by marginalising:

    implied(outcome | matchup) = sum over bands of  P(band | matchup, class)  x  P(outcome | band, features)

per team-game, for the offence side, using Read B's carried arm for `P(band | ...)` and Read A's
arm for `P(outcome | band ...)`, against the class-level served pair (the served possession-outcome
class mix with the served make / rebound models, which is `B0` x `R`).

- Targets per team-game, from ALL attempts (not located only): eFG on jumpers and threes
  combined; three-point make rate; and-one rate on made jumpers and threes (partial, section 3);
  OREB rate on jumper and three misses.
- Metrics: MAE of implied vs realised per team-game; the dispersion check (SD of realised minus
  implied against the SD the implied distribution predicts); responsiveness of implied vs realised
  by quintile of the offence's as-of band profile and of the defence's.
- Floor: paired team-game bootstrap SE of the MAE difference, 200 replicates.
- Reported for located-only games and all games side by side; for F1 the located-only set is the
  BIASED SUBSET and is labelled so.

## 6. Decision rule (read on F2; F1 reported, not gated)

- **REFUSED** if either Read B rating family fails the leak test. Not re-tuned.
- The shot-location stage is **CANDIDATE (offline)** for a class only if ALL of:
  1. Read B is `MATCHUP` or `PROFILE` for that class (predictable before tipoff).
  2. Read C improves at least one of {eFG, three-point make rate, OREB rate} beyond its floor for
     that class's attempts, with no other Read C target worse beyond its floor.
  3. The Read C responsiveness lines pass on the offence profile, and on the defence profile when
     Read B is `MATCHUP`.
  4. Read A is `SIGNAL` on at least the make channel for that class (a stage with nothing
     downstream to condition on has no reason to exist).
- Read A `SIGNAL` with Read B `FLAT` is a FINDING ("location explains outcomes but the per-matchup
  mix is not predictable beyond league average"), not a candidate, and is recorded in the fg_make
  and rebound ledgers as an upper bound on what any location-aware feature could buy.
- Ties go to the simpler arm and to the served class-level cascade.
- Nothing stronger than CANDIDATE (offline) is stated. Serving needs: an engine location stage
  behind a default-off flag, downstream make / rebound models retrained on band columns (their own
  rounds, bottom-up), and a paired-seed sim run with no gate regressed. The rim class is untouched
  by any of it.

## 7. Registered diagnostics (reported with the round, not gated)

1. The possessions v2 rim override (JumpShot within 2.27 ft -> `FGA_rim`) uses the same
   outcome-dependent rim placement the audit found. Report the override rate for makes vs misses
   within game; if it differs beyond a paired floor, it goes to the possession-layer queue.
2. Second-source check on 2025 only, from 2025-02-10 (the window where hoopR and CBBD agree within
   1 ft on 91-95% of pairs): band shares per team-game from hoopR vs CBBD, and the share of
   attempts whose band differs between the two. Descriptive.
3. Band drift by season and month (section 2).

## 8. Scripts, artifacts, grader

- `scripts/build_shot_location_bands_v1.py` (bands, cells, as-of band ratings, leak test, coverage
  tables; `--part` selectable).
- `scripts/train_shot_location_v1.py` (Read B arms), `scripts/train_fg_make_v6_loc_v1.py`,
  `scripts/train_rebound_v2_loc_v1.py`, `scripts/train_foul_draw_v1.py` (Read A wrappers; they add
  columns and change nothing else; each disclosed in its docstring).
- `scripts/grade_shot_location_v1.py`: ONE blind grader for Reads A, B and C; arm codes hidden
  until the tables are written.
- Artifacts under `data/processed/models/shot_location/round1/` (gitignored, HF `model_artifacts`
  key); evidence doc `docs/tests/shot_location_round1_<date>.md`; results appended as section 2 of
  `experiments.md`; a `change_ledger.md` row in the same commit as any status change.
- Suite re-run; parity v10 bit-identical (no `src` change is expected; if one is needed it is
  behind a default-off flag and disclosed).

## 9. Data-extension options (pending, outside this round)

Filled when the 2026-10-27 GUI-source survey and the BigQuery fact sheet land. Any second vendor's
coordinates enter only through a registered agreement check (section 7, item 2 pattern) in a later
round, never by union into the training table of this one.

## 10. Known gaps this spec does not close

- The foul channel is observable only on and-ones (section 3). A location-aware shooting-foul rate
  needs a feed that carries the fouled attempt, which ours does not.
- The rim class is excluded by ruling, so the one channel where the PAE round saw its largest
  per-shot number is not read here. A rim treatment (collapsing the placeholder pixel and unlocated
  rim rows into one "rim, no placement" category) is a separate pre-registration if a non-ESPN
  source ever allows the charting bias to be measured.
- Player-level location profiles (which shooter takes the corner three) are usage / attribution
  territory and are not in this round.
