# Shot-location stage: experiments log (append-only)

## 1. Round 1 (PROMOTED 2026-10-27 11:25 ET, PM Fable 5.1; committed BEFORE any fit)

Promoted verbatim from `experiments_DRAFT_shot_location_stage_2026-10-27.md` (the draft file stays
as the record of the drafting session and its section 9 data-extension history). At this commit no
fit output exists: `data/processed/models/shot_location/` does not exist and none of the section-8
scripts exist. Freeze ON; nothing in this round is served. Hard stop for the round: worker reports
by 15:30 ET 2026-10-27; anything not finished by then is reported as NOT RUN, never rushed.

### 1.0 Origin and the question

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

### 1.1 Data, frame, population

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
- Measurement reliability, REGISTERED (docs/tests/gamezone_frame_calibration_2026-10-27.md section 2):
  split-half (Spearman-Brown) reliability of CBBD 2022-25 team-season band shares, team-seasons with
  >= 200 located attempts in the class: offence short2 / corner3 / wing3 / top3 = 0.89 / 0.79 / 0.67 /
  0.80; defence 0.69 / 0.76 / 0.52 / 0.75. These are the measurement ceiling on every Read B and
  Read C responsiveness line: a predicted-vs-realised slope is read against sqrt(reliability), and
  wing3 and all defence lines are labelled WEAKER in every table. Independent-vendor evidence
  (Sportradar 2013-18, GameZone 2017-24) confirmed the rim exclusion and is summarised in section 9.
- Coverage handling, fixed in advance: a location feature is NEVER imputed for an unlocated
  attempt. Attempt-level reads (section 3) use located rows only and report the located share
  per cell. Team-game reads (sections 4-5) use realised aggregates from ALL attempts, located or
  not, and report located-only games and all games side by side.

### 1.2 Bands and cells (fixed before any fit)

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

### 1.3 Read A: ceiling (attempt level, located rows, F2 selection, F1 reported)

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

### 1.4 Read B: predictability (the matchup read, F2 selection, F1 reported)

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

### 1.5 Read C: implied game aggregates (the sim-relevant read, offline, no engine run)

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

### 1.6 Decision rule (read on F2; F1 reported, not gated)

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

### 1.7 Registered diagnostics (reported with the round, not gated)

1. The possessions v2 rim override (JumpShot within 2.27 ft -> `FGA_rim`) uses the same
   outcome-dependent rim placement the audit found. Report the override rate for makes vs misses
   within game; if it differs beyond a paired floor, it goes to the possession-layer queue.
2. Second-source check on 2025 only, from 2025-02-10 (the window where hoopR and CBBD agree within
   1 ft on 91-95% of pairs): band shares per team-game from hoopR vs CBBD, and the share of
   attempts whose band differs between the two. Descriptive.
3. Band drift by season and month (section 2), with the Sportradar 2015-18 league shares alongside as
   an older-era reference (docs/tests/sportradar_inventory_2026-10-27.md part 3).
4. Split-half reliability of the team band profiles on the round's own design table, to confirm the
   registered section-1 numbers on the exact population used.

### 1.8 Scripts, artifacts, grader

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

### 1.9 Data-extension options (outside this round; second-vendor sources enter only through a registered agreement check)

Status 2026-10-27 (sources: `docs/research/shot_coordinate_sources_2026-10-27.md`,
`shot_chart_gui_sources_2026-10-27.md`, `bigquery_ncaa_basketball_2026-10-27.md`):

| source | vendor | seasons | independent of ESPN | status (2026-10-27 closing) |
|---|---|---|---|---|
| Google BigQuery public `ncaa_basketball.mbb_pbp_sr` | Sportradar | 2013-14 to 2017-18 ONLY (frozen May 2018) | yes | PULLED (user ruling: private internal research). Rim placement shows the same make-vs-miss asymmetry as ESPN 2022-25 (gap -8.3 to -11.1 pts). About a third of D-I games, lopsided per team: league-level shot-mix window only; band shares from 2015-16 match CBBD 2022-25 within 1-3 pts. CLOSED for team profiles. |
| `JackLich10/gamezoneR-data` frozen files | STATS LLC / Stats Perform GameZone | 2008-09 to 2024-01-15 | yes | PULLED (user ruling). Rim canned on two pixels (outcome-encoding in 2008-11); jumper/three placement 4.9-5.8 ft from CBBD on the same shots, frame unmappable, own band profiles noise (reliability -0.03 to 0.72). CLOSED as a calibration reference and as a training source; descriptive use only. Would cover 23-26% of our unlocated 2022-24 games. |
| Any GUI via API Anything | various | n/a | ESPN widget: no | NOT USABLE (no public x/y, or terms, or NCAA surface) |
| hoopR pre-2022 | ESPN | 2014-2021 at 10-21% of FGA | no | CLOSED (sparse, biased); ESPN 2014-17 shows NO rim asymmetry, so the artefact is vendor- and vintage-specific |

Rim verdict: outcome-dependent rim placement is present in two of three vendors and the data cannot
separate charting from physics (missed layups may genuinely start farther out). Rim location is
CLOSED as a training input from every source on disk (PM ruling 11). The data question for this
round is settled: CBBD 2022-2025 alone.

What a second vendor would be used for, each as its own pre-registered round, never by union
into this round's training table:
1. **Rim charting-bias measurement**: the audit's make-vs-miss placement asymmetry and the
   located-share gap, re-measured on the same games under the second vendor. If the asymmetry is
   absent there, the rim channel can be re-opened with the vendor's rim placement.
2. **Gap fill for 2022-24**: the unlocated ESPN games (skewed to power / neutral / postseason) read
   under the second vendor, entered only after the band-agreement check of section 7 item 2 passes
   on the games both locate.
3. **Backward extension (2013-14 onward, Sportradar)**: a longer training window for the as-of
   band profiles and prior-season carry-over, with the vendor-vs-ESPN agreement check as the gate.

### 1.10 Known gaps this spec does not close

- The foul channel is observable only on and-ones (section 3). A location-aware shooting-foul rate
  needs a feed that carries the fouled attempt, which ours does not.
- The rim class is excluded by ruling, so the one channel where the PAE round saw its largest
  per-shot number is not read here. A rim treatment (collapsing the placeholder pixel and unlocated
  rim rows into one "rim, no placement" category) is a separate pre-registration if a non-ESPN
  source ever allows the charting bias to be measured.
- Player-level location profiles (which shooter takes the corner three) are usage / attribution
  territory and are not in this round.


<!-- RESULTS FOR ROUND 1 APPEND BELOW THIS LINE AS SECTION 2 -->

## 2. Round 1 results (RUN 2026-10-27, Opus worker; appended, section 1 untouched)

Evidence with every table, floor, UNDERPOWERED label and the worker disclosures:
`docs/tests/shot_location_round1_2026-10-27.md`. Grade: `results/shot_location_v1/grade_v1.json` (one blind grader,
`scripts/grade_shot_location_v1.py`, committed `ada9729` before it ran; build and trainer scripts committed `5f44dcb`
before the first fit). Freeze ON; nothing adopted, wired or served.

### 2.1 Settings fixed before any arm

| item | value | evidence |
|---|---|---|
| bands, cells | section 1.2 exactly; `da15` not re-chosen; `cell_p` reproduces the PAE table to 3e-8 | build report |
| k / rho, offence | 100 / 0.5 (second-best 100 / 0.75) | fold-1 training seasons 2022-2023, located-attempt-weighted squared error |
| k / rho, defence | 200 / 0.5 (second-best 400 / 0.5) | same |
| leak test | PASS both families, worst abs corr 0.023 (offence) / 0.035 (defence), every season | evidence T4 |
| population | located 79.6 / 84.5 / 89.3 / 98.7% of design FGA, 2022-2025; rebound miss rows joined to the fg design 88.0% (99.3% within shared games) | evidence section 1 |

### 2.2 Verdicts (section 1.6 on F2; F1 reported)

| class | Read B | Read A make | Read C, carried arm B2 | verdict F2 | F1 |
|---|---|---|---|---|---|
| `FGA_jump2` | MATCHUP | SIGNAL | nothing improves (eFG -0.02, OREB +0.41, and-one +0.84 floors) | **NOT CANDIDATE** (clauses 2, 3) | NOT CANDIDATE |
| `FGA_3` | MATCHUP | SIGNAL | 3P make +1.17 and OREB +1.07 floors, and-one -1.95 floors (worse); responsiveness lines fail | **NOT CANDIDATE** (clauses 2, 3) | NOT CANDIDATE (clause 3) |

Read A per channel (F2, gain in floors, A1 / A2): make jump2 +5.77 / +12.07 SIGNAL; make three +1.18 / +2.36 SIGNAL;
and-one jump2 +10.08 / +6.23 SIGNAL; and-one three +0.04 / -0.38 NO SIGNAL; rebound jump2 +1.76 / +1.89 SIGNAL;
rebound three +2.56 / +7.62 SIGNAL. Decile gate held in every F2 cell (F1: A1 make jump2 and A1 rebound three not held).

Read B (F2, floors = paired game-block bootstrap SE): B1 vs B0 +24.5 (jump2) / +22.6 (three); B2 vs B1 +8.9 / +16.8;
every B2 offence and defence line passes (slopes 0.93-1.05); B3 vs B2 +2.3 / +2.6 on F2 but -0.7 / +0.2 on F1
(recorded; GLM carried); B2alt vs B2 +0.0 / +2.2.

The registered finding clause (Read A SIGNAL with Read B FLAT) does not fire. Not run: rebound arms on the served
`S1_weekly` calendar (S0 used, disclosed). Registered diagnostics: the 2.27 ft rim override fires on makes more than
misses within game by +0.85 / +1.05 / +0.98 / +3.34 pp (SE 0.05-0.08) for 2022-2025, so it goes to the possession-layer
queue; hoopR vs CBBD bands differ on 3.5% (jumpers) / 4.4% (threes) of attempts from 2025-02-10; split-half reliability
on this round's table 0.88 / 0.76 / 0.61 / 0.76 offence, 0.67 / 0.71 / 0.46 / 0.71 defence.


## 3. Round 2: props block (P) and variance block (V) (PRE-REGISTERED 2026-10-27 12:40 ET, PM Fable 5.1; committed BEFORE any fit)

**Origin.** Round 1 (section 2) closed the location stage for team markets: the per-matchup band mix is
predictable and known location explains outcomes, but the implied team-game aggregates sit inside the
floor. The user asked for the two places the value could still live that do not need a clean rim source:
(P) player props, where a shooter-specific band profile could change one player's made-threes
distribution while the team aggregate stays put; (V) variance, where team-game band shares are 1.11-1.31x
the multinomial SD (round 1 finding), so shot selection carries a per-game latent that would change the
width of total and margin distributions without moving their centres. At this commit no fit output for
round 2 exists; none of the section-3.6 scripts exist. Freeze ON. Worker: Sonnet (the day's Opus budget
is spent on round 1); hard stop 15:30 ET, anything unfinished reported NOT RUN. The rim class stays ONE
band (ruling 11); nothing in this round touches rim location.

### 3.1 Data

Round-1 design table `data/processed/models/shot_location/round1/design_loc_cols_v1.parquet` (CBBD 2022-25
located jump2 / three attempts, folded frame, bands per 1.2, `shooter_id`, `game_id`, `game_date`) and
the round-1 team band profiles. Folds as in 1.1 (F2 = 2025 selection; F1 reported). Player-game truth for
block P: hoopR `player_box` (`three_point_field_goals_made / _attempted`, `field_goals_made / _attempted`)
joined on game and athlete; the CBBD-shooter to hoopR-athlete map and its join rate are reported, and
player-games that do not join are dropped and counted, never imputed. Class-mix data for block V: the
possessions v2 chances tables and the SERVED possession-outcome artifact's per-chance class probabilities
on each test season (no refit).

### 3.2 Block P: shooter-specific band profile and band-specific skill

Registered first, before any arm: split-half (odd / even games, Spearman-Brown) reliability of (a) the
shooter's band shares within class and (b) the shooter's make rate by band (corner3 vs non-corner; short2
vs long2), over shooter-seasons with >= 100 located attempts in the class; shooter-seasons under 100 are
UNDERPOWERED. These are the measurement ceiling for every block-P responsiveness line.

As-of features (strictly before tipoff, centred on the snapshot league mean, shrunk; k chosen on fold-1
TRAINING seasons by located-attempt-weighted squared error, before any arm):
- `sh_band_c`: the shooter's centred shrunk band log-odds within class, prior = the shooter's team
  profile (round 1) then league.
- `sh_band_make_c`: the shooter's centred shrunk make deviation WITHIN band, prior = the shooter's overall
  `shooter_shrunk_dev_c` (served).
Leak test on both, aggregated to the team-game (mean over the team's attempts) in change form against
the verified own-game margin, |corr| <= 0.15, REFUSED on failure.

| read | unit | arms | primary |
|---|---|---|---|
| P-A ceiling | located attempt, jump2 and three | `R` served B1 refit (round-1 R), `R_seed`, `P0` = R + band (round-1 A1, re-run), `P1` = P0 + `sh_band_make_c` | log loss gain P1 vs P0 against the 1.3 floor; decile battery; responsiveness: make rate by quintile of `sh_band_make_c` within band |
| P-B predictability | located attempt | `B2` (round-1 carried arm) vs `BP` = B2 + `sh_band_c` | multinomial log loss gain vs floor; responsiveness by quintile of the shooter's own as-of band share |
| P-C player-game | player-game, test season | implied E[3PM] over the player's ACTUAL 3PA in the game (attempt counts are the usage model's job and are held at actual): (i) served R; (ii) P1 marginalised over BP's band mix; same for FGM over jump2 + three attempts with rim held at the served class level | MAE and dispersion (SD of actual - implied vs implied SD) of player-game 3PM and FGM; paired player-game bootstrap floor, 200 replicates; by shooter band-profile quintile and by 3PA-per-game tier (1-2, 3-5, 6+); UNDERPOWERED under 2,000 player-games |

Decision (F2): the shooter band profile is **CANDIDATE (offline, props pathway)** only if P-C improves
player-game 3PM MAE beyond its floor with no tier worse beyond its floor, the P-C responsiveness line by
shooter-profile quintile passes (slope ratio in [0.8, 1.2] read against sqrt(reliability)), and P-A or P-B
beats its floor. Otherwise NOT CANDIDATE. Serving would need the usage model to carry the shooter's band
mix and is its own round.

### 3.3 Block V: the per-game shot-selection latent (DIAGNOSTIC, no arm; outcome is DEFECT CONFIRMED / NOT CONFIRMED)

- **V-A structure.** Per team-game residual shares: actual minus B2-predicted band shares (jump2 and three
  bands), and actual minus served-possession-outcome-predicted CLASS shares (3PA, rim, jump2, TOV share of
  chances). Report: over-dispersion ratio (realised SD of the residual / SD implied by the multinomial at
  the game's attempt count) per band and class with a parametric-bootstrap floor; the within-team-game
  correlation matrix of residuals across bands and classes; share of variance in the first principal
  component; correlation of the residual with possession count, final margin, site, and the opponent's
  defence profile.
- **V-B served under-dispersion.** For each test season: implied per-game SD of 3PA share and rim share
  from the served possession-outcome per-chance probabilities (sum of Bernoulli variances) vs the realised
  SD of the residual, overall, by season, by offence quintile; ratio with floor. Ratio > 1 beyond the floor
  = DEFECT CONFIRMED for the possession-outcome owner, and the latent's SD (logit-normal random effect by
  method of moments, per class) is the registered quantity a future variance arm must reproduce.
- **V-C shared latent.** Correlation of the team-game residual 3PA share with the residual corner3 / wing3 /
  top3 and short2 / long2 shares. If the band latent and the class latent are one latent, the fix belongs
  in possession outcome, not in a location stage; state which.

### 3.4 Order of work

V-A, V-B, V-C first (cheap, decisive for the variance queue); then P reliability, leak test, P-B, P-A, P-C.

### 3.5 Metrics conventions

As in 1.3-1.5: floors are max(seed difference, paired block bootstrap SE); GLM fits have a zero seed
floor; decile battery as served; UNDERPOWERED labels; located-only vs all-games side by side where
applicable; F1 reported and labelled BIASED SUBSET.

### 3.6 Scripts, artifacts, grader

`scripts/build_shooter_band_v1.py`, `scripts/train_shot_location_v2_props_v1.py`,
`scripts/diag_shot_selection_dispersion_v1.py`, ONE blind grader `scripts/grade_shot_location_v2.py`.
Artifacts under `data/processed/models/shot_location/round2/` (gitignored, HF `model_artifacts`).
Evidence `docs/tests/shot_location_round2_2026-10-27.md`; results appended as section 4; ledger row in
the same commit; suite re-run; parity v10 unchanged.

<!-- RESULTS FOR ROUND 2 APPEND BELOW THIS LINE AS SECTION 4 -->

## 4. Round 2 results (RUN 2026-10-27, Sonnet worker; appended, sections 1-3 untouched)

Evidence with every table, floor, UNDERPOWERED label and the worker disclosures: `docs/tests/shot_location_round2_2026-10-27.md`. Grade: `results/shot_location_v2/grade_V.json`, `grade_P.json` (ONE blind grader `scripts/grade_shot_location_v2.py`, block V committed `c2a3504`, block P committed `f13137f`, both before they ran; build `18b8f69`, trainer `84350bf` before the first fit). Freeze ON; nothing adopted, wired or served.

### 4.1 Block V (diagnostic): DEFECT CONFIRMED for the possession-outcome owner on 3PA, rim, jump2 and TOV

Served `round2_s1` F2 2025 (853,931 chances, 10,890 team-games; log loss reproduces 1.51543 / 1.49976). Over-dispersion ratio = SD of the team-game share residual / SD implied by the sum of Bernoulli variances; parametric-bootstrap null mean 0.999-1.001; floor = max(null SD, game-bootstrap SE).

| class | ratio | floor | excess (floors) | outcome | latent s2 | **latent SD (logit-normal, registered)** |
|---|---|---|---|---|---|---|
| 3PA share | 1.253 | 0.0088 | +28.8 | DEFECT CONFIRMED | 0.0369 | **0.192** |
| rim share | 1.360 | 0.0097 | +37.0 | DEFECT CONFIRMED | 0.0560 | **0.237** |
| jump2 share | 1.473 | 0.0119 | +39.8 | DEFECT CONFIRMED | 0.1034 | 0.322 |
| TOV share | 1.104 | 0.0073 | +14.2 | DEFECT CONFIRMED | 0.0221 | 0.149 |
| FT trips | 0.946 | 0.0068 | -7.8 | NOT CONFIRMED (under-dispersed) | -0.0140 | 0 |

Bands (B2, team-games with >= 5 located attempts): short2 1.258, corner3 1.200, wing3 1.097, top3 1.144 (F2); 1.255 / 1.181 / 1.085 / 1.118 (F1, BIASED SUBSET); latent SD 0.455 / 0.353 / 0.200 / 0.259 (F2), 0.441 / 0.351 / 0.195 / 0.244 (F1). The ratio is flat across offence quintiles (3PA 1.20-1.28), site does not correlate with any residual, margin correlates with TOV (-0.22) and weakly with rim / 3PA / FT (+0.06 to +0.08).

**V-C: the band latent and the class latent are NOT one.** The 3PA residual is uncorrelated with every band residual (raw -0.015 to +0.010 against null +/-0.001; disattenuated -0.06 to +0.03, SE 0.025-0.04); PC1 of the nine residuals carries 18.2% against 18.5% under the null. The only coupling is the rim / short-jumper boundary (rim vs short2 -0.135 raw, null -0.0005, disattenuated -0.33; jump2 vs short2 +0.141; rim vs jump2 -0.448 against a null -0.281), the boundary the 2.27 ft rim override acts on (round-1 diagnostic 1). The class latent (3PA, rim, jump2, TOV; SDs above) belongs in possession outcome; the band latent is a separate location-stage quantity. F1 class part NOT RUN (no F1 served artifacts).

### 4.2 Block P: NOT CANDIDATE under the literal 3.2 rule (F2; F1 agrees)

Reliability (shooter-seasons >= 100 located attempts; 764 jump2, 2,427 three): band share 0.934 (short2) / 0.817, 0.585, 0.837 (corner3, wing3, top3); make rate 0.257 / 0.302 (short2 / long2), 0.058 / 0.135 (corner3 / non-corner); make-rate contrasts 0.173 and 0.044. Leak test PASS (worst abs corr 0.041). Player map: `player_crosswalk_v2`, attempt join 100%, player-game to hoopR box 99.86% (2025) / 99.78% (2024). k: k_s 5 / 20, k_m 200 / 400 (jump2 / three), fold-1 training.

| read (F2) | jump2 | three |
|---|---|---|
| P-A P1 vs P0 LL gain (floor) | +4.83e-4 (1.63e-4) = +2.96 floors, gate held, SIGNAL | -5.2e-5 (1.30e-4) = -0.40 floors, NO SIGNAL |
| P-A P0 vs R (round-1 A1 replicated) | +5.77 floors | +1.18 floors |
| P-B BP vs B2 LL gain (floor) | +0.0477 (0.0010) = +47.5 floors, short2 line 0.99 | +0.0219 (0.00047) = +46.7 floors, lines 0.99 / 0.92 / 1.03 |

P-C (player-game, attempts held at actual, 71,159 3PM / 95,038 FGM player-games): 3PM MAE 0.6827 -> 0.6823, gain +0.000425 = **+2.18 floors**, no tier worse (+1.81 / +0.14 / +2.00 floors); FGM +0.000782 = +3.80 floors; dispersion ratios 1.005 / 1.003 (3PM). **Responsiveness by shooter-profile quintile (3PM): slope ratio 2.05 (F1 1.35), outside [0.8, 1.2] -> clause fails; FGM line passes (0.99; F1 1.15).** Post-hoc, descriptive: the slope of (actual - R-implied) on the shift is 0.78 [0.63, 0.93] F2 and 0.88 [0.75, 1.00] F1; the registered ratio has a small denominator (realised span 0.7 pp) and a bootstrap 95% interval [1.10, 8.11] (F2). Verdict: **NOT CANDIDATE** (clauses 1, 2, 4 hold; clause 3 fails).

Not run: block V class part on F1; sensitivity fit under the second-best (k_s, k_m). Suite and parity v10 in the evidence doc (section 4).
