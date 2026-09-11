# Rotation model (L4) — who is on the floor, and for how long

Status: **BAKE-OFF RUN 2026-09-10/11, SIX rounds, NO ARM ADOPTED.** Round 6
conditioned the substitution COMPOSITION on who left (L33) and produced the best
per-player minutes of any arm in six rounds -- `K1_cond_class` 8.862 min against
R2's 9.794 and round 5's W4 8.928, with the best per-player minutes K-S D ever
measured (0.0589) and 5 of 6 G8 cells, which only the incumbent has matched --
and it is **not adopted**: it passes 4 of the 8 state cells, and it loses the
second as-of-minutes quintile to W4 by 0.060 min against a 0.016 floor, which the
pre-registered responsiveness condition vetoes. Round 6 settles two questions and
opens one. **The temperature story is dead**: the entry-weight exponent `tau` is
fitted at **1.00 in all six S1 windows** -- exactly round 5's W4 -- and the
likelihood rejects the rank direction by 320,000-350,000 log units, so no
temperature between the two knob-free endpoints is preferred to the one already
in use. **The conditional story is real and half-works**: a substitution REVERSES
the class (a bench player leaving puts a starter on 0.749, a starter leaving puts
one on 0.316, measured on both seasons), W4's unconditional race realises a 20 pp
spread against a real 34 pp, and K1 realises 43 pp -- past the target rather than
onto it. **The binding defect has moved to the EXIT side**: every arm since round
5 takes a starter off at 0.456-0.467 of single swaps against a real 0.587, a
12-13 pp miss, because the rank exit rule was carried over unchanged. Round 6
also measures, report-only, what the gate cannot separate: re-grading the ACTUAL
sequence with the model's as-of starter set moves the opening-ten-minutes cell
**-4.4 pp** and both close-late bands -2.8 pp, so those cells are unreachable for
starter-identification reasons and the arms sit within 1.2 pp of the honest
benchmark; foul trouble, at +0.2 pp of benchmark, is the one veto cell whose
failure is genuinely the model's own. Round 4
changed model family (L25): per-player discrete-time substitution hazards
(sub-out over the five on the floor, sub-in over the eligible bench) in place of
a minutes budget or a donor sequence. It **produces both structural facts three
rounds could not** — the second-half tip goes from R2's 0.79 to 0.94 against a
real 0.97, and the opening ten minutes of a close game from R2's 0.60 to 0.74
against a real 0.78 — **passes the blowout band for the first time in four
rounds**, **passes the within-player minutes SD ratio for the first time ever**,
and beats R2 on per-player minutes MAE by 0.98 min against a 0.015 min floor.
It is **not adopted**: it loses the close-and-late band (0.669 against 0.749,
where R2 is 0.725), so no arm has all eight state cells and the pre-registered
rule adopts nothing. The mechanism is measured, not guessed: the arms substitute
**43% too often** (0.206 against a real 0.152 at a possession boundary) because
independent per-player Bernoulli exits cannot represent a coordinated dead-ball
substitution wave — the same defect that stops the *earned*-reset arm (H2) at
0.85 while its fitted boundary hazards are right to 1 pp. Round 5's arm is this
family with a **joint** dead-ball substitution draw. Also this round: the
incumbent's first-ever Decision-10 closed-loop gate (it passes), and an engine
loop defect fixed (§7 item 9). Earlier rounds: Round 3 added
a fitted close-game keep-starters override in two functional forms (R7 logistic,
R8 per-cell threshold) and adopted neither; it also closed §5's hazard-matrix
defect, whose corrected re-run makes the donor family **worse** than the column
round 2 reported. `R2_hier_dirichlet` remains the best arm in three rounds, but
round 3b shows its "three of four state cells" is boundary-thin — its 6-15 band
was −2.97 pp against a −3.00 pp tolerance and flips to FAIL on a sub-noise move,
so it is **2 of 4 under S1**. Round 3b also **adopts S1 (in-season monthly
walk-forward refit, L21) as this model's training scheme** — no gate regressed
beyond the noise floor and six cells improved beyond it — and shows the round-3
override knobs are **not identified** (R7's keep scale spans the whole grid
across six windows of one season). Read `experiments.md` for the
pre-registrations and the full tables; this file leads with the conclusion and
the relative comparisons.

---

## 1. Purpose

At every possession the possession engine needs the five players on the floor for
each team. That set drives the entire player layer: usage allocation, per-player
rate stats, and every prop. `docs/SIM_GUARDRAILS.md` G8 grades the result on
per-player minutes (mean and SD), the share of team minutes taken by the top
players, the number of players with more than zero minutes, and the per-player
count distributions — broken out by final-margin band, because "bench minutes in
blowouts affect props far more than game markets" (§4). This model is the
stochastic process that produces the five-player sets; it sits between the
possession-outcome layer (L3) and usage allocation, and is called once per
possession boundary per team.

L15 is the reason this layer has to be good rather than adequate: every
player-game rate stat keys on `athlete_id` first (player-beyond-both 10–52% of
variance against 1–3% for coach and 0.3–1.2% for team-season), so *which*
players are on the floor dominates what the possession produces. A rotation
model that gets team minutes right while mis-assigning them across players
produces correct team totals and wrong props.

---

## 2. Target variable

**The on-floor set.** One row per (game, team, possession): the five CBBD player
ids recorded on the possession's first event, from
`data/processed/possessions/possessions_{season}.parquet`
(`on_floor_h1..h5` / `on_floor_a1..a5`).

- **Population.** 2024 and 2025 only — CBBD `onFloor` is empty at the source in
  2022–23 (L13). A game enters only if every possession has a complete ten-player
  on-floor list: 5,320 of 5,593 D-I non-truncated 2025 games (95.1%). Both teams
  must also have at least 3 earlier games that season for an as-of profile to
  exist, which leaves **4,689 eligible 2025 games**.
- **Derived minutes.** A player's minutes are the summed `duration_s` of the
  possessions he is on the floor for. Validated against hoopR box minutes through
  the crosswalk on 2025: corr **0.984**, mean diff **+0.14 min**, MAD **0.63
  min**. Sim and actual are computed by the same function throughout, so every
  comparison is like-for-like.
- **Distribution snapshot (2025 actual).** 9.62 players with > 0 minutes per
  team-game (201.8 team-minutes); top-5 share of team minutes 0.748, top-8 0.956;
  top-1 five-man lineup takes 0.296 of team possessions, top-3 0.545, top-5
  0.692; 14.7 distinct lineups per team-game; the on-floor set changes at 15.2%
  of possession boundaries (≈ 21 changes per team-game, 1.35 players per change).
  Starters take 70% of on-floor slots over a whole game, and in the final eight
  minutes of regulation **0.751 / 0.723 / 0.524** of them at \|margin\| ≤ 5 /
  6–15 / > 15.

---

## 3. Methodology at a glance

- **Data window.** Train 2024, test 2025 (F1, the selection fold). Robustness:
  within-2025 walk-forward, train games before 2025-01-15, test after. 2026 is
  sealed and `seal.assert_not_sealed` guards both trainers.
- **Split strategy.** Temporal walk-forward, no random split. Every pregame
  feature is a `shift(1)` expanding statistic over the team's own game order, and
  `tests/test_rotation.py` proves leak-safety two ways (independent recomputation
  of the strictly-earlier average, and invariance to corrupting the game's own
  rows).
- **Isolation.** The bake-off conditions on the **real possession sequence** of
  each test game — durations, period structure, score path — and simulates only
  *who is on the floor*. That grades the rotation model without the pace and
  possession-outcome models' errors leaking in.
- **Primary metric.** There isn't one: the pre-registration counts **cells inside
  tolerance** across G8 (minutes mean ± 2.0, SD ratio 0.9–1.1, top-5 and top-8
  share ± 2 pp, players with > 0 minutes ± 1.0) and state dependence (starters'
  share of the final eight minutes by margin band, and starters' share while
  carrying ≥ 4 fouls, each ± 3 pp), with an eligibility veto on the
  state-dependence cells. The veto is the point: a rotation model that reproduces
  minutes but not blowout and foul-trouble behaviour is useless for props.
- **Model families tested.** Round 1: single-level Dirichlet + scheduler (R1),
  hierarchical Dirichlet + the same scheduler (R2), per-possession logistic stint
  hazards (R3), empirical stint-sequence resampling (R4). Round 2: R2 as the
  incumbent reference, a donor-sequence/override hybrid (R5), and R3 respecified
  with player × state interactions (R6).
- **Noise floor.** 50 seed-varied runs per arm on 150 games. Every G8 and
  state-dependence gap discussed below is 3–10× the measured seed SD, so none of
  the failures are seed noise.

---

## 4. The CBBD ↔ ESPN player crosswalk (a prerequisite, not a result)

`docs/SIM_GUARDRAILS.md` §4 makes this a reported number. Built by
`scripts/build_player_crosswalk.py` / `src/cbb_sim/data/player_ids.py` in **3
CBBD API calls** (one `/teams/roster?season=S` per season returns every team's
roster, 1,519 team-rows for 2025), far inside the 300-call budget:

| Scope | Roster rows | Row match | On-floor players matched | **Possession-weighted match rate** |
|---|---:|---:|---:|---:|
| Overall | 34,938 | 99.94% | 14,965 / 14,986 (99.86%) | **99.9868%** |
| 2024 | 14,056 | 100.00% | 4,945 / 4,945 (100.00%) | **100.0000%** |
| 2025 | 15,215 | 99.97% | 5,046 / 5,051 (99.90%) | **99.9991%** |
| 2026 (sealed, reported only) | 5,667 | 99.72% | 4,974 / 4,990 (99.68%) | **99.9618%** |

By method: 33,375 `source_id_verified`, 1,534 `source_id_unverified`, 8
`name_team_pbp`, 21 `unmatched`. The crosswalk is essentially exact because CBBD's
roster endpoint carries the ESPN athlete id as `sourceId`; name/jersey matching is
only a fallback and did no real work. Downstream it is verified three ways: the
`source_id_verified` ids all exist in that season's hoopR `player_box`
(`tests/test_rotation.py`), 99.977% of as-of feature rows resolve and 99.758% join
a hoopR box row, and hoopR's `did_not_play` agrees with "zero on-floor
possessions" on 99.02% of them.

---

## 5. Result: no arm adopted, in either round

### 5.1 Round 1 (1,200 of 4,689 eligible games, 5 seeds, 86,687 rotation player-games)

| arm | G8 cells | state cells | eligible | what it gets right | what it misses |
|---|---:|---:|---|---|---|
| R1 `dirichlet` | 5/6 | 2/4 | NO | minutes level and pooled dispersion, top-5/top-8 share, nonzero count | within-player SD ratio 1.30; pulls starters too early in blowouts (−6.1 pp at \|margin\| > 15) and at 6–15 (−3.3 pp) |
| R2 `hier_dirichlet` | 4/6 | **3/4** | NO | the most cells of any arm (7); both close-game bands and foul trouble | within-player SD 1.35, top-5 share +2.1 pp, blowout band −5.2 pp |
| R3 `stint_hazard` | 2/6 | 1/4 | NO | pooled SD | everything concentrated: top-5 +3.1 pp, top-8 +2.6 pp, 8.57 players used, all three late bands |
| R4 `stint_resample` | 4/6 | 1/4 | NO | **lineup concentration almost exactly** (K-S D 0.036 on the top-1 lineup share vs 0.19–0.25 for every other arm) | over-benches in blowouts (+8.1 pp) and under-benches in close games (−5.6 pp) |

The relative picture is the finding. **The minutes-share arms and the
stint-sequence arms fail in opposite directions**: R1/R2 reproduce the minutes
table and fail lineup concentration (top-1 lineup share 0.231/0.238 against a real
0.296) and over-respond to blowouts; R4 reproduces lineup concentration to within
seed noise and fails the minutes table and the state response. That is why round 2
was specified to separate "who plays together" from "when the coach deviates".

### 5.2 Round 2 (1,600 of 4,689 eligible games, subset seed 2025, 5 seeds, 115,702 rotation player-games)

| arm | G8 cells | state cells | eligible | what it gets right | what it misses |
|---|---:|---:|---|---|---|
| R2 `hier_dirichlet` (incumbent) | 4/6 | **3/4** | NO | both close-game bands (-2.1, -3.0 pp) and foul trouble (+0.4 pp); reproduces its round-1 numbers to four decimals on a different game subset | blowout band -4.7 pp; within-player SD 1.354; top-5 share +2.1 pp |
| R5 `hybrid` | 2/6 | 2/4 | NO | **lineup concentration better than anything else tried** (top-1 lineup share 0.2876 vs a real 0.2940, K-S D **0.029**); blowout band -2.9 pp; foul trouble +0.7 pp | both close-game bands (-4.3, -6.9 pp); 8.57 players used; top-5 +2.9 pp, top-8 +2.3 pp |
| R6 `stint_hazard_v2` | 2/6 | 2/4 | NO | the respecification works where round 1's R3 could not: blowout band +2.0 pp (R3 was +4.0) | both close-game bands (-6.5, -7.2 pp); 8.58 players used; lineup concentration worst of all six arms (K-S D 0.265) |

**No arm is eligible: every one misses at least one state-dependence cell, and the
misses are structured.** R2 is right in close games and too bench-heavy in a
blowout; R5 and R6 are right in a blowout and too bench-heavy in close games. The
round-2 arms did what they were specified to do — R5 reproduces which fives play
together almost exactly, and R6's interactions give the hazard a blowout response
R3 could not represent — but neither holds the close-game end while doing it.

**A defect qualifies the R5 and R6 columns.** `build_hazard_training` never
advanced the foul counter, so every fitted hazard (R3's in round 1, R5's and R6's
in round 2) saw `fouls = 0` on every training row; all five foul coefficients came
back as exactly `+0.0000`, which is how it was caught. R1, R2 and R4 use no hazard
and are unaffected (R2's two runs agree to four decimals). The matrix is fixed and
R5's overrides re-fitted — the foul terms now carry the expected signs, reported in
`experiments.md` §5 — but the corrected full re-run was cut off by the session's
hard stop and is **OPEN**. Section 4's R5/R6 columns are inert-foul-term results.
Note also that both arms *passed* the `>= 4 fouls` cell with an inert foul term,
because that state pools fouled-out players (forced off, share 0) with four-foul
players; the pre-registered "at exactly 4 fouls" diagnostic row exposes it at
0.82 / 0.80 against a real 0.52.

### 5.3 Round 3 (same 1,600 games, 5 seeds; `experiments.md` §6-7)

Round 3 ran the incumbent, the **corrected** R5 (closing §5's OPEN item) and two
fitted close-game keep-starters overrides: R7, a logistic on-floor propensity
applied as a state deviation with a per-game tolerance draw, and R8, the simpler
threshold form (one fitted scalar against the training season's own
starters'-share-by-(time × margin) table).

| arm | G8 | state | eligible | final-8:00 b0 / b1 / b2 / ≥4 fouls (actual 0.7491 / 0.7240 / 0.5223 / 0.4613) |
|---|---:|---:|---|---|
| R2 `hier_dirichlet` | 4/6 | **3/4** | NO | 0.7281 / 0.6943 / 0.4750 / 0.4648 |
| R7 `keep_logistic` | 2/6 | 0/4 | NO | 0.7074 / 0.6749 / 0.5994 / 0.5125 |
| R5 `hybrid` (corrected) | 1/6 | 0/4 | NO | 0.6942 / 0.6566 / 0.5911 / 0.4964 |
| R8 `keep_cell` | 1/6 | 0/4 | NO | 0.6982 / 0.6599 / 0.5956 / 0.4997 |

Three findings, all in `experiments.md` §7.9.

1. **Fixing the foul defect made the donor family worse.** With the corrected
   hazard matrix R5's blowout band goes from −2.9 pp (a PASS under the defect) to
   **+6.9 pp**, and foul trouble from +0.7 to +3.5 pp. The corrected foul terms
   take load off the margin terms, the block stops emptying the bench when the
   game is decided, and the arm plays its starters through garbage time. §5 is
   closed and its answer is negative.
2. **The close-game cell is unreachable by this override family.** Over R7's
   whole 24-point knob grid the largest close-band share produced on the training
   season is 0.6994 against a target of 0.7110; raising the keep scale or its
   base rate *lowers* it. The fitted starter-vs-bench separation of the keep
   score in the close-and-late state is only ≈1.0 in log odds, so any base rate
   large enough to keep starters also keeps bench players, and each kept bench
   player displaces an un-kept starter. R8's grid does reach b0 = 0.7475 at
   θ = 1.0, but only by pushing b1, b2 and foul trouble further out, so the joint
   fit selects θ = 0.1 and R8 is R5 with a rounding error.
3. **Neither keep arm clears the refit noise floor.** R7 gains +1.3 / +1.8 pp on
   the two close bands; a spec-identical refit under a second seed moves the same
   cells 1.5 / 1.4 pp. R8's single knob is outright unstable across refits
   (θ 0.1 vs 0.3, close band moving 6.1 pp — twice the gate tolerance).

Two things round 3 banks: R7 has the **best lineup concentration of any arm in
three rounds** (top-1 five-man lineup share 0.2888 against a real 0.2940, K-S D
**0.0302**, per-player minutes K-S 0.0782), and every arm's quintile slope
against the pregame team prior is right (actual +0.692; R2 +0.645, R5 +0.742,
R8 +0.728, R7 +0.826), so all four failures are level failures, not
responsiveness failures.

### 5.4 The diagnostic that matters most: starter identification vs rotation

The state-dependence rows count, on each side, the five that side actually
started. The model's as-of starter set overlaps the real starting five on **4.57
of 5** players, so part of every gap is picking the wrong fifth man rather than
rotating him wrongly. Re-grading the **actual** on-floor sequence with the
**model's** starter set splits the two:

| cell | ACTUAL (own starters) | ACTUAL (as-of starter set) | of which: starter identification |
|---|---:|---:|---:|
| final 8:00, \|margin\| ≤ 5 | 0.7513 | 0.7256 | −2.6 pp |
| final 8:00, \|margin\| 6–15 | 0.7230 | 0.6947 | −2.8 pp |
| final 8:00, \|margin\| > 15 | 0.5237 | 0.5080 | −1.6 pp |
| ≥ 4 fouls | 0.4596 | 0.4630 | +0.3 pp |

So roughly 1.6–2.8 pp of each late-window miss is "wrong five", and the rest is
genuinely "wrong rotation". A recency-weighted starter predictor (EWMA of the
starter flag, decay fitted on 2024) already bought 4.22 → 4.60 of 5 over a flat
expanding mean; closing the remaining 0.4 players needs availability information
the as-of feature set does not have (injury reports), not a better estimator.

### 5.5 Round 4 (same 1,600 games, 3 seeds under S1 + 3 static; `experiments.md` §10-11)

A new family: two per-player discrete-time hazards at every possession boundary,
`p_out` over the five on the floor and `p_in` over the eligible bench, with
exits drawn as independent Bernoulli and entrants taken in an
Efraimidis–Spirakis exponential race weighted by `p_in / (1 − p_in)`. Three
arms: **H1** logistic + a hard reset to the predicted starters at the first
possession of period 2, **H2** the same hazards with the reset left to be
*earned*, **H3** H1 with a LightGBM hazard. `R2_hier_dirichlet` is the
reference, under S1 and static.

| cell (S1) | ACTUAL | R2 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|---:|
| final 8:00, \|m\| ≤ 5 | 0.7491 | **0.7254** ✓ | 0.6692 | 0.6719 | 0.6797 |
| final 8:00, \|m\| 6–15 | 0.7240 | 0.6914 | 0.6518 | 0.6489 | 0.6636 |
| final 8:00, \|m\| > 15 | 0.5223 | 0.4694 | **0.5446** ✓ | **0.5471** ✓ | **0.5517** ✓ |
| starters at ≥ 4 fouls | 0.4613 | **0.4626** ✓ | **0.4344** ✓ | **0.4341** ✓ | 0.4005 |
| **H2 tip, ≤ 5 / 6–15 / > 15** | 0.968 / 0.961 / 0.956 | 0.789 / 0.798 / 0.734 | **0.939 / 0.937 / 0.943** ✓ | 0.855 / 0.849 / 0.848 | **0.939 / 0.942 / 0.945** ✓ |
| **H1 20:00–10:00, ≤ 5** | 0.7822 | 0.5988 | 0.7434 | 0.7450 | **0.7581** ✓ |
| per-player minutes MAE (min) | 0.0 | 9.794 | **8.819** | 8.897 | **8.674** |
| within-player minutes SD ratio | 1.000 | 1.342 | **1.079** ✓ | **1.080** ✓ | **1.022** ✓ |
| distinct lineups / team-game | 14.84 | 15.51 | 19.73 | 19.93 | 20.41 |
| substitution rate at a boundary | 0.1518 | 0.1437 | 0.2058 | 0.2052 | 0.2050 |

Reading, in the order the evidence forces:

1. **The family reaches the cells the round existed for.** The two structural
   facts of L25 are produced, the blowout band passes for the first time in four
   rounds, and the within-player minutes SD ratio — which had failed for every
   arm in every round — passes on all three hazard arms. Per-player minutes MAE
   improves by 61–76 noise floors. None of it comes from a knob; round 4 has
   none, and a spec-identical refit under a second seed moves six of eight cells
   by 0.04–0.81 pp (round 3's override knobs were *not identified*).
2. **The reset must be given, not earned — and the fit is not what fails.** On
   the 2024 training rows the model predicts a bench player's exit at a period
   boundary at **0.786** against an actual **0.791**, and an off-floor starter's
   entry at **0.771** against **0.780**. H2 still lands 11 pp short of the tip.
   Independent Bernoulli exits capped at the bench size realise about four
   fifths of a coordinated two- or three-player swap, whatever the marginal
   rates are.
3. **The cell it breaks is the one it did not previously fail, and by the same
   mechanism.** The arms substitute 43% too often, run 19.7 distinct lineups
   against 14.84, and compress the late-game spread across margin bands to
   12.4 pp against a real 22.7 pp. Real substitutions bunch at dead balls;
   independent draws spread the same total exits over more boundaries, and every
   extra churn resamples the floor toward the unconditional mix.
4. **The two families fail in orthogonal ways.** R2 gets the close-and-late
   *level* right and the within-game *shape* badly wrong (−16 to −22 pp at the
   tip, −18.3 pp over the opening ten minutes). The hazard family gets the shape
   right and the level wrong. Round 5 is this family with a joint dead-ball
   substitution mechanism, not a third family.

Decision 8 is satisfied by all four arms (slope +0.629 / +0.628 / +0.672 /
+0.706 against an actual +0.692, monotone 4 of 4) and separates none of them.

---

### 5.6 Round 5 (same 1,600 games, 3 seeds under S1; `experiments.md` §12-13)

Round 5 changed the **draw**, not the family. Round 4's fitted per-player
hazards are reused byte for byte as the composition rule and round 5 adds one
object: a 324-cell table (`prev_end` × time cell × margin band × foul state)
carrying a per-(team, boundary) **wave probability** and a **wave size**, plus
one fitted coupling scalar `rho`. Five arms, the four knob-free members of the
composition grid (rank or draw, on each of the exit and entry sides) plus a
coupled variant of the first.

| | ACTUAL | R2 | H1 (round 4) | W1 rank/rank | W2 draw/draw | W4 rank/draw | W5 draw/rank | W3 coupled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **substitutions per boundary** | 0.1509 | 0.1437 | **0.2058** | 0.1570 | 0.1571 | 0.1570 | 0.1575 | 0.1568 |
| **distinct lineups / team-game** | 14.84 | 15.51 | 19.73 | 11.52 | 16.96 | **14.47** | **14.41** | 11.50 |
| per-player minutes MAE | 0.0 | 9.794 | **8.819** | 9.124 | 8.965 | 8.928 | 9.035 | 9.135 |
| final 8:00, \|m\| ≤ 5 | 0.7491 | 0.7254 | 0.6692 | **0.7281** | 0.6494 | 0.7135 | 0.6767 | **0.7288** |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6914 | 0.6518 | **0.7149** | 0.6301 | 0.6914 | 0.6628 | **0.7118** |
| H2 tip, \|m\| ≤ 5 | 0.9678 | 0.7892 | 0.9390 | 0.9367 | 0.9377 | **0.9402** | 0.9319 | 0.9367 |
| H1 20:00-10:00, \|m\| ≤ 5 | 0.7822 | 0.5988 | 0.7434 | 0.7444 | 0.7286 | 0.7332 | 0.7441 | 0.7430 |
| state cells passed (of 8) | | 2 | 5 | 5 | 3 | 4 | 3 | 5 |
| new cells passed (of 2) | | 2 | **0** | 1 | 1 | **2** | **2** | 1 |

**No arm adopted** — no arm passes all eight state cells and both new cells.
Three readings the round does establish:

1. **The joint draw removes round 4's over-substitution outright.** Every arm
   sits at 0.1568-0.1575 per boundary against a real 0.1509 (+4%) and a floor of
   0.0011-0.0020, where round 4's arms were +36%. It costs nothing on the
   primary metric: every wave arm still beats R2 by 45-56 noise floors on
   per-player minutes MAE. The timeout feature excluded for engine
   expressibility since round 4 is **not** needed to make substitutions bunch.
2. **The close-and-late cell is fixed for the first time in the hazard family.**
   W1 −2.1 pp and W3 −2.0 pp against R2's −2.4 and H1's −8.0; the \|m\| 6-15
   band, which no arm in five rounds had passed, passes at −0.9 and −1.2 pp.
3. **The failure moved from WHETHER to WHO.** A ranked entry rule makes the
   floor sticky and concentrates the rotation too far (8.08 players against
   9.64, top-5 minutes 0.800 against 0.747, 11.5 distinct lineups against 14.8);
   a drawn entry rule gets the breadth exactly right (14.47 / 14.41 distinct
   lineups, the best per-player-minutes K-S D of any arm in five rounds) and
   gives back 3-4 pp of late stickiness. The truth is between a zero-temperature
   race and a unit-temperature one, and no knob-free rule can express it.

### 5.7 Round 6 (same 1,600 games, 3 seeds under S1; `experiments.md` 14-15)

Round 6 changed the **entry rule** and nothing else: round 5's wave tables,
round 4's hazards, round 5's rank exit rule and the hard second-half reset are
all reused byte for byte, so a difference between a round-6 arm and W4 is a
difference in the entry composition alone. Three arms -- **T1** one fitted
temperature on the entry weights, **K1** the entry class count conditioned on how
many predicted starters left, **A1** a fitted 4x4 tier-pair log-affinity -- with
R2, W1 and W4 as reference columns from round 5's JSON (a 1-seed W4 reproduction
re-run inside this round moves every state cell by less than its floor).

| | ACTUAL | R2 | W1 | W4 | T1 | K1 | A1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-player minutes MAE | 0.0 | 9.7939 | 9.1244 | 8.9281 | 8.9266 | **8.8622** | 8.9167 |
| K-S D, per-player minutes | -- | 0.0798 | 0.1152 | 0.0747 | 0.0752 | **0.0589** | 0.0730 |
| state cells passed (of 8) | | 2 | **5** | 4 | 4 | 4 | 4 |
| round-5 cells passed (of 2) | | 2 | 1 | **2** | **2** | **2** | **2** |
| G8 cells passed (of 6) | | 5 | 2 | 3 | 3 | **5** | 3 |
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7254 | 0.7281 | 0.7135 | 0.7096 | 0.7001 | 0.7089 |
| starters at >= 4 fouls | 0.4613 | 0.4626 | 0.4432 | 0.4216 | 0.4286 | 0.4212 | 0.4295 |
| distinct lineups / team-game | 14.836 | 15.514 | 11.515 | 14.474 | 14.442 | 14.571 | 14.341 |

Reading, in the order the evidence forces:

1. **`tau = 1.00` in all six windows kills the temperature hypothesis.** T1 is
   therefore W4's rule exactly, and its cells reproduce W4's within the seed
   floor -- a free extra noise reading, and a negative answer to round 5's own
   option (a).
2. **The conditional object is real, and K1 represents it too strongly.** The
   measured joint (a class-reversing swap, 0.749 / 0.316 at size 1 on 2024 and
   0.737 / 0.323 on 2025) is missed by W4's race (20 pp of spread against 34) and
   overshot by K1 (43 pp).
3. **The defect is now on the exit side**, by 12-13 pp of the starter share of
   single-swap leavers, and no entry rule can pay while the leaver mix is wrong.
4. **Most of what the late and opening cells still read is starter
   identification**, not rotation: against the as-of-starter benchmark the arms
   are -1.2 / -0.4 / +1.7 pp on the three late bands and -0.5 pp on the opening
   ten minutes. Foul trouble is the exception and is the model's own miss.
5. **Decision 10 passes at both seed counts**: live/frozen margin SD ratio
   0.9998 at 5 seeds and 0.9934 at 25, with possessions moving 0.027 and 0.016
   against a G1 tolerance of 1.0. Round 6 adds no margin, clock or foul term to
   size under L31 -- the objects that carry state are round 5's, already sized
   there -- so the freeze is the instrument that applies and it finds no new
   channel.


---

## 6. Garbage time: how each arm produces it, from data rather than a rule

The measured target is a collapse of the starters' share of on-floor slots in the
final eight minutes from 0.751 (close) to 0.524 (blowout). No arm has a hand-set
margin threshold anywhere; each gets the behaviour from a different fitted object.

| arm | mechanism | fitted from |
|---|---|---|
| R1, R2 | a **state-tilt lookup table** `TiltTables.state[rank, time, margin]` multiplies each player's target minutes *rate* every possession. Rank 1–5 are the five who started; in the final eight minutes at \|margin\| > 15 the table gives them 0.80–0.82× their game-average share and the deepest bench bucket 2.6×. The scheduler re-targets the remaining slot-seconds through it, so bench minutes rise as the margin opens. | share of on-floor slot-seconds by rank bucket per (time × margin) cell on 2024, normalised on that bucket's exposure-weighted share over the whole game |
| R3, R6 | intrinsic: `\|margin\|`, `margin`, seconds remaining and period are hazard features. R3 entered them flat, which cannot move *who* comes in (a per-possession constant cancels in the entry softmax) and left the arm nearly flat across bands; R6 adds the `is_starter ×` and `period ×` interactions that can. | logistic exit/entry hazards on 2024 substitution events |
| R4 | intrinsic by construction: the game is cut into state blocks (1st half; 2nd half > 8:00; the final 8:00 split by the three margin bands) and each block's donor is drawn from the team's own last k games with probability ∝ (possessions that donor spent in the same band + 1), falling back to the nearest band when the team has never been in one. A blowout stretch is replayed from a real blowout stretch of that team's own season. | the team's own earlier games; k fitted |
| R5 | R4's splice **plus** a fitted state-deviation override: `z = (x − x_neutral)·(w_exit − w_enter)` over (fouls, \|margin\|, seconds remaining) × (is_starter, period), `p_block = sigmoid(scale·z + logit(p0))`, compared against one per-player "coach tolerance" uniform drawn once per game. `z = 0` in an ordinary state, so the override is inert there and bites only in foul trouble, in a blowout, or close-and-late. | override hazards on 2024 substitution events; `scale` and `p0` on 2024's own state cells |

Foul trouble is handled the same way — `TiltTables.foul[fouls, time]` for R1/R2,
features for R3/R6, the override for R5 — and every arm simulates its own foul
process from the player's as-of fouls per minute, because reading the real foul
events of the game being simulated would be a leak of exactly the kind
`is_transition` already committed at L3.

---

## 7. Decisions log

1. **Lineups come from CBBD, minutes accounting from the same source.** hoopR has
   no on-floor field; deriving minutes from the same possession stream as the
   lineups keeps sim and actual on one definition, and the 0.984 correlation with
   hoopR box minutes says nothing material is lost.
2. **Substitution events are derived from on-floor set transitions, not from
   `Substitution` rows.** CBBD pbp carries **zero** `Substitution` rows in 2024
   and only ~35 per game in 2025 against ~145 in 2026. The pre-registration's
   "fitted on 2024 substitution events" is therefore fitted on transitions of the
   on-floor set between consecutive possessions, which is the same event at
   possession-boundary resolution. This is a deviation forced by data and is
   flagged wherever it matters.
3. **The candidate pool is "appeared earlier for this team", and the tail is
   anonymous.** Using the CBBD roster as the pool would let a January addition be
   visible to a November game (the endpoint is one season-level snapshot with no
   as-of date). The "too short" defect CFB's `extend_profile` fixes is repaired
   with fitted geometric tail *slots* carrying synthetic negative ids instead.
4. **Availability is a drawn layer, not a filter.** `p_play[as-of rank]` is the
   fitted P(the player records any minutes); without it no Dirichlet over the
   candidate list can produce the exact zeros real minutes tables are full of, and
   the simulated nonzero count ran 12.1 players against a real 9.6. It is shared
   identically by all arms. Its measured cost is real: an i.i.d. per-game
   availability draw is the main reason the within-player minutes SD ratio is
   1.30–1.35 for R1/R2 (a real player is unavailable in runs, not independently
   each night) — see §9.
5. **Three scheduler parameterisations were tried; two are recorded as rejected.**
   Targeting each player's game *total* equalises everyone by the end of the game.
   Targeting a cumulative *path* is better but sluggish (starters took 0.65 of the
   final eight minutes against a real 0.78). The shipped form tracks the
   state-tilted *rate* with an exponentially-weighted on-floor tracker plus a slow
   long-run correction, and hits the drawn Dirichlet target in expectation.
6. **A period boundary re-sets the floor by target rate, not by deficit.** Ranking
   a fresh start by "how far below your rate are you right now" puts every resting
   player ahead of every on-floor one, so the coach would tip off the second half
   with whoever happened to be sitting. A threshold-free re-shuffle at every
   state-cell change was tried for the same reason and cost 3–4 pp of the
   starters' share in the final eight minutes; it is not used.
7. **The foul tilt is an event study, because it cannot be identified by
   stratification.** Pooled, the data says three fouls makes a player **3× more
   likely** to be on the floor; controlling for as-of rank it still says 1.6×, and
   controlling for rank and minutes-played-so-far 1.4× — a foul count is itself a
   record of floor time. Comparing a player's on-floor rate in the 20 possessions
   after picking up foul *f* with the 20 before, and chaining the per-foul ratios,
   gives the benching response the model actually needs (0.84 / 0.39 / 0.17 / 0.17
   cumulative in the first half).
8. **The state tilt is keyed on the five who started, not on the as-of minutes
   rank, and is normalised on the exposure-weighted mean rather than a baseline
   cell.** Keying on the predicted start order attenuates the table by however
   often the prediction misses; normalising on one cell silently re-levels the
   game total the Dirichlet draw is supposed to set.
9. **R5's override is a state *deviation* with a per-game tolerance draw.** A
   Markov flag version (P(bench) = scale × sigmoid(exit), P(return) =
   sigmoid(entry)) implies a two-thirds blocked fraction even in an ordinary state
   from its own fitted base rates, churns the lineup, and forces the fit to trade
   state dependence against the substitution rate — its fitted scale went to the
   grid ceiling with the substitution rate at 0.212 against a real 0.152.
10. **R5's override strength is fitted against the state cells, not the
    substitution rate.** Fitted against the rate it is selected straight out
    (the donor already substitutes slightly more often than reality, so every
    positive scale moves that number the wrong way and the grid returns 0, i.e.
    plain R4). The substitution rate is the donor's job.
11. **Nothing is adopted.** All three rounds fail the pre-registered
    state-dependence veto. Per the standing rules this is reported, not patched:
    no post-hoc multiplier on the blowout band, no cap on bench minutes.
12. **Round 3's keep override is a two-sided propensity, not a starters-only
    rule.** It was fitted on every candidate, which is what made it fail in a
    diagnosable way rather than by construction: the same fitted coefficients
    that keep a starter close-and-late keep a *bench* player in a blowout, and
    the grid shows the whole reachable frontier. A starters-only keep would have
    hit the close cell and would have been a hand-set rule wearing a fitted
    coefficient; that is the version the standing no-hand-tuning rule forbids.
13. **`rotation.py` was not modified for round 3.** Another worker is reading it
    for engine v0, so the round-3 arms live in `src/cbb_sim/models/rotation_v3.py`
    and import everything else. The round-3 donor simulator draws BOTH tolerance
    vectors whether or not an arm uses them, so R5, R7 and R8 sit at identical
    stream positions and the arms are paired; the cost is that round 3's R5 is
    not bit-identical to `rotation.R5Hybrid` (same rule, different realised
    numbers), which is why its cells differ from `docs/tests/rotation_close_game_audit_2026-09-10.md`
    by 0.1-0.6 pp. That difference is a free extra seed-noise reading and is
    smaller than noise floor A on every cell.
14. **Round 4's design is saturated in (time cell x margin band x is_starter),
    and that was declared before the run.** The audit measures a starter's exit
    hazard running 0.035 -> 0.020 -> 0.054 -> 0.027 -> 0.037 across the nine
    time cells, a bench player's exit hazard jumping to 0.244 in H2 20:00-16:00,
    and the starter/bench ordering reversing sign in the last two minutes of a
    blowout. A linear time term cannot represent that. The gate reads
    *occupancy* in some of those cells, which is a different functional of the
    process -- an equilibrium under the five-on-the-floor constraint -- and the
    L25 reachability probe (`experiments.md` §10.9) is what makes the claim that
    the family can reach the cells falsifiable rather than rhetorical.
15. **Timeouts and prior-season minutes share are measured and excluded.** A
    timeout multiplies every hazard by 3-4x and is the strongest non-boundary
    stoppage signal in the data; it is excluded because the engine has no
    timeout model, so a hazard conditioned on it could be fitted and never
    evaluated in simulation. Prior-season minutes share has a real gradient,
    same-signed and about half the size of the as-of share already in the
    design, and would need a new per-roster-slot engine input array. Both costs
    are sized in `features.md` §4.1, and the timeout exclusion is now the
    leading candidate cause of round 4's binding defect.
16. **Round 4 fits only what it adds; the base fits are reused from round 3b.**
    `rotation_fit_v3.json` and the six `rotation_fit_v3_S1_{YYYYMM}.json` are the
    base parameter sets, so the round-4 `R2 S1` column IS round 3b's and a
    difference between rounds cannot be a difference in R2's fit. The arms live
    in `src/cbb_sim/models/rotation_v4.py`; `rotation.py` and every fit file are
    untouched.
17. **An engine loop defect was found and fixed during this round.**
    `engine/loop.py` called `push_lineups()` *before* the period/halftime block,
    so at a period boundary the rotation was handed the previous possession's
    period and clock -- the five that took the floor for the first possession of
    the second half were chosen with `period = 1, seconds_remaining = 0`, and
    R2's own period-boundary reshuffle fired one possession late. The offline
    samplers have always used the possession's own state, so the engine and the
    bake-off disagreed exactly at the cell round 4 adds. The call now sits after
    the period block. Blast radius on aggregate engine metrics is about one
    possession in 137 and the counter-based streams stay aligned, but it is a
    behaviour change for every engine run started after the fix and any paired
    comparison that straddles it is invalid.
18. **The incumbent has a closed-loop gate for the first time (Decision 10).**
    `ENGINE_ROTATION_FREEZE=1` holds the rotation model's margin and foul counts
    at their pregame values while leaving foul accrual, the foul-out rule and the
    box-score counters live. Paired 5-seed runs over a fixed 500-game subset:
    R2 margin SD ratio 0.9950, home/away correlation -0.0026, possessions +0.061;
    H1 0.9963, +0.0106, +0.094. The rotation's consumption of engine-produced
    state is **not** a feedback channel -- it changes who scores, not how much --
    which is why both pass by two orders of magnitude against L23's `fg_make`
    loop.

19. **Round 5 changes the DRAW, not the family, and fits only the wave.** The
    round-4 hazards and the round-3b base fits are reused byte for byte; round 5
    fits `p_wave`, `p_size` and `rho` per S1 window and nothing else, so any
    difference between a round-5 arm and H1 is a difference in the draw alone.
    The shrinkage constant `k = 300` is the project's UNDERPOWERED threshold,
    declared in the pre-registration and never tuned, and each shrinkage parent
    is a marginalisation of the same counts.
20. **The composition grid is enumerable, and it was enumerated before it was
    fitted.** Rank or draw, on each of the two sides, is four knob-free rules;
    the L25 probe in `docs/tests/rotation_wave_audit_2026-09-11.md` §6 ran all
    four on paper and predicted that the uniform rules would bracket the target
    and the mixed rules would land on it. The bake-off reproduced the ordering
    on every cell and the distinct-lineup level to within 0.67 / 0.58 / 0.02 /
    0.05. An L25 probe is a **ranking** instrument: it was uniformly optimistic
    on the close-late band by 4.0-5.4 pp, which the as-of starter set (−2.8 pp,
    measured in round 3) and the real foul sequence account for.
21. **The timeout stays excluded and round 5 is the evidence that it can.** A
    timeout triples the wave probability (0.1371 → 0.4967) and carries 13.3% of
    all waves, and the engine has no timeout model. Round 4's arms paid for the
    exclusion with a 36-43% excess substitution rate; round 5's pay 4%, because
    the bunching a timeout produces is now in the wave draw rather than left to
    five independent coins.
22. **Decision 8 separated arms for the first time, and it penalised the
    coupling.** `rho = 0.39` is real (residual lift 2.53 after conditioning on
    the shared cell) and identified across six windows (0.386-0.391), and the
    coupled arm's close-late slope ratio is 0.74 against the actual, outside the
    [0.8, 1.2] band, as is W5's 0.75. A shared dead-ball draw moves both benches
    on a common rhythm and flattens the team-to-team spread the cell exists to
    show. Recorded against any future coupling arm.

23. **Round 6 changes the ENTRY RULE and fits only the composition object.**
    Round 5's wave tables, round 4's hazards and round 3b's base fits are reused
    byte for byte and nothing is written to any of them, so a round-6 arm minus
    W4 is the entry rule and nothing else. All three round-6 arms draw the same
    uniforms whether or not they use them (K1's `k_in` draw), so the three are
    paired with each other; the cost, stated, is that they are NOT byte-aligned
    with W4, which is why W4's column comes from round 5's JSON and is checked by
    a 1-seed reproduction re-run.
24. **`tau` is fitted by maximum likelihood on the training window, never
    against a gate cell**, over a grid declared in the pre-registration. It
    returns 1.00 in all six windows. `p_in` was itself fitted by logistic ML on
    those rows, so the argmax at 1 is close to a property of that fit; the
    informative content is the curvature, which rejects the rank direction by
    320k-350k log units and is what makes round 5's option (a) empty.
25. **The as-of-starter benchmark is report-only and was declared so before the
    run.** It re-grades the ACTUAL sequence with the model's predicted five and
    shows which gate cells are reachable at all (-4.4 pp on the opening ten
    minutes, -2.8 pp on both close-late bands, +0.2 pp on foul trouble). It
    changed no tolerance and no verdict; every arm is still scored against each
    side's own real starting five. Whether the gate SHOULD score against the
    benchmark is a PM decision and is not taken inside a running bake-off.

---

## 8. Consumption from the sim

The engine-facing interface is `RotationSampler.next_lineup(state)`. It runs
exactly the offline `simulate()` decision rule — `tests/test_rotation.py` asserts
the two produce the same sequence — so the engine and the bake-off can never be
two different models.

```python
from cbb_sim.models import rotation as R

fit = R.RotationFit.from_json("data/processed/models/rotation/rotation_fit.json")
feats = R.build_asof_player_features(R.player_game_minutes(R.load_team_possessions(2025)))
priors = R.build_priors(feats, fit)            # one TeamPrior per (game_id, team_id)

arm = R.R2HierDirichlet(fit)                   # or R1/R3/R4/R5/R6 behind a flag
rng = R.game_stream(seed, game_id)             # (seed, game_id, "rotation"), home first
home = R.RotationSampler(arm, priors[(game_id, home_team_id)], seed, game_id,
                         is_home=True, expected_total_seconds=2400.0, rng=rng)
away = R.RotationSampler(arm, priors[(game_id, away_team_id)], seed, game_id,
                         is_home=False, expected_total_seconds=2400.0, rng=rng)

last = 0.0
for possession in game:
    st = R.RotationState(period=possession.period,
                         seconds_remaining=possession.seconds_remaining,
                         score_diff=home_score - away_score,   # always home minus away
                         last_possession_seconds=last)
    on_home = home.next_lineup(st)             # tuple of 5 CBBD player ids
    on_away = away.next_lineup(st)
    last = possession.duration_s
```

Notes for the caller:

- **Feature order** is `features.md` §1; `RotationFit` carries every fitted
  object and `from_json` restores it, including the tilt tables.
- **`score_diff` is always home minus away**; the sampler re-signs it.
- **`last_possession_seconds` must be the duration of the possession just
  played**, 0 on the first call. The sampler credits minutes and draws fouls from
  it, so passing 0 every time freezes the rotation.
- **Ids are CBBD player ids.** Map to ESPN with
  `cbb_sim.data.player_ids.cbbd_to_espn_map(crosswalk, season)` before joining
  anything keyed on `athlete_id`.
- **NaN handling / defaults.** A team with fewer than 3 earlier games that season
  has no `TeamPrior` and the engine must fall back (none is provided here — that
  is a known gap, §9). Tail slots carry negative ids and have no box-score
  identity; any downstream consumer must treat them as anonymous replacement-level
  minutes.
- **Output shape.** A tuple of exactly 5 distinct ids, every call, for every arm —
  asserted per possession in the tests.

---

## 9. Artifacts

| Path | What it is |
|---|---|
| `data/processed/player_crosswalk.parquet` | CBBD ↔ ESPN player crosswalk, one row per (season, cbbd_player_id) |
| `data/processed/player_crosswalk_report.json` | match rates by season and method, plus the largest unmatched on-floor ids |
| `data/raw/cbbd/rosters/roster_{2024,2025,2026}.parquet` | the three `/teams/roster` pulls the crosswalk is built from |
| `data/processed/models/rotation/rotation_fit.json` | every round-1 fitted object (role prior, shrinkages, availability table, Dirichlet concentrations, tilt tables, scheduler parameters, hazards) |
| `data/processed/models/rotation/rotation_F1_results.json` | round-1 results, verdicts, noise floor, robustness, decision |
| `data/processed/models/rotation/rotation_F1_table.csv` | round-1 metric table, one row per arm |
| `data/processed/models/rotation/rotation_fit_round2_corrected_hazards.json` | the round-2 fit with the **corrected** hazard/override coefficients (`experiments.md` §5) |
| `data/processed/models/rotation/rotation_F1_round2_SMOKE30_do_not_cite.{json,csv}` | a later 30-game dev smoke run that overwrote the graded round-2 JSON/CSV. **`experiments.md` §4 is the record of the graded round-2 run**; re-running `scripts/train_rotation_v2.py` with the arguments named there reproduces it |
| `data/processed/models/rotation/rotation_fit_v3.json` | the round-3 fit: the round-2 corrected fit plus `s*`, R7's on-floor propensity and both fitted knob pairs. **Nothing overwrites `rotation_fit.json`, which the engine reads** |
| `data/processed/models/rotation/rotation_fit_v3_noisefloor_seed2.json` | the spec-identical refit under fit seed 101 / sim seed 23 that noise floor B is measured from |
| `data/processed/models/rotation/rotation_F1_round3_{results.json,table.csv,section.md}` | round-3 results, verdicts, both noise floors, the quintile slope table and the full knob grids |
| `data/processed/models/rotation/rotation_fit_v3_S1_{YYYYMM}.json` | round 3b: one parameter set per S1 window, named by the window's **start** month, so a game selects the largest `YYYYMM` at or before its own month. The first window's file is byte-identical to `rotation_fit_v3.json` |
| `data/processed/models/rotation/rotation_F1_round3b_S1_results.json` | round 3b, the S1-vs-static scheme confirmation |
| `data/processed/models/rotation/close_game_audit_2026-09-10.json` | the round-3 evidence audit's raw cells (`docs/tests/rotation_close_game_audit_2026-09-10.md`) |

| `data/processed/models/rotation/round5/rotation_v5_wave_{YYYYMM}.json` | round 5: the wave Bernoulli, the wave-size categorical, `rho`, and a COPY of that window's round-4 hazard coefficients so an artifact is self-contained. Six windows plus `rotation_v5_manifest.json` in the `engine/manifest.py` format. Gitignored (`data/processed/models/*/round*/`), HF-synced |
| `data/processed/models/rotation/round5/rotation_v5_wave_seed2_*.json` | round 5 noise floor B: the spec-identical refit under fit seed 101 |
| `data/processed/models/rotation/round5/rotation_v5_wave_nostate_*.json` | round 5 Decision 10 / L31: the refit WITHOUT margin or foul state — the wave table marginalised over both axes and the round-4 hazards refitted with all 14 margin/foul columns dropped (`scripts/train_rotation_v5_nostate.py`). Its `n_boundaries` field reads ~4.96M because the collapsed counts are broadcast back over the six margin × foul cells; the real boundary count is the same 822k-826k as the live fit |
| `data/processed/models/rotation/rotation_F1_round5_{results.json,table.csv}` | round-5 results, verdicts, both noise floors, the reproduction check, the slope table and the decision |
| `data/processed/models/rotation/wave_audit_2026-09-11.json` | the round-5 wave audit's raw cells (`docs/tests/rotation_wave_audit_2026-09-11.md`) |
| `data/processed/models/rotation/wave_reachability_2026-09-11.json` | the L25 on-paper probe over all four composition rules, computed before the pre-registration |
| `data/processed/models/rotation/round6/rotation_v6_comp_{YYYYMM}.json` | round 6: `tau`, `P(k_in \| size, k_out)`, the 4x4 tier log-affinity and the tau likelihood curve, one per S1 window, plus `rotation_v6_manifest.json`. Gitignored (`data/processed/models/*/round*/`), HF-synced |
| `data/processed/models/rotation/round6/rotation_v6_comp_seed2_*.json` | round 6 noise floor B: the spec-identical refit under fit seed 101 |
| `data/processed/models/rotation/rotation_F1_round6_{results.json,table.csv}` | round-6 results, verdicts, floor A, the quintile and per-team blocks, the as-of-starter benchmark, the slope table and the decision |
| `data/processed/models/rotation/rotation_F1_round6_floorB.json` | round-6 floor B (A1), run in its own process so the bake-off JSON is on disk first |
| `data/processed/models/rotation/comp_audit_2026-09-11.json` | the round-6 composition audit's raw cells (`docs/tests/rotation_composition_audit_2026-09-11.md`): the measured joint on both seasons, the state split, and the realised joint under W4/K1/A1 |

---

## 10. Known gaps / followups

- **ROUND 7, and it is the only item that matters.** The **exit rule**. Round 5
  fixed WHETHER and HOW MANY, round 6 shows the entry side can be conditioned and
  that conditioning cannot pay while the leaver mix is wrong: every arm since
  round 5 takes a starter off at 0.456-0.467 of single swaps against a real
  0.587. The object is K1's mirror, `P(k_out | size, state)` drawn first and the
  leavers raced within class -- the same counts pass, the same declared shrinkage
  constant, the same engine primitives. The one state term the composition
  measurably needs is the final eight minutes of a decided game, where
  P(a starter enters | a bench player leaves) is 0.53 against 0.75 everywhere
  else (`docs/tests/rotation_composition_audit_2026-09-11.md` section 2).
- **Foul trouble is the only veto cell left that is the model's own.** -3.2 to
  -4.2 pp against the like-for-like benchmark, with the "at exactly 4 fouls"
  share 0.59-0.61 against a real 0.517. It is a foul-process and
  benching-response question and nothing in the substitution draw will fix it.
- **SUPERSEDED BY ROUND 6 (kept for the record): ROUND 5.** The round-4 family needs a
  **joint dead-ball substitution draw**: one Bernoulli per (team, boundary) for
  "is there a substitution wave here", conditioned on `prev_end` and the state,
  then a wave size and a composition drawn from the same per-player hazards.
  Measured evidence that this is the binding defect, not a guess:
  (a) the fitted period-boundary hazards are right to 1 pp and the *earned*-reset
  arm still misses the tip by 11 pp; (b) the arms' boundary change rate is
  0.2058 against a real 0.1518 while their per-player exit rate is right;
  (c) distinct lineups per team-game 19.7 against 14.84. A timeout indicator
  would supply the same bunching directly and is excluded because the engine has
  no timeout model (`features.md` §4.1) — the joint draw gets the effect from
  `prev_end` alone, which the engine does carry.
- **The hazard family's close-and-late level is 8 pp low and the reachability
  probe says it should not be.** The saturated cell form of the same family,
  fitted on 2024 and run on 400 2025 games with the real starting five, lands at
  0.7393 against 0.7388 (audit §8). The arms land at 0.669. 2.8 pp of the gap is
  the as-of starter set (measured); the rest is item 1's over-substitution.
- **Lineup concentration got worse, not better.** Top-1 five-man lineup share
  0.198–0.220 against a real 0.294 and R2's 0.229; K-S D 0.27–0.37 against R7's
  0.030, which remains the best any arm has produced. A unit-level substitution
  block is the shape to aim at, and it is the same object as item 1.
- **H3 (LightGBM hazard) is not expressible in the sim loop.** `CLAUDE.md` bans
  live model calls there; shipping a tree hazard needs the booster discretised
  into a lookup table. It did not earn that work in round 4 (it loses the
  foul-trouble cell at −6.1 pp and the pooled minutes SD ratio at 0.856).

- **REFUTED BY ROUND 3 (kept for the record).** The hypothesis below — that R5
  spends its starter time too early and needs a within-game time-profile
  constraint — is wrong. `docs/tests/rotation_close_game_audit_2026-09-10.md` §5
  measures it directly: in the close band R5 is **−5.3 pp early and −5.4 pp
  late**, so the total is not right and its distribution over the game is not the
  problem. The close-game miss is a level shift across the whole game. What the
  audit found instead, and what round 4 should carry: (a) the second half tips
  off at a **0.90** starters' share in every margin band and no arm exceeds 0.80,
  a −10 to −15 pp miss in the largest single cell of the second half that no gate
  currently reads; (b) R2 is −17.0 pp at the opening tip in close games while its
  pooled minutes table is right; (c) with the foul terms working, the donor
  family's binding failure is the blowout band and foul trouble, not the close
  bands. Neither (a) nor (b) was added to a gate mid-bake-off; both belong in the
  next pre-registration.
- **For round 3 (SUPERSEDED): the close-game late-starter miss is a *level*
  problem, not a state-response problem.** After round 2 the binding cell is the opposite of
  round 1's: R5 and R6 are inside tolerance in a blowout and 4-7 pp short of the
  starters' share in the final eight minutes when the game is close or moderate.
  Four pieces of evidence point at one cause. (1) The miss is almost the **same
  size in both close bands** (R5 -4.3 / -6.9, R6 -6.5 / -7.2 pp) and the *slope*
  across bands is right, so it is a level shift, not a failure to respond to the
  margin. (2) Starter identification accounts for only 2.6-2.8 pp of it: against
  the as-of-starter-set benchmark R2 is +0.7 / -0.2 pp, i.e. **essentially exact**,
  while R5 is -1.5 / -4.1 and R6 -3.8 / -4.4. So the arms that replay or model
  stint sequences, not the scheduler, carry the residual. (3) Both of those arms
  also use **too few players** (8.57-8.58 vs 9.64) and concentrate **too much** of
  the minutes table (top-8 share +2.3 / +2.6 pp) -- they are spending a fixed
  amount of starter time too early in the game, so there is less of it left for
  the last eight minutes. (4) R5's donor depth was fitted on lineup concentration
  alone (k = 3 of {3, 5, 8, all}), which has no term for *when* in the game a
  lineup appears.
  The round-3 hypothesis this suggests: the donor/hazard arms need a **within-game
  time-profile constraint**, not a stronger late-game override. Concretely, fit
  the donor splice and the hazard against the starters' share in each
  (time bucket x margin bucket) cell jointly -- the same lookup table R1/R2
  already use as `TiltTables.state`, which is exactly why R2 gets the close bands
  right -- rather than against end-of-game aggregates. The cheap test of the
  hypothesis is to check whether R5's starter share is *above* the real one in the
  first half by about the amount it is below in the final eight minutes; if it is,
  the total is right and only its distribution over the game is wrong, and a
  time-profile constraint fixes both the close bands and the top-8 share at once.
  A stronger late override would instead trade the blowout cell back.
- **The blowout band is the blocking failure of round 1.** Every arm of round 1 misses
  starters' share of the final eight minutes at \|margin\| > 15, and they miss it
  in *both* directions (R1/R2 −5 to −6 pp, R4 +8 pp). The next pre-registration
  should treat the blowout response as its own fitted object rather than a
  by-product of whichever lineup mechanism is chosen, and should consider that the
  real margin band at a possession is a poor conditioner: a coach responds to how
  long the game has been decided, not to the instantaneous margin.
- **Within-player minutes dispersion is 30% too wide for R1/R2.** Diagnosed to the
  i.i.d. per-game availability draw: a real player's unavailability is serially
  correlated (an injury spell), and drawing it independently each night adds
  game-to-game variance the actual side has no counterpart for. A Markov
  availability state is the obvious fix and is not built.
- **Lineup concentration is unsolved for the Dirichlet arms.** Top-1 five-man
  lineup share 0.231/0.238 against a real 0.296 (K-S D 0.19–0.23), while R4 sits
  at 0.276 with K-S D 0.036. The scheduler has no notion of a *unit*; real coaches
  substitute in blocks that recur.
- **No fallback prior for a team with fewer than 3 earlier games.** November games
  are simply excluded from the test universe. A preseason prior (prior-season
  roster carry-over with the transfer attenuation L15 measured) is needed before
  the engine can simulate a full season.
- **`Substitution` rows are unusable in 2024 and thin in 2025.** A followup
  should check whether the 2026 rows (~145/game) reconcile with on-floor
  transitions, which would let the hazards be fitted on real events once the seal
  lifts.
- **Overtime is a single tilt bucket.** `time_bucket = 4` pools all OT periods; OT
  is 5.6% of games and the rotation there is not separately validated.
- **The foul process is calibrated on totals, not on the upper tail.** Simulated
  starters reach ≥ 4 fouls more often than real ones even after the event-study
  tilt (foul-out rate was 2.6× too high before it; it is still high). Real fouls
  are not independent across possessions.
- **2026 is untouched.** Both trainers call `seal.assert_not_sealed`; the 2026
  crosswalk numbers in §4 are descriptive only, the same way the data audits
  report 2026.
