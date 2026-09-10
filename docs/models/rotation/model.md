# Rotation model (L4) — who is on the floor, and for how long

Status: **BAKE-OFF RUN 2026-09-10, two rounds, NO ARM ADOPTED.** One defect in
the hazard training matrix is open and qualifies the round-2 R5/R6 columns
(`experiments.md` §5). Read
`experiments.md` for the pre-registrations and the full tables; this file leads
with the conclusion and the relative comparisons.

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

### 5.3 The diagnostic that matters most: starter identification vs rotation

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
11. **Nothing is adopted.** Both rounds fail the pre-registered state-dependence
    veto. Per the standing rules this is reported, not patched: no post-hoc
    multiplier on the blowout band, no cap on bench minutes.

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

---

## 10. Known gaps / followups

- **For round 3: the close-game late-starter miss is a *level* problem, not a
  state-response problem.** After round 2 the binding cell is the opposite of
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
