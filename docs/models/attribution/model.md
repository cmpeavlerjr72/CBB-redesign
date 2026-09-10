# L4 PLAYER ATTRIBUTION (attribution)

Pre-registration and the full grid: [`experiments.md`](experiments.md). Feature
provenance: [`features.md`](features.md). Code:
`src/cbb_sim/models/attribution.py`. Trainer:
`scripts/train_attribution_v1.py`. Tests: `tests/test_attribution.py`.

> **Status: BAKE-OFF COMPLETE 2026-09-10.** Both folds (F1 selection, within-2025
> robustness) ran to completion on all eight targets with the full
> pre-registered grid (3 LightGBM rungs, 3 F1 seed refits, 40 game-level draws,
> 200 bootstrap reps); no runtime lever was needed (wall time 682.4 s). A
> `build_team_asof` construction bug (found while resuming; every team's league
> as-of rate for the three binaries was silently zero/NaN) was fixed first --
> `docs/tests/attribution_team_asof_lg_defect_2026-09-10.md`. Winners: 6 of 8
> targets adopt an arm on F1 (`REB_off`->cond_logit, `REB_def`->lgbm,
> `steal`->proportional, `block`->lgbm, `assisted`->aware_ridge,
> `blocked`->aware_ridge); `assist` and `stolen` adopt NO WINNER on F1
> (`assist` recovers a winner, `lgbm`, on the within-2025 fold; `stolen` fails
> calibration on every arm on both folds).
> Full per-target table, both folds: `experiments.md` section 3.

---

## 1. Purpose

The cascade has already decided everything about the event except who gets the
box-score line. `possession_outcome` decided that the chance ended in a missed
shot or a turnover; `fg_make` decided made vs missed; `rebound` decided that the
offence (not the defence, not a dead ball) recovered the miss; `rotation` decided
which ten players were on the floor; `usage` decided which of the five offensive
players took the shot or lost the ball. This model decides the **secondary
credit**: which of the rebounding team's five secured the board, whether a made
field goal was assisted and by whom, whether a turnover was a steal and by whom,
and whether a miss was blocked and by whom.

It feeds every non-scoring player prop (rebounds, assists, steals, blocks) and is
the last player-level stage before markets. It decides **no counts**: the number
of rebounds, turnovers and misses belongs to its upstream neighbours, and the
three binaries here own only the conditional share of those counts that carries a
secondary credit.

## 2. Target variables

Five **choice** targets and three **binary** targets. One row per event of the
target's population, restricted to events whose candidate five is fully resolved.

| target | population | candidates | credited | K | uniform log loss |
|---|---|---|---|---:|---:|
| `REB_off` | live offensive rebounds | offence five | rebounder | 5 | 1.609438 |
| `REB_def` | live defensive rebounds | defence five | rebounder | 5 | 1.609438 |
| `assist` | made FGA that **are** assisted | offence five **minus the shooter** | assister | 4 | 1.386294 |
| `steal` | turnovers that **are** steals | defence five | stealer | 5 | 1.609438 |
| `block` | missed FGA that **are** blocked | defence five | blocker | 5 | 1.609438 |
| `assisted` | made FGA | — | was it assisted | — | — |
| `stolen` | turnovers | — | was it a steal | — | — |
| `blocked` | missed FGA | — | was it blocked | — | — |

Offensive and defensive rebounds are modelled **separately**, as the
pre-registration requires: they are different skills on different bodies, and
pooling them would force one rate per player where the data plainly carries two.

`DeadBallReb` is excluded from both R targets — it carries no player credit at the
source and `rebound` already models it as a fixed per-miss-type share (L17).
Free-throw rebounds **are** live rebounds and are included; ESPN's administrative
reset rebound between two free throws of one trip is dropped by exactly the rule
`event_stream` and `possessions._collect_trip` use.

Population filter: D-I, non-truncated, `pbp_complete` games, seasons 2024 and
2025 only (on-floor ids are empty at the source in 2022-2023, L13). Coverage per
target and season — how many events have a resolved candidate set, how many have a
credited id, and how many are therefore modelled — is **reported** in
`experiments.md` section 2.1, never filtered silently.

Candidates are sorted ascending by CBBD player id so the alternative order is a
function of the lineup and never of the feed's column order, and `y` is the slot
index of the credited player.

## 3. Methodology at a glance

- **Data window.** 2024 and 2025 (L13). 2026 is sealed; `fold_slices` and
  `walkforward_slices` both call `seal.assert_not_sealed`.
- **Split.** Temporal walk-forward, never random. F1 trains 2024 and tests 2025
  and is the selection fold. The robustness fold is a within-2025 walk-forward,
  train before 2025-01-15, test after.
- **Primary metric.** Per-target log loss. Eligibility is gated on calibration by
  predicted-probability decile **and** by as-of-rate decile (≤ 2 pp),
  responsiveness under `ARCHITECTURE_DECISIONS.md` Decision 8, and the game-level
  checks carried over from `usage`.
- **Families tested.** Three choice arms (proportional over the shrunk as-of
  rate; conditional logit; LightGBM grouped-softmax ranker) and three binary arms
  (team-level logistic ridge; shooter/miss-class-aware logistic ridge;
  LightGBM). Nothing is chosen for familiarity: the shrinkage prior and strength,
  the ridge penalty and the tree parameters are all fitted on training data only
  and the winner falls out of `train_attribution_v1.decide`.

### 3.1 The Decision-8 reading used here

The pre-registration cites Decision 8 directly rather than restating a step
count, so the gate is that decision's own minimum: **slope ratio in [0.8, 1.2]
AND monotone in at least 3 of 4 quintile steps, with a driver whose realised
quintile span is under 2 pp exempt from BOTH clauses** (a slope ratio over a
noise-sized span is itself noise). `usage` reads clause (b) as 4-of-4 relaxing to
3-of-4, because *its own* pre-registration specified 4 of 4 and Decision 8 only
relaxes it. Both readings are reported for every arm here
(`strict_4of4_pass` alongside `pass`), so the stricter reading stays auditable and
the difference between the two models' numbers is a documented reading, not a
silent divergence.

### 3.2 The game-level checks, and what a binary can and cannot be held to

For a **choice** target the checks are `usage`'s trio, re-attributing the ACTUAL
event sequence over the ACTUAL fives: per-player per-game count SD ratio
(0.9-1.1), players credited with ≥ 1 per team-game (± 0.5), and top-1 / top-3
share (± 2 pp). That isolation is the point — how many events a team-game has and
who was on the floor at each come from reality, so a failure here cannot be
confused with a failure in rotation or in the event model.

A **binary** credits no player, so that trio has no player axis to live on. The
statistic of the same family one level up does exist and is what the binary is
gated on: the **per-team-game count** of the credit and its dispersion (mean
within ± 0.5, SD ratio 0.9-1.1). The three per-player fields are reported as
`null` with that reason rather than omitted. The **composed** per-player check
(binary then choice, the quantity a player prop actually needs) is implemented as
`attribution.composed_check` and is a **reported diagnostic only** — it is not in
the pre-registration's decision rule and never enters `decide`.

## 4. Winner

Per target, on **F1** (train 2024, test 2025 -- the selection fold):

| target | F1 winner | F1 log loss | WF2025 (within-2025 robustness) winner |
|---|---|---:|---|
| `REB_off` (choice, K=5) | `cond_logit` | 1.431192 | `cond_logit` |
| `REB_def` (choice, K=5) | `lgbm` | 1.559167 | `cond_logit` |
| `assist` (choice, K=4) | NO WINNER | (lgbm best, 1.292465) | `lgbm` |
| `steal` (choice, K=5) | `proportional` | 1.586249 | `proportional` |
| `block` (choice, K=5) | `lgbm` | 1.273376 | NO WINNER (17,234-row test, flagged underpowered) |
| `assisted` (binary) | `aware_ridge` | 0.558057 | NO WINNER (calibration, 141k-row test -- not underpowered) |
| `stolen` (binary) | NO WINNER | (lgbm best, 0.642243) | NO WINNER |
| `blocked` (binary) | `aware_ridge` | 0.264337 | `aware_ridge` |

Full per-arm tables (Brier, top-1/top-3, calibration, transfer subsets,
LightGBM importances, conditional-logit coefficients), the responsiveness
slope check and the noise-floor SD for every decision: `experiments.md`
section 3. `stolen` is the one target that fails calibration on every arm on
both folds -- reported as a clean "not usably modellable with the
pre-registered feature set" result, not patched.

## 5. Robustness

The pre-registered within-2025 walk-forward fold ran to completion on all
eight targets (`experiments.md` section 3.3). Five of eight targets carry the
same winner (or NO WINNER) as F1, or a winner inside the other's noise floor;
`assist` and `block` flip direction between folds (`assist` gains a winner on
WF2025, `block` loses one, and the WF2025 loss is flagged as likely
underpowered at a 17,234-row test set rather than presented as a reversal).

The composed diagnostic (binary-then-choice, the quantity a player prop
actually needs) is a **reported-only** check per the pre-registration and was
never a gate -- but it did not run in either the interrupted launch or this
completed run, because `scripts/train_attribution_v1.py` never actually calls
`attribution.composed_check` from its per-target loop; `--skip-composed`
exists as a flag but `args.skip_composed` is never read. This is a
pre-existing gap in the trainer, found while completing the bake-off, not a
choice made by this run. It does not affect any adopted winner. See
`experiments.md` section 3.2.

## 6. Decisions log

1. **The event stream is rebuilt, not imported.** `event_stream` drops the block
   rows after folding them into a flag (losing the blocker's id), carries no
   assist columns, and keeps no play `id` to join a stealer onto its turnover.
   It is a shared prerequisite two shipped models train against, so it is not
   edited; `build_attr_stream` re-does the same cleaning with three extra columns
   and the same repairs. `features.md` section 1.1.
2. **The shooter is `shot_shooter_id`, not `participant_1_id`.** On 2025 assisted
   made field goals `participant_1_id` is the **assister** on 48.9% of rows —
   CBBD's participant order is not stable on a two-participant row. Reading the
   shooter off it removed the wrong man from the assist choice set and made 49%
   of the assist population unmodellable. `features.md` section 1.2. **This is a
   data defect with a scope beyond this model** and is logged as such in the
   change ledger.
3. **Offensive and defensive rebounds are two targets, not one with a side
   flag.** The pre-registration requires it, and the rate tables confirm why: the
   same player has two different rates and one model would average them.
4. **The assist has FOUR alternatives.** A made field goal cannot be assisted by
   the player who made it, so a five-alternative assist model would spend
   probability mass on an impossible outcome and its log loss would not be
   comparable to a four-alternative one. The removal happens in exactly one place
   (`usable`), and the sampler enforces the same rule at draw time.
5. **Exposure is per opportunity of the target's own population, on the floor.**
   So the candidates' rates sum to ≈ 1 over the candidate set, P1's normalisation
   is a re-scaling, and the fitted `m` reads as "pseudo opportunities of
   history". This is a **different** denominator from
   `rebound.player_rebound_rates` (share of *available* rebounds) and the two
   numbers are not interchangeable. `features.md` section 2.
6. **Four RNG families, one per credit.** `attr_rebound`, `attr_assist`,
   `attr_steal`, `attr_block`. An engine that stops drawing blocks must not change
   which player gets the steals;
   `test_c_drawing_one_credit_does_not_move_another_credits_sequence` pins it.
   The two rebound targets share the `attr_rebound` family because they are the
   same credit asked on two sides of the ball and no miss draws both.
7. **The binary and the choice share one family and one counter.** The binary
   uniform is drawn first and the choice uniform only when the binary fires, so a
   game with no steals consumes exactly as many uniforms as it had turnovers and
   the stream is a function of the event sequence alone.
8. **The trainer's re-attribution path keys on `(seed, game_id, family,
   ordinal)`**, not on the game alone. `usage` measured what keying on the game
   costs: 3.42 players with a three-point attempt per team-game against a real
   6.69, because every event of a team-game drew the same uniform.
9. **P3 optimises the grouped-softmax likelihood, not a binary one.** The
   pre-registration calls it a "ranker"; a lambdarank score has an arbitrary scale
   so softmaxing one is miscalibrated by construction, and a binary objective
   softmaxed afterwards optimises a different likelihood and came out
   systematically over-sharpened in the usage bake-off (2.24 pp worst calibration
   gap on an arm whose log loss looked fine). `group_softmax_objective(K)` is the
   K-generic form of `usage._group_softmax_objective`, and the arm is offset by
   `log q` so at zero trees it **is** P1.
10. **Row bagging is off in P3.** A row is one alternative of a choice set, so
    `subsample` would split choice sets across the in-bag boundary. Column
    sampling is unaffected and kept.
11. **State features enter P2 only as interactions.** A conditional logit
    differences out anything constant across the alternatives, so the
    pre-registered state and shot-class columns are *unidentified* as plain
    columns, not merely weak. `usage` decision 9.
12. **The prior-season block is dropped on F1 and the drop recorded.** Not
    because it is constant — `build_choice_design` falls it back to the position
    prior, so it varies — but because its *meaning* changes between a fold with no
    prior season and one with.
13. **The arms see only the pre-registered features.** Exposure, minutes,
    transfer status and period are carried onto the design for the **reported
    subsets** and are deliberately not features of any arm. Adding them would be
    an unregistered feature bundle; they are listed as a known gap instead.
14. **Team and own-share rates are league-centred before they enter a matrix**,
    per `CLAUDE.md`'s ban on raw levels.
15. **The game-level port is pinned to `usage`'s.** `usage`'s cell machinery is
    fixed at five alternatives, at `alt_*` column names and at the `usage_alloc`
    family; it is ported here over K and over `cand_*` with identical definitions,
    and `test_d_game_level_port_matches_usage_on_five_alternatives` asserts the
    two agree, so a number quoted next to a usage number is the same statistic.

## 7. Consumption from the sim

```python
from cbb_sim.models import attribution as AT

# ---- once per (game, seed), at game setup --------------------------------
# `rates` is credit -> {cbbd_player_id: relative rate}, assembled from the
# pregame as-of table (or from a shipped rate table keyed the same way).
state = AT.new_game_state(game_id=game_id, seed=seed, rates=rates)

# ---- per resolved team-level event ---------------------------------------
who = AT.draw_rebounder(off_five, state, offensive=True)       # -> player id
who = AT.draw_assist(off_five, shooter_id, state, p_assisted)  # -> id or None
who = AT.draw_steal(def_five, state, p_steal)                  # -> id or None
who = AT.draw_block(def_five, state, p_block)                  # -> id or None
```

- Artifact: `data/processed/models/attribution/attribution_params_v1.json`
  carries, per target, the fitted shrinkage prior and strength, the
  conditional-logit feature order and coefficients, the tree parameters, the
  binary ridge coefficients and the RNG family.
- `five` is any iterable of exactly five CBBD player ids; order does not matter
  (the model normalises over the set) and a length other than five raises.
- `p_*` is the binary probability from this model's binary half. Omitting it
  asserts the caller already knows the credit happened.
- A candidate set in which nobody carries any rate falls back to the **uniform** —
  never a silent pick of the first id.
- One uniform of the credit's own `(seed, game_id, family)` stream is consumed per
  call, so a game's draws are reproducible and independent of every other game.

## 8. Artifacts

| Path | What it is |
|---|---|
| `data/processed/models/attribution/events_{pop}_{version}.parquet` | the five population tables, one row per event with its candidate five, its state and its coverage flags |
| `data/processed/models/attribution/asof_{version}.parquet` | one row per (season, player, game) with every pregame as-of player input |
| `data/processed/models/attribution/team_asof_{version}.parquet` | one row per (season, team, game) with the binaries' team inputs |
| `data/processed/models/attribution/build_report_{version}.json` | universe, rim override, per-target coverage, stream repair counts, crosswalk join rate |
| `data/processed/models/attribution/results_v1.json` | every arm's full metric block on every fold that ran |
| `data/processed/models/attribution/attribution_params_v1.json` | the fitted parameters the sim consumes |
| `data/processed/models/attribution/report_v1.md` | the generated results markdown appended to `experiments.md` |
| `data/processed/models/attribution/train_log_v1.txt` | the run log, including every grid rung |

## 9. Known gaps / followups

1. **The bake-off is finished (2026-09-10 evening).** Both folds ran on all
   eight targets at the full pre-registered grid; see section 4 for the
   winners and `experiments.md` section 3 for the full evidence. The one
   remaining procedural gap: the composed per-player diagnostic is
   implemented (`attribution.composed_check`) but is never called from
   `scripts/train_attribution_v1.py`'s per-target loop, regardless of
   `--skip-composed` -- a trainer wiring gap, not a decision-rule gap, and it
   does not affect any adopted winner (item 9 below already noted the check as
   reported-only). Wiring it in is the next thing to do if the composed
   numbers are wanted.
2. **P3 is the runtime bottleneck.** A Python grouped-softmax objective is called
   once per boosting iteration over K·n rows, and `REB_def` and `miss_fga` are the
   two largest populations in the project. Options, in order of preference: cut
   the parameter grid (it is three rungs), subsample whole **games** for the
   parameter search only, or move the objective to a compiled form. None of these
   changes what is compared.
3. **No arm sees exposure or minutes.** The pre-registration's player input list
   is the as-of rate, the prior-season rate and the position group, so a player
   with two opportunities of history and one with two hundred are distinguished
   only through the shrinkage. That is a defensible reading of the contract and
   almost certainly leaves signal on the table; it needs a *new* pre-registration,
   not a quiet column.
4. **No lineup-aggregate feature.** The candidate five's summed as-of rate is the
   obvious "how much competition is there for this board" term and is not
   pre-registered. Same treatment as item 3.
5. **Team rebounds have no player to credit.** `participant_1_id` is blank on
   ~18-25% of offensive-rebound rows and ~7% of defensive ones (the feed records
   a team rebound). Those rows cannot enter a who-did-it model and are dropped
   with the count reported, never imputed — the same treatment `usage` gives team
   turnovers. The consequence is that `REB_off` is trained on ~81% of offensive
   rebounds, which is the lowest coverage of any target here.
6. **The prior-season block has only ever been identified on one fold.** L13 caps
   this model at two seasons of on-floor data, so `prior_rate` and
   `log_prior_share` are dropped on F1 and the within-2025 fold is the only place
   they are exercised. A third season makes the transfer question answerable
   rather than merely reportable, and is the first thing to redo after the seal
   lifts.
7. **The sim path for a tree or logit winner is not a lookup table yet.**
   `CLAUDE.md` forbids live model calls in the sim loop. The samplers serve the
   rate-table arm (P1) directly; a tree or logit winner needs its per-event scores
   materialised against the pregame feature bins first. Same open item `usage`
   carries.
8. **`shot_class` on a rebound row is the previous row's class, not the chance
   table's.** It is read off the immediately preceding miss in the stream. That is
   correct by construction once the block rows are dropped, but it is a stream
   adjacency rather than an authoritative field, and it is labelled one.
9. **The composed check is a diagnostic, not a gate.** The pre-registration grades
   the binary and the choice separately, so a target can pass both halves and
   still compose badly. `composed_check` measures exactly that and is reported;
   making it a gate would require a new pre-registration.
10. **Technical free throws and dead-ball rebounds are out of scope**, as is the
    offensive-rebound **tip-out** (the feed records it as a rebound, so a tip-out
    to a team-mate is credited to the tipper or to nobody depending on the feed's
    mood). No rule is imposed here.
