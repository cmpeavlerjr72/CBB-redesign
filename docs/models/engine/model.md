# ENGINE v0 -- the possession-level simulator

Created 2026-09-10. Implements Decision 1 (possession-by-possession Monte Carlo,
Control as the yardstick) and Decision 7 (pace is emergent; the L2 winner is a
prior feature, not a sampler). Code: `src/cbb_sim/engine/`, `scripts/build_engine_inputs.py`,
`scripts/run_engine.py`. Gate report: `docs/tests/engine_v0_F2_2026-09-10.md`.
Resume state: `docs/models/engine/RESUME.md`.

This is **v0**: the wiring, not a validated engine. Four of its seven adapters
stand on sub-models whose own bake-offs adopted nothing, and the engine says so
in every artifact it writes. Nothing here is tuned to make a gate pass; section 7
lists the gate failures with a component diagnosis instead.

---

## 1. Architecture

One batch of N concurrent simulations, advanced one possession at a time.

    GameState (src/cbb_sim/engine/state.py)
        struct of arrays over N = (games x seeds). Every field is a NumPy array
        whose row i is one simulation: period, seconds remaining (within the
        period), score, possession team, team fouls per period, the bonus and
        double-bonus thresholds (the RULE ERA, read from
        data/processed/models/free_throw/bonus_era.json per season), chance
        number, previous possession end type, the ten on-floor roster slots,
        per-player fouls and seconds, overtime count, the seven contract box
        counters per team and five per player.

    EngineInputs (src/cbb_sim/engine/inputs.py, built by build_engine_inputs.py)
        every quantity that is CONSTANT within a simulated game, as dense
        arrays: (G, 2, 28) team-level as-of form, (G, 2, 15, 19) per-roster-slot
        as-of form, the rotation prior, the usage shrunk rates, the per-player
        rebound rates, the ESPN athlete ids, and the data-derived rule
        constants. Side index 0 is always home; a `*_off` block is the
        OFFENCE's view.

    FeaturePlan (inputs.py)
        per sub-model, the splice: which of its declared features come from the
        static team block, the static slot block, or the per-step state block,
        and at which index. Assembling a design matrix is two fancy-index
        assignments, no pandas, no per-row work. A feature that resolves in
        none of the three blocks is a hard error -- the engine never hands a
        model a silently-zero column.

    loop.py
        the possession loop. One outer step advances every active simulation by
        exactly one possession; the OREB chain and the free-throw trip are inner
        loops over a shrinking SUBSET of rows, never over games.

### 1.1 Why this shape satisfies "lookup tables and vectorized NumPy"

CLAUDE.md bans live model calls in the sim loop. The engine splits every feature
into *static* (pregame, constant within a game) and *state* (a function of the
live simulation). All static work is done once by `build_engine_inputs.py` and
thereafter only indexed. Per step the loop makes **one batched predict per
sub-model for the whole batch** -- a single call covering every active
simulation, not a call per game. The tree models are not further reduced to
binned lookup tables; section 6 reports the measured throughput that decision
buys and what the lookup export would cost.

---

## 2. The possession loop, step by step

| stage | sub-model | what the engine does |
|---|---|---|
| (a) clock | `clock.QuantileTreeArm.pmf` + `clock.sample_from_pmf` | one batched pmf over 0..90 s for every active simulation, one inverse-CDF draw. A draw that exceeds the period's remaining time is the **censored** case the bake-off drops from training: the possession consumes what is left and still resolves. |
| (b) event | `possession_outcome` first/cont references | one batched predict per population. First chances -> the `first` reference (lgbm, `C_plus_state`, 21 features); continuation chances -> the `cont` reference (cascade, 22 features). Six classes in `PO.CLASSES` order, sampled by inverse CDF. |
| (c) allocation | `usage.draw_player`'s rule | the five on-floor slots' shrunk as-of class rates, `usage.normalise`d over the five, one uniform. The fitted (prior, m) per class come from `usage_params_v1.json`: rim (position, 50), jump2 (league, 50), three (position, 25), TOV (league, 200), FT-trip (position, 200). |
| (d) outcome | `fg_make` per class; `free_throw`; `rebound` | see section 3. |
| (e) rotation | `rotation_adapter.next_lineup` | both teams, credited with the possession's own duration; the scheduler decision rule of `rotation.RotationSampler`, vectorised. |
| (f) period | -- | halftime: clock 1200, **team fouls reset**, possession to whoever LOST the opening tip. Overtime: a 5-minute period opened by a fresh jump ball, **team fouls CARRY OVER** because an NCAA extra period is an extension of the second half. Repeat until untied (cap 12, counted). |
| (g) bookkeeping | -- | `possessions` written as the mean of the two sides' counts (the convention `reference.load_actual_possessions` grades against); `n_periods = 2 + n_ot`; minutes from per-player seconds. |

### 2.1 The composition order that makes `is_transition` legal

`possession_outcome`'s `is_transition` is `duration <= 8 AND the possession
started on a DREB or a TOV`. It is a function of the possession's own duration,
which is why `clock.BANNED_FEATURES` forbids the clock model from seeing it. The
engine draws the duration **first**, so by the time the event model is called the
value is already determined and is supplied with exactly its training definition.
The clock model never sees it. This is a property of the loop's ordering, stated
here because it is the one place the engine uses a feature another model is
banned from.

### 2.2 End-of-period composition

Real games contain `end_period` possessions (excluded from the event model's
training population). The engine instead reads a duration draw longer than the
remaining clock as **censoring**: the possession consumes the remaining time and
still produces a terminal event. The count of such possessions is reported
(`possessions_censored_by_period_end`). The alternative -- a no-event possession
-- loses roughly two possessions and two points per game and was rejected as the
worse of two stated approximations, not tuned between.

---

## 3. Outcome rules, all derived from data

| rule | value (F2 train = 2022-2024) | source |
|---|---|---|
| dead-ball rebound share by miss type | rim 0.00456, jump2 0.00643, three 0.00923, ft 0.00799 | `rebound.deterministic_dead_share` on the train fold (L17: a fixed share costs 0.49 noise floors vs a third class) |
| and-one rate given a made shot | rim 0.0989, jump2 0.0467, three 0.0067 | measured on `fg_make` design train rows with `made` |
| 3-attempt share of shooting-foul trips | 0.03998 | `trips_v1_era.parquet`, `foul_class == "shooting"` |
| 3-attempt share of double-bonus trips | 0.01720 | same, `foul_class == "double_bonus"` |
| non-shooting fouls per possession that award no attempt | 0.12335 | (603,991 personal fouls - 327,946 trip fouls) / 2,237,980 possessions |
| bonus thresholds | 6 / 9 prior team fouls, every season 2022-2026 | `bonus_era.json` (L18: no era boundary in the data) |
| median chance-elapsed seconds by chance number | 1 -> 16, 2 -> 3, 3+ -> 2 | `fg_make` design train rows |

### 3.1 The free-throw rule table

Implemented from `free_throw.TRIP_RULES`, keyed on GameState's team-foul counter
and the era thresholds:

- and-one (shooting foul on a made basket): **1** attempt, regardless of team fouls.
- shooting foul, defence below 6 team fouls: **2**, or **3** at the measured share.
- one-and-one, defence at 6-8: **1**, and a second **iff the first is made**.
- double bonus, defence at 9+: **2**, or **3** at the measured share.
- a missed last attempt is a live rebound with miss type `ft`; a made last
  attempt ends the possession with `prev_end = made_FT`.

`FT_trip_bonus` emitted while not in the bonus is awarded as a two-shot trip and
counted (`ft_bonus_awarded_out_of_bonus`), never silently dropped.

### 3.2 The foul-accrual gap (flagged `provisional_foul_accrual`)

The L3 class vocabulary has **no class for a non-shooting foul that awards no
free throw**, because such a foul ends no chance and is invisible in the chance
table. But it drives the bonus, and without it the bonus arrives far too late and
FTA collapses. The engine therefore draws one Bernoulli per possession at the
measured rate above and increments the defence's TEAM foul count only.

Personal fouls -- the ones that foul a player out and move minutes -- come from
`rotation`'s own fitted `fpm` hazard, which is how `rotation.py` does it. **The
two counters are not reconciled in v0**: a team's foul total and the sum of its
players' personal fouls need not agree. This is the engine's largest
self-declared structural gap after the clock.

---

## 4. Adapters, and exactly which are provisional

Adoption state is each sub-model's own verdict file, not a judgement made here.

| adapter | artifact | adopted? | flag |
|---|---|---|---|
| event (`possession_outcome`) | `reference_not_adopted_first.pkl` (lgbm, C_plus_state, F2), `reference_not_adopted_cont.pkl` (cascade) | **NO** -- round 1 had 0 of 11 arms pass; round 2 was still running when this was built and has written no winner | `provisional_event=True`, `ENGINE_EVENT=reference` |
| clock | `reference_not_adopted_lgbm_quantile.pkl` (9 quantile boosters, C_plus_score, F2) -- the best-CRPS arm | **NO** -- 0 of 20 eligible | `provisional_clock=True`, `ENGINE_CLOCK=reference` (`reference_empirical` also wired) |
| rotation | `rotation_fit.json`, R2 hierarchical Dirichlet + the fitted scheduler | **NO** -- state-dependence veto in both rounds | `provisional_rotation=True`, `ENGINE_ROTATION=reference` |
| usage | `usage_params_v1.json` (fitted prior + m per class) | winner is the LightGBM choice arm, whose **booster is not persisted anywhere**; the engine runs `draw_player`'s own U1 proportional path | `provisional_usage=True` |
| fg_make | `winner_FGA_rim.joblib`, `winner_FGA_jump2.joblib` (lgbm, C_plus_state, F2); FGA_3 per `ENGINE_FG3` | **YES** | `provisional_fg=False`; `ENGINE_FG3` recorded |
| rebound | refit here: lgbm / C_plus_state on F2 train | **YES** -- but `train_rebound_v1.py` persists no model (its docstring promises `winner.joblib` and never writes it), so the engine refits the winner's own spec from its own module | `provisional_rebound=False`, refit recorded |
| free_throw | refit here: lgbm on F2 train | **YES** -- same, no artifact exists | `provisional_free_throw=False`, refit recorded |

### 4.1 The FGA_3 / Decision 8 conflict

Decision 8 amended the responsiveness gate and states explicitly that *"FGA_3
changes from team_baseline to LightGBM under the corrected gate"*.
`winner_FGA_3.joblib` predates that decision (written 15:36, Decision 8 at 15:47)
and still holds `team_baseline`, whose shooter slope ratio is **0.0086** -- a
three-point model that cannot separate a 15% shooter from a 51% one.

`fg_make` is read-only for this deliverable, so the engine does not re-export it.
Instead `build_engine_inputs.py` fits the Decision-8 model into its **own**
directory and `ENGINE_FG3` chooses:

- `decision8` (default): the engine's lgbm / C_plus_state refit for FGA_3.
- `artifact`: the stale `team_baseline` winner, for a like-for-like read against
  anything already graded against it.

Which one ran is in `run_meta.json` under `adapter_flags.ENGINE_FG3`.

### 4.2 Static / state split per sub-model

| sub-model | static per game-team | static per player | state |
|---|---|---|---|
| possession_outcome | 8 centred style rates, 4 own-ridge ratings, site_home/away, season_idx, days_since_start | -- | period, seconds_remaining, score_diff, in_bonus, is_transition (+ chance_number for `cont`) |
| clock | off/def_tempo_rel, tempo_prior_game (the **L2 multiplicative winner as a prior feature**, Decision 7), 4 ratings, site | -- | 5 prev-end dummies, period, is_ot, seconds_remaining, score_diff, score_diff x seconds_remaining / 1200, in_bonus |
| fg_make | off_make_c, def_allow_c (per shot class), 4 ratings, site, season_idx | shooter_make_c, shooter_att_c, prior_season_make_c (per class), has_prior_season, pos_G/F/C, shooter_games_asof, shooter_fga_asof | period, seconds_remaining, score_diff, in_bonus, chance_number, chance_elapsed_s, is_transition_f |
| free_throw | season_idx | shooter_ft_asof, shooter_fta_asof, prior_season_ft, has_prior_season | seconds_remaining, period, score_diff, in_bonus |
| rebound | off_oreb_c, opp_def_dreb_c, 4 ratings, site | -- (L17: the model is team-level; per-player rates are attribution only) | miss_rim/jump2/three, blocked_f, period, seconds_remaining, score_diff, in_bonus |
| usage | -- | shrunk as-of class rate per event class | -- (the U1 path is state-free) |

Sign conventions: `score_diff` is **offence minus defence** for every sub-model
except `RotationState`, which takes home minus away; `seconds_remaining` is
always **within the period**; `usage.sec_remaining` is derived as
`(2 - period) * 1200 + seconds_remaining` in regulation.

### 4.3 As-of feature staleness, stated

A roster slot with no `fg_make` / `free_throw` / `usage` design row for the game
being prepared takes his most recent **earlier** row
(`merge_asof(direction="backward")`): at most one of his own games stale, never
forward-looking. A slot with no earlier row at all falls to the model's own
no-history state -- a centred rate of exactly `0.0`, which on a league-centred
scale **is** the league mean, and the class prior for usage. 134,707 of 171,300
roster slots are named players; the rest are `rotation.extend_profile`'s
anonymous fitted tail slots.

### 4.4 The rotation fallback (10.0% of team-games)

`rotation.build_priors` yields nothing for a team with fewer than
`min_prior_games` games of as-of history. The engine calls it with
`min_prior_games=1` and, for the 1,144 of 11,420 team-games still without a
prior (a team's first game of the season), substitutes a profile built **only
from fitted training-season parameters**: `fit.role_prior` shares by as-of rank,
`fit.p_play` availability, `fit.fpm_league` foul rate, anonymous negative ids. No
team identity, no look-ahead, and those slots are excluded from
`players.parquet` because they have no athlete identity. The alternative -- using
the game's observed participants -- would be a look-ahead, and dropping the games
would empty out November.

### 4.5 Two deliberate divergences in the rotation adapter

`rotation.py` is read-only and its `RotationSampler` holds one team of one game
in Python objects with a sequential `PCG64`. At 3.1e8 calls that cannot be the
engine's rotation layer, so `rotation_adapter.py` re-expresses the **same
decision rule** over (2N, 15) arrays. Two things differ and are not hidden:

1. **RNG.** Availability, the Dirichlet targets and the foul hazard come from the
   engine's counter-based (seed, game_id, side, family, ordinal) stream, with the
   side folded into the key the way `usage.event_stream_keys` folds the event
   ordinal. The decision rule is identical; the realised numbers are not the
   ones the offline sampler would produce for the same (seed, game_id).
2. The foul hazard draws one uniform per roster slot and masks to the five on the
   floor, rather than five uniforms for the five. Same per-player distribution,
   different stream positions.

---

## 5. RNG discipline

Every draw goes through `engine/rng.py`, which adds nothing to
`control/rng.py`'s construction -- same splitmix64, same `stream_keys` -- and
only generalises `uniforms(keys, index)` to a **per-row** index, because the
engine's rows sit at different points of their own game's stream.
`tests/test_engine.py::test_uniforms_at_matches_control_rng` pins the
equivalence bit for bit.

`StreamBook` holds one counter per (family, simulation) and advances **only the
rows it draws for**, so a simulation that skipped a possession does not consume a
draw and no other simulation is affected. Families: clock, event, usage, fg_make,
free_throw, rebound, and_one, foul_accrual, rotation, rotation_foul, tipoff.

**This is the L19 fix, generalised.** The usage bake-off's shared-uniform bug was
a scalar index: every event of a game drew the same uniform and the game's
allocation collapsed onto one player (top-1 usage share 60.9% vs 32.3% real,
3.42 distinct three-point shooters per team-game vs 6.69).
`test_two_consecutive_draws_in_one_game_differ` asserts, for every family, that
three consecutive draws in one game-sim are three different numbers, and
`test_allocation_does_not_collapse_onto_one_player` asserts the downstream form.

`test_a_games_draws_do_not_depend_on_the_batch_it_is_in` pins the property the
RNG rule exists to give: dropping games from the batch moves no remaining game's
result, so paired arms and paired seeds difference game by game.

---

## 6. Performance

Design: one batched predict per sub-model per step over every active simulation;
all state updates in NumPy. Measured single-core predict throughput on this
machine (50,000 rows, `OMP_NUM_THREADS=1`):

| sub-model | rows/s/core | share of a possession |
|---|---|---|
| clock (9 quantile boosters x 300 trees, num_leaves 63) | **4,900** | 1 per possession |
| possession_outcome `first` (lgbm, 6-class, 400 trees) | 48,000 | 1 per chance |
| possession_outcome `cont` (cascade, 5 logits) | 2,400,000 | ~0.12 per possession |
| fg_make rim / jump2 (lgbm) | 309,000 / 407,000 | ~0.6 per possession |
| fg_make FGA_3 (A_team ridge) | 5,500,000 | ~0.25 per possession |

**The clock arm is ~85% of the model cost of the engine.** It is nine LightGBM
quantile regressions of 300 trees each, called once per possession; everything
else together is under a sixth of it.

End-to-end, the MEASURED rate at a 6,000-simulation batch on a contended machine
is **862 possessions/s/core** (862,621 possessions in 1,000.9 s), which projects
to **2.7 hours for 5,710 x 200 on 20 cores and therefore MISSES the 2-hour
target**. The per-model benchmarks above would project ~3,500 poss/s/core on a
free core; no point in the build session had free cores, so the dedicated-core
rate is unmeasured and the 5.7x gap between the two is unexplained. Full working:
`docs/tests/engine_v0_F2_2026-09-10.md` section 2.

On that evidence the lookup export is REQUIRED work, not a contingency. The
cheapest version of it is already on disk: the
clock bake-off's `reference_not_adopted_empirical.pkl` **is** a lookup table --
`EmpiricalArm.level_pmfs` is a (n_cells, 91) pmf array over 7 nested binned state
dimensions of sizes (6, 5, 2, 3, 3, 5, 2) -- and is wired behind
`ENGINE_CLOCK=reference_empirical`. Exporting the quantile arm itself to bins
over its 20 features is the remaining work; its binning error has **not** been
measured and must not be assumed small.

---

## 7. Known defects, with the component diagnosed

Ordered by size. None of these is corrected anywhere in the engine.

1. **Possessions per game run high and points per possession run low, nearly
   cancelling in the total.** On a 6,000-simulation sample (300 games x 20 seeds)
   the engine produced 71.89 possessions/game against an actual 67.875 and 143.74
   total points against an actual 145.5 -- i.e. PPP 1.000 against 1.071 -- while
   every per-possession rate the L3 sub-models own landed within half a point
   (3PA share -0.005, FTA/FGA 0.000, OREB% -0.001). **Component: the clock
   model.** Its own bake-off adopted nothing precisely because no arm passed the
   emergent G1 gate; the engine inherits that, and Decision 7's acknowledged risk
   ("G1 now depends on the clock-consumption model being right by state") is
   realised. This is also the textbook multi-level-evidence case from CLAUDE.md:
   a passing total hides two offsetting errors underneath.
2. **Overtime rate is low** (1.3% on that sample against 5.58% actual). Ties
   at the end of regulation are rarer than they should be, which is a
   *dispersion* symptom at the end of the second half, not an overtime-rule
   symptom. Component: the clock model's late-period duration distribution plus
   the engine's end-of-period censoring read (section 2.2).
3. **The team-foul / personal-foul split is unreconciled** (section 3.2). Team
   fouls drive the bonus and come from events plus a measured accrual rate;
   personal fouls drive foul-outs and come from the rotation hazard.
4. **FGA_3 is flat in shooter skill** under `ENGINE_FG3=artifact`, and under
   `decision8` the engine is running a model the adopted artifact does not
   contain (section 4.1). Either way this is a sub-model bookkeeping defect, not
   an engine one.
5. **`ast` is a placeholder.** There is no assist model; `players.parquet` writes
   0 and `run_meta.json` carries `ast_is_placeholder: true`. G8's assist leg must
   read NEEDS-INSTRUMENTATION.
6. **Under-concentrated top usage** is inherited from L19's open defect (the U1
   path spreads events too evenly across the five). It will show in G8's top-1
   share, not in game markets.
7. **Dead balls are handed to the defence.** The rebound model's `DEAD` class
   says nobody rebounded; only 58.1% of dead balls are followed by the shooting
   team's next action, and the engine has no model for the split, so it gives
   them to the defence and counts them (`dead_ball_rebounds`).

---

## 8. Reproduce

    .venv/Scripts/python.exe scripts/build_engine_inputs.py --fold F2 --season 2025
    .venv/Scripts/python.exe scripts/run_engine.py --fold F2 --season 2025 --seeds 200 \
        --workers 16 --games-per-block 250 --seeds-per-block 25
    .venv/Scripts/python.exe scripts/eval_gates.py --results results/engine_v0/F2_2025 --season 2025
    .venv/Scripts/python.exe scripts/grade_market_games.py --results results/engine_v0/F2_2025 --season 2025
    .venv/Scripts/python.exe -m pytest tests/test_engine.py -q

Season 2026 is refused unless `CBB_UNSEAL=1`; the fold's test season is also put
through `cbb_sim.data.seal.assert_not_sealed` before anything is read.
