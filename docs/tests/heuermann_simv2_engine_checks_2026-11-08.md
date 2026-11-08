# Heuermann sim_v2 exposures: are they in OUR engine? (2026-11-08)

Opus worker diagnostic for the reply to Matt Heuermann's `CBB Model/Heuermann/study/sim_v2/PROGRESS_2026-11-07.md`
(s2, s4.3) and `Handoffs - Communication/2026-11-07_Heuermann_INTRO_cbb-sim-v2.md` (s2 items 3, 4, 6).
**Diagnostic only. No engine code changed, nothing adopted, nothing fitted to sim output.** Seasons 2022-2025
(2021-22 .. 2024-25) only. 2025-26 stays SEALED and was not loaded.

Script: `scripts/diag_heuermann_simv2_checks_v1.py` (about 1 min; one instrumented in-process replay of
`cbb_sim.pbp.possessions._GameMachine` with the v4 switch set over CBBD plays 2022-2025, plus the served-v3
closed-loop trajectories already on disk). CSVs (all < 6 KB) went to
`rfq-collab/CBB Model/Handoffs - Communication/2026-11-08_Peavler_REPLY_cbb-sim-v2/engine_*.csv`.

**Sim source.** No new sim run was needed. `results/trajectories/eg_ctrl_o{0,1000,2000,3000,4000}_s25`:
the late-game round-7 served-v3 control (2026-10-13, engine commit 12788a5, no ENGINE_* override, `ENGINE_CLOCK=v5b_r8K2_glat_pmean`,
`ENGINE_FOUL_JOINT=R9ao3`, `ENGINE_EVENT=round2_s1`). 500 verified 2024-25 games
(`stride500_verified_minswap_v1_F2_2025`) x 125 seeds = 62,500 game-seeds, 8.6 M possessions. The freeze has been on since,
and nothing has been adopted since 10-07, so this is the stack we serve now.

## Verdicts

| item | his finding | our engine | verdict |
|---|---|---|---|
| A late-game race | independent foul / TOV / shot clocks, earliest one plays; leader shoots .16 vs .10 at 31-60 s, lead 4-6 | **no race.** One duration draw, then one categorical class draw from a model fitted on real terminal-event frequencies, then K2 redraws the time given the class. **The symptom is still there:** leader possession with an FGA .232 sim vs .140 real; foul before any shot .651 vs .731. The cause is a margin response that is too flat (1-3 runs the other way), not censoring. His TOV-timing symptom is reversed: our leader TOs run long (12.4 s vs 6.9 s) | **PARTLY**: same symptom from a different mechanism; TOV timing opposite |
| B1 row-less two-shot trips | 2.3-2.7% of two-shot trips; fit counted rows, so the sim reached the bonus early | 0.59-0.69% (CBBD, builder definition), 0.90-1.26% (hoopR second source); 0.21-0.22 row-less trips per game (hoopR 0.26-0.34; his .375). The engine charges a team foul for EVERY trip, and the accrual target nets the row-less trips out. The data's bonus LABEL lags on 0.20-0.24% of possessions. The sim does not reach the bonus early: in 2H it is LATER and less often (.897 vs .953 of team-halves) | **NOT EXPOSED** to his early-bonus mechanism; small label lag in training data; separate 2H bonus shortfall |
| B2 0-s period-end row / no-event rows | the real log writes a 0-s row at each period end; sim wrote it only in period 2 | the builder never opens a possession on an End Period row. An open possession closes as `end_period` (censored, one per period at most). **But a possession that would have held the ball to the horn after a close (DREB/made FG at 0:0x, nothing logged after) is DROPPED on the real side: 1.17 periods per game, mean 9.5 s unlogged.** The engine always plays that tail and resolves it with an event | **PARTLY / different exposure**: no 0-s phantom, but a real-vs-sim possession-definition gap of about 1 per game at period ends |
| B3 technicals as team fouls | counted as ordinary team fouls (.37/g) | technical rows never increment team fouls in the builder; only the rare duplicate PersonalFoul row for a technical does (0.002-0.013/g). The engine has no technical fouls. 0.32-0.46 technical rows per game | **NOT EXPOSED** |
| C fit-walker phantom segment | every period-end possession got a second censored segment (9,485 of 47,341) | one record per possession and per chance by construction; 0 duplicate keys in every table the served clock reads; at most one censored row per (game, period) in the clock design; the design-to-censoring join is enforced 1:1 | **NOT EXPOSED** |

## A. Late-game possession resolution

### A.1 Mechanism (code)

There is no clock race anywhere in the served loop:

1. **Duration first, once.** `src/cbb_sim/engine/loop.py:474-484`: one uniform per possession (`book.draw("clock")`), one
   inverse-CDF draw from the clock pmf on the state block. A draw past the period end is capped (`censored = dur >= left`, :482;
   `used = np.minimum(dur, left)`, :484). The served clock's score information is the P3 end-game indicators
   (`src/cbb_sim/models/clock_v3.py:1008-1021`): trailing-big / leading-big / close, with margin 4 and the last 120 s of H2/OT only.
2. **Then ONE categorical class draw.** `loop.py:533` `cls = categorical(book.draw("event"), probs)` over the six classes
   `TOV, FGA_rim, FGA_jump2, FGA_3, FT_trip_shooting, FT_trip_bonus` (`src/cbb_sim/models/possession_outcome.py:81-83`).
   `probs` comes from the possession-outcome model (`ENGINE_EVENT=round2_s1`, multiclass LightGBM, `possession_outcome.py:669`),
   trained on real chance terminal events (`:406`) with state features `period, seconds_remaining, score_diff, in_bonus,
   is_transition` (`:297-299`). The served foul joint then shifts the two FT-trip logits by half, live defence fouls and
   foul differential (`loop.py:530`, `src/cbb_sim/engine/foul_joint.py` "trip" table). A foul ending the possession is a
   CLASS with its own fitted frequency, so "the defence fouls because the leader won't shoot" enters as an observed outcome
   share. Nothing infers it from a shot hazard with the foul as independent censoring.
3. **Then time given class.** K2 (`loop.py:115-129`, called at `:767`): the first-chance time is redrawn at the SAME uniform
   from d1 | chance-1 end class (`src/cbb_sim/engine/clock_adapter_v3.py:469`), and an OREB adds a continuation time.
4. Non-trip fouls (`loop.py:736-764`) are a separate accrual draw that changes the team-foul count only. They never end or
   censor the possession.

So the factorisation is P(class | state) x P(time | class, cell). It is categorical-then-conditional and has no independence
assumption between foul, TOV and shot. **Heuermann's specific bias (a shot hazard read high because fouls are treated as
independent censoring) cannot arise here.**

### A.2 Measured cells (`engine_lategame_cells.csv`)

Possessions starting in period 2 with 31-60 s left (`start_clock`), split by the offence's pre-possession margin. The
classes are count-based and identical on both sides: **FGA** = possession has >= 1 FGA. **foul, no shot** = FTA > 0 and no
FGA (in both the feed and the engine a shooting foul on a missed shot is not an FGA). **TOV, no shot** = TOV with no FGA or FTA.

| cell | source | n | FGA | foul, no shot | TOV, no shot | foul / (foul+FGA) | mean dur | TOV mean dur | TOV >= 30 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| leader 1-3 | real 2022-25 | 3,801 | .463 | .367 | .167 | .442 | 17.5 | 13.7 | .060 |
| leader 1-3 | real 2025 | 957 | .490 | .352 | .157 | .418 | 18.1 | 14.1 | .087 |
| leader 1-3 | **sim** | 11,371 | .434 | .408 | .158 | .485 | **11.3** | 11.7 | .055 |
| **leader 4-6** | real 2022-25 | 4,233 | **.140** | **.731** | .129 | **.839** | 7.4 | **6.9** | **.017** |
| leader 4-6 | real 2025 | 1,036 | .139 | .716 | .145 | .838 | 7.5 | 7.4 | .020 |
| **leader 4-6** | **sim** | 11,863 | **.232** | **.651** | .117 | **.737** | 7.5 | **12.4** | **.127** |
| leader 7-12 | real 2022-25 | 7,448 | .186 | .669 | .144 | .782 | 8.6 | 11.4 | .135 |
| leader 7-12 | sim | 22,039 | .241 | .609 | .150 | .717 | 7.9 | 12.7 | .109 |
| trailer 4-6 | real 2022-25 | 4,319 | .725 | .172 | .101 | .192 | 10.2 | 7.4 | .000 |
| trailer 4-6 | sim | 12,535 | .732 | .165 | .103 | .184 | 10.3 | 8.3 | .003 |

The real same-500-game rows (n = 87-155 per cell) are in the CSV and are **UNDERPOWERED**. The 2022-25 and 2025-only real rows
agree, so the real level is stable across seasons. Cell occupancy matches: 0.19 leader-4-6 possessions per game in both the real
data and the sim. SE of the real 4-6 FGA share is about .005, so the .09 gap is about 17 SE.

The real leader dead-ball TOs (not stolen) in the 4-6 cell: n = 209, mean 6.6 s, 2.9% (6 of 209) last >= 30 s,
14.8% team-credited. **The feed has no shot-clock-violation type** (CBBD `Lost Ball Turnover` text is "X Turnover."; hoopR
has the same type list). We therefore report two proxies:
(a) a TOV possession lasting >= 30 s, which can be computed on both sides;
(b) a team-credited TOV (no participant on the row), real side only. Proxy (b) also catches 5-s inbound and 10-s backcourt calls.
The engine has no violation class at all. A shot-clock turnover can appear only as a TOV whose drawn time is >= 30 s.

Reading:
- **Same symptom, different cause.** At lead 4-6 the sim leader reaches a shot too often (.232 vs .140) and the defence fouls
  first too rarely (.651 vs .731; conditional .737 vs .839, close to his .80-vs-.725 pattern but on the other side of real).
  At lead 1-3 the errors flip (FGA .434 vs .463, foul .408 vs .367). At 7-12 the sim is too high on FGA again.
  **The engine's response to margin is too flat**: the event model sees `score_diff` and `seconds_remaining` as smooth inputs
  pooled over all possessions, and it does not separate the one-possession regime (1-3) from the intentional-foul regime (4-6)
  sharply enough. The trailer side is right (.732 vs .725).
- **TOV timing is reversed from his.** His sim leader TOs are too short (14.3 vs 17.5 s). Ours are too long: 12.4 s vs 6.9 s,
  and 12.7% vs 1.7% last >= 30 s. In the real data, at 4-6 with 31-60 s left the trailer fouls at once, and the leader's
  turnovers are quick inbound or pressure turnovers. The K2 TOV-conditional time law cannot see this: its score input is the
  coarse P3 indicator (leading-big covers 4-6 and 7-12 alike).
- **Overall duration is right at 4-6 (7.5 vs 7.4) and badly short at 1-3 (11.3 vs 17.5).** The same root cause shows up in
  the clock cell: below a margin of 4, P3 calls it "close", which pools leaders who hold with tied and trailing teams.
- His real 17.5 s and 23% shot-clock-violation numbers do not match our real 6.6-6.9 s and 2.0-2.9%. That points to a
  definitional difference in "dead turnover", or a different cell, which should be reconciled with him before either side
  quotes the other.
- This is the area the late-game round 7 and the clock/endgame follow-up already worked on, both NO ARM
  (`docs/tests/endgame_round_2026-10-13.md`, `docs/tests/endgame_clock_followup_AB_2026-10-14.md`; block A's cell term failed
  the trailer slope gate). These cells give a sharper target: the margin-band x time interaction of the possession-outcome
  class mix, plus the TOV time law in the leader-4-6 cell. Any fix is a bake-off, not a scale.

## B. Feed artefacts

### B.1 Two-shot trips with no foul row (`engine_rowless_ft_trips.csv`, `engine_rowless_ft_trips_hoopr.csv`)

| season | CBBD 2-shot row-less | CBBD all trips row-less | CBBD row-less / game | hoopR 2-shot row-less | hoopR row-less / game |
|---|---:|---:|---:|---:|---:|
| 2022 | 451 / 75,155 = 0.60% | 1.26% | 0.206 | 0.90% | 0.262 |
| 2023 | 507 / 81,517 = 0.62% | 1.31% | 0.222 | 0.94% | 0.325 |
| 2024 | 516 / 87,413 = 0.59% | 1.20% | 0.214 | 0.96% | 0.279 |
| 2025 | 612 / 88,266 = 0.69% | 1.20% | 0.217 | 1.26% | 0.336 |

CBBD definition: the builder's own. A trip reaches `_handle_ft_trip` from the main loop with `prior_fouls=None` (no foul row
immediately before it, inert rows already dropped; technical trips excluded via the v4 lookahead). The D-I universe is used.
hoopR definition: a run of `MadeFreeThrow` rows whose previous non-inert row is neither `PersonalFoul` nor `Technical Foul`,
over all games in the file. **Our shares are a third to a half of his 2.3-2.7%. Per game, the hoopR count (.336 in 2025) is
close to his .375.** His denominator or trip definition probably differs; worth a one-line check with him.

Treatment in our stack:
- **Builder:** a row-less trip uses the defence's CURRENT count to classify the trip
  (`src/cbb_sim/pbp/possessions.py:938-942`) but **does not add a team foul**. Only `_handle_foul` increments (`:857-874`).
  The data's `def_team_fouls` / `off_in_bonus` therefore lag by the row-less trips. Measured: the defence count differs on
  2.0-2.2% of regulation possessions, the bonus label on 0.20-0.24%, the double-bonus label on 0.23-0.25%
  (`engine_rowless_bonus_label_effect.csv`).
- **Engine:** every FT trip charges the defence one team foul (`loop.py:633`; and-one `:612`), so a row-less trip is a foul in
  the sim. Non-trip fouls come from the served accrual table, whose target is
  `y_nt = (def_silent + def_trip + off_silent + off_trip - trips) >= 1` (`scripts/train_foul_joint_v1.py:80-81`). Foul ROWS minus
  ALL trips: a row-less trip makes the difference negative and cannot create a phantom non-trip foul. His M95 mechanism (rows
  counted, so the bonus comes early) is absent. The residual is the opposite, small effect: the accrual and trip-offset tables
  are INDEXED by the lagged label count, the engine queries them with the full count, and the two disagree on about 2% of
  possessions.
- **Does the sim reach the bonus at the right time?** (`engine_bonus_onset.csv`; same 500 games, real n = 984 team-halves,
  sim 125,000):

| half | source | share of team-halves reaching bonus | mean onset (s left) | median |
|---|---|---:|---:|---:|
| 1H | real (label) | .711 | 334 | 320 |
| 1H | real (row-less counted) | .715 | 336 | 320.5 |
| 1H | sim | .682 | 346 | 331 |
| 2H | real (label) | .953 | 474 | 493 |
| 2H | real (row-less counted) | .954 | 476 | 493 |
| 2H | sim | **.897** | **458** | **469** |

  Counting the row-less trips moves the real side by only +0.4 pp and about 2 s. The sim reaches the 2H bonus **later and less
  often** (-5.6 pp; SE of the real share about .007). In 1H it reaches it slightly less often but with an earlier conditional
  onset. Caveat: the real label counts the pre-open foul at the possession's open (`build_foul_accrual_design_v2.py`
  docstring); the sim's `off_in_bonus` is read before the possession's own fouls. That can move the real onset earlier by up to
  one possession but does not change the reaching-share gap. This 2H shortfall is the known leader-bonus-occupancy deficit
  that late-game LF2 moved 1.6-2.2 floors toward real (round 7, NO ARM). It is not a feed artefact.

### B.2 The 0-second period-end row, and rows with no event (`engine_period_end_rows.csv`)

Code:
- `End Period` / `End Game` rows (all at 0 s in CBBD) are boundaries. They close an OPEN possession with terminal
  `end_period` and never open one (`possessions.py:677-689`; a stray open possession at end of feed closes the same way, `:742`).
  So no 0-s phantom row exists on the real side.
- An `end_period` possession (the period ended with the ball live and nothing else logged) is **kept**. It counts as a
  possession (pace target) and is the clock's censored row (`src/cbb_sim/models/clock.py:491`; the served horn flag, which also
  covers buzzer shots, turnovers and trips at 0 s, is attached 1:1 at `clock_v3.py:126`). The possession-outcome trainer drops
  it, together with `unknown` terminals, because those are not one of the six classes (`possession_outcome.py:406`).
- `unknown`-terminal possessions (mismatch-guard closes with nothing pending) are kept as possessions.

Measured, regulation only, per game:

| source | games | poss/game | `end_period` (no event) | `unknown` | 0-s duration | start <= 2 s left | start <= 5 s left | **periods whose last logged possession ends with time left** | mean unlogged tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real 2022-25 | 21,969 | 135.45 | 0.27 | 0.35 | 2.16 | 0.17 | 0.46 | **1.17** | 9.5 s |
| real 2025 same 500 | 492 | 136.12 | 0.30 | 0.13 | 2.16 | 0.21 | 0.54 | 1.14 | 12.5 s |
| sim | 62,500 | 136.99 | 0 | 0 | 2.50 | 0.30 | 0.76 | 0 | -- |

**A different exposure from his.** After a close at 0:0x (DREB, made FG or FT, TOV), if nothing else is logged before the horn,
the builder opens no possession: `_open` is only ever called by an event, and the End Period row finds `cur is None`. The real
table therefore **omits the team's final hold-to-horn possession in 1.17 of 2 regulation periods per game** (mean 9.5 s of
clock). The engine always plays that tail (`loop.py:482-484`, censored but "still resolves" with an event; the round-4 buzzer
no-event law at `:495` is DEFAULT OFF). Consequences:
- On the same 500 games, sim possessions/game exceed the real build by 0.87, against 1.14 unlogged tails per game. A
  pbp-built possession count therefore reads about 0.6% low against the sim for definitional reasons alone. (Our pace/total
  gates grade against verified finals and box possessions, not this table. Any possession-count comparison against
  `possessions_v*` should carry this offset.)
- The clock's period-end censoring set is missing these tails entirely: neither a censored nor a complete row. The dropped
  possessions are exactly the long holds, so the near-horn duration law is trained on a sample with them removed.
- The sim has more possessions starting with <= 5 s left (0.76 vs 0.46 per game) and gives each an event, where in real games
  many of these are no-event holds. This links to his J14 (horn endings high in his sim) and should be a candidate for a
  pre-registered period-end round, not a patch.

### B.3 Technical fouls

- 0.32-0.46 `Technical Foul` rows per game in CBBD (0.31-0.45 in hoopR). His .37 is in range.
- Builder: `_handle_technical` (`possessions.py:876-897`) never touches `team_fouls`. The duplicate `PersonalFoul` row the feed
  sometimes logs for a technical DOES count (`:863-871`, deliberate: the bonus thresholds were validated on counting every
  PersonalFoul row, `:132-141`). That is 0.013 / 0.008 / 0.006 / 0.002 per game in 2022-2025 under the v4 lookahead.
  The foul-accrual design logs technicals separately and keeps them out of `def_silent`/`def_trip`
  (`scripts/build_foul_accrual_design_v2.py:92-94`).
- Engine: no technical-foul mechanism and no team-foul increment for one (no reference anywhere in `src/cbb_sim/engine/`).
- **Not exposed to M48.** (Caveat: the served v2 tables were built WITHOUT the technical lookahead, so a technical's free
  throws that follow a duplicate PersonalFoul row are classed there as an ordinary trip. That is the documented v2 defect
  fixed in v3/v4; `possessions.py:175-242`.)

## C. Fit-walker phantom censored segment (`engine_fitwalker_duplicates.csv`)

Code: the builder emits exactly one row per `_Possession` and one per `_Chance` (`possessions.py:1135-1225`). A possession is
closed once: `_close` sets `cur = None` (`:625-640`). Our clock fitters do not walk possessions into segments. They read those
rows directly:
- The censored flag is set per possession row (`clock.py:491`).
- The served horn flag is merged with `validate="1:1"` and a row-count assert (`clock_v3.py:126-131`).
- K2's d1 rows are censored only when the possession has ONE chance and was horn-censored, and its continuation rows exclude
  censored possessions (`scripts/train_clock_r8_chance_v1.py:74-76, 100`).
- The engine caps a censored draw once per possession (`loop.py:482-484`).

Counts on real data, 2022-2025:

| table | check | result |
|---|---|---|
| possessions_v2 / v4, each season | duplicate (game, period, poss_index) and (game, poss_index) | 0 |
| chances_v2 / v4 | duplicate (game, period, poss, chance) | 0 |
| chances_v2 / v4 | possession with > 1 `end_period` chance; `end_period` on a non-last chance | 0 / 0 |
| possessions_v2 / v4 | (game, period) with > 1 `end_period` possession | 0 except 2 in 2025 (both versions; 2 of 1,808 / 1,680 period-ends, period-label oddities, not walker duplicates) |
| clock `r6_L2/design_v2` (served clock design, 2,453,120 rows) | duplicate possession key; (game, period) with > 1 censored row | 0 / 0 |
| clock `r6_L2/clock_censoring` (3,002,772 rows) | duplicate key; design rows with no censoring row | 0 / 0 |
| clock_censoring | (game, period) with > 1 `censored_horn` row | 715 of 19,074 |

The 715 are not phantoms. They are distinct real possessions that both end at 0 s: a buzzer shot and then the fouled
opponent's 0-s trip, a 0-s TOV, and so on (`engine_horn_multi_by_terminal.csv`; mean durations 2.6-18 s by terminal class).
A zero-length censored row is harmless to a Kaplan-Meier risk set. **Not exposed.**

## Caveats

- Sim = 500 games of 2024-25 only. The real comparators are 2022-25, 2025-only and the same 500 (underpowered). The sim
  shares carry game-context clustering (500 contexts), so treat sub-.01 sim-real gaps as noise. The A gaps are 5-10x that.
- Sim classes come from box diffs in the trajectory (`src/cbb_sim/engine/trajectory.py`). The trajectory does not record the
  chance-1 class, so the classes are count-based on both sides. The real chance-1 split (`p_c1_fga`, the bonus-class share) is
  in the CSV for context.
- Shot-clock violations are proxies only: the feed has no type, and the engine has no violation class.
- His numbers come from his own definitions and his 2026 reads. Ours are 2022-25. The A.2 TOV-timing disagreement in the real
  data and the B.1 denominator need a definition check with him before anyone quotes a cross-team comparison.
