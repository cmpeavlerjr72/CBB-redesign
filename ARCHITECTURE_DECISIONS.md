# Architecture Decisions Log

Every major design decision, the reasoning, and the alternatives considered. Format: Decision, Why, Alternative considered, Risk acknowledged. Decisions are numbered and never deleted; a superseded decision gets a dated note pointing at its replacement.

---

## Decision 1: Possession-level simulation, with a Control engine as the yardstick (2026-09-10)

**Decision:** Build a possession-by-possession Monte Carlo engine. On each possession, the players on the floor, offence and defence strengths, and the game state (score, time, period, fouls, bonus) feed a cascade of sub-models that produce a terminal event and clock consumption. Scores, box lines, and every market fall out of the same trajectory.

**Why:**
- One engine produces spreads, totals, moneylines, team totals, halves, and player props at once, and the correlations between them emerge from shared game flow instead of being modelled (cfb-props-sim Decision 1).
- "Bottom-up, prove every stage" needs stages. A box-score draw has no bottom.
- Last year's box-score draw had no residual signal after controlling for the closing line (t = 0.86 on margin, t = -0.74 on total, n = 5,229). There is nothing to repair.

**Control engine:** Week 1 builds a cheap team-level rate engine done right (rates per possession, features centred on snapshot league means, home court, one shared pace draw per sim, fitted dispersion, refit on all data). It is the yardstick. The possession engine must match or beat Control on every game-level gate before any prop work is built on it. If it cannot, that is a defect in a sub-model to be found, not a reason to skip the gate.

**Alternative considered:** Fixing last year's team-level engine. Rejected: no residual signal, no player layer possible, no correlation structure. Its corrected form survives only as the Control baseline.

**Risk acknowledged:** A possession engine at ~140 possessions per game costs roughly what the CFB play-by-play engine costs per game, far more than last year's twelve-vector draw. Mitigation: lookup tables and vectorized NumPy from day one; the 196-core EC2 box for full-season seed sweeps.

---

## Decision 2: Vegas-independent; the team-strength anchor is a bake-off, not a pick (2026-09-10)

**Decision:** Betting lines are a comparison layer only, never a feature. The team-strength anchor (offence and defence efficiency per possession, tempo) is chosen by bake-off among: (a) KenPom point-in-time snapshots centred on each snapshot's league mean; (b) our own regularised ratings fitted from hoopR box data with home flag and recency; (c) a blend. Each arm passes the leak test before it enters a feature table.

**Why:**
- Circularity: props derived from the same team totals as the book find no disagreement.
- Last year's single largest defect was raw KenPom levels on a drifting scale (league-mean AdjO 100 -> 109.3). Centring is non-negotiable in arm (a).
- Arm (b) is fully compliant and reproducible, and is the fallback if KenPom access ever becomes a problem. The user asked to keep the KenPom snapshot store regardless.

**Alternative considered:** Picking KenPom outright as CFB picked SP+. Rejected under the bake-off rule; the cost of building arm (b) is small and it doubles as a leak-free control.

---

## Decision 3: Data source is hoopR-mbb-data; several sources are retired (2026-09-10)

**Decision:** Play-by-play, player box (with minutes), team box, schedules, shots, rosters (2025-26 on), and player bios come from sportsdataverse/hoopR-mbb-data (GitHub, CC BY 4.0, daily cron, built from ESPN's public JSON). Seasons 2021-22 through 2025-26 for backtests. ESPN's own JSON is used lightly in-season only for same-day items (lineups, injuries, lines snapshot).

**Retired:** Sports-Reference (terms ban ML/AI use of content and bots; last year's gamelog scraper is not carried over), barttorvik.com and masseyratings.com direct scraping (robots disallow AI crawlers), PrizePicks/Underdog/DraftKings scraping.

**Why:** Verified 2026-09-10 (`docs/postmortem/06`). Last year had zero player-level data on disk; this source fills it with one download.

**Risk acknowledged:** Historical rosters exist only from 2024-25. Availability for earlier seasons is derived from player_box flags. Embedded ESPN market columns in pbp are a leak channel and are stripped from features.

---

## Decision 4: Lines and grading, accuracy-first (user decision 2026-09-10)

**Decision:** Free line sources only for now. Grade primarily on score and stat accuracy; the market scorecard runs wherever lines exist. Verified 2026-09-10: the CollegeBasketballData API accepts the existing CFBD key and serves lines from 2013 (modeled providers through 2022; ESPN BET from the 2022-23 season; ESPN BET, Bovada and DraftKings with open/close split for 2025-26). Together with the SBR lines on disk this gives three seasons of real-book closes for walk-forward market grading and one season with opens for the CLV check. The Odds API and paid odds archives are deferred; no paid data without asking the user again.

**Why:** User direction. Accuracy gates are the bottom-up proof anyway; ROI is the final check, not the first.

---

## Decision 5: Tech stack and layout (2026-09-10)

**Decision:** Python 3.12, per-project venv, `src/cbb_sim` small tested package plus flat prefix-named `scripts/`. numpy/pandas/pyarrow/duckdb for data, scikit-learn/xgboost/lightgbm/statsmodels/scipy for models, numpy + lookup tables + optional numba for the sim loop. Docs contract copied from cfb-props-sim (`docs/models/<m>/{model,features,experiments}.md`, change ledger, fixplan template, investigate board). Git from day one; bulk data on the private HF dataset `mvpeav/cbb-sim-data`.

**Why:** Proven on cfb-props-sim over five working weeks. The flat scripts layer with naming prefixes is what let that project move without a refactor tax.

---

## Decision 6: Phased build and walk-forward folds (2026-09-10)

**Decision:** Eight-week plan in `docs/FRAMEWORK_PLAN.md` section 7. Selection folds: train through 2022-23 / test 2023-24; train through 2023-24 / test 2024-25 (selection metric). 2025-26 with lines is sealed until selection is done. Validated game markets are the committed deliverable; player props are second; social publishing is out of scope.

**Why:** Season tips Nov 1-3, 2026. The CFB project's highest-value phase was a dedicated audit sprint on a working engine; it is budgeted explicitly as week 6.

---

## Decision 7: Pace is emergent; the L2 winner is a prior feature, not a sampler (2026-09-10)

**Decision:** The possession engine does not draw possessions per game. Possessions emerge from per-possession clock consumption (L5) across regulation and an explicit overtime model. The L2 bake-off winner (multiplicative as-of tempo formula, `docs/models/pace/`) is used as a pregame tempo prior feature and as the Control engine's pace model.

**Why:** All 32 L2 arms failed PIT calibration on the game-level count, for the same reason (overtime and fat tails). Feature richness and model class were inside noise (L14). A sampler that is uniformly miscalibrated cannot be a gate-passing component; an emergent count gets the overtime skew for free.

**Alternative considered:** Mixture or skewed families for the game-level count. Rejected: it models the symptom of a clock process the engine already simulates.

**Risk acknowledged:** G1 now depends on the clock-consumption model being right by state; L5 carries a by-month and by-terminal-event duration gate.

---

## Decision 8: Responsiveness gate must test slope, not only step monotonicity (2026-09-10)

**Decision:** The responsiveness gate in every bake-off is: (a) slope ratio of predicted-vs-actual across driver quintiles within [0.8, 1.2], AND (b) monotone in at least 3 of 4 quintile steps, with the 4-of-4 requirement dropped when the driver's realised quintile span is below 2 pp (the steps are then noise). Applies to every driver (shooter, team, defence). Scope clarification (2026-09-10, after the fg_make re-decision surfaced the ambiguity): a driver whose realised quintile span is below 2 pp is exempt from BOTH clauses, the band and the steps, because a slope ratio over a noise-sized span is itself noise; the gate is then evaluated on the remaining drivers only, and the exemption is recorded per driver in the experiments doc. This supersedes the steps-only wording used in the fg_make, usage, rebound, free_throw and possession_outcome pre-registrations.

**Why:** In the fg_make FGA_3 bake-off the steps-only gate selected the team-level baseline, whose shooter slope ratio is 0.0086 (0.31 pp predicted span against 35.9 pp realised), over LightGBM (slope 0.988, 92 noise floors better on log loss) because the tree dipped at one interior step on a defence driver with a 1.37 pp total span. A gate that admits a flat model violates the standing matchup-specific rule ("must slope with actuals, not sit flat at the mean"); the gate was mis-specified, not the model.

**How the amendment is applied honestly:** It is generic and prospective. For already-decided models it is re-applied mechanically and the effect recorded in each experiments.md: every previously adopted winner already had slope ratios inside [0.8, 1.2] (rebound 0.96/1.10; free throw 0.977; usage 4/4 with slopes near 1; possession outcome 0.97-1.02), so no earlier winner changes. FGA_3 changes from team_baseline to LightGBM under the corrected gate, and both readings stay on record.

**Alternative considered:** Leaving the steps-only rule and shipping the flat FGA_3 model. Rejected: it fails the standing rule and the G4 oracle-tercile check showed the too-narrow-spread signature sourced in exactly that decision.

**Risk acknowledged:** Amending a gate after seeing a result is the kind of act the bake-off rule exists to prevent. The mitigation is that the amendment is stated as a general rule, applied to all prior decisions with the effect reported, and this entry exists.

## Decision 9: Opponent adjustment and conference alignment are MANDATORY BAKE-OFF ARMS, not yet standing rules (user decision 2026-09-10; CLOSED 2026-09-30 against opponent adjustment, the conference flag and conference alignment as mandatory arms; refit CADENCE stays open; see the closing amendment)

**Decision:** (a) Every sub-model that consumes an as-of team or player rate must carry opponent-adjusted versions of that rate as bake-off arms (at minimum: raw-centred as the reference, one-pass opponent-mean adjustment, iterative adjustment to convergence). Adjustment is NOT adopted anywhere until it beats the raw-centred reference beyond the noise floor on possession-outcome round 3 AND the result is confirmed on at least one other sub-model's S1 confirmation pass. If it loses or ties, the raw-centred rate stays and this entry is amended to say so. The user's instruction on 2026-09-10 was explicit: the mechanism is logical, and logical things have been disproven by data in this project before; prove it. (b) A conference-game flag is a first-class feature in every scoring-stage model, audited like home/away/neutral. (c) The in-season refit scheme's cadence and alignment are bake-off dimensions: calendar-monthly (the S1 definition adopted in L21), weekly, and conference-aligned (each team refits at its own first conference game, then on the standing cadence). Possession outcome runs this first; the other sub-models' S1 confirmation passes inherit its winner.

**Why:** The first four to six weeks of the season are mostly non-conference games against opponents of very different quality from the conference schedule that follows. A team that has played Mississippi Valley State three times carries some of the best raw rates in the country into January, and a model without adjustment sees big numbers and predicts they continue against Duke. Monthly refit boundaries are not aligned to the conference start, so a refit on the first of the month can carry two to three weeks of non-conference-only fits into conference play. S1's calibration gain (L21) was measured with monthly cadence; weekly and aligned cadences were never arms, so the cadence rests on convenience, which the bake-off rule bans.

**How it is applied:** Diagnostic first: the round-2 S1 possession-outcome residuals re-bucketed by week relative to each team's first conference game, not by calendar week. Then a pre-registered possession-outcome round 3 crossing alignment arms with feature arms (conference flag, opponent-adjustment method). Every previously decided sub-model gets the winning feature bundle and scheme through its S1 confirmation pass (HANDOFF "QUEUED"), each re-gated, none reopened on model class.

**Alternative considered:** Keep monthly S1 and rely on own_ratings (already opponent-adjusted) to carry quality while style rates stay raw-centred. This is the F0/F1 reference in round 3; if the tree already extracts the schedule-strength correction from own_ratings, the adjusted arms will sit inside the floor and F0/F1 wins by the simplicity rule. That outcome is a legitimate result, not a failure.

**Risk acknowledged:** Per-team aligned refits multiply fitted objects; the trainer refits once per distinct boundary date and shares across teams. Opponent adjustment must be strictly as-of (only games before the date) or it becomes a leak; the leak harness runs on every adjusted feature.

**Amendment 2026-09-11 12:25 EDT (PM, after possession-outcome rounds 3-4 and the S1 confirmation passes; status still PENDING EVIDENCE, leaning against):** (a) Opponent adjustment: possession-outcome round 3 found one-pass and iterative adjustment and the conference flag inside the noise floor (best +0.000225 vs floor 0.000804); the raw-centred reference stands there. Not confirmed on any second sub-model. (b) Conference flag: same result, inside the floor. (c) Alignment and cadence: free throw adopted S1_conf_aligned on calibration (first-4-conference-weeks gap 0.94 vs 1.27 pp) and remains the only sub-model where alignment won. Possession-outcome round 3's marginal conference-aligned win on the continuation model DID NOT REPRODUCE in round 4 (identical cells, log loss to 1e-6, gain 0.237 pp under threshold on a decile-boundary sensitivity); the continuation scheme winner is back to S1_monthly. The two tree alignment cells on the first-shot model have now been NOT RUN in two rounds because each is 5-6 hours at the local thread cap; they are minutes on the AWS box and are the first job of the next box session. Rebound's early-conference calibration failure (open under every scheme) is the other place alignment could still earn its keep. Until the tree cells run, this decision cannot close either way; no sub-model adopts adjustment or alignment on the strength of this entry. Evidence: `docs/tests/possession_outcome_conference_regime_2026-09-10.md`, `docs/tests/possession_outcome_early_season_2026-09-11.md`, possession_outcome experiments.md sections 7-9, free_throw experiments.md section 8.

**Closing amendment 2026-09-30 (PM recommendation of 2026-09-18, user ruling 2026-09-30: "close decision 9"):** CLOSED against (a) opponent adjustment, (b) the conference-game flag and (c) conference ALIGNMENT as mandatory bake-off arms. The decision's own test was "beats the raw-centred reference beyond the noise floor on possession outcome AND confirms on a second sub-model"; it failed both halves. Possession outcome: adjustment and the flag inside the floor in round 3 (best +0.000225 vs floor 0.000804); the conference-aligned tree cell A1, finally run 2026-09-18, is 0.24 floors with every segment inside the retrain spread (PO experiments.md sections 16, 18, 19; `docs/tests/possession_outcome_alignment_cells_2026-09-18.md`). Second sub-models: the arms were null or negative in rebound round 3, shot_block round 1 and foul accrual round 6 (all 2026-09-18; under 0.1 floor in foul accrual), and the 2026-09-11 conference-aligned continuation win did not reproduce. The raw-centred rate is the standing reference; the alternative in this entry ("the tree already extracts the schedule-strength correction from own_ratings") is the outcome the data chose. What this does NOT change: (1) free throw keeps its adopted `S1_conf_aligned` scheme; that adoption rests on its own calibration evidence and is not reopened. (2) Refit CADENCE stays an open bake-off dimension: the weekly-cadence cell A2 was inside the floor on the primary (0.66 floors) but moved the weeks 0-3 gap 3.83 -> 2.51 pp, more than the 0.8 pp retrain spread, on one seed; it needs seed 1 and the `cont` cell before it is read either way. (3) Any of the three may still be entered as an OPTIONAL arm where a round's own evidence motivates it (the team-rate estimator round of 2026-09-30 registers opponent adjustment as optional arm E9); it is no longer required, and a round is not incomplete without it. (4) The leak rule for any adjusted feature is unchanged: strictly as-of, through the leak harness.

## Decision 10: Closed-loop gate for every engine-produced state feature (2026-09-10)

**Decision:** Any sub-model feature that the engine itself generates during a simulated game (score margin, team fouls, possession index, and any interaction of these with clock) is adopted only after a closed-loop check inside the engine: a paired-stream run with the feature live vs frozen at its pregame value must keep margin SD ratio, home/away score correlation, and possessions per game inside the G1/G2 tolerances. Offline log loss and calibration on real game states remain necessary but are no longer sufficient for state features.

**Why:** L23. fg_make's `score_diff` passed every offline gate and then tripled margin variance in the engine; the clock's `score_diff` produced the whole +4.24 possession miss. Real game states carry a stable confound (the leading team is the better team) that a sim breaks by construction.

**How it is applied:** Prospectively to every bake-off with state features. Retroactively to fg_make and clock, which get pre-registered round-2/round-3 arms that re-parametrise state; usage and rotation are audited for engine-produced inputs and gated the same way in their current rounds. The gate is run with the engine's `diag_engine_multilevel.py` ablation, 5 seeds minimum for the SD ratio, 200 for the final read.

**Alternative considered:** Drop `score_diff` everywhere. Rejected: it removes real end-game effects (12 points of total in the ablation) and is a deletion chosen because it is convenient, which the bake-off rule bans.

**Risk acknowledged:** A closed-loop gate depends on the rest of the engine being right; a feature can fail it because another sub-model is broken. Ablations are therefore run one sub-model at a time with all others frozen, and the attribution of a failure is written with the ablation table, never from the aggregate.

## Decision 11: Fixes that expose a hidden compensation ship as a SET, judged on the combined stack (PM, 2026-09-30)

**Decision:** When a pre-registered fix passes its own primary and fails only veto lines whose miss is owned by a different, named defect that the fix stops compensating for, the fix is NOT refused and NOT shipped alone: it is marked VALIDATED-PENDING-SHIP-ACTION in the ledger with the owning defect named, and it enters the next combined closed loop with the other pending fixes and the owner's candidate. The vetoes are applied to the combination, at 200 seeds with the multi-draw floor of SIM_GUARDRAILS revision 2026-09-30(c). No fix is ever shipped alone on the strength of this entry, and a combination that fails is not rescued by dropping the member that fails: the owner must be found.

**Why:** Three times on 2026-09-30 a mechanism repair that removed a compensation was vetoed by the line the compensation had been propping up: the joint foul rounds (the over-dispersed served foul constant was carrying total variance), the drawn block flag (more second chances expose the total-accuracy line and the TOV% denominator), and, on 09-18, the possession-outcome shrinkage arm. The served engine passes the total-bias line only because +2.0 possessions offsets a points-per-possession deficit (`g1_g5_possessions_corr_diagnostic_2026-09-30.md`). CLAUDE.md bans accepting a downstream stage that compensates for an upstream bias; the mirror is that removing compensation must not be punished one fix at a time, or the engine can never leave the compensated state.

**How it is applied:** Every such fix is registered default-off with exact parity, has its own primary passed, its vetoes read with the game bootstrap and a multi-draw seed floor, and the owner of each failed veto named in the ledger row. The PM schedules the combined loop; the combined loop's arms are pre-registered like any other.

**Alternative considered:** Waive the failing veto for the fix in isolation. Rejected: that is how a compensated engine ships; the veto is information about a defect elsewhere.

**Risk acknowledged:** "Pending ship action" can become a parking lot. Each pending fix carries the date and the owner; if the owner is not in a registered round within two weeks, the fix is re-read and either shipped on its own merits or refuted.

## Decision 12: Closed-loop noise floors are multi-draw and game-bootstrapped (PM, 2026-09-30)

**Decision:** A paired closed-loop line's noise floor is the larger of (a) the spread across at least four seed-offset reference runs and (b) the paired game bootstrap interval (resampling games with each game's paired seeds kept together). A 500-game x 25-seed loop decides sub-model primaries and mechanism lines; it does not decide G5 ratio or correlation lines, which need the 200-seed read.

**Why:** The single seed-offset draw used as the G5 floor since 09-18 understated the line's seed variation about fivefold (`shot_block_drawn_flag_2026-09-30.md`, supplementary served runs on seeds 2000-2024 and 3000-3024). Vetoes were being read as 5-14 "floors" on a line whose true 25-seed SD was 0.005.

**How it is applied:** graders report both floors on every veto line; the ledger row of every 25-seed veto read on 2026-09-18 and 2026-09-30 is marked PROVISIONAL until the 200-seed read.
