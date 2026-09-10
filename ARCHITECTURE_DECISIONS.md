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
