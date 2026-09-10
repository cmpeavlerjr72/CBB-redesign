# CBB Clean-Sheet Framework Plan

Version 0 (draft for review), 2026-09-10. Author: Fable 5.1 (PM). Evidence: `docs/postmortem/01..06`.

Target: a possession-level Monte Carlo engine for NCAA men's basketball that prices game markets (spread, total, moneyline, team totals, halves) and player props from one simulation, validated bottom-up with hard gates, every method chosen by blind bake-off. Season tips Nov 1-3, 2026. That is 7.5 weeks from today.

---

## 0. What last year's engine was, and why it lost

CBB-Monte was a team-level box-score draw, not a possession simulator. Six independent GLMs (3PA, 2PA, FTA counts; 3P%, 2P%, FT%) produced attempts and percentages per team, makes were binomial, and points were arithmetic. The two teams were drawn independently. Nothing was simulated below the box-score line. That is also why it ran so much faster than the CFB play-by-play engine.

Measured over 5,229 graded games of the 2025-26 season (`02_season_scorecard.md`):

| Metric | Model | Closing line |
|---|---:|---:|
| Spread MAE vs actual | 9.74 | 9.10 |
| Total MAE vs actual | 16.47 | 13.36 |
| Total bias (pred minus actual) | +9.44 | -0.48 |
| Win-prob Brier | 0.201 | 0.180 (de-vigged) |
| ATS following the model, all games | 51.3% | breakeven 52.4% |
| O/U following the model, all games | 48.7% (Over picked 96% of the time) | |

After regressing actual outcomes on both the closing line and the model line, the model's coefficient is indistinguishable from zero (t = 0.86 on margin, t = -0.74 on total). Patching the two level biases flat still leaves ATS at 51.0% and O/U at 49.2%. There was no residual signal to salvage, which is the data reason for a clean sheet rather than a repair.

Root causes, ranked by measured impact (`01_engine_review.md`):

1. **Raw KenPom levels on a drifting scale.** League-mean AdjO went 100 (Nov 2021) to 109.3 (Apr 2026) and moves ~6 points within every season. Every feature entered as a raw level with a positive coefficient, so all six sub-models inflated together. This is the totals bias.
2. **No home-court advantage anywhere.** The home/away/neutral column was dropped at the merge step (`merge_kp_and_enrich_gamelogs.py:21`) although it was populated in the schedule file. Home margin projected +2.2 vs actual +5.3. The model took the home side ATS 16.6% of the time.
3. **Full independence between the two teams and no shared pace.** Sim SD of margin equalled sim SD of total (24.6 vs 24.7), the signature of independent draws. Realised residual SDs were 12.4 and 17.9. The sim was twice too wide on margin, which is why win probabilities were underconfident at both tails.
4. **Dispersion was read off feature columns, not fitted.** The observed game-to-game SD of shooting percentage (which is almost entirely binomial noise) was injected as true-talent variance and then binomial noise was added again on top.
5. **Models were the 60% training slice ending Jan 2024 and were never refit.** They priced all of 2025-26.
6. **No player layer at all.** A player box-score scraper was written but never run. Injuries changed nothing.
7. **Process failures.** Five engine forks never reconciled; the two "improved" variants were generated after the games were played (file timestamps prove it) so their better numbers are unusable; no season-long pick record exists; the tournament fork added a post-hoc calibration curve on win probabilities; the daily pipeline crashed 82 times on one unretried scraper and died in June.

Worth keeping as ideas, not code: the shift(1) pregame feature builder; the point-in-time KenPom snapshot store (weekly 2022-25, daily 2026, all on disk); train-time medians persisted with each model plus a preflight feature contract; the counts x percentages decomposition; the shared-possession draw sketched in the unshipped "improved" branch; the per-game artifact layout; the Brier/ROI/CLV evaluation harness, which diagnosed the model correctly all season.

---

## 1. Standing rules (all carried into `CLAUDE.md` on repo creation)

1. **PM delegation.** Fable 5.1 plans, specifies, reviews, decides. Opus/Sonnet/Haiku workers do the reading, coding, and analysis.
2. **Bake-off rule.** No model class, distribution family, sampling scheme, feature set, or anchor rating is adopted without a pre-registered comparison of multiple candidates on walk-forward holdouts. The spec (candidates, metric, holdout, noise floor, decision rule) is written and committed before the experiment runs.
3. **No hand tuning on output.** No multipliers, caps, clips, calibration curves, or blends on sim output. A failed gate means a sub-model is wrong; find it and fix it there. Exact wording: `cfb-props-sim/docs/SIM_GUARDRAILS.md` core principle and section 5.
4. **Bottom-up the cascade.** Build and gate the smallest unit first (possession outcome, shot make, rebound, foul), then aggregates. Never accept a downstream stage that compensates for a known upstream bias.
5. **Matchup-specific, not league-average.** Every sub-model must show responsiveness: predictions bucketed by team or player prior quintile must slope with actuals.
6. **Multi-level evidence.** Every finding and fix shows overall, per-game, per-team, per-possession-type, and per-player evidence. Underpowered cells are labelled underpowered.
7. **Profitability frame.** Every model evaluation carries a market scorecard: calibration vs de-vigged market, ROI by edge bucket at real odds, and the CLV cross-check. An edge that beats the close but cannot predict line movement is presumed leaked.
8. **Walk-forward only.** Season-level folds for selection; the 2025-26 season with its lines is the blind final test and is not opened until a sub-model has been chosen on 2023-24 and 2024-25.
9. **Every backtest row must satisfy created_at < tipoff**, enforced in code. No retroactive generation.
10. **One engine.** Variants live behind flags over a shared core. No forks.
11. **Append-only experiment docs**, never-deleted investigate board, closed items require a PROOF block.
12. **Git from day one**, commit after every meaningful change, bulk data on a private HF dataset.

---

## 2. Target architecture (a hypothesis to be proven, not a decision)

The engine simulates a game possession by possession. On each possession the players on the floor, the offense and defense strengths, and the game state (score, time, period, fouls, bonus) feed a cascade of sub-models that produce a terminal event and clock consumption. Scores, box lines, and every market fall out of the same trajectory, so correlations between team totals, player stats, and game outcome emerge instead of being modelled.

Why this shape: it is the only one that produces player props and game markets from a single engine with emergent correlation (CFB Decision 1), and it is the only one where "bottom-up, prove every stage" has stages to prove. A box-score draw has no bottom.

The bake-off that keeps this honest: a **Control engine** is built in week 1. It is last year's decomposition done right (rates per possession, features centered on snapshot league means, home court, one shared pace draw per sim, fitted dispersion, refit on all data). It is cheap, and it is the yardstick. The possession engine must match or beat Control on every game-level gate before any prop work is built on it. If it cannot, that is a defect in a sub-model to be found, not a reason to skip the gate.

### 2.1 Cascade, in build order

| Layer | Sub-model | Output | Candidate methods to bake off |
|---|---|---|---|
| L0 | Data, IDs, reference tables | Clean pbp, player box, team box, schedules 2021-22 to 2025-26; one ID crosswalk keyed on ESPN team_id and athlete_id; empirical gate targets | n/a (audit only) |
| L1 | Team strength anchor | Offense and defense efficiency per possession, tempo, both point-in-time | (a) KenPom centered on its snapshot's league mean; (b) own ridge ratings from hoopR box data with home flag and recency; (c) blend. Each leak-tested with the change-form correlation test. |
| L2 | Pace | Possessions per game, one shared draw per simulated game | Multiplicative tempo formula; ridge on tempo features; GBM. Family: Poisson vs NegBin vs Normal, chosen on held-out dispersion. |
| L3 | Possession outcome | Terminal event: turnover, rim/2pt FGA, 3pt FGA, shooting foul, non-shooting foul in bonus, end of period; then make/miss by shot type; then OREB continuation; and-1 | Multinomial GBM vs nested binary cascade vs hierarchical logit; shot-make models with player and defense terms; OREB model |
| L4 | Player layer | Who is on the floor (rotation/minutes), who takes the shot (usage), player-specific make rates, rebound/assist/steal/block attribution | Rotation from pbp substitution events; hierarchical Dirichlet usage allocation (from cfb-props-sim `usage_alloc.py`); empirical-Bayes shrinkage of player rates toward role priors; keyed on what the variance decomposition says (coach vs team vs player) |
| L5 | Game state and clock | Seconds per possession distribution; end-of-half; end-game behaviour (fouling, deliberate fouls); overtime; garbage-time rotation | Empirical resampling by state vs parametric; explicit end-game policy model gated by data |
| L6 | Markets and grading | Spread/total/ML/team totals/1H; player props; scorecard; CLV leak detector; bias monitor | Ported from cfb-props-sim graders |

### 2.2 Non-negotiables baked into the design from the evidence

- Every rating feature is expressed relative to its own snapshot's league mean.
- Home/away/neutral is a first-class feature in every scoring-stage model, and each model's feature list is audited for it (CFB INV-50 lesson).
- One pace realisation per simulated game, both teams scaled by it.
- Dispersion comes from the model's own variance function, validated against realised residual SD. Targets come from the data: last year's realised residual SD was 12.4 on margin and 17.9 on total.
- Rates per possession, not counts, so shot mix sums to the possession count.
- RNG seeded on (seed, game_id, family) so paired bake-off arms share aligned streams.
- Ties are resolved by an overtime model, not discarded.
- Models refit on all available data before deployment, on a schedule, with the chronological split used for selection only.

---

## 3. Bake-off protocol

Every choice follows the same procedure, using the `docs/models/<model>/{model,features,experiments}.md` contract and the `fixplan` template from cfb-props-sim (`05_cfb_methodology_extract.md` sections 1 and 3).

1. Spec first. Candidates, feature bundles, holdout folds, primary metric, segment breakdowns, noise floor (a spec-identical retrain under another seed), and the decision rule are written to `experiments.md` and committed before any run.
2. Folds. Fold 1: train through 2022-23, test 2023-24. Fold 2: train through 2023-24, test 2024-25. Fold 2 is the selection metric. 2025-26 with lines is sealed until selection is done.
3. Blind evaluation. One grading script scores all arms on the holdout and writes the table. The PM reads the table, not the arms.
4. A winner must beat the noise floor. Ties go to the simpler model.
5. Sim-level confirm. An offline winner ships only after a paired-seed sim run shows the relevant gates did not regress (CFB's "INT hybrid" lesson: three offline wins, three sim-level regressions, never shipped).
6. Results are appended, never edited. The change ledger records status with the CFB vocabulary (SHIPPED / TESTING / REFUTED / CONFIRMED-INCUMBENT / ...).

---

## 4. Validation gates (CBB analog of CFB Gates 1-8)

Targets are computed from 2021-22 through 2024-25 real data in week 1 and stored in `data/processed/reference/`. They are not guessed. Every gate is evaluated overall, by team tier, and by game (and by player where relevant). Any failed gate triggers sub-model debugging, never an output patch.

| Gate | Quantity | Level |
|---|---|---|
| G1 | Possessions per game: mean and SD | overall, by tempo tier |
| G2 | Points per possession | by offense tier, by defense tier |
| G3 | Shot mix: rim/2pt, 3PA, FTA shares per possession | by team |
| G4 | Four factors: eFG%, TOV%, OREB%, FT rate | by team, sim vs real |
| G5 | Residual SD of margin and of total; correlation of the two team scores | overall, by spread band |
| G6 | Home-court margin at home vs neutral | overall, by conference tier |
| G7 | Overtime rate and tie handling; period-level scoring split | overall |
| G8 | Player minutes distribution (starters vs bench), usage shares, per-player points/reb/ast distribution tails | by player role |
| G9 | Spread and total calibration vs actual; win-prob calibration by decile | overall, by month |
| G10 | Market scorecard: MAE vs close, Brier vs de-vigged, ROI by edge bucket, CLV sign agreement | by market |

Meta-gates: created_at < tipoff on every backtest row; a seed-count study fixes the minimum seeds before any ROI is read; every external feature passes the leak test (|change-form corr| <= 0.15 against own-week margin); grading truth (final scores, OT accumulation) is verified against a second source before any grade is trusted (CFB INV-44).

---

## 5. Data plan

| Need | Source | Status | Notes |
|---|---|---|---|
| Play-by-play 2021-22 to 2025-26 | sportsdataverse/hoopR-mbb-data (CC BY 4.0) | verified, free | 1.3M to 2.9M rows/season; player ids, shot coords, clock; embedded ESPN spread/win-prob columns must be stripped from features |
| Player box with minutes | same | verified | 2024-25: 6,290 games, 207k player-rows |
| Team box, schedules, shots | same | verified | schedules already include 2026-27 |
| Rosters / game rosters | hoopR (2025-26 on); CBBD `/teams/roster`, `/lineups`, `/substitutions`, `/plays` onFloor | verified | CBBD covers lineups historically; point-in-time-ness of CBBD `/ratings/adjusted` unverified, treat as leak until checked |
| Point-in-time team ratings | KenPom snapshots on disk (weekly 2022-25, daily 2026); own ridge ratings | on disk / to build | own ratings are the compliant fallback and a bake-off arm in their own right |
| Closing lines 2025-26 | SBR scrape on disk (130 dates) + ESPN pickcenter | on disk | the blind final test |
| Closing lines 2013-2026 | CollegeBasketballData API `/lines` (existing CFBD key works) | verified 2026-09-10 | modeled providers 2013-2022; ESPN BET from 2022-23; opens populated 2025-26 |
| Player prop lines | The Odds API ($30-59/mo; props from May 2023) | deferred by user 2026-09-10 | props graded on stat accuracy until then |
| Injuries, in-season | Covers.com, ESPN injuries endpoint | to build | unstructured; manual override CSV as in CFB |
| Retired | Sports-Reference (ToS bans ML use), Torvik and Massey direct scraping (robots disallow AI crawlers), PrizePicks/Underdog/DK scraping | do not use | last year's SR gamelog scraper is retired |

Week-1 data audit items, learned from CFB: OT accumulation in finals, neutral-site and early-season tournament labelling, season_type ambiguity around conference tournaments, frozen or stale fields, and the vendor's rating-date convention (rating after vs entering a date).

---

## 6. Repo skeleton and process

Copied from cfb-props-sim with names locked before any doc is written into them:

```
CBB-clean-sheet/
  CLAUDE.md                 standing rules, reading order, git practice
  PROJECT_STATUS.md         current state; HANDOFF.md session log (one canonical pointer)
  ARCHITECTURE_DECISIONS.md decision / why / alternative rejected
  docs/SIM_GUARDRAILS.md    the gates above, numbered
  docs/LEARNINGS.md         findings ledger, each entry ends "For the sim:"
  docs/models/<m>/{model,features,experiments}.md + README index + change_ledger.md
  docs/tests/               dated gate runs and experiments, append-only
  docs/plans/               fixplan_<id>.md, pre-registered
  docs/postmortem/          this review (01-06)
  src/cbb_sim/              small tested package: game_state, engine, usage_alloc, lookups, variance, clients
  scripts/                  flat, prefix-named: pull_ build_ train_ run_ exp_ diag_ grade_ leak_ chain_
  data/raw (HF-synced) / data/processed (git) / results (HF)
  tests/
```

Engineering rules from the CFB experience: lookup tables and vectorized NumPy in the sim loop, never live model calls; thread counts pinned in any container; `PYTHONIOENCODING=utf-8`; rule-era flags live in GameState, not in sub-models; verdict-vs-shipped-artifact consistency check in the retrain logger.

Compute: 20 local cores. A possession engine at ~140 possessions per game will cost roughly what the CFB engine costs per game, so the 196-core EC2 box and its launch chain, Dockerfile, and parity-digest gate are reused for full-season seed sweeps.

---

## 7. Timeline (7.5 weeks, Sep 10 to Nov 3)

The CFB project's most valuable phase was a 2.5-week audit sprint on an already-working engine. That phase is budgeted explicitly here rather than assumed.

| Week | Dates | Deliverable | Exit gate |
|---|---|---|---|
| 1 | Sep 10-17 | Repo skeleton and docs; hoopR ingest 2021-2026; ID crosswalk; data audit; empirical reference tables; leak-test harness; **Control engine** | Control passes G1, G5, G6, G9 on 2024-25; reference tables committed |
| 2 | Sep 17-24 | L1 anchor bake-off; L2 pace bake-off; variance decomposition (coach / team / player) on four factors, tempo, usage | Winners chosen on fold 2 with noise floor; prior keys decided from the decomposition |
| 3-4 | Sep 24 - Oct 8 | L3 possession cascade bake-offs; L5 clock and game state; engine v0 producing scores | G1-G7 pass on 2024-25; engine v0 matches or beats Control |
| 5 | Oct 8-15 | L4 player layer: rotation/minutes, usage allocation, player make rates | G8 passes; game gates do not regress |
| 6 | Oct 15-22 | Audit sprint: full walk-forward 2024-25, then unseal 2025-26 with lines; market scorecard; seed-noise study; fix defects at model level | G9-G10 reported honestly; leak detector clean |
| 7 | Oct 22-29 | Daily ops: schedule pull, ratings refresh, roster/injury overrides, sim run, artifact publish, grading, bias monitor, CLV log; cloud sweep if needed | Dry run on the exhibition slate end to end |
| 8 | Oct 29 - Nov 3 | Paper-trade opening week; props layer graded vs Odds API if purchased; go/no-go | First live-week grading report |

Delegation: each row becomes worker specs written by the PM. Typical split: Opus for engine core and cascade models, Sonnet for data pipeline, graders, and bake-off runners, Haiku for lookups and summaries. The PM reviews every bake-off table and every gate report before anything is marked SHIPPED.

Risk to the timeline: the L3 cascade is the largest unknown. If week 4 ends without G1-G7 passing, week 5 slips and props become the stretch goal, exactly as live trading was for CFB. Game markets validated honestly are the committed deliverable.

---

## 8. Decisions needed before week 1 starts

1. Historical odds 2021-2025: buy the $119 CSV, or re-verify the free sportsbookreviewsonline archive first? (Recommendation: probe hoopR's embedded spread for two days, then buy if it is not usable.)
2. The Odds API subscription (~$30-59/month) for player prop lines and backtest history. Without it, props ship without a market-graded backtest. (Recommendation: buy at the $30 tier in week 5.)
3. KenPom: last year scraped the public front page. Own ridge ratings from hoopR box data are a bake-off arm and the compliant fallback. Confirm you are comfortable keeping the KenPom snapshot store as the other arm.
4. Git remote name and HF dataset name for this project.
5. Scope confirmation: game markets first, props second, publishing/social layer out of scope for the 8 weeks.
