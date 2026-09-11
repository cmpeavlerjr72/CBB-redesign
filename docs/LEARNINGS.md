# Learnings ledger

Empirical findings that reshape modelling decisions. Each entry ends with "For the sim:". Append-only. Evidence lives in `docs/tests/` or `docs/postmortem/`.

---

## L1. Last year's engine had no residual signal (2026-09-10)

Regressing actual margin on [closing line, model line] over 5,229 games: model coefficient 0.036 (t = 0.86). Same on totals: -0.030 (t = -0.74). Level-bias patches leave ATS at 51.0% and O/U at 49.2%. Evidence: `docs/postmortem/01_engine_review.md` section 0.

For the sim: rebuild, do not recalibrate. Nothing from the old model's fitted objects is reused.

## L2. KenPom's league mean drifts, within and across seasons (2026-09-10)

League-mean AdjO: 100.0 (Nov 2021) to 103.1 (Mar 2022); 104.5 (Oct 2025) to 109.3 (Apr 2026). AdjT 71-72 in November to 67 by March every season. Raw levels fed to six GLMs inflated all of them. Evidence: `01_engine_review.md` section 3.2, verified by the PM against the snapshot files.

For the sim: every rating feature is centred on its own snapshot's league mean; tempo enters as a ratio to the snapshot mean.

## L3. Independence between the two teams shows up as SD(margin) == SD(total) (2026-09-10)

Sim SD 24.6 on margin and 24.7 on total; realised residual SDs 12.4 and 17.9. Evidence: `01_engine_review.md` section 1.4.

For the sim: one pace realisation per game, shared by both teams; G5 checks the score correlation directly.

## L4. Scoring is trending up through free-throw rate and three-point volume (2026-09-10)

Points per team per game 70.4 (2021-22) to 75.1 (2025-26). FTA/FGA 0.305 to 0.352. 3PA share 37.9% to 39.6%. Possessions flat at 68.4-68.9. Evidence: `docs/tests/data_audit_hoopr_2026-09-10.md` section 6.

For the sim: foul and shot-selection models carry season-aware features or are refit each preseason; gate targets are per season; pooled-season training without a season term will under-shoot the current year.

## L5. Pace is about 3 possessions faster in November than in February (2026-09-10)

Every season: November 70.1-71.0, January 67.4-68.2, February 67.2-67.7. Evidence: same audit, section 6 by-month table.

For the sim: the pace model needs a season-progress term or a recency structure; last year's engine got this backwards by letting the rating-scale drift masquerade as tempo change. G1 is checked by month.

## L6. hoopR pbp lacks lineups and its event vocabulary drifts; CBBD pbp carries on-floor players every season (2026-09-10)

hoopR: 24 distinct event types across five seasons, only 19-20 in any one; Substitution events only from 2024-25; shot coordinates 6-26% before 2024-25. CBBD `/plays/date` returns whole days untruncated with a ten-player `onFloor` list on every row. Evidence: `docs/tests/data_audit_hoopr_2026-09-10.md` section 4; `docs/tests/data_audit_cbbd_2026-09-10.md`; bulk-endpoint probe 2026-09-10.

For the sim: CBBD pbp is the event and lineup source for L3-L4; hoopR supplies schedules, box scores, and the canonical ESPN ids. Event mappings are explicit tables with a test that fails on unknown types.

## L7. CBBD season ratings are end-of-season snapshots (2026-09-10)

`/ratings/adjusted`, `/ratings/srs`, `/ratings/elo` return one row per team-season with no date field. Per-game `homeTeamEloStart/End` inside `/games` is point-in-time. Evidence: `docs/tests/data_audit_cbbd_2026-09-10.md`.

For the sim: the season endpoints are banned as pregame features. The per-game Elo start values are a permitted, leak-testable feature.

## L8. Historical line coverage (2026-09-10)

CBBD lines: modelled providers only through 2021-22; ESPN BET from 2022-23; open/close split populated only in 2025-26 (93-94% of games). hoopR pbp's embedded spread is frozen at 2.5 from 2023-24 on and is not a line. Evidence: `docs/tests/data_audit_cbbd_2026-09-10.md`, hoopR audit section 8.

For the sim: real-book market grading is possible on 2022-23, 2023-24, 2024-25 (close only) and 2025-26 (open and close, sealed). CLV checks are 2025-26 only.

## L9. Our own as-of ridge ratings tie centred KenPom within seed noise (2026-09-10)

Control engine, fold 2 (2024-25, 5,700 games, 200 seeds): margin MAE own 9.146, KenPom 9.075, both 9.059; seed noise floor 0.05-0.08. Paired SEs put KenPom ~0.07 points ahead (t = 2.4), real but tiny. Leak test on own ratings: change-form corr 0.05 / -0.06 / 0.00. Evidence: `docs/tests/control_engine_F2_2026-09-10.md`, `docs/models/control_engine/experiments.md`.

For the sim: the compliant, self-contained anchor is good enough to be the default; KenPom stays as a bake-off arm for the possession engine. A market-independent rating built from box scores alone is not the bottleneck.

## L10. Independent attempt-count draws are last year's independence bug one level down (2026-09-10)

The four per-100-possession counts are individually calibrated (model SD vs actual within 5-10%), but their residuals correlate -0.56 (3PA/2PA), -0.36 (2PA/TOV), -0.29 (3PA/FTA) because they compete for the same possessions. Drawing them independently gives team-points SD 13.9 vs actual 9.2, margin SD ratio 1.68, home/away score correlation 0.08 vs 0.23, and PIT K-S p = 2e-79. Same evidence file.

For the sim: the Control's point estimates are usable, its intervals are not. Shot mix must be drawn per possession as one categorical outcome (turnover / 3PA / 2PA / foul), never as independent counts. This is why L3 is a possession-outcome model and not four count models.

## L11. Centred features carry no season level; totals drift 3-4 points a season (2026-09-10)

Fold-2 train mean total 141.8 vs test 145.5. With every feature centred on its snapshot mean, the engine inherited 77% of the drift as a -2.8 point total bias (the close was -0.6 on the same games). The Control's spec deliberately had no season term, so this is a measured cost, not a surprise (L4).

For the sim: outcome models need a season-level term or a recency-weighted refit each preseason, chosen by bake-off; never a post-hoc offset.

## L12. The yardstick: Control trails the close by 0.36 (margin) and 0.54 (total) MAE with no ATS edge (2026-09-10)

2024-25 vs ESPN BET closes: margin MAE 9.106 vs 8.746; total 13.218 vs 12.680; ATS 49.5-50.3% in every disagreement bucket; Brier 0.188 vs 0.175 de-vigged; leak screen clean (surprise corr ~0, CLV agreement 0.53).

For the sim: this is the bar. The possession engine must beat it on G9 and G10 before player props are built on it.

## L13. On-floor lineups exist only from 2023-24 (2026-09-10)

CBBD pbp `onFloor` completeness: 0% in 2022 and 2023 (empty at the source), 90% in 2024, 98% in 2025, 97% in 2026. hoopR Substitution events start in 2025. Evidence: `docs/tests/data_audit_cbbd_pbp_2026-09-10.md`.

For the sim: team-level possession models (L3) train on all five seasons; lineup/rotation/usage models (L4) train on 2024 and 2025 only (F1 = train 2024 test 2025; within-season walk-forward inside 2025 as the second check), 2026 sealed. Sample size for lineup priors is two seasons, so shrinkage strength is a fitted parameter, never assumed.

## L14. Game-level possession counts are not Gaussian; pace should be emergent (2026-09-10)

L2 bake-off, 32 arms on F2: every arm fails PIT (p < 0.05). Overtime games (5.6%) add +8.2 possessions on average, and regulation-only residuals still carry excess kurtosis ~1.8. RMSE spread across all feature sets and model classes is 0.06 (4.82-4.88), inside the 0.157 noise floor; the zero-parameter multiplicative formula (tempo_A x tempo_B x league mean, all as-of) is as good as LightGBM with state features. Responsiveness: tempo-sum slopes correctly. Evidence: `docs/models/pace/experiments.md` R1-R8.

For the sim: no game-level pace model is calibrated enough to draw possessions from. In the possession engine, possessions per game are emergent from per-possession clock consumption over 40 minutes plus the overtime model, which is where the skew comes from naturally. The multiplicative formula is kept as a pregame tempo prior (a feature for the clock-consumption model and the Control engine), not as a sampler. G1 is evaluated on the emergent count.

## L15. CBB's variance decomposition does NOT reproduce CFB's "team-beyond-coach ~= 0%" -- team identity is a real term, and player identity dwarfs everything (2026-09-10)

Sequential nested R^2 decomposition (`src/cbb_sim/analysis/variance.py`, ported from cfb-props-sim), coach_id -> (team_id, season) beyond coach -> opponent-coach beyond both -> residual, on 43,930 D-I non-truncated team-games, 2022-2025, weighted by pbp-derived possessions:

| Metric | Coach (seq) | Team-beyond-coach (seq) | Opponent-coach (seq) | Residual |
|---|---:|---:|---:|---:|
| Tempo (possessions) | 13.9% | 5.9% | 12.6% | 67.6% |
| 3PA share of FGA | 21.5% | 8.6% | 10.8% | 59.1% |
| Rim share of FGA | 11.9% | 7.3% | 9.2% | 71.5% |
| FTA/FGA | 6.4% | 4.6% | 8.3% | 80.7% |
| TOV% (TOV/poss) | 8.8% | 5.8% | 10.1% | 75.3% |
| OREB% | 14.2% | 5.8% | 5.7% | 74.3% |
| eFG% | 8.1% | 3.8% | 7.2% | 80.8% |

Team-beyond-coach is 3.6-8.6% on every metric (defensive sides too, `docs/tests/variance_decomp_2026-09-10.md`), never zero, and stable in sign across a 2022-23-only and 2024-25-only split. Opponent-coach is often as large as or larger than own-coach (FTA/FGA, TOV%) -- these carry a real matchup/defensive-scheme component, not just an own-team tendency.

Player-game level (coach_id -> team-season beyond coach -> athlete_id beyond both -> residual, 340,801-346,559 player-games, players with >=10 minutes, weighted by minutes) shows player identity dominating everything else on every metric, and exceeding CFB's largest-ever measured single-player share (QB red-zone TD, 34%):

| Metric | Coach (seq) | Team-beyond-coach (seq) | Player-beyond-both (seq) | Residual |
|---|---:|---:|---:|---:|
| Usage proxy ((FGA+0.44FTA+TOV)/min) | 1.0% | 0.3% | 34.1% | 64.7% |
| 3PA share of own FGA | 2.9% | 1.2% | 51.7% | 44.2% |
| FT rate (FTA/FGA) | 0.8% | 0.6% | 9.8% | 88.8% |
| Assist rate (AST/min) | 1.6% | 0.7% | 28.9% | 68.9% |
| Rebound rate (REB/min) | 1.0% | 0.5% | 35.9% | 62.7% |
| Points per minute | 1.1% | 0.6% | 19.5% | 78.8% |

Restricted to the 2,484 athletes who changed team_id across seasons in 2022-2025 (163,879 player-games, the transfer natural experiment that separates player identity from team identity the way CFB used coach moves and QB transfers): player-beyond-both stays the largest non-residual term on every metric but runs 5-12 points lower than pooled (e.g. 3PA share 51.7% -> 40.9%, usage proxy 34.1% -> 23.4%) -- a real effect that attenuates in a transfer's first season, not a with-team artifact. Evidence: `docs/tests/variance_decomp_2026-09-10.md`.

For the sim: CFB's rule ("team identity beyond the coach is zero, key tendencies on (coach, season) alone") does not transfer to CBB and must not be copied. Per metric: (1) team-game tendencies (tempo, 3PA share, rim share, OREB%, eFG%) key primarily on (coach, season) but carry a (team_id, season)-level adjustment on top -- CBB's transfer-portal churn makes team identity a real, non-trivial term CFB never had to model; (2) FTA/FGA and TOV% key on the (own coach, opponent coach) pair jointly, since the opponent's scheme explains as much or more than the team's own coach; (3) every player-game rate stat (usage, shot mix, assist rate, rebound rate, scoring rate) keys on athlete_id first, full stop -- coach and team-season are nearly irrelevant (1-3% and 0.3-1.2%) next to a 10-52% player-beyond-both share, confirming the postmortem's prediction that a basketball player's share of team output would exceed CFB's largest QB effect; (4) a transferring player's first-season prior must shrink harder toward a career/position baseline than a continuing player's, same shape as CFB's HC/QB continuity-weighted prior (`docs/postmortem/05_cfb_methodology_extract.md` section 5) but re-derived, not copied, since CBB's attenuation-under-transfer is measured directly here rather than assumed; (5) residual is 60-89% everywhere in Study 1 and 44-89% in Study 2, so none of these become deterministic lookups -- every metric still needs a possession- or player-level stochastic draw on top of whatever prior it's keyed on.

## L16. ESPN's 2025 feed mislabels putbacks as jump shots; fixed at the event layer by shot location (2026-09-10)

In 2024-25 only, "JumpShot" attempts following an offensive rebound sit a median 0.55 ft from the basket (9 ft in every other season), snapped to the same canned coordinate as TipShot rows; 50% within 1 ft vs < 1% elsewhere. hoopR's independent pbp carries the identical label, so the defect is upstream at ESPN. Box-score reconciliation of 3PA/2PA/FTA/OREB/TOV shows no 2025-specific degradation, so the defect is confined to the rim/jumper split. Evidence: `docs/tests/shot_classification_diag_2026-09-10.md`.

For the sim: rim vs jumper is classified with a location override whose threshold is derived from the release-distance distribution of dunks/layups (round 2 of L3). Team style features for first chances are built from first-chance events or box scores only, so continuation labels cannot contaminate them. This is a data fix at the source of the error, the opposite of an output patch.

## L17. Rebounding is a team-level sub-model; dead balls are a fixed share (2026-09-10)

F2 winner: LightGBM with miss type and game state (log loss 0.6456, worst decile gap 1.96 pp, responsive 4/4). Adding the on-floor five's individual rebound rates gains 0.88-1.11 noise floors on the 2024->2025 fold, a straddle, not a win; fitted individual-rate shrinkage 50 pseudo-opportunities. Dead-ball rebounds (< 1% of misses) as a fixed per-miss-type share cost 0.49 floors vs a third class. OREB% drifted 28.0% -> 29.6% across 2022-2025 and every arm carries that level miss (L11, third sub-model). Evidence: `docs/models/rebound/experiments.md`.

For the sim: rebounds are drawn at team level given miss type and state; per-player as-of rates are used for attribution in the player layer only. Dead balls are a deterministic share. Season-level drift is handled by the training scheme chosen in L3 round 2, not by a term added here.

## L18. Free-throw make probability is a shooter-identity model; no bonus-rule era boundary in the data (2026-09-10)

F2 winner: LightGBM on shooter as-of features (log loss 0.5753, gap 1.27 pp) beating empirical-Bayes shrinkage by 3.6 floors; the team-level arm fails by 6.8 pp, almost all shape (it cannot separate a 90% shooter from a 55% one). Fitted shrinkage prior = position group, m = 30 pseudo-attempts, on both folds; the position prior beats the prior-season prior on transfers (L15). Bonus thresholds re-derived per season land at the 7th and 10th team foul in every season 2022-2026; the pre-registered 2024-25 rule change is not in the data. FT attempts per game 34.8 -> 40.8 with the one-and-one share flat at ~28%. Rule violations 2.4-2.7% of trips (ESPN never logging the FGA). Evidence: `docs/models/free_throw/experiments.md`.

For the sim: trip structure is deterministic by rule from GameState's foul counts (thresholds read from `bonus_era.json`, one pair for all seasons); make probability per attempt keyed on the shooter's CBBD id with the ESPN id attached where it resolves.

## L19. Shot allocation: conditioning on the five on the floor removes CFB's "too narrow" defect; every allocator under-concentrates the top (2026-09-10)

Usage bake-off, train 2024 test 2025: LightGBM over the five wins all five event classes (rim by 16.9 floors, FT-trip the only eligible arm; jump2/3 straddle and TOV reverses on the within-2025 fold, recorded UNCONFIRMED). Fitted Dirichlet concentrations come back at "no dispersion" on every class: game-to-game usage variance is already supplied by lineup changes, so the CFB hierarchical Dirichlet is not adopted. Per-player count SD ratios 1.00-1.08 (CFB saw 0.54-0.89). The one gate that bites is top-1/top-3 usage share: all 50 cells negative, the sim spreads events too evenly across the five. Fitted shrinkage in on-floor events orders the classes by player-specificity (3PA 25, rim/jump2 50, TOV/FT 200), reproducing L15. Evidence: `docs/models/usage/experiments.md`.

For the sim: allocation is a per-class model over the actual five; no extra dispersion layer. The under-concentration of top usage is the open defect to attack next (a star's share within a lineup is compressed), and it will show up in G8 top-1 share, not in game markets.

## L20. Horn-truncated possessions must be right-censored, or the clock model manufactures possessions (2026-09-10)

Clock rounds 1-2: no arm passes the emergent G1; round 2's finer end-of-period state improved CRPS by 9-45 floors and made the emergent count worse. Cause: 50.5% of possessions starting with under 10 s left consume all of it, but only 0.2% are flagged censored, so the conditional law near the horn is truncation, not behaviour. A calibrated model draws short, the engine gets time back, draws shorter again, and subdivides the end of each half into possessions never played. Engine v0 at 6,000 sims shows the symptom exactly: possessions +4.0, PPP -0.07, every per-possession rate within 0.5 pp. Also: the end-of-half grading truth was contaminated by feed truncation (only 44.6% of halves' last logged possession ends at 0:00; on clock-complete halves the true share is 0.956 and gamma/hazard already sit inside the floor). Evidence: `docs/models/clock/experiments.md` sections 6-7, `docs/models/engine/model.md`.

For the sim: round 3 models the duration the offence intended, with every horn-ending possession right-censored, and the engine truncates at the horn; a CBBD clock-completeness flag goes into games_universe before the end-of-half gate is re-read. Do not touch any downstream sub-model until G1 is re-read with this fix.

## L21. In-season walk-forward refit (S1) is a calibration fix, not a log-loss fix; it is the default training scheme (2026-09-10)

Possession-outcome round 2, F2: LightGBM first-chance goes from a 2.78 pp worst decile gap (FAIL) to 0.98 pp (PASS) under monthly in-season refit for a log-loss gain of only 0.0014; the cascade on continuation chances 2.21 -> 1.86 pp. Exponential recency weighting (S2) fits half-lives of 365-730 days and is inside the noise floor everywhere; down-weighting old seasons is not the same operation as seeing the current one. Separable causes: the rim override (L16) collapsed continuation miscalibration from 7-12 pp to ~2 pp; first-chance-only style features moved ridge from 2.06 FAIL to 1.94 PASS under a static fit; S1 carried the tree across. Noise floor for the tree arm is PARTIAL (linear bootstrap SE applied; verdicts not at risk). Evidence: `docs/models/possession_outcome/experiments.md` section 5.

For the sim: every sub-model refits monthly in-season on all prior seasons plus the season to date (strictly before the refit date); bake-offs must gate on calibration, never on log loss alone. Switch `DEFAULT_POSSESSION_VERSION` to v2 and `DEFAULT_STYLE_SOURCE` to first-chance together on resume, after the usage shooter-label fix.

## L22. A defensive "missing column -> 0" fallback turned a merge defect into an all-NaN feature and an interrupted bake-off; fallbacks that manufacture data are banned (2026-09-10)

Attribution's `build_team_asof` appended six identically named column groups per season into one list and merged them sequentially; pandas suffixed the duplicates `_x`/`_y`, the bare names never existed, and an `if c not in tg.columns: tg[c] = 0.0` fallback filled every team count with zeros, so every league as-of rate was 0/0 = NaN and all three binary targets crashed. The five choice targets that did finish in the interrupted run had trained on the zeroed team features without any error, which is the worse outcome: a silent wrong feature passes and a loud crash does not. Fixed by merging within season column-wise and concatenating seasons row-wise; rebuilt rates match the reported base rates (assisted league rate 0.512). Evidence: `docs/tests/attribution_team_asof_lg_defect_2026-09-10.md`, `docs/models/attribution/experiments.md` section 3.1.

For the sim: a feature builder never fills a missing column with a constant; it raises. Every as-of table build asserts that no feature column is constant across teams, and every trainer asserts no NaN in X before fitting so the failure is attributed to the builder, not the model. Attribution's within-2025 fold also shows the `assisted` binary miscalibrated by 4-7 pp on 141k rows under a static 2024 fit; attribution was pre-registered before S1 became the default scheme (L21) and is the next candidate for an S1 rerun.

## L23. A state feature that is a consequence of the modelled outcome closes a feedback loop in simulation that offline scoring cannot see; `score_diff` in fg_make triples margin variance (2026-09-10)

Engine v0 first gradeable run (5,710 games x 5 seeds, round-2 event winners, paired streams): margin SD 34.6 vs actual 14.6, home/away score correlation -0.64 vs actual +0.25, possessions +4.24. A per-sub-model ablation of the `score_diff` state feature localises both: with fg_make's `score_diff` frozen, margin SD falls to 11.4 and the correlation to +0.42 (event and clock ablations do not move it); with the clock's `score_diff` frozen, possessions land at 67.84 vs actual 67.88 while the other two ablations leave the count at +3.6 to +3.7. Offline, `score_diff` is a legitimate predictor: a team that leads is on average the better team, and the fg_make bake-off scored C_plus_state on real game states where that confound is stable. In a closed loop the sim draws a lead by chance, the lead raises the make probability, the make grows the lead. Nothing was tuned; deleting the feature is not the fix either (it removes a real garbage-time and late-game effect and costs 12 points of total). Evidence: `docs/tests/engine_v0_F2_2026-09-10.md` addendum, `docs/tests/gates_engine_v0_F2_2025_s5_r2event_2026-09-10.md`, `results/engine_v0/F2_2025_s5_r2event/`.

For the sim: every sub-model that consumes a game-state feature which the engine itself produces (`score_diff`, fouls, possession count, time remaining interacting with margin) must pass a closed-loop gate before adoption: a paired-stream engine run with the feature live vs frozen must not move margin SD, home/away correlation or possession count outside the gate tolerance. State features are re-parametrised so the causal part (garbage time, end-game fouling, intentional slow-down) enters through variables the sim cannot inflate, and the team-strength confound is carried by the pregame ratings only. Round-2 bake-offs for fg_make and clock own this; the clock's censoring fix (L20) and its `score_diff` are two separate hypotheses for G1 and both run as arms.

## L24. Reconcile like with like: the event layer excludes technical free throws on purpose, and the grader compared it to a box that includes them (2026-09-10)

Truth tables v1 showed event-layer FTA/FTM disagreeing with the box on 7-10% of team-games in every season, flat while FGA agreement rose 80% to 97%; the suspect was `ft_trip_ambiguous`. That flag is refuted as a cause: it mislabels trip TYPE on ~41% of attempts but is uncorrelated with the count disagreement (r between -0.01 and -0.06). The cause is the possessions layer's deliberate exclusion of technical free throws (correct for FT-2: coach-chosen shooter, no possession) while hoopR's box and final score include them; netting technicals out explains 94.3% of 3,887 disagreeing team-games, is one-sided (event layer below box on 95-99%), and collapses the apparent FT-rate-quintile slope to zero, so that slope was a volume artefact and not a matchup defect. Flagrants and lane violations are not observable in CBBD's 24-value play vocabulary. Third source on the four flagged 2025 finals: hoopR right on 3, CBBD right on 1, and the side-flip was hoopR's, not CBBD's. Evidence: `docs/tests/ft_trip_reconciliation_2026-09-10.md`, `data/processed/truth/diag_finals_resolution_2025.json`.

For the sim: every deliberate exclusion in the event layer is mirrored in the grader (truth tables carry technical FTA/FTM as their own columns and the reconciliation adds them back before flagging), and the engine needs a technical-foul free-throw rule, sized at 0.14-0.22 points per team-game, one-directional low, listed in `docs/models/free_throw/model.md` section 9. No possessions-layer fix and no free-throw rerun.
