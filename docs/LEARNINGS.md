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
