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
