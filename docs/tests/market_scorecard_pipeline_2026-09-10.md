# Market scorecard v2 -- pipeline description and provisional proof run (2026-09-10)

Built by `scripts/grade_market_games_v2.py` (`scripts/grade_market_games.py`, v1, is
untouched -- CLAUDE.md's scripts convention: graders are versioned filenames,
never overwritten). This doc describes what the pipeline does, the guards
it enforces, and the results of running it once, end to end, on the smallest
available engine run to prove the plumbing before any ROI number is trusted.

## Why a v2

v1 reads market lines from `data/raw/cbbd/lines_{season}.parquet` -- the raw
per-season CBBD pull, joined with a hardcoded DraftKings > ESPN BET > Bovada >
consensus provider preference. That pull was never itself validated: no
coverage check, no de-vig sanity check, no home/away flip check, no check
against a second source. `docs/tests/lines_cbbd_validation_2026-09-10.md`
(+ addendum) did that validation on 2023-2025 and it passed; the PM accepted
CBBD ESPN BET as the free lines source the same day
(`docs/models/change_ledger.md`, "CBBD betting lines ... ACCEPTED"), and its
output is `data/processed/lines/lines_close_v1.parquet` -- de-vigged
(proportional primary, power alongside), one row per (game, provider), with
both `close_*` and (2025 only) `open_*` columns. v1 never learned about that
work; it still reads the unvalidated raw pull. v2 reads the accepted table.

Separately, `docs/tests/engine_seed_count_2026-09-10.md` (run the same day)
found that v1's own 5-seed report
(`docs/tests/market_games_engine_v0_F2_2025_s5_r2event_2026-09-10.md`) prints
ROI, Brier and calibration numbers that its own section 4 says "must be
struck, not interpreted" -- a 5-seed sim win probability can only take the
values {0, 0.2, 0.4, 0.6, 0.8, 1.0}, and the per-game margin-mean seed noise
(14.85 pts at 5 seeds) swamps the ~8-16 pt MAE the report reads as if it were
signal. v1 has no seed-count guard at all. v2 does.

## What the pipeline does

1. **Seed-count / honesty guard.** Reads `n_seeds` from `run_meta.json`.
   Below 200 seeds (`docs/tests/engine_seed_count_2026-09-10.md`'s floor for
   even the slate-level gate report) every table in the output is labelled
   PROVISIONAL. Below ~2,000 seeds (the same doc's floor for a per-game
   ROI/Brier/calibration read) the report REFUSES to print any ROI, Brier,
   calibration or edge-bucket-hit-rate number -- it prints the seed count and
   the threshold instead. Also checks `created_at < tipoff` per row when the
   results contract carries row-level timestamps; today's engine_v0 contract
   does not (only a run-level `created_at`), so the report says
   "created_at not recorded" and states the run's `backtest` flag rather than
   silently skipping the check.

2. **Truth.** Schedule metadata (date, month, neutral site, team ids) comes
   from `games_universe` via `cbb_sim.eval.reference.load_actual_games`;
   scores come from `data/processed/truth/game_finals_v2.parquet` ONLY --
   never the engine's own totals, never a pbp accumulation, never
   `games_universe`'s own score columns. A game flagged by `game_finals_v2`
   as an unresolved cross-source disagreement (nonzero score diff never
   checked against a third source) is excluded from the truth set and
   counted, not silently kept or silently dropped.

3. **Lines.** `data/processed/lines/lines_close_v1.parquet`, provider
   `ESPN BET` (the only book present in every target season and the one
   `lines_close_v1` was built primary around). Proportional de-vig
   (`p_home_close_prop`) is used for every probability comparison; power
   de-vig (`p_home_close_power`) is reported alongside for comparison only.
   Settlement is always at the real posted spread/total (-110, since CBBD
   carries no per-side spread/total price) or the real posted moneyline --
   never at a de-vigged price.

4. **Sim vs market (point estimates).** Model-vs-actual and market-vs-actual
   MAE/bias (context, matching v1's framing) plus a direct model-vs-market
   MAE/bias/correlation on margin and total, with the seed-count study's SE
   of the sim mean attached so the reader can judge how much of any gap is
   seed noise rather than engine behaviour.

5. **Probability calibration (gated with ROI/Brier).** Sim P(home) and the
   de-vigged market P(home), each calibrated against realised outcome in 20
   buckets; a reliability table of (sim prob - market prob) deciles against
   (realised - market prob); Brier for both. Refused below ~2,000 seeds.

6. **Edge buckets (gated with ROI/Brier).** Spread and total, bucketed by
   `|sim - market|` into (0,1] (1,2] (2,3] (3,5] (5,inf) points; moneyline
   bucketed by `|sim P(home) - market P(home)|` into (0,2] (2,4] (4,6] (6,inf)
   percentage points. Each cell reports n, hit rate at the real settlement
   price, a Wilson 95% CI on the hit rate, the seed-count study's SE of the
   sim mean, and an UNDERPOWERED label below 100 games. Reported overall and
   broken down by season, by month, by conference-vs-non-conference (CBBD
   `games_{season}.parquet`'s `conferenceGame` flag), and by quintile of a new
   grading-only `own_rating` (as-of `off_c - def_c` from
   `data/processed/ratings/own_ratings_{season}.parquet`, home+away average,
   joined leak-free via `own_ratings.join_as_of` on each game's own date).
   Refused below ~2,000 seeds.

7. **Leak cross-check (2025 only).** `lines_close_v1`'s `open_spread_home` is
   usable only in 2025 (0% coverage in 2023-2024, 33.5% in 2025, per the
   validation doc's addendum). Where opens exist: correlation of
   (sim margin - open margin) against (close margin - open margin), plus a
   2x2 of edge sign vs movement sign among lines that actually moved, plus
   the CLV sign-agreement rate -- CLAUDE.md: "an edge that beats the close
   but cannot predict line movement is presumed leaked." This is a different
   quantity from v1's `surprise_corr`/`clv_agreement` (which compare against
   the CLOSE, not the open); both are legitimate, non-overlapping checks.

## Proof run: `results/engine_v0/F2_2025_s5_r2event` (5 seeds, season 2025)

Command: `.venv/Scripts/python.exe scripts/grade_market_games_v2.py --results
results/engine_v0/F2_2025_s5_r2event --season 2025`. Output:
`docs/tests/market_games_v2_engine_v0_F2_2025_s5_r2event_2026-09-10.md` (+
`.json`).

- **Guard fired correctly on both thresholds.** 5 seeds < 200 -> whole report
  labelled PROVISIONAL. 5 seeds < 2,000 -> sections 3 (probability
  calibration) and 4 (edge buckets) print only the seed count and the
  threshold, no numbers.
- **Truth accounting**: 5,710 schedule games for 2025, 0 missing a
  `game_finals_v2` row, 0 flagged unresolved -- all 5,710 enter the truth set
  (`finals_source`: 5,709 hoopR, 1 CBBD, matching `game_finals_v2`'s own
  resolution notes). 5,383 of those have a usable ESPN BET close line, the
  same count v1's report reads on this season.
- **created_at**: labelled "not recorded" -- `run_meta.json` has only a
  run-level `created_at` (2026-09-10T23:13:38Z), `backtest=true`, so the
  per-row inequality is structurally impossible and is not mechanically
  checked, matching `contract.validate_tipoff_safety`'s own behaviour.
- **Section 2 (not gated -- point estimates, not ROI/Brier)** printed: model
  margin MAE vs actual 16.03 pts (market 8.74), model total MAE vs actual
  14.49 pts (market 12.73); direct model-vs-market margin MAE 13.43,
  correlation 0.329; total MAE 6.65, correlation 0.669. Seed-noise SE of the
  sim mean at 5 seeds: margin 14.85 pts, total 6.43 pts -- i.e. the margin MAE
  above is smaller than one seed-noise SE, exactly the "must be struck, not
  interpreted" situation the seed-count study warned about, stated plainly
  here rather than left implicit.
- **Section 5 (leak cross-check, not gated by the ROI/Brier refusal but
  labelled PROVISIONAL)**: 1,802 2025 games have an open, 1,086 moved.
  corr(edge at open, open-to-close movement) = 0.0674; CLV sign agreement
  0.5405 on the 1,086 moved lines -- matching v1's `clv_agreement` number on
  this same close-vs-open game set (v1 computes the analogous statistic
  against the close-side disagreement, a different definition; the two
  agreeing to four decimals here is a coincidence of this particular data,
  not a redundant computation).
- **Sections 3/4 (gated)**: correctly refused; no ROI, hit-rate, calibration
  or Brier number appears anywhere in the shipped report.

## What was NOT validated end-to-end with real data

The gated code path (calibration, edge buckets with Wilson CIs and
seed-noise SEs, the season/month/conference/own-rating-quintile breakdowns)
was exercised once by temporarily zeroing the two seed thresholds in a
throwaway interpreter session against this same 5-seed run, to catch bugs
before shipping -- it produced well-formed tables (e.g. 5,383 spread-edge
rows across 4 breakdown dimensions, correct UNDERPOWERED labelling below
n=100, correct handling of a 0-row bucket) with no exceptions. That run was
not saved or adopted as a report; no ROI or Brier number from it has been
read or should be trusted. The pipeline has not yet been run against a
>=2,000-seed engine result, because none exists yet (a 2,000-seed full-season
run is ~56 core-hours per `docs/tests/engine_seed_count_2026-09-10.md`
section 3). That run is the next prerequisite before any ROI or Brier number
from this scorecard is adopted.
