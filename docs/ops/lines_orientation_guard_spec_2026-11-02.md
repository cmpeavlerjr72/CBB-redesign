# Spec: lines orientation guard (neutral-site home/away mismatch). SPEC ONLY, no code today.

Ops worker, 2026-11-02. Rulings from the PM on 2026-11-02 are folded in. Earlier verdicts in `lines_signflip_2026-11-02.md` are superseded by its addendum.

## (a) Defect and evidence
CBBD's Draft Kings (DK) CURRENT `spread` is expressed from the book's home team; `spreadOpen`, `homeMoneyline`, `awayMoneyline` are expressed
from CBBD's home team. On neutral-site games the two designations can differ, so the current spread arrives with the wrong sign relative to
our slate's home.
- NIU (CBBD home) v Grambling (ESPN event 401920912, neutral, Rock Hill SC): CBBD DK spread +8.5, open -5.5, ML -238/+195 (NIU). ESPN DK current: NIU -8.5,
  ML NIU -375; Bovada (CBBD perspective) -8.5. Final Grambling 89-87.
- Merrimack (CBBD home) v App State (401921422, same site): CBBD DK +3.5, open -3.5. ESPN DK: Merrimack -3.5. Bovada -3.0. Final App State 63-49.
- Fetches were pregame (8 h and 11 h before tip, ML unchanged), so this is not a live-line problem.
- The other six sign-contradicting rows of 2026-11-02 are NOT defects: genuine pregame favourite changes confirmed by ESPN DK current; CBBD
  moneylines are the OPEN prices (stale), which is why a spread-vs-ML test gives false positives. No ML-based guard is to be built.
- Suggestive history: 2025-26 DK rows with sign-opposite current vs open and |open|>3: 7 of 17 neutral-site (neutral is 9% of DK rows).

## (b) Rule
For every slate game reconcile two home designations:
1. OUR slate/CBBD home: `slate.home_team_id` (from CBBD `/games` `homeTeamId`; `schedule_daily` rows with `source == "cbbd"`: `home_team`, `away_team`, keyed `game_id` = ESPN event id, `cbbd_game_id`).
2. The book's home: ESPN's `homeAway`. Today `schedule_daily` hoopR rows (`source == "hoopr"`, key `game_id`) carry NaN `home_team`/`away_team`, so the guard must read the home
   designation from hoopR directly (schedule/team box `home_id`/`team_home_away`; exact column names to be verified against the pulled hoopR schedule parquet before coding),
   or from the ESPN scoreboard `competitors[].homeAway` that `pull_espn_odds_snapshot_v1.py` already fetches (store it in the snapshot row: `espn_home_team_id`).
   Comparison is on team ids through the existing hoopR<->CBBD id map, never on names.
- Designations AGREE: nothing changes; flag `orient_rule = "agree"`.
- Designations DIFFER (neutral-site case): the CURRENT spread (and total unchanged) is taken as given in the book (ESPN) perspective and re-expressed in OUR slate's home perspective
  by negation (`spread_ours = -spread_book`), ONLY for DK `spread`; `spreadOpen` and both moneylines stay as delivered. Flag `orient_rule = "swap_current_spread"`.
- Designation unknown (no hoopR/ESPN row): leave as delivered, flag `orient_rule = "unknown"`; if the game is neutral and the current sign contradicts the open sign, set `ats_line_suspect`
  (see d) rather than guessing. Never negate silently; every swap is logged per game and counted (`n_orient_swap`, `n_orient_unknown`).
- Bovada's perspective is not assumed identical to DK's: Bovada is checked against the same reconciliation in (e) before the rule is applied to it. Today it was in CBBD perspective.

## (c) Columns (additive; `publish_id` hash columns unchanged)
`orient_rule` (agree / swap_current_spread / unknown), `book_home_team_id`, `spread_as_delivered`. Moneyline outputs additionally carry `ml_is_open_proxy = True` on the live path
(PM ruling 2: CBBD ML are open prices), so every ML edge/pnl/CLV is labelled open-moneyline based; a second odds source (ESPN DK) is a separate PM decision.

## (d) Grade-stage exclusion for stored publishes made before the guard ships
For any stored row where the designations differ AND the current spread sign contradicts the open spread sign: exclude from the ATS record and ATS CLV with reason
`spread_orientation_suspect`; totals and moneyline grade as stored. Needs only stored columns plus the designation lookup. Applies to NIU-Grambling and Merrimack-App State for 2026-11-02.
No hand-grading at ESPN's line (ESPN is not a pipeline source). Tonight's pass may grade them wrongly; the ledger is regradable.

## (e) Verification plan (2024-25 = fold 2; 2025-26 sealed, not touched)
Population: all DK rows in `lines_2025.parquet` (and ESPN BET rows as the control provider) with `neutralSite == True` plus all rows where hoopR and CBBD home differ.
Per row, before and after the rule, compute (i) orientation correctness against the close-vs-result relation: sign of (final margin + close spread) agreement rate and mean |margin + spread|
(a correctly oriented spread has lower error), (ii) agreement of the sign of the current spread with Bovada's and with the de-vigged-ML favourite where the ML is the close price, (iii) sign
disagreement rate with the open spread. Report the disagreement rate before and after, by neutral/non-neutral and by provider; label cells below 30 rows underpowered. Success: after the rule,
neutral-site disagreement falls to the non-neutral baseline; non-neutral rows are unchanged (bit-identical).
Honesty: the 2024-25 lines parquets carry no fetch time, so only close-vs-result tests apply; no sealed-season data is read.

## (f) Tests
- Unit, from the saved raw 2026-11-02 refetch (`results/diag/lines_signflip_2026-11-02/raw_cbbd_lines_refetch.json`, copy a 5-game subset to `tests/fixtures`): NIU and Merrimack swap; Bryant-Siena and ODU-WMU (genuine moves, designations agree) unchanged; one clean game unchanged.
- Designation unknown on a neutral game with sign contradiction: flagged suspect, spread unchanged.
- No ML-based flag exists: a pick'em and a stale-ML genuine move are never flagged.
- Grade: suspect row has `ats_result == "none"`, no ATS CLV, totals/ML graded; counts reported; `publish_id` bit-identical for rows with `orient_rule == "agree"`.
- Idempotence: guard applied twice equals once. Existing `tests/test_daily_chain_v3.py` and lines tests pass.

## (g) Files and deploy order
Touched: `src/cbb_sim/live/lines.py` (reconciliation + swap + flags + counters; stage dict `n_orient_swap`, `n_orient_unknown`), `scripts/grade_daily_v1.py` and `src/cbb_sim/live/daily.py` (settle/grade_tables/clv_frame exclusion),
`scripts/pull_espn_odds_snapshot_v1.py` (store ESPN home id, only if hoopR does not supply it), `scripts/grade_truth_regrade_v1.py` (regrade path applies (d)). Tests as in (f).
Order: (1) verification in (e) first; (2) ship the guard in lines.py with tests, outside the 20:00 ET window; (3) ship the grade exclusion; (4) regrade 2026-11-02 so NIU-Grambling and Merrimack-App State leave the ATS record. The
three/two wrong spreads stay in today's stored publish and site day untouched.
