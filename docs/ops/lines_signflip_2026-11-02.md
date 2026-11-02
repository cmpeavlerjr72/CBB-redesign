# DK current-spread sign flips, 2026-11-02 (ops worker, report only)

> SUPERSEDED VERDICT: the "Verdict" and "Rule" sections below (feed-inverted sign) were WRONG. See "Addendum: live-line test and corrected diagnosis" at the end. The sign-consistency guard was implemented, tested, then withdrawn (code reverted, nothing committed).

## Verdict
CBBD's defect, not ours. `pull_cbbd.flatten_lines` and `live/lines.py::normalise` pass `spread` through untouched (no home/away name
matching, no sign logic), and a fresh `/lines` refetch (raw JSON: `results/diag/lines_signflip_2026-11-02/raw_cbbd_lines_refetch.json`,
gitignored) returns the same signs. The payload has no `formattedSpread` field (keys: provider, spread, overUnder, homeMoneyline,
awayMoneyline, spreadOpen, overUnderOpen), so candidate (a) "follows formattedSpread" cannot be tested. (b) is excluded (no name match
exists). (c) genuine move is rejected: moneyline fetched at the same instant contradicts the sign, and Bovada for the same game disagrees.

## Rule (confidence: moderate)
Draft Kings' CURRENT `spread` is intermittently delivered with inverted sign (home-perspective convention broken); `spreadOpen` and
the moneylines are consistent. Not deterministic by team name. No trigger found beyond "DK current spread". Open spread: 0 of 117 bad.

## Today (117 DK rows, all line_kind live, fetched 2026-11-02 00:00:02Z)
Current spread contradicts the open sign AND the ML favourite: 7 games. Plus Boston University (-1.5 vs open +1.5, ML exactly 50/50):
undeterminable. Total candidates 8/117 (6.8%); spread_open bad 0/117.

| game | cur | open | home_ml/away_ml | p_home devig | other evidence | strength |
|---|---|---|---|---|---|---|
| Northern Illinois v Grambling | +8.5 | -5.5 | -238/+195 | 0.675 | Bovada cur -8.5 | high |
| Merrimack v App State | +3.5 | -3.5 | -185/+154 | 0.622 | Bovada cur -3.0 | high |
| Bryant v Siena | +2.5 | -2.5 | -155/+130 | 0.583 | none | high |
| Old Dominion v W Michigan | +1.5 | -1.5 | -130/+110 | 0.543 | none | medium |
| Coastal Carolina v E Michigan | -1.5 | +1.5 | +102/-122 | 0.474 | Bovada null | low |
| South Alabama v Ohio | -1.5 | +1.5 | -105/-115 | 0.489 | Bovada -1 (open -1) | low |
| James Madison v Miami (OH) | -1.5 | +1.5 | -105/-115 | 0.489 | Bovada null | low |
| Boston University | -1.5 | +1.5 | 50/50 | 0.500 | none | undeterminable |

Raw CBBD DK fields match stored values exactly for all rows (NIU: overUnder 138.5, overUnderOpen 147.5; Merrimack 133.5/134.5; Bryant
150.5/144.5). Two correct control games exist throughout the slate (corr(sim margin, -spread) 0.91 overall; flipping the 7 lowers it to
0.90, so the sim does not discriminate here, it is not evidence either way).

Note: the 4 small-spread cases could also be true line moves across pick'em; only the first three are strong.

## Historical (same test: ML-implied favourite vs spread sign, |p-0.5|>0.15 "clear" cells)
- `lines_2023..2025` ESPN BET (replay/backtest provider): clear-cell disagreement 0.00-0.03% on spread; 2023 consensus 2.0%,
  teamrankings 1.2% (not used by `REPLAY_PROVIDERS`). Replay path does NOT share the defect.
- `lines_2026` Draft Kings current spread: 4/742 clear cells (0.5%): Oregon-Gonzaga, UCF-FAU, Minnesota-Nebraska, SMU-Miami(OH). Bovada 0.04%,
  ESPN BET 1.9% (648 clear). DK spreadOpen: 0.00%. So the DK-current defect exists in 2025-26 data at low rate, but is far higher today (6.8%).
- `data/raw/lines_daily` today: DK open snapshots 5/51 bad (9.8%), Bovada 0/8.

## Files
- `results/diag/lines_signflip_2026-11-02/corrected_lines_2026-11-02.parquet` -- CANDIDATE sign-corrected spreads (column `spread_proposed`,
  NaN for BU), with confidence. Not applied anywhere. Stored publish/lines parquets untouched.

## Addendum: live-line test and corrected diagnosis (owner question 2026-11-02)

### Was the "current" spread an in-game or post-game line? No.
Timeline (UTC; fetch = DK `line_fetched_at` in each publish / `lines_snapshots`; tip = slate `tipoff_utc`, identical to ESPN `event.date`;
all tips are real, none is a placeholder in the final slate):

| game | tip | last fetches (cur spread) | final |
|---|---|---|---|
| NIU v Grambling (ESPN event 401920912, neutral, Rock Hill SC) | Nov 2 08:00 | Nov 1 00:00 -5.5; Nov 1 12:00 -5.5; Nov 2 00:01 +8.5 (8 h pre-tip) | Grambling 89-87, final before the Nov 2 12:00 fetch |
| Merrimack v App State (401921422, neutral, Rock Hill SC) | Nov 2 11:00 | Nov 1 -3.5, -3.5; Nov 2 00:01 +3.5 (11 h pre-tip) | App State 63-49 |
| Bryant-Siena, ODU-WMU, CCU-EMU, JMU-Miami OH, S Alabama-Ohio, BU-Northeastern | Nov 3 00:00-01:00 | flip between Nov 1 12:00 and Nov 2 00:01, unchanged at Nov 2 12:00 | pending |

The fetch that carries the flip precedes every tip by 8-25 h, so it is a pregame line. The 12:00Z Nov 2 morning publish (ee8a6e68a9, 116 games)
already excludes NIU and Merrimack (tipped). `game_date` 2026-11-02 and `tipoff_utc` match ESPN's date=20261102 events exactly (ESPN has
them on Nov 2 at 08:00Z/11:00Z = 3 am/6 am ET, an odd Rock Hill slot but not a UTC date shift of Nov 1 evening games and not a placeholder);
the date field is `game_date` / CBBD `startDate`, trustworthy here. No date-boundary bug found.
In-game moneyline would have moved; the CBBD moneylines are identical in all fetches of every flagged game.

### What actually happened (ESPN DraftKings core odds API, fetched 2026-11-02 morning ET, independent of CBBD)
1. **Six evening games: genuine pregame line moves, not feed errors.** ESPN's DK current spread equals CBBD's current for all six
   (e.g. Siena -2.5 at Bryant = Bryant +2.5; WMU -1.5 at ODU; BU/JMU/CCU/USA -1.5). The favourite changed between open and now.
2. **CBBD's DK `homeMoneyline`/`awayMoneyline` are the OPEN prices, not the current ones** (ESPN current: Bryant +120 / Siena -142 vs
   CBBD -155/+130 = ESPN open; ODU -102 / WMU -118 vs CBBD -130/+110 = open; unchanged across all our fetches). The moneyline I used as
   "truth" in the first analysis was stale, so the sign-vs-ML test produced false positives on real moves.
3. **Two neutral-site games (NIU-Grambling, Merrimack-App State): home/away mismatch.** ESPN lists Grambling and App State as HOME
   (neutral site); CBBD lists NIU and Merrimack as home. ESPN DK current is NIU -8.5 (ML NIU -375) and Merrimack -3.5 (ML -170), matching Bovada
   (NIU -8.5, ML -340). CBBD's DK current spread (+8.5, +3.5) is in the ESPN-home perspective (Grambling +8.5) while CBBD's own home is NIU, so the
   CBBD-perspective spread should be -8.5 / -3.5. CBBD's spreadOpen and ML open are in CBBD-home perspective. So the defect is a perspective mismatch
   between CBBD's home label and the book's home label on NEUTRAL-site games, i.e. rows where CBBD and ESPN disagree on home. Corroboration (not
   proof): 2025-26 DK sign-opposite cases with |open|>3 are 7 of 17 on neutral sites (neutral share of all DK 2026 rows: 9%); Oregon-Gonzaga
   (Moda Center) and SMU-Miami OH (UD Arena) are neutral. No fetch timestamp exists in lines_2026.parquet, so a tip-time test is not possible there.
   Graded consequence today: NIU -8.5 lost to Grambling (87-89) and Merrimack -3.5 lost (49-63) at the true lines; the stored +8.5/+3.5 would
   grade the OTHER side. These two are the only confirmed wrong-sign rows; the other six are right as stored.

### Guard status: WITHDRAWN
The PM-ruled guard (spread sign vs open spread and a clear de-vigged ML favourite) was implemented in lines.py / daily.py with 6 passing tests, then
reverted before commit: it would have replaced six GENUINE current lines with the open line (stale ML makes every real favourite flip look like a
defect), and it cannot separate those from the two neutral-site cases. Nothing was changed, committed or re-published. A correct guard needs the
home/away definition used by the book (ESPN event `homeAway`) and a neutral-site flag: for neutral games where CBBD home != ESPN home, interpret DK current in the
ESPN-home perspective. That is a join on a second source plus a rule for the unknown provider perspective, and needs PM spec and a bake-off style
verification across the 2025-26 neutral-site rows before code.
