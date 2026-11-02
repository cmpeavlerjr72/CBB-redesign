# DK current-spread sign flips, 2026-11-02 (ops worker, report only)

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
