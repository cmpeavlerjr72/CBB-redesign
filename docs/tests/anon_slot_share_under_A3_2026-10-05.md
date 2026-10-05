# Anonymous-slot share under A3 day-1 seeding (sizing, 2026-10-05)

Question: the FT round found the anonymous-slot block feeds a newcomer FT prior (~0.645) to rotation slots whose real shooters are mostly returners (~0.695), about -1.9 pp of opening-window FT%. How much remains once A3 names returners and incoming transfers? Sizing only; no new model, no re-simulation. Inputs: existing 50-seed window runs in `results/player_day1/runs/` (first 14 days; F2 = 2024-25, 598 graded games; F1 = 2023-24, 643), verified finals from `truth/team_game_shots_v2`. No 2025-26 data read.

Method: `players.parquet` holds named players only (anonymous slots are absent), so anonymous share of FTA = 1 - (named FTA / team FTA from `games.parquet`). Anonymous-slot FT% is backed out from team FT% and named FT% (ftm = pts - 2*(rim+jump2) - 3*fgm3). ANON arm = every slot anonymous.

| fold | state | anon share of sim FTA | sim FT% | actual FT% | gap (pp) | named-slot FT% | implied anon-slot FT% |
|---|---|---|---|---|---|---|---|
| F2 | served history | 42.0% | 67.0 | 70.7 | -3.7 | 68.9 | 64.5 |
| F2 | anonymous | 100% | 64.6 | 70.7 | -6.1 | - | 64.6 |
| F2 | A3 | 12.4% | 67.7 | 70.7 | -3.0 | 68.2 | 63.6 |
| F1 | served history | 43.7% | 67.4 | 70.2 | -2.8 | 69.6 | 64.6 |
| F1 | anonymous | 100% | 64.8 | 70.2 | -5.4 | - | 64.8 |
| F1 | A3 | 7.9% | 68.3 | 70.2 | -1.9 | 68.6 | 63.7 |

Named-slot FT% under A3 (68.2 / 68.6) is itself 2 pp below actual: A3's named players are returners and transfers at ~0.69, so the remaining gap is mostly the named-shooter / thin-sample block (FT section 20), not the anonymous block.

Remaining anonymous-slot contribution under A3: if those slots shot a returner-like 0.695 instead of 0.636-0.637 (gap 5.8-5.9 pp), team FT% rises 0.73 pp (F2) and 0.46 pp (F1); at 37.5 FTA per game that is about 0.27 and 0.17 points of total per game, against A3 window total bias of -5.27 (F2). That is about 3-5% of the total bias. It is an upper bound: the anonymous slots under A3 include true freshmen and non-D1 newcomers, whose 0.64 is legitimate. Under served history the same lever is about 2.0-2.1 pp (0.8 pts/game), matching the ~1.9 pp finding. A3 removes roughly 65-80% of it.

## 2027 opening-day anonymous minutes (`data/raw/cbbd/rosters/roster_2027.parquet`, ESPN, 296 of 365 teams)

Per-player 2025-26 minutes were not read, so the 2027 share is not computed from a minutes prior. Evidence available:
- Roster headcount: 93.5% of 4,261 players carry a CBBD id; 6.5% (278) are new to D1; 299 are transfers in (7.0%); 1,173 are class FR (206 of them new_to_d1 and unmapped; the rest map to a CBBD id but a freshman has no prior-season minutes).
- Analog: under A3 the named-player minute share was 89.4% (F2) and 92.7% (F1) of 200 team minutes, i.e. anonymous minutes 10.6% / 7.3%. Those folds had full rosters and the same freshman/newcomer mix. Estimate for the 296 rostered teams: about 7-11% of minutes anonymous (about 9%).
- The 69 teams with no roster (fetch errors, `season.year 2026 != 2027`) are 100% anonymous (A3 hard-stops without a roster; they fall to the no-history block).
- Team-weighted opening-day anonymous minutes: (296 x ~0.09 + 69 x 1.0) / 365 = about 26%, of which the roster-less teams are 19 points. Closing the 69-team roster gap is worth about 4x more anonymous minutes than the newcomer prior on rostered teams.

## Read

Under A3 on rostered teams the anonymous-slot prior is worth at most ~0.2-0.3 pts/game of total bias (inside the reseed floor of the window total bias scale only marginally) and 0.5-0.7 pp of FT%. A modeling round is not justified before Nov 2; the first-order 2027 risks are the 69 missing rosters and the named-shooter FT level (-2 pp).
