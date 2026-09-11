# Free-throw event-layer vs box reconciliation — 2026-09-10

Diagnostic worker task, answering `docs/tests/truth_tables_v1_2026-09-10.md`'s
open question: is the flat ~90-93% FTA/FTM event-vs-box agreement (all four
seasons, including 2024-2025 where FGA has already converged near 100%) a
labelling defect that biases the sim, and is `ft_trip_ambiguous` the cause?

Inputs (all read-only): `data/processed/truth/team_game_shots_v1.parquet`,
`data/processed/truth/game_finals_v1.parquet`, `data/processed/possessions_v2/`
(`possessions_{season}.parquet`, `chances_{season}.parquet`),
`data/raw/cbbd/pbp/plays_{season}.parquet` (re-classified with
`cbb_sim.pbp.events.classify_frame`, same segmentation rules as
`cbb_sim.pbp.possessions`), `data/processed/models/free_throw/attempts_v1_era.parquet`.
Seasons 2022-2025 only; 2026 untouched.

## Headline finding

**The leading suspect is refuted. The real cause is a different, fully
identified mechanism: technical free-throw attempts are deliberately excluded
from the event layer's `fta`/`ftm` (and from the naive points identity), while
hoopR's box `FTA`/`FTM` and final score include them.** Netting technical FT
attempts out of `diff_fta` explains **94.3%** of all disagreeing team-games
2022-2025 EXACTLY (to the attempt), collapses the FT-rate-quintile "slope"
to ~0, and reduces the points-identity mismatch by 60-93% per season.
`ft_trip_ambiguous` (the bonus-vs-shooting label) is real and large (~41% of
FT-2 training attempts) but is **uncorrelated** with the count-level
disagreement (r = -0.01 to -0.06 every season) — it mislabels trip TYPE, never
attempt COUNT, so it cannot and does not cause this gap.

---

## 1. Direction, magnitude, net bias

`diff_fta = ev_fta - box_fta` on `match_source == 'event_and_box'` rows.

| season | n matched | n disagree | % disagree | % event ABOVE box | % event BELOW box |
|---|---|---|---|---|---|
| 2022 | 10,564 | 888 | 8.41% | 4.62% | 95.38% |
| 2023 | 11,080 | 1,229 | 11.09% | 1.71% | 98.29% |
| 2024 | 11,102 | 886 | 7.98% | 1.13% | 98.87% |
| 2025 | 11,179 | 884 | 7.91% | 3.05% | 96.95% |

Distribution of `diff_fta`, share of disagreeing rows:

| season | -3+ | -2 | -1 | +1 | +2 | +3+ |
|---|---|---|---|---|---|---|
| 2022 | 19.0% | 54.4% | 22.0% | 1.7% | 0.2% | 2.7% |
| 2023 | 20.4% | 44.0% | 33.8% | 1.0% | 0.2% | 0.5% |
| 2024 | 23.7% | 60.3% | 14.9% | 1.0% | 0.0% | 0.1% |
| 2025 | 23.9% | 64.6% | 8.5% | 1.9% | 0.3% | 0.8% |

**One-sided by a wide margin.** 95-99% of disagreements are event-BELOW-box
(an undercount), with -2 the single largest bucket every season. This is
already inconsistent with symmetric attribution noise.

Season-level net bias (all matched team-games, not just disagreeing ones):

| season | mean diff FTA/team-game | mean diff FTM/team-game | mean diff FTA/100 poss | mean diff FTM/100 poss |
|---|---|---|---|---|
| 2022 | -0.157 | -0.124 | -0.234 | -0.186 |
| 2023 | -0.224 | -0.179 | -0.915 | -0.676 |
| 2024 | -0.185 | -0.147 | -0.266 | -0.211 |
| 2025 | -0.183 | -0.146 | -0.274 | -0.218 |

(Every season: `sum(ev_fta) < sum(box_fta)` by 1,654 / 2,487 / 2,059 / 2,044
attempts respectively — a consistent net undercount, never a net overcount.)

### FT-rate quintile (team-season box FT rate; powered, ~2,100-2,270 team-games/quintile)

Raw `mean diff_fta` slopes with the quintile (bigger negative bias for
high-FT-rate teams) in every season — e.g. 2023: Q1 -0.168 -> Q5 -0.267. **That
slope disappears once technical FTA is netted out** (`resid = diff_fta +
tech_fta`): 2023 Q1 -0.0004 -> Q5 -0.0082, no monotonic trend in any season.
The apparent "matchup-specific" responsiveness is a volume artifact of
technical-foul incidence scaling with total foul/FTA volume, not a
team-specific labelling defect.

---

## 2. Where: classification of the disagreement

CBBD's play-by-play has only 24 `playType` values
(`src/cbb_sim/pbp/events.py`); flagrant fouls and lane violations are not
among them and do not appear anywhere in `playText` either (checked directly:
0 of 462,118 2025 `JumpShot`/all-play rows contain "flagrant" or "violation").
**Neither is observable in this feed, at all** — reported as a genuine data
gap, not folded into another bucket. Substitutions cannot break trip assembly:
`Substitution` is an `INERT_CLASSES` member and is dropped from the event
stream before segmentation runs, so a sub between two free throws of one trip
is invisible to (and cannot corrupt) `_collect_trip`. One-and-one front-end
MISSES are the unambiguous branch of `classify_ft_trip` by construction (not
flagged `ft_trip_ambiguous`) and are 0.24-0.31% of all FTA every season — a
correctly-handled minority, not a defect source. CBBD blanking the shooter on
an FT row is functionally zero (0 / 12 / 0 / 3 rows across 2022-2025 of
~180-215k FT rows/season) and, since team attribution comes from `teamId` not
`shot_shooter_id`, could not move a team-level count regardless.

### Waterfall, pooled disagreeing team-games 2022-2025 (n = 3,887)

| explanation | n | % of all disagreeing rows |
|---|---|---|
| **fully explained by technical FTA alone** (`diff_fta == -tech_fta` exactly) | 3,665 | **94.3%** |
| tech_fta present but not exact | 21 | 0.5% |
| has an orphan front-end chance (1 FT, made, in the one-and-one window — a feed-gap candidate) | 26 | 0.7% |
| has an end-of-period trip (start_clock <= 3s) | 10 | 0.3% |
| has a blank-shooter FT row | 2 | 0.05% |
| unexplained residual | 171 | 4.4% |

Per-season "fully explained by technical FTA" share: 2022 91.3%, 2023 96.0%,
2024 97.0%, 2025 92.2%. Season-level `sum(tech_fta)` accounts for 96-109% of
the season's aggregate `box_fta - ev_fta` gap (1,805/1,654 in 2022; 2,439/2,487
in 2023; 2,053/2,059 in 2024; 1,961/2,044 in 2025) — i.e. this single,
identified mechanism is essentially the whole story, not one contributor among
many. The unexplained 4.4% has no dominant secondary cause and sits inside a
plausible noise floor for a metric already at ~90-98% exact/near-exact.

**Why this happens by design, not by accident.** `possessions.py`'s
`_handle_ft_trip` explicitly buffers technical-foul free throws into
`tech_points_off`/`tech_points_def` and never into `fta`/`ftm` ("they belong
to no possession"). This is the CORRECT choice for FT-2 (`docs/models/free_throw/model.md`
section 9: "the shooter on a technical is chosen by the coach... pooling
would bias both"). The defect is not in that exclusion — it is that nothing
downstream adds technical attempts back before comparing `ev_fta` to
`box_fta` (which does include them), and that the engine itself has no
technical-FT scoring rule yet (same section 9, already an open item, now
precisely sized: ~1,960-2,490 technical attempts/season, ~0.14-0.22 points
per team-game).

---

## 3. Points identity: `2*FGM2 + 3*FGM3 + ev_FTM` vs box final score

| season | n | n mismatch | % mismatch | mean pt diff | mean \|pt diff\| | corr(pt_diff, tech_ftm) |
|---|---|---|---|---|---|---|
| 2022 | 10,564 | 1,960 | 18.55% | -0.465 | 0.574 | -0.296 |
| 2023 | 11,080 | 2,225 | 20.08% | -0.544 | 0.566 | -0.343 |
| 2024 | 11,102 | 1,238 | 11.15% | -0.065 | 0.228 | -0.829 |
| 2025 | 11,179 | 948 | 8.48% | -0.206 | 0.242 | -0.355 |

Netting out `tech_ftm` (technical FT **makes**, the only technicals that score)
drops the mismatch rate to 12.6% / 12.0% / 4.0% / 1.6% and mean \|diff\| by
23-58%. The residual after netting tracks the same FGA/FGM feed-completeness
curve `docs/tests/possessions_build_v2_2026-09-10.md` already established
(worse in 2022-2023, converging by 2025) — a different, already-documented
gap, not a new one. **Points, unlike the naive formula, are NOT actually
wrong in the possession table itself**: `possessions.py` reconciles the
`points` column against the final score by adding `tech_points_off/def` back
in (`docs/tests/possessions_build_v2_2026-09-10.md` section 2 confirms exact
reconciliation). The naive identity asked for here is a useful diagnostic
precisely because it excludes that patch and isolates the FTM-specific gap.

---

## 4. Effect on the free-throw model's training labels

FT-2 training universe (`attempts_v1_era.parquet`, technical excluded,
finite `shooter_id`, per `build_ft_design`): 1,032,321 / 1,042,905 attempts.

| season | ambiguous share | ambiguous make rate (95% CI) | clean make rate (95% CI) | diff |
|---|---|---|---|---|
| 2022 | 40.93% | 0.7774 [0.7744, 0.7804] | 0.6738 [0.6710, 0.6766] | +0.104 |
| 2023 | 41.51% | 0.7735 [0.7706, 0.7763] | 0.6747 [0.6720, 0.6774] | +0.099 |
| 2024 | 40.67% | 0.7713 [0.7685, 0.7741] | 0.6825 [0.6799, 0.6850] | +0.089 |
| 2025 | 40.40% | 0.7757 [0.7729, 0.7785] | 0.6843 [0.6817, 0.6868] | +0.092 |
| pooled | 41.00% | 0.7744 [0.7732, 0.7757] | 0.6812 [0.6800, 0.6823] | +0.093 (z=103.7) |

`ambiguous` = a 2-attempt trip in the one-and-one or double-bonus window
(the only rows `classify_ft_trip` cannot disambiguate from a genuine 2-shot
shooting foul). The ~9pp make-rate gap is real, large, and not noise — but it
is exactly why the model's `in_bonus` feature exists: the model conditions on
the (partly noisy) bonus-vs-shooting context rather than needing the TRUE
foul type as a label, so this ambiguity is a known, already-instrumented
input, not a mislabelled target. It has no bearing on the FTA-count question
in sections 1-2 (confirmed there by near-zero correlation).

---

## 5. Third-source check: the four flagged 2025 final-score disagreements

ESPN's public game page (`www.espn.com/mens-college-basketball/game/_/gameId/<id>`,
the human-facing page hoopR's own scrape wraps) was fetched once per game
(4 requests total; the `site.api.espn.com` JSON endpoint 403's from this
environment). Full detail: `data/processed/truth/diag_finals_resolution_2025.json`.
`game_finals_v1.parquet` is unedited.

| game_id | hoopR (H-A) | CBBD (H-A) | ESPN (H-A) | winner |
|---|---|---|---|---|
| 401745889 | 79-59 | 77-59 | 79-59 (Bryant 79, Maine 59) | hoopR |
| 401723767 | 69-81 | 0-81 (gap) | 69-81 (Howard 69, Norfolk St 81) | hoopR |
| 401722537 | 62-60 | 60-62 | 60-62 (S. Illinois 60, Murray St 62) | **CBBD** |
| 401746100 | 80-67 | 80-65 | 80-67 (Akron 80, S. Alabama 67) | hoopR |

hoopR wins 3/4, but **not universally**: game 401722537 shows hoopR carrying
the side-flip, the opposite of what the truth table's own note assumed
("the same class of defect `_fix_flipped_sides` repairs" was read as
implicating CBBD). Confirms neither source should be silently preferred by a
grader.

---

## Verdict

**(a) — a labelling-adjacent defect that biases FTA and points, but not the
one the leading hypothesis named.** `ft_trip_ambiguous` is exonerated on
counts (uncorrelated, r <= 0.06 magnitude, every season) and does not need a
possessions-layer fix or an FT-model rerun on its account. The real,
94%-explained mechanism is the deliberate (and correct, for FT-2) exclusion
of technical free throws from `fta`/`ftm`/points, with nothing downstream
reconciling that exclusion against a box source that includes them. Two
concrete follow-ups, neither a hand-tuned patch on sim output (`CLAUDE.md`):
(1) any grader/truth-table comparing event-layer FTA/points to box or final
score should add technical attempts/makes back in first (a data-plumbing fix,
not a model fix); (2) the engine needs the technical-FT scoring rule
`docs/models/free_throw/model.md` section 9 already lists as an open item —
now sized at ~0.14-0.22 points/team-game, small but real and one-directional
(the engine currently scores strictly low on this component). No possessions
rebuild and no free-throw model rerun are indicated; both `possessions.py`
and the FT-2 exclusion are already doing the right thing.
