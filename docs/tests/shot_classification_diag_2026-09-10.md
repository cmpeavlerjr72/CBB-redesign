# Shot-classification diagnostic: is the 2024->2025 shift real or an artifact? -- 2026-09-10

Companion to `docs/models/possession_outcome/experiments.md` (section 3.6, the F2 `first`-population
decile calibration table) and `docs/tests/possessions_build_2026-09-10.md` (section 3.1, the
continuation-chance putback finding). Produced with `.venv/Scripts/python.exe`, duckdb/pandas over
`data/raw/cbbd/pbp/plays_{2022..2026}.parquet`, `data/raw/hoopr/team_box/team_box_{season}.parquet`,
`data/raw/hoopr/pbp/play_by_play_2025.parquet` and `data/processed/games_universe.parquet`. No file
under `src/cbb_sim/pbp/` or `data/processed/possessions/` was modified; `cbb_sim.pbp.events` and
`cbb_sim.pbp.possessions` were only imported and called.

## Verdict: MIXED, and the two things the pre-registration lumped together are not the same finding

**The two facts named in the question are not one phenomenon; they live in different chance
populations and have different causes.**

1. **The continuation-chance ("cont", i.e. after-OREB) rim-share collapse to 44.5%/41-42% in
   2025 IS AN ARTIFACT.** Confirmed by three independent methods below (playType tabulation,
   shot-location geometry, and an independent second feed). Root cause: ESPN's own 2025
   play-by-play feed mistags a large batch of true tip-in/putback attempts as `JumpShot` instead
   of `TipShot`/`LayUpShot`; `hoopR`'s independent parse of the same underlying broadcast feed
   carries the identical mistagged label on the identical plays, so this is not a bug in this
   repo's ETL or in `events.py` -- it is upstream, at the source. It is, however, exactly the kind
   of defect `events.py`'s own guardrail exists to catch, and `classify_frame` should be patched to
   route around it (proposed patch below) because the ESPN/CBBD `playType` field is not the only
   signal the feed carries for these plays.

2. **The `first`-chance FGA_jump2 (+1.8 pp) / FGA_3 (-1.15 pp) level shift reported in
   `experiments.md` section 3.6 is NOT explained by that artifact at the target-label level** --
   `first` chances are by definition the chances that do not follow an OREB, so the putback defect
   cannot relabel a `first`-chance shot. The aggregate 3PA/2PA-vs-hoopR-box reconciliation (section 2
   below) also shows no 2025-specific degradation, which rules out a broader 2-vs-3 or FGA-total
   defect as the explanation. Most of this gap is genuine, continuing shot-mix drift (L4). **But**
   section 6 below shows a concrete, verified mechanism by which the SAME putback artifact leaks
   into the `first`-chance model's own *predictors* (not its targets) via the `off_rim_c` feature,
   which is built from `sum(fga_rim)/sum(fga)` over the **whole** `possessions_{season}.parquet`
   table -- continuation chances included. That predicts exactly the asymmetric pattern actually
   observed (`FGA_rim` under-predicted by only -0.42 pp, well inside gate, vs `FGA_jump2`
   over-predicted by +1.80 pp, failing gate), while `FGA_3`'s separate and larger contribution to the
   drift (-1.15 pp) is untouched by this channel (3PA are never touched by the putback defect) and
   should be read as the genuine trend continuing out-of-sample.

Net: fix the event-mapping defect (it is real, it is measurable, and it is not a modelling problem);
do not expect that fix to fully close the `first`-population gate on its own, because part of that
gap is real drift, but do expect it to shrink the `FGA_jump2`/`FGA_rim` legs of it once `off_rim_c`
is rebuilt on corrected possessions.

---

## 1. How `events.py` determines FGA_3 vs FGA_jump2 vs FGA_rim

Read directly from `src/cbb_sim/pbp/events.py` (not modified):

* **Rim vs. jump family is decided by `playType` alone, with no fallback.**
  `PLAY_TYPE_TO_EVENT` maps `DunkShot`, `LayUpShot`, `TipShot` -> `FAM_SHOT_RIM` and `JumpShot` ->
  `FAM_SHOT_JUMP` (lines 174-177). `classify_frame` (line 283) sets `ev[fam == FAM_SHOT_RIM] =
  "FGA_rim"` unconditionally for the first group and resolves the second group only into
  `FGA_jump2`/`FGA_3` (lines 296-300). **There is no path by which a `JumpShot`-tagged row can ever
  become `FGA_rim`, regardless of any other column on the row.** This is the exact gap the diagnostic
  below is aimed at.
* **2 vs. 3 is decided by `shot_range` first** (`_is_three`, lines 262-280): `shot_range ==
  'three_pointer'` -> three, else two; `playText` containing "three" is the fallback when
  `shot_range` is null (never triggers in the current data, 0 nulls in all 5 seasons); `scoreValue
  == 3` is the last resort (0.0067% of rows, uniform across seasons -- see section 2).
* `shot_range` is **not** something this repo derives from `playType`. It is CBBD's own
  `shotInfo.range` API field, populated at pull time
  (`scripts/pull_cbbd_pbp.py:292`, `row["shot_range"] = shot_info.get("range")`) -- a genuinely
  separate upstream field from `playType`. **New finding, not previously documented:** empirically,
  across all 2,261,348 de-duplicated `JumpShot` rows in 2022-2026, `shot_range` is **never** `'rim'`,
  and across all `DunkShot`/`LayUpShot`/`TipShot` rows it is **always** exactly `'rim'` (100.000% in
  every season, checked directly — see `rim_jumpshot_check.py` output). So although `shot_range` is a
  separate field from `playType` in the schema, it carries **zero independent information about the
  rim/jump split** -- it is generated in lockstep with `playType` at the source (ESPN), not from
  shot geometry. This means `shot_range` cannot be used to catch or repair a `playType` rim/jump
  mistag, which is exactly why the mistag documented below survives undetected by every check that
  only looks at CBBD's own categorical fields.

## 2. Reconciliation vs. hoopR `team_box` (ground truth), 2022-2026

Universe: `is_d1_game & ~pbp_truncated`, joined `games_universe.game_id <-> team_box.game_id` /
`home_team_id,away_team_id <-> team_id` (ESPN ids on both sides; CBBD's own `teamId` is a different,
internal key and is not used for this join -- see `cbb_sim.pbp.possessions._fix_flipped_sides` for
the home/away resolution this reuses). Per-team-game counts from CBBD plays via
`cbb_sim.pbp.events.classify_frame`: `3PA = count(FGA_3)`, `2PA = count(FGA_rim)+count(FGA_jump2)`,
`FTA = count(FT_made|FT_missed)`, `OREB` excludes the administrative between-free-throws rebound
(same test `possessions.py` uses), `TOV` includes unpaired-steal promotion. n = 10,564-11,433
team-games/season.

**3PA (pbp - box)**

| season | n | mean diff | MAE | % exact match |
|---|---|---|---|---|
| 2022 | 10,564 | +0.011 | 0.040 | 98.48% |
| 2023 | 11,080 | -0.008 | 0.023 | 98.91% |
| 2024 | 11,102 | -0.001 | 0.013 | 98.86% |
| **2025** | **11,179** | **-0.013** | **0.031** | **98.69%** |
| 2026 | 11,433 | +0.026 | 0.059 | 98.40% |

**2PA = FGA-3PA (pbp - box)**

| season | n | mean diff | MAE | % exact match |
|---|---|---|---|---|
| 2022 | 10,564 | -0.376 | 0.434 | 80.43% |
| 2023 | 11,080 | -0.384 | 0.406 | 81.67% |
| 2024 | 11,102 | +0.035 | 0.061 | 94.40% |
| **2025** | **11,179** | **-0.045** | **0.069** | **97.30%** |
| 2026 | 11,433 | +0.047 | 0.080 | 97.69% |

**FTA / OREB / TOV (pbp - box)**

| stat | 2022 MAE | 2023 MAE | 2024 MAE | **2025 MAE** | 2026 MAE |
|---|---|---|---|---|---|
| FTA | 0.028 | 0.014 | 0.003 | **0.021** | 0.038 |
| OREB | 0.258 | 0.271 | 0.322 | **0.243** | 0.035 |
| TOV | 0.073 | 0.043 | 0.035 | **0.028** | 0.046 |

**Reading this: 2025 is unremarkable, or the best of the pre-2026 seasons, on every one of these
five statistics.** 2025's 3PA reconciliation sits mid-pack (better than 2022/2026, worse than
2023/2024) and its 2PA reconciliation is dramatically *better* than 2022/2023 (which have a
structural -0.38 mean bias from a different, known feed-quality issue in those years) and comparable
to 2024/2026. **There is no season-specific spike in 2025 anywhere in this table.** This rules out a
broad 2-vs-3 split defect or an FGA-total defect as an explanation for anything -- the total FGA and
its 3PA/2PA split reconcile to the official box score exactly as well in 2025 as in every other
season. It does **not**, and cannot, rule out a defect in how the 2PA total is *split between* rim and
jump2, because hoopR's team-level box score does not report that split at all. Section 4 closes that
gap with shot-level geometry.

## 3. playText patterns by season

Top templates for `JumpShot` rows (player name replaced with `<PLAYER>`), 2022-2025 vs 2026:

* **2022-2025 (unchanged format across all four seasons):** `<PLAYER> missed Three Point Jumper.`,
  `<PLAYER> missed Jumper.`, `<PLAYER> made Jumper.`, `<PLAYER> made Three Point Jumper.[ Assisted by
  ...]`. These four templates alone cover >99.5% of rows in every one of 2022, 2023, 2024, 2025.
  **No template unique to 2025 exists, and no template's relative frequency changes sharply at the
  2024->2025 boundary.**
* **2026 (a genuine format change, unrelated to the 2025 issue):** ESPN switches to
  `<PLAYER> makes/misses #-foot [pullup/floating/turnaround/step back] jump shot`, i.e. adds shot
  subtype and distance to the text. This is the "makes"/"misses" verbiage change already noted in
  the `events.py` module docstring for `MadeFreeThrow`; it now visibly appears for `JumpShot` text
  too, first in 2026.
* **`TipShot`'s own text is unambiguous in every season, including 2025:** `<PLAYER> made/missed Two
  Point Tip Shot.` (2022: n=1,509; 2025: n=9,175). **When ESPN *does* tag a play `TipShot`, the text
  confirms it correctly, every season.** The problem (section 4) is plays that should be `TipShot`
  but are tagged `JumpShot` instead, carrying the generic `made/missed Jumper.` text with **no**
  textual hint that a rebound was just tipped in. Checked directly: of 462,118 `JumpShot` rows in
  2025, only 42 contain the substring "tip" (vs. 124/431/331/15 in 2022/2023/2024/2026 -- if
  anything, *fewer* than the neighboring seasons) and zero contain "putback". **`playText` carries no
  information that would let a text-only check find or explain this defect; only shot location does
  (section 4).**
* Missed-shot 2-vs-3 (`scoreValue == 0`): confirmed unused by design; `shot_range` alone resolves
  100% of `JumpShot` rows in all five seasons (0 nulls), consistent with the module docstring's
  measured 0.0067% failure rate attributable to `scoreValue` disagreement, itself uniform across
  seasons (2/22/18/110 rows in 2023/2024/2025/2026 respectively -- 2025 is not elevated).

## 4. Putbacks: playType tabulation, and the independent geometric proof

**Definition used** (matches `docs/tests/possessions_build_2026-09-10.md` section 3.1's
methodology): inert classes (`timeout, sub, block, jumpball, challenge`) dropped first via
`cbb_sim.pbp.events.INERT_CLASSES`; a live (non-administrative) `OREB` by team T, followed by the
next surviving event, same team, same period, within 5 seconds by `secondsRemaining`.

**Putback playType distribution (%), n_live_oreb = 117k-134k/season:**

| putback label | 2022 | 2023 | 2024 | **2025** | 2026 |
|---|---|---|---|---|---|
| LayUpShot | 48.00 | 49.42 | 49.60 | **28.49** | 28.73 |
| DunkShot | 4.04 | 3.93 | 3.94 | **1.77** | 1.92 |
| TipShot | 2.26 | 1.67 | 1.01 | **11.88** | 24.39 |
| JumpShot(2) | 12.62 | 12.02 | 11.57 | **22.43** | 10.29 |
| JumpShot(3) | 19.54 | 19.71 | 20.60 | **22.51** | 23.23 |
| **rim total** | **54.29** | **55.03** | **54.55** | **42.14** | **55.03** |

This reproduces (within ~2 pp, an artifact of a slightly different adjacency test) the finding
already reported in `docs/tests/possessions_build_2026-09-10.md` section 3.1. **What follows is new:
independent, geometry-based proof of what these 2025 "JumpShot(2)" putback rows actually are.**

### 4.1 Shot-location geometry (the decisive check)

CBBD's plays carry `shot_location_x`/`shot_location_y` -- a separate ESPN sub-system (a shot-chart
click) from the categorical `playType` tagger, populated on 76-97% of shooting rows depending on
season/type (`shot_location_y` coverage: 76-88% in 2022-2024, 97-99% in 2025-2026). Calibrating the
coordinate system empirically against `DunkShot` (median release point 2.3 ft from the hoop in every
season) establishes baskets at raw coordinate `(52.5, 250)` and `(887.5, 250)` on a 0-940 x 0-500
(0.1 ft units) full-court grid; distance-to-nearest-basket = `min(dist to each basket) / 10` feet.

**Distance from basket of "JumpShot(2)" putback attempts, by season:**

| season | n | median dist (ft) | share within 1 ft | share within 4 ft |
|---|---|---|---|---|
| 2022 | 7,023 | 9.20 | 0.75% | 12.94% |
| 2023 | 7,332 | 9.28 | 0.64% | 12.48% |
| 2024 | 7,646 | 8.98 | 0.71% | 13.52% |
| **2025** | **16,668** | **0.55** | **50.23%** | **56.46%** |
| 2026 | 6,966 | 8.91 | 0.52% | 11.87% |

A "two-point jumper" attempt taken right after an offensive rebound normally sits at a median of
~9 feet from the hoop -- a real short/mid jumper, and this number is essentially identical in 2022,
2023, 2024 and 2026. **In 2025 alone, the median collapses to 0.55 feet: literally standing under the
rim.** Half of all "JumpShot(2)" putback attempts in 2025 are within 1 foot of the basket, against
well under 1% in every other season -- a discontinuity with no basketball explanation.

**These are not scattered near-rim clicks; they are snapped to two exact pixels.** The `(x,y)` pairs
among 2025's within-1-ft cluster are overwhelmingly `(56.4, 250)` and `(883.6, 250)` (10,522 of
11,042 season-wide, ~95%, plus small jitter variants `±5` in y). **This is the identical canned/default
coordinate that legitimate `TipShot` rows use in 2025-2026** -- `TipShot`'s own location coverage
jumps from ~6% (2022-2024, essentially unlocated) to 97.3% in 2025 and 87.5% in 2026, all landing on
this same placeholder pixel (median `TipShot` distance: 4.27/4.18/2.56 ft in 2022/2023/2024, **0.39
ft in 2025 and 2026**). This is convergent, independent evidence for one specific claim: ESPN's shot
chart / location subsystem correctly flags these plays as "at the rim" (a tip-in placeholder), while
its separate categorical shot-type tagger mislabels a large fraction of them `JumpShot` instead of
`TipShot`, specifically and only in the 2025 season.

**Season-wide (not restricted to the 5-second putback window), 5.07% of ALL `FGA_jump2` attempts in
2025** sit within 1 ft of the basket at this exact placeholder location, vs. 0.12-0.47% in
2022/2023/2024/2026 (n = 217,580 `FGA_jump2` rows checked in 2025). That is an excess of roughly
10,300-10,400 misclassified shots league-wide for the season, of which **8,373 (about 80%) fall
inside the strict 5-second post-OREB window** used above; the remainder are presumably slightly
delayed rebound scrambles or other rim-adjacent dead-ball sequences outside that window.

## 5. Cross-check against hoopR's independent parse of the same 2025 games

If this were a CBBD-side ETL bug (a parsing error unique to `scripts/pull_cbbd_pbp.py` or to this
repo's mapping), a second, independently-built feed of the same underlying broadcast data should
disagree. Joined the 8,373 CBBD-flagged "JumpShot(2) putback, distance <= 1 ft" 2025 rows to
`data/raw/hoopr/pbp/play_by_play_2025.parquet` on `(ESPN game_id via games_universe, shooter name,
period, clock +/- 5s, shooting_play == True)` -- `shot_shooter_id` in CBBD is a CBBD-internal roster
key, not an ESPN athlete id, so name is the only usable join key across feeds; 8,345 of 8,373
(99.7%) matched a hoopR shooting play within the window.

**Result: 8,310 of 8,345 matched rows (99.6%) are ALSO tagged `JumpShot` by hoopR**, with
byte-identical text (`<player> made/missed Jumper.`). Sample:

```
CBBD text                        hoopR type_text   hoopR text
Toyaz Solomon made Jumper.       JumpShot          Toyaz Solomon made Jumper.
Mouhamed Dioubate made Jumper.   JumpShot          Mouhamed Dioubate made Jumper.
Kadin Shedrick missed Jumper.    JumpShot          Kadin Shedrick missed Jumper.
```

**Conclusion: this is not a CBBD parsing bug, and it is not introduced anywhere in this repository.**
Two independently-built downstream feeds (CBBD/ESPN's own play-by-play API and hoopR's separate ESPN
scrape/parse) carry the identical wrong categorical tag on the identical plays. The defect is
upstream, in ESPN's 2025-season play-type classification itself, for tip-in/putback attempts
specifically. The fix nonetheless belongs in `cbb_sim.pbp.events.classify_frame`, because that is the
one place this repo's "explicit mapping table, no silent fallback" guardrail
(`docs/SIM_GUARDRAILS.md` section 4) can route around an upstream vendor defect using a signal
(shot location) the vendor did get right.

## 6. A verified contamination channel into the `first`-chance model's own features

`docs/models/possession_outcome/features.md` section 1: `off_rim_c` is computed as
`100 * sum(fga_rim) / sum(fga)` (centred) **over `data/processed/possessions/possessions_{season}.parquet`**
-- the POSSESSION table, not the chance table. `cbb_sim.pbp.possessions._emit` sums `fga_rim`/`fga_jump2`
*across every chance of the possession*, first and continuation alike (`"fga_rim": sum(c.fga_rim for c
in chs)`). **So a team's rolling, as-of `off_rim_c` feature -- used as a predictor for the `first`
population, including F2's 2025 test season -- is computed from a mix that includes the mislabeled
2025 continuation-chance shots.** A team's true rim share is therefore mechanically understated (and
its true jump2 share overstated) in its own 2025 `off_rim_c` history, and that understated feature
feeds the model's *predictions* for that team's later 2025 games, independent of what the `first`
population's *target* labels look like (those are unaffected, since `first` chances never follow an
OREB by construction).

This predicts exactly the **asymmetric** shape actually observed in `experiments.md` section 3.6 for
the lowest-loss (`lgbm`+`C_plus_state`) arm on F2 `first`:

| class | level shift (pp) | consistent with the feature-contamination channel? |
|---|---|---|
| FGA_rim | -0.424 (small, non-gating) | yes -- rim under-predicted, matching an understated `off_rim_c` |
| FGA_jump2 | **+1.802 (gating)** | yes -- the mass that should have counted toward rim is, in a team's own history, sitting in the jump2 bucket instead |
| FGA_3 | -1.145 (gating) | **no** -- 3PA are never touched by the putback defect (`JumpShot(3)` putback distance is a stable ~24.6 ft in every season, section 4); this leg of the drift is untouched by this channel |

Fully quantifying how much of the +1.80/-0.42 pp rim/jump2 pair this channel explains requires
rebuilding `possessions_{2022..2025}.parquet` with the location-based fix (section 7) and refitting
`off_rim_c` / refitting the F2 model -- out of scope for a read-only diagnostic that was told not to
touch the possessions parquet files. Recommend it as the concrete next step. **The FGA_3 leg of the
gap is not explained by anything found in this diagnostic and should be read as genuine, continuing
three-point-rate drift (L4) that a season-index feature under-shoots one year out-of-sample** -- this
part of the calibration failure is real, not an artifact.

## 7. Proposed fix (diff, not applied -- `src/cbb_sim/pbp/` was not modified per instructions)

Add a location-based override to `classify_frame` in `src/cbb_sim/pbp/events.py`, applied only to
rows already resolved to the jump family, so it can only ever move a shot *toward* `FGA_rim` and
never touch `DunkShot`/`LayUpShot`/`TipShot` rows or the three-point rows (`JumpShot(3)` putback
distance is a stable ~24.6 ft with zero rows near the rim in any season -- no risk of clipping a real
three):

```diff
--- a/src/cbb_sim/pbp/events.py
+++ b/src/cbb_sim/pbp/events.py
@@ def classify_frame(plays: pd.DataFrame) -> pd.Series:
     # SHOT_RIM / SHOT_JUMP resolve directly.
     ev[fam == FAM_SHOT_RIM] = "FGA_rim"
     jump = fam == FAM_SHOT_JUMP
     ev[jump & is_three] = "FGA_3"
     ev[jump & ~is_three] = "FGA_jump2"
+
+    # RIM-LOCATION OVERRIDE (docs/tests/shot_classification_diag_2026-09-10.md).
+    # ESPN's 2025-season feed mistags a large batch of true tip-in/putback
+    # attempts `JumpShot` instead of `TipShot`/`LayUpShot` (confirmed
+    # independently in hoopR's separate parse of the same games -- this is an
+    # upstream vendor defect, not a CBBD or events.py bug). shot_range cannot
+    # catch it (it is generated in lockstep with playType, never 'rim' for a
+    # JumpShot row, in any season). shot_location can: these rows carry the
+    # same rim-distance placeholder coordinate TipShot rows use. A JumpShot
+    # row within RIM_OVERRIDE_MAX_FT of a basket is a rim attempt regardless
+    # of its playType tag.
+    RIM_OVERRIDE_MAX_FT = 1.5
+    BASKET_1, BASKET_2 = (52.5, 250.0), (887.5, 250.0)
+    loc_x = pd.to_numeric(plays["shot_location_x"], errors="coerce")
+    loc_y = pd.to_numeric(plays["shot_location_y"], errors="coerce")
+    d1 = np.sqrt((loc_x - BASKET_1[0]) ** 2 + (loc_y - BASKET_1[1]) ** 2)
+    d2 = np.sqrt((loc_x - BASKET_2[0]) ** 2 + (loc_y - BASKET_2[1]) ** 2)
+    dist_ft = np.minimum(d1, d2) / 10.0
+    near_rim = (jump & (dist_ft <= RIM_OVERRIDE_MAX_FT)).to_numpy()
+    ev[near_rim] = "FGA_rim"
```

**Threshold choice, made explicit rather than hidden in one number:**

| threshold | 2025 putback JumpShot(2) reclassified | excess vs. season baseline (~0.6%/~13%) | implied post-fix `cont` population rim / jump2 share (2025) |
|---|---|---|---|
| <= 1.0 ft (matches the exact placeholder pixel, near-zero false-positive rate: 0.5-0.8% baseline in every clean season) | 8,373 of 16,668 (50.2%) | ~8,270 excess | rim 31.63 -> **38.7%**, jump2 19.76 -> **12.7%** (slightly overshoots the 2022-2024/2026 band of rim 37.9-39.2% / jump2 14.4-15.1%) |
| <= 4.0 ft (also captures genuine close-in push shots/floaters at the normal ~13% clean-season rate, so nets out to the same excess estimate) | 9,411 of 16,668 (56.5%) | ~7,245 excess | rim 31.63 -> **37.8%**, jump2 19.76 -> **13.6%** (lands inside the 2022-2024/2026 band on both classes) |

The 4 ft threshold reproduces the pre-2025 continuation-population rim/jump2 shares almost exactly on
both sides simultaneously, which is itself a strong internal-consistency check on the diagnosis (not
just the direction, the *magnitude* of the correction matches what a clean season looks like). The
diff above uses 1.5 ft as a conservative middle value that stays essentially entirely inside the
placeholder-pixel cluster; **the exact cutoff is a modelling choice and belongs in a pre-registration
per `docs/tests/possessions_build_2026-09-10.md` section 3.1's own stated policy**, not asserted here.
Either choice requires `shot_location_x`/`shot_location_y` to be added to
`cbb_sim.pbp.events.PLAY_COLUMNS` (currently loaded, just not read by `classify_frame`) and to
`scripts/build_possessions.py`'s column list if not already present there.

**What this fixes and what it does not:** this repairs the `cont`-population rim/jump2 split for 2025
(section 4) and should shrink, not necessarily close, the `first`-population `FGA_rim`/`FGA_jump2`
gap via the `off_rim_c` feature-contamination channel (section 6). It will not move `FGA_3` at all --
that part of the F2 `first` calibration failure is real drift and no event-mapping fix addresses it.

## 8. CBBD feed completeness per season, and a proposed `pbp_complete` flag

Methodology: for every `is_d1_game` (no other pre-filtering), take the CBBD feed's own last-recorded
running `homeScore`+`awayScore` (by `id` order within `gameId`) and compare to
`games_universe.home_score + away_score` (the official final). This is a direct, single-purpose
completeness check that does not depend on the possession state machine, tech-FT accounting, or the
0.44 free-throw-trip approximation.

| season | n games | no pbp rows at all | pbp final short of box | pbp final exceeds box (data issue, rare) | **% incomplete** |
|---|---|---|---|---|---|
| 2022 | 5,485 | 124 | 75 | 4 | 3.63% |
| 2023 | 5,749 | 117 | 85 | 8 | 3.51% |
| 2024 | 5,731 | 87 | 82 | 9 | 2.95% |
| 2025 | 5,769 | 117 | 95 | 3 | 3.67% |
| 2026 | 5,767 | 41 | 126 | 2 | 2.90% |

Cross-tabulated against the existing (hoopR-derived) `games_universe.pbp_truncated` flag: in every
season, the "short of final" games found here are overwhelmingly (91-98%) already inside
`pbp_truncated == True` (e.g. 2022: 75 of 79 `pbp_truncated` games are short here; only 4 games flip
the other way, i.e. flagged truncated by hoopR's own check but reaching the final score in CBBD's
feed). **This measurement disagrees substantially with the "of which feed-incomplete" column in
`docs/tests/possessions_build_2026-09-10.md` section 2** (1,028/5,282 = 19.5% in 2022, computed
*within* the `~pbp_truncated` subset from the segmented possession table's own summed `points`
field rather than from the feed's raw running score column directly). That check and this one are
not measuring the same thing and the gap between 3.6% (here, all games) and 19.5% (there, already
`~pbp_truncated`-filtered games only) is large enough to be worth reconciling on its own -- a
plausible cause is technical-free-throw point accounting in the possession table's `points` field
(carried separately as `tech_points_off`/`tech_points_def` per that doc's own section 2, so a check
against `points` alone rather than `points + tech_points` would manufacture a false shortfall on any
game with a technical foul) -- but resolving that discrepancy is outside this diagnostic's scope and
is flagged here as a followup rather than adjudicated.

**Proposed flag**, using this diagnostic's direct definition (auditable independent of the possession
state machine):

```
pbp_complete = has_pbp_rows AND (sum(homeScore, awayScore) at the last CBBD play by id order
                                  >= games_universe.home_score + games_universe.away_score)
```

computed once per `cbbd_game_id` at the `games_universe` build step, alongside the existing
(hoopR-derived) `pbp_truncated`, rather than replacing it -- the two flags come from independently
sourced feeds and disagreeing rows (the 4-9 per season above) are themselves informative about which
single feed failed for a given game.

---

## Appendix: scripts used (scratch, not committed)

All analysis scripts referenced above were written to the session scratchpad
(`recon.py`, `recon_summary.py`, `text_patterns.py`, `rim_jumpshot_check.py`, `cbbd_location_check.py`,
`cbbd_geom_check.py`, `putback_analysis.py`, `putback_v2.py`, `putback_hist.py`, `jump2_all_dist.py`,
`hoopr_crosscheck.py`, `rim_text_templates.py`, `completeness.py`) and are reproducible against the
parquet files cited throughout; none modified any file under `src/` or `data/processed/possessions/`.
