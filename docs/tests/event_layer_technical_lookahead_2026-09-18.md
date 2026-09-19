# Event layer: the technical free-throw lookahead (2026-09-18)

Lane: event-layer bug fix, VERSIONED SIBLING ONLY. **No default was changed
and no consumer was switched.** `data/processed/possessions{,_v2}` and
`data/processed/models/free_throw/{trips,attempts}_v1_era.parquet` are
untouched, and the default code path is proved bit-identical below (section 3).

Follows `docs/tests/free_throw_technicals_round1b_2026-09-18.md` and
`docs/models/free_throw/experiments.md` section 10 (commits `b8d20c1`,
`8fa8334`), which identified the defect but could not touch `src/cbb_sim/`.

Code: `src/cbb_sim/pbp/possessions.py` (module docstring, "TECHNICAL FREE
THROWS"; `technical_ft_index`, `_GameMachine(tech_lookahead=...)`),
`src/cbb_sim/models/event_stream.py` (`build_stream(tech_lookahead=...)`),
`src/cbb_sim/models/free_throw.py` (pass-through).
Scripts: `scripts/build_possessions_techfix_v3.py`,
`scripts/diag_technical_blast_radius_v1.py`. Tests:
`tests/test_technical_lookahead.py`.

**Wall clock (system clock and git timestamps, not estimates).** Lane start
19:50 EDT -> first commit `a3899a5` at **20:24:12 EDT = 34 minutes** for the
investigation, the fix, the tests, the builds and the blast radius. Inside
that: the v3 build (four seasons of possessions + chances + both trip/attempt
siblings, one process) ran **20:13:08 -> 20:16:52 = 223.5 s** by the script's
own timer; the blast-radius comparison **29.5 s**; the new unit-test file
**0.55 s**. The full-suite run is section 7. All on
`.venv/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, thread env vars pinned
to 1, never more than two of this lane's own processes at once, and no process
started or signalled that this lane did not start itself.

---

## 1. What the inserted administrative rows are

Census over every single-team `Technical Foul` moment in CBBD's raw pbp,
seasons 2022-2025 (inert rows dropped first, so "the next row" means what the
possession machine means by it):

| next row's playType | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| `MadeFreeThrow` (the case the one-row rule catches) | 854 | 1,305 | 897 | 829 |
| **`Lost Ball Turnover`** | **288** | **572** | **329** | **298** |
| **`PersonalFoul`** | **67** | **46** | **50** | **6** |
| `Technical Foul` (a second T at the same clock) | 11 | 6 | 12 | 8 |
| rebounds / field goals (all types) | 19 | 20 | 16 | 5 |

On the bug population (the moments whose next row is not a free throw): the
inserted row is at the **identical frozen clock on 100.0%** of moments in all
four seasons, belongs to the **offending team on 98.9-99.7%**, and a
beneficiary free throw is found later at that same frozen clock on 87.5-94.5%.
The gap is exactly **one row on 99.0%** of them and never exceeds six
(2022-2025 maximum), which is where `TECH_LOOKAHEAD_MAX_ROWS = 6` comes from.

The inserted `Lost Ball Turnover` **names the same player as the technical**:
on the rows where both carry a `participant_1_id`, they match on 99.53% /
99.58% / 100.00% / 99.65% (2022-2025). The 102-129 rows a season that carry no
player id are the team/bench technicals, and their `playText` is the team name
with an empty player slot ("`Texas  Turnover.`" against "`Technical Foul on
Texas.`"). Hand-read transcripts, 16 games across 2024-2025, all show the same
shape; two of them:

```
game 401591429 (2024) p2 563s
  563s  Technical Foul       team 333   Technical Foul on Nick Pringle.
  563s  Lost Ball Turnover   team 333   Nick Pringle Turnover.        <-- inserted
  563s  MadeFreeThrow        team 2413  Riley Minix made Free Throw.
  563s  MadeFreeThrow        team 2413  Riley Minix made Free Throw.

game 401583794 (2024) p1 910s
  910s  Technical Foul       team 2     Technical Foul on Chad Baker-Mazara.
  910s  PersonalFoul         team 2     Foul on Chad Baker-Mazara.    <-- inserted
  910s  MadeFreeThrow        team 239   Ja'Kobe Walter made Free Throw.
  910s  MadeFreeThrow        team 239   Ja'Kobe Walter made Free Throw.
```

### 1.1 The box evidence: the turnover is REAL, so the fix does not delete it

The job's question -- is the `Lost Ball Turnover` a real turnover or a feed
artefact, i.e. does the fix also have to remove phantom turnovers and
possessions -- is decided against hoopR's team box (`turnovers`), over 43,925
matched team-games, 2022-2025:

| inserted technical turnovers on this team-game | n team-games | mean(event `Lost Ball Turnover` rows - box `turnovers`) |
|---:|---:|---:|
| 0 | 41,444 | +0.0123 |
| 1 | 2,302 | +0.0191 |
| 2 | 161 | +0.0745 |
| 3 | 18 | +0.3333 |

Paired against the k=0 baseline: **k=1 gives +0.0068 (SE 0.0051)** and k=2
gives +0.0622 (SE 0.0258). A feed artefact the box does not know about would
give **+1.00** and **+2.00**. The observed shift is 0.7% of that, 195 standard
errors from the artefact hypothesis. **The box counts these turnovers.**

The box is not a re-add of the same pbp rows, so this is not circular: the
event stream and the box disagree on **2.69%** of team-games (1,183 of 43,925;
-3..+3 spread), which they could not do if the box were derived from the pbp
rows being counted. Sanity check on the same population: event free-throw rows
minus box FTA is +0.0007 / -0.0096 / +0.0168 for k = 0 / 1 / 2, so the free
throws themselves are accounted for identically in every group.

**Consequence for the fix: it is a LOOKAHEAD ONLY.** Deleting the inserted
turnover would manufacture a fresh -1 box disagreement per technical. Every
intervening row is still processed exactly as before; only the free throws'
class changes. The measured result is in section 4: **turnover counts are
bit-identical between the old and new tables in all four seasons.**

### 1.2 A SEPARATE defect, measured and NOT patched

On the next live-ball event after the technical trip, the ball belongs to the
**technical'd team** on 65.8% of the moments that carry the inserted turnover
(n = 1,452) against 22.4% of the technical moments that do not (n = 4,106) --
i.e. the row the box counts as a turnover usually does *not* coincide with the
ball changing hands, because NCAA resumes at the point of interruption. The
possession machine still closes a possession on it and opens another. That is a
possible over-count of possessions in exactly these ~1,500 moments a season
(~0.05% of possessions), it is a different defect from the one this lane was
given, and per `CLAUDE.md` it is reported here rather than patched inside a fix
for something else. It is unchanged by v3.

---

## 2. The fix

`_handle_technical` looked at exactly one row ahead. The fix, behind
`tech_lookahead` (default `False`), runs `technical_ft_index`: a bounded
same-clock scan forward for the first free throw by the beneficiary, while
`(period, secondsRemaining)` are unchanged, stepping over at most
`TECH_LOOKAHEAD_MAX_ROWS = 6` non-free-throw rows and stopping at any field
goal attempt, period boundary, game boundary, or a free throw by the wrong
team. If the free throw is the very next row, the pre-fix code path is taken
unchanged.

If it is further ahead, the machine **arms an exact row index** and returns, so
the main loop processes the inserted turnover and the duplicate personal foul
normally; the free-throw branch then consumes that one index as a technical
trip, and `_handle_foul` checks the same marker so a duplicate `PersonalFoul`
row still increments the team foul count but no longer claims the technical's
free throws. Because the marker is a row index and not a clock predicate, it
can never claim a different trip at the same frozen clock. Offsetting
technicals (one per team at one clock) shoot no free throws by rule: the scan
finds none and nothing is armed.

`cbb_sim.models.event_stream._attach_trips` carries the identical defect in the
backward direction (`trip_cause` reads one row back, so it resolves to `none`
or `foul`). It now imports and re-runs the **same** `technical_ft_index` rather
than implementing a second rule, and forces `trip_cause` on the trip it points
at. Nothing else in the stream moves.

### 2.1 Residual against the verified target

Against `technical_target_verified_trips_v1.parquet`, matched at
`(game_id, period, clock, beneficiary)` moment grain:

| season | target moments | old matched | **new matched** | new missed | new extra | old recovery | **new recovery** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 1,281 | 883 | **1,197** | 84 (6.6%) | 48 | 70.4% | **99.4%** |
| 2023 | 2,082 | 1,342 | **1,910** | 172 (8.3%) | 45 | 65.1% | **95.1%** |
| 2024 | 1,342 | 933 | **1,250** | 92 (6.9%) | 65 | 71.1% | **100.3%** |
| 2025 | 1,204 | 869 | **1,140** | 64 (5.3%) | 60 | 73.0% | **101.3%** |

(Recovery = technical trips in the trip table as a percentage of the target's
trip count; it exceeds 100% in 2024/2025 because the "extra" moments below are
counted too.) The residual is **5.3-8.3% missed and 3.7-5.0% extra**, and it is
the *same* double/multi-technical-at-one-frozen-clock edge case the verified
target itself carries a quantified 8-10% residual on -- both this scan and the
target's scan resolve one trip per armed moment, so a second technical on the
same team at the identical clock is under-resolved by one on both sides. The
residual is reported, not patched. Double technicals **at different clocks**
are unaffected (each is its own moment). This is a strict improvement on every
axis in every season: missed moments fall from 398/740/409/335 to 84/172/92/64.

---

## 3. Default-path identity, and the tests

* `segment_season(2025, universe)` called with **no keyword** reproduces
  `data/processed/possessions_v2/possessions_2025.parquet` and
  `chances_2025.parquet` **bit-identically** (`DataFrame.equals` True on both,
  768,540 possession rows / 886,462 chance rows), with the diagnostic reporting
  `tech_lookahead: False, n_tech_lookahead_hits: 0`.
* `FT.build_trips_and_attempts([2025], version="v1")` with no keyword
  reproduces `trips_v1_era.parquet`'s 2025 slice: 118,075 rows and every
  `foul_class` count identical to the row
  (`shooting 49,032 / bonus_one_and_one 32,853 / double_bonus 17,628 /
  and_one 16,150 / unknown 1,533 / technical 879`). The only column-level
  differences are two pre-existing, unrelated parquet round-trip artefacts
  (`game_date` `datetime64[s]` vs `[ms]`, and NaN-vs-NaN on `shooter_id`);
  neither is touched by this change.
* `tests/test_technical_lookahead.py`: **19 tests, 19 passed** (0.52-0.55 s). Every
  sequence is a real CBBD row sequence transcribed from a hand-read game
  (401591429, 401583794, 401597885, 401706151). The suite asserts, with the
  flag OFF, that the machine still reproduces the documented *defect* on both
  inserted-row shapes -- a default-path change would silently invalidate the
  tables on disk -- and that points reconcile under both flags.
* Full suite: see section 7 -- it found one real failure caused by this
  change (a test that used `"v3"` as its example of an *unknown* version
  label), now fixed.

---

## 4. Blast radius: possessions, v2 (old) vs v3 (new)

v3 is v2's event layer plus the lookahead, so the diff below is the fix alone.

| season | poss/game old -> new | TOV old -> new | FTA old -> new | tech points old -> new | FTA/FGA old -> new | TOV% old -> new |
|---|---|---|---|---|---|---|
| 2022 | 137.9341 -> 137.8764 | 133,752 -> **133,752** | 182,013 -> 181,358 | 1,418 -> 1,921 | .30134 -> .30025 | 18.3582 -> 18.3659 |
| 2023 | 137.3958 -> 137.3016 | 138,124 -> **138,124** | 197,090 -> 196,199 | 1,945 -> 2,652 | .31258 -> .31116 | 18.1429 -> 18.1554 |
| 2024 | 138.8408 -> 138.7812 | 131,091 -> **131,091** | 210,174 -> 209,454 | 1,628 -> 2,186 | .32418 -> .32307 | 17.0031 -> 17.0104 |
| **2025 (fold 2 test)** | **137.4636 -> 137.4110** | **131,810 -> 131,810** | **211,763 -> 211,128** | **1,576 -> 2,089** | **.32673 -> .32575** | **17.1441 -> 17.1507** |

**Turnovers are bit-identical in every season** -- the designed consequence of
section 1.1. TOV% rises by 0.007-0.013 pp only because its denominator
(possessions) fell; the numerator did not move.

**Does the truth itself move?** Barely. Possessions per GAME fall by 0.053
(2025), 0.060 (2024), 0.094 (2023), 0.058 (2022) -- i.e. **0.026 per team-game
on fold 2**, 294 possessions out of 768,540 (0.038%). The engine's G1
possession mean is +2.0 against this truth; the truth moves by 0.03 of that, so
**the G1 possession bias is essentially unchanged and this fix does not
explain it**. Points reconcile exactly: total points fall by precisely the
increase in `tech_points` (2025: 812,175 -> 811,662, tech 1,576 -> 2,089,
both 513).

FTA by trip class (the free throws move out of the ordinary buckets and into
`tech_points`, which is where a technical belongs):

| season | FT_trip_shooting FTA | FT_trip_bonus FTA | FTA on other terminals |
|---|---|---|---|
| 2022 | 90,368 -> 89,955 (-413) | 72,029 -> 71,820 (-209) | 19,616 -> 19,583 (-33) |
| 2023 | 97,310 -> 96,667 (-643) | 79,005 -> 78,779 (-226) | 20,775 -> 20,753 (-22) |
| 2024 | 105,792 -> 105,314 (-478) | 81,637 -> 81,427 (-210) | 22,745 -> 22,713 (-32) |
| **2025** | **107,496 -> 107,130 (-366)** | **81,800 -> 81,548 (-252)** | **22,467 -> 22,450 (-17)** |

Rows changed. A game is "changed" if its possession sequence differs at all:
350 / 584 / 372 / **320** games (6.63% / 10.54% / 6.70% / **5.72%**). The
positional row-diff is 5.81% / 9.44% / 5.91% / **5.07%** of possession rows,
but that figure is an **upper bound inflated by index shift**: removing one
possession from a game re-aligns every later row in that game. The rows whose
content actually changed are the ~2 possessions per recovered technical
(346 lookahead hits in 2025, so on the order of 700 rows, ~0.09%), plus the
294 net possessions that disappear.

## 4.1 Blast radius: trips and attempts (fold 2 = 2025 in bold)

| trip class | 2025 trips old -> new | 2025 FTA old -> new |
|---|---|---|
| **technical** | **879 -> 1,219 (+38.7%)** | **1,959 -> 2,612 (+33.3%)** |
| unknown | 1,533 -> 1,202 (-331) | 2,463 -> 1,826 |
| shooting | 49,032 -> 49,030 (-2) | 98,027 -> 98,024 |
| bonus_one_and_one | 32,853 -> 32,847 (-6) | 59,918 -> 59,907 |
| double_bonus | 17,628 -> 17,627 (-1) | 35,179 -> 35,177 |
| and_one | 16,150 -> 16,150 (0) | 16,176 -> 16,176 |
| **total trips** | **118,075 -> 118,075 (0)** | |

2022/2023/2024 technical trips: 902 -> 1,273, 1,355 -> 1,979, 954 -> 1,346.
The total trip count never moves: this is a pure relabelling.

FT-2's own training universe (`attempts_v1_era`, technicals excluded):

| season | attempts | reclassified | % | make rate of the moved attempts | FT-2 universe make rate old -> new |
|---|---:|---:|---:|---:|---|
| 2022 | 183,817 | 666 | 0.362% | 0.7673 | 0.7162 -> 0.7161 |
| 2023 | 199,529 | 925 | 0.464% | 0.7989 | 0.7157 -> 0.7153 |
| 2024 | 212,226 | 740 | 0.349% | 0.7730 | 0.7186 -> 0.7184 |
| **2025** | **213,722** | **653** | **0.306%** | **0.8025** | **0.7212 -> 0.7210** |

This confirms round 1b's flagged contamination directly and quantitatively:
the leaked attempts make **0.77-0.80**, against the 0.72 ordinary base, exactly
the coach-selected-shooter signature. 96-97% of them came out of the `unknown`
class, not out of `shooting` / `bonus`.

---

## 5. Which sub-models consume the changed rows, and what to retrain

Nothing is switched: every model below still reads the old tables.

| sub-model | what it reads | fraction of its rows that move | recommendation |
|---|---|---|---|
| **free_throw, technical rate (round 1b arms)** | `trips_v1_era` `foul_class == "technical"` | **+38.7% of the target count on fold 2** | **RETRAIN / re-grade.** This is not a marginal move; round 1b's own X1/X5/X7 level-calibration table was computed against a target the trip table under-supplied by 27%. |
| **free_throw, FT-2 (make probability)** | `attempts_v1_era`, technicals excluded | 0.31-0.46% of attempts leave the universe | **Probably immaterial** -- the universe make rate moves 0.0002-0.0004 (0.02-0.04 pp on a 0.72 base). Cannot be confirmed without running FT-2's own seed noise floor; a 0.02 pp shift is far below any calibration band the model is graded on, so the honest call is "expected immaterial, unverified". |
| **possession_outcome** | possession / chance rows, terminal-event classes | 0.038% of possessions removed; `FT_trip_shooting` -0.36%, `FT_trip_bonus` -0.29% of their counts on fold 2 | **Immaterial, expected.** Per-100-possession terminal shares move in the fourth decimal. Cannot be *proved* under its noise floor without a re-run; recommend re-running only when the model is next retrained for another reason. |
| **clock** | possession durations and counts | possessions/game -0.053 (0.04%) on fold 2 | **Immaterial.** A pace latent fitted on 137.46 vs 137.41 possessions/game cannot move measurably. Unverified. |
| **usage / attribution** | the event stream's free-throw rows and `foul_class` | 0.31% of FT attempts change class on fold 2 | **Check, do not assume.** If the usage build allocates technical free throws to a shooter's usage the way it allocates ordinary ones, 653 attempts a season move; whether that is inside its noise floor cannot be told without running it. |
| **late_game** | possessions in the closing minutes | 5.7% of games touched somewhere, but the technical moments are dead-ball and spread across the game | **Unknown without running it.** The touched-game share is high enough that "immaterial" should not be asserted from these numbers alone. |

Ranking: the free-throw technical-rate lane is the only one where the change is
large relative to what it models. Everything else is a fourth-decimal move that
is *expected* to sit under its own noise floor, and this doc says so as an
expectation, not as a measured pass.

---

## 6. Sibling artifacts written

All four seasons were built, not only fold 2's -- confirmed on disk at
20:58:44 EDT: `possessions_v3/` holds `possessions_{2022,2023,2024,2025}` and
`chances_{2022,2023,2024,2025}`, eight files, 68 MB, written 20:13-20:16.

```
data/processed/possessions_v3/possessions_{2022..2025}.parquet   (gitignored, >20MB)
data/processed/possessions_v3/chances_{2022..2025}.parquet       (gitignored, >20MB)
data/processed/possessions_v3/build_report.json
data/processed/models/free_throw/trips_v1_era_techfix.parquet
data/processed/models/free_throw/attempts_v1_era_techfix.parquet
results/event_layer_technical/blast_radius.json                  (gitignored)
```

`possessions_v3/` is over 20 MB and is gitignored per `CLAUDE.md`'s data rule;
the PM syncs it with an `hf_sync_data.py` bulk key. Nothing existing was
overwritten; no worker's current read target moved.

Rebuild command (idempotent, ~224 s for all four seasons, one process):

```
.venv/Scripts/python.exe scripts/build_possessions_techfix_v3.py
.venv/Scripts/python.exe scripts/diag_technical_blast_radius_v1.py
```

---

## 7. The full test suite: one real failure, caused by this change, fixed

`.venv/Scripts/python.exe -m pytest tests/ -q -x --durations=10`, run to
completion in the foreground: **started 20:25:53 EDT, ended 20:58:14 EDT,
1,940.35 s = 32 min 20 s**, result **1 failed, 385 passed** (`-x` stops at the
first failure, so the ~157 tests after that point did not run).

```
FAILED tests/test_possession_outcome.py::test_possessions_version_resolves_and_refuses_an_unknown_label
```

**The failure was this change's fault, and it was a genuine one.** That test
asserted `possessions_dir("v3")` raises `KeyError` -- it used `"v3"` as its
example of an unrecognised version label. Registering `v3` as a real build
took the placeholder out from under it. The test's *intent* -- an unknown
label must raise rather than silently fall back and read the wrong table -- is
untouched and still worth having, so the assertion now uses a label no build
will ever claim (`"no_such_version"`) and gains a positive assertion that
`v3` resolves to `possessions_v3`. Re-run at 20:58:43: the amended test, its
`DEFAULT_POSSESSION_VERSION == "v1"` companion and all 19 new tests --
**21 passed in 0.52 s**.

This is worth recording as more than a bookkeeping note: the suite caught a
real consequence of adding a version that no amount of reading the diff would
have surfaced, and the `-x` flag then hid the rest of the suite behind it.

**Why the suite is slow, and why it is not hanging.** `--durations=10` says the
cost is the clock-model fixtures on a loaded machine, not a hang:

```
1103.86s setup  tests/test_clock.py::test_no_fitted_arm_ever_uses_a_banned_column
 229.34s call   tests/test_clock.py::test_arm_fits_are_reproducible
  93.06s call   tests/test_possession_outcome.py::test_arm_fits_are_deterministic[lgbm]
```

An earlier attempt (launched 20:02:46, stopped by this lane at 20:24:59 after
22 min 13 s) appeared to emit nothing; that was an artefact of piping pytest
through `tail`, which buffers until the process exits, and NOT evidence of a
hang. Reported here because the first version of this section drew the wrong
conclusion from it.

**PARTIAL, and precisely bounded**: the ~157 tests that sit after the
`-x` stop point have not been run since the fix. Resume with the full suite,
no `-x`, allowing ~35 min on a loaded machine:

```
.venv/Scripts/python.exe -m pytest tests/ -q --durations=10
```

Nothing else in this change can reach a test that does not pass
`tech_lookahead=True`: every entry point defaults to `False`, and the two
production tables were reproduced bit-identically without the keyword
(section 3). The one thing that *could* reach other tests was the new entry in
`POSSESSION_VERSIONS`, and that is exactly what failed and is now fixed.

---

## 8. Learnings paragraph (for the PM to lift into `docs/LEARNINGS.md`)

**A one-row lookahead is an assumption about a vendor's feed, and it should be
written down as one.** `possessions.py::_handle_technical` and
`event_stream.py::_attach_trips` both encoded "the free throw is the adjacent
row" -- one forward, one backward -- and CBBD inserts an administrative row
before it on 11-26% of technical moments, so both silently mis-tagged
~1,200-1,900 attempts a season for four seasons. It survived because every
check that could have caught it was an aggregate: total FTA reconciled (the
attempts were still counted, just in the wrong bucket), total points reconciled
(the same points landed on a possession instead of `tech_points`), and the
turnover count was right all along. It took a *class-level* comparison against
an independently scanned target to surface it -- exactly the multi-level
evidence rule. Two second-order lessons. First, **the second source settles
what to fix, not just whether something is broken**: the natural reflex was to
delete the inserted `Lost Ball Turnover` as a phantom, and the box says it is a
real turnover (+0.007 against +1.00 expected for an artefact), so deleting it
would have traded a free-throw defect for a turnover defect. Second, **the same
rule implemented twice drifts twice**: the fix imports one
`technical_ft_index` into both builders rather than repairing each in place,
and the adjacent finding -- that the ball stays with the technical'd team on
65.8% of these moments, so the possession split may itself be wrong -- is
recorded as its own open defect instead of being folded into this patch.
