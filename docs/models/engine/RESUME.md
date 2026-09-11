# ENGINE v1 -- resume state (2026-09-11, overnight session)

Supersedes the 2026-09-10 evening version, which is kept below in full because
every diagnosis in it is still live. Gate read:
`docs/tests/engine_v1_gates_F2_2025_s200_2026-09-11.md` (PROVISIONAL, 50 of 200
seeds). Architecture: `docs/models/engine/model.md`.

---

## 0. What changed on 2026-09-11 (the rewiring)

**Every sub-model that has adopted a winner or a scheme is now served BY
DEFAULT.** Until tonight four were wired but not defaulted, so a run that did
not export the right environment variable silently served a superseded arm.

| flag | now | authority |
|---|---|---|
| `ENGINE_INPUTS_VERSION` | `v2` | required by `round4_B1` |
| `ENGINE_EVENT` | `round2_s1` | possession_outcome round 2 |
| `ENGINE_FG_MAKE` | `round4_B1` | fg_make s20.6 |
| `ENGINE_REBOUND` | `s1_weekly` (23 artifacts) | rebound S1 confirmation |
| `ENGINE_FREE_THROW` | `s1_conf_aligned` (29) | free_throw S1 confirmation |
| `ENGINE_ROTATION_SCHEME` | `s1` (6) | rotation round 3b s9.4 |
| `ENGINE_CLOCK` | `v3c_srfloor_P3_s1` | the clock lane owns it (L31) |

1. **Engine inputs v2** (`build_engine_inputs.py --version v2`, 8.6 s) --
   versioned siblings `*_F2_2025_v2.*`, composed from v1 so byte-identity of
   every non-shooter array is a property of the construction and is asserted,
   not claimed. Three things change: the 15 fg_make shooter slot columns
   (re-keyed on `shot_shooter_id`), 10 round-4 columns appended, and
   `usage_rate` rebuilt off the usage round-2 panel with every (prior, m)
   unchanged. `EngineInputs.load` resolves `ENGINE_INPUTS_VERSION`; an
   explicitly named missing version RAISES.
2. **`run_meta.json` carries `max_train_date` per family per artifact**, and
   `grade_market_games_v2.py` now asserts `max_train_date < tipoff` on **9
   artifact schedules across 6 families** (5710/5710 rows each) instead of 2
   paths. `usage` reports static and is not fabricated a date.
3. **`engine_commit` + a per-file dirty-tree list** in `run_meta.json` (commit
   `0cfd68a`), so a paired design can prove it did not straddle an engine
   change instead of bisecting for it.
4. **Parity digest v3**: `docs/ops/parity_reference_windows_v3.json`, sha256
   `09260d82...`, git `493a818eeb`, reproducible across worker/block splits.
   NOT comparable to v2 -- four flags moved, three are new.
5. **20/20 engine tests**, five of them new.

## 1. The gate read (PROVISIONAL, 50 seeds)

`results/engine_v0/F2_2025_s200_rewire1`: 5,710 games x **50 complete seeds of
200 requested**, 41.6M possessions, 500 of 1,920 blocks, 5,368 poss/s on 12
local workers (447/core, contended). A full 200-seed slate needs ~8.3 h at
that rate.

**PASS 0 / FAIL 8 / NEEDS-INSTRUMENTATION 1 at the gate level -- and the
engine is much better than v0, which was 0/6/3.** G4 and G8 moved from
unmeasured to measured-and-failing, which is progress, not regression. At the
LINE level v1 passes 7 lines v0 passed 0 of.

| | engine v1 | engine v0 | actual |
|---|---:|---:|---:|
| margin SD ratio | **1.0265 PASS** | 1.5977 | 1.0 |
| margin bias | **-0.210 PASS** | -1.819 | 0 |
| total bias | -0.888 PASS* | +0.014 | 0 |
| calibration slope | 0.891 | 0.181 | 1.0 |
| possessions/game | 69.880 (+2.00) | 72.118 (+4.24) | 67.875 |
| PPP | 1.0350 | 1.0090 | 1.0730 |
| home/away score corr | +0.027 | -0.615 | +0.253 |
| OT rate | 3.00% | 1.21% | 5.57% |
| per-team-quintile slope | **0.843**, 4/4 | 0.555 | 1.0 |
| market margin MAE / corr | **9.29 / 0.908** | 16.03 / 0.329 | close 8.74 |

*Total bias passes on TWO CANCELLING ERRORS (OREB% -1.58 pp and FTA/FGA -1.24
pp against possessions +2.00) and must be read as a fail.

## 2. The three things to fix next, in size order

1. **Possessions +2.0 while every per-possession rate lands within 1.6 pp.**
   The clock owns it; round 4 is already on L31's duration-level shortfall.
2. **Home/away score correlation +0.027 vs +0.253, total SD ratio 0.79.** Two
   views of one defect: simulated opponents do not co-move. Nothing in the
   cascade couples the two teams' scoring within a game except the shared
   possession count.
3. **The player layer: 8.79 players used vs 9.80, minutes SD ratio 1.23.** L30
   named the cause; rotation round 5 pre-registers the wave model against it.

## 3. Still open from this session

- **The 200-seed run itself.** 50 seeds is a quarter of the floor.
- **The seed-offset noise floor** (`--seed-offset 1000`) -- PENDING, never
  startable tonight.
- **`provisional_fg` still reads True**: the round-4 B1 joblibs carry
  `adopted: false` (written before s20.6 decided). The fg_make lane should
  re-export; the engine reports the artifact's own flag rather than overriding.
- **fg_make B1 is monotone 2/4 on the jumper and three ON THE ENGINE'S OWN
  roster-slot population** (4/4 on the attempt-weighted design population).
  Not a Decision-8 failure -- a different cell of the same model. New, logged.
- **Rotation S1 is half-scope**: it reaches the scheduler, the Dirichlet
  concentrations and the tilt tables; the as-of prior construction still uses
  the static fit (`rotation_s1_scope`). Inputs-v3 item.
- **Per-class fg_make serving** (s20.8 item 2): `FGA_rim`'s own winner is BR,
  worth 64.6 floors, and the adapter resolves one round directory for all
  three classes.
- The run is **not reproducible from a commit** (uncommitted clock lane file in
  the tree at launch): `results/engine_v0_F2_2025_s200_rewire1_code_provenance.json`.

---
---

# (superseded) ENGINE v0 -- resume state (2026-09-10, evening session)

Supersedes the 16:35 version. Architecture: `docs/models/engine/model.md`.
Gate read: `docs/tests/engine_v0_F2_2026-09-10.md` (read the **ADDENDUM** at the
end; everything before it was written when the run had produced no output).
Seed requirements: `docs/tests/engine_seed_count_2026-09-10.md`.

---

## 0. What changed this session

1. **All tests run and pass: 13/13**, in three flag configurations
   (`reference`; `round2_s1`; `round2_s1` + `reference_empirical`). The five
   tests that had never been executed -- determinism, batch-independence,
   five-on-floor, period/OT transitions, allocation spread -- **found no engine
   bug**. No test was weakened. Three new tests cover dated artifacts.
2. **`winner_FGA_3.joblib` is no longer stale.** The `fg_make` worker
   re-exported it at 15:52, after Decision 8 at 15:47. It holds
   lgbm / C_plus_state and is **bit-identical** to the engine's own Decision-8
   refit (max |diff| 0.0 over 4,000 probes, same 25 features in the same order,
   both 400 trees; shooter slope 0.175 against the stale model's 0.0086).
   `ENGINE_FG3=decision8` and `artifact` now agree and the conflict is closed.
3. **The possession-outcome ROUND-2 WINNERS ARE WIRED** behind
   `ENGINE_EVENT=round2_s1` (lgbm+S1 first, cascade+S1 cont).
   `provisional_event` is **False** under that flag; `reference` still works and
   still sets it True.
4. **S1 monthly selection is implemented generically** in
   `src/cbb_sim/engine/manifest.py`, because S1 is the standing default for
   EVERY sub-model (L21), not an event-layer special case. Section 2.
5. **The seed-count study is done** and invalidates most of today's 5-seed
   market numbers. Section 5.
6. **`ENGINE_CLOCK=reference_empirical` now actually works** -- it previously
   failed at adapter load -- so the binning error is measured for the first
   time. Section 6.
7. **A first-order defect was found and localised to a named sub-model.**
   Section 4. It displaces the clock as "the first thing to fix".

---

## 1. Current gate read (PROVISIONAL, 5 seeds)

`results/engine_v0/F2_2025_s5_r2event`: 5,710 games x 5 seeds, 4,117,924
possessions, 669 s on 8 workers, `partial=False`.

**PASS 2, FAIL 6, NEEDS-INSTRUMENTATION 4.** Headlines: possessions/game
**72.118** vs 67.875 (**+4.24**), PPP **1.0090** vs 1.0730, total bias
**+0.014** (Control -2.826; better than the close), margin SD ratio 1.598,
home/away score correlation **-0.615** vs +0.253, OT rate 1.21% vs 5.57%,
per-team-quintile monotone 4/4 but **slope ratio 0.555**, every
per-possession-type rate within 1.4 pp. Full tables and the G10 scorecard are in
the test doc's addendum.

G3, G4 and G8 remain NEEDS-INSTRUMENTATION because of the **truth side**, not
the engine: G3's rim share needs a pbp shot-location truth table, G4's eFG%
needs make counts the results contract does not carry (attempts only), G8 needs
a player truth table. The engine writes all seven box pairs and a 452,618-row
`players.parquet`. The earlier claim that it makes these gradeable is withdrawn.

---

## 2. The dated-artifact mechanism (S1 for every sub-model)

`src/cbb_sim/engine/manifest.py`. One rule in one place that every sub-model
passes through, so seven adapters cannot implement it seven ways.

**Manifest format (JSON):**

    {
      "model": "possession_outcome",     # names the run_meta flag
      "scheme": "S1",                    # "S1" | "static"
      "fold": "F2", "season": 2025,
      "key": "first",                    # OPTIONAL: population / class / target
      "artifacts": [
        {"refit_date": "2024-11-01",           # required, ISO date
         "path": "first_2024-11-01.joblib",    # relative to the manifest's dir
         "max_train_date": "2024-04-08",       # REQUIRED (see below)
         "n_train": 1908534}                   # optional provenance
      ]
    }

Entries need not be sorted. A model with several keys writes one manifest per
key, or one file whose top level maps key -> that object.

**Selection.** Per game, the artifact with the latest `refit_date` strictly
before tipoff. `max_train_date` is **required, not optional**: the rule that
actually binds is `max_train_date < game_date`, asserted at load for every game
(`CLAUDE.md`: created_at < tipoff, enforced in code). A manifest omitting it is
rejected unless the caller passes `require_max_train_date=False` and says why.

**A static model is a manifest of length one.** `ArtifactManifest.static(...)`
sets `is_static`, written to `run_meta.json` as `scheme_static_<model>=True`.
Today: `possession_outcome` **False**; clock, fg_make, free_throw, rebound,
usage and rotation all **True** -- i.e. every other sub-model is still S0 and
the flag says so. That is how a reader tells a deliberately-static model from
one silently serving a stale S1 artifact, which is the failure the change ledger
warns about ("a stale artifact degrades toward S0, which fails the gate").

**Two defects the shared rule caught on its first run**, both missed by the
purpose-built check inside `build_engine_event_round2.py`:

- *A timezone leak.* Comparing a refit date against `tipoff_utc` put every
  late-evening game in the last days of a month into the NEXT month's artifact
  (a 19:50 US tip on 31 March carries `tipoff_utc` of 1 April, and the 1 April
  refit trained through 31 March). Selection now happens on the same calendar
  the schedule was built on.
- *An off-by-one-month over-correction.* A strict `refit_date < game_date` would
  serve November's model to 1 December games. S1's own partition scores a game
  ON the first with THAT month's refit (fitted strictly before the first), so
  the engine now reproduces the partition the bake-off actually scored --
  otherwise the measured calibration would not be the calibration the engine
  gets.

**Tests** (`tests/test_engine.py`):
`test_a_game_never_gets_an_artifact_refit_at_or_after_its_own_month` (per-game
refit-date and max-train-date checks, the month-level form, and an assertion
that all six artifacts are actually used so a collapsed schedule cannot pass as
S1); `test_a_static_manifest_is_a_manifest_of_length_one_and_says_so`;
`test_a_manifest_whose_training_window_reaches_the_game_is_rejected` (the guard
must fail on a leaky schedule, not merely pass on a clean one).

**For the model workers:** produce dated artifacts under your own versioned
directory plus a manifest in the format above and hand the PM the path. Do not
write into `data/processed/models/engine/`.

---

## 3. Artifacts, flags, commands

`data/processed/models/engine/`:

    games_F2_2025.parquet / arrays_F2_2025.npz / names_F2_2025.json
    rebound_F2.joblib, free_throw_F2.joblib        adopted winners, refit here
    fg_make_FGA_3_decision8_F2.joblib              now identical to the fg_make winner
    event_round2_s1_F2_2025/                       46.9 MB, 12 joblibs + team_block.npz + index.json

`event_round2_s1_F2_2025/` is built by `scripts/build_engine_event_round2.py`
(~8 min) and holds six monthly refits per population (2024-11-01 .. 2025-04-01)
plus a **round-2 team-form block**. That block is necessary, not incidental:
round 2 changed the event layer, the style-rate source (`first_chance`) and the
universe (`pbp_complete`), so matched team-game to team-game on 2025 the same
feature NAMES correlate only 0.956-0.978 with round 1's and `off_3pa_c`'s own SD
shrinks 5.25 -> 4.65. Serving round-2 arms round-1 columns would be a silent
train/serve skew. `arrays_F2_2025.npz` is NOT modified, so `reference` reads
exactly what it read before. 530 of 11,420 team-games are uncovered by the
round-2 design; 529 take that team's most recent earlier covered game
(`merge_asof` backward), 1 falls to the league mean.

**Per PM decision 2026-09-10**, `data/processed/models/engine/event_round2_s1_*/`
is now in `.gitignore` and syncs to the private HF dataset via
`scripts/hf_sync_data.py`. On a fresh machine, pull it or rebuild it.

| flag | values | default | effect |
|---|---|---|---|
| `ENGINE_EVENT` | `reference` \| `round2_s1` | `reference` | `round2_s1` runs the adopted round-2 winners and clears `provisional_event` |
| `ENGINE_CLOCK` | `reference` \| `reference_empirical` | `reference` | provisional either way; `reference_empirical` is the binned lookup table and now loads |
| `ENGINE_ROTATION` | `reference` | `reference` | provisional |
| `ENGINE_FG3` | `decision8` \| `artifact` | `decision8` | now equivalent |

`run_meta.json` also carries `scheme_static_<model>` for all seven sub-models.
`CBB_UNSEAL=1` is required for season 2026 and stays unset.

    # build the round-2 S1 event artifacts (once per fold/season, ~8 min)
    .venv/Scripts/python.exe scripts/build_engine_event_round2.py --fold F2 --season 2025

    # the run (5 seeds ~11 min on 8 workers; 200 seeds ~2.6 h on 20 cores)
    ENGINE_EVENT=round2_s1 .venv/Scripts/python.exe scripts/run_engine.py \
        --fold F2 --season 2025 --seeds 200 --workers 8 \
        --games-per-block 60 --seeds-per-block 25 --tag F2_2025_s200_r2event

    # grade
    .venv/Scripts/python.exe scripts/eval_gates.py             --results results/engine_v0/<tag> --season 2025
    .venv/Scripts/python.exe scripts/grade_market_games.py     --results results/engine_v0/<tag> --season 2025
    .venv/Scripts/python.exe scripts/grade_market_props.py     --results results/engine_v0/<tag> --season 2025
    .venv/Scripts/python.exe scripts/diag_engine_multilevel.py --results results/engine_v0/<tag> --season 2025

    # the studies
    .venv/Scripts/python.exe scripts/exp_engine_seed_count.py  --fold F2 --season 2025 --games 300 --seeds 200 --workers 8
    .venv/Scripts/python.exe scripts/diag_engine_throughput.py --fold F2 --season 2025 --games 300 --seeds 10 --workers 8

    # tests
    .venv/Scripts/python.exe -m pytest tests/test_engine.py -q

---

## 4. THE FIRST THING TO FIX -- `score_diff` feedback (was: the clock)

The clock is no longer the largest defect. A per-sub-model ablation (60 games x
20 seeds, paired streams, the `score_diff` and
`score_diff x seconds_remaining` columns redirected to an always-zero column for
one adapter at a time; the harness reproduces the baseline exactly):

| arm | margin SD | total SD | per-team pts SD | corr(home,away) | poss/game |
|---|---|---|---|---|---|
| baseline | 34.61 | 16.36 | 19.77 | **-0.6363** | 71.61 |
| clock off | 33.01 | 14.55 | 18.37 | -0.6750 | **67.84** |
| event off | 39.36 | 16.46 | 22.09 | -0.7041 | 71.48 |
| **fg_make off** | **11.35** | 17.79 | **10.65** | **+0.4214** | 71.82 |
| all off | 14.42 | 15.30 | 10.64 | +0.0593 | 68.38 |
| **actual 2025** | **14.63** | **18.95** | ~11 | **+0.2532** | **67.88** |

**Two separate defects with two separate owners.**

1. **`fg_make` owns the dispersion blow-up.** Its `score_diff` alone triples
   margin variance and flips the home/away correlation from +0.42 to -0.64.
   The event model is *damping* the loop, not driving it.
2. **`clock` owns the possession inflation.** Its `score_diff` alone accounts
   for the ENTIRE +4.24 G1 miss (71.61 -> 67.84 against an actual 67.88). The
   clock bake-off's L20 horn-truncation root cause is still real, but it is not
   what G1 is measuring here.

**Diagnosis.** `score_diff` is a CONSEQUENCE of the outcome being simulated,
used as a DRIVER of it. In training it is exogenous and carries team-quality
information, so a shot-make model that sees it learns a coefficient that is
partly a selection effect. In simulation that closes a loop: a random early lead
is read as evidence of a strong offence, which scores more, which widens the
lead. Team strength is already carried by the ratings and shooter features, so
the engine double-counts it, with feedback.

**What was NOT done.** No shrinkage, clipping, rescaling or recalibration
(`CLAUDE.md`, "no hand tuning on engine output"). The ablation names the
responsible sub-model; it is not a fix. Deletion is not the fix either: the "all
off" row costs 12 points of game total (133.24 vs an actual 145.51), because the
state block genuinely carries signal -- it cleared its own noise floor by 30-140x
at L3.

**What needs pre-registering**, for the PM: a bake-off over parametrisations of
game state that cannot re-encode team strength -- `score_diff` residualised
against the pregame rating difference and elapsed fraction; a "surprise" term
(`score_diff` minus its expectation given the matchup and time remaining); or
dropping it from the scoring-stage models while keeping it in the clock. This
blocks G5, G7 and every market number, and it is larger than anything currently
in the clock round-3 queue.

---

## 5. Seeds: what may and may not be read

`docs/tests/engine_seed_count_2026-09-10.md` (300 games x 200 seeds, 8.6M
possessions, paired by construction).

| seeds | per-game margin SE | per-game win-prob SE | slate margin SE |
|---|---|---|---|
| 5 | **14.85 pt** | **21.6 pp** | 0.772 |
| 25 | 6.36 | 9.4 pp | 0.272 |
| 100 | 3.03 | 4.2 pp | 0.112 |

- **Game-level win-prob SE < 1 pp needs ~2,100 seeds** (4,300 for the p90 game).
  That is mostly a counting requirement, not an engine one: SE is
  `sqrt(p(1-p)/k) <= 0.5/sqrt(k)`, so no simulator reaches 1 pp under ~1,900.
- **Minimum for the G1-G9 gate report: 200 seeds** (slate SE ~0.08 pt, an order
  of magnitude inside every tolerance).
- **No ROI, Brier or per-game market number may be quoted below ~2,000 seeds.**
  Today's G10 margin MAE of 16.03 is mostly the 14.85-point MC error, and its
  calibration deciles show probabilities of exactly 0.2/0.4/0.6/0.8/1.0 -- the
  5-seed quantisation. G5's PIT is likewise a comb at 5 seeds and unreadable.
- These requirements are inflated ~6x by the section-4 defect (seed count scales
  with the square of the within-game SD) and should be **re-measured, not
  rescaled**, after it is fixed.

---

## 6. Throughput and the lookup table

Contended measurement: 20 logical cores, 76-80% total CPU busy, 12 Python
processes (four other model workers active).

| arm | poss/s/core |
|---|---|
| `ENGINE_CLOCK=reference` (live 9-quantile LightGBM) | 873.7 |
| `ENGINE_CLOCK=reference_empirical` (binned pmf lookup) | 1,111.6 |

Independent reads: 817.4 poss/s/core (seed study, 8 workers, 8.6M poss) and 769
(full-slate run, 8 workers). Honest contended rate: **820-880**.

**The lookup table buys 1.27x, not the ~5x the plan assumed.** "The clock is 85%
of model cost" came from isolated per-model benchmarks; in the real loop the
cost is dominated by state bookkeeping, matrix assembly and the other six
adapters. 5,710 x 200 seeds = 1.65e8 possessions: **2.6 h live vs 2.05 h binned
on 20 cores**, so the export only just reaches the 2-hour target and no further
clock optimisation can help.

**The binning error, measured for the first time** (paired, same games/seeds):
possessions/game **-0.980**, per-game possessions **MAE 1.652** (SD 1.847),
total -2.320, PPP -0.0024. **Not small** -- the MAE alone exceeds G1's +/-1.0
tolerance. It is an upper bound on a faithful export's error, because the
empirical arm is a different model rather than the quantile arm binned.

`reference_empirical` previously failed at load, for two reasons now fixed: it
declares `chance_number_at_start` (identically 1.0 at a possession's start; the
engine now supplies that constant), and `EmpiricalArm._codes` reads raw
`prev_end` (string) and `season` columns absent from the arm's own feature list
(`ClockAdapter` now rebuilds both).

---

## 7. Still unfinished

1. **The 200-seed full-season run** (~2.6 h on 20 cores). Today's read is 5
   seeds.
2. **The paired-seed noise floor** (`--seed-offset 1000`).
3. **The `score_diff` re-parametrisation bake-off** (section 4) -- blocks G5,
   G7 and every market number.
4. **A faithful binned export of the quantile arm**, and its own binning error
   separated from the model difference (section 6).
5. **`ast` has no model.** `players.parquet` writes 0; `ast_is_placeholder: true`.
6. **Truth-side instrumentation for G3, G4 and G8** (section 1) -- a pbp
   shot-location truth table, make counts in the results contract, a player
   truth table.
7. **The change ledger** has no engine-v0 row; the PM owns it.
