# Possession-outcome round 6: the team-foul accrual law and the conditional bonus trip -- OFFLINE tables (2026-09-18)

Lane: foul accrual / bonus law. **OFFLINE ONLY. NOTHING IS ADOPTED, NO DEFAULT IS
CHANGED, NO ENGINE FILE IS TOUCHED.** The paired closed loop of section 13.7 rule 7
was NOT run (the machine shuts down tonight); section 6 below leaves the exact
resume command.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 13
(commit 3b75c3c) as amended by **section 15 (AMENDMENT A, commit ac18555,
written and pushed BEFORE any fitting)**. Motivating evidence:
`docs/tests/g4_oreb_fta_diagnostic_2026-09-18.md`.

Code, all new, all versioned siblings:

    scripts/build_foul_accrual_design_v1.py    instrumented replay -> the target
    scripts/train_foul_accrual_v1.py           Block F arms, both folds, two seeds
    scripts/train_foul_bonus_cond_v1.py        Block T arms + the technical anti-join
    scripts/grade_foul_accrual_v1.py           ONE blind grading pass over every arm

    data/processed/models/possession_outcome/round6/   every artifact

---

## 0. The target, and why it had to be rebuilt

Section 15.1. The served engine accrues non-shooting team fouls from one scalar,
`silent_foul_per_possession = 0.123346`, drawn once per possession and always
charged to the DEFENCE. Two facts change what the honest target is:

1. **The accrual has two channels and the engine has one.** Fouls charged to a
   team while it is on OFFENCE are NCAA team fouls and the segmenter counts them.
   Measured over 2022-2025: **0.0222 offensive non-trip fouls per possession**,
   against 0.0769 defensive ones. The served constant is a pooled
   `(all fouls - trip fouls)/possession` that re-attributes every one of them to
   the defending team. **The engine has no offensive-foul mechanism at all.**
2. **`def_team_fouls` cannot be differenced.** It is read at possession OPEN and
   `_handle_ft_trip -> _ensure` opens the possession AT the foul whenever the
   previous one has closed. A possession ending in `FT_trip_bonus` shows a mean
   defence-side increment of **0.158** over its own window against **1.023** for
   an and-one, so most bonus-trip fouls fall in the previous window -- where the
   fouling team is the OFFENCE. Differencing yields a *negative* silent rate
   (-0.081/possession).

The target is therefore built by **replaying `_GameMachine` with `_handle_foul` /
`_handle_technical` wrapped in this lane's own process** (`src/cbb_sim/` is
unchanged; the `diag_late_game_tap_v1.py` pattern). Every personal foul is logged
with the possession that was open when it happened; a foul with no possession
open is attributed to the next possession opened.

| season | games | possessions | fouls with no open possession | dropped unresolved |
|---|---:|---:|---:|---:|
| 2022 | 5,282 | 728,568 | 138,132 | 30 |
| 2023 | 5,541 | 761,310 | 146,156 | 32 |
| 2024 | 5,553 | 770,983 | 144,080 | 22 |
| 2025 | 5,593 | 768,834 | 143,716 | 42 |

3,029,695 possessions, 26.4 s wall clock on 4 workers, **126 fouls (<0.02%)
unattributable and dropped**, counted not imputed. Per-possession means:
`def_silent` 0.07692, `def_trip` 0.12285, `off_silent` 0.02220, `off_trip`
0.00125. Total fouls 0.2233/possession -> 15.4 per team-game, consistent with the
box.

**Technicals never increment `team_fouls`** (`_handle_technical`), so the
technical labelling defect cannot touch Block F. It does touch Block T; see
section 4.

---

## 1. Block F -- the accrual law. Fold 2 SELECTS (2024-25 test, 681,101 fit-window possessions)

Fit and primary mask `period <= 2 and seconds_remaining > 120` (13.5 breakdown 8).
`floor` = |seed 0 - seed 7| on the same primary; the deterministic arms have an
*exactly* zero seed sensitivity, so the **applied floor is round 4b's 0.000804**
and the measured 4.3e-6 on the one stochastic arm is reported beside it.

| arm | log loss | measured floor | calib gap pp | minute-curve max abs gap pp | team SD ratio | prior-quintile slope |
|---|---:|---:|---:|---:|---:|---:|
| `F0` reference (one fitted constant) | 0.2739820 | 0.0 | -0.081 | 4.876 | **0.000** | **0.000** |
| `F0_served` (the literal 0.123346) | 0.2847055 | 0.0 | **+4.530** | **9.487** | 0.000 | 0.000 |
| `F1` per half | 0.2738989 | 0.0 | -0.081 | 5.259 | 0.005 | 0.002 |
| `F2` clock x site cells | 0.2696477 | 0.0 | -0.080 | 0.351 | 0.033 | 0.007 |
| `F2m` + margin cell | 0.2695107 | 0.0 | -0.077 | 0.387 | 0.142 | 0.054 |
| `F3` GLM state | 0.2717722 | 0.0 | -0.082 | 3.229 | 0.197 | 0.093 |
| `F3b` + bonus state | 0.2604190 | 0.0 | -0.113 | 1.381 | 0.371 | 0.137 |
| `F3c` + as-of rates | 0.2604066 | 0.0 | -0.114 | 1.376 | 0.430 | 0.209 |
| `F3d` prior-season carry | 0.2604121 | 0.0 | -0.141 | 1.407 | 0.467 | 0.269 |
| `D9a` opponent-adjusted | 0.2604072 | 0.0 | -0.111 | 1.389 | 0.432 | 0.210 |
| `D9b` + conference flag | 0.2604068 | 0.0 | -0.113 | 1.374 | 0.434 | 0.211 |
| `D9c` conference-aligned fit | 0.2603935 | 0.0 | -0.111 | 1.372 | 0.450 | 0.222 |
| `H1` site interactions | 0.2603751 | 0.0 | -0.111 | 1.369 | 0.429 | 0.212 |
| **`F5` GBM** | **0.2448577** | 0.0000043 | **-0.117** | **0.293** | **0.589** | **0.252** |

**`F5` is the only arm that clears the reference by a margin that means anything**:
-0.029124 = **36 applied floors** (6,773 measured floors of its own seed spread).
Fold 1 confirms with the same ordering: `F5` 0.2428148 against `F0` 0.2713440,
`F3c` 0.2582059, `F2` 0.2670832.

**Eligibility (13.7 rules 4-5, 15.6).**

* Minute curve (the offline DRIVER of the occupancy curve): `F5` 0.293 pp max
  absolute per-bucket gap against `F0` 4.876 and the served constant 9.487.
  Signed, `F5` is -0.21 / -0.24 / -0.15 / -0.10 / -0.29 / +0.01 / +0.09 / +0.03 pp
  over the eight buckets -- **no bucket is made worse to fix another**, which is
  the failure mode rule 4 exists to catch.
* Prior-season quintile slope: `F5` 0.252, monotone 3/4, on **362 powered teams
  (>= 300 fit-window possessions each, ~72 per quintile)**. Every constant arm
  scores exactly 0.000 by construction. The slope is still weak in absolute terms
  and the round says so: **no arm here is matchup-responsive at anything
  approaching 1.0.**
* Spread of team estimates (15.5): realised team SD 0.868 pp. `F5` predicts
  0.511 pp, **ratio 0.589**, correlation +0.329. No arm compresses spread relative
  to the reference -- the reference has **zero** spread -- so the round-5 failure
  mode does not arise here. It does cap what the arm can buy: 41% of the
  between-team spread is still unmodelled.

**Held-out final 2:00 of regulation + overtime** (87,733 possessions, never
fitted): `F5` calibration gap **+0.027 pp** against `F0` +5.016 and the served
constant **+9.626**. The law generalises into the late window without being
trained there. That window remains the late-game lane's.

---

## 2. The mechanism: the silent-foul rate is a function of the bonus state, and the served constant ignores it

Gap = predicted - actual, pp, fold 2, all cells powered:

| defence's team fouls | n | actual rate pp | `F0_served` gap | `F0` gap | `F5` gap |
|---|---:|---:|---:|---:|---:|
| 0-3 | 376,065 | 7.92 | **+4.41** | -0.20 | -0.20 |
| 4-6 | 190,225 | 11.78 | +0.56 | **-4.05** | -0.13 |
| 7-9 | 91,796 | 1.01 | **+11.32** | **+6.71** | +0.15 |
| 10+ | 23,015 | 0.17 | **+12.16** | **+7.55** | +0.22 |

**The rate rises from 7.9% to 11.8% as the defence approaches the bonus and then
collapses to 1.0% once it is in it** -- because in the bonus a personal foul
awards free throws and stops being silent by definition. The served law fires a
12.3% Bernoulli in all four states. That is the structural defect, and it is why
`F3b` (the arm that first sees the bonus state) buys 0.0114 of the 0.0136 total
log-loss gain and why the flat law cannot produce the diagnostic's by-minute sign
flip.

**An unresolved contradiction, stated rather than smoothed.** Offline, the served
constant is **58% HIGHER** than the measured defensive non-trip rate (0.1233 vs
0.0780) and over-predicts in three of the four foul-count cells; the closed-loop
diagnostic measures the engine **too rarely** in the bonus at minutes 30-37
(-15.27 pp). Both cannot be the whole story. The offline evidence confirms the
SHAPE the diagnostic found (a flat law against a strongly state-dependent truth)
and does **not** reconcile the LEVEL. **Reconciling it needs the paired closed
loop, which this round did not run**, and until it is run no share of the
-1.384 pp bonus-trip channel may be claimed as closed by Block F.

Multi-level, fold 2, `F5` vs `F0` calibration gaps in pp: site away/home/neutral
-0.11/-0.11/-0.17 (`F0` -0.06/-0.06/-0.22); conference/non-conference
-0.14/-0.07; Nov-Dec/Jan/Feb-Apr -0.11/-0.02/-0.16. **Every one of these cells is
flat for every arm**, which is the signature of a defect that lives in state, not
in season or site. The site asymmetry Block H was written for does not appear in
the accrual object: `H1` beats `F3c` by 0.0000315, three hundredths of a floor.

---

## 3. Block T -- the conditional bonus trip. Fold 2 (183,130 in-bonus fit-window chances)

Object redefined by 15.3: `P(FT_trip_bonus | off_in_bonus)`, its own binary, fitted
independently of Block F. Rows anti-joined against the verified technical trips
(section 4).

| arm | log loss | floor | calib gap pp | team SD ratio | prior-quintile slope |
|---|---:|---:|---:|---:|---:|
| `T0` reference (served `C_plus_state` bundle) | 0.4108482 | 0.0 | +0.139 | 0.364 | 0.269 |
| `T1` + as-of foul / drawn-foul rates | 0.4107385 | 0.0 | +0.145 | 0.400 | 0.368 |
| **`T2` + raw team-foul COUNTS** | **0.3923356** | 0.0 | +0.077 | **0.767** | 0.330 |
| `T3` = `T1` + `T2` | 0.3923117 | 0.0 | +0.073 | 0.771 | 0.376 |

Fold 1 confirms the ordering (`T0` 0.4090113, `T1` 0.4088319, `T2` 0.3901581,
`T3` 0.3901346).

**`T2` is the eligible winner.** It beats the reference by 0.018513 = **23 applied
floors**. `T3` beats `T2` by 0.0000239 = **0.03 of an applied floor**, a tie, and
13.7 rule 2 sends a tie to the simpler arm in the pre-registered order
`T0 < T2 < T1 < T3`. `T1` -- the as-of league-centred rate features -- buys
0.00011, one eighth of a floor, and is **not eligible on either fold**.

What `T2` fixes is exactly what 13.1 said it would: the served bundle sees only
the binary `in_bonus` and cannot tell 7 fouls from 11. By foul count, `T0`'s
calibration gap is **+11.41 pp at 4-6 fouls, -3.77 at 7-9 and -10.02 at 10+**;
`T2` moves them to +6.52 / -4.52 / +3.79. It also nearly doubles the spread of
team estimates (0.364 -> 0.767 of the realised team SD). Both arms leave the
held-out final 2:00 badly under-predicted (calibration gap -2.97 pp for `T2`,
-4.69 pp for `T0`) -- that window is the late-game lane's and the gap is reported,
not fitted.

---

## 4. The technical free-throw anti-join (PM instruction, 15.4)

Verified table `data/processed/models/free_throw/technical_target_verified_trips_v1.parquet`
(5,909 verified technicals, 2022-2025), matched to FT-trip chances on
(season, game_id, period) with the technical's clock inside the chance's own clock
window widened by 25 s (tolerance fixed before the run).

| | value |
|---|---:|
| FT-trip chances in the design | 337,790 |
| chances dropped as mislabelled technicals | **4,059 (1.20%)** |
| `FT_trip_bonus` chances before / after | 153,619 / **151,843** (-1.16%) |
| bonus-trip rate per chance before / after | 0.050555 / **0.050038** (-0.052 pp) |

On the diagnostic's own scale the actual bonus-trip rate falls from 0.05088 to
about 0.05034 per chance, so the sim-actual gap narrows from -9.97% to -8.99%:
**roughly 10.6% of the -1.384 pp bonus-trip channel, about 0.15 pp of FTA/FGA, was
mislabelled technical attempts and is not the engine's to close.** The PM's
estimate of "up to roughly 0.18 pp" is confirmed at the low end.

The **occupancy curve barely moves** -- P(in bonus) by five-minute bucket changes
by at most 0.03 pp (e.g. minutes 10-14, 0.23462 -> 0.23443; minutes 35-37, 0.80599
-> 0.80584) -- while **P(trip | bonus) moves by 0.09 to 0.35 pp** (minutes 35-37
0.17264 -> 0.17150). So the defect sits entirely in the CONDITIONAL half of the
59/41 split, not the occupancy half, and Block F's truth is untouched. Every Block
T number in section 3 is on the cleaned target.

---

## 5. What this round did NOT establish

1. **No closed loop.** No share of the -1.384 pp is claimed as closed in the
   engine. The occupancy target of 13.4 (max per-bucket gap <= 4 pp against the
   served 15.27) is unmeasured; only its offline driver is.
2. **The level contradiction of section 2 is open** and is the first thing the
   closed loop must resolve.
3. **`F4` was not run** -- promoting the silent foul to a seventh
   possession-outcome class rewrites `PO.CLASSES` and every stored artifact.
4. **`D9c` is a proxy.** "Conference-aligned refit cadence" is implemented as
   separate conference and non-conference fits, not as an `S1`-style refit
   calendar. It changes nothing (0.0000131 under `F3c`), and Decision 9 remains
   unresolved for this sub-model on all three cells.
5. **Block T pools the `first` and `cont` populations** with an `is_cont` flag
   rather than fitting them separately as the served model does; a declared cost
   deviation (15.3).
6. **The measured noise floor is degenerate.** Every arm but `F5` is
   deterministic given its data, so a second seed reproduces it exactly. The
   applied floor is round 4b's 0.000804 and every verdict above is quoted against
   it.
7. **No arm is matchup-responsive.** The best prior-quintile slope in the round is
   0.252 (`F5`, Block F) and 0.376 (`T3`, Block T). Both are far below 1.0.

---

## 6. Exact resume command for the closed loop

The arms are not wired. Wiring is: export `F5`'s predictions as a lookup table
over `(period, clock bucket, margin bucket, def_team_fouls, off_team_fouls, site)`
behind a DEFAULT-OFF `ENGINE_FOUL_ACCRUAL` flag in `src/cbb_sim/engine/loop.py`
beside the existing `silent_foul` draw, prove the default path bit-identical
against `docs/ops/parity_reference_windows_v6.json`, then

    .venv/Scripts/python.exe scripts/run_po4b_closed_loop.py --arm round6_F5 --seeds 25 --tag po6_F5_s25

paired against today's `results/engine_v0/po4b_R_s25` reference **only if the
engine commit's simulated values are unchanged** (the round-5 precondition smoke
`smoke60x5_po4b_wiring` is the test); otherwise re-run the reference. Read:
bonus occupancy by minute, FTA/FGA by half and overall excluding and including the
final 2:00, G1-G9 no-regression in floors, team slope, total SD ratio.
