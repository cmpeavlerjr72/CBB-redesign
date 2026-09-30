# Late-game regime, ROUND 2: window-gated arms, wired DEFAULT-OFF, paired closed loop (2026-09-30)

Lane D. **NOTHING IS ADOPTED. No served default is changed.** Every arm is
reachable only through the new DEFAULT-OFF flag `ENGINE_LATE_GAME`; the PM
decides. Pre-registration: `docs/models/late_game/experiments.md` section 4,
committed at `1c7aa68` before any arm was wired or run.

Code (all committed):

    src/cbb_sim/models/late_game.py            window cell-law class + L3 columns (importable, so the pickles load in workers)
    src/cbb_sim/engine/late_game_adapter.py    LateGameClock / LateGameEvent wrappers, ENGINE_LATE_GAME parser
    src/cbb_sim/engine/adapters.py, loop.py    minimal hooks: the flag, and `in_double_bonus` APPENDED to STATE_COLS
    scripts/train_late_game_r2_v1.py           refit + verify the four served window objects
    scripts/run_late_game_r2_closed_loop.py    tapped paired closed loop (pins and asserts the served stack)
    scripts/grade_late_game_r2_v1.py           the one grader (primary, floors, vetoes, reported lines)
    scripts/diag_late_game_r2_levels_v1.py     kernel / per-game / per-team tables

Outputs (gitignored bulk): `results/engine_v0/lg2_*_s25/`, `results/late_game/round2/grade_stride.json`,
`levels_stride.json`. Artifacts: `data/processed/models/late_game/round2/` (gitignored, reproducible bit-for-bit by the trainer).

---

## 1. Wiring and parity

- **Default path bit-identical, twice.** (a) `run_engine.py` 60 games x 5 seeds
  at `1c7aa68` + the hooks: `digest_engine_run.py --compare
  docs/ops/parity_reference_windows_v6.json` -> **PASS, bit-identical digest**.
  (b) The round's own reference `lg2_R_s25` (HEAD `26f2ff6`) against the stored
  `po4b_R_s25`: 12,500 rows, **28 of 28 columns equal**, 1,750,476 possessions in both.
- **Refits reproduce round 1.** `clk_C2`, `clk_D`: test-row PMFs EXACTLY equal
  to `round1_clock/pmf/*_F2.npy`. `ev_BL3` max|d| 3.7e-9 vs `c027`, `ev_L0S0`
  1.5e-8 vs `c024` (78,314 rows). All S0, fold 2, max_train_date 2024-04-04/08,
  before every 2024-25 tipoff.
- **Outside the window every arm is the served engine**: in all six arms the
  per-simulation first-half points (both teams) and first-half possession count
  are bit-identical to R in **12,500 / 12,500** simulations (veto passed by every arm).
- Engine code was identical across all eight runs (HEAD moved only through other
  lanes' non-engine commits; each `run_meta.json` records its commit; tap row
  mismatches 0 in every run).

## 2. Power

Pre-registered test (4.4): floor on `P(0)/P(1)` <= 0.10 and >= 300 regulation
ties in R. Measured: floor **0.0623**, R ties **390** -> **adequate; the
close-game-enriched sample was NOT triggered**. Sims reaching `|m| <= 6` at the
2:00 mark: 4,106 of 12,500 (32.8%; actual 36.3% on the sample, 37.7% season).

## 3. Decision table (500 games x 25 paired seeds, fold 2; floor = |R_floor - R|)

Target: `P(0)/P(1)` **1.546** (season), OT **0.0557**, band 0.046-0.055. The
sample's own actual: P(0) 0.0680, P(1) 0.0360, ratio 1.889 (descriptive).

| arm | P(0) | P(1) | P(0)/P(1) | floors | paired SE | OT in band | G1 mean | G1 SD | G5 m / t | G9 m / t | half share | 1st half | vetoes |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---:|:--:|---|
| R | 0.0312 | 0.0566 | 0.551 | -- | -- | no | 70.019 | 5.609 | 0.969 / 0.835 | +0.11 / -0.96 | 0.4703 | -- | -- |
| R_floor | 0.0274 | 0.0560 | 0.489 | -- | -- | no | 69.920 | 5.552 | 0.956 / 0.835 | -0.11 / -1.32 | 0.4710 | -- | -- |
| W_C2 | 0.0306 | 0.0563 | 0.543 | **-0.13** | 0.038 | no | 70.201 (+1.85 fl) | 5.630 | 0.971 / 0.840 | +0.12 / -0.61 | 0.4692 (+1.55 fl) | identical | **FAIL** G1 mean, half share |
| **W_D** | **0.0437** | **0.0511** | **0.854** | **+4.88** | 0.047 | no (0.0437) | 70.255 (+2.39 fl) | 5.721 (+1.96 fl) | 0.971 / 0.849 | +0.16 / -0.46 | 0.4688 (+2.03 fl) | identical | **FAIL** G1 mean, G1 SD, half share |
| E_BL3 | 0.0318 | 0.0550 | 0.579 | +0.46 | 0.033 | no | 70.017 | 5.588 | 0.970 / 0.838 | +0.11 / -0.98 | 0.4703 | identical | PASS |
| E_L0S0 (control) | 0.0326 | 0.0566 | 0.576 | +0.41 | 0.025 | no | 70.029 | 5.596 | 0.971 / 0.835 | +0.12 / -0.96 | 0.4703 | identical | PASS |
| W_C2+E_BL3 (pre-registered combo) | 0.0314 | 0.0517 | 0.607 | +0.90 | 0.044 | no | 70.203 (+1.86 fl) | 5.611 | 0.970 / 0.841 | +0.13 / -0.60 | 0.4691 (+1.59 fl) | identical | **FAIL** G1 mean, half share |
| W_D+E_BL3 (EXPLORATORY, not pre-registered) | 0.0420 | 0.0494 | 0.851 | +4.82 | 0.049 | no | 70.234 (+2.18 fl) | 5.694 (+1.48 fl) | 0.971 / 0.847 | +0.15 / -0.50 | 0.4689 (+1.85 fl) | identical | **FAIL** G1 mean, G1 SD, half share |

Floors: P(0) 0.00384, P(1) 0.00064, ratio 0.0623, G1 mean 0.099, G1 SD 0.057,
G5 margin 0.0135, G5 total 0.00084, G9 margin 0.221, G9 total 0.353, half share
0.00074. Actual: G1 68.33 / 5.19, half share 0.4731. "fl" in a veto cell = how
much worse than R, in floors. **The combined arm is W_C2+E_BL3 by the
pre-registered rule 4.2.6** (both duration arms fail a veto, so C2 by default);
W_D+E_BL3 was run with spare wall clock and is labelled exploratory.

**Reading, against the pre-registration.**

1. **Only `D_clk` moves the primary: +4.88 floors** (0.551 -> 0.854; paired
   game-bootstrap SE 0.047, so ~6.4 SE). OT 0.0312 -> 0.0437, just under the
   0.046 band floor. Section 1.3's rule applies: `P(0)/P(1)` stays below 1.0,
   so **D is recorded as NOT fixing the defect**; it closes ~30% of the ratio gap.
2. **`C2_clk` -- round 1's offline CRPS winner -- does not move it at all**
   (-0.13 floors), although it changes the window's clock more than D does.
3. **The event half is not the lever** (as round 1 said): E_BL3 +0.46 floors,
   and its cadence control E_L0S0 +0.41, so enrichment itself is ~+0.05 floors.
   Both pass every veto.
4. **Every duration arm fails the G1-mean and half-share vetoes** (and D also
   G1 SD). See section 5: this is real window over-production, not only the
   engine's upstream G1 excess.

## 4. Why D moves ties and C2 does not (per-possession-type evidence)

Closed-loop window durations (intended / consumed, s), by role and clock bucket:

| cell | R | W_C2 | W_D |
|---|---|---|---|
| tied, (0,10] | 12.2 / 4.1 | 7.0 / 2.8 | **20.4 / 4.7** |
| tied, (10,30] | 11.4 / 9.5 | 7.2 / 6.8 | **17.4 / 14.4** |
| leading, (0,10] | 10.3 / 3.6 | 5.7 / 2.5 | **2.6 / 1.5** |
| leading, (10,30] | 10.8 / 8.9 | 4.7 / 4.6 | **3.6 / 3.6** |
| trailing, (0,10] | 10.5 / 4.3 | 7.2 / 3.1 | 8.7 / 3.7 |

D's tied offence **holds for the last shot**; C2's does not. Mechanism, checked
offline on the fold-2 window rows: C2's hierarchy is `(prev_end, bucket,
period_type, eg_role6, tempo)`, and tied rows inside 30 s fall back to mean level
2.78 of 5 -- BELOW the role dimension -- so they are served a role-free law
(predicted mean 7.3 s; actual uncensored 11.2 s with 54% horn-censored). D puts
`role3` first; tied rows keep the role (predicted 18.8 s). The fallback ORDER of
the hierarchical Kaplan-Meier, not the floor, decides whether the hold exists.

**Kernel, P(regulation tie | |margin| at the 2:00 mark = k):**

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| actual 2024-25 (pbp anchor, verified OT) | 0.244 | 0.181 | 0.193 | 0.119 | 0.138 | 0.075 | 0.072 |
| actual n | 172 | 331 | 347 | 329 | 355 | 294 | 278 |
| R | 0.149 | 0.122 | 0.120 | 0.096 | 0.074 | 0.058 | 0.027 |
| W_C2 | 0.128 | 0.125 | 0.106 | 0.096 | 0.090 | 0.038 | 0.029 |
| W_D | **0.216** | **0.181** | **0.171** | **0.140** | 0.094 | 0.062 | 0.046 |
| E_BL3 | 0.140 | 0.122 | 0.120 | 0.099 | 0.084 | 0.058 | 0.029 |

D reproduces the kernel for k = 0-3; the residual sits at k = 4-6 and in the
share of simulations that arrive close (32.8% vs 36-38%).

**Regulation-margin distribution P(|m| = 0..5):** R .0312 .0566 .0463 .0469 .0443 .0418;
W_C2 .0306 .0563 .0463 .0454 .0388 .0434; W_D .0437 .0511 .0473 .0461 .0388 .0389;
E_BL3 .0318 .0550 .0462 .0458 .0450 .0430; W_C2+E_BL3 .0314 .0517 .0481 .0454 .0385 .0428;
W_D+E_BL3 .0420 .0494 .0491 .0445 .0410 .0406.

## 5. The veto failures, by level

- **Window possessions per game, by |m@2:00| (k = 0..6)**: actual 7.59 7.80 7.74
  7.93 7.79 7.03 6.71; R 7.68 7.93 7.98 7.74 7.75 6.68 5.92; W_C2 9.03 9.02 9.04
  8.71 8.43 7.29 6.56; W_D 8.23 8.77 8.47 8.43 8.04 6.95 6.18. On this
  like-for-like count (pbp possession segmentation for the actual) **R is already
  right, C2 over-produces by ~1.1-1.3 and D by ~0.3-0.9 window possessions**. So
  the G1-mean failure is largely the arm's own, not only the engine's upstream
  +1.7 possessions; the half-share failure is its arithmetic consequence (more H2
  points). The first half is bit-identical, so neither failure is upstream damage.
  Caveat: round 1's "9.24-11.34" was a different count and is not comparable.
- **G9 total bias improves** under every duration arm (-0.96 -> -0.46 for D) and
  G5 margin/total SD ratios do not regress.

## 6. Role splits (offence-signed) and final-2:00 FTA

Window FIRST chances, live role, from the served probabilities; the tap reads
each row's own offence `score_diff`, so no row carries the opponent's sign.

| arm | lead-minus-trail bonus-FT | trail-minus-lead 3PA share | leading bonus-FT | final-2:00 FTA / game (all / close) |
|---|---:|---:|---:|---:|
| R | +0.286 | +0.158 | 0.411 | 5.28 / 5.48 |
| W_C2 | +0.323 | +0.189 | 0.443 | 5.58 / 6.24 |
| W_D | +0.324 | +0.193 | 0.444 | 5.44 / 5.91 |
| E_BL3 | +0.285 | +0.162 | 0.416 | 5.31 / 5.56 |
| W_C2+E_BL3 | +0.321 | +0.192 | 0.447 | 5.61 / 6.33 |
| W_D+E_BL3 | +0.322 | +0.197 | 0.448 | 5.46 / 5.97 |
| actual | +0.393 (fold-2 window first chances) | +0.198 | 0.543 | **5.10 / 6.00** (season, pbp); 4.98 / 6.07 (this sample) |

**For lane A:** served final-2:00 FTA is **5.28 per game (5.48 in games within 6
at 2:00)** against a pbp actual of 5.10 (6.00). The closed-loop leading-offence
bonus-FT rate is 0.41 against 0.54 actual although round 1's arm A predicts 0.538
offline on actual states: the gap is the state the engine arrives in (bonus
status), i.e. foul accrual, not the event model. Final-2:00 is "from the first
H2 possession starting at <= 120 s to the end of regulation"; the actual is
pbp-accumulated, so it is descriptive, not a graded line.

## 7. Other levels

- **Per game** (paired, 25 seeds): D raises the tie count in 35.4% of games and
  lowers it in 14.4% (mean +0.31 ties/game); C2 21.2% / 23.2% (mean -0.02); the
  reseed floor itself 27% / 30%.
- **Per team**: 1 of 347 teams reaches 200 simulations -> **UNDERPOWERED**; not read.
- **Per player**: not applicable.

## 8. What did not run, and resume commands

Everything pre-registered ran; the enriched sample was not triggered (section 2).
Not run: a 200-seed gate on any arm; an OT-inclusive window; a D-variant that
keeps the hold but not the extra possessions (would need its own pre-registration).

    # artifacts (exact reproduction)
    .venv/Scripts/python.exe scripts/train_late_game_r2_v1.py
    # one arm (flag values: off | clk_C2 | clk_D | ev_BL3 | ev_L0S0 | clk_X+ev_Y)
    .venv/Scripts/python.exe scripts/run_late_game_r2_closed_loop.py --late-game clk_D --seeds 25 --workers 4 --tag lg2_W_D_s25
    .venv/Scripts/python.exe scripts/run_late_game_r2_closed_loop.py --tag lg2_Rfloor_s25 --seeds 25 --seed-offset 1000
    # enriched sample, if ever wanted: add --sample enriched
    .venv/Scripts/python.exe scripts/grade_late_game_r2_v1.py --ref lg2_R_s25 --floor lg2_Rfloor_s25 \
        --arms lg2_W_C2_s25 lg2_W_D_s25 lg2_E_BL3_s25 lg2_E_L0S0_s25 lg2_W_C2_E_BL3_s25 lg2_W_D_E_BL3_s25 \
        --out results/late_game/round2/grade_stride.json
    .venv/Scripts/python.exe scripts/diag_late_game_r2_levels_v1.py --ref lg2_R_s25 --runs lg2_Rfloor_s25 lg2_W_C2_s25 lg2_W_D_s25 ... --out results/late_game/round2/levels_stride.json

## 9. Incident

The first `ev_L0S0` refit ran LightGBM unpinned (it ignored `OMP_NUM_THREADS`)
at ~6 cores for ~17 minutes (11:03-11:20), above the lane's 4-core cap. I
stopped my own process (PID 100292, parent 107824) and refit with `n_jobs=3`
pinned in the trainer; no other process was touched. The change-ledger row for
this round is NOT written (the file is PM-only today).
