# The drawn block flag in the engine: wiring, parity, paired closed loop (2026-09-30)

Lane C. Pre-registration: `docs/models/shot_block/experiments.md` section 5
(commit 6dbf4ac, before any wiring or run). Wiring 5187017; lookups, tests,
runner and grader b2f83ed. **NOTHING ADOPTED; the flag is default-off; no
served input was overwritten.**

## 1. Verdict

The drawn flag does what the G4 diagnostic said it would: pooled OREB%
0.2836 -> **0.2912** (`SB_K2O`) / 0.2910 (`SB_K2`), **+38 / +37 floors toward**
the 0.2984 actual, and G4's pooled OREB% line flips FAIL -> PASS. The mechanism
lines land on the actuals. **Neither arm is put forward** under the
pre-registered rule: two vetoes fire in both arms, G5 total SD ratio (-12 /
-13 floors) and G4 TOV% (-1.5 floors). The anchor buys about one floor of OREB%
over plain K2 in the closed loop.

| line (target) | `SB0` served (`po4b_R_s25`) | `SB_K2O` | floors | `SB_K2` | floors | floor |
|---|---:|---:|---:|---:|---:|---:|
| **G4 OREB% pooled** (0.2984) | 0.2836 FAIL | **0.2912 PASS** | **+38.0** | 0.2910 PASS | +37.0 | 0.0002 |
| team offence OREB%, mean gap pp | -1.63 | -0.87 | +118 | -0.90 | +114 | 0.006 |
| team defence OREB% allowed, mean gap pp | -1.53 | -0.77 | +90 | -0.78 | +88 | 0.009 |
| team offence MAE pp / defence MAE pp | 4.73 / 4.83 | 4.60 / 4.76 | | 4.60 / 4.77 | | |
| team offence prior-quintile slope | 0.571 | 0.546 | -0.41 (no fall) | 0.544 | -0.44 | 0.061 |
| G1 possessions mean (68.33) | 70.019 | 69.953 | +0.67 | 69.950 | +0.70 | 0.099 |
| G1 possessions SD (5.19) | 5.609 | 5.581 | +0.49 | 5.580 | +0.51 | 0.057 |
| **G5 total SD ratio** (1.0) | 0.8355 | 0.8247 | **-12.0 VETO** | 0.8236 | **-13.2 VETO** | 0.0009 |
| G5 margin SD ratio (1.0) | 0.9695 | 0.9754 | +0.44 | 0.9764 | +0.51 | 0.0135 |
| G5 home/away corr (0.237) | 0.1063 | 0.1009 | -0.47 | 0.1005 | -0.50 | 0.0115 |
| G9 margin bias (0) | +0.112 | -0.072 | +0.18 | -0.085 | +0.12 | 0.221 |
| G9 calibration slope (1.0) | 0.892 | 0.896 | +0.11 | 0.898 | +0.16 | 0.039 |
| G4 eFG% (0.5086) | 0.4990 | 0.4990 | 0.00 | 0.4990 | 0.00 | 0.0006 |
| **G4 TOV%** (0.1739) | 0.1762 | 0.1768 | **-1.50 VETO** | 0.1768 | **-1.50 VETO** | 0.0004 |
| G4 FT rate (0.3295) | 0.3195 | 0.3197 | +0.11 | 0.3196 | +0.06 | 0.0018 |
| G9 total bias, points (0; NOT a veto) | -0.962 | -0.521 | +1.25 | -0.535 | +1.21 | 0.353 |

"floors" = movement TOWARD the target in units of the seed-offset floor
|`po4b_R_s25_floor` - `po4b_R_s25`| (team rows: signed change over the team
floor). Gate numbers are the gate functions' own (`cbb_sim.eval.gates`,
`docs/gates.yaml`).

**Total bias.** Pre-stated expectation +0.4 to +0.6 points; measured **+0.44
(K2O) / +0.43 (K2) points**, toward 0 (the total sits below the actual here).
Reported, not a veto, as the PM ruled.

## 2. Mechanism lines (same 500 games; actual from the rebound design's own rows)

| miss type | blocked share: actual | K2O | K2 | OREB% blocked: actual | K2O | OREB% unblocked: actual | K2O |
|---|---:|---:|---:|---:|---:|---:|---:|
| rim | 0.2581 | 0.2655 | 0.2525 | 0.418 | 0.425 | 0.374 | 0.365 |
| jump2 | 0.0788 | 0.0826 | 0.0844 | 0.403 | 0.396 | 0.282 | 0.272 |
| three | 0.0143 | 0.0143 | 0.0143 | 0.519 (n = 213) | 0.410 | 0.285 | 0.274 |
| all FGA misses | 0.0957 | 0.1028 | 0.0996 | | | | |

The drawn flag reproduces the blocked share per type within about 1 pp and the
blocked/unblocked OREB split within about 1 pp on rim and jump2 (three-point
blocks are underpowered on the actual side). The unblocked rates sit ~1 pp
below the actuals in every type: that is the remaining season-drift LEVEL
channel of the rebound model (offline -1.14 pp; the Stage B `TO` arm's job),
and the pooled OREB% gap left after the flag, -0.72 pp, is its size.

## 3. The two vetoes, read (nothing fixed here)

- **G4 TOV% +0.0006 (-1.5 floors).** More offensive rebounds mean more chances
  per possession, and a chance can end in a turnover, so turnovers per
  possession rise mechanically. The served engine's TOV% was already above the
  actual (0.1762 vs 0.1739); the drawn flag exposes that, it does not create
  it. The floor (0.0004) is small.
- **G5 total SD ratio 0.8355 -> 0.8247.** Not the within-game spread (15.842 ->
  15.781, -0.4%) but the denominator SD(actual - sim mean): 18.962 -> 19.135,
  because the correlation between the sim's game-mean total and the actual
  total falls 0.293 -> 0.270 (excluding the unplayed game; the seed-offset
  run gives 0.299). The per-game changes the flag makes are anti-correlated
  with the served run's own residuals (-0.12; the seed-offset run's changes
  +0.14). Where the block model moves game totals, it moves them slightly the
  wrong way. Caution on the floor: this line's floor is ONE unpaired seed-offset
  draw (0.0009) while per-game mean totals move by SD 4.8 points between seed
  sets, so the floor probably understates the line's noise; the rule is
  applied as registered all the same. Owner of the next look: which games move
  (block-rate extremes), and a 200-seed read with its own floor pair
  (commands below).

## 4. Proofs (section 5.5)

1. Flag unset: `run_engine.py --seeds 5 --max-games 60` digest
   `0d4ddccc...` **equals** `docs/ops/parity_reference_windows_v6.json`
   (PASS, bit-identical).
2. Flag unset: the edited engine reproduces `po4b_R_s25` **bit-for-bit** on 6
   of its games x 25 seeds (150 rows, 26 columns;
   `scripts/diag_shot_block_default_parity_v1.py`), so `SB0` is `po4b_R_s25`.
3. `tests/test_engine.py` + `tests/test_smoke.py` 22 passed after the loop
   edit; `tests/test_shot_block_engine.py` 3 passed: default-off, the new RNG
   family leaves every other family's draws unchanged, and the engine path
   (lookup gather + live state) reproduces the offline K2_Ocell probability on
   real 2025 rows (median |diff| < 1e-6).
4. Lookups validated against the training design: team `def_block_c` 11,185 /
   11,185 team-games exact; shooter rate exact on every roster-matched shooter
   (91% of the design's shooters are on the 15-slot engine roster).

## 5. What was wired (files, flag)

- Flag: **`ENGINE_SHOT_BLOCK`** = `K2_Ocell` | `K2`; unset / `reference` = served.
- `src/cbb_sim/engine/shot_block.py` (new): loads
  `data/processed/models/engine/shot_block_<arm>_F2_2025.npz`, checks the game
  order against the slate, returns P(blocked | missed FGA) from the 17-input
  logit + per-type anchor.
- `src/cbb_sim/engine/rng.py`: family `"shot_block"` appended (drawn only when on).
- `src/cbb_sim/engine/loop.py`: five small hunks: import; load; the shooter
  slot carried alongside `miss_rows` (only when on); the draw replacing
  `blocked_f = 0.0` for FGA misses (FT misses stay 0; fg_make untouched);
  `sb_*` diagnostic counters.
- `scripts/build_engine_shot_block_lut_v1.py` (sibling inputs; F2 fit, strictly
  as-of by game date), `scripts/run_shot_block_closed_loop_v1.py`,
  `scripts/grade_shot_block_closed_loop_v1.py`,
  `scripts/diag_shot_block_default_parity_v1.py`, `tests/test_shot_block_engine.py`.
- Runs: `results/engine_v0/sb_K2O_s25`, `sb_K2_s25` (463 s / 344 s, 4 workers);
  grade `results/shot_block_round2/closed_loop_grade_v1.json`.

Not wired: block credit to a defender (player props); lookups exist for the F2
2025 slate only (other slates need the builder run for them).

## 6. Caveats (stated in advance, not fixed)

The 500-game sample contains one unplayed game (401714278, actual total 0);
it inflates the G5 denominator by ~1.0 point in every arm and does not drive
the arm's move (without it: 17.934 -> 18.121). The served engine inputs carry
same-game leaks being rebuilt tonight: paired deltas are valid, absolute levels
are provisional. The block lookups themselves are strictly as-of.

## 7. Resume / box commands (200 seeds, own floor pair)

    # served reference and its seed-offset floor, then both arms, same subset
    python scripts/run_po4b_closed_loop.py --arm round2_s1 --tag sb_R_s200 --seeds 200 --workers <N>
    python scripts/run_po4b_closed_loop.py --arm round2_s1 --tag sb_R_s200_floor --seeds 200 \
        --seed-offset 1000 --workers <N>
    python scripts/run_shot_block_closed_loop_v1.py --shot-block K2_Ocell --tag sb_K2O_s200 --seeds 200 --workers <N>
    python scripts/run_shot_block_closed_loop_v1.py --shot-block K2 --tag sb_K2_s200 --seeds 200 --workers <N>
    python scripts/grade_shot_block_closed_loop_v1.py sb_R_s200 sb_K2O_s200 sb_K2_s200 \
        --floor sb_R_s200_floor,sb_R_s200
    # local re-run of this round (4 workers, ~8 min per arm)
    .venv/Scripts/python.exe scripts/build_engine_shot_block_lut_v1.py
    .venv/Scripts/python.exe scripts/run_shot_block_closed_loop_v1.py --shot-block K2_Ocell --tag sb_K2O_s25 --seeds 25 --workers 4
    .venv/Scripts/python.exe scripts/run_shot_block_closed_loop_v1.py --shot-block K2 --tag sb_K2_s25 --seeds 25 --workers 4
    .venv/Scripts/python.exe scripts/grade_shot_block_closed_loop_v1.py po4b_R_s25 sb_K2O_s25 sb_K2_s25

The box needs the two committed `shot_block_*_F2_2025.npz` lookups (0.45 MB
each, in git) and the served engine inputs; nothing else new.

## 8. Incidents

None this job. Every process was my own (children identified by command line
and parent PID); `loop.py` / `rng.py` were edited only when `git diff` showed
no other lane's hunk and committed within minutes (5187017, 30 lines, mine
only); `git diff --stat` checked for line-ending churn before every commit.

## 9. PM follow-up: game-sampling intervals, the G5 decomposition, TOV as a count (13:07 EDT)

`scripts/diag_shot_block_game_bootstrap_v1.py` -> `results/shot_block_round2/game_bootstrap_v1.json`.
Paired bootstrap over the 500 GAMES (2,000 draws, seed 20260930; each game keeps
its 25 paired seeds in both arms), gate formulas unchanged, arm minus served:

| line | seed-offset floor | K2_Ocell diff [95% game CI] | K2 diff [95% game CI] | excludes 0 |
|---|---:|---|---|---|
| G5 total SD ratio | 0.0008 | -0.0108 [-0.0189, -0.0041] | -0.0119 [-0.0199, -0.0052] | yes / yes |
| G4 TOV% pooled | 0.0004 | +0.00058 [+0.00042, +0.00073] | +0.00056 [+0.00040, +0.00072] | yes / yes |
| G4 OREB% pooled | 0.0002 | +0.00756 [+0.00717, +0.00795] | +0.00738 [+0.00700, +0.00777] | yes / yes |
| G9 total bias (pts) | 0.353 | +0.441 [+0.333, +0.545] | +0.428 [+0.319, +0.533] | yes / yes |

Under game sampling both vetoes still exclude zero; the game interval is about
10x the seed floor on the G5 line but does not reach it.

**How the G5 total SD ratio is computed** (`cbb_sim.eval.gates.gate_g5`):
per game, the sim mean and SD of the total over its seeds; ratio =
mean over games of the sim SD / SD over games of (actual total - sim mean
total). The denominator mixes dispersion with prediction accuracy: it grows
when the sim's game means track the actual totals less well. Components
(served -> K2_Ocell): mean within-game sim SD 15.842 -> 15.781 (diff -0.061,
game CI [-0.174, +0.051], includes 0); SD(actual - sim mean) 18.962 -> 19.135
(diff +0.174, CI [+0.087, +0.270], excludes 0). The veto is carried by the
ACCURACY component (correlation of sim mean totals with actual 0.293 -> 0.270),
not by a narrower sim.

**TOV%'s denominator** is not the engine's possession count: it is the box
estimate FGA - OREB + TOV + 0.44 FTA on the sim's own box (the same formula on
the actual box), so it does not carry the +1.6 possession bias, but it does
fall mechanically as OREB rises. TOV per team-game as a count: served 12.253,
K2_Ocell 12.281, K2 12.279, actual 12.021 (hoopR box, same games). The count
moves +0.028 per team-game (+0.23%) away from the actual; the rest of the
+0.33% relative TOV% move is the OREB term in the denominator. Engine
possessions per game: 70.019 -> 69.953 / 69.950.
