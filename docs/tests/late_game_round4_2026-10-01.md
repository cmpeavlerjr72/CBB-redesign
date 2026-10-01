# Late-game round 4: end-of-period possession value and the leading team's clock (2026-10-01)

Lane L, day session 2026-10-01, 09:04-10:40 EDT (wall clock; run log `results/late_game/round3/run_log.txt`).
**NOTHING IS ADOPTED. No served default changes.** All new behaviour sits behind default-off flags.

- Pre-registration: `docs/models/late_game/experiments.md` section 9, commit `9ca9bcd`. It was pushed before any arm was fitted, wired or run.
- Base: served stack v2, parity v9.

## 0. Verdict

Screen: 500 verified games x 25 paired seeds, fold 2. Floor (Decision 12) = max(draw SD over R9 + four seed offsets, 2 x paired game-bootstrap SE). All floor draws were re-run under the round-4 tap; their games are bit-identical to round 3's.

| arm (9.3) | P(0)/P(1) | floors vs R9 | OT (dP0, SE) | vetoes failing | candidate |
|---|---:|---:|---|---|---|
| R9 (served v2) | 0.552 | -- | 0.0290 | -- | -- |
| Dt (`clk_Dt`, round 3) | 0.672 | +2.13 | 0.0341 (+0.0051, 0.0012) | none | no (< 1.0) |
| Dt+a1 (`clk_Dt` + `ENGINE_LG_BUZZER=BZ3`) | 0.796 | +2.93 | 0.0383 (+0.0094, 0.0016) | G9 total bias +3.5 fl, half share +1.9, G1 SD +1.6 | no |
| Dt+a (+ `ENGINE_LG_MAKE=MK2`) | 0.832 | +3.06 | 0.0382 (+0.0092, 0.0017) | G9 total +5.6, half share +2.2, G1 SD +1.5, **H1 buzzer test (overshoot)** | no |
| Dt+b (`clk_DtL`) | 0.701 | +1.87 | 0.0410 (+0.0121, 0.0017) | G1 mean +5.0, G1 SD +2.4, half share +3.6, window possessions | no |
| Dt+a+b (full set) | 0.848 | +3.02 | **0.0454** (+0.0165, 0.0019) | G1 mean +3.9, G1 SD +3.0, G9 total +1.9, half share +5.4, window possessions, H1 buzzer test | no |

**No arm qualifies.** None reaches P(0)/P(1) 1.0, and none is in the 0.046-0.055 OT band; the full set is closest at 0.0454.

**No arm is a clear best.** The four set arms' primaries sit within one floor of each other (+2.93 / +3.06 / +3.02, floors about 0.09). So under section 9.4 the outcome is NO ARM ADOPTED. No `d1001_L_2.md` full-size request was filed: the PM's condition for one ("qualifies or is the clear best") is not met. Status: RUN.

**Offline guards (both folds, all pass; fold 2 selects):**

| law | selected | F2 vs baseline | F1 vs baseline | notes |
|---|---|---:|---:|---|
| No-shot law | `BZ3`: period x start bucket x role x start type | +18.2 floors vs `BZ0` (no time term) | +16.3 | `BZ3` beats `BZ2` by +2.67 floors on F2 (+1.17 on F1) |
| Buzzer make law | `MK2`: class x period x {0 s, 1 s} | +12.7 floors vs `MK0` (no buzzer term) | +15.8 | `MK2` beats `MK1` by +6.4 (+3.9 on F1) |
| Leading clock law `LGL` | -- | +13.1 floors vs A on leading rows | +13.2 | Its lead 1-3 (30,60] profile is 16.9 s against 17.9 actual (A: 11.8) |

Sources:
- `data/processed/models/late_game/round4/buzzer_grade.json`
- `make_grade.json`
- `results/late_game/round4/clock_offline.json`

## 1. Who owns the last-second possession value (9.1; the motivating evidence)

`scripts/diag_late_game_r4_owner_v1.py`, output `results/late_game/round4/owner_R9.json`. Possessions starting at <= 35 s, sim vs 2024-25.

1. **No-shot at the horn (owner 1).**
   - The data end 18-29% of H1 possessions starting inside 10 s with no shot (`end_period`). The figure is 21-29% for tied and 20-25% for trailing period-2 possessions inside 6 s.
   - The engine produces none. `possession_outcome` drops these as "the clock model's job", and the engine runs the event cascade on every possession.
2. **Make rate of the shot at the horn (owner 2).** First-chance shots with <= 1 s left at the shot:

   | half | sim (three / jumper / rim) | actual (three / jumper / rim) |
   |---|---|---|
   | H1 | 0.267 / 0.364 / 0.558 | 0.163 / 0.213 / 0.369 |
   | H2 | 0.223 / 0.370 / 0.638 | 0.133 / 0.169 / 0.441 |

   fg_make has the features but does not reproduce this in the loop.
3. **The KD timing feed is NOT the owner.** The share of shots that come at <= 1 s left is close: sim 0.76 / 0.65 (H1 / H2) against actual 0.85 / 0.72.
4. **Shot mix is a small term.** A sequential split of the PPP gap gives:
   - H1: volume -0.02 to -0.11, make -0.02 to -0.14;
   - tied P2: make -0.15 to -0.34.

## 2. What each arm does, by level

### 2.1 End-of-period PPP, possessions starting <= 10 s (`results/late_game/round4/levels.json`)

| line | actual | R9 | Dt+a1 | Dt+a | Dt+a+b |
|---|---:|---:|---:|---:|---:|
| H1, all | 0.730 | 0.943 | **0.764** | 0.642 | 0.642 |
| H1, home | 0.817 | 1.002 | 0.814 | 0.693 | 0.693 |
| H1, away | 0.673 | 0.886 | 0.709 | 0.590 | 0.590 |
| H1, neutral | 0.636 | 0.943 | 0.792 | 0.649 | 0.649 |
| H2, all | 1.009 | 1.017 | 0.939 | 0.856 | 0.864 |

- **The first-half buzzer test (free extra test).** The no-shot law alone lands H1 PPP at 0.764 against 0.730, and it holds by site. The make law then overshoots (0.642; floor 0.022) and fails the test.
- **Why the make law overshoots.** It fixes the <= 1 s cell, but the sim's shots at 1-10 s left are too poor. The data's H1 threes at 1-6 s make 0.43-0.51, against 0.30 in the sim: a quick-shot selection effect fg_make lacks. Removing the buzzer excess therefore removes the compensation.
- **Why H2 falls below the actual.** The sim's leading offence is shooting, not being fouled (owner (b)).

### 2.2 Where the ties are lost now (`results/late_game/round4/tieloss.json`)

| measure | R9 | full set | actual |
|---|---:|---:|---:|
| conversion share of the gap at T = 1:00 | -0.0212 | -0.0031 | -- |
| kernel when tied at 0:10 | 0.67 | 0.83 | 0.85 |
| tied final possession, scores 0 | 46% | 59% | 64% |
| tied final possession, game goes to OT | 35% | 54% | 60% |

- The end-game kernel is now mostly right.
- What remains is arrival: too few games reach a tie at 1:00 / 0:10.
- A second piece is the excess of one-point finishes. P(1) is 0.054 in the full set against 0.037 actual, so the ratio stays at 0.85.

### 2.3 Other levels

- **Site.** OT rate rises in neutral and non-neutral games alike (full set: 0.047 / 0.045 against actual 0.068 / 0.054).
- **Per game, paired against R9, full set:** ties rise in 45.6% of games and fall in 16.2%; the mean is +0.41 ties per game over 25 sims.
- **Per team:** UNDERPOWERED (1 of 347 teams reaches 200 sims).
- **Per player:** not applicable.

### 2.4 The veto failures are informative

1. **G9 total bias.** The no-shot law removes about 0.37 points per game, and the make law about 0.23 more. This exposes the engine's existing total under-prediction (-0.80): a compensation, as in Decision 11.
2. **Half share and the first-half count.** The sim has about 35-80% MORE possessions starting in the last 10 s of the first half than the data: 0.16-0.21 against 0.12 per game per bucket. Every per-possession fix at the horn therefore removes too many first-half points. That is a clock defect (the possession count at the half's horn), not this round's.
3. **G1 mean and window possessions.** The leading law adds possessions in the loop (+0.2 per game) although it matches the leading durations offline by band. The excess sits outside the leading cells, because the trailing offence is too short in the sim (trail 1-3 (30,60]: 12.1 s against 15.7 actual). Offline, the same five-band law on trailing rows (`LGLall`) is also +10 floors on F2, but it was NOT pre-registered and was not run.

## 3. (c) Pricing the leading-team FT make defect (9.5; reported, never applied)

`scripts/diag_late_game_r4_ftprice_v1.py`, output `results/late_game/round4/ftprice_R9.json`. This is a first-order re-scoring of R9's log with no behavioural response.

**Inputs.** Sim vs actual leading-offence FT make, final 2:00:

| cell | sim | actual | conversion |
|---|---:|---:|---:|
| lead 1-3, (10,30] | 0.642 | 0.774 | 0.37 |
| lead 1-3, (0,10] | 0.640 | 0.761 | 0.34 |

Other cells are within 0.05.

**When the FT model is fixed, expect:**

| line | change | SE | from -> to |
|---|---:|---:|---|
| OT rate | **-0.0014** | 0.0005 | -- |
| P(1) | -0.0057 | -- | 0.0525 -> 0.0468 |
| P(0)/P(1) | **+0.037** | 0.015 | -- |

The fix COSTS tie rate but HELPS the ratio, because it removes one-point finishes faster than ties. Lane B's and lane I's set should be judged with this in hand.

## 4. Lane H's clock change (9.6), checked against commit `7383999`

- **K1** (`cont_mode`, pre-cascade draw kept). The window laws survive: `dur` is the window draw. But `_clock_cont` recomputes `used` after the cascade, so the no-shot law's "runs to the horn" is overwritten. A one-line guard would be needed (`used[ns] = left` after `_clock_cont`).
- **K2** (`redraw_end` at the same uniform). The window laws do NOT survive. `LateGameClock` forwards `redraw_end` to the served clock through `__getattr__`, so the outcome-conditioned redraw replaces the window draw. Those laws would have to be refit as outcome-conditioned window laws.
- **The make law.** It survives either way; it wraps fg_make.

## 5. Wiring, parity, tree state

| flag | file | what it does |
|---|---|---|
| `ENGINE_LG_BUZZER=BZ1-3` | `late_game_adapter.Buzzer` + a 14-line `loop.py` hunk | Draws on its own stream `(seed, game_id, "lg_buzzer")`. A no-shot possession runs to the horn and records no box event. |
| `ENGINE_LG_MAKE=MK1-2` | `late_game_adapter.LateGameMake`, wrapped in `adapters.py` | Replaces the make probability on first-chance shots with `rint(sec - chance_elapsed_s) <= 1`, periods 1-2. |
| `ENGINE_LATE_GAME=clk_DtL` | `late_game_adapter.py` | `clk_D` on tied rows, `LGL` on leading rows. |

All three are off by default.

- **Artifacts.** The fold-2 laws are force-added: `data/processed/models/late_game/round4/{clk_LGL.pkl, buzzer_BZ*_F2.json, make_F2.json}`, under 1 MB together.
- **Parity.** v9 digest PASS, bit-identical. `src/` contained only my edits at the time.
- **Round-3 arm unchanged.** `clk_Dt` reproduces `lg3_Dt9_s25` bit-for-bit.
- **Floors.** `lg4_R9_f1..f4` equal `lg3_R9_f1..f4` bit-for-bit.
- **Tree state.** All four arm runs started on a clean `src/`. The floor re-runs started with only an untracked file of another lane present (`fg_two_stage.py`), and they match round 3 exactly.
- **Code commits.**
  - `9ca9bcd`: pre-registration, tap, trainers.
  - `52a2d7a`: wiring and artifacts.
  - This commit: graders, diagnostics, results.

## 6. NOT run

- **Fold-1 closed loop.** No fold-1 engine inputs exist; fold 1 enters offline only.
- **The full-size box read.** No qualifier or clear best.
- **`LGLall` / a trailing-role law.** Not pre-registered.
- **Any fix for the horn possession count, the quick-shot make, or the leading-offence fouling.** Out of scope for this round; identified as owners above.

## 7. Recommended next step

1. **Clock, first-half horn.** The sim makes about 35-80% too many possessions start in the last 10 s of a half. Until that is fixed, any honest end-of-period fix fails the half-share veto. This belongs to the clock lane (H).
2. **Late-game round 5.** Pre-register `BZ3` plus a five-band window law for BOTH trailing and leading rows (`LGLall` on non-tied rows, `clk_D` on tied rows), so the window possession count is preserved by construction. Hold the make law back until fg_make's quick-shot selection (1-10 s left) is modelled. Its decision line stays section 1.3.
3. **Compensations to carry under Decision 11.** The no-shot law exposes about -0.37 points per game of total under-prediction. The FT fix (c) costs about -0.0014 OT rate and adds +0.04 to the ratio.

## 8. Resume commands

```
.venv/Scripts/python.exe scripts/run_late_game_r4_closed_loop.py --workers 3 --tag lg4_DtA_s25 --env ENGINE_LATE_GAME=clk_Dt --env ENGINE_LG_BUZZER=BZ3 --env ENGINE_LG_MAKE=MK2
.venv/Scripts/python.exe scripts/grade_late_game_r4_v1.py --base lg4_R9_s25 --draws lg4_R9_f1_s25 lg4_R9_f2_s25 lg4_R9_f3_s25 lg4_R9_f4_s25 \
    --arms lg3_Dt9_s25 lg4_DtBZ_s25 lg4_DtA_s25 lg4_DtL_s25 lg4_DtLA_s25 --out results/late_game/round4/grade.json
.venv/Scripts/python.exe scripts/train_late_game_r4_buzzer_v1.py ; .venv/Scripts/python.exe scripts/train_late_game_r4_make_v1.py ; .venv/Scripts/python.exe scripts/train_late_game_r4_clock_v1.py
```
