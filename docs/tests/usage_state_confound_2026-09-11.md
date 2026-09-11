# usage: an own-row delta audit of every state feature the U5 tree consumes -- 2026-09-11

Author: Opus worker (usage Decision-10 lane), 2026-09-11. Evidence for
`ARCHITECTURE_DECISIONS.md` Decision 10 and `docs/LEARNINGS.md` L27/L28, run
BEFORE `docs/models/usage/experiments.md` section 10 (the round-3 data fix and
refit it feeds). Script: `scripts/diag_usage_state_confound.py`. Numbers:
`data/processed/models/usage_v2/state_confound.json`.

Trigger: L28 records that the adopted U5 LightGBM tree consumes four
engine-produced state features (`score_diff`, `sec_remaining`, `period`,
`chance_number`), 26-31% of split importance, and that wiring the tree into the
engine is blocked on exactly this audit -- the same own-row delta method L27
used to find fg_make's leaked `score_diff` before it doubled that model's
margin variance in the engine.

Population: usage's own modelled window, D-I / non-truncated / `pbp_complete`
universe, seasons **2024-2025** (on-floor ids do not exist before 2024, L13).
2026 is sealed and never read. Method: for every event, reconstruct the state
as of the END of the PREVIOUS row of the SAME game's cleaned event stream
(`cbb_sim.models.event_stream.build_stream`'s own row order, which is what
`usage.build_usage_events` consumes) and compare it to the feature's actual
value on the row. This is the fg_make audit's method
(`docs/tests/fg_make_state_confound_2026-09-10.md`), applied to usage's own
event population (all five usage classes) instead of fg_make's raw-plays
population.

---

## 0. HEADLINE

**`score_diff` is POST-OUTCOME, the same defect and the same construction L27
found in `fg_make`.** `usage.build_usage_events` (line 343, before this fix)
reads it off `home_score`/`away_score` on the event's OWN row
(`np.where(off_home, hs - as_, as_ - hs)`) -- byte-identical in shape to
`fg_make`'s banned construction. `sec_remaining`, `period` and `chance_number`
are clean.

## 1. `score_diff` -- own-row score move, by usage event class (2024-2025 pooled)

The move is this row's `home_score`/`away_score` minus the PREVIOUS row's, from
the offence's own side, restricted to rows with a valid previous row in the
same game (a game's own first row is excluded, as in the fg_make audit).

| class | n made | delta == shot's own value | n missed | delta == 0 |
|---|---:|---:|---:|---:|
| `FGA_rim` (worth 2) | 258,321 | **99.959%** | 186,119 | 99.968% |
| `FGA_jump2` (worth 2) | 125,765 | **99.969%** | 187,676 | 99.982% |
| `FGA_3` (worth 3) | 158,814 | **99.985%** | 310,792 | 99.988% |

| class | n | delta == 0 (no scoring event) |
|---|---:|---:|
| `TOV` | 248,468 | 99.978% |

| class | n made | delta == 1 (the FT's own point) | n missed | delta == 0 |
|---|---:|---:|---:|---:|
| `FT_trip` (first attempt of a foul trip) | 152,269 | 99.683% | 67,450 | 99.806% |

**Every usage event class reproduces the leak at 99.7-99.99%** -- higher than
fg_make's 93.6-94.5% on made shots, because this population is
`event_stream.build_stream`'s already-cleaned stream (blocked-shot duplicate
rows and free-throw-trip administrative rebounds removed) rather than the raw
`plays` table fg_make's audit read directly; the ~6% of makes fg_make
attributed to a same-clock dead-ball row landing between the shot and the next
score update is largely the population this cleaning removes. The residual
0.02-0.3% (a handful of rows per class) is not read as a competing mechanism --
it is within the range of clock/score parsing noise at this row count and never
mattered to fg_make's finding either.

**The manufactured share is essentially 100% of the feature's own-row
identity, on EVERY class this model allocates, not only the field-goal
classes.** `TOV` and `FT_trip` -- the two classes L28 and round 2 both flagged
as immune to the shooter-key defect -- are NOT immune to this one: `score_diff`
is alternative-invariant (`features.md` section 1.3), so the leak lives in the
event-state block shared by all five classes, and it is present regardless of
which class or which player is being decided.

## 2. `sec_remaining` and `period` -- clean by construction

Both are read directly off the pbp clock columns (`period`,
`secondsRemaining`) with no dependence on `made` / `scoringPlay`
(`usage.py` `build_usage_events`, the `period`/`sec` block). A clock reading
cannot carry this row's own scoring outcome -- there is no "previous value" for
it to leak from, unlike a cumulative score. No own-row delta applies.

## 3. `chance_number` -- pre-outcome at the row level, confirmed two ways

**Code proof.** `usage._chance_number` builds
`new_poss = concatenate([[True], ends[:-1] | (game boundary)])`, so
`poss_id[i]` -- and therefore `chance_number[i]` -- is a function of
`ends[0..i-1]` **only**. Row `i`'s own `ends[i]` (which depends on `made[i]`)
never enters the computation of chance_number AT row `i`; it only affects rows
strictly AFTER it. An FGA/TOV/FT_trip row's own `oreb` term is always 0
(`cls != 'OREB'`), so it cannot contribute to its own cumulative sum either.

**Empirical confirmation**, isolated from legitimate downstream propagation:
flip `made` on only the LAST FGA row of each game (so there is no later row in
that game for the flip to legitimately propagate to via the next chance's
count) and recompute.

| season | games tested | flipped rows whose OWN chance_number changed |
|---|---:|---:|
| 2024 | 5,178 | **0 of 5,178** |
| 2025 | 5,445 | **0 of 5,445** |

Zero exceptions in both seasons. (An earlier version of this check flipped
EVERY FGA row in the stream at once and found ~9% of those rows' chance_number
"changed" -- that was a defect in the check, not a finding: flipping every FGA
row in a game also flips every EARLIER row than the one under test, and an
earlier row's outcome legitimately moves a LATER row's chance number, which is
causality, not a leak. Isolating the flip to the last FGA row of each game
removes that confound and the result is unambiguous.)

## 4. What this means for U5

`score_diff` cannot be adopted at its round-1/round-2 construction. It is
post-outcome on every usage event class, at 99.7-99.99%, which is a stronger
and cleaner reproduction of the fg_make defect than fg_make's own numbers.
`sec_remaining`, `period` and `chance_number` need no change. The fix and the
paired refit of the adopted LightGBM arm are in `docs/models/usage/
experiments.md` section 10 (round 3, data fix). This audit is a precondition
for, and does not by itself resolve, the Decision-10 closed-loop gate
(`docs/tests/usage_decision10_gate_2026-09-11.md`).
