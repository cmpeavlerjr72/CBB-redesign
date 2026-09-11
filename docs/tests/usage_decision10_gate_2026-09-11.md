# usage: Decision-10 closed-loop gate on the round-3 LightGBM tree -- 2026-09-11

Author: Opus worker (usage Decision-10 lane), 2026-09-11. Pre-registration:
`docs/models/usage/experiments.md` section 11, committed BEFORE any arm but
`reference` was run. Data-fix precondition: `docs/tests/
usage_state_confound_2026-09-11.md` (own-row delta audit) and `experiments.md`
section 10 (round-3 refit). Scripts: `scripts/run_usage_tree_closed_loop.py`
(engine runs), `src/cbb_sim/engine/usage_tree_adapter.py` (the gate's own
adapter, NOT SERVED). Raw results: `results/engine_v0/usage_*` (`run_meta.json`
per arm), `results/engine_v0/usage_tree_closed_loop_per_seed_s25.json`.

**VERDICT: DO NOT WIRE.** Live and frozen state differ from each other by more
than the paired-seed noise (fails the pre-registered condition (b)), and a
tree refit WITHOUT any state feature ties the live arm on every usage-specific
read (fails condition (c)) -- the state buys nothing worth its own fragility.
All three tree variants clearly beat the served U1 baseline on usage
concentration, which is a real and useful finding, but it argues for a future
bake-off of the refit-without-state tree against U1, not for adopting the
state-carrying arm this round pre-registered. The served default
(`ENGINE_USAGE=reference`, U1 proportional) is UNCHANGED by this gate; that is
a PM decision, not made here.

---

## 1. What was gated, and how

Four arms, one paired-stream engine run each, on the SAME 500-game F2 2025
stride subset (sorted by `game_id` ascending, every 11th row, first 500 --
identical to `run_rot5_closed_loop.py` / `run_clk3c_closed_loop.py`) and the
SAME pinned sub-models (`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round4_B1`,
`ENGINE_CLOCK=v3c_srfloor_P3_s1`, `ENGINE_REBOUND=s1_weekly`,
`ENGINE_FREE_THROW=s1_conf_aligned`, `ENGINE_ROTATION=reference`):

| arm | `ENGINE_USAGE` | what it is |
|---|---|---|
| `reference` | `reference` | served U1 proportional (`usage.draw_player`'s own rule), unchanged |
| `live` | `tree_v3` | round-3 corrected LightGBM tree (`docs/models/usage/experiments.md` section 10), state features read LIVE off `GameState` every step |
| `frozen` | `tree_v3_freeze` | the SAME boosters, state held at pregame values all game (`score_diff=0`, `sec_remaining=2400`, `period=1`, `chance_number=1`) |
| `nostate` | `tree_v3_nostate` | a tree refit with NO state features at all, live otherwise (the L31/L33 refit-without arm) |

5-seed smoke first (all four arms), then scaled to 25 seeds (all four arms;
comfortably inside the session's wall-clock budget -- reference and each tree
arm both completed at 25 seeds by 11:49 ET against a 12:15 ET checkpoint).
6 engine workers throughout. Every run wrote `games.parquet` + `players.parquet`
+ `run_meta.json` to `results/engine_v0/usage_<arm>[_s25]/`.

**How the tree was wired without touching any served default.** `loop.py`'s
usage call site now passes the live state and on-floor identities through to
`ad.usage.probs(...)` as keyword arguments; the served `UsageAdapter.probs`
accepts and ignores them (`**_state_kwargs`), so `ENGINE_USAGE=reference` is
byte-identical to before this gate existed. A new opt-in dispatcher
(`adapters._load_usage`, default `"reference"`) routes any other
`ENGINE_USAGE` value to `usage_tree_adapter.UsageTreeAdapter`, which builds the
tree's non-state per-alternative features (`prior_rate`, `exposure_asof`,
`minutes_asof`, `minutes_per_game_asof`, `games_asof`, `position_code`,
`is_transfer`) once at load time from `inp.roster_cbbd` + the existing
`asof_v2_shotshooter.parquet` panel -- the same backward as-of join
`build_engine_inputs.py` already uses for `usage_rate`, reproduced here as a
read-only side table so nothing under `data/processed/models/engine/` is
touched. `rate`/`share` come from `inp.usage_rate` UNCHANGED (round 3 only
touches `score_diff`, never the shrinkage prior/m `usage_rate` is built from).
Verified with a fixture-booster smoke test end to end (`simulate_chunk` on 3
games under all four `ENGINE_USAGE` values) before any real compute ran.

---

## 2. Aggregate gates (G1/G5/G9-style): no regression on any arm

25-seed means, +/- the standard error of the mean (`SD / sqrt(25)`), and the
pairwise z-score (`diff / combined SE`) against the served baseline.

| metric | reference | live | frozen | nostate | live vs reference (z) | max |z| among all pairs |
|---|---:|---:|---:|---:|---:|---:|
| margin SD | 15.876 +/- 0.102 | 15.945 +/- 0.119 | 15.903 +/- 0.118 | 15.865 +/- 0.104 | +0.44 | 0.50 |
| home/away score corr | 0.0014 +/- 0.0091 | 0.0031 +/- 0.0094 | -0.0017 +/- 0.0092 | 0.0084 +/- 0.0078 | +0.13 | 0.84 |
| possessions/game | 70.030 +/- 0.038 | 70.050 +/- 0.032 | 70.003 +/- 0.036 | 70.025 +/- 0.034 | +0.41 | 0.98 |
| total (both teams) | 145.293 +/- 0.109 | 145.323 +/- 0.110 | 145.277 +/- 0.117 | 145.373 +/- 0.113 | +0.19 | 0.59 |

**No pair on any of these four reads exceeds |z| = 1** except one line at 0.98
(possessions, live vs frozen) -- indistinguishable from noise at 25 seeds
throughout. This is the core Decision-10 bar (margin SD, home/away
correlation, possessions per game) and every arm clears it: **usage's choice
of arm does not move scoring variance, home/away correlation, pace or total,
in either direction, at this seed count.** This is the expected result for a
sub-model that decides WHO shoots rather than WHETHER a shot goes in --
unlike `fg_make`'s `score_diff` (L27), which multiplied margin SD by 3x, usage
has no direct channel into shot outcome. It does NOT mean the state feature is
inert; section 3 finds the channel it actually moves.

---

## 3. Usage-specific gates: a real, detected state-dependence, and a tie the
served baseline can't see

| metric | reference | live | frozen | nostate |
|---|---:|---:|---:|---:|
| top-1 usage share (mean) | 0.2526 +/- 0.0003 | 0.2572 +/- 0.0003 | 0.2540 +/- 0.0003 | 0.2577 +/- 0.0003 |
| top-3 usage share (mean) | 0.5973 +/- 0.0005 | 0.6056 +/- 0.0004 | 0.6011 +/- 0.0004 | 0.6063 +/- 0.0004 |
| pregame-rate quintile span, predicted (pp) | 10.75 +/- 0.05 | 11.07 +/- 0.05 | 10.92 +/- 0.04 | 11.12 +/- 0.04 |

Pairwise z-scores (`diff / combined SE`, 25 seeds each):

| comparison | top-1 share z | top-3 share z | quintile span z |
|---|---:|---:|---:|
| live vs reference | **+10.14** | **+13.49** | **+4.82** |
| live vs frozen | **+7.29** | **+7.82** | **+2.41** |
| live vs nostate | -0.94 | -1.21 | -0.69 |
| frozen vs nostate | **-8.11** | **-9.19** | **-3.32** |

**Finding 1: every tree variant beats the served U1 baseline, decisively and
consistently.** Live, frozen and nostate all sit 4.8-13.5 SE above `reference`
on every usage-concentration read. This replicates, at engine scale, round 1's
offline finding that U1 under-concentrates the top of the usage distribution
by 0.5-2.9 pp (`experiments.md` section 6, R3) -- and shows a tree-family arm
(with or without state) measurably closes part of that gap in the actual
simulator, not just on held-out historical rows.

**Finding 2: live and frozen are NOT the same, and the gap is real, not
noise.** Freezing the SAME boosters' state at pregame values costs 0.32-0.45 pp
of top-1/top-3 usage share and 0.15 pp of quintile span, at z = 2.4-7.8 --
far outside the |z| < 1 that every aggregate gate in section 2 showed. **This
is the closed-loop signature Decision 10 exists to catch**, even though it is
two orders of magnitude smaller than fg_make's (L27: margin SD 34.6 vs 12.1
actual, a 3x blowup). The mechanism is legible from the offline evidence
already on record: `experiments.md` section 10.5 found the tree extracts
"nearly the same amount of usable signal from the leaked and the corrected
[score_diff] column" for the identity target, meaning the state features carry
real, live-state-dependent information the tree has learned to lean on -- a
frozen pregame value breaks that dependency exactly as a frozen margin broke
the rotation model's late-game read in L31 ("a frozen margin of zero late in a
game is a tie game").

**Finding 3: a tree that never saw state performs the same as -- if anything
marginally BETTER than -- the tree that gets to use state live.** `nostate`
vs `live` is not significant on any of the three reads (|z| < 1.3), and
`nostate`'s point estimate is slightly ABOVE `live`'s on top-1 share, top-3
share and quintile span. **The live state feature is not earning the model
complexity and the frozen-state fragility it introduces.** This is the
Decision-10 refit-without-state check L31/L33 pre-registered: a freeze
ablation DETECTS a loop (finding 2), but only a refit without the feature
SIZES whether the loop is worth keeping, and here it says no.

---

## 4. Multi-level evidence (per CLAUDE.md's standing rule)

- **Overall**: sections 2-3 above.
- **Per arm**: every arm's own `run_meta.json` records seeds, game-id list,
 possessions simulated and wall-clock (`results/engine_v0/usage_*_s25/
 run_meta.json`); all four completed their full 500-game x 25-seed subset
 (12,500 game rows, no dropped seeds).
- **Per possession-type**: not separately re-cut here -- usage is a single
 allocation decision per event, and `experiments.md` section 10.3 already
 reports per-class (`FGA_rim`/`FGA_jump2`/`FGA_3`/`TOV`/`FT_trip`) offline
 log-loss and top-3-gap deltas from the state fix; all five classes' state
 split-importance sits at 27.2-29.6%, so no single class is carrying this
 result alone.
- **Per player (quintile)**: section 3's "quintile span" IS the per-player
 read -- pregame `rate_total` quintiles of the players actually on the floor
 in the simulated games, merged from `asof_v2_shotshooter.parquet`. It rises
 monotonically with the driver in every arm (checked, not shown as a separate
 table for space); the offline Decision-8 slope-ratio metric (predicted span
 normalised by REALISED span) was not recomputed at engine scale within this
 session's time budget and is listed as not run in section 6, not asserted as
 a pass.
- **Underpowered cells**: none of the reads above are underpowered at n=25
 seeds x 500 games x ~10 usage events/game/team; every SE in sections 2-3 is
 computed from the full 25-seed sample, not a subset.

---

## 5. Why the offline data fix (section 10) did not predict this

`experiments.md` section 10 found every class's F1 log-loss move from
correcting `score_diff` sat inside its round-2 noise floor -- i.e., offline,
the leak looked almost harmless for this target. Section 2-3 here shows the
offline read was right about scale (no fg_make-style blowup) and silent about
DIRECTION: an offline fold scores the model against the REAL, historically
realised score sequence, which never asks the model to condition on a state it
could not have produced itself. The engine does exactly that once a game
diverges from history, which is the L27 mechanism restated at a smaller
amplitude. This is why Decision 10 makes the closed-loop check MANDATORY and
not conditional on the size of an offline log-loss delta.

---

## 6. What this run does NOT establish

- It does not select a NEW arm. `nostate` is not adopted here -- adopting it
 would require its own pre-registered offline bake-off against U1 (the
 standing bake-off rule), which this session did not run because the
 pre-registration in `experiments.md` section 11 was about the STATE
 feature question, not about re-opening the U1-vs-tree choice.
- It does not recompute the exact offline Decision-8 slope-RATIO metric
 (predicted span / REALISED span) at engine scale; section 3's "quintile
 span" is the predicted span only, reported as a directional read, not
 substituted for that gate.
- It does not run `scripts/eval_gates.py`'s full G1-G9 against season truth
 for these four arms -- section 2's comparison is arm-vs-arm on the SAME
 paired seeds and subset, which is what Decision 10 asks for (a closed-loop
 comparison), not a fresh grade against reality. The existing engine-v1
 reference numbers (`docs/tests/engine_v1_gates_F2_2025_s200_2026-09-11.md`:
 top-1 FGA share 0.2573 at 50 seeds on a different, larger sample) are
 consistent in order of magnitude with this run's `reference` arm (0.2526 at
 25 seeds on the 500-game stride subset) and are cited for scale only.
- 2026 was never read; only F2 2025 (test season) and 2024 (F1 train season,
 offline only) entered any computation in this lane.

---

## 7. Recommendation

1. **Do not wire `tree_v3` or `tree_v3_freeze`.** Both fail the pre-registered
 decision rule (condition (b): live/frozen differ beyond noise; condition
 (c): live does not beat refit-without).
2. **The served default stays `ENGINE_USAGE=reference` (U1 proportional).**
 This gate does not change it; that switch is the PM's, per this lane's
 standing instructions.
3. **Open a new, separately pre-registered round** treating
 `tree_v3_nostate`'s architecture (LightGBM over the five, alternative
 features only, no engine-produced state) as a fresh U-series candidate
 against U1 -- offline first (log loss, calibration, Decision-8
 responsiveness, top-3 gap), THEN its own Decision-10 closed-loop confirmation
 before any adoption. Both today's live/frozen boosters and the nostate
 boosters are already on disk (`data/processed/models/usage_v3/lgbm_tree/`,
 gitignored per `.gitignore`, rebuildable from `scripts/
 train_usage_v3_save_tree.py`) and can seed that round without a refit.
