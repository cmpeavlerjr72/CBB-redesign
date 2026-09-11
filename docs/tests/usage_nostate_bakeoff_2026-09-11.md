# usage: `tree_v3_nostate` vs served U1 -- adoption bake-off graded against truth -- 2026-09-11

Author: Sonnet worker (usage round-3 lane), 2026-09-11, 11:55-12:45 ET.
Pre-registration: `docs/models/usage/experiments.md` section 12 (committed
`c41ec25`, BEFORE any grading number in this doc was computed). Trigger:
`docs/tests/usage_decision10_gate_2026-09-11.md` section 7 item 3 and
`docs/models/usage/experiments.md` section 11's recommendation to open a
fresh U-series bake-off for `tree_v3_nostate` against U1.

**VERDICT: DO NOT ADOPT `tree_v3_nostate`.** It clears the Decision-8 slope
ratio gate and shows no regression on `eval_gates.py` G1-G9, but it moves
FURTHER from the actual truth top-1/top-3 usage share than the served U1
allocator does, at both the player level and the team level, by far more
than the seed noise (12-14 noise floors). The concentration lift over U1
found in section 11 (arm-vs-arm, z = 4.8-13.5) is real but is a move in the
wrong direction relative to ground truth on this sample -- U1 already sits
closer to what actually happened. The served default (`ENGINE_USAGE=reference`)
is UNCHANGED; that switch is a PM decision, not made here, and this result
argues against it.

---

## 1. What was graded, and how

Two arms, the SAME F2 2025 500-game stride subset (sorted by `game_id`
ascending, every 11th row, first 500) and the SAME pinned sub-models
(`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round4_B1`,
`ENGINE_CLOCK=v3c_srfloor_P3_s1`, `ENGINE_REBOUND=s1_weekly`,
`ENGINE_FREE_THROW=s1_conf_aligned`, `ENGINE_ROTATION=reference`), 25 seeds,
6 engine workers:

| arm | `ENGINE_USAGE` | what it is |
|---|---|---|
| `reference` | `reference` | served U1 proportional allocator, unchanged |
| `tree_v3_nostate` | `tree_v3_nostate` | round-3 corrected-lineage LightGBM tree, refit with NO engine-produced state feature, live otherwise |

**Config-match check (`docs/models/usage/experiments.md` section 12.2):** the
`run_meta.json` of `results/engine_v0/usage_reference_s25/` and
`results/engine_v0/usage_tree_v3_nostate_s25/` (both produced for section 11's
Decision-10 closed-loop gate) were diffed field-for-field -- identical
pinned sub-models, identical 25 seeds (0-24), identical 500 `game_id` list,
identical subset rule. **These runs are reused byte-identical; no new engine
run was executed for this round.** This is a legitimate reuse under the
pre-registration's item (d), not a shortcut around it: the only thing this
round adds is grading the SAME simulated output against a data source
section 11 explicitly did not use (real truth), rather than against the
other arms' own simulated distributions.

Truth source: `data/processed/truth/player_game_v1.parquet` (hoopR
player_box, `cbbd_player_id` join key, 99.9% filled for season 2025),
restricted to the 500 game_ids above. Pregame driver: `rate_total` from
`data/processed/models/usage_v2/asof_v2_shotshooter.parquet`, season 2025.
`usage_events = fga + fta` on both the truth and the simulated side (TOV
dropped from this comparison ONLY because sim `players.parquet` carries no
per-player TOV column to compare against; `usage.build_usage_events` itself
is untouched and still credits `TOV` as its own class).

---

## 2. Gate (a): Decision-8 slope ratio at engine scale

Quintile bin edges for `rate_total` were fixed from the REALISED (truth)
side (5 bins, edges `[0, 0.1406, 0.1742, 0.2020, 0.2375, 1.0]`) and applied
identically to the truth table and to both arms' simulated `players.parquet`,
so "predicted span" and "realised span" are read off the SAME five buckets.

### Realised (truth) quintile table, n = 9,377 truth player-games merged to `asof`

| quintile | mean driver (`rate_total`) | mean actual share | n |
|---|---:|---:|---:|
| Q1 | 0.1047 | 0.0453 | 1,790 |
| Q2 | 0.1582 | 0.0758 | 1,789 |
| Q3 | 0.1881 | 0.0986 | 1,789 |
| Q4 | 0.2193 | 0.1295 | 1,790 |
| Q5 | 0.2774 | 0.1638 | 1,788 |

Realised span (Q5 - Q1): **11.8528 pp**. Monotone 5/5 -- a real, matchup-
specific driver-response in the truth data, not a flat line.

### Predicted quintile tables (same bins, sim `players.parquet`, 25 seeds x 500 games)

| quintile | driver | `reference` pred share | `tree_v3_nostate` pred share |
|---|---:|---:|---:|
| Q1 | 0.1115 | 0.0643 | 0.0627 |
| Q2 | 0.1585 | 0.0928 | 0.0925 |
| Q3 | 0.1881 | 0.1123 | 0.1124 |
| Q4 | 0.2193 | 0.1382 | 0.1391 |
| Q5 | 0.2714 | 0.1736 | 0.1759 |

| arm | predicted span (pp) | realised span (pp) | slope ratio | monotone steps | verdict |
|---|---:|---:|---:|---:|---|
| `reference` (U1) | 10.9307 | 11.8528 | **0.9222** | 4/4 | inside [0.85,1.15] and [0.8,1.2] |
| `tree_v3_nostate` | 11.3209 | 11.8528 | **0.9551** | 4/4 | inside [0.85,1.15] and [0.8,1.2] |

**Both arms pass gate (a).** `tree_v3_nostate` sits closer to a slope ratio
of 1.000 (0.9551 vs 0.9222) -- a real, if modest, improvement in matching
the SHAPE of the driver-response curve. This is the one reading in this
round that favours `tree_v3_nostate`.

---

## 3. Gate (b): top-1 / top-3 usage-share gap vs actual

n = 998 truth team-games in the 500-game subset (actual top-1 share mean
0.2490, actual top-3 share mean 0.5945).

| arm | sim top-1 mean +/- seed SE | gap top-1 vs actual (pp) | sim top-3 mean +/- seed SE | gap top-3 vs actual (pp) |
|---|---:|---:|---:|---:|
| `reference` (U1) | 0.2526 +/- 0.0003 | **+0.361** | 0.5973 +/- 0.0005 | **+0.279** |
| `tree_v3_nostate` | 0.2577 +/- 0.0003 | **+0.867** | 0.6063 +/- 0.0004 | **+1.175** |

Noise band for the gap DIFFERENCE (the actual side is a fixed historical
quantity with no seed variance, so only the sim-side seed SE enters):
top-1 ~0.04 pp, top-3 ~0.06 pp. Observed `nostate` minus `reference`:
**+0.506 pp top-1 (~12 noise floors), +0.896 pp top-3 (~14 noise floors)**.

**`tree_v3_nostate` moves further from truth than `reference` on both reads,
far beyond noise. Gate (b) FAILS for `tree_v3_nostate`.** Both arms
over-predict top-1/top-3 share relative to truth in this subset (`reference`
by a small amount, `nostate` by roughly 2.4-4.2x more) -- neither arm
under-concentrates here; the direction of section 11's arm-vs-arm finding
(tree beats U1 on concentration) is confirmed, but grading against truth
instead of against U1's own distribution flips which arm is preferable.

### Per-team cross-check (multi-level evidence, team layer)

347 teams have truth coverage in this subset; 342 have a matching simulated
team. Per-team top-1 share averaged across that team's games and all 25
seeds, compared to that team's own actual top-1 share:

| arm | mean \|gap\| (pp) | gap SD (pp) | teams within +/-2pp of actual | team-level corr(sim, actual) |
|---|---:|---:|---:|---:|
| `reference` (U1) | 2.605 | 3.348 | 164/342 (48%) | 0.427 |
| `tree_v3_nostate` | 2.706 | 3.367 | 149/342 (44%) | 0.421 |

`reference` is closer to truth on every count except the (statistically
tied) team-level correlation -- **the team-layer cut confirms the
player-layer finding under gate (b): `tree_v3_nostate` does not improve, and
mildly worsens, alignment with actual team usage concentration.** Neither
arm is meaningfully more matchup-specific than the other at the team level
(0.427 vs 0.421 is within sampling noise for n=342 teams); both correlations
are well above 0 (i.e., neither arm is flat against team-level usage
quality, satisfying the standing matchup-specific rule at the team layer).

### Per-class note (multi-level evidence, event-class layer)

Usage is a single allocation draw per possession event; there is no engine-
scale per-class (`FGA_rim`/`FGA_jump2`/`FGA_3`/`TOV`/`FT_trip`) split of
top-1/top-3 share to re-cut in the same way section 11.4 already explained.
The offline per-class evidence already on record (`experiments.md` section
10.3) showed the `score_diff` data fix moved every class's log loss by less
than its round-2 noise floor and all five classes' state split-importance
sat in a narrow 27.2-29.6% band, i.e., no single class drove the tree's
offline behaviour differently from the others; that evidence is cited for
context and not re-run here.

### Underpowered cells

None of the reads in this section are underpowered: 998 team-games and
9,377-176,313 player-quintile rows (truth and sim respectively) are well
above `min_cell_n`. The per-team cut (342 matched teams) is adequately
powered for a directional read but not for team-by-team individual
significance; it is reported as a distributional summary, not a per-team
significance claim.

---

## 4. Gate (c): `eval_gates.py` G1-G9 vs truth, no regression

Run directly against the existing (reused) results directories --
`docs/tests/gates_usage_reference_s25_2026-09-11.md` and `docs/tests/
gates_usage_tree_v3_nostate_s25_2026-09-11.md` (new files; neither the
engine-v1 200-seed gate doc nor any other lane's gate doc was touched).

| gate | reference | nostate | regression? |
|---|---|---|---|
| G1 -- possessions/game | FAIL | FAIL | no (means 70.030 vs 70.025, SDs 4.604 vs 4.614) |
| G2 -- PPP by tercile | FAIL | FAIL | no (cell values match to 3-4 decimals) |
| G3 -- shot mix | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION | n/a |
| G4 -- four factors | FAIL | FAIL | no (cell values match to 3-4 decimals) |
| G5 -- dispersion | FAIL | FAIL | no (margin SD ratio 0.9520 vs 0.9504; home/away corr 0.0013 vs 0.0079, both inside section-11's |z|<1 band; total SD ratio 0.7381 vs 0.7449) |
| G6 -- home margin | PASS | PASS | no |
| G7 -- OT / half split | FAIL | FAIL | no |
| G8 -- player layer | FAIL | FAIL | no status change; top-1 FGA share (gates.py's own 458-game truth-restricted convention) 0.2576 (ref) vs 0.2608 (nostate) vs truth 0.2476 -- consistent in direction and magnitude with section 3's gap-vs-actual finding |
| G9 -- spread/total accuracy | FAIL | FAIL | no (cell values match to 3-4 decimals) |

**Every gate status is identical between arms; every numeric line moves by
less than the section-11 aggregate noise band (|z| < 1). Gate (c) PASSES: no
regression from adopting `tree_v3_nostate`.** The pre-existing FAILs (G1,
G2, G4, G5, G7, G9) are known engine-v1 characteristics of this 500-game/
25-seed subset driven by clock/fg_make/event, not by usage; they are
reported for completeness per the gate line and are out of this lane's
scope to fix.

---

## 5. Multi-level evidence summary (per CLAUDE.md's standing rule)

- **Overall**: sections 2-4.
- **Per team**: section 3's per-team cross-check, 342 matched teams.
- **Per player (quintile)**: section 2's five-bucket table, 9,377-176,313
  rows.
- **Per class**: not re-cut at engine scale (usage is one allocation draw
  per event); offline per-class evidence cited from experiments.md section
  10.3, where no single class carried the tree's behaviour.
- **Underpowered cells**: none in this round's own reads (section 3);
  eval_gates.py's own UNDERPOWERED labels (month-level G1 cells, individual
  G3 team rows) are unchanged from the standard gates.yaml `min_cell_n` rule
  and are not usage-specific.

---

## 6. Decision (per `experiments.md` section 12.5's pre-registered rule)

Adoption required (a) AND (b) AND (c). (a) and (c) pass; **(b) fails** --
`tree_v3_nostate` is further from truth than `reference` on both top-1 and
top-3 usage share, well beyond the noise band, at both the player and the
team level. Per the pre-registered rule, a pass on (a) does not override a
fail on (b). **`tree_v3_nostate` is NOT ADOPTED.** No change is made to
`src/cbb_sim/engine/adapters.py`'s served default
(`ENGINE_USAGE=reference`); that switch is a PM decision and this result
recommends against it, not for it.

No change-ledger line is added: `docs/models/change_ledger.md` records
status changes when an arm IS adopted, and this round's outcome is "stays
on `reference`," which is the pre-existing status, not a change.

---

## 7. Unfinished / not established by this round

- Only F2 2025 (test) and truth entered this round; 2026 remains sealed.
- No new engine run was executed (section 1); if the PM wants a fresh,
  non-reused closed-loop pair (different seed offset) as an additional
  check, that is a follow-up, not something this round's wall-clock budget
  covered.
- The Decision-8 slope-ratio noise floor (how much `slope_ratio` itself
  moves seed-to-seed) was not separately derived in this round -- only the
  aggregate margin/home-away/possessions/total noise band from section 11.5
  was available and was used for gates (b) and (c). A seed-resampled SE on
  the slope ratio itself is a legitimate follow-up if a borderline case
  arises in a future round (this round's two ratios, 0.92 and 0.96, are not
  borderline against either band).
- This does not reopen or re-test the state-carrying arms (`tree_v3`,
  `tree_v3_freeze`); section 11's DO NOT WIRE verdict on those stands.
