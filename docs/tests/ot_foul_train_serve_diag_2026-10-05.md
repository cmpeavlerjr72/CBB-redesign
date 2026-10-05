# OT team-foul train/serve mismatch: which side is wrong (engine worker, 2026-10-05)

Task: readiness_gaps_2026-10-05.md "2027 rule constants" row and PM ruling 4 (fix at the source, default-off flag, no post-hoc OT adjustment). Re-verified on HEAD `882bf97`. Builds on `docs/tests/ot_team_foul_state_audit_2026-10-01.md` (parts 1-3), which already did the data-side build.

## Verdict: the TRAINING side is wrong; the engine (serving) side follows the rule

- Rule: `docs/ops/day1_readiness_2027_2026-10-01.md` rows 5-6, source S2 (NCAA/NFHS Major Basketball Rules Differences 2025-26, NCAA Men column): "Team Fouls Reset: End of the first half". An overtime continues the second-half count. S1 (NCAA.org 2026-05-08) announces no foul change for 2026-27.
- Engine (HEAD): `engine/loop.py:776` resets `team_fouls` at halftime only; the overtime branch (`:790-800`) does not touch `team_fouls` (carry). `state.in_bonus()` / `in_double_bonus()` read the carried count. Correct.
- Training data: `pbp/possessions.py:666-668` resets the team-foul dict at every period boundary, OT included, unless the default-off machine switch `ot_foul_carry` is set; `models/event_stream._attach_team_fouls` groups by (game, period) unless `CBB_OT_FOUL_CARRY=1`. Wrong in OT.
- The feed agrees with the rule (audit part 1): OT FTA per possession 0.552, the regulation in-bonus level (0.515), not the no-bonus level (0.224).

So no engine flag was added. An `ENGINE_OT_FOULS=reset` serving switch would make the engine match the training tables by serving the wrong rule, which is the post-hoc pattern ruling 4 bans. The only code change is a stale comment (`state.py:128` said "reset every period"; it now says halftime only, carries into OT). It's a comment-only change, so behaviour and parity are unchanged.

## Served consumers that need a retrain on carried OT state

| served model (default key) | foul-state input | trained on |
|---|---|---|
| possession outcome (PO event round 2 S1) | `in_bonus`, `in_double_bonus` (state block, `loop.py:193,225`) | reset |
| clock `v5b_r6L2_glat_pmean` | `in_bonus` (`clock_adapter_v3.py:393,496`) | reset |
| fg_make `round4_B1` | `in_bonus` from event-stream prior fouls | reset |
| rebound `s1_weekly` | same | reset |
| free throw `s1_conf_aligned` (trip class, `foul_class`) | `trip_prior_fouls` | reset |
| foul joint `R9ao3` (accrual OT cell `pi=2`, trip offsets `h = period>=2`) | `def_team_fouls_true` | reset (OT cell fitted near 0 fouls, served at about 11) |
| shot block `K2_Ocell` | `in_bonus` | reset (inferred; its design is NOT wired into the carry switch) |
| late game (G3 adapter, `late_game_adapter.py:269`) | `in_double_bonus` | reset where OT rows are used |

New point not in the 10-01 audit: the rotation hazard families (`ENGINE_ROTATION=round4..round9`, `models/rotation_v4.py` feature `team_fouls_frac`) also read the team-foul count and were fitted on the reset table (the adapter records this as stated divergence 4). They are default-off; the served rotation (`reference`, R2) does not read team fouls. If a hazard arm is ever adopted, its retrain needs `possessions_v4otc` too; the carry chain keeps rotation on `possessions_v4`.

## Status of the source fix (already built, default off, not run)

`chain_full_retrain_v1.py --ot-foul-carry` (lane F, 2026-10-01) retrains PO, clock, the foul-state overlay and R9ao3 inputs, fg_make, rebound and free throw on carried tables, with regulation rows bit-identical to the stored tables and an OT parity stage. Checked today on HEAD: carry dry-run plan lists `v4otc`, `otc_designs`, `ft_train`, the OT parity stage; preflight 29/29 inputs present. Not wired: the shot-block and late-game designs. The full-size read is the existing box request `docs/ops/box_queue/d1001_F_2.md` (FRc/FTc vs FRb/FTb, 200 seeds, `ENGINE_OT_STATS=1`), still unrun (no `FRc`/`FRb` results on disk). No retrain was run today, and no new box spec was written because d1001_F_2 already covers it.

Scope note: this cannot move the G7 OT rate, because regulation state is correct on both sides. It can only move scoring and FT volume inside overtime (about 0.9% of possessions).
