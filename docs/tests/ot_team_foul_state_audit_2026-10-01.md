# OT team-foul state audit (lane F, 2026-10-01). Diagnose only: no trainer, table or served artifact changed.

Script `scripts/diag_ot_team_foul_state_v1.py`, output `results/ot_team_foul_state.json` (gitignored). Seasons are end years (2025 = 2024-25). Fold 1 = train 2022-2023, test 2024; fold 2 = train 2022-2024, test 2025.

## Answer in one paragraph

The rule says CARRY: men's team fouls reset only at the end of the first half (NCAA/NFHS Major Rules Differences 2025-26, NCAA Men column: "Team Fouls Reset: End of the first half"), so an overtime continues the second half's count and a team that reached the bonus in the second half is in it from the first OT possession. The ENGINE carries (`engine/loop.py:689` resets at halftime only; `:708-711` carries into OT). The TRAINING DATA does not: the possession builder resets at every period boundary including each overtime (`pbp/possessions.py:661`, `:669`), and the event layer counts "prior personal fouls in the current period" grouped by (game, period) (`models/event_stream.py:373-394`). In the training tables 99.8% of games' first OT possession shows a defence foul count of 0 (1,292 games with OT), while the second-half count the defence actually carried averaged at least 9.1 (lower bound). Every served model that reads a bonus or foul-count feature was therefore trained on reset counts on OT possessions and is served carried counts in the sim: a train/serve mismatch, on 95% of OT possessions. It touches OT possessions only (about 0.9% of possessions), so it cannot move how often games reach OT (that is decided by regulation, where the state is correct); it can move scoring and foul/FT behaviour once in OT.

## OT possessions affected (state differs between the table's reset count and the carried count)

Carried count = table count + the defence's end-of-regulation count (lower bound: the max count the table shows in period 2, so fouls in the last possession are missing; the true number affected is slightly higher). "Bonus state differs" = defence at 6+ prior fouls under one count and not the other.

| scope | OT games | OT possessions | bonus state differs | double-bonus state differs | any differs | share of OT possessions |
|---|--:|--:|--:|--:|--:|--:|
| 2022 | 303 | 6,525 | 6,269 | 4,498 | 6,305 | 96.6% |
| 2023 | 341 | 7,037 | 6,689 | 4,769 | 6,733 | 95.7% |
| 2024 | 335 | 7,139 | 6,816 | 4,982 | 6,857 | 96.0% |
| 2025 | 313 | 6,848 | 6,541 | 4,937 | 6,592 | 96.3% |
| FOLD 1 train (2022-2023) | 644 | 13,562 | 12,958 | 9,267 | 13,038 | 96.1% |
| FOLD 1 test (2024) | 335 | 7,139 | 6,816 | 4,982 | 6,857 | 96.0% |
| FOLD 2 train (2022-2024) | 979 | 20,701 | 19,774 | 14,249 | 19,895 | 96.1% |
| FOLD 2 test (2025) | 313 | 6,848 | 6,541 | 4,937 | 6,592 | 96.3% |

Training tables flag 1.3-1.5% of OT possessions as in the bonus (`off_in_bonus`); the carried count puts at least 95.7-96.6% in it.

## Which rule is right, from the feed itself

Free throws per possession (FTA/poss) from the possession design: regulation second half, defence at 7+ fouls (in the bonus): 0.515 (n 440,246); at 3 or fewer: 0.224 (n 661,988). Overtime possessions: 0.552 when the CARRIED defence count is 7+ (n 25,035, i.e. 96% of OT possessions), 0.358 when it is 3 or fewer (n 88, underpowered). Overtime FT volume sits at the bonus level of regulation, not at the no-bonus level that a reset count would predict. The feed agrees with the rule: carry is correct, the table is wrong on OT.

## Which count each model's trainer read on OT possessions

| consumer | where the feature comes from (file:line) | trained count on OT | served count in sim | status |
|---|---|---|---|---|
| possession outcome (PO) | `models/possession_outcome.py:450` `in_bonus = off_in_bonus` from the builder's chances table | reset | carried (`engine/loop.py:201`, `state.in_bonus()`) | MISMATCH (traced) |
| free throw | `models/free_throw.py:562` `in_bonus` from `foul_class`, which comes from `trip_prior_fouls` = event_stream period count | reset | carried | MISMATCH (traced) |
| clock | `models/clock.py:524` `in_bonus = off_in_bonus`; `clock_v3.py:902` also has `is_ot` | reset | carried | MISMATCH (traced) |
| fg_make, rebound | `models/fg_make.py:403-404`, `rebound.py:235-236`: `ES.in_bonus(fouls_opp_prior)`; `event_stream.py:385` groups by (game, period) | reset | carried | MISMATCH (traced) |
| foul joint R7 / R8 / R9ao3 accrual and trip offsets | `scripts/train_foul_joint_v1.py:84,202` `def_team_fouls_true` (the builder's reset count); engine cells `engine/foul_joint.py:122` `pi` has its own OT cell, `:156` pools H2 + OT for trip offsets | reset | carried | MISMATCH (traced). The OT cell was fitted on near-zero counts and is read at carried counts. R9ao3's and-one model uses clock and period, not the count. |
| shot block (K2_Ocell) | `scripts/train_shot_block_v1.py:78` feature `in_bonus` from the event design | reset | carried | MISMATCH (inferred from the shared event design, not traced line by line) |
| late game (rounds 1-3) | design `late_game/design_v1.parquet` from the possessions table; `eg_role6_code` uses `per >= 2` (OT included), `in_double_bonus` read at `engine/late_game_adapter.py:140`; `in_window` is period 2 only | reset in OT where used | carried | MISMATCH for any arm that reads OT possessions (inferred for the design file); round 1's window excludes OT |
| rotation | player fouls only, no team count | n/a | n/a | not affected |

## What this means for lane L (OT rate, late-game round 3)

- The mismatch cannot explain the G7 OT-rate gap (0.0305 vs 0.0557): whether regulation ends tied is decided before any OT possession, and regulation counts are correct in both table and engine.
- It does bear on what happens after OT starts (points per OT possession, number of OT periods, FT rate in OT). Any round-3 arm that adds OT possessions to its training window or reads a foul-state feature there inherits the reset counts unless the design uses carried counts.
- A fix is a data-layer correction (carry the second-half count into the OT segment in `possessions.py` and `event_stream._attach_team_fouls`, then rebuild the tables and refit every consumer above), i.e. a retrain-set item. Not proposed as a hot patch, and not done.

## Not done / limits

No sim was run to size the effect on scores. The carried count is a lower bound. Shot block and late-game rows are inferred from shared designs. 2025-26 was not touched.
