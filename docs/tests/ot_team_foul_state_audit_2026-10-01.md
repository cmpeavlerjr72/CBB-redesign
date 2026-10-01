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

---

# Part 2 (same day): the corrected foul-state tables as versioned siblings, regulation identity, OT row diffs, retrain switch

Nothing existing was overwritten; no trainer default, served artifact or engine behaviour changed. The retrain on the siblings is not run.

## What was built

| sibling | where | how | consumers it serves |
|---|---|---|---|
| possessions and chances, event layer v4 + OT carry (`v4otc`) | `data/processed/possessions_v4otc/{possessions,chances}_{2022..2025}.parquet` (gitignored, 68 MB, rebuild about 4 min: `.venv/Scripts/python.exe scripts/build_possessions_v4otc_v1.py --seasons 2022 2023 2024 2025`) | `cbb_sim.pbp.possessions` machine switch `ot_foul_carry` (default off): the team-foul dict is not reset at the 2-to-3 boundary or later ones (period 1 to 2 still resets). Registered as version label `v4otc`; `VERSION_EVENT_FIXES["v4"]` and every earlier version are untouched | PO design (`off_in_bonus`), clock (`in_bonus`), FT-trip terminal classes, late-game design |
| foul state (counts, bonus, double bonus, silent / trip decomposition) on the v4otc machine | `data/processed/models/possession_outcome/round6_v4otc/foul_accrual_poss.parquet` (+ `build_report.json`; gitignored by the `round*` rule) | lane D's `build_foul_state_v4_v1.build_state("v4otc", ...)`, imported and unedited apart from the argparse choice | foul joint R7 / R8 / R9ao3 trainers, PO `in_bonus` overlay |
| event-stream team-foul counts with OT carry | `cbb_sim.models.event_stream` env switch `CBB_OT_FOUL_CARRY=1` (unset = as built); no stored table, the consumer designs are rebuilt with the env set | `_attach_team_fouls` groups by (game, 1 or 2 for periods >= 2) instead of (game, period); `trip_prior_fouls` follows | fg_make, rebound, free-throw designs (and the shot-block design that shares them) |

Machine detail that mattered: the first version carried whenever the machine's current period was 2 or later at an `end_period` row, which changed 196 regulation possessions (a halftime marker the feed labels with period 2 wiped second-half fouls). The final rule carries only when the NEXT real event is in period 3 or later (or the unannounced boundary is into period >= 3). Caught by the regulation-identity check.

## Proof: regulation rows are bit-identical

`DataFrame.equals` on the regulation rows (period <= 2), same columns and dtypes, same row keys, v4 (on disk) vs v4otc, per season (`data/processed/possessions_v4otc/build_report.json`):

| table | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| possessions, regulation rows | 715,609 identical | 747,463 identical | 756,996 identical | 755,531 identical |
| chances, regulation rows | 818,929 identical | 856,243 identical | 870,364 identical | 871,182 identical |

Foul-state table (v4 machine rebuilt in a scratch dir vs v4otc, 3,002,772 rows, keys equal): 2,975,599 regulation rows identical (`results/foul_state_v4otc_vs_v4.json`). Event stream (`scripts/diag_ot_carry_event_stream_v1.py`, flag unset vs `1`): regulation rows identical in all four seasons (1,541,127; 1,615,886; 1,648,750; 1,653,967). The default machine path is unchanged because the switch is off: `tests/test_event_layer_v4.py` and `tests/test_ot_foul_carry.py` pass.

## OT row diffs per fold (seasons are end years)

Possessions (v4 vs v4otc; nearly every OT possession changes its counts):

| scope | OT possessions | counts changed | `off_in_bonus` changed | `off_in_double_bonus` changed | `terminal_event` changed (FT trip class) | `ft_trip_ambiguous` changed |
|---|--:|--:|--:|--:|--:|--:|
| 2022 | 6,427 | 6,427 | 6,243 | 4,768 | 1,426 | 1,430 |
| 2023 | 6,934 | 6,927 | 6,632 | 5,093 | 1,542 | 1,551 |
| 2024 | 7,034 | 7,034 | 6,784 | 5,304 | 1,631 | 1,649 |
| 2025 | 6,778 | 6,778 | 6,580 | 5,250 | 1,627 | 1,635 |
| fold 1 train (2022-2023) | 13,361 | 13,354 | 12,875 | 9,861 | 2,968 | 2,981 |
| fold 1 test (2024) | 7,034 | 7,034 | 6,784 | 5,304 | 1,631 | 1,649 |
| fold 2 train (2022-2024) | 20,395 | 20,388 | 19,659 | 15,165 | 4,599 | 4,630 |
| fold 2 test (2025) | 6,778 | 6,778 | 6,580 | 5,250 | 1,627 | 1,635 |

Chances (what the PO design reads): OT chances 7,417 / 8,012 / 8,100 / 7,839 for 2022-2025; `off_in_bonus` changed on 7,218 / 7,675 / 7,824 / 7,614, `off_in_double_bonus` on 5,446 / 5,876 / 6,093 / 6,039, `terminal_event` on 1,470 / 1,578 / 1,676 / 1,685 (about 21% of OT chances: a shooting-foul trip becomes a bonus trip once the carried count is used).

Foul-state table: OT possessions per season 6,427 / 6,934 / 7,034 / 6,778; `def_team_fouls_true` changed on 6,427 / 6,926 / 7,034 / 6,778, `off_in_bonus_true` on 6,275 / 6,659 / 6,809 / 6,617, `off_in_double_bonus` on 4,768 / 5,093 / 5,304 / 5,250. Part 1 used the round-6 v2-machine accrual table (6,848 OT possessions in 2025); this is the v4 machine, whose segmentation differs slightly (6,778), so the counts are not interchangeable.

Event stream (OT event rows, all changed): 15,379 / 16,642 / 16,872 / 16,430 for 2022-2025; `fouls_opp_prior` changed on 14,813 / 16,013 / 16,223 / 15,769; `trip_prior_fouls` on 3,292 / 3,547 / 3,745 / 3,738. Mean opponent prior count on 2025 OT rows: 1.43 (reset) vs 11.18 (carry).

The trained TARGETS move too, not only the features: about 23% of OT possessions change terminal class (FT_trip_shooting vs FT_trip_bonus), so PO's OT labels and the FT-trip-class counts differ under the correct rule.

## Retrain switch (applied; default off)

`scripts/chain_full_retrain_v1.py` was re-read and `git diff` was clean immediately before the edit (no other lane's hunks). It gains `--ot-foul-carry` (default off). With it: the possessions stage builds / checks `possessions_v4otc` (via `scripts/build_possessions_v4otc_v1.py`), the `po_design`, `clock` and foul-state stages read version `v4otc`, and the subprocess environment sets `CBB_OT_FOUL_CARRY=1`; the flag is part of the tag-resume compatibility check. Three scripts gained `"v4otc"` in an argparse `choices` list and nothing else: `build_po_design_v4_v1.py`, `train_clock_chain_v1.py`, `build_foul_state_v4_v1.py`. Off path: the dry-run plan of the edited chain without the flag is byte-identical (28 lines, variant F_T, timestamps stripped) to the plan of the committed HEAD version. With the flag the planned commands show `--poss-version v4otc`, `--machine v4otc`, `build_possessions_v4otc_v1.py` (dry run only; nothing run).

NOT wired (stated, not hidden): (1) the fg_make / rebound / free-throw / shot-block designs are prebuilt files in the chain (`FG_DESIGN`, the rebound design) and are not rebuilt by it; to carry OT fouls into them, rebuild their designs with `CBB_OT_FOUL_CARRY=1` in the environment and point the retrain at them. (2) Rotation keeps reading `possessions_v4` (on-floor lineups, not foul counts). (3) The engine-inputs event-block replay (`inputs_base`) is team-rate features and was not checked for a foul-count dependence. (4) The size of the effect on served models needs the retrain; not done.

## Files

`src/cbb_sim/pbp/possessions.py` (switch + version registry), `src/cbb_sim/models/event_stream.py` (env switch), `scripts/build_possessions_v4otc_v1.py`, `scripts/diag_ot_carry_event_stream_v1.py`, `scripts/chain_full_retrain_v1.py` (+ three `choices` edits), `tests/test_ot_foul_carry.py`, `.gitignore` (one line for the 68 MB sibling dir).

---

# Part 3 (same day, task 4): every foul-state consumer wired to `--ot-foul-carry`, OT parity stage, OT columns

Code at commit `8080bda` and later. Nothing existing was overwritten; the default chain plan is byte-identical to the committed one (56 lines, F_R and F_T).

## What the switch now does (`scripts/chain_full_retrain_v1.py --ot-foul-carry`, default off)

| consumer | before (part 2) | now |
|---|---|---|
| possession outcome, clock, foul-state overlay | `possessions_v4otc`, machine `v4otc` | unchanged |
| fg_make | prebuilt `design_v2_shotshooter.parquet` (reset counts), NOT rebuilt | new stage `otc_designs` rebuilds the fg events (`FG.build_fg_events`) and design with `CBB_OT_FOUL_CARRY=1`; `fg_design` starts from the carried design (the ratings swap runs on top as before) |
| rebound | prebuilt `design_round3.parquet`, NOT rebuilt | same stage rebuilds `rebound/events_v1.parquet` and the round-3 design (`build_rebound_round3_design_v1.py`, env-redirected input and output dirs, default unchanged) |
| free throw | served S1 schedule, not retrained by the chain | new stage `ft_train`: `train_free_throw_s1_fold_v1.py --fold F2 --attempts <carried attempts>` (new default-off `--attempts`), one serial process; `build_engine_inputs_chain_v1.py --ft-manifest` writes the override `adapters.FT_S1_MANIFEST` so the gate serves it |
| shot block | design shares the event layer | not retrained by the chain (the K2_Ocell table is a served lookup, not trained here); its `in_bonus` feature reads the carried count only if its design is rebuilt: NOT wired |

New files: `scripts/build_ot_carry_designs_v1.py` (builds the carried events / designs / attempts and writes `build_report.json` with the default-path identity vs the stored tables and the regulation / OT diffs), `scripts/diag_ot_carry_parity_v1.py` (the parity stage extension), `scripts/diag_ot_carry_read_v1.py` (OT block of the paired read), `tests/test_ot_carry_chain.py`.

## Parity stage now compares OT rows

With the switch, `st_parity` additionally runs `diag_ot_carry_parity_v1.py`. It recounts each team's carried prior fouls from the plays independently of the stream's own switch (a groupby / cumsum with the segment rule period 1 | periods >= 2) and compares, on regulation AND overtime rows, the carried `off_in_bonus` / `off_in_double_bonus` of the fg and rebound event tables, joined on (cbbd_game_id, period, seconds_remaining, offence side) with repeated keys dropped. FAIL rule: mismatch rate <= 0.5% in regulation and in overtime, and at least 5,000 joined OT rows per table (so the sample always contains the OT rows). Result on the built tables (`results/ot_carry_parity.json`): fg events 0 of 2,008,625 regulation and 0 of 15,327 OT rows mismatch (trained OT bonus share 0.9755 = expected 0.9755); rebound events 0 of 1,313,437 and 0 of 10,794 (0.9753 = 0.9753). NEGATIVE control, the stored (reset) fg table: 14,952 of 15,327 OT rows mismatch (97.6%), regulation 0: the check fails it. The chances tables' OT rows are reported informationally (OT `off_in_bonus` share 0.971-0.985, double-bonus 0.73-0.77 by season).

## Builder identity and regulation proof (`results/` / `data/processed/models/otc_designs/build_report.json`)

Default path (flag unset) rebuilt from raw vs the stored tables, 2022-2025: fg events 2,241,195 rows, rebound events 1,549,406, FT attempts 809,294: every column equal (`DataFrame.equals` is False only through dtype round trips; `differing_columns` is empty, `value_identical`). Carried vs default rebuild: regulation rows differing 0 in all three and in both designs; OT rows differing: fg events 17,253 of 17,660 (`off_in_bonus` 17,235, `off_in_double_bonus` 12,687), rebound events 13,511 of 13,801, FT attempts 14,322 of 14,521 (`trip_prior_fouls`, `foul_class` 13,551). Designs (carried vs stored): fg 17,243 of 17,650 OT rows (regulation 0 of 2,222,028), rebound 13,243 of 13,528 (regulation 0 of 1,523,042).

## Local runs

- Dry run: default plan identical to HEAD~ for F_R and F_T; the carry plan shows `build_possessions_v4otc_v1.py`, `--poss-version v4otc`, `--machine v4otc`, `otc_designs`, the carried design paths in `fg_design` / `rb_design`, `ft_train`, `--ft-manifest`, and the OT parity command. Preflight with the switch: 29 of 29 inputs present.
- Timed slices (1 core): `build_possessions_v4otc_v1.py` 218 s (4 seasons), foul-state replay 73 s on 2 workers, `otc_designs` 440 s, the `ft_train` stage end to end 452 s (29 S1 segments, same refit dates as the served manifest; fold-2 log loss 0.5752, calibration and responsiveness checks pass; manifest served through the override: FreeThrowAdapter loads it, and `build_engine_inputs_chain_v1.py --ft-manifest` writes `adapters.FT_S1_MANIFEST` into overrides.json).
- Carried smoke through the chain (`--smoke --ot-foul-carry --stages fg_design,fg_train,rb_design,rb_train`, carried tables copied in): fg_design 13 s, fg_train and rb_train smoke PASS.
- ENGINE_OT_STATS (default off) for the paired OT read: `loop.py` snapshots points, FTA, FGA and possessions at the first OT and writes `{home,away}_ot_{pts,fta,fga,poss}` as extra game columns. Off path: parity v9 PASS bit-identical (60 x 5) with the edit in place; on: the shared columns equal the off run exactly (300 games x 10 seeds), OT columns are 0 in regulation sims, 87 OT sims of 3,000 (2.9%), about 8-9 possessions per team per 5-minute period.

## Not done

The retrain itself (box request `docs/ops/box_queue/d1001_F_2.md`); the shot-block design on carried columns; sizing the effect on served models.
