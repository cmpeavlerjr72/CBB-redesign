# Engine-level stack reads: experiments (append-only)

Whole-stack closed-loop reads that no single sub-model owns: a full retrain of every served sub-model, and fold
confirmations of a served stack. Sub-model experiments stay in their own `docs/models/<model>/experiments.md`; the
E3 and ratings C sections there point here for the reads registered below. Nothing in this file adopts anything.

---

## 1. Full retrain x served stack v2, fold 2 (lane D, registered 2026-10-01 about 06:40 EDT, COMMITTED BEFORE ANY RUN)

**Question.** Last night's retrained stacks (`scripts/chain_full_retrain_v1.py`: every served sub-model the foundation
touches retrained on possessions v4, the corrected foul state and own ratings C; `docs/ops/full_retrain_chain_2026-09-30.md`)
were read with the loop-level switches of the OLD served stack. Do they beat served stack v2 when the four adopted
loop-level switches are on? This one read settles E3 (`team_rate_estimator/experiments.md` Stage C, through F_T) and
ratings C (`own_ratings/experiments.md` 4.3, "retrain the rating consumers, then repeat the paired loop", through F_R).

### 1.1 Arms (all fold 2: train through 2023-24, test 2024-25; 5,710 games; v3 tag-path inputs)

| arm | artifacts | loop-level switches | inputs |
|---|---|---|---|
| **S2** reference | served (no overrides) | served stack v2 defaults (no `ENGINE_*` set) | `data/processed/models/engine_v3` |
| **FRa** | chain tag `FR_box_v1` (F_R: served expanding-mean team rates; ratings C; v4; corrected foul state) | `--gate-stack adopted`: K2_Ocell, R9ao3, G3, KD explicit; clock = the chain's own clock stage (equals L2, 1.3) | `FR_box_v1/inputs/set/inputs` (v4 event block replay, ratings C) |
| **FTa** | chain tag `FT_box_v2` (F_T: E3 v4 table through `team_rate_adapter_v2`, the skew-free fg trainer; otherwise FRa) | as FRa | `FT_box_v2/inputs/set/inputs` |

The chain gate keeps `ENGINE_CLOCK=v5b_glat_pmean` as the KEY because the chain's clock root has the base layout;
under the chain's `CK_DIR` / `V5_PARAMS` rebinding that key serves the chain's v4 refit. Section 1.3 proves that refit
is the adopted L2 artifact byte for byte, so the clock is applied once.

Nothing is refitted or reselected for this read: FR_box_v1 and FT_box_v2 are last night's box artifacts (HF
`model_artifacts`, `full_retrain_v1/**`), and their parity stage (FT_box_v2) passed.

### 1.2 Metric, floors, decision

- **Seeds.** 200 per arm, seeds 0-199, paired (the RNG is keyed on (seed, game_id, family)). Grading `eval_gates.py
  --season 2025` with `CBB_TRUTH=verified_v1`.
- **Floors (Decision 12).** S2 re-run at seed offsets 1000, 2000, 3000, 4000 (four draws of the served v2 default; none
  exist yet) plus a 2,000-resample paired game bootstrap (`scripts/ops_pair_bootstrap_v1.py`); a line's floor is the
  larger. If the four S2 draws cannot run, the S0 draws `v3full_S0f{1..4}` are the fallback and the report says so.
- **Primary.** G9 calibration slope (target 1.0; S2 = 0.948).
- **Vetoes (must not move away from target beyond the floor):** G1 possessions mean; G4 eFG%, OREB%, FTA/FGA, TOV per
  team-game; G5 margin SD ratio, total SD ratio, home/away correlation; G9 total bias, margin bias; G6 non-neutral home
  margin; G7 OT rate. Also: no gate verdict flips PASS -> FAIL, and the count of G2 cells inside and G1 months inside
  does not fall.
- **Decision lines.**
  - **L-R (ratings C + clean foundation): FRa vs S2.** WIN if the slope moves toward 1.0 beyond its floor and no veto
    fires. NEUTRAL if the slope is within its floor and no veto fires. LOSE if any veto fires. Only WIN advances; NEUTRAL
    keeps the incumbent (ties go to the incumbent, which needs no retrain).
  - **L-E3 (E3 features): FTa vs FRa** on the same rule, floors from the S2 draws (NOT a draw SD that mixes FRa in; the
    box caveat in `laneD_3.done.md`). E3 advances only on a WIN here; FTa vs S2 is co-reported.
- **Segments (reported, not deciding):** G9 slope and margin MAE by month and in weeks 0-7 (ratings C's own cell);
  per-team responsiveness (Decision 8 quintile slopes of PPP and possessions); per-possession-type PPP (G2 cells).
- **Underpowered.** A local 500 x 25 tap of the same arms is DIRECTION ONLY (Decision 12) and decides nothing.

### 1.3 Clock: established before the read (not a selection)

`train_clock_chain_v1.py --poss-version v4 --ratings-dir data/processed/ratings` reproduces `clock/r6_L2` exactly
(design, six S1 pickles byte-equal, censoring tables, B1 sigma 0.04682263314741274); with ratings C the six pickles and
the sigma are still byte-equal (only four unused rating columns of the design differ). Script
`scripts/diag_chain_clock_vs_L2_v1.py`.

---

## 2. Fold-1 closed-loop confirmation of served stack v2 (lane D, registered 2026-10-01 about 06:40 EDT, COMMITTED BEFORE ANY RUN)

**Question.** Every full-size read of the adopted set is fold 2. Does the move SERVED_V1 -> served stack v2 replicate
on fold 1 (train through 2022-23, test 2023-24, season 2024)?

### 2.1 Arms

| arm | switches | artifacts |
|---|---|---|
| **V1** | `adapters.SERVED_V1` | fold-1 artifacts of every served sub-model; clock `v5b_glat_pmean` spec refitted for 2024 |
| **V2** | served stack v2 defaults | the same fold-1 artifacts; clock L2 spec (v4) refitted for 2024; fold-1 tables of K2_Ocell, R8b + AO3 (R9ao3), G3, KD |

Fold-1 artifacts are produced by the SAME trainers at the SAME spec with `--fold F1 --season 2024` (or a versioned
sibling that only parameterises the fold): no reselection, no refitting of a selected hyperparameter on fold 1.
Inputs: 2023-24 engine inputs built the way the fold-2 v3 inputs were (v1 -> v2 -> live replay per date, as-of 30 min
before the date's first tip), as versioned siblings; nothing fold-2 is overwritten.

### 2.2 Metric, floors, decision

- 5,640 games x 200 seeds (full size; box), seeds 0-199 paired; `eval_gates.py --season 2024`, `CBB_TRUTH=verified_v1`
  (verified finals cover 5,640 games of 2024).
- Floors: V1 at offsets 1000-4000 plus the paired game bootstrap (Decision 12).
- **CONFIRMED** if (i) no line in the 1.2 veto list moves away from its target beyond its fold-1 floor, and (ii) the three
  lines the fold-2 adoption moved beyond the floor in the target direction (G1 possessions mean down, G4 OREB% up, G9
  slope up) move the same way beyond their fold-1 floors. **NOT CONFIRMED** if (i) fails. **INCONCLUSIVE** if (i) holds
  and any line in (ii) is within its floor.
- Not confirming does not un-adopt anything; it goes to the PM.
- A local tap (500 games x 25 seeds, stride sample of 2024 verified games) is DIRECTION ONLY and decides nothing.
