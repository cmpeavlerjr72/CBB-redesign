# Variance spec FACT PACK: per-game class-mix latent (possession outcome) + clock variance function (VD1)

Written 2026-10-27 by a research worker for the PM, who writes the pre-registered spec. Facts only: every number is quoted from a file, every code reference is file:line at HEAD `5ad33cb` (working tree has an unrelated uncommitted edit to `src/cbb_sim/engine/shot_block.py`, not read here). No fits, no engine runs. Where something was not measured it says ABSENT. The PM wrote none of the cited lines in this file.

Sources read: `PROJECT_STATUS.md` 2026-10-27 / 10-14 blocks, `docs/tests/shot_location_round2_2026-10-27.md` block V, `results/shot_location_v2/grade_V.json`, `docs/SIM_GUARDRAILS.md`, `docs/models/clock/experiments.md` (s16, s21-22, s41-52), `docs/tests/clock_env_round10_full_2026-10-14.md`, `docs/ops/box_queue/d1001_H_3.done.md`, `docs/tests/adoption_clock_K2_2026-10-07.md`, `docs/models/shared_shooting`, `docs/models/shared_possession`.

---

## 1. How the served engine draws the terminal class per chance

**Call chain (one chance, vectorised over all active simulations):**

| step | file:line | what happens |
|---|---|---|
| streams | `src/cbb_sim/engine/loop.py:268` | `book = StreamBook(seeds, gids)`; one counter-based stream per (seed, game_id, family), families listed `src/cbb_sim/engine/rng.py:41-56` (`"event"` at :43). Keys come from `cbb_sim.control.rng.stream_keys(seed, game_ids, family)` (`control/rng.py:60`), built in `StreamBook.__init__` (`rng.py:~85-101`). |
| state block | `loop.py:518-526` | `xx = _state_block(...)`; `is_transition`, `chance_elapsed_s` overwritten from the live clock draw (first chance: `e1`; continuation: `ce_lut`). |
| probabilities | `loop.py:527-528` | `probs = ad.event.predict(t_off, xx, is_first, gidx[rows], off[rows])` (or the `kidx=` variant under Stage C). |
| foul-joint trip shift | `loop.py:529-532` | `probs = fj.adjust_trips(probs, period, def fouls, off fouls, CLS_FT_BONUS, CLS_FT_SHOOT, rows, def_side)` (served `R9ao3`; shifts ONLY the two FT-trip logits by the fitted (half, def fouls, foul-diff) offset and rescales the other four classes, `foul_joint.py:152-184`). |
| class draw | `loop.py:533` | `cls = categorical(book.draw("event", a_rows), probs)`: inverse-CDF on the (m, 6) matrix with ONE fresh `event` uniform per active chance (`rng.py:134-141`; `StreamBook.draw` `rng.py:103-115` advances the counter per draw). |

Literal lines (`loop.py:527-533`):

```
probs = (ad.event.predict(t_off, xx, is_first, gidx[rows], off[rows]) if kk is None
         else ad.event.predict(t_off, xx, is_first, gidx[rows], off[rows], kidx=kk[rows]))
if fj is not None:
    probs = fj.adjust_trips(probs, st.period[a_rows], st.team_fouls[a_rows, dfn[rows]], ...)
cls = categorical(book.draw("event", a_rows), probs)
```

**The model behind `ad.event.predict`:** `EventAdapter.predict`, `src/cbb_sim/engine/adapters.py:368-411` (`class EventAdapter` at :232). Served mode is `round2_s1` (default, `adapters.py:1097`: `ev_mode = os.environ.get("ENGINE_EVENT", "round2_s1")`). Note this is a LIVE batched `predict_proba` per (population, monthly refit), not a lookup table (`adapters.py:408-411`):

```
team_r2 = self.team_block[gidx, off] if kidx is None else self.team_block_k[kidx, gidx, off]   # adapters.py:395
...
out[r] = models[k].predict_proba(np.ascontiguousarray(m, dtype=np.float32))                     # adapters.py:411
```

- Populations: `first` = LightGBM multiclass (21 features), `cont` = nested binary GLM cascade (22 features, adds `chance_number`). Classes in fixed order `PO.CLASSES = (TOV, FGA_rim, FGA_jump2, FGA_3, FT_trip_shooting, FT_trip_bonus)` (`src/cbb_sim/models/possession_outcome.py:81-84`; class constants `loop.py:75-81`).
- What the per-chance probability depends on (from `data/processed/models/engine/event_round2_s1_F2_2025/index.json`): 16 STATIC team columns read from `team_block[gidx, off]` (shape (G, 2, 16)): `off_3pa_c, off_rim_c, off_tov_c, off_ftr_c, opp_def_3pa_c, opp_def_rim_c, opp_def_tov_c, opp_def_ftr_c, off_rating_off_c, off_rating_def_c, def_rating_off_c, def_rating_def_c, site_home, site_away, season_idx, days_since_start`; plus 5 LIVE state columns `period, seconds_remaining, score_diff, in_bonus, is_transition`; plus `chance_number` on `cont`. The team block is served from the inputs-v3 replay sibling (`adapters.py:130-145`, `ENGINE_EVENT_TEAM_BLOCK` default `v3`). Monthly refit index is a per-game constant from the manifest (`adapters.py:402`); refit dates 2024-11-01 ... 2025-04-01 (6 segments).
- So per-chance P depends on: (game, offence side) static block + live game state. Nothing is shared across chances of a game except the static block and the monthly refit. There is no per-simulation random effect on the class probabilities. (This is the defect block V names.)
- Already-existing per-simulation hooks on this exact path: (a) Stage C `team_rate_draw` perturbs `team_block` per simulation via index `kidx` into a (K, G, 2, 16) array (`team_rate_draw.py:1-24`, picked once per game from the `team_rate` family; DEFAULT OFF, `loop.py:271-273`). (b) `anchor_first` (G, 6) raw-score offset added before softmax, per game (`adapters.py:403-408` region, `season_anchor_serving.py:80-101` `predict_with_offset`; DEFAULT OFF). (c) `fj.adjust_trips` adds a per-(simulation, defence-side) log-odds shift `lnA[rows, def_side]` to the two FT-trip logits (`foul_joint.py:167-170`) when a whistle arm is on; the served `R9ao3` has NO whistle (`foul_joint.py:56-58`, whistle only in `R8bS`/`R8bU`), s2 = 0.00421 (`whistle_var_v1.json` `served_s2_F2`).

**Where a per-game class-mix latent would have to enter (arrays / dimensions):**

1. Between `ad.event.predict` and `categorical` (`loop.py:527` and `:533`): a per-simulation vector `u[n_sim, 6]` (or 4 free logit shifts for TOV / rim / jump2 / 3PA with the FT-trip logits held) indexed by the simulation row, i.e. `u[a_rows]`; same indexing pattern as `lnA[rows, ...]` (`foul_joint.py:168`) and `SharedShooting.u[rows, side, type]` (`shared_shooting.py:116-120`). Side index `off[rows]` is available at the call site if a per-offence latent is wanted (`kk`, `gidx`, `off`, `dfn` are in scope at :527).
2. Order relative to the foul block: the FT-trip logits are shifted AFTER `predict` by `fj.adjust_trips`. A latent applied before `adjust_trips` would be rescaled by it; applied after, the four non-FT classes must be rescaled so the FT-trip masses are untouched (this is how FT trips are "held fixed", see section 7).
3. Draw site: `init_*` at game start, like `fj.init_whistle(seeds, gids)` (`loop.py:306`) and `ssl.init_game(seeds, gids, book.keys["clock"])` (`loop.py:333`). Each uses its OWN stream family via `StreamBook(seeds, game_ids, families=(FAMILY,))` and `draw_block(...)` (`rng.py:113-126`, `foul_joint.py:108-110`, `shared_shooting.py:68`). A new family name would not shift any other family's stream (`rng.py:36-38` comment), so paired arms still difference game by game. "event" is NOT the recommended family (a new `ordinal` on it would work: see the clock latent's `1 << 40` trick, section 2, but every existing `event` draw uses ordinals 0, 1, 2, ...).
4. K2 coupling that exists regardless of the latent: the clock draws the first-chance time BEFORE the cascade (`loop.py:470-481`) and, in K2 (served), re-draws it after the cascade conditional on the chance-1 end class (`_clock_cont`, `loop.py:108-127`, `_end1_code` :108-114). So a shifted class mix changes possession durations (and hence possession count) mechanically through the served K2 law; this is a mechanism, not a draw-level correlation.
5. Block V scored the model probabilities BEFORE `adjust_trips` (and before any other loop-level shift); see section 4.

---

## 2. The pace latent: template for a class-mix latent

**Where.** `LatentClockAdapter`, `src/cbb_sim/engine/clock_adapter_v3.py:780-975`; served mode `v5b_r8K2_glat_pmean` (`adapters.py:128` `CLOCK_DEFAULT`; `ADOPTED_MODES` :693-714 includes K2). Adopted: `v5b_glat_pmean` 2026-09-11 (ledger), L2 2026-10-01, K2 2026-10-07 (`docs/tests/adoption_clock_K2_2026-10-07.md`).

**Distribution.** One log-normal scale `A` per simulated game, multiplies every drawn duration (first chance, K2 redraw, OREB continuations) and rounds (`clock_adapter_v3.py:916-936`, :947-960, :968-974):

```
u = uniforms_at(k, np.full(len(k), LATENT_ORDINAL, dtype=np.int64))      # :922, LATENT_ORDINAL = 1 << 40 (:776)
m = float(loc_for(self.sigma, self.loc_kind, self.log_c))                # :927
return np.exp(self.sigma * ndtri(u) + m)                                 # :928  (served path)
d = np.clip(np.rint(self._latent(keys, team) * t), 0.0, float(CK.DURATION_CAP))   # :956
```

- `A = exp(sigma z + sigma^2/2)` (`loc_kind = "plus_half"`, `models/clock_v5.py:89-101`): location chosen so E[1/A] = 1, i.e. the possession COUNT mean is preserved (`E[A] = e^{s2}`), not the duration mean. This is the Jensen-correct location for a latent entering through a reciprocal; a logit-normal class latent has its own mean shift on the probability scale (section 7).
- Served sigma (K2, F2): `B1_sigma = 0.04710077845445259`, `B1_sigma2 = 0.0022184833` from `data/processed/models/clock/r8_K2/v5b_bakeoff/v5b_bakeoff_report.json` (`params.F2`, 4,479 training games, 605,911 training possessions). Identity: same value in the adoption note ("B1_sigma is equal (0.04710077845445259)").
- One draw per simulation, game-level: `keys` is `book.keys["clock"][act]`, i.e. the (seed, game_id, "clock") key; `loop.py:475-476`. Both teams share it ("one pace realisation per simulated game").
- Draw-time correlation: NONE with any other draw. The uniform sits at ordinal `1 << 40` on the `clock` stream; the possession counter never approaches 2^40 so it cannot collide with a duration draw (`:771-776`). It is a function of (seed, game_id) only; independent of the `event`, `fg_make`, `usage`, `rebound` streams (different family keys). The only dependence on other models is mechanical (durations -> K2 end-class law, chance time `KD`, state features `seconds_remaining`).
- Optional dispersion function (`sigma_fn = "pace2"`, `:896-936`): `sigma^2 = b0 + b1 (t - tbar) + b2 (t - tbar)^2` in the PREGAME tempo feature `tempo_prior_game` with the B1 location applied row-wise at that row's sigma; tested as round 5d and NOT adopted (ledger row at `change_ledger.md:237`: L9 1.029 -> 1.041).

**How the SD was fitted** (`experiments.md` s16 "Fitting rule", ~:2017-2032; s21-22): method of moments on the fold's TRAINING seasons only, per S1 refit, using only rows with `game_date <= max_train_date_k` and the artifact's own predictions: `s2 solves mean_g[ V_iid(g)(1 + s2) + s2 * mbar_g^2 ] = Var_g(rbar)` (arm A1/B1), where V_iid is the i.i.d. variance of the game's mean duration under the conditional law and `Var_g(rbar)` the realised between-game variance of the game-mean duration residual. Parameter, not an output adjustment (guardrails section 5). For the env round the latent sigma is refit on the arm's law with `exp_clk5b_mean_consistent.py --fit-only` (`experiments.md` s41, "the B1 latent sigma refitted on the arm's law ...").

**How it was validated.** Offline blind score (`clock_v3.score_arm_v3`, unedited) on per-game possession SD ratio 0.6877 -> 1.0043 (A1, 56 floors; s21 text) and 0.688 -> 1.009 (B1); 25-seed paired closed loop: possession SD ratio 0.804 -> 1.032, total SD ratio 0.736 -> 0.823, `corr(home, away)` 0.003 -> 0.101, `corr(P, eFG%)` -0.211 -> -0.122, G1 possession mean +1.704 -> +1.649 (ledger `change_ledger.md:230`). The model-level fact the latent was built to honour: it fills the "98.9% of the per-game Var(Dbar) gap is a within-game LEVEL" gap (s21). Evidence docs: `docs/tests/clock_duration_dispersion_2026-09-11.md`, `docs/tests/clock_mean_consistent_latent_2026-09-11.md`, `docs/tests/pace_efficiency_sign_2026-09-11.md`. Since K2 (10-07) the pace latent has not been re-validated for G1 SD alone: served G1 SD is 5.629 vs actual 5.474 (section 3 / 5).

**Correlation with other latents.** The pace latent is drawn independent of everything. The only coupling ever PRE-REGISTERED to it is shared shooting arm `GP`: `u = s (rho z_pace + sqrt(1-rho^2) z)` reading `z_pace` from the clock key at `LATENT_ORDINAL` without advancing it (`shared_shooting.py:12-14, 75-84`); fitted `GP_rho = 0.0903` (`data/processed/models/shared_shooting/params_v1.json`). `GP` was NOT eligible; the served arm is `G3` (below).

**The closest served precedent for a class-mix latent (same lineage, same RNG pattern):** `SharedShooting` arm `G3`, ADOPTED 2026-10-01 (`shared_shooting.py:124` `DEFAULT = "G3"`; `docs/tests/adoption_served_v2_2026-10-01.md`): a per-game 3-vector logit shift on the fg_make probability of (rim, jump2, three), `u ~ N(0, Sigma)` via eigen-factor of `G3_Sigma`, own family `shared_shooting` (`shared_shooting.py:61-100`), identical for both teams' attempts (shared), applied by `shift(p, rows, side, type_idx)` (`:116-120`) at `loop.py:592` (`p_make = ssl.shift(...)`). Fitted values (`params_v1.json`, fold-2 train, SD on the make logit): rim 0.141 (var 0.01994), jump2 0.154 (0.02367), three 0.0653 (0.00427); covariances rim-jump2 -0.0036, rim-three +0.0018, jump2-three +0.0045. Not mean-corrected (Jensen shift is read through realised rates). This shows (i) a joint multivariate latent with a fitted covariance matrix has already been served, (ii) `params_v1.json` records `offline_winner: G3`, `eligible: [G3]` (the per-team control `U1` was a control, not an eligible arm).

**Prior shared TOV latent evidence** (`docs/models/shared_possession/experiments.md` s2, 2023-24 pooled): measured shared (between-team, game-level) TOV logit s2 = 0.0074 (SE 0.0011), OREB shared s2 = 0.0013 (SE 0.0016, not detectable), OREB x TOV term -0.0046 (0.0009). `T1` / `OT` were eligible offline (+2.23 / +2.68 floors fold 2) but NOT WIRED under the pre-condition that the served engine must UNDER-produce the between-team TOV covariance in points (sim 0.84 vs actual 0.76 pts^2). Block V's TOV latent (s2 0.0221, SD 0.149) is per team-game, a different unit; the two are not decomposed anywhere.

---

## 3. Clock variance function / VD1

**What it is.** The name `VD1` exists ONLY in the clock s46 draft and its promoted section: `docs/models/clock/experiments.md:4924` and `experiments_DRAFT_endgame_clock_followup_2026-10-14.md:71`. It is not defined in `docs/models/clock/model.md` (no match). Quote (`experiments.md:4924`):

> `| D arm (optional, Q4) | VD1: game-latent sigma and chance-count dispersion refit with the s41.B estimator; the SD is scored, never matched; a variance-function arm, not a scale |`

Costed at (`:4960`) "D (VD1, only if approved) | local | 1 | 1 | $1.5 - $5". PM ruling 4 (`:4983`): "VD1: NOT this round. G1 possession SD is a registered line only (veto for A and B, co-primary for C). The variance function gets its own spec if finding (5) survives this round." Block D is "NOT RUN (ruling 4)" (`:5162`).

**Finding (5) as registered** (`experiments.md:4866`): "Served control G1 possession SD already 5.629 vs actual 5.474 (floor 0.014; +0.155 = 11 floors); owner = clock latent / variance function."

**What the 10-13 / 10-14 blocks registered:**

- G1 line (`experiments.md:4906-4912`): "Served control: 5.629 vs actual 5.474 (floor 0.014; 11.1 floors too wide); E1 5.650. Printed in every table header and registered for EVERY arm"; role: veto for A, B; co-primary for C.
- Full-size read (`experiments.md` s45; `docs/tests/clock_env_round10_full_2026-10-14.md` section 4.1): G1 possession SD actual 5.474, control 5.629, E1 5.650, floor 0.014, FAIL (E1 away by 1.5 floors); 50-seed-draw read 5.632 / 5.659 / 0.020. G1 possession mean actual 67.875, control 68.748, E1 68.728, floor 0.023.
- Structural finding (`experiments.md:5154`): "Every arm that moves the env slope also raises the between-game variance of expected possessions: offline SD +0.04 to +0.06, floor 0.025-0.034. Through the variance identity that moves the loop G1 SD away from 5.474 (translation 5.629 -> 5.65-5.66), as E1 did at full size (5.650). The G1 SD co-primary and the env primaries therefore conflict through the variance function: the latent sigma and chance-count dispersion (finding 5, VD1), not the env term."
- PM ruling (PROJECT_STATUS 10-14 midday, ruling 4): "VD1 (variance function) is now REQUIRED as a co-registered arm in the next ENV round, not this one: env responsiveness cannot be read without it."
- PM ruling (10-14 afternoon): "The offline G1 SD proxy is RETIRED as a gate; G1 SD is read only in the loop (screen and full size)." (The offline proxy read +0.082 for CA1 vs +0.001 in the loop.)
- Process ruling (10-14): "a closed-loop sample read is a SCREEN, never a candidate call"; the 500 x 25 floors were 5-20x wider than full size (G1 SD even had the opposite sign at 500 x 25).
- Mechanism in the fact base: the pace latent sigma is fitted by method of moments on TRAINING between-game variance of the game-mean duration residual under the arm's own conditional law (section 2). G1 SD (`poss_sd_pooled`, `scripts/grade_clock_r8_loop_v1.py:142`: `sqrt(average(poss_var + (sim - mean_sim)^2))`) is the pooled per-game possession SD. Which of (latent sigma, chance-count dispersion, K2 end-class mix) owns the +0.155 is NOT decomposed anywhere in the files read; the s46 text only names "the latent sigma and chance-count dispersion".
- Interaction to note: a class-mix latent changes the shot / TOV / FT mix and, through the served K2 end-class law, the possession duration and therefore the possession count (loop.py:108-127). Whether it moves G1 SD in the same direction as the over-dispersion (+0.155) or against it is unmeasured.

---

## 4. Possession-outcome training scheme, artifact, residual table

**Training scheme** (`docs/models/possession_outcome/model.md` s3-4):

- Walk-forward folds: F1 trains {2022, 2023}, tests 2024; F2 trains {2022, 2023, 2024}, tests 2025 (selection fold). 2025-26 sealed.
- S1: refit at the start of each calendar month of the test season on all prior seasons plus the test season to date strictly before the refit date; each test game scored by the most recent refit at or before its own date (`adapters.py:295-302` comment, `ArtifactManifest` re-derivation `adapters.py:326`). Six segments for 2025 (2024-11-01 ... 2025-04-01), `n_train` 1,908,534 (`first`, Nov) and 276,163 (`cont`, Nov).
- `first` winner: LightGBM multiclass, feature set `C_plus_state`, F2 log loss 1.51543, worst decile gap 0.98 pp, noise floor 0.000763. `cont` winner: nested binary GLM `cascade`, F2 log loss 1.49976, floor 0.001982 (no tree arm passed calibration).
- Features: the 16 team columns + 5 state columns (+ `chance_number` on `cont`) listed in section 1; every rate centred on its own as-of league mean; strictly-before expanding means; `docs/models/possession_outcome/features.md` s1.
- Engine reads the monthly artifacts from `data/processed/models/engine/event_round2_s1_F2_2025/` (`first_<refit>.joblib`, `cont_<refit>.joblib`). **The engine holds F2 artifacts only; there is no F1 served artifact set** (block V disclosure 2); an F1 replay is `scripts/build_engine_event_round2.py --fold F1 --season 2024` and is a refit, not a read.
- Honest note for any walk-forward latent fit: the registered SDs (below) were measured on the 2025 TEST season with artifacts trained through 2024. A latent fitted on training seasons needs out-of-fold per-chance probabilities for those seasons; those exist only if an S1 replay is run for the earlier season (F1 test = 2024 is the only one the registered scripts support).

**The round-2 residual table** (`scripts/diag_shot_selection_dispersion_v1.py`, graded by `scripts/grade_shot_location_v2.py --block V`; committed `c2a3504`):

- Inputs on disk in `data/processed/models/shot_location/round2/V/`: `chance_probs_F2_2025.parquet` (853,931 rows: `game_id, offense_team_id, defense_team_id, offense_is_home, neutral_site, game_date, poss_index, chance_number, y, population, p_TOV, p_FGA_rim, p_FGA_jump2, p_FGA_3, p_FT_trip_shooting, p_FT_trip_bonus`), `class_context_F2_2025.parquet` (10,890 team-games: `off_3pa_c, off_rim_c, off_tov_c, opp_def_*_c, offense_is_home, neutral_site`), `margins.parquet` (22,680 team-game margins, verified finals), `band_probs_F{1,2}.parquet`. `population` is in the chance table so a `first` vs `cont` split is derivable; the grader did NOT report one (ABSENT).
- Per team-game residual: `r = (count - sum p) / n`, implied SD `sqrt(sum p (1 - p)) / n`, ratio = SD(r) / sqrt(mean implied variance) (disclosure 4). Latent (disclosure 7): first-order logit shift `u ~ N(0, s2)` applied to every chance of a game for one class against the rest, `s2 = (mean r_dm^2 - mean implied share variance) / mean((sum p (1 - p) / n)^2)`, SD = sqrt(max(s2, 0)); each class is binary vs rest, so the class latents are MARGINAL, not a joint softmax latent; bootstrap SEs are on s2. The mean residual is separate (3PA +0.36 pp, jump2 -0.41 pp, i.e. class level shifts).
- Floors: ratio floor = max(parametric-bootstrap null SD, game-resampled SE), 200 replicates; null mean 0.999-1.001.

**Registered class results (F2 2025, 10,890 team-games)** (`results/shot_location_v2/grade_V.json` `F2.rows`; round-2 doc s0 and s1):

| class | ratio | null mean | floor | excess (floors) | latent s2 | latent SD | s2 boot SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3PA | 1.253 | 1.001 | 0.00876 | +28.8 | 0.03686 | 0.192 | 0.00142 |
| rim | 1.360 | 1.001 | 0.00970 | +37.0 | 0.05604 | 0.237 | 0.00175 |
| jump2 | 1.473 | 0.9996 | 0.0119 | +39.8 | 0.1034 | 0.322 | 0.00307 |
| TOV | 1.104 | 1.001 | 0.00730 | +14.2 | 0.02209 | 0.149 | 0.00161 |
| FT trips | 0.946 | 0.9991 | 0.00680 | -7.8 (UNDER) | -0.01404 -> 0 | 0 | 0.00168 |

**Latent SD by cell in that worker's tables:**

- By offence quintile (`F2.by_offence_quintile`, 2,178 team-games per quintile; quintiles of the offence's own as-of profile: `off_3pa_c` for 3PA and jump2, `off_rim_c` for rim, `off_tov_c` for TOV): reported as ratio and latent s2 per quintile, NO per-cell floor (ABSENT; the overall s2 SEs above are the only floors).
  - 3PA s2 by quintile: 0.0385, 0.0296, 0.0402, 0.0375, 0.0376 (ratio 1.243, 1.200, 1.272, 1.264, 1.276). Flat within about +-0.005 of the overall 0.0369.
  - rim s2: 0.0703, 0.0625, 0.0565, 0.0522, 0.0428 (ratio 1.405 -> 1.302): monotone DECLINING with the offence's rim share; Q1 is 1.64x Q5.
  - TOV s2: 0.0255, 0.0171, 0.0203, 0.0193, 0.0277 (U-shaped; ratio 1.113, 1.078, 1.096, 1.093, 1.138).
  - jump2 s2 (by `off_3pa_c`): 0.1002, 0.0974, 0.1037, 0.0915, 0.1291 (ratio 1.499 ... 1.509).
- By season: ABSENT for the class part (2025 only; no F1 served artifact). The band part has F1 and F2 (band latent SDs agree, e.g. short2 0.441 vs 0.455) but is a different latent (below).
- By site: ABSENT as a split of the latent. Only the correlation of the residual with `site_home` (neutral excluded, n 9,476): TOV -0.007, rim -0.005, jump2 +0.018, 3PA -0.005, all inside the floor 0.010 (not beyond), i.e. the class MEAN residual does not track home/away in this table.
- By chance type (`first` vs `cont`): ABSENT (pooled).
- Residual covariates (corr, Fisher floor 1/sqrt(n) = 0.0096, beyond-floor flagged): margin vs TOV -0.216 (a mechanical outcome link, residual TOV tracks losing), margin vs rim +0.077, margin vs 3PA +0.058, n_chances vs rim +0.055, 3PA -0.026, jump2 -0.023, opp-defence profile vs TOV +0.031, 3PA +0.029. Reported as description; not a decomposition.
- Class residual correlation matrix (observed vs parametric-bootstrap null, n 10,356): rim-jump2 -0.448 (null -0.281), rim-3PA -0.417 (-0.384), jump2-3PA -0.308 (-0.314), TOV-rim -0.212 (-0.263), TOV-jump2 -0.202 (-0.199), TOV-3PA -0.216 (-0.268), FT-rim -0.135 (-0.228), FT-jump2 -0.092 (-0.148), FT-3PA -0.188 (-0.206), TOV-FT -0.129 (-0.151). The excess over the multinomial null is concentrated in rim-jump2 (the 2.27 ft override boundary, hypothesis in the doc). A latent covariance across the four classes is therefore only partly implied by independent marginals.
- Cross-team within-game correlation of the class residual (is the latent game-shared or team-specific): ABSENT. The unit of the table is the team-game; the table `chance_probs_F2_2025.parquet` carries what is needed to compute it without a fit.
- Pace-vs-class-latent correlation: ABSENT (no pace quantity in the residual table; `n_chances` is the nearest, corr values above).
- Band latent (separate, not served, ruling 19): short2 SD 0.455, corner3 0.353, wing3 0.200, top3 0.259; the 3PA class residual is uncorrelated with every band residual (|r| <= 0.06 disattenuated); only coupling rim vs short2 -0.135 raw / -0.33 disattenuated.

---

## 5. Gates a variance arm must pass, and current served readings

Gate definitions: `docs/SIM_GUARDRAILS.md` s3 (G1, G3, G5; tolerances there are provisional; floors from the 200-seed control draws replace them in every closed-loop read per revision-log amendment c). Round-10 line grammar: floor = max(2 x SD of four control draws at the full 200 seeds, 2 x paired game-bootstrap SE); the 500 x 25 floors are 5-20x too wide to call a candidate (10-14 process ruling).

**Latest full-size read of the served stack v3 (control): `docs/tests/clock_env_round10_full_2026-10-14.md` s4.1 (5,710 games x 200 seeds, fold 2, four control draws). G3 / PIT / score-correlation are not in that grader; those come from the K2 full read `docs/ops/box_queue/d1001_H_3.done.md` (2026-10-01, 5,710 x 200, `ENGINE_CLOCK=v5b_r8K2_glat_pmean`, the same K2 stack as served v3 per the adoption note).**

| gate / line | served v3 value | actual | floor | source and note |
|---|---:|---:|---:|---|
| G1 possession SD | 5.629 | 5.474 | 0.014 | round 10 s4.1. Too wide by 0.155 = 11 floors (raw tolerance +-0.75 PASSES). The line the pace latent / VD1 owns. |
| G1 possession mean | 68.748 | 67.875 | 0.023 | round 10 s4.1 (raw tolerance +-1.0 PASS) |
| G5 margin SD ratio | 1.0416 | 1.0 | 0.0023 | round 10 s4.1 (1.0412 in the K2 read). Inside [0.95, 1.05] but 18 floors above 1; ABOVE the target before any new latent adds variance. |
| G5 total SD ratio | 0.9454 | 1.0 | 0.0028 | round 10 s4.1; the K2 read has 0.9734 PASS (different total definition; the round-10 grader's regulation-total line `DISP regulation total SD ratio` is 0.9936, actual 0.994 [0.974, 1.014], band 0.020). Not the same quantity; the spec must state which. |
| G5 home/away score correlation | 0.1698 (K2 read) | 0.2283 | no floor in the K2 read | FAIL on raw tolerance +-0.05; F1 K2 0.1875 vs 0.2522. A GAME-shared latent raises this (pace did 0.003 -> 0.101 at 25 seeds); a per-team latent does not. |
| G5 PIT K-S p | 0.00235 (K2 read, F2) | > 0.1 | n/a | FAIL (F1 K2: 0.044) |
| G3 3PA share (season, pooled, PROVISIONAL) | 0.3868 | 0.3906 | none; PASS inside +-1.5 pp | K2 read. -0.38 pp. |
| G3 rim share (season, pooled, PROVISIONAL) | 0.3723 | 0.3733 | none | K2 read. -0.10 pp. |
| G3 FTA/FGA (season, pooled, PROVISIONAL) | 0.3305 | 0.3295 | none | K2 read |
| G3 by team (3PA share, FTA/FGA, rim share) | NEEDS-INSTRUMENTATION | | | `0/0 powered team-metrics`; the team-level G3 the spec's G3-by-team line would need does not exist in the grader today |
| G4 pooled TOV% / OREB% / eFG / FT rate | 0.1753 / 0.2886 / 0.5098 / 0.3305 | 0.1739 / 0.2984 / 0.5086 / 0.3295 | none | K2 read, pooled PROVISIONAL |
| within-game N on OREB / FGM / TOV / FTA | -0.0074 / 0.3474 / 0.4720 / 0.2127 | -0.023 / 0.381 / 0.422 / 0.200 | 0.0013 / 0.0010 / 0.0009 / 0.0005 | round 10 s4.1 (all pass; the TOV line is the one a TOV latent moves) |
| DISP margin SD ratio, abs-spread Q1 / Q2 / Q3 / Q4 / Q5 | 1.122 / 1.040 / 1.046 / 1.069 / 0.982 | 1.0 | 0.007 / 0.006 / 0.008 / 0.006 / 0.006 | round 10 s4.1; all pass the literal 0.009 band |
| DISP pace SD ratio overall; Q1 / Q2 / Q3 / Q4 / Q5 | 1.0112; 1.009 / 1.029 / 1.026 / 1.044 / 0.933 | 1.0 | 0.0045; 0.011 / 0.0128 / 0.010 / 0.014 / 0.010 | round 10 s4.1; band [0.95, 1.05] |
| DISP regulation total SD ratio | 0.9936 | 0.994 [0.974, 1.014] | band 0.020 | round 10 s4.1 |
| G9 margin slope / margin bias | 0.9568 / -0.254 | 1 / 0 | 0.0035 / 0.022 | round 10 s4.1 |
| G7 OT rate | 0.0351 | 0.0557 | 0.0005 | round 10 s4.1 |
| half share H1 / H2 | 0.4694 / 0.5249 | 0.4671 / 0.5200 | 0.0001 | round 10 s4.1 |
| D1 eFG home-away gap | 0.0292 | 0.0293 | 0.0002 | round 10 s4.1 |

Reproduction targets for the new arm (not gates, from block V; 10-27 PM ruling 19): 3PA 0.192, rim 0.237, jump2 0.322, TOV 0.149; FT trips must not widen (ratio 0.946, latent 0); the band latent is recorded only and not served.

Gate conflicts the spec must read together (facts, not a recommendation): the margin SD ratio is already above 1 (1.0416, 18 floors) while the total SD ratio is below 1 (0.9454 / 0.9734); G1 possession SD is already 11 floors above the actual; G5 score correlation is 0.058 below actual. The class-mix latent adds between-game variance in points; whether it adds to margin SD, total SD, or correlation depends on whether it is game-shared or team-specific (open question 1).

---

## 6. Compute facts

- **500 x 25 loop screen (local, ~2 min per run).** Sample: `data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet` (500 verified fold-2 games; possession-level real lines use the 492 games whose pbp final equals the verified final). Launcher: `scripts/chain_clk10_env_loop_v1.sh <S seeds> <ctrl|E1> <offset>...` which runs, per tag, `scripts/run_engine_window_v2.py --ids-file <sample> --fold F2 --season 2025 --input-dir data/processed/models/engine_v3 --seeds 25 --seed-offset <o> --workers 12 --games-per-block 25 --seeds-per-block 25 --tag clk10_<lab>_o<o>_s25 --trajectory-seeds <o+S> --results-dir results/clock_env10/loop`, then `scripts/diag_c4_extract_v1.py --tag ...` and `scripts/diag_margin_dispersion_v2.py --prefix results/diag_c4/<tag>_ --seeds 25 --out ...`. Control `ctrl` = no `ENGINE_*` flag; arm = one `ENGINE_*` flag; seeds 0-24 shared; floor draws at offsets 1000, 2000, 3000, 4000; 125-135 s per run (round 10, `docs/tests/clock_env_round10_2026-10-13.md` s2). Grader `scripts/grade_clock_env_loop_v1.py`. A screen is NEVER a candidate call (10-14 process ruling).
- **5,710 x 200 full-size box read (AWS spot).** Instance `c7a.48xlarge` (192 vCPU, 94 workers per run, two runs at a time), launch `scripts/ops_launch_spot_v1.sh spot` (`VOLGB` default 200; AZ order 2c, 2b, 2a, do not pin 2a; `docs/ops/aws_launch_chain.md` s20, :1239-1245); bootstrap `scripts/box_bootstrap_v1.sh install | clone <SHA> | pull | build | check`; run driver pattern `scripts/box_clk10_full_v1.sh parity | runs | post | grade` (control `o0`, arm `o0`, four control draws `o1000..o4000`, each 5,710 x 200; thread vars pinned to 1; `CBB_TRUTH=verified_v1`; run command inside the loop: `scripts/run_engine.py --fold F2 --season 2025 --seeds 200 --seed-offset <o> --workers 94 --input-dir data/processed/models/engine_v3 --tag ... --no-players --trajectory-seeds <o+200> --results-dir ...`). Round 10 timeline: pull 12:39-12:45, build to 12:55, parity 34 s, sims 37 min, post-processing 7 min, grading 10 + 37 min; total ~101 min, about $4.9 (spot $2.70/h; instance-minutes x launch price, not reconciled against the bill). A 100 GB root filled during post-processing (27 GB `data/processed` baked in the image, 29 GB build cache, 11 GB trajectories): use 200 GB or `docker builder prune -af` after `docker build`. Grader `scripts/grade_clock_env_full_v2.py`. The grader grew to 37 min because six 200-seed trajectories load sequentially (~6 min each). A game-shared latent that changes trajectories needs the extra `diag_c4_extract_chunked_v1.py` step already in `post`.
- **Parity v10 (served stack v3, clock K2).** `.venv/Scripts/python.exe scripts/run_parity_smoke_v1.py` (default ref `docs/ops/parity_reference_windows_v10.json`; exit 0 only on a bit-identical digest; drops every `ENGINE_*` from the caller's env; it runs `run_engine.py --fold F2 --season 2025 --seeds 5 --max-games 60 --input-dir data/processed/models/engine_v3` then `scripts/digest_engine_run.py --compare`). On the box: `scripts/box_clk10_full_v1.sh parity` (`dr scripts/run_parity_smoke_v1.py --workers 32 --tag box1014_parity_v10`). Flag-off path of any new latent must be bit-identical to v10 (house pattern: `ENGINE_*` default-off, "no draw, no stream touched").
- New RNG family: add a family name via `StreamBook(..., families=(NAME,))` as `shared_shooting`/`foul_joint` do (they are not in `FAMILIES`, `rng.py:41-56`); no existing stream moves.

---

## 7. Open questions the spec must decide (questions only)

1. Game-shared or team-specific? One class-mix latent per simulated game (both offences share it, like the pace latent and shared shooting `G3`) versus one per team-game (like block V's unit, like shared-shooting control `U1`)? Block V cannot say: the cross-team within-game correlation of the class residual is ABSENT. The choice moves G5 score correlation (0.170 vs 0.228) and margin vs total SD differently.
2. One latent vector for all four classes (TOV, rim, jump2, 3PA) with a fitted 4 x 4 (or 3 x 3 free-logit) covariance, or four independent marginal latents? The excess rim-jump2 residual correlation (-0.448 vs null -0.281) suggests the marginal latents are not independent; the block-V latents are marginal binary-vs-rest, so the joint softmax latent is unfitted.
3. Parametrisation: logit-normal shifts on the (m, 6) probability vector (multinomial-logit), or Dirichlet-multinomial / Dirichlet shares? How is the shift renormalised across the other classes, and does it enter before or after `fj.adjust_trips`?
4. Does the latent's SD depend on state: constant per class, or a function of the offence profile (rim s2 falls 0.070 -> 0.043 across offence rim quintiles; TOV U-shaped; 3PA flat), season part, site or population (`first` vs `cont`)? Per-cell floors are ABSENT; which of these cells are powered for a variance FUNCTION, as opposed to a scalar?
5. Offence and defence: separate latents for offence and defence team-games, or one latent per game-side? The residual correlates beyond floor with the opponent defence profile for 3PA / TOV (|corr| 0.03); is that part of the latent or a model-level miss?
6. How are FT trips held fixed? Options visible in the code: shift only the four non-FT classes and rescale them so FT-trip mass is unchanged; or draw the latent and then let `fj.adjust_trips` set the trips. FT trips are UNDER-dispersed (0.946, latent 0); must the arm leave them exactly alone, or is a negative-latent (tightening) allowed?
7. Interaction with the game-state block: `xx` includes `score_diff`, `in_bonus`, `period`, `seconds_remaining`, which already make the per-chance probability vary within a game. Is the latent a pure additive shift on the logits (state response preserved) or a draw of the static team block (like Stage C `team_block_k`)? Does it need a martingale argument (disclosure 6 of block V) that the null is still the right null after adding it?
8. Mean preservation: a logit-normal shift moves the class mean (Jensen). Is the location fixed analytically (like the clock's `+s2/2`), by method of moments on training rows, or left free and read through the realised G3 levels (3PA share 0.3868 vs 0.3906; rim 0.3723 vs 0.3733 already sit slightly low), as shared shooting `G3` did? What counts as a "regression" on G3 / G4 pooled levels?
9. Fit: how is the latent SD fitted walk-forward? Block V's SDs are test-season (2025) values on artifacts trained through 2024; training-season residuals need out-of-fold probabilities (only an S1 replay on 2024 exists as a registered path). Fold 1 class part was NOT RUN. Is the 0.192 / 0.237 / 0.322 / 0.149 set the reproduction TARGET, or a fit target on training seasons that must then land within a stated floor of them on fold 2?
10. Joint with the pace latent: drawn independent (pace latent today), correlated with a registered rho (`GP` precedent: rho 0.0903 was not eligible), or fitted jointly with the clock variance function? What measurement of the pace x class-mix correlation exists to register (ABSENT today; the nearest is `n_chances` vs class residual, |corr| <= 0.055)?
11. VD1 definition: is "VD1" the pace latent sigma refit, a sigma that is a function of tempo (round 5d `pace2` was NOT adopted), or the chance-count dispersion (K2 end-class mix and OREB chains)? Which component owns the +0.155 G1 SD excess? The class-mix latent itself changes the end-class mix and therefore possession count through K2, so the two arms are not separable by construction; is the decision rule on G1 SD one-sided (toward 5.474) or a veto against moving away?
12. Band latent: not served (no location stage, ruling 19). Is it drawn anyway (recorded, unused), or omitted? The class latent and band latent are uncorrelated (|r| <= 0.06 disattenuated) except the rim-short2 coupling (-0.33 disattenuated), so omitting it removes nothing the class latent carries.
13. Gate form: G5 margin SD ratio is 1.0416 and G5 score correlation is 0.058 low before any new variance; what is the registered direction for each of margin SD ratio, total SD ratio, score correlation, G1 SD and the five round-10 hard vetoes, and does any of them count as co-primary? Which of the three total-SD definitions is used (G5 total 0.9454 vs K2-read 0.9734 vs regulation total 0.9936)?
14. Screening and decision: the 10-14 rule says the 500 x 25 loop is a screen only and the full-size 5,710 x 200 read (~$5, 200 GB disk, four control draws) is the only candidate call. Is the offline gate "reproduce block V on fold 2 within floor" (and on fold 1 once an F1 artifact exists), and what is the minimum seed count study?
15. Which grader scores the arm blind (one grading script for all arms, house rule)? The block-V grader (`scripts/grade_shot_location_v2.py --block V`) is the existing scoring path for the class dispersion; it scores the model probabilities, not the engine's post-`adjust_trips` sim output. Is a sim-output class-share dispersion line (per team-game 3PA / rim / jump2 / TOV share SD in the 200-seed trajectories vs actual) added as a primary? It does not exist today (G3 by-team is NEEDS-INSTRUMENTATION).
