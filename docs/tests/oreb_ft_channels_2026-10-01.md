# OREB% and FT% channels of the PPP deficit on served stack v2, plus two responsiveness lines (lane I, 2026-10-01)

Lane I, day session 2026-10-01 (06:14-10:50 EDT). Nothing adopted, no served default changed, no engine code edited.
Pre-registration: `docs/models/rebound/experiments.md` section 12 and `docs/models/free_throw/experiments.md` section 13
(commit 4716a08, before any arm ran). Verified truth everywhere. Served stack v2 = plain default; a local plain-default run
passes parity v9 (bit-identical digest), and the local reference tap equals the box adopted run
(`v3full_COMB9GCTKD_s200_o0`) on 12,500 of 12,500 (game, seed) rows, 0 mismatching cells.

## 0. Answer

1. **(a) OREB%: `TO` is registered (team_rate_estimator 7a.2), fold 2 had run, fold 1 had not. Fold 1 CONFIRMS** (TO - R
   -6.9 floors, tie with T; level -0.06 pp, team slope 1.02 vs R 0.64). **Full size, served v2 with only rebound = TO
   (`RBTO`): OREB% 0.2887 -> 0.2977 (+45 floors; target 0.2984), team OREB slope 0.65 -> 1.05 offence, 0.61 -> 1.14
   defence, G9 total bias -0.34 -> +0.27. One registered veto fires: G4 TOV% 0.1751 -> 0.1757 (6 floors away).** Not put
   forward by the letter; reading for the PM: a Decision 11 case (more second chances expose possession_outcome's TOV level).
2. K2_Ocell team responsiveness at full size: OREB slope cost -0.031 (5.6 floors offence, 5.4 defence), far smaller than
   the 500 x 25 veto read; the block flag itself is level-right and compressed (defence slope 0.61, offence 0.46).
3. **(b) FT%: not mainly a prior problem.** No train/serve skew. Channels (FT% pp of -0.90): unknown shooters on anonymous
   slots about -0.27 (a NOVEMBER candidate-roster gap: 73% of those shooters have a prior season), named who-shoots -0.26
   (of which late-game leading-team trips about -0.22: the sim sends worse shooters; usage is state-blind), state fed to the
   model about +0.38 pp of p in the swap (score_diff the largest), model level -0.12.
4. **FT arms:** offline winner **N1** (height / position / D-I years; -9.9 / -8.8 floors); full size FT% +26 floors, no gate
   veto, but team FT% slope 0.88 -> 0.80 (-8 floors, unregistered). Input arm **A1** (anonymous-slot prior from the team
   roster): FT% +22 floors, no veto (h/a corr -1.96 floors, at the threshold), team FT% slope 0.88 -> 0.93: **put forward**.
   Each is worth about +0.06 points per game.
5. **(c) Team FT slope 0.617 vs 0.958: owner = the grading sample** (legacy -> verified 500-game sample): same inputs 0.92 ->
   0.50 (v2) and 0.94 -> 0.55 (v3); same sample, inputs move it +0.02 to +0.04. Underpowered at 500 games (bootstrap SD
   0.37-3.3). The real full-size number is 0.53 (old) / 0.60 (served v2): a genuine team FT-RATE compression.

---

## 1. (a) OREB% level, -0.75 points: rebound `TO`

### 1.1 Status of `TO` when the day began

`TO` = Stage B arm (E3 v4 rebound features + season anchor `O`), registered in `team_rate_estimator/experiments.md`
section 7a.2 under the 7.1 rule (fold 2 selects, fold 1 confirms). Fold 2 offline had run on the box on 09-30; fold 1
and every closed loop had not (spot reclaim). Section 12 of the rebound file executes fold 1 and adds the closed loop
inside served stack v2 with ONLY the rebound sub-model replaced (arm `RBTO`).

### 1.2 Offline, both folds (primary: three-class log loss, S1_weekly)

| fold | R | R2 (seed 1) | T | TO | TO - R (floors) | TO - T (floors) | floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2 (box, 09-30) | 0.644522 | 0.644642 | 0.643946 | 0.643672 | -0.000850 (-7.1) | -0.000274 (-2.3) | 1.2e-4 |
| F1 (today, local) | 0.619556 | 0.619601 | 0.618813 | 0.618732 | -0.000824 (-6.9) | -0.000081 (-0.7) | 1.2e-4 |

Floor = max(registered 6.7e-5, |R2 - R| on the fold, the F2 R2 spread 1.2e-4) = 1.2e-4 on both folds. (The 09-30 table
quotes TO - R as -12.7 floors against the smaller 6.7e-5 floor; the same difference is 7.1 floors against 1.2e-4.)
**Fold 1 confirms** under the 7.1 / 7a.2 rule: same sign as fold 2 against both R and T, and TO loses to neither.
Against T alone fold 1 is a tie (-0.7 floors).

Lines (cell JSONs):

| fold | arm | held-out level (pp) | worst-decile calib (pp, gate 2.0) | team-quintile slope ratio | gap by team quintile (pp) |
|---|---|---:|---:|---:|---|
| F2 | R | -0.94 | 1.67 | 0.684 | +0.31 / -0.65 / -0.93 / -1.45 / -1.77 |
| F2 | T | -0.90 | 1.78 | 1.045 | -0.97 / -1.13 / -0.84 / -0.74 / -0.68 |
| F2 | TO | -0.04 | 1.82 | 1.055 | -0.14 / -0.25 / +0.03 / +0.09 / +0.22 |
| F1 | R | -0.49 | 1.34 | 0.635 | +0.84 / +0.08 / -0.76 / -0.78 / -1.56 |
| F1 | T | -0.59 | 1.44 | 1.017 | -0.65 / -0.39 / -0.86 / -0.30 / -0.54 |
| F1 | TO | -0.06 | 1.79 | 1.023 | -0.17 / +0.15 / -0.33 / +0.27 / -0.01 |

The E3 features (T) fix the team slope (0.63-0.68 -> 1.02-1.05); the anchor (O) fixes the level (-0.5 to -0.9 pp ->
-0.05 pp). Both folds, every gate passes.

### 1.3 Closed loop `RBTO` (served v2 with rebound = TO only)

Train/serve parity PASS: the served `off_oreb_c` / `opp_def_dreb_c` of `engine_v3_I_RBTO` equal the trained E3 values on
11,185 of 11,185 (game, offence) rows (max diff 0); no other input array changed; anchor offsets cover all 5,710 games.

Local 500 x 25 (verified stride sample; DIRECTION ONLY, underpowered per Decision 12), RBTO minus reference:

| line | reference | RBTO - ref |
|---|---:|---:|
| G4 OREB% | 0.2876 | +0.0096 (FAIL -> PASS) |
| G9 total bias | +0.42 | +0.76 (PASS -> FAIL on this sample) |
| G9 margin bias | +0.37 | -0.02 |
| G9 calibration slope | 0.902 | +0.018 |
| G5 total / margin SD ratio | 0.899 / 0.981 | +0.010 / +0.017 |
| G5 home/away corr | 0.118 | +0.013 |
| G1 mean / SD | 68.92 / 5.42 | -0.03 / +0.09 |
| G4 eFG / TOV / FT rate | | +0.0002 / +0.0006 / +0.0005 |

On this sample the reference already over-predicts totals (+0.42), so the +0.76 shows as a G9 FAIL; at full size the
served stack's total bias is -0.34, and the PPP decomposition prices the OREB channel at -0.75 points.

**Full size (box `d1001_I_1`, 5,710 x 200, ref `v3full_COMB9GCTKD_s200_o0`, floors = max |draw - ref| over lane D's four
served-v2 draws; `scripts/diag_laneI_full_read_v1.py`, Decision 12 tool table `results/d1001I/boot_vs_S2.md`):**

| line | ref | RBTO | floors toward target |
|---|---:|---:|---:|
| G4 OREB% (primary) | 0.2887 | 0.2977 | **+45** |
| team OREB slope, offence / defence | 0.651 / 0.615 | 1.054 / 1.141 | +112 / +47 (no fall) |
| G9 total bias | -0.337 | +0.270 | +3.0 |
| G9 margin bias / slope | -0.261 / 0.948 | -0.251 / 0.945 | +0.7 / -1.0 |
| G5 total / margin SD ratio | 0.925 / 1.042 | 0.924 / 1.040 | -1.3 / +1.0 |
| G5 home/away corr | 0.1261 | 0.1277 | +3.0 |
| G1 mean / SD | 68.82 / 5.50 | 68.76 / 5.50 | +6.8 / +0.9 |
| G4 TOV% | 0.1751 | 0.1757 | **-6 (VETO)** |
| G4 eFG / FT rate | 0.5080 / 0.3271 | 0.5081 / 0.3278 | +1 / +7 |
| FT% | 0.7123 | 0.7122 | -1.4 |

Multi-level (`scripts/diag_laneI_rbto_levels_v1.py`, 5,700 graded games): OREB% gap to actual by month, ref -> RBTO: Nov
-1.20 -> -0.11, Dec -0.95 -> +0.12, Jan -1.20 -> -0.20, Feb -0.21 -> +0.51, Mar -0.44 -> +0.01 pp (Apr 17 games
UNDERPOWERED); home/away -0.78 -> +0.10, neutral -1.04 -> -0.12. Per game: +0.89 pp mean, 73% of games up, per-game MAE vs
actual 5.03 -> 4.90 pp. Per team: see the slopes (every quintile within 0.3 pp offline).

**Verdict (registered 12.3): primary +45 floors, fold 1 confirms, responsiveness up, but the G4 TOV% veto fires (6 floors
away; +0.0006 absolute). NOT PUT FORWARD by the letter.** The TOV count per team-game rises +0.029: each extra offensive
rebound is an extra chance with possession_outcome's own TOV rate, whose level is already above target (the decomposition's
TOV channel). This is the Decision 11 pattern (a fix exposing a compensation); the PM rules.

### 1.4 K2_Ocell team responsiveness (descriptive; open line from HANDOFF judgment call 2)

Team OREB% by 2024-prior quintile, FULL SIZE (5,710 x 200, all games; floors = the four `v3full_S0f*` draws):

| run | pooled OREB% | offence slope | defence slope | offence gap by quintile (pp) |
|---|---:|---:|---:|---|
| S0 (old served) | 0.2829 | 0.684 | 0.647 | -0.45 / -0.98 / -1.41 / -2.15 / -2.40 |
| K2O (K2_Ocell alone) | 0.2905 | 0.653 | 0.619 | +0.40 / -0.18 / -0.62 / -1.44 / -1.73 |
| served v2 | 0.2887 | 0.651 | 0.615 | +0.25 / -0.37 / -0.82 / -1.62 / -1.90 |
| floor | 0.0001 | 0.0055 | 0.0052 | |

K2_Ocell lowers both team slopes by about 0.031 (5.6 / 5.4 floors) at full size: a real but much smaller loss than the
500 x 25 read (-33 "floors" against a 0.0019 single-draw floor). The underlying compression (slope 0.68) is the rebound
model's own (offline R slope 0.684 on F2), which `TO` repairs.

Block rates (served-v2 tap, all games x 2 seeds; sim = sum of K2_Ocell P(blocked | miss) / missed FGA; actual = box blocks
/ opponent missed FGA): pooled 0.1019 vs 0.1029. Defence (blocking) by 2024-prior quintile: sim 0.090 / 0.098 / 0.099 /
0.105 / 0.115 vs actual 0.083 / 0.095 / 0.099 / 0.111 / 0.124, slope 0.61, team corr 0.88. Offence (being blocked):
sim 0.098 / 0.100 / 0.103 / 0.104 / 0.105 vs 0.096 / 0.099 / 0.104 / 0.106 / 0.111, slope 0.46, corr 0.74. The drawn
block flag is level-right and compressed across teams, mostly on the offence side; K2_Ocell's team-slope cost on OREB% is
consistent with a near-uniform block uplift. Not an arm; no decision.

---

## 2. (b) FT%, -0.34 points

### 2.1 Decomposition on served v2 (full size, FT% 0.7123 vs 0.7213)

| channel (total -0.90 pp; rows overlap, not additive) | FT% (pp) | how measured |
|---|---:|---|
| unknown shooters: anonymous roster slots get an all-zero FT block | about -0.27 | 5.5% of sim FTA at 0.648 vs real off-roster 5.2% at 0.695 |
| who goes to the line among named players (usage FT-trip allocation) | -0.26 | served model, states held fixed, sim vs real FTA weights per slot |
| of which: late-game leading-team trips | about -0.22 | last 2:00, leader shooting: fixed-state p 0.707 sim vs 0.730 real, 9.5% of attempts |
| state fed to the FT model (score_diff, bonus, clock) | swap restores +0.38 pp of mean p | served-v2 tap swap; score_diff alone +0.20, bonus -0.04 |
| FT model level (offline) | -0.12 | served model on real 2025 rows 0.7200 vs 0.7212 |

Supporting facts:
- NO train/serve skew in the FT shooter block: for 202,415 real attempts by on-roster shooters, served slot values equal the
  design values (has_prior exact on 100%, as-of FT% on 99.96%; p difference -0.0004 pp).
- Real off-roster shooters (4.4% of attempts) are a NOVEMBER phenomenon: 19.5% of November attempts, under 1% in every
  later month. 73% of them HAVE a prior season and 89% have 0 as-of attempts: they are players the as-of candidate roster
  has not seen yet this season, not newcomers. The served model on their real features gives 0.680 (real 0.695); the
  all-zero block the engine serves gives 0.652.
- The served-v2 tap swap (full season, 2 seeds, 377,463 sim attempts, 98% matched to real attempts of the same game and
  team): sim mean p 0.7118 vs real 0.7191; shooter block swap +0.38 pp (prior-season FT% alone +0.48, has-prior alone
  +0.13), state swap +0.38 pp (score_diff +0.20, in_bonus -0.04); both +0.67 pp. Sim has_prior at the line 0.686 vs
  0.736; in-bonus share 0.552 vs 0.447; shooting team's score diff +0.28 vs +0.75.
- Late game (PM request, lane L's 0.64 vs 0.77): on the served-v2 tap, attempts by the LEADING team in the last 2:00 are
  9.5% of sim attempts (real 9.5%); sim mean p 0.741 vs real realised 0.758 (model on real rows 0.761). With the state
  held fixed, sim leaders' shooters score 0.707 vs real 0.730 (-2.3 pp): reality sends its better FT shooters to the line
  (shooter as-of FT% +0.028 centred vs -0.003 in the sim). The engine's usage draw ignores the game state
  (`UsageAdapter.probs` accepts and ignores `score_diff` / `sec_remaining`), so intentional-foul trips go to the usual
  FT-trip drawers. Owner: usage FT-trip attribution in late-game fouling states. Last 1:00: 0.727 vs 0.752 (-2.5 pp).
  The score_diff state adds about +3.4 pp in the sim and +3.2 pp in reality for leaders, so it is not the late-leader
  owner. (My window and lane L's differ; I did not reconcile 0.64.)

### 2.2 Model arms for the newcomer prior (section 13.2; S1_monthly paired calendar, stated deviation)

Primary F2 log loss; floor max(|N0 s1 - N0| = 4e-6, registered 0.000147) = 0.000147. `N0` reproduces the registered
S1_monthly numbers exactly (F2 0.575237, F1 0.578044).

| arm | F2 log loss | F2 floors | F1 log loss | F1 floors | worst calib F2 / F1 (pp) | newcomer gap F2 / F1 (pp) | at 0 as-of att. F2 | Nov F2 | verdict |
|---|---:|---:|---:|---:|---|---|---:|---:|---|
| N0 | 0.575237 | | 0.578044 | | 0.84 / 1.80 | -0.31 / +0.01 | -1.34 | -1.32 | reference |
| N2 (+3pt block) | 0.573694 | -10.5 | 0.576584 | -9.9 | 1.30 / **2.16** | +0.26 / **+0.67** | -1.87 | -1.60 | fails F1 calib gate and newcomer guard |
| N1 (+height, position, D-I years) | 0.573779 | -9.9 | 0.576747 | -8.8 | 0.57 / 1.68 | -0.32 / +0.12 | -1.09 | -1.32 | **WINS** |
| N3 (both) | 0.573155 | -14.2 | 0.576055 | -13.5 | 1.33 / 1.89 | +0.20 / **+0.63** | -1.29 | -1.55 | fails the newcomer guard on F1 |

Decision by the registered rule: **N1 wins** (simplest passing arm; N3 is better on log loss but its F1 newcomer gap grows
by 0.62 pp > 0.25). Responsiveness: Decision 8 slope 0.96-0.99 every arm; prior-season-FT quintile slope 0.99-1.02.
N1 improves log loss on every segment but barely moves the newcomer LEVEL (F2 newcomer gap -0.32 vs -0.31 pp).

Serving: N1 on the served S1_conf_aligned calendar: F2 log loss 0.573735 vs served 0.575210. Train/serve parity PASS
(height / D-I years exact; position 99.5-100%). Served with no code change (tagged inputs `engine_v3_I_N1` with two extra
slot columns + `adapters.FT_S1_MANIFEST` override; off path = plain default).

Local 500 x 25 (direction only): FT% +0.06 pp; every other line within noise (G9 margin bias +0.10, G9 slope +0.016).

**Full size (box `d1001_I_1`):** FT% 0.71230 -> 0.71402 (+26 floors toward 0.7213); G9 total bias -0.337 -> -0.286 (+2.3);
G5 h/a corr -1.6, G5 margin ratio -0.6, every other line inside 1 floor; no gate verdict flips. By month FT% (actual / ref /
N1): Nov 0.7108 / 0.6873 / 0.6937, Dec 0.7216 / 0.7144 / 0.7149, Jan 0.7220 / 0.7189 / 0.7186, Feb 0.7275 / 0.7193 /
0.7208, Mar 0.7265 / 0.7238 / 0.7240. **Unregistered responsiveness line, reported against N1: team FT% by 2024-prior
quintile, slope 0.883 -> 0.801 (floor 0.010 from the four served-v2 draws, -8 floors).** Offline winner by the rule; in the
sim it trades team responsiveness for level. Not recommended for serving until that is owned.

### 2.3 Input arm A1 (anonymous-slot FT prior, section 13.3)

Build: 26,566 of 28,470 anonymous slots filled; only those four slot columns changed (asserted). Offline (2025 only; fold
1 has no engine inputs): real off-roster attempts covered 97%; served-model mean p 0.652 (all-zero block) -> 0.680 (A1),
real FT% 0.695; log loss 0.6160 -> 0.6062. Local 500 x 25 (direction only): FT% +0.15 pp, nothing else moves.

**Full size (box `d1001_I_2`):** FT% 0.71230 -> 0.71376 (+22 floors toward); G9 total bias -0.337 -> -0.289 (+2.1); G5 total
ratio +0.9; G5 home/away corr 0.1261 -> 0.1251 (-1.96 floors with the registered max-draw floor 0.00052; the Decision 12
bootstrap tool reads -2.03 with its 0.0005 floor: AT the veto threshold); everything else inside 1 floor. November FT%
0.6873 -> 0.6939 (actual 0.7108); later months move under 0.02 pp. Team FT% slope 0.883 -> 0.927 (+4 floors toward 1).
**Verdict (13.3): primary beyond 2 floors, no veto by the registered floor: PUT FORWARD to the PM, h/a corr flagged.**
Fold 1 cannot be read (no fold-1 engine inputs). The deeper owner is the early-season candidate roster (anonymous slots
stand for real players the as-of rotation prior has not seen yet); A1 only gives those slots the roster's FT history.

### 2.4 Interactions

- Lane B is replacing the FT model's `score_diff` (a same-game-form proxy) with a form latent (round 16). N1 keeps the
  served `score_diff`; if B's change lands, N1 must be refit spec-identically on B's feature set. No arm here uses
  `score_diff`.
- The largest remaining FT channels are not FT-model channels: the anonymous-slot block (A1 / the early-season candidate
  roster) and the usage FT-trip attribution, especially late-game intentional fouls.

---

## 3. (c) Team FT slope 0.617 on v3 vs 0.958 on v2: owner = the GRADING SAMPLE

The line is `grade_foul_joint_closed_loop_v2.team_stats`: team FTA/FGA (FT RATE, not FT%; the shooter mix cannot move
it) by 2024-prior quintile, sim span / actual span, on the graded games. Evaluated on finished runs (no new sims):

| run (stack, inputs) | legacy 500 | verified 500 | all 5,710 |
|---|---:|---:|---:|
| 09-18 served stack, v2 inputs (full size) | 0.919 | 0.503 | 0.520 |
| 09-30 S0, v3 inputs (full size) | 0.941 | 0.547 | 0.526 |
| served v2, v3 inputs (full size) | 0.983 | 0.646 | 0.600 |
| reference 500 x 25 as recorded | 0.958 (v2) | 0.617 (v3) | |

- Same inputs, different sample: 0.919 -> 0.503 (v2), 0.941 -> 0.547 (v3). Same sample, different inputs: +0.02 / +0.04,
  and +0.006 on all games. The 0.958 -> 0.617 move is the switch from the legacy to the verified 500-game sample (10
  games in common), not the inputs.
- Why: with about 2.7 games per team the actual quintile means are noise. Legacy actual by quintile 0.325 / 0.342 / 0.322 /
  0.332 / 0.360 (not monotone), verified 0.299 / 0.330 / 0.345 / 0.348 / 0.335; full season 0.306 / 0.324 / 0.328 / 0.343 /
  0.348. Paired game-bootstrap SD of the slope on one 500-game sample: 0.37-3.3. The line is UNDERPOWERED at 500 games.
- The honest number is the full-size one: 0.526 (old served) and 0.600 (served v2): the sim's team FT-rate span is about
  half to 0.6 of reality (full-season sim 0.315 / 0.323 / 0.327 / 0.331 / 0.340 vs actual 0.306 / 0.324 / 0.328 / 0.343 /
  0.348). That is a real responsiveness defect of the foul / FT-rate model's team inputs (R9ao3 improved it +0.07), not
  owned by any line here.

---

## 4. Not run, incidents, files

**NOT RUN:** a joint N1 + A1 read; a usage arm for late-game FT-trip attribution (owner found, nothing registered);
fold-1 closed loops (no fold-1 engine inputs); a T-vs-TO closed loop (only TO was read in the sim).

**Incidents:** none caused by this lane to other lanes' processes; only my own PIDs were started and none were killed.
The first F1 `TO` launch died on `--team-rate-missing raise` (the E3 table has no F1 key for the 392,538 2025 rows,
which F1 never uses); relaunched with `keep_served`, recorded. The sample closed-loop runner pins the pre-adoption clock
(`run_po4b_closed_loop.PINNED_SUBMODELS`), so my local taps go through `scripts/run_laneI_overlay_served_v1.py`, which
corrects that one pin in-process (reference tap = box adopted run on 12,500 of 12,500 rows); the shared runner is
not edited. The tree had another lane's staged edits in `src/cbb_sim/models/event_stream.py` and
`src/cbb_sim/pbp/possessions.py` during my runs (not engine modules; parity v9 PASS on this tree). A file in my session
scratch directory was overwritten by another lane's script (`append.py`); I stopped using it. The box operator's first
attempt at `d1001_I_1` failed on a root-owned results dir (their disclosure) and the second ran as written. The operator's
note reads FT% target as 0.695 (that is the off-roster shooters' FT%); the target is 0.7213.

**Files.** Scripts (all committed): `diag_ft_slot_skew_v1.py`, `diag_ft_who_shoots_v1.py`, `diag_ft_offroster_v1.py`,
`diag_ft_late_leader_v1.py`, `diag_team_ft_slope_owner_v1.py`, `diag_team_oreb_resp_v1.py`, `diag_oreb_ft_tap_v1.py`,
`diag_oreb_ft_tap_grade_v1.py`, `diag_rbto_parity_v1.py`, `train_free_throw_v3_newcomer.py`,
`grade_free_throw_newcomer_v1.py`, `train_free_throw_v3_newcomer_artifacts.py`, `build_engine_inputs_v3_ftbio_v1.py`,
`build_engine_inputs_v3_A1_v1.py`, `run_laneI_overlay_served_v1.py`, `chain_laneI_taps_v1.sh`,
`diag_laneI_tap_compare_v1.py`, `diag_laneI_full_read_v1.py`, `diag_laneI_rbto_levels_v1.py`, `diag_laneI_ft_levels_v1.py`.
Outputs (gitignored): `results/laneI_1001/*.json`, `results/laneI_1001/ft/`, `results/d1001I/`,
`results/engine_v0/{laneI_*_s25, d1001I_*_s200_o0}`. Artifacts (local, untracked, not committed):
`data/processed/models/rebound/round_stageb_F1_laneI/`, `data/processed/models/free_throw/laneI_N1/` (82 MB),
`data/processed/models/engine_v3_I_{RBTO,N1,A1}/`. Box requests: `docs/ops/box_queue/d1001_I_1.md`, `d1001_I_2.md` (both DONE).
