# Possession-outcome variance round: per-game class-mix latent, fitted jointly with the pace latent (VD1) (DRAFT, PM Fable 5.1, 2026-10-27)

Status: DRAFT. Not promoted, nothing fitted. Promotion = this text copied verbatim into
`docs/models/possession_outcome/experiments.md` as its next section, plus a cross-reference row in
`docs/models/clock/experiments.md` (VD1 is defined there at section 46), in the commit BEFORE the first
fit, on the next Opus day. Freeze is ON; a survivor of the full-size read is the first candidate for a
freeze lift, which is the PM's and the user's call, not the worker's.

Facts this draft relies on: `variance_spec_factpack_2026-10-27.md` (same folder; file:line pointers for
every claim below), `docs/tests/shot_location_round2_2026-10-27.md` block V, PROJECT_STATUS rulings 19
and 4 (10-14).

## 0. The defect, and the principle

The served engine draws each chance's terminal class from `EventAdapter.predict` (`adapters.py:368-411`)
at `loop.py:527-533` with nothing shared across a game's chances. Against the model's own per-chance
probabilities, real team-games over-disperse their class mix: 3PA share ratio 1.253 (+28.8 floors), rim
1.360 (+37.0), jump2 1.473 (+39.8), TOV 1.104 (+14.2); FT trips 0.946 (under-dispersed). Registered
latent SDs (logit-normal, method of moments, F2 2025): 3PA 0.192, rim 0.237, jump2 0.322, TOV 0.149.

The fix is at the source: a per-game latent in the class draw, with its size ESTIMATED from data and
validated against realised dispersion, exactly as the pace latent is (`clock_adapter_v3.py:780-975`,
sigma 0.04710 by method of moments on training seasons, one draw per game). No post-hoc widening of sim
output, no tuned multiplier. Because K2 redraws the first-chance duration conditional on the class, a
class-mix latent moves the possession count mechanically, so G1 possession SD (served 5.629 vs actual
5.474, floor 0.014, already 11 floors too wide) is a co-primary and the pace latent's sigma must be
re-fitted jointly (that is VD1 in this round).

## 1. Step 0: structure diagnostics, BEFORE any arm, on fold-1 TRAINING residuals only

Residual table: per team-game, actual class shares minus the model's per-chance probabilities, where the
probabilities for every training season come from OUT-OF-FOLD predictions (each season predicted by the
S1 scheme refit excluding it, or by the walk-forward within-season fits; the worker states which and
why; the 2025 test season is never used to fit anything in this round). Report, with floors:

1. Cross-team within-game correlation of the class residuals (offence A's 3PA residual vs offence B's).
   Decides game-shared vs team-specific structure (section 2, arm family).
2. Correlation of each class residual with the game's pace residual (actual possessions minus the clock
   model's expected possessions). Decides whether the joint draw carries a pace-class correlation.
3. Latent SD by season, by site, by `first` vs `cont`, by offence quintile (rim s2 fell 0.070 -> 0.043
   across quintiles in block V; this is the heteroscedastic structure the arm may need).
4. The class residual correlation matrix (block V: rim-jump2 -0.448 observed vs -0.281 null).
5. Reconcile the three total SD ratios in circulation (0.9454, 0.9734, 0.9936; fact pack section 5) and
   state which definition the full-size grader uses; that one is the registered G5 line.

Step 0 output is a section of the evidence doc. It does not change the arm list; it fixes the
structural choices within each arm (section 2) before any arm is fitted.

## 2. Arms (class-mix latent x pace-sigma treatment)

Class-mix latent, applied to the log-odds of the four non-FT classes (3PA, rim, jump2, TOV) with the
FT-trip logits held FIXED and the four renormalised around them (the `fj.adjust_trips` precedent), one
draw per offence per simulated game (team-specific; a game-shared component is added only if step 0
item 1 shows a cross-team correlation beyond its floor, and then as a two-level draw with both variances
fitted), own RNG family `classmix`, drawn once per game at a fixed ordinal:

| arm | latent | parameters fitted (method of moments on OOF training residuals, walk-forward) |
|---|---|---|
| `CM0` | none (served control) | none |
| `CM1` | diagonal logit-normal | four SDs |
| `CM2` | full-covariance logit-normal | 4x4 covariance (the shared-shooting `G3` precedent) |
| `CM3` | Dirichlet-multinomial on the five-class vector with FT share fixed | one concentration (the simplest comparator) |

Mean preservation: each arm's expected class share under the latent must equal the model's per-chance
probability; the correction (analytic for the logit-normal second-order term, exact for the Dirichlet)
is part of the model, is fitted on training, and is checked in section 4 as a gate (G3 levels within
floor of the control). A mean shift is a failed arm, not a tuning target.

Pace-sigma treatment, crossed with the class arms:

| treatment | pace latent sigma |
|---|---|
| `P0` | served 0.04710 (unchanged) |
| `VD1` | re-fitted jointly: sigma chosen so that the between-game possession variance implied by the class latent plus the pace latent matches the realised between-game possession variance on OOF training seasons (method of moments on the joint residual), with the pace-class correlation from step 0 item 2 carried when it is beyond its floor |

Heteroscedastic variants (latent SD as a function of the offence's as-of class-rate quintile, item 3)
are entered ONLY for the arm that wins the homoscedastic read, as a single disclosed follow-up within
the same round, never as a new arm family after the fold-2 read.

## 3. Folds and fitting

- Fold 1: latent parameters from OOF residuals on 2022-2023; read on 2024. Reported, not gated.
- Fold 2: parameters from OOF residuals on 2022-2024; read on 2025. SELECTION fold.
- No parameter is fitted on, or chosen by looking at, a test season. The 2025-26 season stays sealed.

## 4. Reads and gates

**Offline (both folds; no engine).** For each arm, simulate the class mix for every test team-game from
the model's per-chance probabilities with the latent drawn (1,000 draws per team-game), and read:
- Primary: per-class over-dispersion ratio (realised residual SD / simulated residual SD) for 3PA, rim,
  jump2, TOV: must sit within its floor of 1.0 (floor = max(seed difference, parametric bootstrap SE)).
- FT-trip ratio unchanged from control within floor (the latent must not widen trips).
- G3 class-share levels within floor of control (mean preservation).
- PIT of each team-game's class shares under the arm (K-S p reported; calibration-shape line).
- The by-quintile, by-site, by-season dispersion lines (responsiveness: the latent must not flatten a
  real quintile gradient; read against the step-0 structure).

**Loop screen (500 x 25, local, ~2 min per run; SCREEN ONLY, never a candidate call).** Arms that pass
offline run paired against `CM0 x P0` on aligned streams: G5 margin and total SD ratios, G1 possession
SD and mean, G3 levels, home/away score correlation, PIT. Any hard veto from the round-10 list fires on
the screen = the arm does not go to the box.

**Full-size box read (5,710 x 200 vs four 200-seed control draws, c7a.48xlarge spot, ~$5, 200 GB disk;
`scripts/box_clk10_full_v1.sh` pattern).** For every screen survivor. CANDIDATE only here.
- Primary 1: G5 total SD ratio (registered definition from step 0 item 5) closer to 1.0 than the control
  beyond the floor (0.0028).
- Primary 2: G5 margin SD ratio closer to 1.0 than the control beyond the floor (0.0023).
- Co-primary (VD1): G1 possession SD not farther from 5.474 than the control beyond the floor (0.014);
  an arm that improves the G5 lines by widening G1 further is REFUTED.
- Gates: G1 possession mean within floor of control; G3 3PA and rim shares within floor of control; PIT
  K-S p not worse than control; home/away score correlation not farther from 0.228 than control; every
  other round-10 veto line (pace SD ratio by spread quintile, G9 margin slope, first-half share, D1
  eFG home-away gap) not worse than control beyond its floor; every responsiveness line read as
  distance-to-actual with the actual's sampling error in quadrature (10-14 ruling 2).
- Decision: the arm with the best primary-1 reading among those passing everything is CANDIDATE; ties
  go to the simpler arm (CM1 over CM2 over CM3 is NOT the simplicity order; CM3 one parameter, CM1 four,
  CM2 ten: CM3 < CM1 < CM2). Nothing stronger than CANDIDATE is stated by the worker; the freeze-lift
  decision is the PM's with the user.

## 5. What is deliberately out

- The band latent (block V, separate from the class latent) is not drawn: there is no location stage.
- Shared shooting (`G3`), team-rate draw and the whistle shift stay at their served defaults; this round
  changes the class draw and the pace sigma only.
- No change to the possession-outcome model's mean structure, features or training scheme.
- No hand constants anywhere: every number in the served arm is a fitted parameter with its fitting
  evidence in the doc.

## 6. Scripts, artifacts, evidence

`scripts/diag_classmix_structure_v1.py` (step 0), `scripts/fit_classmix_latent_v1.py` (arms, OOF
residuals, VD1 joint fit), `scripts/grade_classmix_offline_v1.py` (blind offline grader), engine
integration behind a default-off flag `ENGINE_CLASSMIX_LATENT={CM1,CM2,CM3}` and `ENGINE_PACE_SIGMA=
{served,vd1}` at the `loop.py:527-533` draw site with its own RNG family; the loop screen and box
scripts cloned from the round-10 chain with the grader extended by the class-share dispersion lines.
Artifacts under `data/processed/models/possession_outcome/classmix_round1/` (gitignored, HF
`model_artifacts`). Evidence `docs/tests/classmix_round1_<date>.md`; results appended to
`possession_outcome/experiments.md`; `clock/experiments.md` gets the VD1 cross-reference; ledger rows in
the same commit; suite; parity v10 bit-identical with the flags off.

## 7. Compute and budget

Offline and loop screen local (same day). Full-size read on the box only for survivors, ~$5 per read,
at PM discretion (user approval 2026-10-14). One Opus day for offline + screen; the box read may run
on the same day if a survivor exists before 14:00 ET, otherwise the next session.
