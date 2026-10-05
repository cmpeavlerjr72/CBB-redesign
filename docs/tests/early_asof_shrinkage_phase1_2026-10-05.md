# Early-season as-of team features: bias / shrinkage check, Phase 1 (2026-10-05)

Early-season worker. Diagnostic only: nothing was fitted, served or adopted. 2025-26 was not loaded.
Question: are the thin early as-of team TOV and FTA-rate features (possession_outcome `off_tov_c`, `opp_def_tov_c`,
`off_ftr_c`, `opp_def_ftr_c`) biased or mis-shrunk? If so, does that drive the days 0-14 total bias of about -4 pts?

**Verdict: REFUTED as the owner of the window level bias. Phase 2 was not run.**
- The features are unbiased in every bucket.
- They are badly under-shrunk early, but that is a slope defect. It is already owned and bake-offed:
  - team_rate_estimator: E1/E2 = n/(n+k), with and without a prior-season carry, and E3 selected;
  - possession_outcome round 4: G1/G2/G4.
- Shrunk features do not move the window class level of the downstream model (1c).

Scripts:
- `scripts/diag_early_asof_bias_v1.py` (1a);
- `scripts/diag_early_asof_bias_po_v1.py` (1b);
- `scripts/diag_early_asof_bias_po_v2.py` (1c).

Tables are in `results/early_asof_bias/` (gitignored).

## 1a. Feature vs realised rest-of-season rate (served build: possessions v2 first chances, pbp_complete; per 100)

- Target: the team's realised rest-of-season rate, from this game onward, centred on the league rate over the same dates.
- Bias = mean(feature - target). Slope = target regressed on feature. A well-shrunk feature has slope 1.
- SEs come from a team-block bootstrap.
- The rows with n0 (feature exactly 0) are included.
- E3 is the shrunk reference: box-based, scale differs slightly.

| fold | feature | d0-14 bias (SE) | d0-14 slope | d15-45 bias / slope | d46+ bias / slope | E3 d0-14 slope |
|---|---|---|---|---|---|---|
| F1 | tov off | -0.10 (0.15) | **0.17** | -0.03 / 0.33 | -0.14 / 0.59 | 1.04 |
| F1 | tov def | -0.00 (0.15) | **0.17** | +0.01 / 0.36 | +0.11 / 0.63 | 0.79 |
| F1 | ftr off | +0.45 (0.44) | **0.08** | +0.32 / 0.23 | +0.19 / 0.53 | 0.89 |
| F1 | ftr def | +0.22 (0.42) | **0.16** | +0.50 / 0.34 | -0.37 / 0.60 | 0.98 |
| F2 | tov off | -0.01 (0.13) | **0.16** | -0.03 / 0.31 | -0.17 / 0.59 | 0.89 |
| F2 | tov def | -0.01 (0.14) | **0.18** | -0.04 / 0.39 | +0.00 / 0.67 | 0.86 |
| F2 | ftr off | +0.23 (0.45) | **0.08** | +0.51 / 0.21 | +0.28 / 0.52 | 0.87 |
| F2 | ftr def | +0.19 (0.46) | **0.11** | +0.15 / 0.28 | -0.35 / 0.60 | 0.72 |

- **No level bias.** Every bias is within about 1.2 SE of 0, in every bucket, in both folds. Feature means are about 0 by construction, since the features are centred.
- **Mis-shrinkage is severe early.** The d0-14 slope is 0.08-0.18 (games 1-3: 0.07-0.18). The next-game target gives the same picture.
- League context, first-chance level per 100:
  - FTA/FGA d0-14 vs the season: F1 32.3 vs 31.7, F2 33.7 vs 32.2.
  - TOV per possession d0-14 vs the season: 16.1 vs 15.2 (F1), 15.9 vs 15.4 (F2).
  - The early league level is a calendar fact. The model's `days_since_start` carries it, and the centring does not.

## 1b. Served PO `first` (T0) window level gap: is it in the thin-sample rows?

Gap = mean p - realised, in pp. SEs come from a game-block bootstrap. `both_n0` means both teams have 0 prior games, so every team feature is exactly 0, which is perfectly shrunk.

| fold | class | d0-14 all | d0-14 both_n0 | d0-14 n1-3 | d15-45 n4+ | d46+ |
|---|---|---|---|---|---|---|
| F1 | TOV | +1.35 (0.15) | +1.23 (0.30) | +1.33 (0.18) | +0.44 | +0.37 |
| F2 | TOV | +0.28 (0.15) | +0.44 (0.29) | +0.35 (0.19) | -0.15 | -0.09 |
| F1 | FT trip | -0.48 (0.10) | -0.27 (0.19) | -0.49 (0.12) | -0.17 | -0.15 |
| F2 | FT trip | -0.38 (0.11) | +0.06 (0.20) | -0.62 (0.16) | -0.05 | +0.05 |

- **TOV.** The window gap is the same in rows whose features are exactly the league mean. So it is the level/calendar drift (section 31), not thin-feature noise.
- **FT trips.** The F2 gap sits in the thin-sample rows (n1-3 vs both_n0: -0.68 pp, about 2.6 SE). F1 does not confirm: the same difference is -0.22 pp, inside 1 SE.
- Signed feature quintiles show no FT over-reaction: the gap is flat to negative across quintiles.

## 1c. Do shrunk features move the window level? (paired, existing retrains, F2 only)

The arms are two existing fold-2 retrains on identical rows, scored by a game-block bootstrap:
- FR: the served features (`full_retrain_v1/FR_box_v1`);
- FT: E3 v4 shrunk features (`FT_box_v2`).

Fold 1 has no FT retrain, so it cannot be read.

| bucket | TOV gap FR -> FT (d, SE) | FT-trip gap FR -> FT (d, SE) | multiclass LL d (SE) |
|---|---|---|---|
| d0-14 | +0.30 -> +0.28 (-0.02, 0.05) | -0.47 -> -0.44 (+0.03, 0.04) | **-0.0046 (0.0007)** |
| d0-14 n1-3 | +0.38 -> +0.27 (-0.11, 0.07) | -0.67 -> -0.58 (+0.09, 0.05) | -0.0047 |
| d15-45 | -0.09 -> -0.02 (+0.07, 0.03) | -0.15 -> -0.08 (+0.06, 0.02) | -0.0018 |
| d46+ | -0.14 -> -0.07 (+0.06, 0.01) | +0.06 -> +0.01 (-0.05, 0.01) | +0.00002 |

- Shrinkage buys the known Stage B likelihood gain, concentrated early (d0-14 LL -0.0046, about 6 SE).
- It does NOT move the window class level: TOV -0.02 and FT trips +0.03 pp, both inside 1 SE.
- Even the n1-3 FT-trip deficit closes by only 0.09 of 0.67 pp.

## Reading and recommendation

1. The as-of features are not biased, and their mis-shrinkage is not what makes November totals low.
   - A pre-registered n/(n+k) or prior-season arm would repeat team_rate_estimator E1/E2 and PO round 4 G1/G2.
   - Given 1c, it would be expected to move the window level by about 0, so Phase 2 was not run.
2. E3 shrinkage stays worth serving for likelihood/ordering reasons. That is the existing Stage B/C track (F_T retrain, not adopted 10-05), not the early-total fix.
3. The window level miss lives in three places:
   - (a) the season-level TOV drift (F1);
   - (b) an FT-trip deficit of -0.4 to -0.5 pp in both folds, which is NOT repaired by team shrinkage and is partly concentrated in games 2-4 of a team's season (F2 only);
   - (c) FT make: player-level `shooter_fta_asof` / anonymous slots, `total_bias_decomp` reading 4.
   - The thin-sample object worth a registration is player-level (FT-2 shooter reliability), not team-level.
   - The FT-trip (b) deficit belongs to the foul/PO FT-trip owner (decomp "not owned here" item).
