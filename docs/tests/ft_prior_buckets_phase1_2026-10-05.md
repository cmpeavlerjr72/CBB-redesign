# Player-FT prior by shooter sample size, Phase 1 (2026-10-05)

Player-FT worker. Diagnostic only, nothing fitted for serving. 2025-26 not loaded.
Script `scripts/diag_ft_prior_buckets_v1.py`. Output `results/ft_prior_buckets/phase1_v1.{json,csv}` (gitignored).
Model: served FT model X0 (served `FT_FEATURES`, S1_monthly, seed 0). Its predictions reproduce `results/ft_exposure/preds_X0_*` exactly (max abs diff 0.0).
"anon" = the same real attempt with the shooter block zeroed (has_prior 0, fta_asof 0, ft_asof 0, prior_ft 0), which is what an engine anonymous slot feeds the model.

## Gap = mean p - realised (pp), by the shooter's prior-season FTA

| prior-season FTA | F2 d0-14 | F2 d15-45 | F2 d46+ | F1 d0-14 | F1 d15-45 | F1 d46+ | raw prior FT% fed (F2) | realised (F2, all days) |
|---|---|---|---|---|---|---|---|---|
| 0 (n 5.9k / 11k / 37k) | **-2.71** | -0.72 | +0.18 | **-1.21** | +0.89 | -0.04 | league (0.70-0.72) | 0.70 |
| 1-10 (n 1.5k / 3.0k / 8.9k) | **-2.55** | **-2.18** | -0.10 | **-1.84** | **-1.56** | -0.60 | **0.62-0.64** | 0.71 |
| 11-40 (n 4.1k / 8.1k / 26k) | **-1.73** | **-1.60** | -0.34 | -0.73 | -0.96 | -0.71 | 0.67-0.68 | 0.72 |
| 41+ (n 12k / 23k / 72k) | -1.14 | +0.46 | +0.43 | -0.13 | +0.11 | -0.01 | 0.72 | 0.73 |

Binomial SE per cell: 0.6-1.2 pp (d0-14, thin rows; the 1-10 d0-14 cells are near the 1,000-attempt underpowered line), 0.16-0.50 pp elsewhere.
Bucketed by known FTA (prior season + as-of), the shooters with no history at all (known 0) are at -2.3 (F2) / -1.4 (F1) in d0-14. Later in the season they are +4.5 to +5.0, but those cells are UNDERPOWERED (n 541-990): late first-time shooters are walk-ons (realised 0.58-0.60).

**Anonymous block.** Scored on every real attempt, the all-zero block gives mean p 0.671 (F2) / 0.655 (F1), against realised 0.721 / 0.719, and 0.645 in d0-14.
The block is the "certain newcomer with no history" state. Real off-roster shooters are 73% returners (section 13.1). The block mislabels them, and the model correctly prices a no-history newcomer low.

## Reading: two owners

1. **FT model (free_throw, owner of this round): the prior-season rate is fed with no sample size.**
   - `FT_FEATURES` has `prior_season_ft` (the raw rate, centred) but not `prior_season_fta`. A 3-for-8 prior season (0.375) is read as a bad shooter, so it cannot be shrunk.
   - Thin prior samples (1-10 and 11-40 FTA) are under-predicted by 1.6-2.7 pp (F2) and 0.7-1.8 pp (F1), all the way through d15-45 until as-of attempts take over.
   - No-history shooters (0) are under-predicted early only. The pooled no-history prior mean mixes November regulars with late-season walk-ons, which is the X1 calendar term's object.
   - This is a mis-sized shrinkage defect for thin prior samples, plus a pooled prior mean for no-history shooters. The 41+ rows are fine apart from F2 d0-14.
2. **Anonymous-slot block (player layer: A3 day-1 prior, selected for 2026-27; FT section 13.3 `A1`).**
   - The block feeds the no-history state (p about 0.645 early) to slots whose real occupants shoot 0.695.
   - An input-prior defect, not an FT-model defect. It is not reopened here.

Phase 2 (`free_throw/experiments.md` section 20) targets owner 1.
