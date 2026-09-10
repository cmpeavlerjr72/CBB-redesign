# Gate reference tables -- 2026-09-10

Source: `scripts/build_gate_reference.py`, computed on D-I, non-truncated games (`data/processed/games_universe.parquet`: `is_d1_game & ~pbp_truncated`). Tables at `data/reference/gate_targets_{season}.parquet` (tidy long format: season, breakdown, group, side, metric, value, n).

Team tiers (terciles of each team's own-season mean point differential) are **not** leak-free -- a team's tier uses its full season including games after the one being described. That is fine for a descriptive reference table but this exact per-season tier must never be joined onto a game as a pregame feature; see `cbb_sim.data.seal` and the leak-test rule in FRAMEWORK_PLAN.md.

## Season-level headline numbers

| season | poss/gm (SD) | PPP off | eFG% off | TOV% off | OREB% off | FT rate | 3PA share | rim+layup share | home margin (non-neutral) | total pts (SD) | margin SD | OT rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 68.07 (5.54) | 1.029 | 50.2% | 18.5% | 27.8% | 0.310 | 37.8% | 34.7% | 4.98 (14.21) | 139.80 (19.29) | 14.13 | 5.7% |
| 2023 | 67.90 (5.53) | 1.042 | 50.6% | 18.3% | 28.2% | 0.323 | 37.4% | 35.5% | 5.38 (13.94) | 140.53 (21.47) | 13.87 | 6.2% |
| 2024 | 68.39 (5.48) | 1.062 | 50.6% | 17.2% | 28.7% | 0.336 | 37.3% | 36.3% | 5.36 (14.45) | 145.13 (19.05) | 14.34 | 6.0% |
| 2025 | 67.88 (5.47) | 1.073 | 51.0% | 17.4% | 29.6% | 0.337 | 39.1% | 35.6% | 5.74 (14.90) | 145.51 (18.95) | 14.63 | 5.6% |
| 2026 | 68.16 (5.49) | 1.097 | 51.6% | 16.9% | 30.3% | 0.358 | 39.5% | 38.6% | 5.72 (15.17) | 149.05 (20.67) | 15.02 | 5.7% |

## Readings

**Possessions/game.** poss_per_game_mean by season -- 2022: 68.068, 2023: 67.896, 2024: 68.391, 2025: 67.875, 2026: 68.164. Pace is basically flat across 2022-2026 (~68-69 poss/gm); no secular tempo shift to design around.

**Points per possession (offense).** ppp_mean by season -- 2022: 1.029, 2023: 1.042, 2024: 1.062, 2025: 1.073, 2026: 1.097. PPP has risen steadily -- shooting efficiency and FT rate both climbed while pace held flat, so scoring/game rose with them.

**FTA/FGA.** ft_rate_mean by season -- 2022: 0.310, 2023: 0.323, 2024: 0.336, 2025: 0.337, 2026: 0.358. This is the clearest secular trend in the data: FTA/FGA rose from the low-0.30s in 2022 toward the mid-0.35s by 2026, consistent with rule/officiating shifts widely discussed in the sport; any possession model must let this drift rather than fitting one static rate.

**3PA share.** three_pa_share_mean by season -- 2022: 0.378, 2023: 0.374, 2024: 0.373, 2025: 0.391, 2026: 0.395. Share of FGA that are threes rose modestly, another reason shot-mix should be fit per-season, not pooled.

**eFG% / TOV% / OREB% (four factors, offense).** eFG% by season -- 2022: 0.502, 2023: 0.506, 2024: 0.506, 2025: 0.510, 2026: 0.516. TOV% by season -- 2022: 0.185, 2023: 0.183, 2024: 0.172, 2025: 0.174, 2026: 0.169. OREB% by season -- 2022: 0.278, 2023: 0.282, 2024: 0.287, 2025: 0.296, 2026: 0.303. Four factors are far more stable year to year than FTA/FGA -- they are the safer anchors for the possession cascade's shot-outcome sub-model.

**Home-court margin.** home_margin_mean_nonneutral by season -- 2022: 4.978, 2023: 5.376, 2024: 5.364, 2025: 5.738, 2026: 5.717. Rising gently from ~5.0 to ~5.7 points at non-neutral sites, D-I-vs-D-I games only. Notably smaller than the hoopR audit's unfiltered figure (~8-9 points, `docs/tests/data_audit_hoopr_2026-09-10.md` section 6) -- most of that gap is buy-game blowouts (a D-I home team hosting a weak non-D-I opponent), which this table deliberately excludes via the D-I filter. Either way, home/away/neutral must be a first-class feature (last year's engine dropped it entirely, CLAUDE.md postmortem).

**Margin SD / total SD.** margin_sd by season -- 2022: 14.133, 2023: 13.870, 2024: 14.339, 2025: 14.633, 2026: 15.017. total SD by season -- 2022: 19.294, 2023: 21.468, 2024: 19.046, 2025: 18.954, 2026: 20.674. Total SD noticeably exceeds margin SD every season, the signature of positively correlated team scores (shared pace) -- an engine with independent team draws (last year's defect) will show margin SD roughly equal to total SD instead.

**OT rate.** ot_rate by season -- 2022: 0.057, 2023: 0.062, 2024: 0.060, 2025: 0.056, 2026: 0.057. Stable at ~5-6% every season; ties must resolve through an OT model, not be discarded (already a standing rule).
