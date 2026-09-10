# control_engine experiments (append-only)

## Pre-registration (2026-09-10, before any code)

Grid:

| Dimension | Values |
|---|---|
| Anchor feature set | A_own (own ridge ratings, as-of daily) ; B_kp (centred KenPom as-of snapshot) ; C_both |
| Rate dispersion | Poisson ; NegBin (chosen per target by training-residual overdispersion test, recorded) |
| Folds | F1 train 2022-2023 test 2024 ; F2 train 2022-2024 test 2025 (selection) |
| Seeds | 200 per game; noise floor = same spec, seed offset +1000 |

Metrics: margin MAE/bias, total MAE/bias, Brier, calibration slope, G1/G5/G6/G9, G10 vs ESPN BET close (2024, 2025).

Decision rules, committed now:
- Anchor set with the lowest F2 margin MAE wins for the Control; if A_own and B_kp are within the seed noise floor, C_both is used only if it beats both by more than the floor, else A_own (compliant, self-contained).
- A rate target uses NegBin only if the Poisson deviance/df on training residuals exceeds 1.2.
- Any G9 bias outside tolerance on F2 is reported as FAIL and traced to a component; nothing is adjusted.

Results: (appended when run)
