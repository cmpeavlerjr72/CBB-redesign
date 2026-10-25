# fg_make PAE round v1: evidence (2026-10-25)

Pre-registration: `experiments.md` section 31 (commit fe34135, before any fit). Full tables: section 32. Freeze ON. Nothing adopted, wired or served.

## Verdicts (registered rule, F2)

| class | P1 (+ team offence / defence PAE) | P2 (P1 + per-attempt cell rate + missing flag) |
|---|---|---|
| rim | CANDIDATE (offline), +1.67e-4 = 2.12 floors | CANDIDATE (offline), +2.20e-2 = 39.4 floors; preferred over P1 |
| jump2 | NOT CANDIDATE (0.99 floors; calibration clause; offence-PAE slope 0.79) | NOT CANDIDATE (P1 gate; +13.6 floors on its own) |
| three | NOT CANDIDATE (0.41 floors) | NOT CANDIDATE (P1 gate; +4.96 floors on its own) |

The leak test passed on both ratings (pooled change-form correlation +0.054 / -0.064, every season |corr| <= 0.079).

## Settings and the evidence that chose them

All three hyperparameters were chosen on fold-1 training data only, before any arm ran.

| parameter | chosen | evidence |
|---|---|---|
| grid g | `da15` (2 ft distance x 12 deg angle) | LOSO log loss 0.651351, against g25 0.651768, g15 0.652729, g10 0.656021 |
| k (offence / defence) | 800 / 1600 located attempts | fitted on season 2023 |
| rho (offence / defence) | 0.5 / 0.5 | fitted on season 2023 |

## Coverage and coordinate frame

- **FGA coordinate coverage:** 80.0 / 84.3 / 87.6 / 98.6% for 2022-2025.
- **Per-team p10:** 55.9 / 65.0 / 78.9 / 96.8%.
- **The coverage gap is per game.** It is lowest in March, at 64-76% before 2025.
- **Frame:** x is full court. A team shoots at one end per half and switches at halftime in 99.98% of team-games. Folding x to the half court puts the basket at x = 52.5, and make rate by distance peaks at the rim (70.6% at 0-1 ft).
- **Design<->pbp join:** 100% matched.

## Caveats for the PM

1. **Missingness is informative and season-unstable.**
   - In 2022-2024, unlocated rim attempts make about 3-4 pp more than located ones.
   - In 2025, unlocated rim attempts make 81% against 58% located.
   - P2's gain on those few rows is partly an outcome proxy. Its verdict rests on located rows (36-38 floors on rim).
2. **Canned pixels.**
   - 9% of located rim attempts sit on the L16 canned pixels and make 69-70%, against 57% elsewhere.
   - This is consistent with real shot quality, but outcome-dependent charting cannot be excluded.
   - Audit coordinate placement before building on P2.
3. **P2 is unservable.**
   - The engine samples no shot location.
   - Serving P2 needs a shot-location stage upstream of fg_make, gated on its own (cascade rule).
4. **Responsiveness.**
   - The served model is flat on the offence-PAE quintile line for jumpers and threes (slope 0.55 / 0.65 on F2).
   - P1 lifts both lines to 0.79 / 0.87 at a gain inside the floor.

## Reproduce

```
.venv/Scripts/python.exe scripts/build_pae_shot_rating_v1.py --part coverage
.venv/Scripts/python.exe scripts/build_pae_shot_rating_v1.py --part build
bash scripts/chain_fg_make_pae_v1.sh        # 8 runs of scripts/train_fg_make_v5_pae_v1.py
.venv/Scripts/python.exe scripts/grade_fg_make_pae_v1.py
```

The design table `pae_v1/design_pae_cols_v1.parquet` (sha256 93bb463bf560...) and the team ratings (4a7f85e220ad...) are on HF under the `model_artifacts` key, along with all runs in `round_pae/`.
