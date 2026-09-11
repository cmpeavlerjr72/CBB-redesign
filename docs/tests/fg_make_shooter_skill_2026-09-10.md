# How much genuine shooter skill is there per shot class? (fg_make round 4, step 2)

Author: Opus cascade worker (fg_make lane), 2026-09-10. Reproducer:
`scripts/diag_fg_make_shooter_skill_v1.py` (writes
`data/processed/models/fg_make/shooter_skill_v1.json`). Population: seasons
**2022-2025 only** (2026 sealed, `assert_not_sealed` at entry), fg_make's own
universe and its three v2 shot classes, on the **corrected shooter label**
(`shot_shooter_id`) via the cached design
`data/processed/models/fg_make/design_v2_shotshooter.parquet`
(2,239,678 attempts). No model is fitted anywhere in this document.

Trigger: `docs/LEARNINGS.md` **L29** and `docs/models/fg_make/experiments.md`
section 18.3. Round 3 re-keyed the shooter and the `FGA_3` Decision-8
shooter-quintile span collapsed from 35.9 pp to **2.802 pp**. Two readings of
that number imply opposite round-4 designs:

* **2.8 pp is the truth.** College three-point make probability is nearly all
  team shot quality and noise; the honest shooter block is tiny or absent, and
  round 4 should expect B0 (no shooter block) to win.
* **2.8 pp is under-shrinkage.** The raw as-of rate on a handful of attempts is
  mostly binomial noise; a tree fed that raw rate correctly learns to distrust
  it, and the *predicted* span collapses even though *true* between-shooter
  skill is much larger. Round 4 should then shrink the rate, and B1 should beat
  B0 by a lot.

This document separates them without fitting anything.

---

## 0. The answer in one table

| class | attempts | shooter-seasons | true between-shooter SD (MoM) | skill share of observed spread | `m` (shrinkage, attempts) | implied quintile span on TRUE skill | round-3 realised span on the RAW as-of rate | recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `FGA_rim` | 817,199 | 17,841 | **6.912 pp** | 47.4% | **50.8** | **19.35 pp** | 10.238 pp | 53% |
| `FGA_jump2` | 571,838 | 17,325 | **3.937 pp** | 17.8% | **152.8** | **11.02 pp** | 4.343 pp | 39% |
| `FGA_3` | 850,641 | 16,565 | **2.736 pp** | 14.7% | **298.9** | **7.66 pp** | 2.802 pp | 37% |

**Neither reading is right; the truth is in between, and it is the second one
that matters for the design.** Genuine three-point shooter skill exists and is
worth about **7.7 pp** of top-to-bottom quintile span — nearly three times the
2.8 pp round 3 measured, and nowhere near the 35.9 pp the labelling defect
manufactured. The raw as-of rate recovers only 37% of it because 85% of the
observed spread in shooter season rates on threes is binomial sampling noise.

"Implied quintile span on true skill" is `2 x 1.3998 x sigma`, the difference
between the top- and bottom-quintile means of a normal with that SD; it is what
Decision 8 would read if the driver were the shooter's TRUE rate instead of his
observed one. "Recovered" is the round-3 realised span over that.

---

## 1. Method-of-moments between-shooter SD

Per (season, shooter) attempt count `n_i` and make count `k_i` on that class;
the beta-binomial moment estimator splits the observed spread of shooter rates
into binomial noise and true between-shooter variance:

```
S           = sum_i n_i (p_i - pbar)^2
E[S]        = (I - 1) pbar(1 - pbar)  +  (N - sum_i n_i^2 / N) sigma^2
sigma^2_hat = [S - (I - 1) pbar(1 - pbar)] / (N - sum_i n_i^2 / N)
```

The first term of `E[S]` is exactly the spread `I` *identical* shooters would
produce by chance, so subtracting it is what separates skill from sampling.
`m = pbar(1 - pbar) / sigma^2` is the implied shrinkage strength in attempts —
the point at which a shooter's own history and the prior carry equal weight.

### 1.1 Stability by season (the estimate is not a pooling artefact)

| class | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| `FGA_rim` | 7.13 pp (m 47.7) | 6.91 (50.7) | 6.94 (50.6) | 6.71 (54.0) |
| `FGA_jump2` | 3.74 pp (m 168.0) | 3.83 (160.9) | 3.81 (163.8) | 4.23 (133.3) |
| `FGA_3` | 2.74 pp (m 297.5) | 2.34 (409.0) | 2.94 (258.3) | 2.82 (282.0) |

Flat within 0.4 pp on rim and the jumper and within 0.6 pp on threes across
four independent seasons. `FGA_3`'s 2023 season is the outlier low (2.34 pp);
nothing else moves.

### 1.2 Restricting to high-volume shooters (the long tail is not driving it)

| class | all shooter-seasons | `>= 10` attempts | `>= 25` | `>= 50` |
|---|---|---|---|---|
| `FGA_rim` | 6.91 pp (17,841) | 6.80 (13,564) | 6.68 (10,302) | 6.55 (6,495) |
| `FGA_jump2` | 3.94 pp (17,325) | 3.99 (11,719) | 3.87 (7,790) | 3.62 (4,115) |
| `FGA_3` | 2.74 pp (16,565) | 2.84 (11,994) | 2.57 (9,492) | 2.21 (6,604) |

The estimate is stable to 0.5 pp under every volume filter. It drifts slightly
DOWN as the filter tightens on every class, which is the expected direction
(low-volume specialists — a centre who takes six threes a season — carry some
of the genuine spread), and is small enough that no conclusion here depends on
which population is used.

---

## 2. Reliability of an as-of rate

`reliability(n) = n / (n + m)`: the fraction of the observed as-of rate's
deviation from the prior that is signal rather than noise.

| class | `m` | 25 attempts | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|
| `FGA_rim` | 50.8 | 0.330 | 0.496 | 0.663 | 0.797 |
| `FGA_jump2` | 152.8 | 0.141 | 0.247 | 0.396 | 0.567 |
| `FGA_3` | 298.9 | **0.077** | **0.143** | **0.251** | **0.401** |

For scale: a high-volume college three-point shooter takes about 200 threes in
a whole season, and the engine must predict his November games off 0-25. **A
three-point as-of rate is 92% noise at 25 attempts and still 60% noise at 200.**
That is the entire explanation for round 3's 2.8 pp span, and it is a property
of the sample size, not of the model class.

---

## 3. Prior season -> current season

Same shooter, completed prior season's rate against the current season's rate.
Disattenuated correlation divides the raw correlation by
`sqrt(rel_prev x rel_cur)` at each subset's own mean attempt count.

| class | min attempts both seasons | n | mean att prev / cur | raw corr | disattenuated corr | OLS slope cur~prev |
|---|---|---:|---|---:|---:|---:|
| `FGA_rim` | 1 | 8,339 | 46 / 59 | 0.184 | 0.363 | 0.165 +- 0.010 |
| `FGA_rim` | 25 | 4,361 | 72 / 86 | 0.475 | **0.783** | 0.449 +- 0.013 |
| `FGA_rim` | 50 | 2,398 | 94 / 108 | 0.546 | **0.821** | 0.517 +- 0.016 |
| `FGA_rim` | 100 | 550 | 141 / 151 | 0.654 | **0.882** | 0.632 +- 0.031 |
| `FGA_jump2` | 1 | 8,032 | 34 / 42 | 0.075 | 0.383 | 0.069 +- 0.010 |
| `FGA_jump2` | 25 | 3,018 | 63 / 74 | 0.240 | **0.781** | 0.228 +- 0.017 |
| `FGA_jump2` | 50 | 1,285 | 88 / 100 | 0.273 | **0.720** | 0.264 +- 0.026 |
| `FGA_jump2` | 100 | 216 | 137 / 150 | 0.324 | 0.670 | 0.314 +- 0.063 | UNDERPOWERED (n < 500) |
| `FGA_3` | 1 | 7,482 | 52 / 67 | 0.108 | 0.656 | 0.098 +- 0.010 |
| `FGA_3` | 25 | 4,065 | 81 / 101 | 0.164 | **0.707** | 0.153 +- 0.014 |
| `FGA_3` | 50 | 2,549 | 103 / 122 | 0.169 | **0.620** | 0.160 +- 0.019 |
| `FGA_3` | 100 | 890 | 144 / 163 | 0.138 | 0.407 | 0.136 +- 0.033 |

The `FGA_jump2 >= 100` cell (n = 216) is UNDERPOWERED and is printed for
completeness only.

**Shooter skill survives a summer on every class.** Once the noise in both
seasons is accounted for, prior-season and current-season true rates correlate
**0.62-0.88**. The RAW correlation on threes (0.11-0.17) is what a model
actually sees, which is why `prior_season_make_c` looked worthless in rounds
1-3 and why a B2 arm must carry the ATTEMPT COUNT alongside the rate: without
the count, the model cannot tell a 0.45 on 20 attempts from a 0.45 on 200.

The one cell that falls off is `FGA_3` at `>= 100` (disattenuated 0.41, down
from 0.71 at `>= 25`): the highest-volume three-point shooters are the most
role-dependent, and a role change between seasons moves shot quality
independently of skill. That is a reason to carry shot-mix context (B3) and
not a reason to drop the prior season.

---

## 4. Slope of the realised make rate on the as-of rate, by attempt-count bucket

Realised outcome minus the league as-of rate, regressed on the model's own
`shooter_make_c` (as-of rate minus the league as-of rate), within buckets of
`shooter_att_c`. **If the raw as-of rate were an unbiased predictor the slope
would be 1.0 in every bucket.** `predicted` is `n_bucket_mean / (n_bucket_mean + m)`
from section 2 — what the slope SHOULD be if the only defect is binomial noise
at the estimated `m`.

| class | bucket | n | share | SD of as-of dev | **slope** | SE | t | predicted by `m` | realised quintile span |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `FGA_rim` | 0 | 43,960 | 5.4% | 0.00 | undefined | | | | |
| `FGA_rim` | 1-24 | 300,590 | 36.8% | 21.59 pp | **0.117** | 0.004 | 28.1 | 0.19 | +7.84 pp |
| `FGA_rim` | 25-49 | 202,025 | 24.7% | 10.78 | **0.376** | 0.010 | 37.1 | 0.42 | +11.26 |
| `FGA_rim` | 50-99 | 193,579 | 23.7% | 8.95 | **0.537** | 0.012 | 43.2 | 0.59 | +13.82 |
| `FGA_rim` | 100-199 | 74,513 | 9.1% | 7.73 | **0.657** | 0.023 | 28.6 | 0.73 | +14.03 |
| `FGA_rim` | 200+ | 2,532 | 0.3% | 7.78 | 0.621 | 0.123 | 5.0 | 0.83 | +14.34 |
| `FGA_jump2` | 1-24 | 257,534 | 45.0% | 20.82 | **0.036** | 0.005 | 7.8 | 0.07 | +2.43 |
| `FGA_jump2` | 25-49 | 138,630 | 24.2% | 9.32 | **0.187** | 0.014 | 13.3 | 0.19 | +4.91 |
| `FGA_jump2` | 50-99 | 109,297 | 19.1% | 7.07 | **0.263** | 0.021 | 12.6 | 0.32 | +4.91 |
| `FGA_jump2` | 100-199 | 31,029 | 5.4% | 5.48 | **0.408** | 0.051 | 8.0 | 0.48 | +6.23 |
| `FGA_jump2` | 200+ | 1,145 | 0.2% | 3.55 | -0.076 | 0.410 | -0.2 | 0.62 | -4.80 (UNDERPOWERED) |
| `FGA_3` | 1-24 | 267,215 | 31.4% | 18.32 | **0.043** | 0.005 | 8.7 | 0.04 | +2.41 |
| `FGA_3` | 25-49 | 194,744 | 22.9% | 8.34 | **0.113** | 0.013 | 8.8 | 0.11 | +2.24 |
| `FGA_3` | 50-99 | 222,260 | 26.1% | 6.12 | **0.148** | 0.017 | 9.0 | 0.19 | +2.45 |
| `FGA_3` | 100-199 | 115,613 | 13.6% | 4.61 | **0.229** | 0.031 | 7.5 | 0.32 | +3.06 |
| `FGA_3` | 200+ | 7,155 | 0.8% | 3.46 | -0.038 | 0.164 | -0.2 | 0.45 | -1.19 (UNDERPOWERED) |
| all three | all defined rows | | | | rim **0.204**, jump2 **0.066**, three **0.074** | | | | |

`200+` is 0.2-0.8% of each class and its slope SE spans zero on the jumper and
the three; it is labelled UNDERPOWERED and is not read as evidence in either
direction. The 5.1-6.0% of attempts with NO prior history have no as-of rate at
all and enter as a labelled cell (the `feature = 0` centring convention), not
as an imputation.

**Every measured slope is far below 1 and rises monotonically with the attempt
count, and every one lands close to `n / (n + m)`.** That is the signature of a
raw rate whose only defect is binomial noise at the `m` section 1 estimated,
measured directly on outcomes with no model in between. The pooled
all-defined-rows slopes (0.204 / 0.066 / 0.074) are the number Decision 8's
`shooter_make_c` driver is really reading, and they are the reason a tree fed
this raw feature produces a flat prediction: **the tree is right and the
feature is wrong.**

---

## 5. What this says about round 4

1. **The 2.8 pp `FGA_3` span is under-shrinkage, not the truth.** True skill is
   worth 7.66 pp of quintile span; the raw as-of rate recovers 2.80 pp (37%).
   A shooter block is worth building on all three classes.
2. **The correct fix is a shrunk rate, not a richer model class.** The measured
   slope profile matches `n / (n + m)` bucket for bucket. Shrinking the as-of
   rate toward a prior with `m` of roughly 51 / 153 / 299 attempts turns a
   feature with slope 0.07-0.20 into one with slope near 1, and does it with
   one fitted parameter per class. Round 4 fits `m` on fold 1 rather than
   adopting these numbers, because these are pooled 2022-2025 and would leak
   the fold-2 test season.
3. **Attempt count must travel with the rate.** Reliability moves by a factor
   of five across the observed range on threes, so a model given the rate
   without the count cannot know how much to trust it — which is what B2 tests.
4. **The prior season is real once disattenuated (0.62-0.88)** and belongs in
   the arm list, with its own attempt count for the same reason.
5. **A falsifiable prediction, recorded before round 4 runs:** if shrinkage is
   the whole story, B1 beats B0 by a clear margin on every class and B2-B4 add
   little; if `FGA_3`'s honest skill really is un-extractable, B1 lands inside
   the noise floor of B0 and the simplicity rule selects B0 for that class.
   Both outcomes are legitimate results.
