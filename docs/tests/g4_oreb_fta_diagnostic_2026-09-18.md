# Gate G4: where the OREB% (-1.6 pp) and FTA/FGA (-1.2 pp) misses come from (2026-09-18)

Lane: G4 cause tracing. **DIAGNOSIS AND PRE-REGISTRATION ONLY.** Nothing is
fitted, no served default is changed, no arm is proposed as adopted, and no
number anywhere below is a post-hoc adjustment to sim output. Every channel
table is arithmetic on measured distributions and **closes**: the channels are
defined so they sum to the measured gap, and the residual is printed next to
them.

Inputs (all read-only):

    results/engine_v0/F2_2025_s200_v5b_A            served v5b stack, 75 seeds, 5,710 games
    results/engine_v0/F2_2025_s200_v1_clockv3c_A    v3c clock, 200 seeds (power cross-check)
    results/g4_diag/tap/                            NEW instrumented tap, 500 games x 10 seeds
    data/raw/hoopr/team_box/team_box_{2022..2025}   the ACTUAL the grader reads
    data/processed/possessions_v2/chances_2025      the pbp event layer
    data/processed/models/rebound/events_v1.parquet rebound opportunities 2022-2025
    data/processed/models/rebound/s1_confirm/S1_weekly/F2/   the SERVED rebound artifacts

    scripts/diag_g4_oreb_fta_v1.py   truth reconciliation, actual mix/rates, sim aggregates,
                                     offline scoring of the served rebound arm on fold 2 (new)
    scripts/diag_g4_tap_v1.py        the instrumented sim tap (new)
    scripts/diag_g4_report_v1.py     the channel decomposition and the multi-level cuts (new)
    results/g4_diag/{g4_offline_*.json, g4_report.json}   machine-readable output

**The tap is proved, not asserted, to be the served engine.** `diag_g4_tap_v1.py`
wraps `ad.event`, `ad.reb`, `loop.categorical` and `loop._shoot_trip` **in its own
process only** (the pattern `diag_late_game_tap_v1.py` established); `src/cbb_sim/`
is unchanged. On the 5,000 (game, seed) pairs it shares with the served 75-seed run,
its `games.parquet` is **BIT-IDENTICAL** on every column. The 500-game subset is
systematic (every 11th game in schedule order) and sits 0.035 pp (OREB%) and
0.020 pp (FTA/FGA) from the full run.

---

## 0. The two misses, and the source of the ACTUAL

| quantity | sim v5b (75 seeds) | sim v3c (200 seeds) | ACTUAL (hoopR box) | gap |
|---|---:|---:|---:|---:|
| OREB% pooled | 0.28280 | 0.28289 | 0.29841 | **-1.561 pp** |
| FTA/FGA pooled | 0.31726 | 0.31682 | 0.32955 | **-1.229 pp** |

The two clock arms agree to 0.01 / 0.04 pp, an order of magnitude inside both
misses, and the 200-seed paired A-B noise band on these lines is 0.0001
(`gate_noise_band_F2_2025_s200_v1_clockv3c_2026-09-11.md`). **Both misses are
real and neither is a clock effect.** Everything below is read off the served
v5b stack.

**Grading truth is the verified box, never pbp accumulation** (CLAUDE.md). The
two sources are reconciled first, on the 11,179 team-games that carry both:

| | box / team-game | pbp / team-game | pbp - box | team-games disagreeing |
|---|---:|---:|---:|---:|
| OREB | 10.349 | 10.390 | +0.0404 | 24.0% |
| DREB | 24.349 | 24.329 | -0.0201 | 20.6% |
| FTA | 19.117 | 18.934 | **-0.1828** | 7.9% |
| FGA | 58.006 | 57.949 | -0.0578 | 2.7% |

Pooled: OREB% box 0.29827 / pbp 0.29923 (**+0.096 pp**); FTA/FGA box 0.32957 /
pbp 0.32674 (**-0.282 pp**, of which -0.318 pp is the FTA count and +0.035 pp
the FGA count). The FTA line is the already-identified technical-free-throw
exclusion: `docs/tests/ft_trip_reconciliation_2026-09-10.md` showed technical
attempts explain 94.3% of disagreeing team-games exactly and 96% of the 2025
aggregate (1,961 of a 2,044-attempt gap). **This is a channel, not a nuisance:
the engine has no technical-FT rule, so it cannot produce those attempts even
in principle.**

One bookkeeping question closed by direct measurement rather than assumption:
hoopR `total_rebounds - (offensive_rebounds + defensive_rebounds) = 0 on all
11,400 2025 team-games, every row`. **Team and dead-ball rebounds are not inside
the graded numerator or denominator**, so the engine dropping its DEAD class
from both is the matching convention, not a leak (and in any case DEAD cancels
exactly: the engine composes `p = (1-dead) x p_binary`, so
`oreb/(oreb+dreb) = p_binary` regardless of the dead share).

---

## 1. OREB%: the channel chain

Four channels, each measured as a step from the one above, so they close by
construction. `B` and `C` come from scoring the **served** S1_weekly rebound
artifacts on the **real fold-2 rows** (the 388,338 live 2025 opportunities),
choosing each row's artifact by the same date rule `ArtifactManifest._select`
uses, with the trained feature order asserted.

| level | value | channel to the next | pp |
|---|---:|---|---:|
| A' ACTUAL, hoopR box (what G4 grades) | 0.29841 | grading source | **+0.082** |
| A ACTUAL, pbp event layer (the model's own target) | 0.29923 | fold-2 model calibration | **-1.137** |
| B served arm on real rows, TRUE `blocked_f` | 0.28786 | `blocked_f = 0` engine feed | **-0.740** |
| C served arm on real rows, `blocked_f = 0` | 0.28046 | sim state / mix distribution | **+0.234** |
| D SIM realised (full 75-seed run) | 0.28280 | | |

| channel | pp | share of the -1.561 pp gap |
|---|---:|---:|
| grading source (box -> pbp) | +0.082 | -5.3% |
| **fold-2 calibration of the served rebound arm** | **-1.137** | **+72.9%** |
| **engine feeds `blocked_f = 0`** | **-0.740** | **+47.4%** |
| sim state / mix distribution | +0.234 | -15.0% |
| **closure** | **-1.561** | residual **+0.0000 pp** |

Two channels are negative and two positive, which is why the negative shares
exceed 100%; the arithmetic is exact.

### 1.1 Channel 2 (73%): the rebound model is a season behind

The served arm under-predicts its own fold-2 test season by 1.137 pp, in
**every** calibration decile (-0.40 to -2.54 pp) — a level error, not a shape
error. The cause is a monotone season drift the pooled fit averages away:

| season | live OREB% (event layer) | box OREB% |
|---|---:|---:|
| 2022 | 0.28243 | 0.28126 |
| 2023 | 0.28683 | 0.28528 |
| 2024 | 0.29155 | 0.28994 |
| 2025 | **0.29923** | **0.29841** |

`S1_weekly` refits 23 times through the test season, but on the **pooled**
2022-to-date window with no season term and no recency weight. Its training
pool never gets near the 2025 level:

| refit | max_train_date | n_pool | pool OREB% | share of pool from 2025 |
|---|---|---:|---:|---:|
| 2024-11-04 | 2024-04-08 | 1,135,739 | 0.28704 | 0.0% |
| 2024-12-30 | 2024-12-29 | 1,277,231 | 0.28869 | 11.1% |
| 2025-02-24 | 2025-02-23 | 1,457,085 | 0.29000 | 22.1% |
| 2025-03-24 | 2025-03-23 | 1,521,053 | 0.29015 | 25.3% |
| 2025 season level | — | — | **0.29923** | — |

The model's mean prediction (0.28786) tracks its pool, not the test season, and
its calibration error shrinks as the pool absorbs 2025 without ever closing:

| month of 2025 | n | actual | model (true `blocked_f`) | cal err pp |
|---|---:|---:|---:|---:|
| Nov | 82,739 | 0.30231 | 0.28804 | -1.427 |
| Dec | 62,627 | 0.30081 | 0.28772 | -1.310 |
| Jan | 97,136 | 0.30392 | 0.28699 | -1.693 |
| Feb | 91,844 | 0.29343 | 0.28798 | -0.545 |
| Mar | 52,792 | 0.29448 | 0.28919 | -0.529 |
| Apr | 1,200 | 0.27750 | 0.28564 | +0.814 |

`rebound.build_design` already computes `season_idx`, and **no feature set
contains it**; `TEAM_FEATURES`/`MISS_FEATURES`/`STATE_FEATURES` have no season,
no week, and the trainer applies no sample weight. The model cannot see the
drift and is not told to forget the old seasons.

### 1.2 Channel 3 (47%): the engine tells the rebound model nothing was blocked

`loop.py`'s rebound block sets `xr[:, I["blocked_f"]] = 0.0` on every
opportunity, because **the cascade has no shot-block model** — there is no event
class, no rate table and no `blocked` draw anywhere in the engine. `blocked_f`
is nevertheless in the served `C_plus_state` bundle, and blocked misses are both
common and very different:

| miss type | blocked share of live opps | actual P(OREB) unblocked | actual P(OREB) blocked |
|---|---:|---:|---:|
| rim | 0.2617 | 0.3688 | 0.4237 |
| jump2 | 0.0816 | 0.2822 | 0.4113 |
| three | 0.0143 | 0.2857 | 0.4274 |
| ft | 0.0000 | 0.1377 | (n=1) |

**The model handles the feature correctly.** On blocked rows with the true
feature it predicts 0.4225 / 0.3995 / 0.3997 against actuals of 0.4237 / 0.4113
/ 0.4274. Zeroing it costs -6.3 pp (rim), -13.1 pp (jump2), -12.9 pp (three) on
those rows, and -0.740 pp pooled:

| miss type | model true | model `blocked_f=0` | cost pp |
|---|---:|---:|---:|
| rim | 0.37803 | 0.36166 | -1.637 |
| jump2 | 0.27955 | 0.26890 | -1.065 |
| three | 0.27325 | 0.27141 | -0.184 |
| ft | 0.12882 | 0.12882 | -0.001 |

This is an **engine-feed defect, not a model defect**: the engine is asking a
correctly-fitted model a counterfactual question ("what happens when nothing is
ever blocked?") and getting the honest answer to it.

### 1.3 Channel 4 (+0.234 pp): the sim's own distributions are NOT the problem

Split exactly:

| sub-channel | pp |
|---|---:|
| miss-type MIX (sim vs real rows) | +0.055 |
| within-type rate (state distribution) | +0.070 |
| Monte-Carlo realisation (368,826 opps, 1 binomial SE = 0.074 pp) | +0.074 |
| tap subset vs full run | +0.035 |
| total | +0.234 |

| miss type | sim live-opp mix | actual mix | sim mean p(OREB) | real-rows p(OREB) at `blocked_f=0` |
|---|---:|---:|---:|---:|
| rim | 0.25445 | 0.24538 | 0.36149 | 0.36166 |
| jump2 | 0.24328 | 0.24433 | 0.26965 | 0.26890 |
| three | 0.41286 | 0.42276 | 0.27245 | 0.27141 |
| ft | 0.08942 | 0.08753 | 0.13028 | 0.12882 |

### 1.4 Explicitly ruled out

- **The rotation.** The served rebound arm is **team-level** (`C_plus_state`;
  `D_plus_lineup` lost on its own fold, `experiments.md` section 5) and
  `loop.py` calls it with `inp.team_static[...]` plus the shared state block —
  **no slot block is passed**. The known 25%-bench-heavy floor composition
  cannot move team OREB% through this path. Structural, not statistical.
- **The miss mix / the L16 rim-override fix.** +0.055 pp, 3.5% of the gap and
  the wrong sign to be a cause.
- **Possession-continuation bookkeeping.** `oreb_chain_truncated` fired **9
  times in 428,250 game-sims**; DEAD cancels in the ratio; the graded box
  denominator excludes team rebounds (section 0).
- **Grading source.** +0.096 pp, and the wrong sign.

### 1.5 Multi-level evidence (OREB%)

| cut | cells | delta (sim - box), pp |
|---|---|---|
| per-game | 5,700 games | mean -1.412, median -1.283, SD 6.250, MAE 5.125, P(sim<act) 0.576 |
| site | away / home / neutral | -1.51 / -1.56 / -1.73 (flat) |
| conference | non-conf / conf | -1.81 / -1.42 |
| month | Nov..Mar | -1.93 / -1.72 / -1.92 / -0.98 / -1.16 (Apr n=34 **UNDERPOWERED**) |
| period | 1 / 2 / OT1 | -1.72 / -1.62 / -3.16 (OT2 n=179, OT3 n=12, **UNDERPOWERED**) |
| game minute | 9 buckets, 24.8k-46.7k opps each | -1.21 to -2.43, no trend |

**Flat everywhere in clock and state, sloping hard in team.** That is the
signature of a level defect plus a responsiveness defect, and it is exactly
what sections 1.1-1.2 describe.

---

## 2. FTA/FGA: the channel chain

FTA/FGA is a ratio of two per-possession rates, so the engine's +2.0
possessions/game **cancels exactly** and is not a channel. Each entry below is
`(FTA of that kind / FGA)_sim - (FTA of that kind / FGA)_actual`, so the table
closes into the measured gap.

| channel | pp | share of the -1.229 pp gap |
|---|---:|---:|
| technical FTs + event/box feed gap (engine has no technical-FT rule) | -0.282 | +23.0% |
| and-one FTA / FGA | +0.122 | -10.0% |
| shooting-foul trip FTA / FGA | +0.378 | -30.8% |
| **bonus-trip FTA / FGA** | **-1.384** | **+112.6%** |
| other event-layer FTA (168 chances outside the three kinds) | -0.044 | +3.6% |
| tap subset vs full run (measurement, not a defect) | -0.020 | +1.6% |
| **closure** | **-1.230** | unexplained residual **+0.001 pp** |

### 2.1 It is the bonus-trip RATE, not the trip size

| trip kind | sim trips / 100 poss | actual | ratio | sim FTA / trip | actual | ratio |
|---|---:|---:|---:|---:|---:|---:|
| and-one | 2.2169 | 2.0975 | 1.057 | 1.0000 | 1.0003 | 1.000 |
| shooting foul | 7.2938 | 7.2446 | 1.007 | 2.0402 | 1.9955 | 1.022 |
| **bonus** | **5.2586** | **5.8686** | **0.896** | 1.8739 | 1.8660 | 1.004 |

**Trip size is right everywhere** (<= 2.2% on every kind). The 1-and-1 vs
double-bonus split is right too: 52.9% of the engine's bonus trips are
2-attempt double-bonus trips against the actual's 51.4%. The engine's
`shooting_trip_three_attempt_share` (4.0% of shooting trips get three) against
the actual's 5.3% is the whole of the +2.2% shooting-trip size difference, and
it pushes the *wrong* way for the gate. **Nothing about the 2-vs-3-vs-1-and-1
trip table, or the era flag, is a channel.** The bonus trip rate being 10.4%
low is.

### 2.2 The bonus trip rate: 59% state occupancy, 41% conditional

| | sim | actual |
|---|---:|---:|
| P(offence in bonus) | 0.28373 | 0.30201 |
| P(bonus trip \| in bonus) | 0.16058 | 0.16763 |
| P(bonus trip \| not in bonus) | 0.00034 | 0.00037 |
| bonus trips per chance | 0.04581 | 0.05088 (-9.97%) |

Shapley split of the rate gap: **bonus-state OCCUPANCY 59.0%, CONDITIONAL trip
rate 41.0%** (residual 9e-19).

The occupancy channel has a named mechanism. The engine accrues non-shooting
team fouls from **one constant**,
`silent_foul_per_possession = 0.123346` (`engine_rules_from_data`; `loop.py`,
"non-shooting foul that awards no attempt"), applied identically in both halves
and independent of clock, score and role. Team fouls therefore accrue linearly,
and the bonus arrives at a fixed fraction of each half:

| game minute | sim P(in bonus) | actual | delta pp |
|---|---:|---:|---:|
| 0-4 | 0.0007 | 0.0003 | +0.04 |
| 5-9 | 0.0580 | 0.0317 | **+2.63** |
| 10-14 | 0.3039 | 0.2343 | **+6.95** |
| 15-19 | 0.6102 | 0.5790 | +3.13 |
| 20-24 | 0.0024 | 0.0093 | -0.69 |
| 25-29 | 0.0975 | 0.1431 | **-4.56** |
| 30-34 | 0.3984 | 0.5432 | **-14.48** |
| 35-37 | 0.6533 | 0.8060 | **-15.27** |
| 38-40 | 0.7935 | 0.8037 | -1.02 |

Every bucket is powered (57,480-100,408 sim chances, 61,022-111,927 actual).
**The engine reaches the bonus too early in the first half and up to 15 pp too
rarely in the second**, and the second half is where a bonus minute is worth the
most free throws. Conditional on actually being in the bonus, the served
possession-outcome model is close to right in the second half and low in the
first and in the last two minutes:

| game minute | sim P(trip \| bonus) | actual |
|---|---:|---:|
| 10-14 | 0.1070 | 0.1212 |
| 15-19 | 0.1255 | 0.1301 |
| 25-29 | 0.1385 | 0.1391 |
| 30-34 | 0.1576 | 0.1567 |
| 35-37 | 0.1736 | 0.1710 |
| 38-40 | 0.2480 | 0.2696 |

So most of the "41% conditional" share is the engine querying a correctly-shaped
model at the **wrong (bonus, clock) states**, plus a genuine end-game shortfall.

The FT supply follows the occupancy error exactly, sign flip and all:

| game minute | sim FTA/FGA | actual | delta pp | share of the box gap |
|---|---:|---:|---:|---:|
| 0-4 | 0.1648 | 0.1483 | +1.65 | -17.5% |
| 5-9 | 0.2229 | 0.2048 | +1.81 | -15.9% |
| 10-14 | 0.2772 | 0.2588 | +1.84 | -9.8% |
| 15-19 | 0.3499 | 0.3444 | +0.55 | -7.9% |
| 20-24 | 0.2472 | 0.2475 | -0.03 | +2.1% |
| 25-29 | 0.2947 | 0.3178 | -2.31 | +22.1% |
| 30-34 | 0.3623 | 0.4234 | **-6.11** | **+56.9%** |
| 35-37 | 0.4588 | 0.5165 | -5.78 | +32.3% |
| 38-40 | 0.8102 | 0.9027 | -9.25 | +13.1% |

(Shares sum to +75.5%, the event layer's share of the box gap; the other 23% is
the technical channel.) **The engine's FT supply is not uniformly low — it is
mis-timed.** A fix that raises the overall foul rate would make minutes 0-14
worse.

### 2.3 The final 2:00 is 21% of the gap, and the late-game lane already owns it

Exact four-term split of the event-layer gap by clock window (FTA numerator and
FGA denominator, each per total chance, so the possession count cancels):

| window | term | pp | share of the -1.229 pp box gap |
|---|---|---:|---:|
| final 2:00 (reg) | FTA | +0.043 | -3.5% |
| final 2:00 (reg) | FGA (denominator) | -0.297 | +24.2% |
| rest of game | FTA | -0.588 | +47.9% |
| rest of game | FGA (denominator) | -0.086 | +7.0% |
| | event-layer gap -0.928 pp | sum -0.928 | residual +0.0000 |

**Final 2:00 total: -0.254 pp = 20.7% of the box gap.** Material, but not the
owner, and it arrives through an unexpected door: the window's own FTA term is
slightly **positive**. The engine's local FT rate in the window is 0.788 against
the actual's 0.929, but it plays 6.82% of its chances there against the actual's
5.99% and takes **5.66% of its FGA there against the actual's 4.81%** — because
in reality the trailing team is fouling instead of shooting. The window hurts
FTA/FGA mainly by inflating the denominator.

**This is the late-game regime, and it is already pre-registered.** The
`docs/models/late_game/experiments.md` section-1 pre-registration (written
2026-09-11, running now) names the fouling channel as arm rank 1 and measured
exactly this: trailing-minus-leading bonus-FT rate +0.265 actual vs +0.024 sim,
9% reproduced. This lane's numbers agree and **size** it: the final 2:00 is
20.7% of G4's FTA/FGA miss, split -3.5% numerator and +24.2% denominator.
**Nothing about the late window is pre-registered here. No duplicate is filed,
and `docs/models/late_game/` is not touched.** The other ~79% sits in minutes
25-37, outside that lane's window, and that is what section 4 pre-registers.

### 2.4 Multi-level evidence (FTA/FGA)

| cut | cells | delta (sim - box), pp |
|---|---|---|
| per-game | 5,700 games | mean -1.551, median -0.780, SD 10.762, MAE 8.532, P(sim<act) 0.529 |
| site | away / home / neutral | **-0.43 / -1.62 / -2.63** |
| conference | non-conf / conf | **-2.28 / -0.61** |
| month | Nov..Mar | **-2.99** / -1.21 / -0.95 / -0.20 / -0.67 (Apr n=34 **UNDERPOWERED**) |
| game minute | 9 buckets | +1.84 (min 10-14) to -9.25 (min 38-40), sign flip at minute 20 |

Two further findings that neither of the two named mechanisms explains and that
belong in the pre-registration as gates:

1. **The home whistle is under-reproduced.** Actual FT rate home 0.3486 vs away
   0.3068, a +4.18 pp home advantage; the sim gives 0.3324 vs 0.3025, +2.99 pp.
   The engine reproduces 72% of it. `site_home`/`site_away` are already in the
   served possession-outcome feature set, so this is not a missing column.
2. **The miss is an early-season and non-conference problem** (-2.99 pp in
   November against -0.20 pp in February; -2.28 pp non-conference against
   -0.61 pp conference — and the two cuts overlap heavily).

---

## 3. The per-team quintile slope: a third defect both misses share

Teams bucketed by their **2024** (prior-season) box value of the same metric
(teams with >= 10 games in 2024), then graded on 2025. Prior-season bucketing,
not same-season, so the cut is not circular.

| quintile | teams | prior OREB% | sim 2025 | actual 2025 | delta pp |
|---:|---:|---:|---:|---:|---:|
| 1 | 73 | 0.23204 | 0.26119 | 0.26541 | -0.42 |
| 2 | 72 | 0.26738 | 0.27617 | 0.28628 | -1.01 |
| 3 | 72 | 0.28631 | 0.27853 | 0.29286 | -1.43 |
| 4 | 72 | 0.30893 | 0.29090 | 0.31208 | -2.12 |
| 5 | 73 | 0.34595 | 0.30381 | 0.32846 | **-2.47** |

sim span 0.0426 vs actual span 0.0631, **slope ratio 0.676**, monotone 4/4,
**gap slope Q5-Q1 = -2.04 pp**.

| quintile | teams | prior FT rate | sim 2025 | actual 2025 | delta pp |
|---:|---:|---:|---:|---:|---:|
| 1 | 73 | 0.26615 | 0.30698 | 0.30787 | -0.09 |
| 2 | 72 | 0.30093 | 0.31285 | 0.31968 | -0.68 |
| 3 | 72 | 0.32544 | 0.31673 | 0.33075 | -1.40 |
| 4 | 72 | 0.35324 | 0.32133 | 0.34139 | -2.01 |
| 5 | 73 | 0.39618 | 0.32912 | 0.35021 | **-2.11** |

sim span 0.0221 vs actual span 0.0423, **slope ratio 0.523**, monotone 4/4,
**gap slope Q5-Q1 = -2.02 pp**.

**Both gaps slope, and both slope the same way: the engine is nearly right on
the teams that rebound and foul least and misses by 2.0-2.5 pp on the teams that
do it most.** A pure level shift would give a flat gap, so this is a third,
separate defect: the engine compresses team-level spread to 0.68 (OREB) and 0.52
(FT rate) of the actual.

The mechanism is visible in the season profile. Both `rebound`'s `off_oreb_c` /
`opp_def_dreb_c` and `possession_outcome`'s `off_ftr_c` are as-of, league-centred
expanding means that start at **exactly 0.0 (the league mean) for every team**
and carry nothing from the prior season:

| segment | OREB% slope ratio | gap Q1 / Q5 pp | FT rate slope ratio | gap Q1 / Q5 pp |
|---|---:|---|---:|---|
| Nov-Dec | **0.561** | -0.30 / -3.25 | **0.356** | -0.63 / -3.39 |
| Jan | 0.799 | -1.36 / -2.50 | 0.636 | -0.25 / -1.73 |
| Feb-Apr | 0.738 | +0.09 / -1.55 | 0.663 | +0.41 / -0.99 |

(~71-72 teams per quintile per segment, teams with >= 5 graded games in the
segment.) **The compression is worst in November-December and never fully
recovers.** This is the same defect `possession_outcome` round 4 found for its
own style rates (early-season shrinkage, arms G1-G4, section 9) and that round
4b decided in favour of `G2` (prior-season shrink) — **the rebound model has
never had that arm run against it at all.**

---

## 4. Ownership, and what is pre-registered

| miss | owner | channels |
|---|---|---|
| **OREB% -1.56 pp** | **`docs/models/rebound/`** | pooled-season level -1.14 (73%); `blocked_f=0` engine feed -0.74 (47%); team-slope compression (on top, -2.0 pp Q5-Q1) |
| **FTA/FGA -1.23 pp** | **`docs/models/possession_outcome/`** (foul accrual + the `FT_trip_bonus` class), with **`docs/models/free_throw/`** owning the technical-FT scope gap | bonus-trip rate -1.38 (113%), of which occupancy 59% / conditional 41%; technicals -0.28 (23%); and-one +0.12; shooting trips +0.38; final 2:00 -0.25 (21%, **already owned by `late_game`**) |

Three PROPOSED pre-registrations are appended in the same commit as this
document, none of them run and none of them changing a default:

- `docs/models/rebound/experiments.md` **section 9** — round 3: the season-drift
  level, the `blocked_f` feed, and prior-season carry for the as-of form
  features.
- `docs/models/possession_outcome/experiments.md` **section 13** — round 6: the
  team-foul accrual law that sets the bonus state, and the `FT_trip_bonus`
  conditional.
- `docs/models/free_throw/experiments.md` **section 9** — the technical-FT
  scoring rule, already an open item in that model's `model.md` section 9 and
  now sized at 23% of G4's FTA/FGA miss.

`docs/models/late_game/` is deliberately **not** edited: the final-2:00 channel
is 20.7% of the FTA/FGA miss and is already inside that lane's live
pre-registration.

---

## 5. What this document does NOT establish

1. **No counterfactual was simulated.** Every channel is arithmetic on measured
   distributions plus one offline re-scoring of the served arm on real rows.
   Which channel actually moves G4 most in a closed loop is the pre-registered
   experiment's own measurement, not a claim here.
2. **The tap is 500 games x 10 seeds.** It is bit-identical to the served run on
   its 5,000 overlapping (game, seed) pairs and within 0.035 pp of the full run
   on both graded lines, but every tap-derived number carries that subset. The
   two headline gaps are quoted from the full 75-seed run.
3. **The 59/41 occupancy/conditional split is a pooled Shapley split** of a
   quantity whose two parts interact through the clock (section 2.2). The
   by-minute tables are the real evidence; the single number is a summary.
4. **The and-one and shooting-trip channels are positive** (+0.12 and +0.38 pp).
   They are reported because the table must close, not because anything should
   be done about them; touching either would make G4 worse.
5. **`silent_foul_per_possession` is a measured constant, not a fitted model.**
   Calling it the mechanism behind the occupancy error is a claim about its
   *shape* (one number for the whole game), backed by the by-minute table. What
   the right law is — and whether it belongs to `possession_outcome` as a class
   or to a separate foul sub-model — is arm A vs arm D of the pre-registration,
   not decided here.
6. **The team-slope compression is measured against a prior-season bucketing.**
   It refutes a league-mean-start as-of feature; it does not by itself identify
   which of the several as-of features carries it, and the OREB and FT rate
   quintile cuts are one-way.
7. **Nothing here is a fix.** No multiplier, cap, clip, offset or blend on sim
   output is proposed anywhere, and none would be acceptable
   (`docs/SIM_GUARDRAILS.md`).
