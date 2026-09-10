# `build_team_asof` league-rate columns were silently zeroed for all three L4 binaries

Author: Sonnet bake-off runner (attribution completion), 2026-09-10. Found while
resuming the L4 player-attribution bake-off (`docs/models/attribution/RESUME.md`)
and diagnosing why the interrupted run's `assisted` / `stolen` / `blocked` targets
failed with `ValueError: Input X contains NaN` in `LogisticRegression`.

Fixed in `src/cbb_sim/models/attribution.py::build_team_asof`. Not a modelling
choice and not a change to what is compared -- a construction bug in the pregame
feature table, found and fixed per `CLAUDE.md`'s "no hand tuning" rule ("if the
engine's output has to be adjusted to resemble reality, the engine is wrong and
the responsible sub-model must be found and fixed").

---

## 1. What the defect actually is

`build_team_asof` computes, per season, six (population x side) count groups
(`made`/`ast`, `opp_made`/`ast_allowed`, `tov`/`tov_stolen`, `opp_tov`/`steals`,
`miss`/`miss_blocked`, `opp_miss`/`blocks`) and was accumulating **every**
season's groups into one flat `rows` list, then merging all of them
sequentially with one running `tg.merge(r, ..., how="outer")` loop.

Both seasons' six groups carry the **same** column names (`made`, `ast`,
`opp_made`, ...) -- correct within one season, where the names are unique, but
when the accumulated merge reaches the second season's `made`/`ast` group,
pandas sees a repeated non-key column name against the first season's already-
merged `made`/`ast` columns and silently suffixes both `_x`/`_y` rather than
raising. The bare column names (`made`, `ast`, ...) then never exist in `tg` at
all. The very next block,

```python
for c in TEAM_COUNT_COLS:
    if c not in tg.columns:
        tg[c] = 0.0
```

exists to backstop a genuinely-absent count (e.g. a class with zero events in
some season) -- but here it fires for **every one of the twelve** count columns,
because none of the bare names survived the merge, and manufactures a column of
all zeros for each. Every downstream quantity inherits the zero: the expanding
sum is zero, the day-level league sum is zero, and `_safe_div(0, 0)` (the league
as-of rate) is `NaN` on every row of every season for all three binaries.

The bug is invisible on any table built from a **single** season (the column-name
collision needs two seasons merging against each other) and invisible on the five
**choice** targets, which do not go through `build_team_asof` at all -- it exists
only to feed the three **binary** targets' team-level features
(`off_rate_c`, `def_rate_c` in `TEAM_FEATURES`).

## 2. Reproduction

Standalone re-execution of the merge loop on the cached 2024+2025 population
tables (`events_{made_fga,tov,miss_fga}_v2.parquet`) confirms it exactly: the
merged `tg` frame's dtype listing has `made_x`, `ast_x`, ..., `made_y`, `ast_y`,
... (12 suffixed pairs) and **no** bare `made`/`ast`/... columns at all.

Before the fix, `data/processed/models/attribution/team_asof_v2.parquet`:

| column | NaN rows | of 21,246 |
|---|---:|---:|
| `off_assisted_lg` | 21,246 | 100.0000% |
| `def_assisted_lg` | 21,246 | 100.0000% |
| `off_stolen_lg` | 21,246 | 100.0000% |
| `def_stolen_lg` | 21,246 | 100.0000% |
| `off_blocked_lg` | 21,246 | 100.0000% |
| `def_blocked_lg` | 21,246 | 100.0000% |

`off_{binary}_num` / `_den` were **not** NaN -- they read as a clean, real-looking
0.0 for every team-game, which is the more dangerous half of the bug: had the
run not crashed on the `_lg` `NaN`, `team_shrunk` (`(m * prior + num) / (m +
den)`) would have returned exactly the league prior for every row regardless of
a team's actual history, and the `off_rate_c` / `def_rate_c` features that
`team_ridge`, `aware_ridge` and the binary `lgbm` all depend on would have been
silently uninformative (zero variance in the num/den, whatever variance survived
in `lg` before the NaN masked it too) rather than erroring. The crash is what
surfaced it; a quieter downstream consumer would not have been protected by one.

## 3. The fix

`build_team_asof` now merges each season's six groups column-wise (safe --
their twelve names are unique within one season) into one per-season frame, and
concatenates the two seasons' frames row-wise (`pd.concat`, seasons never share
a `game_id` so this is a clean union, not a join). No other line of the function
changed.

After the fix, `team_asof_v2.parquet` (rebuilt via `--rebuild`):

| column | NaN rows | mean | SD | max |
|---|---:|---:|---:|---:|
| `off_assisted_num` | 0 | 191.919 | 126.843 | 730.0 |
| `off_assisted_den` | 0 | 372.668 | 235.888 | 1147.0 |
| `off_assisted_lg` | 0 | 0.5119 | 0.0078 | 0.523 |
| `off_stolen_lg` | 0 | 0.5499 | 0.0085 | 0.565 |
| `off_blocked_lg` | 0 | 0.1003 | 0.0016 | 0.108 |

The league rates now track the base rates already reported in
`experiments.md` section 2.2 (assist ~50.6-51.5%, steal ~54.9-56.5%, block
~9.9-10.1% across 2024/2025), and the per-team num/den now carry real,
team-varying exposure (SD 118-236 across teams) instead of a constant zero.

`pytest tests/test_attribution.py -q` is unaffected: 27 passed before and after
(none of the 27 tests exercised `build_team_asof`'s multi-season path -- a gap
worth closing, noted but not acted on here to stay in scope).

## 4. Scope

Confined to `cbb_sim.models.attribution.build_team_asof`. Grep basis
(`build_team_asof`, `TEAM_COUNT_COLS`, `TEAM_RATES`) across `src/cbb_sim/` and
`scripts/`: the function has exactly one caller
(`scripts/train_attribution_v1.py::build_tables`) and exactly one artifact
(`team_asof_{version}.parquet`), consumed only by the three L4 binary targets.
No other sub-model's league-rate or team-as-of construction shares this code
path (`rebound.team_rebound_form`, `fg_make.team_shot_form`,
`possession_outcome.build_team_form` and `usage`'s team-level pieces each have
their own, unaffected, single-pass expanding-sum builders). Referenced from
`docs/models/attribution/experiments.md` in the same commit as the completed
bake-off results.
