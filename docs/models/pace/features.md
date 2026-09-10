# pace -- features

Layout per `docs/models/DOCUMENTATION_STANDARD.md`. Absolute numbers live in `experiments.md`; this file is the
"exactly what goes in" reference for whoever wires the sim. Everything here is **pregame by construction**: every
rating is an as-of join with strictly-before semantics, every style rate is `shift(1)+expanding().mean()` (uses
only that team's STRICTLY EARLIER games that season), and rest days come from the previous calendar date in that
team's own schedule. `tests/test_pace.py` corrupts one game's box line, rebuilds the table, and requires that
game's own features to be bit-identical while later games' rolling/as-of features move.

---

## 1. Source manifest (canonical table)

Built by `cbb_sim.models.pace.build_pace_table`, one row per game.

| Feature | Dtype | Source file | Computation | Fallback value | Notes |
|---|---|---|---|---|---|
| `home_tempo_rel` / `away_tempo_rel` | float64 | `data/processed/ratings/own_ratings_{season}.parquet` | own ridge tempo ratio to the as-of league mean (`cbb_sim.ratings.own_ratings`), joined strictly before the game date (`orat.join_as_of`) | train-time median (~1.0) | KenPom AdjT convention; > 1 = faster than the as-of league mean |
| `home_kp_adj_t_rel` / `away_kp_adj_t_rel` | float64 | `data/processed/kenpom_snapshots.parquet` | centred KenPom `adj_t_rel`, latest snapshot strictly before the game date (`cbb_sim.control.features.join_kenpom_as_of`, reused verbatim from the Control) | train-time median (~1.0) | Centred column only -- raw `adj_t` is banned (CLAUDE.md) |
| `league_tempo_mean_asof` | float64 | own ratings | the as-of table's own league mean tempo for that (season, date) | train-time median | Feeds `multiplicative` only; identical for both sides of a game by construction |
| `neutral` | float64 {0,1} | `data/processed/games_universe.parquet` | `neutral_site` | 0 | The only site term this bake-off carries (no home/away pace edge) |
| `days_since_start` | float64 | games universe | calendar days since that season's first D-I game date | train-time median | 0-155ish |
| `month` | float64 | games universe | calendar month of `game_date` | train-time median | 11, 12, 1, 2, 3, 4 |
| `season_index` | float64 | games universe | `season - 2022`, a FIXED anchor so F1 and F2 share one scale | train-time median | A time-trend covariate, not a banned raw rating level (SIM_GUARDRAILS section 4: "outcome models need a season-level term") |
| `home_own_style_{tpa,fta,tov}_per100` | float64 | hoopR `team_box` via `orat.load_team_games` | `100 * count / game_poss` for that team-game, then `groupby(season, team_id).shift(1).expanding().mean()` | train-time median | Strictly the team's own STRICTLY EARLIER games that season |
| `home_own_style_oreb_pct` | float64 | same | `oreb / (oreb + opponent's dreb)` per team-game, same shift+expanding treatment | train-time median | |
| `home_opp_allowed_style_{tpa,fta,tov}_per100` / `..._oreb_pct` | float64 | same | attach the OPPONENT's own raw per-game rate for that game (self-merge on `game_id`+`opp_team_id`), then the SAME `groupby(season, team_id).shift(1).expanding().mean()` on the team whose defense allowed it | train-time median | "What has this team's defense allowed so far this season", pregame |
| `away_*` | -- | -- | mirror of every `home_*` column above for the away team | -- | -- |
| `home_rest_days` / `away_rest_days` | float64 | games universe, that team's own schedule | `game_date - previous game_date` for that team within the season | train-time median | NaN on a team's first game of the season |
| `home_b2b` / `away_b2b` | float64 {0,1} | derived from rest_days | `1.0 if rest_days <= 1 else 0.0` | train-time median | NaN when `rest_days` is NaN (first game) |
| `conf_game` | float64 {0,1} | `data/raw/hoopr/schedules/mbb_schedule_{season}.parquet` | `home_conference_id == away_conference_id`, both non-null | train-time median | Same conference-id columns `cbb_sim.data.universe` uses for the D-I flag |

### 1a. Targets (not features)

| Target | Source | Computation |
|---|---|---|
| `poss_box` (T_box) | hoopR `team_box` | `FGA - OREB + TOV + 0.44*FTA`, averaged over both teams (`orat.load_team_games`'s `game_poss`) |
| `poss_pbp` (T_pbp) | `data/processed/possessions_pbp.parquet` | See section 4 below and `scripts/build_possessions_pbp.py`'s module docstring for the full rule |

### 1b. Fallback policy

Every model class persists its own train-time median per feature (`pace._fit_medians`) and fills a missing
value with it before scoring (`pace._fillna`), the same contract as the Control's `FittedModel.design`.
Missing rates in practice: 3.0-3.3% of team-game rows (a team's first game of the season, where no strictly-prior
game exists for the rolling/rest features) -- expected and not a data defect.

---

## 2. Feature sets tested

| Set name | Included features | Rationale |
|---|---|---|
| `A_tempo` | both teams' `tempo_rel` (own + KenPom) + `neutral` | The minimal, tempo-only bundle -- literally what a multiplicative KenPom-style formula needs |
| `B_plus_season` | A + `days_since_start`, `month`, `season_index` | Tests whether a season-level/seasonal-progress term helps pace specifically (L4/L5: scoring and pace both show seasonal and cross-season drift) |
| `C_plus_style` | B + both teams' own and opponents-allowed as-of per-100 rates of 3PA/FTA/TOV/OREB% | Tests whether shot-selection/rebounding style (not just tempo) carries independent pace signal |
| `D_plus_state` | C + rest days, back-to-back flag (each team), conference-game flag | Tests schedule-state effects on pace (fatigue, familiarity) |

**Chosen: `A_tempo`**, model class `multiplicative` -- see `experiments.md` R4 for the decision rule and the
measured noise floor that produced it (every richer bundle and every fitted model class landed inside that
floor).

---

## 3. Rejected features

| Feature | Why not |
|---|---|
| Raw KenPom `adj_t` / raw own-ratings tempo level | CLAUDE.md bans raw rating levels; only the as-of centred/relative columns (`adj_t_rel`, `tempo_rel`) are used |
| Home/away pace-specific coefficients | This bake-off's site term is `neutral` only, matching the Control's own pace model (`docs/models/control_engine/features.md` section 1b); home-court pace edges are not part of this spec |
| CBBD season ratings endpoints | End-of-season snapshots; banned as pregame features (`docs/SIM_GUARDRAILS.md`, L7) |
| Embedded ESPN market columns in hoopR pbp | Stripped from features by the data rules (CLAUDE.md) |
| Pre-summed (`home + away`) tempo/style columns | Considered (it is the Control's own convention) and rejected for this fresh bake-off: two columns per quantity let ridge/GLM/LightGBM learn any combination, symmetric or not, at no extra cost. `model.md` decisions log item 1 |

---

## 4. T_pbp -- the exact possession-ending rule (summary; full rule in the script)

`scripts/build_possessions_pbp.py`'s module docstring is the authoritative version (event-role mapping, the
full state machine, and every known limitation with its measured size). Summary:

- Every hoopR pbp `type_text` (24 distinct values across 2022-2026) is mapped to one of SHOT / FT / OREB / DREB
  / TOV / PERIOD_END / TECH / IGNORE; an unmapped type raises `ValueError` (`classify()`,
  `tests/test_pace.py::test_unknown_pbp_event_type_raises`) per `docs/SIM_GUARDRAILS.md`'s standing requirement
  that every pbp-derived feature come from an explicit mapping table with a test that fails on an unknown type.
- A forward pass per game (sorted by `game_play_number`, the column verified 100% monotonic within `game_id` --
  hoopR's own `sequence_number` is NOT reliable, only 58-75% of games are monotonic on it) counts a TERMINAL
  event (a made field goal not followed by an and-one free-throw trip, a defensive rebound, a turnover, the
  last made free throw of a trip, or the period/game ending with a possession still open) against the team
  whose possession just ended.
- `poss_pbp` per game is the mean of the two teams' own terminal-event counts (mirrors T_box's "mean over both
  teams").
- Excludes: `pbp_truncated` games, games with no pbp coverage at all (`has_pbp == False`, ~1.3-2.0%/season,
  distinct from truncation), and non-D-I games.
- Reliability, pooled 2022-2025 (21,965 games with both definitions): corr 0.9717 with T_box, mean difference
  (pbp - box) +0.219 possessions, mean absolute difference 0.854, SD of the per-game difference 1.339 --
  the pre-registered "unless it is unreliable" clause does not trigger. `experiments.md` R1.
