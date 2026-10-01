# Daily chain v3: sim, publish, grade, bias / CLV monitor (lane F, 2026-09-30)

`scripts/chain_daily_v3.py` is a sibling of `chain_daily_v2.py` (untouched; stages 1-8 are reused by import). It adds the four stages v2 left as stubs.
No engine sampling code, no served default and no sim output is edited anywhere. All outputs are under `results/daily*/` (gitignored).

## 1. Files

| file | role |
|---|---|
| `src/cbb_sim/live/daily.py` | pure logic: tipped-game split, sim summaries, publish frame, settlement, ledger upsert, rolling bias, per-stat bias, CLV, leak verdict wiring |
| `src/cbb_sim/live/lines.py` | line sources (CBBD client only), provider preference, de-vig |
| `scripts/run_daily_sim_v1.py` | stage SIM (`run_sim_stage`) |
| `scripts/run_daily_publish_v1.py` | stage PUBLISH |
| `scripts/grade_daily_v1.py` | stage GRADE |
| `scripts/diag_daily_bias_clv_v1.py` | stage BIAS / CLV monitor |
| `scripts/chain_daily_v3.py` | the chain; `--replay-season S --slate-date D` replays stages 9-12 on a past season with a faked clock |
| `scripts/diag_daily_replay_check_v1.py`, `scripts/run_daily_replay_f2_v1.sh` | the replay evidence below |
| `tests/test_daily_chain_v3.py` | 11 tests |

Live order inside one morning run: 1-8 as v2, then GRADE and BIAS_CLV (yesterday's slate, retried back 14 days for pending games), then SIM and PUBLISH (today's slate).

## 2. What each stage does

**SIM.** Loads the slate (`--schedule-source cbbd` with the 2027 games file and `team_crosswalk_v2`, tip times overridden from the hoopR schedule where `time_valid`; or `universe` for replay),
splits it by the injected clock into not-yet-tipped and already-tipped, refuses and logs the tipped ones (`skipped.json`, `[sim] SKIP` lines; a game with no tip time counts as tipped), builds live inputs
(`build_live`, as-of = created_at = the clock), runs the served engine in process on one core at `--seeds`, stamps every row with `created_at` and `tipoff_utc`, asserts `created_at < tipoff` per row
(`guards.assert_created_before_tipoff`) before writing, and writes the results contract to `results/daily/sim/<slate_date>/<run_id>/` (`games.parquet`, `run_meta.json` with `live: true`, `slate.parquet`, `skipped.json`, `_DONE.json`).
`run_id` defaults to `s{seeds}_o{offset}`. `--strict` raises `LeakGuardError` on any tipped game instead of skipping. Season 2026 requires a deliberate `CBB_UNSEAL=1`; the chain never sets it for this stage.

**PUBLISH.** Per game: sim margin and total mean, SD and 5/25/50/75/95 quantiles, home win frequency, OT rate, per-team box means, and market probabilities at the stored line: P(home covers), P(away covers), P(push), P(over), P(under), P(push),
de-vigged moneyline probability, vig, ML edge, ATS and O/U disagreement in points, the lean on each, EV at flat -110 (CBBD has no price for spreads and totals). The line, provider and `line_fetched_at` are stored in every row; `published_at < tipoff`
and `line_fetched_at <= published_at` are asserted. Lines come only from the CBBD client (`pull_cbbd.http_get`, quota floor enforced); provider preference Draft Kings, Bovada, then consensus (ESPN BET ended). Output: `slate.parquet`, `slate.csv`, `slate.md`,
`lines.parquet`, `publish_meta.json` under `results/daily/publish/<slate_date>/<run_id>/<publish_id>/`. All probabilities are raw sim frequencies; nothing is calibrated or adjusted. `publish_id` hashes the run id and the stored lines: identical lines are a no-op,
moved lines make a new publication beside the first.

**GRADE.** Takes the EARLIEST publication per game (the line that was actually available) and joins verified finals only: live source = `finals_verified_{season}.parquet` (hoopR and CBBD agree on both scores, from the daily ingestion);
replay source = the eval harness's verified truth (`CBB_TRUTH=verified_v1`). Games without a verified final stay pending (`pending.parquet`, retried by the next run). Settlement is at the stored line: ATS and O/U at flat -110 (win +1, loss -1.1, push 0, harness convention),
ML at the stored real odds. Rows are upserted into `results/daily/grade/ledger.parquet` on (game_id, run_id, publish_id). The report gives margin and total MAE and bias, MAE against the line, Brier against the de-vigged market, ATS / O/U ROI by disagreement bucket,
ML ROI by edge bucket, calibration deciles (n >= 50), game-clustered bootstrap CIs, each labelled UNDERPOWERED below its floor; for the day and for the whole ledger. Actual per-team event-layer counts are attached from `team_game_shots_v2`.

**BIAS / CLV monitor.** From the ledger: margin and total bias with SE and z over 14 d, 30 d and all; per-stat bias (3PA, rim and jump 2PA, FTA and the matching makes) where the truth table has the game; predicted-margin quintile vs actual margin;
closing-line value of every published lean (publish-time stored line vs the close from the same provider: points for ATS and O/U, de-vigged probability for ML), CLV agreement (share of moved lines that moved toward the lean), the surprise correlation, and the leak verdict
from `market.leak_verdict` with `docs/gates.yaml` tolerances. The CLAUDE.md rule is printed at the top of every report: *an edge that beats the close but cannot predict line movement is presumed leaked*. Alarms: |z| > 3 at n >= 50 for bias and per-stat bias,
LEAK-SUSPECT, mean CLV z < -3. It reports and alarms (stage status `alarm`); it never writes back to the sim, the publish files or the ledger. A unit test asserts the input frame is unchanged. Live close source = `lines_daily` `close` snapshots written by the chain's lines step.

## 3. Replay evidence (fold 2, 2024-25; clock faked; 2025-26 not touched)

Replay driver: `scripts/run_daily_replay_f2_v1.sh` (3 dates, 16 seeds, one core, root `results/daily_replay_f2`). Clock = 14:00Z of the slate date for sim and publish, 14:00Z next day for grade and monitor.
Replay line source: open lines exist in `lines_2025.parquet` only from 2025-02 (0% before), so the three dates are 2025-02-11, 2025-02-25, 2025-03-04. Publish used `replay_open` (spread and total from the OPEN columns; the moneyline is the close because the file has no open moneyline, flagged `ml_is_close_proxy`, so ML CLV is not computable in replay). `line_fetched_at` is the fake clock.

| date | games | sim rows | sim wall (build + engine) | graded | pending |
|---|---:|---:|---:|---:|---:|
| 2025-02-11 | 37 | 592 | 140 s (20:50:43 to 20:53:04) | 37 | 0 |
| 2025-02-25 | 35 | 560 | 162 s | 35 | 0 |
| 2025-03-04 | 44 | 704 | 183 s | 44 | 0 |

(about 30 s of each is the live input build; the engine is about 0.15 s per game-seed on one core.)

**created_at < tipoff firing on a late game.** Slate 2025-02-11, clock 2025-02-12T01:30Z (37 games, 23 tipped at or before the clock): `--strict` raised `LeakGuardError: 23 row(s) have created_at >= tipoff` before any engine work; the default mode logged
`[sim] SKIP game ...` for the 23, simulated the other 14, and the output has min tip 02:00Z and max created_at 01:30Z (`results/daily_replay_f2/sim/2025-02-11/late_demo`). The same case is a unit test.

**Idempotency.** Re-running sim, publish and grade for 2025-02-11 returned `cached: true`, `cached: true`, `new_ledger_rows: 0`; SHA-256 of `games.parquet`, the publication `slate.parquet` and the ledger identical before and after. A forced sim re-run under another run id at a later clock
reproduced all 592 rows identically except `created_at` (RNG keyed on seed and game id).

**Grade stage vs the eval harness** (`diag_daily_replay_check_v1.py`: the sims published against the replay CLOSE so the stored line equals the harness's, graded at the next-day clock, then `gates.build_grading_frame` + `market.gate_g10` on the same 116 games). 72 quantities compared: games graded,
games with a spread, model and close MAE on margin and total, Brier model and market, every cell of the ATS, O/U and ML-edge bucket tables (n, wins, losses, pushes, win %, ROI): **0 mismatches** (all exact to 1e-9). The three bootstrap mean ROIs differ by 4e-4, 4e-4 and 2.6e-3 (the bootstrap
resamples rows under one seed, so it depends on row order; compared as approximate). Output: `results/daily_replay_f2_check.json`.

**Monitor on the 116 replay games (open-to-close CLV).** Margin bias -1.26 (SE 1.03), total bias -3.86 (SE 1.78, z -2.2), no alarm. ATS leans: 108 with a close, 61 moved, CLV agreement 0.459, mean CLV -0.02 pt (SE 0.11); O/U: 87 moved, agreement 0.540, mean +0.12 pt (SE 0.18).
Surprise correlation 0.047. Verdict UNDERPOWERED (needs >= 100 moved lines for the agreement). The predicted-margin quintiles slope with the actual margin (-11.4 to +14.9 predicted, -8.3 to +14.7 actual). This is a plumbing proof at 16 seeds, not a performance read.

**Full chain dry run on the live 2027 files** (`chain_daily_v3.py --dry-run --date 2026-09-30 --slate-date 2026-11-02`, 20:51 EDT, 4 CBBD calls): schedule, lines, ingest, injuries, overrides ok; ratings BLOCKED (day-1 choices file); inputs BLOCKED (14 of 14 lane-G breakage rows reproduced); grade and bias_clv skipped (no publication, no ledger);
sim BLOCKED with the census (118 mapped, 38 unmapped non-D-I, 118 would simulate at an evening-before clock) and the prerequisites it needs; publish BLOCKED (dry run). The live line fetch was exercised against CBBD for 2026-11-02: 1 call, 0 rows (no 2027 lines yet, as expected).

## 4. Findings the PM needs

1. **Tip times decide whether day 1 is simulated at all.** Of the 118 mapped games on 2026-11-02, 91 still carry CBBD's midnight-ET placeholder (`startTimeTbd`) after the hoopR override (8 gained a real tip from the hoopR schedule, 19 have a CBBD tip). With the clock at 14:00Z on game day the guard refuses 94 of 118 (91 placeholder + 3 early real tips) and simulates 24;
   at an evening-before clock (2026-11-01 22:00Z) it refuses 0. Until real tips exist (hoopR / ESPN scoreboard refresh, `--tips hoopr` already merges them), run the sim stage the evening before game day (before midnight ET), or accept the refusals. The guard is never loosened.
2. Open lines exist in the 2025 file only from February; no January replay can test CLV. In live, publish-time = the current CBBD line (`line_kind: live`), close = the next-morning `close` snapshot.
3. Moneyline CLV is not testable in replay (no open moneyline in the file).

## 5. Choices recorded (nobody could be asked)

- Replay publishes the OPEN spread / total as "the line available at run time" and labels it; the harness match uses a separate close-line publication in `results/daily_replay_f2_close`.
- The earliest publication per game is the graded and CLV'd one.
- Spread and total price flat -110 (harness convention); ML settles at stored real odds; de-vig only for probability comparison.
- Monitor thresholds (|z| > 3, n >= 50; leak verdict needs >= 100 moved lines) are reporting thresholds, not model adjustments.

## 6. Still missing for day 1 (the stages block on these; nothing here is invented)

- **Real tip times** for the 91 placeholder games (finding 1).
- **Rosters**: CBBD 2027 rosters are empty (`/teams/roster`); candidates and positions need them.
- **2027 adapter artifacts and rule constants**: only `F2_2025` dated-refit sets exist (`event_round2_s1_F2_2025`, clock, rebound, FT, fg, rotation manifests) and `names_F2_2025_v2.json`; the sim prerequisites check lists `event_round2_s1_F2_2027`, `names_F2_2027_v2.json`, bonus-era 2027, dead-ball share, and-one, foul accrual.
- **Own ratings as of the slate date** (ratings stage blocked on `data/overrides/ratings_day1_choices.json`, six decisions) and the 2027 branch of `stage_inputs` (still `NotImplementedError` in v2).
- **Day-1 priors** (team form, rotation, usage, shooter families are degenerate with no prior-season carry) and the player crosswalk for 2027.
- **Lines**: none for 2027 yet. Re-probe 2026-10-26 and 2026-11-02 (readiness doc); `fetch_cbbd_lines` is the 1-call probe (`Draft Kings`, `Bovada`). Live CLV needs the chain's lines step to run every morning so the `close` snapshots exist.
- **Live finals path** is exercised by tests only (the replay used the harness truth): `finals_verified_{season}` does not exist until the first ingested game day. Per-stat truth comes from `team_game_shots_v2`, which the ingestion rebuilds.
- Seed count for live publishing (`--seeds`, chain default 200) is not set by a seed-count study; ROI numbers must not be read before that study (CLAUDE.md).
- Rebounds, turnovers and assists per-stat bias need a per-game truth table in the ledger join (none exists in this stage).
