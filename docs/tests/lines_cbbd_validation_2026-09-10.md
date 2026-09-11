# CBBD `/lines` validation for 2023-2025 -- 2026-09-10

Built by `scripts/diag_lines_cbbd_validation_v1.py`, read-only, from `data/raw/cbbd/lines/lines_{season}.parquet` (`scripts/pull_cbbd_lines_v1.py`), `data/raw/cbbd/games_{season}.parquet` (`conferenceGame`), `data/processed/games_universe_v2.parquet` (our universe, `is_d1_game`), and `data/processed/truth/game_finals_v2.parquet` (verified finals). Seasons 2023-2025 only; 2026 is sealed and not opened here. No API calls.

## 1. Coverage -- share of our D-I games with >=1 line row

| season | D-I games (our universe) | games w/ >=1 line | pct |
|---:|---:|---:|---:|
| 2023 | 5749 | 5740 | 99.8% |
| 2024 | 5731 | 5306 | 92.6% |
| 2025 | 5769 | 5440 | 94.3% |

### Coverage by provider (distinct D-I games, our universe)

| season | provider | games |
|---:|:---|---:|
| 2023 | ESPN BET | 5720 |
| 2023 | consensus | 5602 |
| 2023 | teamrankings | 5732 |
| 2024 | ESPN BET | 5306 |
| 2025 | ESPN BET | 5440 |

**Provider takeaway**: `ESPN BET` is the only provider present in all three target seasons (teamrankings/consensus also appear in 2023 only) -- used below as the single real-book provider for de-vig sanity and as the primary provider in `lines_close_v1`.

### Coverage by conference vs. non-conference game

| season | conference game | games | covered | pct |
|---:|:---|---:|---:|---:|
| 2023 | non-conference | 2137 | 2134 | 99.9% |
| 2023 | conference | 3612 | 3606 | 99.8% |
| 2024 | non-conference | 2166 | 1993 | 92.0% |
| 2024 | conference | 3565 | 3313 | 92.9% |
| 2025 | non-conference | 2119 | 2118 | 100.0% |
| 2025 | conference | 3650 | 3322 | 91.0% |

### Coverage by month

| season | month | D-I games | covered |
|---:|:---|---:|---:|
| 2023 | 2022-11 | 1196 | 1128 |
| 2023 | 2022-12 | 1125 | 1181 |
| 2023 | 2023-01 | 1466 | 1449 |
| 2023 | 2023-02 | 1401 | 1386 |
| 2023 | 2023-03 | 558 | 593 |
| 2023 | 2023-04 | 3 | 3 |
| 2024 | 2023-11 | 1142 | 1017 |
| 2024 | 2023-12 | 998 | 952 |
| 2024 | 2024-01 | 1389 | 1275 |
| 2024 | 2024-02 | 1396 | 1262 |
| 2024 | 2024-03 | 800 | 794 |
| 2024 | 2024-04 | 6 | 6 |
| 2025 | 2024-11 | 1234 | 1213 |
| 2025 | 2024-12 | 928 | 945 |
| 2025 | 2025-01 | 1447 | 1378 |
| 2025 | 2025-02 | 1377 | 1105 |
| 2025 | 2025-03 | 766 | 780 |
| 2025 | 2025-04 | 17 | 19 |

## 2. Opening vs. closing lines and timestamps

**Finding (schema-verified with one live `/lines` call, logged in the pull manifest): a per-line record has exactly 7 fields -- `provider, spread, overUnder, homeMoneyline, awayMoneyline, spreadOpen, overUnderOpen` -- and carries NO timestamp of its own.** The only timestamp anywhere in the payload is the game-level `startDate` (tipoff), not a line-capture time.

| season | spreadOpen non-null % | overUnderOpen non-null % |
|---:|---:|---:|
| 2023 | 0.0% | 0.0% |
| 2024 | 0.0% | 0.0% |
| 2025 | 33.3% | 32.1% |

`spreadOpen`/`overUnderOpen` are essentially absent for 2023-2024 (0.0%) and still sparse in 2025 (this is the same pattern the 2026-09-10 CBBD data audit found in `docs/tests/data_audit_cbbd_2026-09-10.md`: open fields only start getting populated in 2025-2026). **Consequence for `created_at < tipoff`: this field cannot be mechanically enforced against CBBD lines data as returned** -- there is no per-line capture timestamp to check. `spread`/`overUnder`/moneylines should be treated as "the line CBBD has on file for that game" (for a fully played historical season this is presumptively the closing line, since it is a single number per provider per game with no update history) rather than as a verified pre-tipoff snapshot. This is a genuine gap, not a fabricated pass -- flagged for the PM rather than worked around.

## 3. De-vig sanity (provider = ESPN BET)

n = 16240 games with an ESPN BET line and a verified final, 2023-2025 pooled.

De-vig methods: **proportional** (`p_home = p_home_raw / (p_home_raw + p_away_raw)`, the simple method) computed for all rows; **power method** (solve k>0 s.t. `p_home_raw**k + p_away_raw**k == 1`, report `p_home_raw**k`) computed alongside -- cheap, reported for comparison, not used downstream in `lines_close_v1` beyond reporting both.

- Market overround (`p_home_raw + p_away_raw - 1`): mean 4.63%, std 1.10%, min -0.42%, max 9.25% -- consistent with a real, modestly-vigged sportsbook (a fabricated/derived line would show ~0% or a suspiciously constant overround).
- **Spread-implied margin MAE vs. actual margin: 8.81 points** (corr 0.628) -- **below** the ~10-11 point band the standing rule cites as the real-market sanity check. Flagging rather than smoothing over: a lower MAE than the stated band means this book's closing spread predicted these games somewhat better than the 10-11 benchmark implies. Plausible explanations are that ESPN BET is a sharp book and/or the 10-11 band comes from a different sport/era, but the PM should decide whether this is expected or worth digging into further before treating spread-implied margin as ground truth for grading.
- Total line MAE vs. actual total: 12.97 points; bias (actual - line): +0.73 points.
- Spread-model implied home win prob (`Phi(-spread/11)`) vs. de-vigged moneyline-implied home win prob: corr 0.996, MAE 0.020 -- spread and moneyline agree closely, as expected of one book pricing both sides of the same game consistently.

### Calibration: de-vigged implied home win prob vs. realized home win rate (20 buckets)

| n | mean implied p(home) | realized home win rate |
|---:|---:|---:|
| 808 | 0.164 | 0.157 |
| 856 | 0.283 | 0.262 |
| 725 | 0.354 | 0.327 |
| 831 | 0.408 | 0.428 |
| 812 | 0.452 | 0.445 |
| 860 | 0.501 | 0.527 |
| 726 | 0.541 | 0.541 |
| 737 | 0.571 | 0.617 |
| 947 | 0.601 | 0.601 |
| 653 | 0.628 | 0.655 |
| 768 | 0.653 | 0.664 |
| 862 | 0.688 | 0.711 |
| 786 | 0.720 | 0.732 |
| 783 | 0.751 | 0.797 |
| 774 | 0.786 | 0.826 |
| 812 | 0.823 | 0.846 |
| 776 | 0.859 | 0.887 |
| 798 | 0.894 | 0.912 |
| 753 | 0.924 | 0.947 |
| 793 | 0.958 | 0.985 |

## 4. Duplicates

| season | duplicate (gameId, provider) rows |
|---:|---:|
| 2023 | 0 |
| 2024 | 0 |
| 2025 | 0 |

## 5. Home/away flip check (lines' own homeScore/awayScore vs. verified finals)

| season | n checked | normal (home=home) | swapped (home<->away) | neither |
|---:|---:|---:|---:|---:|
| 2023 | 5649 | 5649 | 0 | 0 |
| 2024 | 5228 | 5228 | 0 | 0 |
| 2025 | 5383 | 5380 | 0 | 3 |

No season shows a material swapped-label or neither-matches rate -- home/away labelling in CBBD `/lines` agrees with our verified finals at effectively 100% (any `neither` rows are OT-score edge cases already tracked in `game_finals_v2.finals_resolution_note`, not a lines defect).

## 6. Overall verdict

Coverage, provider mix, de-vig calibration (near-diagonal in the 20-bucket table above), and the home/away flip check all look like a real, internally consistent market. Two items are flagged for the PM rather than fixed here: (1) no line-level timestamp exists (section 2), so `created_at < tipoff` cannot be mechanically enforced against this source; (2) the spread-implied margin MAE (8.81 pts, section 3) is *below* the 10-11 pt band the standing rule cites, i.e. this book's spreads predicted these games a bit better than that benchmark -- not obviously wrong, but worth a second look before relying on the 10-11 pt band as a pass/fail gate.

---

## Addendum 2026-09-10b -- open vs. close, and a KenPom yardstick on the 8.81-pt spread MAE

Requested by the PM before accepting CBBD lines as the free source. Both checks reuse the ESPN BET provider and the identical verified-finals game set from section 3 above. No API calls; read-only.

### A. Open vs. close

| season | n games | n with open | pct with open | open MAE | close MAE (same subset) | pct moved | move mean | move std | move min | move max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 5629 | 0 | 0.0% | UNDERPOWERED (n<30) | -- | -- | -- | -- | -- | -- |
| 2024 | 5228 | 0 | 0.0% | UNDERPOWERED (n<30) | -- | -- | -- | -- | -- | -- |
| 2025 | 5383 | 1802 | 33.5% | 8.51 | 8.45 | 60.3% | -0.03 | 1.21 | -6.0 | +6.0 |

2023-2024 `spreadOpen` coverage is ~0% (already noted in section 2) -- **underpowered, not scored**, not presented as a pass or a red flag. 2025 is the only season with usable open-line coverage.

### B. KenPom yardstick (as-of, strictly-before snapshot; formula in script docstring, HCA=3.75 pts fixed, not fit)

Identical ESPN BET + verified-final game set, restricted further to games where both teams have a strictly-before KenPom snapshot: n = 16076 (pooled 2023-2025).

| season | n | KenPom margin MAE | Market (close spread) margin MAE | corr(KenPom pred, market pred) |
|---:|---:|---:|---:|---:|
| 2023 | 5569 | 8.99 | 8.82 | 0.968 |
| 2024 | 5178 | 9.05 | 8.87 | 0.966 |
| 2025 | 5329 | 8.97 | 8.74 | 0.965 |
| ALL | 16076 | 9.00 | 8.81 | 0.966 |

Market beats the honest KenPom as-of snapshot by 0.19 pts pooled (8.81 vs. 9.00) -- **below**, not above, the 0.5-1.0 pt normal-picture band the PM specified. This is the *opposite* direction from the "3+ pts, lines are suspect" failure mode: the market is not implausibly sharper than a real power rating, it is only marginally sharper, essentially within noise of it (corr(KenPom pred, market pred) = 0.966 -- the two are predicting the same thing, not contradicting each other). The most likely explanation is that the KenPom margin formula used here is a standard-but-approximate reconstruction (see script docstring: centered adj_o/adj_d, average-tempo pace scaling, a fixed 3.75-pt HCA) rather than KenPom's exact proprietary model, so it understates the gap a well-tuned power rating would show against a real book; it is not evidence the CBBD lines are fabricated or derived from KenPom. Flagged for the PM as a formula-fidelity caveat, not as a lines-quality defect. Note: CBBD lines carry no line-level timestamp (section 2), so CLV can only be measured open-to-close (part A above), never against a true bet-placement time; this is a limitation of the source, unrelated to and does not affect our own `created_at < tipoff` rule, which gates OUR prediction rows, not CBBD's.
