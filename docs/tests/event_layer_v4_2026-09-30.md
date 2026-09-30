# Event layer v4: phantom possessions removed at the handler that makes them (2026-09-30)

Lane B, event-layer job (PM ruling 2026-09-30: the phantom possessions found in
`docs/tests/g1_g5_possessions_corr_diagnostic_2026-09-30.md` section 5 are an
EVENT-LAYER defect). **A versioned sibling only.** `possessions`, `_v2` and
`_v3` are untouched, no consumer is switched, nothing is retrained, nothing is
adopted.

## 1. Design, written and committed before the code (commit `ef7a0cd`)

Three new switches on `cbb_sim.pbp.possessions._GameMachine` /
`segment_season`, **all default False**. With every switch off, the machine's
code path is the one that wrote `possessions_v2`, and it must reproduce it
bit-identically (section 3.1). v4 = `tech_lookahead=True` (the 09-18
technical-FT fix, v3) plus all three switches.

Class letters are those of the diagnostic (section 5.1 there).

| class | what creates it today | v4 rule (switch) | effect |
|---|---|---|---|
| **A** missed and-one FT -> defensive rebound | `_handle_fga`'s and-one branch closes the shooter's possession at the free throw (`reason_next = made_FG`) whether the FT was made or missed. The rebound of a MISS then finds no open possession, and `_handle_dreb` opens and immediately closes a 1-s possession for the shooter (`unknown`). | `andone_live_miss`. And-one FT **made**: close at the FT with `reason_next = made_FT`, which is what happens on the floor (a dead-ball inbound after a made FT). And-one FT **missed**: do NOT close. The chance stays open awaiting the rebound (`pending_terminal` = the made FGA's class), exactly as a missed last FT of a trip does. | phantom removed; the possession ends at the real rebound; the next possession starts `DREB` |
| **E** shooter's own OREB of a missed and-one FT | the same premature close: the OREB finds no open possession and opens a NEW possession for the shooter | fixed by the same switch: the OREB now continues the open possession as chance 2 (`start_reason = OREB`) | restart removed |
| after an and-one, start-type label | always `made_FG` | same switch: `made_FT` (FT made) / the rebound outcome (FT missed), per the floor evidence (diagnostic 5.3) | label follows the floor; this is the engine's convention |
| **B1** stray DREB row at a free-throw moment | `_handle_dreb` with no open possession opens and closes a possession for the non-rebounder | `stray_reb_guard`: a DREB that arrives with no open possession **within 3 s** of the last close (`prev_end_clock - secondsRemaining <= 3`) is treated as an administrative row and ignored. It changes no possession, no score, and no foul count. | phantom removed; its seconds stay with the real next possession |
| **B2** DREB with no open possession, elsewhere | same path | same switch, same 3-s rule. **<= 3 s** (a stray or out-of-order rebound row): ignored. **> 3 s**: unchanged. It stays a real possession whose missed shot the feed did not log (terminal `unknown`). | mixed class split at the rule's own threshold |
| **C1** possession opened by a stray OREB | `_handle_oreb` with no open possession calls `_ensure`, which opens a possession for the rebounder, although the rebounder's team just ended its own possession (made FG, made FT, TOV) | `stray_oreb_guard`: an OREB (not the administrative FT-reset kind, which is already skipped) arriving with no open possession, **by the team whose possession was the last one closed in this period**, is ignored | phantom removed; the other team's real possession keeps its seconds |
| **C2** mismatch close (offensive or loose-ball foul, unlogged turnover) | real possession whose terminal event is not in the feed | **unchanged** (terminal `unknown`) | real, kept |
| **D** contradictory OREB/DREB pair | real possession, terminal lost | **unchanged** | real, kept |

What stays **unresolved**:
- **B2 > 3 s.** Whether each of these is a real possession: rows read in
  2022-23 say mostly a missing shot, but the box slope swings by season.
- **B1 > 3 s** (10-30% of B1): kept as real, by the same threshold.
- **C2's terminal.** It is often an offensive foul the box counts as a
  turnover, but the layer cannot label it without the row.
- **The 3-s threshold.** It is the diagnostic's measured split (A and B1 are
  97-98% and 68-92% at 3 s or less; C2 is 95% longer), not a fitted value.
- **Out-of-order rows** where the rebound precedes its own shot at the same
  second: ignoring the rebound lets the later shot row be processed normally,
  but the shot's own rebound is then missing, so that possession closes on
  the next event.

Fouls, scores, points, technical free throws and every event count are
untouched by all three rules: an ignored row is a rebound row that credits
nothing. Section 3 checks it.
