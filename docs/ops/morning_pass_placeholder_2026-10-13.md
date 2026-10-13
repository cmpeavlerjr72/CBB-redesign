# Morning pass refuses placeholder tips: intended rule, not a bug (ops worker, 2026-10-13)

Verdict (a). No code changed.

## Evidence
- `src/cbb_sim/live/tips.py` `select_for_pass` L94-95: `if pass_name == "morning": reason[ph] = "placeholder tip time (not real); morning pass needs a real tip"`. Every placeholder row is refused in the morning pass regardless of clock. The refusal is by the placeholder flag, not by the 00:00 ET bound comparison.
- The clock/bound logic (L88-91, `effective_tip`) is correct and not involved: the evening pass at the same 2026-10-13 13:00 UTC clock keeps both rows.
- The counter name `would_refuse_already_tipped` (`scripts/chain_daily_v3.py` L108, also the `skipped.json` key `already_tipped`) is only a legacy label for "all refused rows"; the 91 are not judged tipped. Reason string in skipped.json is the placeholder one.
- Reproduction (placeholder 2026-11-02 05:00Z = 00:00 ET plus a real 23:00Z tip, `select_for_pass`):
  - evening, clock 2026-10-13 13:00 UTC: simulates [placeholder, real], refuses none.
  - evening, clock 2026-11-01 20:00 UTC: simulates both.
  - morning, clock 2026-10-13 13:00 UTC: simulates [real], refuses placeholder ("morning pass needs a real tip").
  - morning, clock 2026-11-01 20:00 UTC: same.
- Documented design: `docs/ops/day1_readiness_2027_2026-09-30.md` L27 and `2026-10-01.md` L80 ("the morning pass refuses placeholders ... and recovers every game once the table holds real tips"); `docs/ops/scheduler_registration_2026-10-09.md` L9, L28, L39.

## Why it makes sense
The morning pass is a same-day re-publish. A placeholder game at 09:00 ET on game day has an unknown tip, so created_at < tipoff cannot be proven; it is correct to refuse it. The 10-13 morning log is a 3-weeks-early rehearsal of the 11-02 rule, not a clock error (the clock is the real wall clock, as designed).

## Consequence for 2026-11-02
At the 09:00 ET morning pass, the 91 games whose tip is still a placeholder (as of the 2026-10-13 tip table: 123 placeholders of 173 in the window, 91 of 118 mapped) get no morning refresh unless real tips land in the table first. Their evening-pass (11-01 20:00 ET) publication stands, flagged `pre_tip_verified = False` / placeholder basis; the grader keeps the earliest publication anyway. Games whose real tip is published by morning are refreshed. Real-tip counts normally rise toward game day, so the morning set should grow beyond 27.
