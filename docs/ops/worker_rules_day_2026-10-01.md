# Worker rules, day session 2026-10-01 06:15 EDT -> 17:00 EDT (hard stop)

PM-owned. Every lane brief points here. `docs/ops/worker_rules_overnight_2026-09-30.md` binds in full (compute hygiene, foundations, standing rules, git on the shared tree, reporting) EXCEPT the sections replaced below.

## Clock (replaces the overnight clock section)

Real wall clock only: `Get-Date -Format "yyyy-MM-dd HH:mm:ss K"` at the start and before / after every compute step, logged. Size runs from a measured slice of at least 3 minutes, never a guess.

- 15:15 EDT: launch no new compute that the measured timing says will not finish by 15:45.
- 15:45 EDT: stop computing. Kill only your own PIDs. Unfinished work is marked NOT RUN or PARTIAL with the resume command.
- 16:15 EDT: your final report is back with the PM, committed and pushed. A late report is worth less than a partial one on time.

A lane that finishes early reports early; it does not invent extra scope.

## Served stack (new since the overnight rules were written)

The engine defaults are SERVED STACK v2 (adopted 02:06 EDT 2026-10-01): `ENGINE_CLOCK=v5b_r6L2_glat_pmean`, `ENGINE_SHOT_BLOCK=K2_Ocell`, `ENGINE_FOUL_JOINT=R9ao3`, `ENGINE_SHARED_SHOOTING=G3`, `ENGINE_CHANCE_TIME=KD`, event team block v3. The reference for every paired read today is a plain default run (parity reference v9, `docs/ops/parity_reference_windows_v9.json`); the old stack is `adapters.SERVED_V1`. Evidence: `docs/tests/adoption_served_v2_2026-10-01.md`. Truth default is verified (still set `CBB_TRUTH=verified_v1` explicitly).

You adopt nothing and change no served default. New behaviour goes behind a default-off flag whose off path is bit-identical to parity v9 (prove it).

## Compute (replaces the AWS and box-queue bullets)

- Home box, 20 cores, shared by seven lanes; your cap is in your brief.
- AWS is NOT approved for this session unless the PM tells you so. No lane touches AWS. Write box requests exactly as in the overnight rules (`docs/ops/box_queue/<lane>_<n>.md`, code committed and pushed first, not committed themselves); name them with today's lane letter. If the box is approved an operator lane will run them; if not, they are the first jobs of the next box session. Do not wait on a request: finish your offline work and local taps, and report the request as PENDING.
- Local closed-loop taps (for example 500 x 25) are for direction and parity only. Decision 12 applies: they cannot decide a G5 ratio line or any line whose floor needs the full-size read; say "underpowered".

## Known hazards from last night (do not repeat)

- `docs/models/change_ledger.md` is CRLF. Do not let your editor convert it; check `git diff --stat` shows only your rows before committing.
- A shared tree means another lane's uncommitted engine edit can be live in your run. Before any closed-loop run, `git status --short src/` and `git diff --stat src/`; if a file you did not edit is dirty, record it in the run log and either wait or say the run is on a dirty tree.
- Train/serve skew: any retrain must go through the parity stage of `scripts/chain_full_retrain_v1.py` or an equivalent check that the trained feature equals the served feature.
- `laneN_R_s25_o*` runs from 09-30 are not valid floor draws.
