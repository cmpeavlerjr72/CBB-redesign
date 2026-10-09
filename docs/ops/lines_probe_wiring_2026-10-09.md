# Lines-source probe wired into chain_daily_v3 (2026-10-09)

Stage `lines_probe` in `scripts/chain_daily_v3.py`, calling `scripts/probe_lines_sources_v1.py` (always exits 0; writes `data/processed/lines/probe_lines_sources_v1_<date>.json`).

- **When:** evening pass only (`--pass evening`, the default). The morning pass reports `skipped`. Position: right after the existing `lines` and `lines_snapshot` steps, before `ingest`.
- **Default on.** `--no-lines-probe` skips it. The earlier date gate ("starts Oct 20") was removed so the stage runs by default.
- **Window:** `--start <slate date> --end <slate date + 7 days>` (the script's own default span, anchored on the slate instead of today, so an October run probes the Nov 2 slate rather than an empty week).
- **Never fatal:** a non-zero rc, timeout or exception returns status `failed_nonfatal`; `exit_code` counts only `error`, and the live loop only stops on `error`. It feeds no sim, publish or grade number.
- **Dry run:** passes the script's `--dry-run` (prints the plan, no network). When live it makes 3 CBBD calls (/games, /lines, /lines/providers) that are NOT counted in the chain's `cbbd_calls_chain` tracker (the script uses its own client), plus ESPN scoreboard x8 and summary x<=15.
- **Proof (dry run, 2026-10-09, `chain_daily_v3.py --dry-run --dry-run-sim`):** `[OK] lines_probe (0.1s) rc 0, range 2026-11-02..2026-11-09`; sim stage OK (`would_simulate` 118); zero files under `results/daily` modified during the run (the dry run never writes the served output, so the cached 2026-11-02 sim is untouched). Tests: `tests/test_chain_lines_probe.py` (4).
