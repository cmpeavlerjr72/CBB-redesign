# Handoff Log

Append one `## SESSION {date}` block per work session, newest at the bottom.
See `PROJECT_STATUS.md` for current overall state and reading order.

---

## SESSION 2026-09-10

**Who:** repo-skeleton setup pass.

**Did:**
- Initialized git (`main`, remote `origin` ->
  `github.com/cmpeavlerjr72/CBB-redesign.git`), `.gitignore` modeled on
  `cfb-props-sim`.
- Created `.env` (HF_TOKEN only, not committed) and `.env.example`.
- Built `.venv` (Python 3.12), installed the sim/dev dependency set, wrote
  `pyproject.toml` (package `cbb_sim`, src layout), `pip install -e .`.
- Laid out `src/cbb_sim/{sim,analysis,clients}`, `scripts/` (+ README with
  the prefix convention), `tests/` (smoke test), `data/{processed,reference}`
  (tracked), `data/raw` and `results` (gitignored, HF-synced).
- Ported `scripts/hf_sync_data.py` from `cfb-props-sim` to
  `mvpeav/cbb-sim-data` (single wave, `data/raw` + `results`). Created the
  private HF dataset repo and verified connectivity with `status`.
- Wrote `docs/models/` scaffolding (`DOCUMENTATION_STANDARD.md` copied
  verbatim, empty `README.md` index, empty `change_ledger.md`) and
  `docs/plans/TEMPLATE_fixplan.md` (football content stripped).
- Did NOT touch `docs/FRAMEWORK_PLAN.md` or `docs/postmortem/` (PM-authored,
  pre-existing). Did NOT write `CLAUDE.md` or `ARCHITECTURE_DECISIONS.md`
  (PM's to write).
- Left `data/raw/hoopr/**` alone — a concurrent process was downloading
  hoopR data into it during this session; nothing under `data/raw` was
  committed.

**Next session should:**
- PM writes `CLAUDE.md` and `ARCHITECTURE_DECISIONS.md`.
- Start `docs/FRAMEWORK_PLAN.md` §7 Week 1: hoopR ingest 2021-2026, ID
  crosswalk, data audit, empirical reference tables, leak-test harness,
  Control engine.
- Once `data/raw/hoopr` finishes populating, run
  `.venv/Scripts/python.exe scripts/hf_sync_data.py push` to mirror it to
  `mvpeav/cbb-sim-data`.

**Open questions / blockers:** none from this pass. Decisions needed before
week 1 proper starts are listed in `docs/FRAMEWORK_PLAN.md` §8.
