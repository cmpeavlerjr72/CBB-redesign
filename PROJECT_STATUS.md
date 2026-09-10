# Project Status

As of 2026-09-10.

## What exists

This is the repo skeleton for the CBB clean-sheet rebuild — a possession-level
Monte Carlo simulation for NCAA men's basketball, built from scratch after the
2025-26 CBB-Monte engine's postmortem (see `docs/postmortem/`). Nothing beyond
scaffolding has been built yet: no data pull, no model, no engine.

As of this commit:

- Git repo initialized, remote `origin` set to
  `https://github.com/cmpeavlerjr72/CBB-redesign.git`.
- Python 3.12 venv at `.venv/`, package `cbb_sim` installed editable
  (`src/cbb_sim/{sim,analysis,clients}`), `pyproject.toml` modeled on
  `cfb-props-sim`.
- `scripts/` — flat, prefix-named (see `scripts/README.md` for the
  convention) with `hf_sync_data.py` ported from `cfb-props-sim` for the
  `data/raw` / `results` <-> HuggingFace mirror.
- Private HF dataset `mvpeav/cbb-sim-data` created for bulk-data sync.
- `docs/models/` doc-standard scaffolding (`DOCUMENTATION_STANDARD.md`
  copied verbatim from `cfb-props-sim`, empty `README.md` index, empty
  `change_ledger.md`).
- `docs/plans/TEMPLATE_fixplan.md` — the pre-registered fix-design template,
  ported from `cfb-props-sim` with all football-specific content stripped.
- `tests/` — one trivial smoke test, passing.
- `docs/FRAMEWORK_PLAN.md` and `docs/postmortem/` were authored by the PM
  before this skeleton pass and are untouched by it.

## Reading order

1. **`CLAUDE.md`** (to be written by the PM) — standing rules, reading order,
   git practice.
2. **`docs/FRAMEWORK_PLAN.md`** — target architecture, bake-off protocol,
   validation gates, data plan, timeline, and the repo-skeleton spec this
   setup pass implements (§6).
3. **`docs/postmortem/`** — why the 2025-26 engine lost (01-06), the evidence
   base for every standing rule in the framework plan.
4. **`docs/models/`** — per-model documentation as sub-models get built
   (`DOCUMENTATION_STANDARD.md` defines the required shape; `README.md` is
   the index; `change_ledger.md` tracks every investigated change).

## Not yet done

Everything in `docs/FRAMEWORK_PLAN.md` §7 timeline: hoopR ingest, ID
crosswalk, reference tables, leak-test harness, control engine, and every
sub-model in the cascade. See `HANDOFF.md` for the next session's starting
point.
