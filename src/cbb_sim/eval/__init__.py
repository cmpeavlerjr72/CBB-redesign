"""cbb_sim.eval -- the engine-agnostic evaluation harness.

Any sim engine (Control, or any later candidate) is graded by the SAME code
here, provided it writes its output in the shape `contract.py` defines. No
module in this package imports from `cbb_sim.control`, `cbb_sim.models`, or
`cbb_sim.pbp` -- those are engine internals; this package only ever reads
`results/<engine_tag>/{games,players}.parquet` + `run_meta.json` plus the
season-level truth tables under `data/`.

    contract.py    the results schema + validator (deliverable 1)
    reference.py   truth-table loaders (games_universe, gate_targets,
                   team-game box, CBBD lines) -- season-level, engine-free
    gates.py       G1-G9 (SIM_GUARDRAILS.md section 3)
    market.py      G10 + the settle/de-vig/bootstrap machinery shared by
                   scripts/grade_market_games.py and scripts/grade_market_props.py
"""
