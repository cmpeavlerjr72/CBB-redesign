"""
engine -- ENGINE v0, the possession-level simulator (Decision 1, Decision 7).

Spec and provisional-adapter inventory: `docs/models/engine/model.md`.
Resume state: `docs/models/engine/RESUME.md`.

Modules
-------
state             GameState as a struct of arrays over N concurrent simulations;
                  the rule era (bonus thresholds) lives here, not in a sub-model.
rng               the (seed, game_id, family, ordinal) stream discipline, built
                  on `cbb_sim.control.rng` and bit-identical to it for a
                  constant ordinal.
inputs            the per-game lookup tables and the static/state feature splice.
adapters          one batched, vectorised adapter per sub-model, each declaring
                  whether it is provisional.
rotation_adapter  `rotation.RotationSampler.next_lineup`'s decision rule,
                  vectorised over both teams of every simulation.
loop              the possession loop: clock, event, allocation, outcome,
                  rotation, period/overtime, bookkeeping.
"""

from __future__ import annotations

__all__ = ["adapters", "inputs", "loop", "rng", "rotation_adapter", "state"]
