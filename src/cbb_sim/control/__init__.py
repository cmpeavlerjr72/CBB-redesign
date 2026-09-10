"""
control -- the Control engine: last year's counts x percentages decomposition,
done correctly, as the yardstick the possession engine must beat.

Spec: `docs/models/control_engine/model.md` (pre-registered 2026-09-10).
It is NOT a bake-off candidate; its components are fixed by that spec.

Modules
-------
features   per team-game pregame feature table (own ridge ratings as-of,
           centred KenPom as-of, home/away/neutral, days since season start,
           and the opponent's same) + the three anchor feature bundles.
models     the component models: pace (Gaussian GLM, fitted residual SD),
           per-100-possession count GLMs (Poisson, or NegBin where the
           training deviance/df exceeds 1.2), and trials-weighted Binomial
           GLMs with fitted Beta-Binomial overdispersion.
rng        counter-based RNG keyed on (seed, game_id, family) so a game's
           stream never depends on which other games are in the run.
simulate   the vectorised simulator: one shared possession draw per game per
           seed, attempts scaled by it, Beta-Binomial makes, points, and the
           explicit overtime stub.
"""

from __future__ import annotations

__all__ = ["features", "models", "rng", "simulate"]
