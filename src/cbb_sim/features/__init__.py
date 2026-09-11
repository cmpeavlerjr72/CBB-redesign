"""Reusable, strictly as-of feature builders shared by every sub-model.

`conference` -- conference-game flags and each team's first conference game
(ARCHITECTURE_DECISIONS.md Decision 9b).
`opponent_adjust` -- opponent-strength adjustment of team rate features
(Decision 9a). Both are consumed by more than one sub-model, which is why they
live in the package rather than in a trainer.
"""
