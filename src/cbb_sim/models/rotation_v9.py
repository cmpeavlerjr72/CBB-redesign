"""
rotation_v9.py -- ROUND 9 of the L4 rotation bake-off: the exit hierarchy
INVERTED onto the powered composition marginal.

Why round 9 exists
------------------
Round 8 conditioned the exit count on the composition, `P(k_out | size, state,
n_st)`, and shrank it to X1's row -- the LEVEL `P(k_out | size, state)`. The
shrinkage parent, not the axis, is the binding constraint (`experiments.md`
19.1, 19.13 item 3):

    at n_st = 1 the 2,764 training rows split over 18 exit cells leave ~154 per
    cell, so at k = 300 the data carry 154 / (154 + 300) = 0.34 and the fitted
    rate is 0.34 x 0.37 + 0.66 x 0.5576 = 0.49 against a real 0.37.

Y1 therefore recovers only 46% of the real 36.5 pp composition span (19.7). The
composition MARGINAL `P(k_out | size, n_st)` is amply powered -- 379 / 2,764 /
12,646 / 28,762 / 28,011 / 9,582 rows at n_st = 0..5, a 29 pp span, no state
split at all -- so round 9 turns the hierarchy the other way up:

    level 0   M0(s)          = P(k_out | size)                  (pooled root)
    level 1   M1(s, n_st)    = P(k_out | size, n_st)   -> M0     (THE MARGINAL)
    level 2   Z(s, ce, n_st) = P(k_out | size, ce, n_st) -> M1   (the state cell)

A thin (state x composition) cell now falls back on the composition rate, which
has a fixed point at three starters on the floor, instead of on the level, which
has none and drifts -0.071 per swap.

    Z1  `exit_marg`      level 2, continuous shrinkage at the fitted k
    Z2  `exit_interact`  level 2 where the cell carries >= 300 rows, M1 exactly
                         otherwise (a hard gate on the interaction)

Pre-registration: `docs/models/rotation/experiments.md` section 20, written and
committed (3525996) before this file existed.

What is NOT changed
-------------------
The counts are round 8's own (`rotation_v8.build_exit8_training`, byte for
byte), and the sampler IS round 8's (`rotation_v8.run_wave8`), called through a
table shim so the uniform order -- round 5's coupling draw, the wave draw, the
size draw, the `k_out` uniform, K1's `k_in` uniform, the bench race vector -- is
identical and the round-9 arms are byte-aligned with X1 and Y1 (20.13 item 1).
Z2's gate is applied at FIT time, so the sim-side gather is Z1's exactly and the
engine cost is Y1's (20.12).

The shrinkage constant is FITTED, not chosen by hand: leave-one-fold-out
multinomial log-likelihood on the 2024 training season over the declared grid
{30, 100, 300, 1000, 3000} (20.4). It never sees a gate cell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cbb_sim.models.rotation import GameScript, RotationFit, TeamPrior, draw_available
from cbb_sim.models.rotation_v4 import _side_arrays
from cbb_sim.models.rotation_v5 import MAX_WAVE, WaveFit, _WaveArm
from cbb_sim.models.rotation_v6 import CompFit
from cbb_sim.models.rotation_v7 import K_SHRINK, N_EXIT_CELL
from cbb_sim.models.rotation_v8 import N_ST, ExitFit8, _shrink, build_exit8_training
from cbb_sim.models import rotation_v8 as V8

#: the UNDERPOWERED threshold Z2's interaction gate uses -- the project's own
#: since round 5, stated once and not tuned here (20.3).
MIN_CELL = 300

#: the declared grid the shrinkage constant is fitted over (20.4). Not extended
#: after any number is seen.
K_GRID = (30.0, 100.0, 300.0, 1000.0, 3000.0)

#: ties within this many nats per held-out row go to the LARGER k (20.4).
K_TIE_NATS = 0.001


# ===========================================================================
# 1. The fitted object
# ===========================================================================
@dataclass
class ExitFit9:
    """The two round-9 exit tables plus the two parents they are built from.
    One artifact carries all four so the arms are fitted on identical rows."""

    p_m0: list = field(default_factory=list)   # (MAX_WAVE, MW+1)
    p_m1: list = field(default_factory=list)   # (MAX_WAVE, N_ST, MW+1)
    p_z1: list = field(default_factory=list)   # (MAX_WAVE, N_EXIT_CELL, N_ST, MW+1)
    p_z2: list = field(default_factory=list)   # (MAX_WAVE, N_EXIT_CELL, N_ST, MW+1)
    n_cell: list = field(default_factory=list)  # (MAX_WAVE, N_EXIT_CELL, N_ST)
    n_marg: list = field(default_factory=list)  # (MAX_WAVE, N_ST)
    k_shrink: float = K_SHRINK
    k_selected_by: str = ""
    k_grid_scores: dict = field(default_factory=dict)
    n_waves: int = 0
    n_pairs: int = 0
    wave_source: str = ""
    comp_source: str = ""
    hazard_source: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def M1(self) -> np.ndarray:
        return np.asarray(self.p_m1, dtype=np.float64).reshape(
            MAX_WAVE, N_ST, MAX_WAVE + 1)

    @property
    def Z1(self) -> np.ndarray:
        return np.asarray(self.p_z1, dtype=np.float64).reshape(
            MAX_WAVE, N_EXIT_CELL, N_ST, MAX_WAVE + 1)

    @property
    def Z2(self) -> np.ndarray:
        return np.asarray(self.p_z2, dtype=np.float64).reshape(
            MAX_WAVE, N_EXIT_CELL, N_ST, MAX_WAVE + 1)

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "ExitFit9":
        return ExitFit9(**json.loads(Path(path).read_text(encoding="utf-8")))


# ===========================================================================
# 2. The fit -- the hierarchy, inverted
# ===========================================================================
def _levels(c1: np.ndarray, cy1: np.ndarray, k: float):
    """`(M0, M1, Z1, Z2)` from round 8's own counts at a shrinkage constant.

    `_shrink` is round 7's, byte for byte (one-level shrinkage of a `(k_out,)`
    count row toward a parent at `k`, support clipped to `0..size`,
    renormalised), so the only difference from `fit_exit8` is WHICH row is the
    parent of the state cell.
    """
    m0 = np.zeros((MAX_WAVE, MAX_WAVE + 1))
    m1 = np.zeros((MAX_WAVE, N_ST, MAX_WAVE + 1))
    z1 = np.zeros((MAX_WAVE, N_EXIT_CELL, N_ST, MAX_WAVE + 1))
    z2 = np.zeros_like(z1)
    cmarg = cy1.sum(axis=1)                      # (MAX_WAVE, N_ST, MW+1)
    for s in range(MAX_WAVE):
        tot = c1[s].sum()
        root = c1[s].sum(axis=0)
        root = root / tot if tot > 0 else np.eye(MAX_WAVE + 1)[0]
        # level 0: the pooled size row, clipped to the support like every level
        m0[s] = _shrink(np.zeros(MAX_WAVE + 1), root, 1.0, s + 1)
        for ns in range(N_ST):
            # level 1: THE COMPOSITION MARGINAL, pooled over all exit cells
            m1[s, ns] = _shrink(cmarg[s, ns], m0[s], k, s + 1)
        for ce in range(N_EXIT_CELL):
            for ns in range(N_ST):
                # level 2: the state cell, shrunk to the MARGINAL (20.2)
                z1[s, ce, ns] = _shrink(cy1[s, ce, ns], m1[s, ns], k, s + 1)
                # Z2: the interaction only where the cell is powered (20.3)
                z2[s, ce, ns] = (z1[s, ce, ns] if cy1[s, ce, ns].sum() >= MIN_CELL
                                 else m1[s, ns])
    return m0, m1, z1, z2


def heldout_loglik(counts_fit: dict, counts_out: dict, k: float) -> tuple:
    """Multinomial log-likelihood of the held-out fold's `(size, cell, n_st)`
    counts under the level-2 table fitted on the remaining folds (20.4).

    Returns `(total nats, held-out rows)`. Zero-probability support cells cannot
    be reached by a held-out row (`k_out <= size` at every level by
    construction); the 1e-12 floor is a numerical guard, not a model choice."""
    _, _, z1, _ = _levels(counts_fit["c1"], counts_fit["cy1"], k)
    co = counts_out["cy1"]
    ll = float(np.sum(np.where(co > 0, co * np.log(np.maximum(z1, 1e-12)), 0.0)))
    return ll, float(co.sum())


def select_k(fold_counts: list, grid=K_GRID) -> dict:
    """Leave-one-fold-out selection of the shrinkage constant on TRAINING data
    only (20.4). No gate cell, no test row and no MAE is visible here."""
    scores = {}
    for k in grid:
        tot_ll = tot_n = 0.0
        for i in range(len(fold_counts)):
            fit = {"c1": sum(fc["c1"] for j, fc in enumerate(fold_counts) if j != i),
                   "cy1": sum(fc["cy1"] for j, fc in enumerate(fold_counts) if j != i)}
            ll, n = heldout_loglik(fit, fold_counts[i], k)
            tot_ll += ll
            tot_n += n
        scores[str(k)] = float(tot_ll / tot_n) if tot_n else float("-inf")
    best = max(scores.values())
    # argmax; ties within K_TIE_NATS per held-out row go to the LARGER k
    near = [float(kk) for kk, v in scores.items() if best - v <= K_TIE_NATS]
    return {"k": float(max(near)), "scores": scores, "n_folds": len(fold_counts),
            "grid": [float(g) for g in grid], "tie_nats": K_TIE_NATS,
            "criterion": "leave-one-fold-out multinomial log-likelihood per "
                         "held-out row, 2024 training season only"}


def fit_exit9(counts: dict, k: float = K_SHRINK, wave_source: str = "",
              comp_source: str = "", hazard_source: str = "",
              k_selected_by: str = "", k_grid_scores: dict | None = None) -> ExitFit9:
    """The two objects and their two parents, from round 8's own counts and a
    shrinkage constant fitted by `select_k`. No optimiser and no grid search
    against any gate cell."""
    c1, cy1 = counts["c1"], counts["cy1"]
    m0, m1, z1, z2 = _levels(c1, cy1, k)
    return ExitFit9(p_m0=m0.tolist(), p_m1=m1.tolist(), p_z1=z1.tolist(),
                    p_z2=z2.tolist(), n_cell=cy1.sum(axis=3).tolist(),
                    n_marg=cy1.sum(axis=(1, 3)).tolist(), k_shrink=float(k),
                    k_selected_by=k_selected_by,
                    k_grid_scores=dict(k_grid_scores or {}),
                    n_waves=int(counts["n_waves"]), n_pairs=int(counts["n_used"]),
                    wave_source=wave_source, comp_source=comp_source,
                    hazard_source=hazard_source,
                    notes={"team_games": int(counts["team_games"])})


# ===========================================================================
# 3. The sampler -- round 8's `run_wave8`, byte for byte, with a table shim
# ===========================================================================
class _TableShim:
    """Presents a round-9 table where `run_wave8` expects round 8's `Y1`, so the
    mechanism, the uniform order and every clip are round 8's exactly and the
    ONLY difference between a round-9 arm and Y1 is the fitted table (20.1)."""

    __slots__ = ("Y1", "Y2")

    def __init__(self, tab: np.ndarray):
        self.Y1 = tab
        self.Y2 = tab          # never read: round 9 always gathers in mode "Y1"


def run_wave9(prior: TeamPrior, script: GameScript, avail: np.ndarray, wf: WaveFit,
              cf: CompFit, zf: ExitFit9, fit: RotationFit, rng: np.random.Generator,
              mode: str = "Z1", hard_reset: bool = True,
              prev_end: np.ndarray | None = None,
              team_fouls: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """One team-game's on-floor sequence. Delegates to `rotation_v8.run_wave8`
    with the round-9 table, in round 8's own `Y1` gather position."""
    tab = zf.Z2 if mode == "Z2" else zf.Z1
    return V8.run_wave8(prior, script, avail, wf, cf, _TableShim(tab), fit, rng,
                        mode="Y1", hard_reset=hard_reset, prev_end=prev_end,
                        team_fouls=team_fouls)


# ===========================================================================
# 4. Arms
# ===========================================================================
class _ExitArm9(_WaveArm):
    mode = "Z1"

    def __init__(self, fit: RotationFit, wave: WaveFit, comp: CompFit, exit_: ExitFit9,
                 side_state: dict | None = None):
        super().__init__(fit, wave, side_state)
        self.comp = comp
        self.exit = exit_

    def simulate(self, prior: TeamPrior, script: GameScript,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        avail = draw_available(prior, rng)
        pe, tf = _side_arrays(self.side_state, (prior.game_id, prior.team_id),
                              script, script.n)
        return run_wave9(prior, script, avail, self.wave, self.comp, self.exit,
                         self.fit, rng, mode=self.mode, hard_reset=self.hard_reset,
                         prev_end=pe, team_fouls=tf)


class Z1ExitMarg(_ExitArm9):
    """the state cell shrunk to the POWERED composition marginal"""

    name = "Z1_exit_marg"
    simplicity_rank = 16
    mode = "Z1"


class Z2ExitInteract(_ExitArm9):
    """Z1 with the (state x composition) interaction gated at 300 rows"""

    name = "Z2_exit_interact"
    simplicity_rank = 17
    mode = "Z2"


ARMS = {"Z1_exit_marg": Z1ExitMarg, "Z2_exit_interact": Z2ExitInteract}

__all__ = ["ARMS", "ExitFit8", "ExitFit9", "K_GRID", "MIN_CELL", "N_ST",
           "Z1ExitMarg", "Z2ExitInteract", "build_exit8_training", "fit_exit9",
           "heldout_loglik", "run_wave9", "select_k"]
