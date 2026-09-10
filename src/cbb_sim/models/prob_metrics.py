"""
prob_metrics.py -- the scoring battery the REBOUND and FREE-THROW bake-offs are
graded with.

These are the L3 round-1 metric definitions (`cbb_sim.models.possession_outcome`
section 4), generalised from that module's fixed six-class vocabulary to an
arbitrary class list so that a three-class rebound target and a two-class
free-throw target can be graded by literally the same code. The definitions --
including the level/shape split of a calibration miss and the GAME-level block
bootstrap -- are deliberately unchanged, because a bake-off whose numbers are
not comparable to the previous round's numbers is a bake-off that cannot be
read next to it.

Nothing in this module ever changes a prediction. `decile_calibration`'s
level/shape split is a DIAGNOSTIC that says which fix to look for -- a level
miss means the class's overall rate is wrong, a shape miss means the model
orders events wrongly -- which is the question `docs/SIM_GUARDRAILS.md`
section 5 requires be asked before anything is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

#: The pre-registered calibration gate: worst absolute predicted-minus-actual
#: gap over the deciles of predicted probability, on classes with at least a
#: 5% share. Same numbers as L3 round 1.
CALIB_GATE_PP = 2.0
CALIB_MIN_SHARE = 0.05

#: "monotone in 4 of 4 steps" for the rebound pre-registration; five quintiles
#: give four steps. The free-throw pre-registration inherits the L3 reading
#: (3 of 4). Each trainer passes its own number, so neither is hard-coded here.
DEFAULT_MIN_STEPS = 4


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Multiclass log loss. `y` is the integer class index of each row."""
    y = np.asarray(y)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean())


def per_class_brier(y: np.ndarray, p: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    out = {}
    for j, c in enumerate(classes):
        o = (np.asarray(y) == j).astype("float64")
        out[c] = float(((p[:, j] - o) ** 2).mean())
    return out


def multiclass_brier(y: np.ndarray, p: np.ndarray, classes: tuple[str, ...]) -> float:
    """Sum over classes of the per-class Brier score (the standard multiclass
    Brier). Reported alongside the per-class numbers so a single scalar exists
    for the decision rule's tie-breaks."""
    return float(sum(per_class_brier(y, p, classes).values()))


def decile_calibration(y: np.ndarray, p: np.ndarray, classes: tuple[str, ...],
                       n_bins: int = 10) -> dict[str, dict]:
    """Per class: predicted vs actual rate inside deciles of the predicted
    probability, the worst absolute gap, and its level/shape split."""
    y = np.asarray(y)
    out: dict[str, dict] = {}
    for j, c in enumerate(classes):
        pj = p[:, j]
        oj = (y == j).astype("float64")
        edges = np.quantile(pj, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        idx = np.clip(np.searchsorted(edges, pj, side="right") - 1, 0, n_bins - 1)
        rows, gaps = [], []
        worst = 0.0
        for b in range(n_bins):
            m = idx == b
            if m.sum() == 0:
                continue
            pred = float(pj[m].mean())
            act = float(oj[m].mean())
            gap = (pred - act) * 100
            rows.append({"bin": b, "n": int(m.sum()), "pred": round(pred, 5),
                         "actual": round(act, 5), "gap_pp": round(gap, 3)})
            gaps.append(gap)
            worst = max(worst, abs(gap))
        level = float(np.mean(gaps)) if gaps else 0.0
        residual = max((abs(g - level) for g in gaps), default=0.0)
        out[c] = {"share_pct": round(float(oj.mean() * 100), 3),
                  "max_abs_gap_pp": round(worst, 3),
                  "level_shift_pp": round(level, 3),
                  "max_abs_gap_pp_after_level_shift": round(residual, 3),
                  "bins": rows}
    return out


def calibration_verdict(calib: dict[str, dict], gate_pp: float = CALIB_GATE_PP,
                        min_share: float = CALIB_MIN_SHARE) -> tuple[bool, float, str]:
    """(passes, worst gated gap in pp, the class that produced it).

    Only classes whose ACTUAL share is at least `min_share` are gated: a class
    that happens on 0.5% of rows cannot have a decile structure worth gating,
    and pretending otherwise would fail every arm for a reason that is really a
    sample-size statement."""
    worst, who = 0.0, ""
    for c, d in calib.items():
        if d["share_pct"] < min_share * 100:
            continue
        if d["max_abs_gap_pp"] > worst:
            worst, who = d["max_abs_gap_pp"], c
    return (worst <= gate_pp), round(worst, 3), who


def quintile_responsiveness(driver: np.ndarray, y: np.ndarray, p: np.ndarray,
                            class_index: int, n_q: int = 5) -> dict:
    """The matchup-specific rule (`CLAUDE.md`): bucket the test rows by a
    driver feature's quintile and check that the PREDICTED class share slopes
    with the ACTUAL one instead of sitting flat at the league mean.

    "Monotone" is judged against the ACTUAL direction of travel, so an arm that
    slopes the right way but flatter than reality still passes the SHAPE test
    and is caught by `slope_ratio` instead -- flatness and wrong-direction are
    different failures and want different fixes."""
    v = np.asarray(driver, dtype="float64")
    y = np.asarray(y)
    edges = np.quantile(v, np.linspace(0, 1, n_q + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    q = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_q - 1)
    pred, act, ns = [], [], []
    for b in range(n_q):
        m = q == b
        ns.append(int(m.sum()))
        pred.append(float(p[m, class_index].mean()) if m.sum() else np.nan)
        act.append(float((y[m] == class_index).mean()) if m.sum() else np.nan)
    pa, aa = np.array(pred), np.array(act)
    dp, da = np.diff(pa), np.diff(aa)
    span_pred = float(pa[-1] - pa[0])
    span_act = float(aa[-1] - aa[0])
    ref = np.sign(span_act) if span_act else np.sign(span_pred)
    return {
        "n": ns,
        "pred_share": [round(x, 5) for x in pred],
        "actual_share": [round(x, 5) for x in act],
        "steps_agreeing": int(np.sum(np.sign(dp) == np.sign(da))),
        "pred_monotone_steps": int(np.sum(np.sign(dp) == ref)),
        "actual_monotone_steps": int(np.sum(np.sign(da) == ref)),
        "n_steps": int(len(dp)),
        "span_pred": round(span_pred, 5),
        "span_actual": round(span_act, 5),
        "slope_ratio": round(span_pred / span_act, 4) if span_act else None,
    }


def responsiveness_verdict(resp: dict[str, dict], min_steps: int = DEFAULT_MIN_STEPS) -> tuple[bool, int]:
    """(passes, the smallest monotone-step count over the reported drivers)."""
    if not resp:
        return False, 0
    worst = min(int(d["pred_monotone_steps"]) for d in resp.values())
    return worst >= min_steps, worst


def segment_calibration(seg_key: pd.Series, y: np.ndarray, p: np.ndarray,
                        classes: tuple[str, ...]) -> dict[str, dict]:
    """Predicted vs actual class share inside each level of a segment key, and
    the segment's own log loss. Used for the by-miss-type breakdown the rebound
    pre-registration asks for, and for the free-throw bonus/shooting split."""
    y = np.asarray(y)
    keys = pd.Series(seg_key).to_numpy()
    out: dict[str, dict] = {}
    for k in pd.unique(keys):
        m = keys == k
        if m.sum() == 0:
            continue
        seg = {"n": int(m.sum()), "log_loss": round(log_loss(y[m], p[m]), 5)}
        for j, c in enumerate(classes):
            seg[f"{c}_pred_pct"] = round(float(p[m, j].mean() * 100), 3)
            seg[f"{c}_actual_pct"] = round(float((y[m] == j).mean() * 100), 3)
            seg[f"{c}_gap_pp"] = round(seg[f"{c}_pred_pct"] - seg[f"{c}_actual_pct"], 3)
        seg["max_abs_gap_pp"] = round(max(abs(seg[f"{c}_gap_pp"]) for c in classes), 3)
        out[str(k)] = seg
    return out


def block_bootstrap_se(game_ids: np.ndarray, y: np.ndarray, p: np.ndarray,
                       n_rep: int = 200, seed: int = 12345) -> float:
    """Game-level block bootstrap SE of the multiclass log loss.

    Events inside one game are not independent -- the same lineups, the same
    officials, the same pace -- so the resampling unit is the GAME. This is the
    noise floor for the linear arms; the tree arm's floor is a seed-varied
    refit, which has its own randomness to measure."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    games = np.asarray(game_ids)
    order = np.argsort(games, kind="stable")
    gs = games[order]
    starts = np.flatnonzero(np.concatenate([[True], gs[1:] != gs[:-1]]))
    ends = np.concatenate([starts[1:], [len(gs)]])
    blocks = [order[s:e] for s, e in zip(starts, ends, strict=False)]
    n_g = len(blocks)
    ll_row = -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))
    block_sums = np.array([ll_row[b].sum() for b in blocks])
    block_ns = np.array([len(b) for b in blocks])
    losses = np.empty(n_rep)
    for r in range(n_rep):
        pick = rng.integers(0, n_g, n_g)
        losses[r] = block_sums[pick].sum() / block_ns[pick].sum()
    return float(losses.std(ddof=1))


def expanding_asof(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Cumulative sums of `cols` over rows STRICTLY BEFORE each row, within
    `keys`, in the frame's current order.

    Identical in behaviour to `possession_outcome._expanding_asof`; duplicated
    here rather than imported because that module belongs to another worker and
    the leak-safety proof in `tests/test_rebound.py` has to bind to the code
    THIS model actually calls."""
    grp = df.groupby(keys, sort=False)
    out = {}
    for c in cols:
        out[c] = grp[c].cumsum().to_numpy() - df[c].to_numpy()
    out["n_prior"] = grp.cumcount().to_numpy()
    return pd.DataFrame(out, index=df.index)


__all__ = [
    "CALIB_GATE_PP",
    "CALIB_MIN_SHARE",
    "block_bootstrap_se",
    "calibration_verdict",
    "decile_calibration",
    "expanding_asof",
    "log_loss",
    "multiclass_brier",
    "per_class_brier",
    "quintile_responsiveness",
    "responsiveness_verdict",
    "segment_calibration",
]
