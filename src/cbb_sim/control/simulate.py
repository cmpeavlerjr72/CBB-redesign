"""
simulate.py -- the Control engine's vectorised simulator.

model.md section 3, implemented literally:

  "for each game and sim draw, one shared possessions draw; each team's
   attempts from its rate model scaled by the shared possessions; makes from
   Beta-Binomial; points = 3 x 3PM + 2 x 2PM + FTM. Ties resolve by
   re-simulating a 5-minute period at the same per-possession rates with
   possessions drawn as 5/40 of a game's draw (an explicit stub, flagged,
   replaced by the L5 overtime model). RNG seeded on (seed, game_id,
   'control')."

The shared possession draw is what couples the two team scores (CLAUDE.md
modelling rule 3), and G5's home/away score correlation is the gate that reads
whether it is doing its job.

NO LIVE MODEL CALLS IN THE LOOP. Every model is reduced to one linear
predictor per game per side ONCE, in `prepare()`; the seed loop is pure numpy
plus inverse-CDF variate draws. This is also where the feature preflight runs:
`prepare` calls `models.preflight` for every component model, so a sim row
missing any model feature fails loudly before a single draw.

STUB, FLAGGED. The overtime rule above is a stub: the pace target
(`game_poss`) is the OBSERVED possession count, which already includes any
overtime a real game played, so a tied sim adds a further 5/40 of a game on
top. The expected size of that double count is P(sim tie) x 5/40 x poss, about
+0.2 possessions per game, and it is reported against G1 rather than corrected
-- correcting it here would be exactly the post-hoc adjustment the guardrails
ban, and the real fix is the L5 overtime model plus a regulation-only pace
target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import rng as crng
from .models import RATE_TARGETS, ControlModels, preflight

# Stream index layout within one (seed, game_id) stream. OT round r shifts the
# whole block by OT_STRIDE * r, so no draw is ever reused.
IDX_POSS = 0
IDX_RATE_HOME = 1          # +0..3 -> tpa, fg2a, fta, tov
IDX_RATE_AWAY = 5          # +0..3
IDX_BETA_HOME = 9          # +0..2 -> tp, fg2, ft
IDX_BINOM_HOME = 12        # +0..2
IDX_BETA_AWAY = 15         # +0..2
IDX_BINOM_AWAY = 18        # +0..2
OT_STRIDE = 64

OT_FRACTION = 5.0 / 40.0   # a 5-minute period as a fraction of a 40-minute game
MAX_OT_PERIODS = 10

PCT_ORDER: tuple[str, ...] = ("tp_pct", "fg2_pct", "ft_pct")
# 3PM counts 3, 2PM counts 2, FTM counts 1
POINT_VALUE = np.array([3.0, 2.0, 1.0])
# which rate target supplies the trials for each pct target
PCT_TRIALS = {"tp_pct": "tpa", "fg2_pct": "fg2a", "ft_pct": "fta"}


@dataclass
class SimInputs:
    """Per-game lookup tables: everything the seed loop needs, precomputed."""

    game_ids: np.ndarray
    mu_poss: np.ndarray
    sd_poss: float
    eta: dict[str, np.ndarray]        # f"{side}_{target}" -> log-rate per 100 poss
    rate_family: dict[str, str]
    rate_alpha: dict[str, float]
    p: dict[str, np.ndarray]          # f"{side}_{pct}" -> success probability
    pct_rho: dict[str, float]
    n_games: int
    features_checked: list[str] = field(default_factory=list)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prepare(games: pd.DataFrame, team_games: pd.DataFrame, bundle: ControlModels) -> SimInputs:
    """Reduce every component model to per-game arrays and preflight features.

    `games` is one row per game (pace features); `team_games` is one row per
    team-game and must cover both sides of every game in `games`.
    """
    # Preflight: the pace model's features live on the game table, every other
    # model's on the team-game table. A missing feature raises here, before a
    # single draw.
    preflight(games, bundle.pace)
    for m in (*bundle.rates.values(), *bundle.pcts.values()):
        preflight(team_games, m)
    checked = sorted({f for m in bundle.all_models() for f in m.features})

    g = games.reset_index(drop=True)
    home = team_games[team_games["team_id"] == team_games["home_team_id"]]
    away = team_games[team_games["team_id"] == team_games["away_team_id"]]
    home = g[["game_id"]].merge(home, on="game_id", how="left")
    away = g[["game_id"]].merge(away, on="game_id", how="left")
    if home["team_id"].isna().any() or away["team_id"].isna().any():
        raise ValueError("prepare(): team_games does not cover both sides of every game")

    eta, p = {}, {}
    for side, tgs in (("home", home), ("away", away)):
        for t in RATE_TARGETS:
            eta[f"{side}_{t}"] = bundle.rates[t].linpred(tgs)
        for t in PCT_ORDER:
            p[f"{side}_{t}"] = _expit(bundle.pcts[t].linpred(tgs))

    return SimInputs(
        game_ids=g["game_id"].to_numpy(dtype="int64"),
        mu_poss=bundle.pace.linpred(g),
        sd_poss=float(bundle.pace.resid_sd),
        eta=eta,
        rate_family={t: bundle.rates[t].family for t in RATE_TARGETS},
        rate_alpha={t: float(bundle.rates[t].alpha) for t in RATE_TARGETS},
        p=p,
        pct_rho={t: float(bundle.pcts[t].rho) for t in PCT_ORDER},
        n_games=len(g),
        features_checked=checked,
    )


def _draw_counts(inp: SimInputs, keys, base_idx: int, side: str, poss: np.ndarray,
                 tile: int) -> dict[str, np.ndarray]:
    out = {}
    for j, t in enumerate(RATE_TARGETS):
        mu = np.exp(np.tile(inp.eta[f"{side}_{t}"], tile)) * poss / 100.0
        if inp.rate_family[t] == "negbin":
            out[t] = crng.negbin(keys, base_idx + j, mu, inp.rate_alpha[t])
        else:
            out[t] = crng.poisson(keys, base_idx + j, mu)
    return out


def _draw_points(inp: SimInputs, keys, beta_idx: int, binom_idx: int, side: str,
                 counts: dict[str, np.ndarray], tile: int) -> np.ndarray:
    pts = np.zeros_like(counts["tpa"])
    for j, t in enumerate(PCT_ORDER):
        trials = counts[PCT_TRIALS[t]]
        made = crng.beta_binomial(
            keys, beta_idx + j, binom_idx + j, trials,
            np.tile(inp.p[f"{side}_{t}"], tile), inp.pct_rho[t],
        )
        pts += POINT_VALUE[j] * made
    return pts


def simulate(
    inp: SimInputs,
    seeds: np.ndarray,
    family: str = "control",
    chunk_seeds: int = 20,
) -> pd.DataFrame:
    """One row per (game, seed): home_pts, away_pts, possessions, n_ot.

    Vectorised across `chunk_seeds` seeds at a time; the RNG is keyed on
    (seed, game_id, family) so the chunking is an implementation detail that
    cannot change a single draw.
    """
    seeds = np.asarray(seeds, dtype=np.int64)
    n_g = inp.n_games
    out_h, out_a, out_p, out_ot, out_seed, out_gid = [], [], [], [], [], []
    n_poss_floor = 0

    for start in range(0, len(seeds), chunk_seeds):
        chunk = seeds[start:start + chunk_seeds]
        tile = len(chunk)
        keys = np.concatenate([crng.stream_keys(int(s), inp.game_ids, family) for s in chunk])

        poss = crng.normal(keys, IDX_POSS, np.tile(inp.mu_poss, tile), inp.sd_poss)
        floored = poss < 1.0
        n_poss_floor += int(floored.sum())
        poss = np.maximum(poss, 1.0)   # numerical floor only; never binds in practice

        ch = _draw_counts(inp, keys, IDX_RATE_HOME, "home", poss, tile)
        ca = _draw_counts(inp, keys, IDX_RATE_AWAY, "away", poss, tile)
        pts_h = _draw_points(inp, keys, IDX_BETA_HOME, IDX_BINOM_HOME, "home", ch, tile)
        pts_a = _draw_points(inp, keys, IDX_BETA_AWAY, IDX_BINOM_AWAY, "away", ca, tile)

        # ---- explicit overtime stub -------------------------------------
        total_poss = poss.copy()
        n_ot = np.zeros(len(poss), dtype=np.int16)
        tied = pts_h == pts_a
        rnd = 0
        while tied.any() and rnd < MAX_OT_PERIODS:
            rnd += 1
            base = OT_STRIDE * rnd
            idx = np.flatnonzero(tied)
            k_ot = keys[idx]
            poss_ot = poss[idx] * OT_FRACTION
            # index the per-game lookup tables down to just the tied rows
            gi = idx % n_g
            sub = SimInputs(
                game_ids=inp.game_ids[gi], mu_poss=inp.mu_poss[gi], sd_poss=inp.sd_poss,
                eta={k: v[gi] for k, v in inp.eta.items()}, rate_family=inp.rate_family,
                rate_alpha=inp.rate_alpha, p={k: v[gi] for k, v in inp.p.items()},
                pct_rho=inp.pct_rho, n_games=len(gi),
            )
            ch_ot = _draw_counts(sub, k_ot, base + IDX_RATE_HOME, "home", poss_ot, 1)
            ca_ot = _draw_counts(sub, k_ot, base + IDX_RATE_AWAY, "away", poss_ot, 1)
            pts_h[idx] += _draw_points(sub, k_ot, base + IDX_BETA_HOME, base + IDX_BINOM_HOME,
                                       "home", ch_ot, 1)
            pts_a[idx] += _draw_points(sub, k_ot, base + IDX_BETA_AWAY, base + IDX_BINOM_AWAY,
                                       "away", ca_ot, 1)
            total_poss[idx] += poss_ot
            n_ot[idx] += 1
            still = pts_h[idx] == pts_a[idx]
            tied = np.zeros(len(poss), dtype=bool)
            tied[idx[still]] = True

        out_h.append(pts_h.astype(np.int16))
        out_a.append(pts_a.astype(np.int16))
        out_p.append(total_poss.astype(np.float32))
        out_ot.append(n_ot)
        out_seed.append(np.repeat(chunk, n_g).astype(np.int32))
        out_gid.append(np.tile(inp.game_ids, tile))

    df = pd.DataFrame({
        "game_id": np.concatenate(out_gid),
        "seed": np.concatenate(out_seed),
        "home_pts": np.concatenate(out_h),
        "away_pts": np.concatenate(out_a),
        "possessions": np.concatenate(out_p),
        "n_ot": np.concatenate(out_ot),
    })
    df.attrs["n_poss_floor"] = n_poss_floor
    return df


def summarise(sims: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per-game summary: mean/SD margin and total, p_home, sim possessions,
    OT rate, and the actual result carried alongside for grading."""
    s = sims.copy()
    s["margin"] = s["home_pts"].astype("int32") - s["away_pts"].astype("int32")
    s["total"] = s["home_pts"].astype("int32") + s["away_pts"].astype("int32")
    s["home_win"] = np.where(s["margin"] > 0, 1.0, np.where(s["margin"] < 0, 0.0, 0.5))
    g = s.groupby("game_id")
    out = pd.DataFrame({
        "sim_margin_mean": g["margin"].mean(),
        "sim_margin_sd": g["margin"].std(),
        "sim_total_mean": g["total"].mean(),
        "sim_total_sd": g["total"].std(),
        "sim_home_pts_mean": g["home_pts"].mean(),
        "sim_away_pts_mean": g["away_pts"].mean(),
        "sim_poss_mean": g["possessions"].mean(),
        "sim_poss_sd": g["possessions"].std(),
        "p_home": g["home_win"].mean(),
        "sim_ot_rate": g["n_ot"].apply(lambda x: float((x > 0).mean())),
        "n_seeds": g.size(),
    }).reset_index()
    return games.merge(out, on="game_id", how="inner")
