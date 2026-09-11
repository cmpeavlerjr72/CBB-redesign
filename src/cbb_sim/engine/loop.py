"""
loop.py -- the possession loop, vectorised over N concurrent simulations.

Deliverable 2, and Decision 1 / Decision 7 implemented literally: possessions
are NOT drawn. Each possession draws a duration, consumes it, and the game ends
when the clock does; the possession count per game is emergent, and overtime is
an explicit period, which is where the count's right skew comes from (L14).

ONE OUTER STEP ADVANCES EVERY ACTIVE SIMULATION BY EXACTLY ONE POSSESSION.
Within a step:

  (a) CLOCK. One batched `ClockAdapter.pmf` over every active simulation, then
      an inverse-CDF draw. A draw that exceeds the period's remaining time is
      the censored case the clock bake-off drops from training: the possession
      consumes what is left and still resolves. Drawing the duration FIRST is
      also what makes `is_transition` legitimately available to the event model
      (it is `duration <= 8 AND the previous possession ended in a DREB or a
      TOV`, a function of the duration the engine has already drawn, and one the
      clock model is forbidden to see).
  (b) EVENT. One batched `EventAdapter.predict` per population over the
      simulations with a live chance: first chances to the round-1 `first`
      reference (lgbm), continuation chances to the `cont` reference (cascade).
  (c) ALLOCATION. `usage.draw_player`'s own proportional rule over the five on
      the floor, one uniform per event.
  (d) OUTCOME. fg_make per shot class (three disjoint row sets, one predict
      each); free throws by the rule table with a shooter-keyed make
      probability; rebound for every live miss with dead balls as the measured
      fixed share per miss type (L17).
  (e) ROTATION. `rotation_adapter.next_lineup` for both teams, credited with
      the possession's own duration.
  (f) PERIOD / HALFTIME / OVERTIME. Second-half possession goes to whoever lost
      the opening tip; each overtime is a 5-minute period opened by a fresh
      jump ball; team fouls reset at HALFTIME ONLY, because an NCAA extra period
      is an extension of the second half; repeat until untied.
  (g) BOOKKEEPING at the end of the game.

The inner loops (the OREB chain, the free-throw trip) iterate over a SHRINKING
SUBSET, never over games: an OREB chain is at most `MAX_CHANCES` iterations and
a trip at most three, each on the rows still live.

NO LIVE MODEL CALL HAPPENS PER GAME OR PER POSSESSION. Everything static was
reduced to an indexable block by `scripts/build_engine_inputs.py`; every model
call in this file is one batched predict for the whole batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbb_sim.engine import rotation_adapter as RA
from cbb_sim.engine import state as S
from cbb_sim.engine.adapters import STATE_INDEX, Adapters
from cbb_sim.engine.inputs import EngineInputs
from cbb_sim.engine.rng import StreamBook, categorical
from cbb_sim.models import fg_make as FG
from cbb_sim.models import possession_outcome as PO
from cbb_sim.models import rebound as RB

I = STATE_INDEX  # noqa: E741

#: possession_outcome class -> usage event class index, and the shot class.
CLS_TOV = PO.CLASS_INDEX["TOV"]
CLS_RIM = PO.CLASS_INDEX["FGA_rim"]
CLS_JUMP = PO.CLASS_INDEX["FGA_jump2"]
CLS_3 = PO.CLASS_INDEX["FGA_3"]
CLS_FT_SHOOT = PO.CLASS_INDEX["FT_trip_shooting"]
CLS_FT_BONUS = PO.CLASS_INDEX["FT_trip_bonus"]

SHOT_CLASSES = {CLS_RIM: "FGA_rim", CLS_JUMP: "FGA_jump2", CLS_3: "FGA_3"}
SHOT_KEY = {CLS_RIM: "rim", CLS_JUMP: "jump2", CLS_3: "three"}
SHOT_POINTS = {CLS_RIM: 2, CLS_JUMP: 2, CLS_3: 3}
BOX_OF_SHOT = {CLS_RIM: "fga2_rim", CLS_JUMP: "fga2_jump", CLS_3: "fga3"}
#: the matching MAKE counter per shot class (added 2026-09-10 for G4 eFG%).
BOX_MADE_OF_SHOT = {CLS_RIM: "fgm2_rim", CLS_JUMP: "fgm2_jump", CLS_3: "fgm3"}
MISS_COL = {CLS_RIM: "miss_rim", CLS_JUMP: "miss_jump2", CLS_3: "miss_three"}

#: usage event class index per possession_outcome class (usage.EVENT_CLASSES
#: order is fixed by `EngineInputs.usage_classes`).
USAGE_OF_CLASS = {
    CLS_TOV: "TOV", CLS_RIM: "FGA_rim", CLS_JUMP: "FGA_jump2", CLS_3: "FGA_3",
    CLS_FT_SHOOT: "FT_trip", CLS_FT_BONUS: "FT_trip",
}

PREV = S.PREV_END_CODE
PREV_DUMMY = ("prev_end_DREB", "prev_end_TOV", "prev_end_made_FG",
              "prev_end_made_FT", "prev_end_other")

#: An OREB chain longer than this is a pathology, not basketball. Counted.
MAX_CHANCES = 8
#: Hard step cap. A 40-minute game at the data's mean duration is ~140
#: possessions; the cap exists so a pathological draw cannot hang a run.
MAX_STEPS = 700


@dataclass
class ChunkResult:
    games: pd.DataFrame
    players: pd.DataFrame
    diag: dict
    n_possessions: int
    seconds: float


# ---------------------------------------------------------------------------
# the state block
# ---------------------------------------------------------------------------
def _state_block(st: S.GameState, act: np.ndarray, n_state: int,
                 off_sd: np.ndarray, bonus: np.ndarray) -> np.ndarray:
    """The part of the state block every sub-model shares."""
    x = np.zeros((len(act), n_state), dtype=np.float64)
    per = st.period[act].astype(np.float64)
    sr = st.seconds_remaining[act].astype(np.float64)
    sd = off_sd[act].astype(np.float64)
    x[:, I["period"]] = per
    x[:, I["seconds_remaining"]] = sr
    x[:, I["score_diff"]] = sd
    x[:, I["in_bonus"]] = bonus[act]
    x[:, I["is_ot"]] = (per >= 3.0)
    x[:, I["x_score_diff__seconds_remaining"]] = sd * sr / 1200.0
    x[:, I["chance_number"]] = st.chance_number[act].astype(np.float64)
    # `clock.FEATURES_A`'s pre-registered "chance number within possession",
    # which `clock.build_design` sets to a literal 1.0 on every row because a
    # possession begins on its first chance by the segmentation rule. Every
    # clock arm drops it as zero-variance. The engine supplies the same
    # constant rather than the live chance number: the clock is drawn once per
    # POSSESSION, at its start, so 1.0 is that column's training definition and
    # anything else would be a different feature under the same name.
    x[:, I["chance_number_at_start"]] = 1.0
    # --- fg_make round 2 (experiments.md section 13) ----------------------
    # The engine's margin is already PRE-shot (`off_score_diff()` is read
    # before the attempt resolves), so `score_diff_pre` is the same live value
    # under the name the round-2 arms were trained on. The three indicators
    # use `fg_make`'s own constants and its own `regulation_seconds_remaining`,
    # so the simulated definition cannot drift from the trained one.
    x[:, I["score_diff_pre"]] = sd
    gsr = FG.regulation_seconds_remaining(per, sr)
    reg = per <= 2.0
    x[:, I["gt_flag"]] = (reg & (np.abs(sd) >= FG.R2_GT_MARGIN)
                          & (gsr <= FG.R2_GT_SECONDS))
    x[:, I["eg_trail"]] = (reg & (sd <= -FG.R2_EG_LO) & (sd >= -FG.R2_EG_HI)
                           & (gsr <= FG.R2_EG_SECONDS))
    x[:, I["eg_lead"]] = (reg & (sd >= FG.R2_EG_LO) & (sd <= FG.R2_EG_HI)
                          & (gsr <= FG.R2_EG_SECONDS))
    pe = st.prev_end[act]
    for k, name in enumerate(PREV_DUMMY, start=1):
        x[:, I[name]] = (pe == k).astype(np.float64)
    return x


# ---------------------------------------------------------------------------
# the chunk simulator
# ---------------------------------------------------------------------------
def simulate_chunk(inp: EngineInputs, ad: Adapters, game_index: np.ndarray,
                   seeds: np.ndarray, keep_players: bool = True,
                   progress: int = 0) -> ChunkResult:
    """Simulate the cross product already expanded into (game_index, seeds).

    `game_index[i]` and `seeds[i]` together identify simulation `i`. Both arrays
    are the same length; the caller decides the batch shape.
    """
    t_start = time.time()
    n_state = len(I)
    game_index = np.asarray(game_index, dtype=np.int64)
    seeds = np.asarray(seeds, dtype=np.int64)
    n = len(game_index)
    gids = inp.games["game_id"].to_numpy()[game_index]
    seasons = inp.games["season"].to_numpy()[game_index]
    book = StreamBook(seeds, gids)

    # ---- opening tip -----------------------------------------------------
    u_tip = book.draw("tipoff")
    first_off = (u_tip < 0.5).astype(np.int8)

    st = S.new_state(game_index, seeds, seasons, inp.n_slots,
                     S.load_bonus_era(), first_off)
    rules = inp.rules
    dead_share = np.array([rules["dead_share"][m] for m in RB.MISS_TYPES], dtype=np.float64)
    miss_type_index = {m: i for i, m in enumerate(RB.MISS_TYPES)}
    and_one_rate = {c: float(rules["and_one_rate_given_made"][SHOT_CLASSES[c]])
                    for c in SHOT_CLASSES}
    p_three_shoot = float(rules["shooting_trip_three_attempt_share"])
    p_three_double = float(rules["double_bonus_three_attempt_share"])
    silent_foul = float(rules["silent_foul_per_possession"])
    ce_med = {int(k): float(v) for k, v in rules["chance_elapsed_median_by_chance"].items()}
    # lookup tables so the hot loop never runs a python comprehension
    ce_lut = np.array([ce_med.get(min(max(j, 1), 3), 3.0) for j in range(4)], dtype=np.float64)
    ucls_lut = np.array([ad.usage.class_index[USAGE_OF_CLASS[c]]
                         for c in range(len(PO.CLASSES))], dtype=np.int64)

    # ---- rotation: 2n rows, home block then away block -------------------
    # Home and away need INDEPENDENT streams off the same (seed, game) pair.
    # The offline sampler gets that from one sequential Generator drawn by home
    # first; a counter-based stream needs the side folded into the key, the same
    # injective trick `usage.event_stream_keys` uses for the event ordinal.
    RA.assert_max_candidates(inp.n_slots)
    two = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    gg = np.concatenate([game_index, game_index])
    rot_rows = np.arange(2 * n)
    rot_book = StreamBook(np.concatenate([seeds, seeds]),
                          np.concatenate([gids, gids]) * 2 + two,
                          families=("rotation", "rotation_foul", "rotation_sub"))
    # ENGINE_ROTATION=round4 serves the per-player substitution-hazard family
    # (`models/rotation_v4.py`) from an S1 manifest; `reference` is the R2
    # hierarchical Dirichlet + scheduler. The mode is read here rather than in
    # `adapters.py`, which another deliverable owns.
    r4 = RA.load_round4(inp.games) if RA.rotation_mode() == "round4" else None
    # ENGINE_ROTATION_SCHEME=s1: R2's fitted objects are a SCHEDULE (rotation
    # round 3b, experiments.md s9.4), so each of the 2N rows carries its own
    # game's refit. `gg` is already the (2N,) game row index, home block then
    # away block, which is exactly what the gather needs.
    rot_s1 = getattr(ad, "rot_s1", None)
    rot = RA.init_batch(
        ad.rot_fit,
        inp.rot_share[gg, two], inp.rot_srank[gg, two], inp.rot_fpm[gg, two],
        inp.rot_pavail[gg, two], rot_book, rot_rows,
        round4=None if r4 is None else RA.round4_rows(r4, gg),
        fitset=None if rot_s1 is None else RA.r2_s1_fitset(rot_s1, gg))
    # the rotation decides foul-outs off the same counters the box score reports,
    # so the (2n, S) block is the single source of truth until the game is over
    fouls2 = np.zeros((2 * n, inp.n_slots), dtype=np.int8)

    def push_lineups() -> None:
        # `prev_end` is the possession's own start reason -- the dead-ball
        # opportunity the round-4 hazards condition on -- and is exactly the
        # possessions table's `start_reason`. `team_fouls` is each side's own.
        five = RA.next_lineup(
            rot,
            np.concatenate([st.period, st.period]),
            np.concatenate([st.seconds_remaining, st.seconds_remaining]),
            np.concatenate([hsd, hsd]),
            np.concatenate([last_dur, last_dur]),
            fouls2, rot_book, rot_rows,
            np.concatenate([st.active, st.active]),
            prev_end=np.concatenate([st.prev_end, st.prev_end]),
            team_fouls=np.concatenate([st.team_fouls[:, 0], st.team_fouls[:, 1]]))
        st.on_floor[:, 0, :] = five[:n]
        st.on_floor[:, 1, :] = five[n:]

    hsd = st.home_score_diff()
    last_dur = np.zeros(n, dtype=np.float64)
    push_lineups()

    rows_all = np.arange(n)
    # Read once, outside the step loop: whether the clock adapter routes its
    # own dated artifacts per game (round-3c/4 S1 arms) or is a single static
    # object (the incumbent).
    clock_gidx = bool(getattr(ad.clock, "wants_game_index", False))
    diag: dict[str, int] = {}

    def bump(k: str, v: int = 1) -> None:
        diag[k] = diag.get(k, 0) + int(v)

    step = 0
    while st.active.any() and step < MAX_STEPS:
        step += 1
        act = np.flatnonzero(st.active)
        m = len(act)
        gidx = st.game_index[act]
        off = st.off[act].astype(np.int64)
        dfn = 1 - off
        off_sd = st.off_score_diff()
        bonus = st.in_bonus()
        dbonus = st.in_double_bonus()

        team_off = inp.team_static[gidx, off]

        # ---- (a) clock ---------------------------------------------------
        st.chance_number[act] = 1
        x = _state_block(st, act, n_state, off_sd, bonus)
        # An S1 clock schedule needs the row's own game to pick that game's
        # monthly refit, exactly as `ad.event.predict` and `ad.fg.predict` do.
        # The incumbent static `ClockAdapter` declares no `wants_game_index`
        # and keeps its three-argument call unchanged.
        dur = (ad.clock.draw(team_off, x, book.draw("clock", act), gidx)
               if clock_gidx else ad.clock.draw(team_off, x, book.draw("clock", act)))
        left = st.seconds_remaining[act].astype(np.int64)
        censored = dur >= left
        bump("possessions_censored_by_period_end", int(censored.sum()))
        used = np.minimum(dur, left).astype(np.float64)
        st.poss_duration[act] = used.astype(np.int16)

        # `is_transition` exactly as the model was trained: the drawn duration
        # is <= 8 s and the previous possession ended in a DREB or a TOV.
        prev = st.prev_end[act]
        trans = ((dur <= 8) & ((prev == PREV["DREB"]) | (prev == PREV["TOV"]))).astype(np.float64)

        # ---- (b)-(d) the chance cascade ----------------------------------
        live = np.ones(m, dtype=bool)          # rows whose possession is still open
        chance = np.ones(m, dtype=np.int64)
        end_code = np.full(m, PREV["other"], dtype=np.int8)
        for c_iter in range(MAX_CHANCES):
            rows = np.flatnonzero(live)
            if not len(rows):
                break
            a_rows = act[rows]
            st.chance_number[a_rows] = chance[rows].astype(np.int8)
            bonus = st.in_bonus()
            dbonus = st.in_double_bonus()
            xx = _state_block(st, a_rows, n_state, st.off_score_diff(), bonus)
            is_first = chance[rows] == 1
            xx[:, I["is_transition"]] = np.where(is_first, trans[rows], 0.0)
            xx[:, I["is_transition_f"]] = xx[:, I["is_transition"]]
            xx[:, I["chance_elapsed_s"]] = np.where(
                is_first, used[rows], ce_lut[np.minimum(chance[rows], 3)])
            t_off = inp.team_static[gidx[rows], off[rows]]

            probs = ad.event.predict(t_off, xx, is_first, gidx[rows], off[rows])
            cls = categorical(book.draw("event", a_rows), probs)

            # ---- (c) allocation: who of the five ------------------------
            five = st.on_floor[a_rows, off[rows]]            # (k, 5) slot indices
            ucls = ucls_lut[cls]
            r5 = inp.usage_rate[gidx[rows][:, None], off[rows][:, None], five, ucls[:, None]]
            pick = categorical(book.draw("usage", a_rows), ad.usage.probs(r5.astype(np.float64)))
            shooter = five[np.arange(len(rows)), pick]

            cont = np.zeros(m, dtype=bool)       # step-space: chance continues on an OREB
            # ================= turnovers =================================
            tov = cls == CLS_TOV
            if tov.any():
                r = rows[tov]
                st.box["tov"][act[r], off[r]] += 1
                end_code[r] = PREV["TOV"]

            # ================= field-goal attempts ========================
            miss_rows: list[np.ndarray] = []
            miss_kind: list[np.ndarray] = []
            ft_rows: list[np.ndarray] = []
            ft_shooter: list[np.ndarray] = []
            ft_natt: list[np.ndarray] = []
            ft_oao: list[np.ndarray] = []
            for sc in (CLS_RIM, CLS_JUMP, CLS_3):
                sel = cls == sc
                if not sel.any():
                    continue
                r = rows[sel]
                ar = act[r]
                sh = shooter[sel]
                st.box[BOX_OF_SHOT[sc]][ar, off[r]] += 1
                st.player_box["fga"][ar, off[r], sh] += 1
                if sc == CLS_3:
                    st.player_box["fg3a"][ar, off[r], sh] += 1
                xs = xx[sel].copy()
                xs[:, I["blocked_f"]] = 0.0
                slot_blk = inp.slot_static[gidx[r], off[r], sh]
                p_make = ad.fg.predict(SHOT_CLASSES[sc], inp.team_static[gidx[r], off[r]],
                                       slot_blk, xs, gidx[r])
                made = book.draw("fg_make", ar) < p_make
                pv = SHOT_POINTS[sc]
                if made.any():
                    mr = r[made]
                    st.pts[act[mr], off[mr]] += pv
                    st.box[BOX_MADE_OF_SHOT[sc]][act[mr], off[mr]] += 1
                    st.player_box["pts"][act[mr], off[mr], sh[made]] += pv
                    st.player_box[BOX_MADE_OF_SHOT[sc]][act[mr], off[mr], sh[made]] += 1
                    end_code[mr] = PREV["made_FG"]
                    # and-one: a shooting foul on a made basket, one attempt,
                    # at the measured rate per shot class
                    ao = book.draw("and_one", act[mr]) < and_one_rate[sc]
                    if ao.any():
                        a2 = mr[ao]                       # step-space rows
                        st.team_fouls[act[a2], dfn[a2]] += 1
                        ft_rows.append(a2)
                        ft_shooter.append(sh[made][ao])
                        ft_natt.append(np.ones(len(a2), dtype=np.int64))
                        ft_oao.append(np.zeros(len(a2), dtype=bool))
                if (~made).any():
                    xr = r[~made]
                    miss_rows.append(xr)
                    miss_kind.append(np.full(len(xr), miss_type_index[SHOT_KEY[sc]]))

            # ================= free-throw trips ===========================
            for tc in (CLS_FT_SHOOT, CLS_FT_BONUS):
                sel = cls == tc
                if not sel.any():
                    continue
                r = rows[sel]
                ar = act[r]
                st.team_fouls[ar, dfn[r]] += 1
                sh = shooter[sel]
                if tc == CLS_FT_SHOOT:
                    three = book.draw("free_throw", ar) < p_three_shoot
                    natt = np.where(three, 3, 2)
                    oao = np.zeros(len(r), dtype=bool)
                else:
                    db = dbonus[ar].astype(bool)
                    bo = bonus[ar].astype(bool) & ~db
                    out = ~db & ~bo
                    bump("ft_bonus_awarded_out_of_bonus", int(out.sum()))
                    three = book.draw("free_throw", ar) < p_three_double
                    natt = np.where(db, np.where(three, 3, 2), np.where(bo, 2, 2))
                    oao = bo
                ft_rows.append(r)                         # step-space rows
                ft_shooter.append(sh)
                ft_natt.append(natt)
                ft_oao.append(oao)

            if ft_rows:
                fr = np.concatenate(ft_rows)              # step-space
                fs = np.concatenate(ft_shooter)
                fn = np.concatenate(ft_natt)
                fo = np.concatenate(ft_oao)
                last_missed = _shoot_trip(st, inp, ad, book, act[fr], fs, fn, fo, n_state)
                # a missed last attempt is a live rebound (miss type "ft");
                # a made last attempt ends the possession
                end_code[fr[~last_missed]] = PREV["made_FT"]
                if last_missed.any():
                    miss_rows.append(fr[last_missed])
                    miss_kind.append(np.full(int(last_missed.sum()), miss_type_index["ft"]))

            # ================= rebounds ===================================
            if miss_rows:
                mr = np.concatenate(miss_rows)
                mk = np.concatenate(miss_kind)
                amr = act[mr]
                xr = _state_block(st, amr, n_state, st.off_score_diff(), st.in_bonus())
                for name in ("miss_rim", "miss_jump2", "miss_three"):
                    xr[:, I[name]] = 0.0
                for key, col in (("rim", "miss_rim"), ("jump2", "miss_jump2"),
                                 ("three", "miss_three")):
                    xr[:, I[col]] = (mk == miss_type_index[key]).astype(np.float64)
                xr[:, I["blocked_f"]] = 0.0
                p3 = ad.reb.predict(inp.team_static[gidx[mr], off[mr]], xr, gidx[mr])
                # dead balls as the measured fixed share per miss type (L17),
                # composed through the module's own function
                p3 = RB.compose_binary_plus_fixed_dead(
                    p3[:, RB.CLASS_INDEX["OREB"]]
                    / np.maximum(p3[:, RB.CLASS_INDEX["OREB"]] + p3[:, RB.CLASS_INDEX["DREB"]], 1e-12),
                    np.array(RB.MISS_TYPES)[mk], dict(ad.reb.dead_share))
                out3 = categorical(book.draw("rebound", amr), p3)
                is_o = out3 == RB.CLASS_INDEX["OREB"]
                is_d = out3 == RB.CLASS_INDEX["DREB"]
                is_x = out3 == RB.CLASS_INDEX["DEAD"]
                if is_o.any():
                    ro = mr[is_o]
                    st.box["oreb"][act[ro], off[ro]] += 1
                    _credit_rebound(st, inp, book, act[ro], off[ro], 0)
                    cont[ro] = True
                if is_d.any():
                    rd = mr[is_d]
                    st.box["dreb"][act[rd], dfn[rd]] += 1
                    _credit_rebound(st, inp, book, act[rd], dfn[rd], 1)
                    end_code[rd] = PREV["DREB"]
                if is_x.any():
                    bump("dead_ball_rebounds", int(is_x.sum()))
                    end_code[mr[is_x]] = PREV["other"]

            live[:] = False
            live[cont] = True
            chance[cont] += 1
            if c_iter == MAX_CHANCES - 1 and live.any():
                bump("oreb_chain_truncated", int(live.sum()))
                end_code[np.flatnonzero(live)] = PREV["DREB"]
                live[:] = False

        # ---- non-shooting foul that awards no attempt (measured gap) ------
        sf = book.draw("foul_accrual", act) < silent_foul
        if sf.any():
            rs = np.flatnonzero(sf)
            st.team_fouls[act[rs], dfn[rs]] += 1

        # ---- possession bookkeeping and the clock ------------------------
        st.poss_count[act, off] += 1
        st.prev_end[act] = end_code
        st.off[act] = dfn.astype(np.int8)
        st.chance_number[act] = 1
        st.seconds_remaining[act] = (left - used).astype(np.int16)
        st.player_seconds[act[:, None], off[:, None], st.on_floor[act, off]] += \
            used[:, None].astype(np.float32)
        st.player_seconds[act[:, None], dfn[:, None], st.on_floor[act, dfn]] += \
            used[:, None].astype(np.float32)

        # ---- (e) rotation: state only; the call itself is after (f) --------
        last_dur[:] = 0.0
        last_dur[act] = used
        hsd = st.home_score_diff()

        # ---- (f) period / halftime / overtime ----------------------------
        ended = st.active & (st.seconds_remaining <= 0)
        if ended.any():
            e = np.flatnonzero(ended)
            per_at_end = st.period[e].copy()   # split BEFORE any period is advanced
            half = e[per_at_end == 1]
            if len(half):
                st.period[half] = 2
                st.seconds_remaining[half] = S.HALF_SECONDS
                st.team_fouls[half] = 0          # team fouls reset at HALFTIME
                st.off[half] = 1 - st.first_off[half]
                st.prev_end[half] = PREV["period_start"]
                st.chance_number[half] = 1
            reg = e[per_at_end >= 2]
            if len(reg):
                tied = st.pts[reg, 0] == st.pts[reg, 1]
                fin = reg[~tied]
                st.active[fin] = False
                ot = reg[tied]
                cap = st.n_ot[ot] >= S.MAX_OT_PERIODS
                if cap.any():
                    bump("overtime_cap_reached", int(cap.sum()))
                    st.active[ot[cap]] = False
                ot = ot[~cap]
                if len(ot):
                    st.period[ot] += 1
                    st.n_ot[ot] += 1
                    st.seconds_remaining[ot] = S.OT_SECONDS
                    # NCAA: an extra period is an extension of the second half,
                    # so team fouls CARRY OVER; only halftime resets them.
                    st.off[ot] = (book.draw("tipoff", ot) < 0.5).astype(np.int8)
                    st.prev_end[ot] = PREV["period_start"]
                    st.chance_number[ot] = 1

        # ---- (e, continued) the rotation chooses the NEXT possession's five
        # DEFECT FIX (2026-09-10, rotation round 4). This call used to sit
        # before block (f), so at a period boundary the rotation was handed the
        # PREVIOUS possession's period and clock: the five that took the floor
        # for the first possession of the second half were chosen with
        # `period = 1, seconds_remaining = 0`, and the R2 adapter's own
        # period-boundary reshuffle fired one possession late. The offline
        # samplers (`run_scheduler`, `run_sub_hazard`) have always used the
        # possession's own state, so the engine and the bake-off disagreed
        # exactly at the cell round 4 adds. Moving the call after (f) makes the
        # engine match the samplers. It changes R2's engine lineups at every
        # period boundary and is a behaviour change, stated here rather than
        # buried.
        push_lineups()

        if progress and step % progress == 0:
            print(f"    step {step}: {int(st.active.sum())}/{n} active", flush=True)

    if st.active.any():
        bump("step_cap_reached", int(st.active.sum()))
        st.active[:] = False

    # mirror the rotation's foul view back into the state
    st.player_fouls[:, 0, :] = fouls2[:n]
    st.player_fouls[:, 1, :] = fouls2[n:]
    diag.update(rot.diag)

    games, players = _finalise(inp, st, gids, seeds, keep_players)
    n_poss = int(st.poss_count.sum())
    return ChunkResult(games, players, diag, n_poss, time.time() - t_start)


# ---------------------------------------------------------------------------
# free-throw trip: the rule table, shot one attempt at a time
# ---------------------------------------------------------------------------
def _shoot_trip(st: S.GameState, inp: EngineInputs, ad: Adapters, book: StreamBook,
                rows: np.ndarray, shooter: np.ndarray, n_att: np.ndarray,
                one_and_one: np.ndarray, n_state: int) -> np.ndarray:
    """Shoot every attempt of a trip. Returns, per row, whether the LAST attempt
    taken was missed (which makes it a live rebound).

    The one-and-one is the rule, not a model: attempt two exists if and only if
    attempt one was made (`free_throw.TRIP_RULES["bonus_one_and_one"]`).
    """
    side = st.off[rows].astype(np.int64)
    gidx = st.game_index[rows].astype(np.int64)
    alive = np.ones(len(rows), dtype=bool)
    last_missed = np.zeros(len(rows), dtype=bool)
    for a in range(3):
        take = alive & (a < n_att)
        if not take.any():
            break
        k = np.flatnonzero(take)
        r = rows[k]
        sh = shooter[k]
        x = _state_block(st, r, n_state, st.off_score_diff(), st.in_bonus())
        p = ad.ft.predict(inp.team_static[gidx[k], side[k]],
                          inp.slot_static[gidx[k], side[k], sh], x, gidx[k])
        made = book.draw("free_throw", r) < p
        st.box["fta"][r, side[k]] += 1
        st.player_box["fta"][r, side[k], sh] += 1
        if made.any():
            mk = k[made]
            st.pts[rows[mk], side[mk]] += 1
            st.box["ftm"][rows[mk], side[mk]] += 1
            st.player_box["pts"][rows[mk], side[mk], shooter[mk]] += 1
        last_missed[k] = ~made
        # a one-and-one front end that misses ends the trip
        stop = one_and_one[k] & (a == 0) & ~made
        alive[k] = ~stop
    return last_missed


def _credit_rebound(st: S.GameState, inp: EngineInputs, book: StreamBook,
                    rows: np.ndarray, side: np.ndarray, kind: int) -> None:
    """Attribute one team rebound to one of that side's five on the floor,
    proportional to their as-of individual rebound rates (L17: the rebound MODEL
    is team-level; per-player as-of rates are for attribution only)."""
    five = st.on_floor[rows, side]
    w = inp.reb_rate[st.game_index[rows].astype(np.int64)[:, None], side[:, None],
                     five, kind].astype(np.float64)
    tot = w.sum(axis=1, keepdims=True)
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 0.2)
    pick = categorical(book.draw("rebound", rows), w)
    st.player_box["reb"][rows, side, five[np.arange(len(rows)), pick]] += 1


# ---------------------------------------------------------------------------
# end-of-game bookkeeping (deliverable 2g)
# ---------------------------------------------------------------------------
def _finalise(inp: EngineInputs, st: S.GameState, gids: np.ndarray, seeds: np.ndarray,
              keep_players: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.DataFrame({
        "game_id": gids.astype("int64"),
        "seed": seeds.astype("int32"),
        "home_pts": st.pts[:, 0].astype("int16"),
        "away_pts": st.pts[:, 1].astype("int16"),
        "possessions": st.possessions().astype("float32"),
        "n_periods": st.n_periods().astype("int16"),
    })
    for stat in S.BOX_STATS:
        g[f"home_{stat}"] = st.box[stat][:, 0].astype("int16")
        g[f"away_{stat}"] = st.box[stat][:, 1].astype("int16")
    if not keep_players:
        return g, pd.DataFrame()

    n, _, sl = st.player_seconds.shape
    secs = st.player_seconds.reshape(-1)
    gi = np.repeat(st.game_index.astype(np.int64), 2 * sl)
    side = np.tile(np.repeat([0, 1], sl), n)
    slot = np.tile(np.arange(sl), 2 * n)
    espn = inp.roster_espn[gi, side, slot]
    cbbd = inp.roster_cbbd[gi, side, slot]
    team = np.where(side == 0, inp.games["home_team_id"].to_numpy()[gi],
                    inp.games["away_team_id"].to_numpy()[gi])
    keep = (secs > 0) & (espn > 0)
    p = pd.DataFrame({
        "game_id": np.repeat(gids.astype("int64"), 2 * sl)[keep],
        "seed": np.repeat(seeds.astype("int32"), 2 * sl)[keep],
        "athlete_id": espn[keep].astype("int64"),
        "cbbd_id": cbbd[keep].astype("int64"),
        "team_id": team[keep].astype("int64"),
        "minutes": (secs[keep] / 60.0).astype("float32"),
        "pts": st.player_box["pts"].reshape(-1)[keep].astype("int16"),
        "reb": st.player_box["reb"].reshape(-1)[keep].astype("int16"),
        # `ast` is a PLACEHOLDER: no assist model exists, so it is written as 0
        # and recorded as absent in run_meta, never as a fabricated count.
        "ast": np.zeros(int(keep.sum()), dtype="int16"),
        "fga": st.player_box["fga"].reshape(-1)[keep].astype("int16"),
        "fg3a": st.player_box["fg3a"].reshape(-1)[keep].astype("int16"),
        "fta": st.player_box["fta"].reshape(-1)[keep].astype("int16"),
        # per-class FGM, added 2026-09-10 (optional columns; REQUIRED_PLAYER_COLUMNS
        # is untouched -- see contract.py).
        "fgm2_rim": st.player_box["fgm2_rim"].reshape(-1)[keep].astype("int16"),
        "fgm2_jump": st.player_box["fgm2_jump"].reshape(-1)[keep].astype("int16"),
        "fgm3": st.player_box["fgm3"].reshape(-1)[keep].astype("int16"),
        "fouls": st.player_fouls.reshape(-1)[keep].astype("int8"),
    })
    return g, p
