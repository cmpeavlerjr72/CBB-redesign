"""diag_g1_unknown_poss_v1.py -- what the pbp layer's `unknown`-terminal possessions are.

Lane B follow-up, 2026-09-30.  DIAGNOSTIC ONLY.  Re-runs the possession
machine of `src/cbb_sim/pbp/possessions.py` (default build, tech_lookahead
off, i.e. the machine that wrote `possessions_v2`) through a SUBCLASS that
only records context whenever a possession is closed with terminal
`unknown`; every handler delegates to the parent unchanged.  The per-game
possession counts are checked against `possessions_v2` (must be identical).

Per unknown close it records the trigger (a DREB arriving with no open
possession, or the mismatch guard closing a possession with nothing pending,
or a period end), the events around it, and whether the previous close was an
and-one whose single free throw was MISSED.  Also records every and-one close
by FT branch, and the OREB-after-missed-and-one case (same-team restart).

Writes results/g1g5_diag/unknown_poss_{season}.parquet, andone_{season}.parquet
and unknown_summary.json.  Usage: diag_g1_unknown_poss_v1.py [seasons...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.pbp import possessions as PZ  # noqa: E402
from cbb_sim.pbp.events import load_plays  # noqa: E402

OUT = ROOT / "results/g1g5_diag"
SEASONS = [int(s) for s in sys.argv[1:]] or [2022, 2023, 2024, 2025]


class TapMachine(PZ._GameMachine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trigger = None
        self.last_andone = None          # (idx_ft, ft_made, shooter_side) of the most recent and-one close
        self.rec: list[dict] = []
        self.andone: list[dict] = []
        self.opened_by = None

    def _ctx(self, i):
        ev = self.ev
        lo, hi = max(0, i - 3), min(self.n, i + 3)
        return "|".join(f"{ev['cls'][j]}:{int(ev['team'][j])}:{int(ev['sec'][j])}" for j in range(lo, hi))

    def _open(self, team, i):
        super()._open(team, i)
        self.opened_by = (self.ev["cls"][i] if i < self.n else "none", i)

    def _close(self, terminal, i, reason_next="other"):
        if terminal == "unknown" and self.cur is not None:
            ev = self.ev
            ii = min(i, self.n - 1)
            prev_andone_missed = (self.last_andone is not None and not self.last_andone[1]
                                  and self.last_andone[0] < ii
                                  and self.last_andone[0] >= ii - 3)
            self.rec.append({
                "cbbd_game_id": self.m["cbbd_game_id"], "game_id": self.m["game_id"],
                "season": self.m["season"], "period": self.period,
                "start_clock": self.cur.start_clock, "end_clock": int(ev["sec"][ii]),
                "start_reason": self.cur.start_reason, "trigger": self.trigger or "other",
                "offense_side": self.cur.offense_team_id,
                "opened_by": self.opened_by[0] if self.opened_by else "none",
                "n_chances": len(self.cur.chances),
                "event_cls": ev["cls"][ii], "event_team": int(ev["team"][ii]),
                "prev_andone_missed": bool(prev_andone_missed),
                "ctx": self._ctx(ii),
            })
        self.trigger = None
        super()._close(terminal, i, reason_next)

    def _handle_dreb(self, i, t):
        if self.cur is None:
            self.trigger = "dreb_no_open_possession"
        super()._handle_dreb(i, t)
        self.trigger = None

    def _ensure(self, team, i):
        if self.cur is not None and self.cur.offense_team_id != team:
            self.trigger = "mismatch_guard"
        super()._ensure(team, i)
        self.trigger = None

    def _handle_fga(self, i, c, t):
        before = len(self.possessions)
        r = super()._handle_fga(i, c, t)
        if len(self.possessions) > before and self.possessions[-1].chances[-1].and_one:
            k = r - 1
            made = self.ev["cls"][k] == "FT_made"
            self.last_andone = (k, bool(made), t)
            nxt = k + 1 if k + 1 < self.n else None
            self.andone.append({
                "season": self.m["season"], "game_id": self.m["game_id"], "period": self.period,
                "ft_made": bool(made), "shooter_side": t,
                "next_cls": self.ev["cls"][nxt] if nxt is not None else "none",
                "next_team": int(self.ev["team"][nxt]) if nxt is not None else -9,
                "next_sec_gap": int(self.ev["sec"][k] - self.ev["sec"][nxt]) if nxt is not None else -9,
            })
        return r

    def _handle_oreb(self, i, t):
        if self.cur is None:
            self.trigger = "oreb_no_open_possession"
        r = super()._handle_oreb(i, t)
        self.trigger = None
        return r


def run_season(season: int):
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    u = u[u["is_d1_game"] & ~u["pbp_truncated"] & (u["season"] == season) & u["cbbd_game_id"].notna()]
    meta = {int(r.cbbd_game_id): {"game_id": int(r.game_id), "cbbd_game_id": int(r.cbbd_game_id),
                                  "season": int(r.season), "home_team_id": int(r.home_team_id),
                                  "away_team_id": int(r.away_team_id)} for r in u.itertuples()}
    plays = load_plays(season, game_ids=set(meta))
    ev = PZ._prepare_events(plays, with_on_floor=False)
    games = ev["game"]
    bounds = np.flatnonzero(np.concatenate([[True], games[1:] != games[:-1], [True]]))
    rec, ao, counts = [], [], {}
    for b in range(len(bounds) - 1):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        gm = meta.get(int(games[lo]))
        if gm is None:
            continue
        sub = {k: ev[k][lo:hi] for k in ("cls", "team", "sec", "period", "hs", "as_", "made", "stolen")}
        sub["on_floor"] = None
        m = TapMachine(gm, sub, tech_lookahead=False)
        m.run()
        rec += m.rec
        ao += m.andone
        counts[gm["game_id"]] = len(m.possessions)
    R = pd.DataFrame(rec)
    A = pd.DataFrame(ao)
    # identity check against the served possession layer
    pv = pd.read_parquet(ROOT / f"data/processed/possessions_v2/possessions_{season}.parquet")
    pc = pv.groupby("game_id").size()
    mine = pd.Series(counts)
    common = pc.index.intersection(mine.index)
    chk = {"games": int(len(common)), "games_count_mismatch": int((pc.loc[common] != mine.loc[common]).sum()),
           "unknown_v2": int((pv["terminal_event"] == "unknown").sum()), "unknown_tap": int(len(R))}
    return R, A, chk


if __name__ == "__main__":
    summ = {}
    for s in SEASONS:
        R, A, chk = run_season(s)
        R.to_parquet(OUT / f"unknown_poss_{s}.parquet")
        A.to_parquet(OUT / f"andone_{s}.parquet")
        summ[s] = chk
        print(s, chk, R["trigger"].value_counts().to_dict(), flush=True)
    (OUT / "unknown_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
