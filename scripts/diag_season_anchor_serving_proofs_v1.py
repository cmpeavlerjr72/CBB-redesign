"""diag_season_anchor_serving_proofs_v1.py -- lane M proofs (c) and (d) for ENGINE_SEASON_ANCHOR (2026-09-30).

(d) `d`: the per-game offsets file equals `cbb_sim.season_anchor`'s own league level for every fold-2 date:
      d1  anchor_O on the design rows ONLY (no engine rows), per date present in the design;
      d2  the module's serving form `asof_level_live` for EVERY engine game date (independent code path);
      d3  the offsets the anchored TRAINER actually fitted with (lane C's add_anchor_columns / `_a5_off`), per row.
(c) `c_po_smoke`: lane C's full-size one-refit anchored `first` model (smoke pkl) served through
      `EventAdapter.predict` with the offsets file vs the pkl's own offline predictions, same design rows (state and
      team values of those possessions).
    `c_po_tiny`: the complete anchored artifact set written by train_possession_outcome_s1_par_anchor_artifacts_v1.py
      (tiny test hooks) loaded through the REAL `EventAdapter.load` + `season_anchor_serving.attach`, vs the model's
      own offline formula (`season_anchor.predict_proba_with_offset` with the trainer's offsets), per S1 segment.
    `c_rb`: the anchored rebound smoke artifact (train_rebound_v3_par_anchor_artifacts_v1.py, one refit) loaded
      through the REAL `ReboundAdapter._load_dated` + `attach`, vs the worker's own offline predictions `p`.
Writes results/lanem/proofs_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim import season_anchor as SA  # noqa: E402
from cbb_sim.engine import adapters as AD  # noqa: E402
from cbb_sim.engine import season_anchor_serving as SAS  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

OUT = ROOT / "results/lanem"
PO_DESIGN = ROOT / "data/processed/models/possession_outcome/round2/design.parquet"
RB_DESIGN = ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
TRAIN, SEASON = [2022, 2023, 2024], 2025


def _inp(d):
    return EngineInputs.load(d, "F2_2025")


def _gpos(inp):
    return {int(g): i for i, g in enumerate(inp.games["game_id"].to_numpy())}


def _side(inp, gidx, off_team):
    return np.where(inp.games["home_team_id"].to_numpy()[gidx] == off_team, 0, 1).astype(np.int64)


def _pick_games(rows: pd.DataFrame, gpos: dict, n: int) -> list:
    g = rows[["game_id", "game_date"]].drop_duplicates("game_id")
    g = g[g["game_id"].map(lambda x: int(x) in gpos)].sort_values(["game_date", "game_id"])
    pos = np.linspace(0, len(g) - 1, n).round().astype(int)
    return [int(x) for x in g["game_id"].to_numpy()[pos]]


def _state(rows: pd.DataFrame, feats) -> np.ndarray:
    st = np.full((len(rows), len(AD.STATE_COLS)), np.nan)
    for f in feats:
        if f in AD.STATE_INDEX:
            st[:, AD.STATE_INDEX[f]] = rows[f].to_numpy(dtype="float64")
    return st


# ----------------------------------------------------------------------------------------------- (d)
def proof_d(a) -> dict:
    import train_possession_outcome_s1_par_anchor_v1 as A
    import train_rebound_v3_par_anchor_v1 as CA
    import train_rebound_v3_round3 as R3
    from build_engine_anchor_offsets_v1 import engine_games

    z = np.load(a.offsets)
    games = engine_games(Path(a.input_dir), "F2", SEASON)
    assert np.array_equal(z["game_ids"], games["game_id"].to_numpy())
    gdate = games["date"].to_numpy().astype("datetime64[D]")
    gpos = {int(g): i for i, g in enumerate(games["game_id"])}
    rep = {"offsets": a.offsets, "n_games": int(len(games)), "n_engine_dates": int(len(np.unique(gdate)))}

    def compare(name, off_eng, season, date, num, den, train_seasons, kind, trainer_rows=None):
        a0 = SA.anchor_O(season, date, num, den, train_seasons, kind)
        off_rows = a0.offset()
        s = np.asarray(season); dd = pd.to_datetime(date).values.astype("datetime64[D]")
        te = s == SEASON
        per_date = pd.DataFrame(off_rows[te]).assign(date=dd[te]).drop_duplicates("date").set_index("date")
        hit = np.isin(gdate, per_date.index.values)
        want = per_date.loc[gdate[hit]].to_numpy()
        e2 = off_eng.reshape(len(games), -1)
        d1 = float(np.max(np.abs(e2[hit] - want)))
        # d2: serving form, every engine date
        Lend = SA.season_end_levels(s, num, den)
        prior = Lend[SEASON - 1]
        nb = pd.DataFrame(num[te]).groupby(dd[te]).sum()
        db = pd.Series(den[te]).groupby(dd[te]).sum()
        lev = {}
        for t in np.unique(gdate):
            lev[t] = SA.asof_level_live(nb, db, pd.Timestamp(t), prior)
        L2 = np.vstack([lev[t] for t in gdate])
        off2 = SA.link(L2, kind) - SA.link(a0.Lbar[None, :], kind)
        d2 = float(np.max(np.abs(e2 - off2)))
        r = {"d1_design_dates_matched": int(len(np.unique(gdate[hit]))), "d1_games": int(hit.sum()),
             "d1_max_abs_offset_diff": d1, "d2_all_engine_dates": int(len(lev)), "d2_max_abs_offset_diff": d2,
             "Lbar": a0.Lbar.tolist(), "day0_level": prior.tolist(),
             "offset_range": [float(e2.min()), float(e2.max())]}
        if trainer_rows is not None:
            gid, toff = trainer_rows
            ok = np.array([int(g) in gpos for g in gid])
            gi = np.array([gpos[int(g)] for g in gid[ok]])
            r["d3_trainer_rows"] = int(ok.sum())
            r["d3_max_abs_offset_diff"] = float(np.max(np.abs(e2[gi] - toff[ok].reshape(ok.sum(), -1))))
        rep[name] = r
        print(name, json.dumps(r)[:600], flush=True)

    if "po_first" in z.files:
        d = pd.read_parquet(a.po_design)
        d["game_date"] = pd.to_datetime(d["game_date"])
        m = (d["population"] == "first") & d["season"].isin([*TRAIN, SEASON])
        sub = d.loc[m]
        num, den = SA.po_inputs(sub, len(PO.CLASSES))
        dA, _ = A.add_anchor_columns(d, "F2", SEASON)
        t = dA[(dA["population"] == "first") & (dA["season"] == SEASON)]
        compare("po_first", z["po_first"], sub["season"].to_numpy(), sub["game_date"].to_numpy(), num, den, TRAIN,
                "multi", (t["game_id"].to_numpy(), t[A.OFF_COLS].to_numpy()))
        del d, dA
    if "rb_oreb" in z.files:
        d = pd.read_parquet(a.rb_design)
        tr, te = RB.fold_slices(d, "F2")
        both = pd.concat([tr[["season", "game_date", "y"]], te[["season", "game_date", "y"]]], ignore_index=True)
        num, den = SA.rebound_inputs(both, RB.CLASS_INDEX["OREB"], RB.CLASS_INDEX["DEAD"])
        CA.install_anchor_O()
        _, te2, _ = R3.add_fold_columns(tr, te)
        compare("rb_oreb", z["rb_oreb"], both["season"].to_numpy(), both["game_date"].to_numpy(), num, den,
                sorted(int(x) for x in tr["season"].unique()), "binary",
                (te2["game_id"].to_numpy(), te2["_a5_off"].to_numpy()))
    return rep


# ----------------------------------------------------------------------------------------------- (c) po
def _po_rows(design_path, table, missing):
    d = pd.read_parquet(design_path)
    if table:
        from cbb_sim.team_rate_adapter import apply
        d = apply(d, table, "possession_outcome", fold="F2", missing=missing)
    d["game_date"] = pd.to_datetime(d["game_date"])
    return d


def proof_c_po_smoke(a) -> dict:
    import build_engine_event_round2 as B
    inp = _inp(a.input_dir)
    gpos = _gpos(inp)
    ck = pickle.load(open(a.smoke_pkl, "rb"))
    d = _po_rows(PO_DESIGN, a.table, "keep_served")
    te = d[(d["season"] == SEASON) & (d["population"] == "first")].reset_index(drop=True)
    assert len(te) == len(ck["pred"]), (len(te), ck["pred"].shape)
    ev = AD.EventAdapter.load(inp, "round2_s1", "F2", SEASON)
    feats = list(ev.plan_first.features)
    arm = PO.LgbmArm(0); arm.clf_ = ck["model"]
    ev.models_first = tuple(arm for _ in ev.models_first)
    ev.anchor_marks = tuple(True for _ in ev.anchor_marks)
    reb = SimpleNamespace(manifest=None, anchor_marks=(), models_by_seg=(), anchor_oreb=None)
    src = SAS.attach(inp, ev, reb, a.offsets)
    games = _pick_games(te, gpos, a.n_games)
    r = np.flatnonzero(te["game_id"].isin(games).to_numpy())
    rows = te.iloc[r]
    gidx = rows["game_id"].map(lambda g: gpos[int(g)]).to_numpy()
    off = _side(inp, gidx, rows["offense_team_id"].to_numpy())
    tb = ev.team_block.copy()
    tf = rows.drop_duplicates(["game_id", "offense_team_id"])
    const = rows.groupby(["game_id", "offense_team_id"])[list(B.TEAM_COLS)].nunique().max().max()
    gi_t = tf["game_id"].map(lambda g: gpos[int(g)]).to_numpy()
    tb[gi_t, _side(inp, gi_t, tf["offense_team_id"].to_numpy())] = tf[list(B.TEAM_COLS)].to_numpy(dtype=np.float32)
    ev.team_block = tb
    st = _state(rows, feats)
    got = ev.predict(np.zeros((len(rows), 1)), st, np.ones(len(rows), bool), gidx, off)
    want = ck["pred"][r]
    ev.anchor_first = None
    unanch = ev.predict(np.zeros((len(rows), 1)), st, np.ones(len(rows), bool), gidx, off)
    rep = {"smoke_pkl": a.smoke_pkl, "offsets": a.offsets, "games": games, "n_rows": int(len(rows)),
           "dates": sorted(set(rows["game_date"].dt.date.astype(str))), "team_cols_constant_within_team_game": int(const) == 1,
           "max_abs_diff_engine_vs_offline": float(np.max(np.abs(got - want))),
           "max_abs_diff_without_offset": float(np.max(np.abs(unanch - want))),
           "mean_abs_shift_by_offset_pp": (np.abs(got - unanch).mean(0) * 100).round(4).tolist(), "attach": src}
    print(json.dumps(rep, default=str)[:1500], flush=True)
    return rep


def proof_c_po_tiny(a) -> dict:
    import train_possession_outcome_s1_par_anchor_v1 as A
    inp = _inp(a.input_dir)
    gpos = _gpos(inp)
    art = Path(a.artifact_dir)
    AD.ENGINE_DIR = art.parent                     # EventAdapter.load reads ENGINE_DIR/event_round2_s1_F2_2025
    ev = AD.EventAdapter.load(inp, "round2_s1", "F2", SEASON)
    assert all(ev.anchor_marks), ev.anchor_marks
    reb = SimpleNamespace(manifest=None, anchor_marks=(), models_by_seg=(), anchor_oreb=None)
    try:
        SAS.refuse_unserved_anchor(ev, reb)
        refused = False
    except ValueError:
        refused = True
    src = SAS.attach(inp, ev, reb, a.offsets)
    d = pd.read_parquet(a.design)                  # design_sample.parquet (overlay + sample already applied)
    d["game_date"] = pd.to_datetime(d["game_date"])
    dA, _ = A.add_anchor_columns(d, "F2", SEASON)
    te = dA[(dA["season"] == SEASON) & (dA["population"] == "first")].reset_index(drop=True)
    te = te[te["game_id"].map(lambda g: int(g) in gpos)].reset_index(drop=True)
    feats = list(ev.plan_first.features)
    gidx = te["game_id"].map(lambda g: gpos[int(g)]).to_numpy()
    off = _side(inp, gidx, te["offense_team_id"].to_numpy())
    blk = ev.team_block[gidx, off]
    import build_engine_event_round2 as B
    same_block = (blk == te[list(B.TEAM_COLS)].to_numpy(dtype=np.float32)).all(axis=1)
    te, gidx, off = te[same_block].reset_index(drop=True), gidx[same_block], off[same_block]
    st = _state(te, feats)
    got = ev.predict(np.zeros((len(te), 1)), st, np.ones(len(te), bool), gidx, off)
    # offline: the model of each row's S1 segment, the trainer's own offsets, lane C's predict formula
    seg = ev.manifests["first"].segments(gidx)
    want = np.empty_like(got)
    X = np.ascontiguousarray(te[feats].to_numpy(dtype="float32"))
    for k in np.unique(seg):
        m = seg == k
        init = SA.lgbm_init_score(te.loc[m, A.OFF_COLS].to_numpy(), 6, list(range(6)))
        want[m] = SA.predict_proba_with_offset(ev.models_first[k].clf_, X[m], init)
    rep = {"artifact_dir": str(art), "offsets": a.offsets, "mode_off_refused": refused,
           "n_rows": int(len(te)), "n_games": int(te["game_id"].nunique()), "n_rows_block_mismatch_dropped": int((~same_block).sum()),
           "segments_used": [int(x) for x in np.unique(seg)],
           "max_abs_diff_engine_vs_offline": float(np.max(np.abs(got - want))), "attach": src}
    print(json.dumps(rep, default=str)[:1500], flush=True)
    return rep


# ----------------------------------------------------------------------------------------------- (c) rb
def proof_c_rb(a) -> dict:
    import train_rebound_v3_par_anchor_v1 as CA
    import train_rebound_v3_round3 as R3
    from cbb_sim.team_rate_adapter import apply
    inp = _inp(a.input_dir)
    gpos = _gpos(inp)
    art = Path(a.artifact_dir)
    reb = AD.ReboundAdapter._load_dated(inp, "F2", "s1_weekly", art / "manifest.json")
    assert all(reb.anchor_marks)
    ev = SimpleNamespace(mode="round2_s1", anchor_marks=(), models_first=(), anchor_first=None)
    src = SAS.attach(inp, ev, reb, a.offsets)
    ck = pickle.load(open(a.ckpt, "rb"))
    d = pd.read_parquet(RB_DESIGN)
    d = apply(d, a.table, "rebound", fold="F2", missing="raise")
    tr, te = RB.fold_slices(d, "F2")
    CA.install_anchor_O()
    _, te, _ = R3.add_fold_columns(tr, te)
    rows_all = te.loc[ck["rows"]["index"]]
    assert np.array_equal(rows_all["game_id"].to_numpy(), ck["rows"]["game_id"])
    a5_same = bool(np.array_equal(rows_all["_a5_off"].to_numpy(), ck["rows"]["a5_off"]))
    games = _pick_games(rows_all.assign(game_date=pd.to_datetime(rows_all["game_date"])), gpos, a.n_games)
    pick = rows_all["game_id"].isin(games).to_numpy()
    rows = rows_all[pick]
    want = ck["p"][pick]
    gidx = rows["game_id"].map(lambda g: gpos[int(g)]).to_numpy()
    feats = list(reb.plan.features)
    team = np.full((len(rows), len(inp.team_names)), np.nan)
    for s_, d_ in zip(reb.plan.team_src, reb.plan.team_dst):
        team[:, s_] = rows[feats[d_]].to_numpy(dtype="float64")
    st = _state(rows, feats)
    got = reb.predict(team, st, gidx)
    reb.anchor_oreb = None
    unanch = reb.predict(team, st, gidx)
    rep = {"artifact_dir": str(art), "offsets": a.offsets, "games": games, "n_rows": int(len(rows)),
           "trainer_a5_off_equals_design_rebuild": a5_same,
           "max_abs_diff_engine_vs_offline": float(np.max(np.abs(got - want))),
           "max_abs_diff_without_offset": float(np.max(np.abs(unanch - want))),
           "mean_oreb_shift_by_offset_pp": float((got[:, 0] - unanch[:, 0]).mean() * 100), "attach": src}
    print(json.dumps(rep, default=str)[:1500], flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proof", choices=["d", "c_po_smoke", "c_po_tiny", "c_rb"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--input-dir", default="data/processed/models/engine")
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--po-design", default=str(PO_DESIGN))
    ap.add_argument("--rb-design", default=str(RB_DESIGN))
    ap.add_argument("--design", default="")
    ap.add_argument("--table", default="")
    ap.add_argument("--smoke-pkl", default="")
    ap.add_argument("--artifact-dir", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n-games", type=int, default=8)
    a = ap.parse_args()
    fn = {"d": proof_d, "c_po_smoke": proof_c_po_smoke, "c_po_tiny": proof_c_po_tiny, "c_rb": proof_c_rb}[a.proof]
    rep = fn(a)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"proofs_{a.name}.json"
    if p.exists():
        raise SystemExit(f"{p} exists: never overwrite")
    p.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
