"""
build_engine_inputs_v3_tag_v1.py -- from RETRAINED ARTIFACTS (and an optional team-rate table) to a tagged
engine-inputs set `engine_v3_<tag>`. Lane G, 2026-09-30. New file; nothing existing is edited.

    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_tag_v1.py --tag E3 \
        --team-rate-table data/processed/team_rate_features_E3_v4.parquet \
        --po-artifacts  data/processed/models/engine_s1_teamrate_v1/team_rate_features_E3_v4 \
        --fg-artifacts  data/processed/models/fg_make/round5_par/team_rate_features_E3_v4/B1 \
        --rb-artifacts  <dir holding manifest.json + seg_*.joblib>

BASE. The base is the assembled v3 replay (`engine_v3`, `build_engine_inputs_v3_replay.py assemble`): it already is
the live path replayed date by date, so backtest and live stay one code path. This builder never recomputes an
as-of feature; it (1) substitutes team-rate columns from a table and (2) builds a serving overlay for the
retrained artifacts. Output: `data/processed/models/engine_v3_<tag>/` (never an existing directory).

1. TABLE (optional; default none = the served expanding-mean features, output bit-identical to the base).
   `--team-rate-table T` is applied through `cbb_sim.team_rate_adapter.apply(frame, T, submodel, fold)` on one frame
   per sub-model built from the base arrays (rows = game x side [x shot class]), and written back:
     possession_outcome  8 style columns (x100) in `team_static` AND in the round-2 event block
     fg_make             off_make_c / def_allow_c per shot class (6 columns of `team_static`)
     rebound             off_oreb_c / opp_def_dreb_c (2 columns of `team_static`)
   plus the ONE slot family that is a function of the team rate: `shooter_shrunk_dev_c__{rim,jump2,three}` (round-4
   B1's shooter shrinkage, built on off_make_raw = off_make_c + league rate). Because
   shrunk = (m r + mk)/(m + att) and r = c + lg, mk/att = smc + lg (smc = shooter_make_c centred):
       shrunk_dev = att (smc - c) / (m + att)        (the league rate cancels; 0 when att = 0)
   with `m` read from the fg artifact directory's `m_fitted.json` (the retrained m when fg was retrained).
   `--table-po/--table-fg/--table-rb` give a sub-model its own table (Topp = E3opp for PO only); `--no-table-for po,fg,rb`
   keeps those sub-models' served columns (mixed stack). `--team-rate-missing` is the
   adapter's policy (default raise). `--variance-table` is only RECORDED (the draw builder consumes it).
2. ARTIFACTS (default = served). They never change an input array. They are what the engine must LOAD, and the
   engine reads them from fixed paths, so the builder writes `overlay/`: a tree that mirrors the repo paths the
   adapters read, to be bind-mounted over the served paths (docker -v) or copied onto a scratch checkout:
     overlay/data/processed/models/engine/event_round2_s1_F2_2025/   index.json, *.joblib, team_block.npz (THIS tag's block)
     overlay/data/processed/models/fg_make/round4/B1/                 manifest_*.json, *.joblib      (only if --fg-artifacts)
     overlay/data/processed/models/rebound/s1_confirm/S1_weekly/F2/   manifest.json, seg_*.joblib    (only if --rb-artifacts)
   With defaults the event dir is still written, because the served adapter reads the round-2 block from there and
   the v3 block is what S0 needs. `docker_mounts.txt` lists the -v flags. No ENGINE_* flag changes.
Report: `builder_report.json` (what changed, per family, with counts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim import team_rate_adapter as TRA  # noqa: E402

MODELS = ROOT / "data/processed/models"
BASE_DEFAULT = MODELS / "engine_v3"
SERVED_EVENT = MODELS / "engine/event_round2_s1_F2_2025"
SERVED_FG = MODELS / "fg_make/round4/B1"
SERVED_RB = MODELS / "rebound/s1_confirm/S1_weekly/F2"
PO_COLS = ["off_3pa_c", "off_rim_c", "off_tov_c", "off_ftr_c",
           "opp_def_3pa_c", "opp_def_rim_c", "opp_def_tov_c", "opp_def_ftr_c"]
FG_CLS = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}
TAG = "F2_2025"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _frames(games: pd.DataFrame, season: int):
    g = games.reset_index(drop=True)
    n = len(g)
    base = {"season": np.repeat(season, 2 * n).astype("int64"),
            "game_id": np.repeat(g["game_id"].to_numpy(), 2).astype("int64")}
    home, away = g["home_team_id"].to_numpy(), g["away_team_id"].to_numpy()
    off = np.empty(2 * n, dtype="int64"); dfn = np.empty(2 * n, dtype="int64")
    off[0::2], off[1::2] = home, away
    dfn[0::2], dfn[1::2] = away, home
    return base, off, dfn


def substitute(arrs: dict, eb: np.ndarray, names: dict, games: pd.DataFrame, tables: dict, fold: str,
               which: set, missing: str, m_fg: dict | None):
    """Returns (team_static, event_block, slot_static, report); inputs are not modified."""
    tn = names["team_names"]; sn = names["slot_names"]
    ts = arrs["team_static"].copy(); eb2 = eb.copy(); sl = arrs["slot_static"].copy()
    n = len(games)
    base, off, dfn = _frames(games, int(games["season"].iloc[0]))
    rep: dict = {}

    def flat(a, j):            # (G,2) column j of a (G,2,F) -> flat rows in (game, side) order
        return a[:, :, j].reshape(-1)

    def back(a, j, v):
        a[:, :, j] = np.asarray(v).reshape(n, 2).astype(a.dtype)

    if "po" in which:
        fr = pd.DataFrame({**base, "offense_team_id": off, "defense_team_id": dfn})
        for c in PO_COLS:
            fr[c] = flat(eb, tn[c]).astype("float32")      # event block cols 0..15 are team_static cols 0..15
        out = TRA.apply(fr, tables["po"], "possession_outcome", fold=fold, missing=missing)
        for c in PO_COLS:
            j = tn[c]
            back(ts, j, out[c].to_numpy()); back(eb2, j, out[c].to_numpy())     # event block cols 0..15 == team_static 0..15
        rep["possession_outcome"] = {"columns": PO_COLS, "rows_kept_served": out.attrs["team_rate_adapter"]["rows_kept_served"]}
    if "rb" in which:
        fr = pd.DataFrame({**base, "off_team_id": off, "def_team_id": dfn})
        for c in ("off_oreb_c", "opp_def_dreb_c"):
            fr[c] = flat(ts, tn[c]).astype("float32")
        out = TRA.apply(fr, tables["rb"], "rebound", fold=fold, missing=missing)
        for c in ("off_oreb_c", "opp_def_dreb_c"):
            back(ts, tn[c], out[c].to_numpy())
        rep["rebound"] = {"columns": ["off_oreb_c", "opp_def_dreb_c"],
                          "rows_kept_served": out.attrs["team_rate_adapter"]["rows_kept_served"]}
    if "fg" in which:
        rows = []
        for cls, key in FG_CLS.items():
            fr = pd.DataFrame({**base, "off_team_id": off, "def_team_id": dfn, "shot_class": cls})
            fr["off_make_c"] = flat(ts, tn[f"off_make_c__{key}"]).astype("float32")
            fr["def_allow_c"] = flat(ts, tn[f"def_allow_c__{key}"]).astype("float32")
            rows.append(fr)
        allf = pd.concat(rows, ignore_index=True)
        out = TRA.apply(allf, tables["fg"], "fg_make", fold=fold, missing=missing)
        for i, (cls, key) in enumerate(FG_CLS.items()):
            seg = out.iloc[i * 2 * n:(i + 1) * 2 * n]
            back(ts, tn[f"off_make_c__{key}"], seg["off_make_c"].to_numpy())
            back(ts, tn[f"def_allow_c__{key}"], seg["def_allow_c"].to_numpy())
            if m_fg is not None:
                m = float(m_fg[cls]["m"] if isinstance(m_fg[cls], dict) else m_fg[cls])
                att = sl[..., sn[f"shooter_att_c__{key}"]].astype("float64")
                smc = sl[..., sn[f"shooter_make_c__{key}"]].astype("float64")
                c = ts[:, :, tn[f"off_make_c__{key}"]].astype("float64")[:, :, None]
                dev = np.where(att > 0, att * (smc - c) / (m + att), 0.0)
                sl[..., sn[f"shooter_shrunk_dev_c__{key}"]] = dev.astype(np.float32)
        rep["fg_make"] = {"columns": [f"{p}__{k}" for k in FG_CLS.values() for p in ("off_make_c", "def_allow_c")]
                          + [f"shooter_shrunk_dev_c__{k}" for k in FG_CLS.values()],
                          "rows_kept_served": out.attrs["team_rate_adapter"]["rows_kept_served"]}
    return ts, eb2, sl, rep


def build_overlay(out: Path, eb: np.ndarray, po_dir: Path, fg_dir: Path | None, rb_dir: Path | None,
                  fg_m: Path | None) -> dict:
    ov = out / "overlay"
    rep = {}
    ed = ov / "data/processed/models/engine/event_round2_s1_F2_2025"
    idx = json.loads((po_dir / "index.json").read_text(encoding="utf-8"))
    for pop in idx["populations"].values():
        for sg in pop["segments"]:
            link_or_copy(po_dir / sg["file"], ed / sg["file"])
    (ed / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")
    np.savez_compressed(ed / "team_block.npz", team_block=eb.astype(np.float32))
    rep["event"] = {"source": str(po_dir), "files": len(list(ed.iterdir()))}
    mounts = [f"-v $PWD/{ed.relative_to(ROOT).as_posix()}:/app/data/processed/models/engine/event_round2_s1_F2_2025:ro"]
    if fg_dir is not None:
        d = ov / "data/processed/models/fg_make/round4/B1"
        for mp in sorted(fg_dir.glob("manifest_*.json")):
            obj = json.loads(mp.read_text(encoding="utf-8"))
            for a in obj["artifacts"]:
                link_or_copy(fg_dir / a["path"], d / a["path"])
            (d / mp.name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
        rep["fg_make"] = {"source": str(fg_dir), "files": len(list(d.iterdir()))}
        mounts.append(f"-v $PWD/{d.relative_to(ROOT).as_posix()}:/app/data/processed/models/fg_make/round4/B1:ro")
    if rb_dir is not None:
        d = ov / "data/processed/models/rebound/s1_confirm/S1_weekly/F2"
        obj = json.loads((rb_dir / "manifest.json").read_text(encoding="utf-8"))
        for a in obj["artifacts"]:
            link_or_copy(rb_dir / a["path"], d / a["path"])
        (d / "manifest.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
        rep["rebound"] = {"source": str(rb_dir), "files": len(list(d.iterdir()))}
        mounts.append(f"-v $PWD/{d.relative_to(ROOT).as_posix()}:/app/data/processed/models/rebound/s1_confirm/S1_weekly/F2:ro")
    mounts.append(f"-v $PWD/{out.relative_to(ROOT).as_posix()}:/app/{out.relative_to(ROOT).as_posix()}:ro")   # the input dir itself
    (out / "docker_mounts.txt").write_text("\n".join(mounts) + "\n", encoding="utf-8")
    rep["docker_mounts"] = mounts
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-dir", type=Path, default=BASE_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <base parent>/engine_v3_<tag>")
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--team-rate-table", type=Path, default=None)
    ap.add_argument("--table-po", type=Path, default=None, help="table for possession_outcome only (e.g. E3opp = Topp); default --team-rate-table")
    ap.add_argument("--table-fg", type=Path, default=None)
    ap.add_argument("--table-rb", type=Path, default=None)
    ap.add_argument("--no-table-for", default="", help="comma list of po,fg,rb kept on served features")
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise")
    ap.add_argument("--variance-table", type=Path, default=None, help="recorded only (draw builder input)")
    ap.add_argument("--po-artifacts", type=Path, default=SERVED_EVENT,
                    help="event artifact dir (index.json + joblibs) or its parent")
    ap.add_argument("--fg-artifacts", type=Path, default=None, help="fg_make arm dir (manifest_*.json); default served")
    ap.add_argument("--fg-m", type=Path, default=None, help="m_fitted.json; default <fg-artifacts>/../m_fitted.json or served")
    ap.add_argument("--rb-artifacts", type=Path, default=None, help="rebound dir (manifest.json); default served")
    a = ap.parse_args()
    base = a.base_dir
    out = a.out_dir or base.parent / f"engine_v3_{a.tag}"
    if out.exists() or out.resolve() == BASE_DEFAULT.resolve() or out.resolve() == base.resolve():
        raise SystemExit(f"{out} exists or is the base: never overwrite")
    z = dict(np.load(base / f"arrays_{TAG}.npz"))
    names = json.loads((base / f"names_{TAG}.json").read_text(encoding="utf-8"))
    games = pd.read_parquet(base / f"games_{TAG}.parquet")
    eb = np.load(base / f"event_block_{TAG}.npz")["team_block"]
    po = a.po_artifacts
    if not (po / "index.json").exists() and (po / "event_round2_s1_F2_2025/index.json").exists():
        po = po / "event_round2_s1_F2_2025"
    fg = a.fg_artifacts
    fg_m_path = a.fg_m or ((fg.parent / "m_fitted.json") if fg is not None else MODELS / "fg_make/round4/m_fitted.json")
    m_fg = json.loads(fg_m_path.read_text(encoding="utf-8")) if fg_m_path.exists() else None
    skip = {x for x in a.no_table_for.split(",") if x}
    report = {"tag": a.tag, "base": str(base), "base_sha256": {f: sha(base / f) for f in (
        f"arrays_{TAG}.npz", f"games_{TAG}.parquet", f"names_{TAG}.json", f"event_block_{TAG}.npz")},
        "team_rate_table": None, "variance_table": str(a.variance_table) if a.variance_table else None,
        "artifacts": {"po": str(po), "fg": str(fg) if fg else "served", "rb": str(a.rb_artifacts) if a.rb_artifacts else "served",
                      "fg_m": str(fg_m_path)}}
    out.mkdir(parents=True)
    changed = False
    arrs_out = z
    eb_out = eb
    tables = {"po": a.table_po or a.team_rate_table, "fg": a.table_fg or a.team_rate_table,
              "rb": a.table_rb or a.team_rate_table}
    if any(v is not None for v in tables.values()):
        which = {k for k, v in tables.items() if v is not None} - skip
        ts, eb2, sl, rep = substitute(z, eb, names, games, tables, a.fold, which, a.team_rate_missing,
                                      m_fg if "fg" in which else None)
        arrs_out = dict(z); arrs_out["team_static"] = ts; arrs_out["slot_static"] = sl; eb_out = eb2
        changed = True
        tn, sn = names["team_names"], names["slot_names"]
        fam = {"team_static": [c for c, j in tn.items() if not np.array_equal(ts[:, :, j], z["team_static"][:, :, j])],
               "slot_static": [c for c, j in sn.items() if not np.array_equal(sl[..., j], z["slot_static"][..., j])],
               "event_block_cols": [i for i in range(eb.shape[2]) if not np.array_equal(eb2[:, :, i], eb[:, :, i])]}
        report["team_rate_table"] = {"paths": {k: str(v) for k, v in tables.items() if v is not None},
                                     "sha256": {k: sha(v) for k, v in tables.items() if v is not None},
                                     "skipped_submodels": sorted(skip), "substitution": rep, "changed_families": fam,
                                     "unchanged_arrays": [k for k in z if k not in ("team_static", "slot_static")]}
    for f in (f"games_{TAG}.parquet", f"names_{TAG}.json"):
        shutil.copyfile(base / f, out / f)
    if changed:
        np.savez_compressed(out / f"arrays_{TAG}.npz", **arrs_out)
        np.savez_compressed(out / f"event_block_{TAG}.npz", team_block=eb_out)
    else:
        for f in (f"arrays_{TAG}.npz", f"event_block_{TAG}.npz"):
            shutil.copyfile(base / f, out / f)
    report["overlay"] = build_overlay(out, eb_out, po, fg, a.rb_artifacts, fg_m_path)
    report["outputs_sha256"] = {f: sha(out / f) for f in (f"arrays_{TAG}.npz", f"games_{TAG}.parquet",
                                                          f"names_{TAG}.json", f"event_block_{TAG}.npz")}
    (out / "builder_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tag", "team_rate_table", "overlay")}, indent=1, default=str)[:3000])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
