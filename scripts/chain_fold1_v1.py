#!/usr/bin/env python
"""chain_fold1_v1.py -- fold-1 (train through 2022-23, test 2023-24) engine inputs and artifacts of every served
sub-model plus the adopted loop-level tables, then the paired read served stack v2 (V2) vs SERVED_V1 (V1).
Lane D, 2026-10-01. Pre-registration: docs/models/engine/experiments.md section 2 (committed a85707f).

Every stage calls the served trainer (or a fold-parameterised sibling proven identical on fold 2) with fold F1. Every
output is a NEW versioned path; nothing served or fold-2 is written:
  data/processed/models/engine_f1/        v1 + v2 backtest inputs F1_2024 + the three static engine joblibs
  data/processed/models/engine_live_f1/   per-date live-replay staging     results/engine_v3_replay_f1/ per-date reports
  data/processed/models/engine_v3_f1/     v3 (live replay) inputs F1_2024 + event block
  data/processed/models/fold1_v1/         clock/, po/, fg/, rb/, ft/, rot/, loop/, engine_scratch/, overrides_V{1,2}.json
HF bulk key proposal: `fold1_v1` -> those four model dirs (today they ride `model_artifacts`, gitignored).

Stages (each writes data/processed/models/fold1_v1/.done/<stage>.json; a rerun skips finished stages):
  inputs_v1 inputs_v2 inputs_v3 clock po fg rb ft loop shot_block engine_dir overrides gate
  ROTATION (stated deviation): no fold-1 rotation fit exists; the served S1 windows train through 2024-03 and the
  manifest guard refuses them for 2023-24 games, so both arms run ENGINE_ROTATION_SCHEME=static, whose
  rotation_fit.json is fitted on 2023-24 (the fold-1 TEST season): a leak common to both arms of the paired read.

    .venv/Scripts/python.exe scripts/chain_fold1_v1.py --cores 5 [--stages a,b] [--dry-run]
    # gate (local tap):  --gate-mode sample --gate-seeds 25 --gate-offsets 0,1000   (sample: every 11th 2024 verified game)
    # gate (box):        --gate-mode full --gate-seeds 200 --gate-offsets 0,1000,2000,3000,4000 --gate-workers 90
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
M = ROOT / "data/processed/models"
F1 = M / "fold1_v1"
DONE = F1 / ".done"
FOLD, SEASON = "F1", 2024
STAGES = ["inputs_v1", "inputs_v2", "inputs_v3", "clock", "po", "fg", "rb", "ft", "loop", "shot_block",
          "engine_dir", "overrides", "gate"]
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "LIGHTGBM_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
ROT = {"ENGINE_ROTATION_SCHEME": "static"}   # STATED DEVIATION: rotation_fit.json trains on 2023-24 (see doc)
SERVED_V1 = {**ROT, "ENGINE_CLOCK": "v5b_glat_pmean", "ENGINE_SHOT_BLOCK": "reference", "ENGINE_FOUL_JOINT": "reference",
             "ENGINE_SHARED_SHOOTING": "reference", "ENGINE_CHANCE_TIME": "reference", "ENGINE_EVENT_TEAM_BLOCK": "v1"}
ADOPTED = {**ROT, "ENGINE_CLOCK": "v5b_r6L2_glat_pmean", "ENGINE_SHOT_BLOCK": "K2_Ocell", "ENGINE_FOUL_JOINT": "R9ao3",
           "ENGINE_SHARED_SHOOTING": "G3", "ENGINE_CHANCE_TIME": "KD"}
SAMPLE = M / "fold1_v1/sample_F1_2024_every11.parquet"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(p) -> str:
    return Path(os.path.relpath(Path(p).resolve(), ROOT)).as_posix()


class Chain:
    def __init__(self, a):
        self.a = a
        self.log = F1 / "chain.log"

    def env(self, extra=None):
        e = dict(os.environ)
        for v in THREADS:
            e[v] = "1"
        e["PYTHONIOENCODING"] = "utf-8"
        e["CBB_TRUTH"] = "verified_v1"
        for k in list(e):
            if k.startswith("ENGINE_"):
                del e[k]                      # the arm's switches are set explicitly per run, never inherited
        e.update(extra or {})
        return e

    def run(self, stage, cmd, extra=None, bg=False):
        cmd = [str(c) for c in cmd]
        line = " ".join(cmd)
        print(f"[{now()}] {stage}: {line}", flush=True)
        if self.a.dry_run:
            return None
        F1.mkdir(parents=True, exist_ok=True)
        lf = open(F1 / f"{stage}.log", "a", encoding="utf-8")
        lf.write(f"\n=== {now()} {line}\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=self.env(extra), stdout=lf, stderr=subprocess.STDOUT)
        if bg:
            return p
        if p.wait() != 0:
            raise SystemExit(f"stage {stage} FAILED; see {F1 / (stage + '.log')}")
        return p

    def wait_all(self, stage, procs):
        bad = [p.args for p in procs if p is not None and p.wait() != 0]
        if bad:
            raise SystemExit(f"stage {stage}: {len(bad)} process(es) FAILED; see {F1 / (stage + '.log')}")

    # ---------------------------------------------------------------- stages
    def st_inputs_v1(self):
        self.run("inputs_v1", [PY, "scripts/build_engine_inputs.py", "--fold", FOLD, "--season", SEASON,
                               "--out-dir", "data/processed/models/engine_f1", "--version", "v1"])

    def st_inputs_v2(self):
        self.run("inputs_v2", [PY, "scripts/build_engine_inputs.py", "--fold", FOLD, "--season", SEASON,
                               "--out-dir", "data/processed/models/engine_f1", "--version", "v2"])

    def st_inputs_v3(self):
        n = max(1, self.a.cores)
        ps = [self.run("inputs_v3", [PY, "scripts/build_engine_inputs_v3_replay_fold_v1.py", "shard", "--shard", k,
                                     "--n-shards", n], bg=True) for k in range(n)]
        self.wait_all("inputs_v3", ps)
        errs = sorted((ROOT / "results/engine_v3_replay_f1/dates").glob("*.err"))
        if errs and not self.a.dry_run:
            raise SystemExit(f"inputs_v3: {len(errs)} dates failed, e.g. {errs[0]}")
        self.run("inputs_v3", [PY, "scripts/build_engine_inputs_v3_replay_fold_v1.py", "assemble"])

    def st_clock(self):
        for spec in ("v5b", "L2"):
            self.run("clock", [PY, "scripts/train_clock_s1_fold_v1.py", "--spec", spec, "--fold", FOLD,
                               "--out", rel(F1 / "clock")])

    def st_po(self):
        self.run("po", [PY, "scripts/train_possession_outcome_s1_par_v2.py", "--fold", FOLD, "--season", SEASON,
                        "--engine-dir", "data/processed/models/engine_f1", "--out-root", rel(F1 / "po"),
                        "--n-jobs", min(12, self.a.cores)])

    def st_fg(self):
        self.run("fg", [PY, "scripts/train_fg_make_v4_par_fold_v1.py", "--fold", FOLD, "--mode", "run", "--arms", "B1",
                        "--no-floor", "--n-jobs", min(18, self.a.cores), "--out-dir", rel(F1 / "fg")])

    def st_rb(self):
        self.run("rb", [PY, "scripts/train_rebound_v3_par_artifacts_v1.py", "--arms", "A0B0C0", "--stage", "2",
                        "--folds", FOLD, "--n-jobs", min(23, self.a.cores), "--out-dir", rel(F1 / "rb")])
        if not self.a.dry_run:
            mp = F1 / "rb/artifacts/s2_F1_A0B0C0_seed0/manifest.json"
            obj = json.loads(mp.read_text(encoding="utf-8"))
            obj["season"] = SEASON                          # the wrapper's manifest label is fold-2 literal
            mp.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def st_ft(self):
        self.run("ft", [PY, "scripts/train_free_throw_s1_fold_v1.py", "--fold", FOLD, "--out-root", rel(F1 / "ft")])

    def st_loop(self):
        self.run("loop", [PY, "scripts/build_loop_tables_fold_v1.py", "--fold", FOLD, "--out", rel(F1 / "loop")])

    def st_shot_block(self):
        self.run("shot_block", [PY, "scripts/build_engine_shot_block_lut_fold_v1.py", "--fold", FOLD,
                                "--input-dir", "data/processed/models/engine_f1", "--out-dir", rel(F1 / "loop")])

    def st_engine_dir(self):
        """Private ENGINE_DIR for the fold: the PO event dir with the v3-replay event team block (the fold-2 served
        block is the inputs-v3 replay block, PM ruling 2026-10-01), plus the three static engine joblibs."""
        eng = F1 / "engine_scratch"
        if self.a.dry_run:
            return
        if eng.exists():
            shutil.rmtree(eng)
        ev = eng / f"event_round2_s1_{FOLD}_{SEASON}"
        shutil.copytree(F1 / "po" / f"event_round2_s1_{FOLD}_{SEASON}", ev)
        import numpy as np
        import pandas as pd
        blk = np.load(M / f"engine_v3_f1/event_block_{FOLD}_{SEASON}.npz")["team_block"]
        z = dict(np.load(ev / "team_block.npz"))
        gi = pd.read_parquet(M / f"engine_v3_f1/games_{FOLD}_{SEASON}.parquet")["game_id"].to_numpy()
        ge = pd.read_parquet(ev.parent.parent / "po" / f"games_{FOLD}_{SEASON}.parquet")["game_id"].to_numpy() \
            if (F1 / "po" / f"games_{FOLD}_{SEASON}.parquet").exists() else gi
        if not np.array_equal(gi, ge) or z["team_block"].shape != blk.shape:
            raise SystemExit("engine_dir: the v3 replay block and the PO event dir disagree on game order / shape")
        z_old = z["team_block"]
        z["team_block"] = blk.astype(z_old.dtype)
        np.savez_compressed(ev / "team_block.npz", **z)
        for name in (f"fg_make_FGA_3_decision8_{FOLD}.joblib", f"free_throw_{FOLD}.joblib", f"rebound_{FOLD}.joblib",
                     f"games_{FOLD}_{SEASON}.parquet"):
            shutil.copy2(M / "engine_f1" / name, eng / name)
        diff = float(np.abs(z_old - blk).max())
        return {"event_block": "v3 replay", "max_abs_vs_backtest_block": diff}

    def st_overrides(self):
        common = {
            "adapters.ENGINE_DIR": rel(F1 / "engine_scratch"),
            "adapters.FG_DIR": rel(F1 / "fg_root"),
            "adapters.RB_S1_MANIFEST": rel(F1 / "rb/artifacts/s2_F1_A0B0C0_seed0/manifest.json"),
            "adapters.FT_S1_MANIFEST": rel(F1 / f"ft/S1_conf_aligned/{FOLD}/manifest.json"),
            "adapters.CK_DIR": rel(F1 / "clock"), "clock_adapter_v3.CK_DIR": rel(F1 / "clock"),
            "clock_adapter_v3.V5_PARAMS": rel(F1 / "clock/v5_bakeoff/v5_bakeoff_report.json"),
            "clock_adapter_v3.PARAMS_FOLD": {"str": FOLD},
        }
        loop = {
            "shot_block.LUT_DIR": rel(F1 / "loop"),
            "foul_joint.LUT_DIR": rel(F1 / "loop"),
            "foul_joint.ARMS": {"items": {"R8b": {"accrual": f"lut_acc_A2_{FOLD}", "trip": f"lut_trip_T2c_{FOLD}"}}},
            "foul_r9.LUT_DIR": rel(F1 / "loop"),
            "foul_r9.ARMS": {"items": {"R9ao3": ["R8b", f"lut_ao_AO3_{FOLD}"]}},
            "shared_shooting.PARAMS": rel(F1 / f"loop/params_G3_{FOLD}.json"),
            "chance_time.LUT3": {"path_str": f"data/processed/models/chance_time/{FOLD}/lut_v3.npz"},
        }
        if self.a.dry_run:
            return
        # fg_make's adapter reads FG_DIR/round4/B1/: expose the fold's B1 dir under that layout
        fr = F1 / "fg_root/round4"
        if fr.exists():
            shutil.rmtree(F1 / "fg_root")
        fr.mkdir(parents=True)
        shutil.copytree(F1 / "fg/B1", fr / "B1")
        (F1 / "overrides_V1.json").write_text(json.dumps(common, indent=1), encoding="utf-8")
        (F1 / "overrides_V2.json").write_text(json.dumps({**common, **loop}, indent=1), encoding="utf-8")
        return {"V1": rel(F1 / "overrides_V1.json"), "V2": rel(F1 / "overrides_V2.json")}

    def st_gate(self):
        if self.a.gate_mode == "sample" and not SAMPLE.exists() and not self.a.dry_run:
            import pandas as pd
            g = pd.read_parquet(M / f"engine_v3_f1/games_{FOLD}_{SEASON}.parquet")
            tr = pd.read_parquet(ROOT / "data/processed/truth/game_finals_v2.parquet")
            ok = g[g["game_id"].isin(tr.loc[tr["season"] == SEASON, "game_id"])].sort_values(["game_date", "game_id"])
            ok.iloc[::11][["game_id"]].assign(rank=range(len(ok.iloc[::11]))).to_parquet(SAMPLE, index=False)
        runs = []
        for arm, env, ov in (("V1", SERVED_V1, "overrides_V1.json"), ("V2", ADOPTED, "overrides_V2.json")):
            for off in [int(x) for x in self.a.gate_offsets.split(",") if x]:
                if arm == "V2" and off != 0 and not self.a.gate_v2_floors:
                    continue
                tag = f"f1c_{arm}_{self.a.gate_mode}_s{self.a.gate_seeds}_o{off}"
                if not (ROOT / "results/engine_v0" / tag / "run_meta.json").exists():
                    if self.a.gate_mode == "sample":
                        rargs = ["--sample-file", rel(SAMPLE), "--arm", "round2_s1", "--fold", FOLD, "--season", SEASON,
                                 "--input-dir", rel(M / "engine_v3_f1"), "--seeds", self.a.gate_seeds,
                                 "--seed-offset", off, "--workers", self.a.gate_workers, "--tag", tag,
                                 "--results-dir", "results/engine_v0"]
                        if arm == "V1":
                            rargs.append("--allow-drift")
                    else:
                        rargs = ["--fold", FOLD, "--season", SEASON, "--seeds", self.a.gate_seeds, "--seed-offset", off,
                                 "--workers", self.a.gate_workers, "--games-per-block", 60, "--seeds-per-block", 25,
                                 "--tag", tag, "--results-dir", "results/engine_v0", "--input-dir", rel(M / "engine_v3_f1")]
                    self.run("gate", [PY, "scripts/run_engine_overlay_v2.py", "--overrides", rel(F1 / ov),
                                      "--runner", self.a.gate_mode, "--", *rargs], extra=env)
                md = ROOT / "results/engine_v0/v3full_grade" / f"{tag}__verified.md"
                if not md.exists():
                    self.run("gate", [PY, "scripts/eval_gates.py", "--results", f"results/engine_v0/{tag}",
                                      "--season", SEASON, "--out", rel(md)])
                runs.append(tag)
        return {"runs": runs}

    # ------------------------------------------------------------------- run
    def go(self):
        only = [s for s in self.a.stages.split(",") if s] or STAGES
        DONE.mkdir(parents=True, exist_ok=True)
        for st in [s for s in STAGES if s in only]:
            mk = DONE / f"{st}.json"
            if mk.exists() and st not in self.a.redo.split(","):
                print(f"[{now()}] {st}: done (skipped)", flush=True)
                continue
            t0 = time.time()
            info = getattr(self, f"st_{st}")() or {}
            if not self.a.dry_run:
                mk.write_text(json.dumps({"finished": now(), "seconds": round(time.time() - t0, 1), **info},
                                         indent=1, default=str))
                print(f"[{now()}] {st}: DONE in {time.time() - t0:.0f} s", flush=True)
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, default=5)
    ap.add_argument("--stages", default="")
    ap.add_argument("--redo", default="")
    ap.add_argument("--gate-mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--gate-seeds", type=int, default=25)
    ap.add_argument("--gate-offsets", default="0,1000")
    ap.add_argument("--gate-workers", type=int, default=0)
    ap.add_argument("--gate-v2-floors", action="store_true", help="also run V2 at the non-zero offsets")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.gate_workers = a.gate_workers or a.cores
    return Chain(a).go()


if __name__ == "__main__":
    raise SystemExit(main())
