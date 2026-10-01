#!/usr/bin/env python
"""chain_full_retrain_v1.py -- THE FULL RETRAIN ON THE CLEAN FOUNDATION as one resumable command (lane D, 2026-09-30).

Definition (HANDOFF 2026-09-30 open item 2, cross-cutting finding 3): every served sub-model the foundation touches,
retrained on
  * event layer v4 (`possessions_v4`; consumers clock, possession_outcome, rotation; late_game is not served),
  * the corrected team-foul state (lane A's replay on the v4 machine -> round 8's in_bonus overlay recipe),
  * own ratings C (`data/processed/ratings_C_v1`),
with engine inputs built by the v3 TAG path and a gate read on verified truth. Team-rate features are a DIMENSION:
  --variant F_R   served expanding-mean as-of features (no table)
  --variant F_T   E3 v4 features through `cbb_sim.team_rate_adapter` (PO, fg_make, rebound; the adapter's scope)
The season-drift anchor O (`TO` arm) is a SWITCH, default off: --anchor O (PO first + rebound anchored trainers,
per-game offsets, ENGINE_SEASON_ANCHOR at the gate).
Served unchanged (no foundation change touches them; docs/ops/full_retrain_chain_2026-09-30.md section 1):
free_throw, usage (priors in the inputs), the static rotation fit / input priors, late_game (not served).

Nothing is chosen here. Every stage calls a trainer whose served-spec reproduction is proven in the doc; the only
inputs that change are the foundation swaps. Nothing is adopted and no served file is written: everything lands in
  data/processed/models/full_retrain_v1/<tag>/   (gitignored; HF key proposed in the doc)

Stages (dependency order; each writes <root>/<stage>/.done.json when it finishes; a rerun skips finished stages;
an unfinished checkpointed trainer stage resumes in place, an unfinished builder stage is cleared and rerun):
  possessions_v4  build data/processed/possessions_v4 if any 2022-2025 file is missing (shared sibling, not per tag)
  rotation        rotation S1 windows on v4 (serial trainer; started in the BACKGROUND first, joined before inputs)
  po_design       PO round-2 design on v4 + ratings C
  foul_state      lane A's replay on the v4 machine + the in_bonus overlay keyed on po_design
  po_train        PO S1 (12 fits) with the in_bonus overlay [+ E3 table]; anchor O -> anchored trainer
  clock           served clock spec (srfloor_P3 S1 + v5b sigma) on v4 + ratings C
  fg_design       fg_make design with the four rating columns from ratings C
  fg_train        fg_make round-4 B1 S1 (18 fits) [+ E3 table, fresh shooter extra cache]
  rb_design       rebound round-3 design with ratings C
  rb_train        rebound S1_weekly A0B0C0 (23 fits) [+ E3 table]; anchor O -> anchored wrapper
  inputs          v3 tag path: ratings C channels + [E3] + overlay + overrides.json (+ anchor offsets)
  gate            closed loop(s) through scripts/run_engine_overlay_v1.py, CBB_TRUTH=verified_v1, eval_gates per run

    .venv/Scripts/python.exe scripts/chain_full_retrain_v1.py --variant F_R --tag FR_v1 --cores 3 --dry-run
    .venv/Scripts/python.exe scripts/chain_full_retrain_v1.py --variant F_R --tag FR_v1 --cores 3
    # smoke (tiny, minutes):  --smoke  (PO every 40th game / 15 trees, rebound 2 cuts / 15 trees, rotation 2 windows,
    #                                    gate on a 30-game sample x 2 seeds)
    # box (full gate read, 5,710 games, 200 seeds, four floor draws):
    #   scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_R --tag FR_v1 --cores 90 \
    #       --gate-mode full --gate-seeds 200 --gate-offsets 0,1000,2000,3000,4000 --gate-workers 90
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
MODELS = ROOT / "data/processed/models"
CHAIN = MODELS / "full_retrain_v1"
RATINGS_C = "data/processed/ratings_C_v1"
SERVED_RATINGS = "data/processed/ratings"
E3_TABLE = "data/processed/team_rate_features_E3_v4.parquet"
FG_DESIGN = "data/processed/models/fg_make/design_v2_shotshooter.parquet"
RB_DESIGN = "data/processed/models/rebound/round3/design_round3.parquet"
SAMPLE500 = "data/processed/truth/stride500_verified_v1_F2_2025.parquet"
SEASONS = [2022, 2023, 2024, 2025]
STAGES = ["possessions_v4", "rotation", "po_design", "foul_state", "po_train", "clock", "fg_design", "fg_train",
          "rb_design", "rb_train", "inputs_base", "inputs", "parity", "gate"]
CHECKPOINTED = {"po_train", "fg_train", "rb_train"}          # resume in place
#: stages whose output does not depend on --variant (F_T can --reuse-from an F_R tag's finished ones)
SHARED = {"rotation", "po_design", "foul_state", "clock", "fg_design", "rb_design", "inputs_base"}
#: independent training branches (--parallel runs them at once)
BRANCHES = [["po_design", "foul_state", "po_train"], ["clock"], ["fg_design", "fg_train"], ["rb_design", "rb_train"]]
#: --gate-stack (lane D, 2026-10-01): the loop-level switches the gate serves on top of the chain's artifacts, set
#: EXPLICITLY so the read does not depend on the caller's environment. `adopted` = served stack v2's four loop-level
#: members (the fifth, clock L2, is the chain's own clock stage; the event team block is the chain's own inputs
#: block through the private ENGINE_DIR). `served_v1` = the switches of last night's laneD_2 / laneD_3 reads
#: (made before the 02:06 adoption). Gate tags carry a suffix so the two never share a results dir.
GATE_STACKS = {
    "adopted": {"ENGINE_SHOT_BLOCK": "K2_Ocell", "ENGINE_FOUL_JOINT": "R9ao3", "ENGINE_SHARED_SHOOTING": "G3",
                "ENGINE_CHANCE_TIME": "KD"},
    "served_v1": {"ENGINE_SHOT_BLOCK": "reference", "ENGINE_FOUL_JOINT": "reference",
                  "ENGINE_SHARED_SHOOTING": "reference", "ENGINE_CHANCE_TIME": "reference"},
}
GATE_STACK_SUFFIX = {"adopted": "_sv2", "served_v1": ""}
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "LIGHTGBM_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def rotation_windows() -> list[str]:
    """The served S1 rotation windows after the first (the first reuses the static fit by construction)."""
    obj = json.loads((MODELS / "engine/rotation_r2_s1_F2_2025.json").read_text(encoding="utf-8"))
    tags = [Path(e["path"]).stem.rsplit("_", 1)[-1] for e in obj["artifacts"]]
    return tags[1:]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


class Chain:
    def __init__(self, a):
        self.a = a
        self.root = (CHAIN / a.tag).resolve()
        self.e3 = a.team_rate_table if a.variant == "F_T" else ""
        self.stem = Path(self.e3).stem if self.e3 else ""
        self.ratings = a.ratings_dir
        # rotation runs in the BACKGROUND (serial: one process; --parallel: one process per S1 window) while the
        # parallel trainers run; the trainers get what is left so the core cap holds
        self.rot_windows = rotation_windows()
        rot_pending = (not a.stages or "rotation" in a.stages.split(",")) and not a.reuse_from and \
            not (CHAIN / a.tag / "rotation" / ".done.json").exists()
        self.rot_procs = (len(self.rot_windows) if a.parallel else 1) if rot_pending else 0
        self.cores = max(1, a.cores - self.rot_procs)
        if a.parallel:
            # every branch runs at once: each trainer takes at most its own task count
            self.share = {"po_train": min(12, self.cores), "fg_train": min(18, self.cores),
                          "rb_train": min(23, self.cores), "clock": 1}
            need = self.rot_procs + self.share["po_train"] + self.share["fg_train"] + self.share["rb_train"] + 1
            if need > a.cores:
                raise SystemExit(f"--parallel needs {need} cores (rotation {self.rot_procs} + PO 12 + fg 18 + rb 23 + "
                                 f"clock 1); --cores is {a.cores}. Drop --parallel or raise --cores")
        else:
            self.share = {}
        self.log = []
        self.bg: dict[str, subprocess.Popen] = {}

    # ----------------------------------------------------------------- helpers
    def d(self, stage: str) -> Path:
        if self.a.reuse_from and stage in SHARED and \
                (CHAIN / self.a.reuse_from / stage / ".done.json").exists():
            return CHAIN / self.a.reuse_from / stage
        return self.root / stage          # not finished in the reuse tag: built under this tag

    def o(self, stage: str) -> Path:
        """a trainer / builder output dir: a fresh subdir, never the stage dir that holds stage.log"""
        return self.d(stage) / "out"

    def done(self, stage: str) -> bool:
        return (self.d(stage) / ".done.json").exists()

    def mark(self, stage: str, info: dict) -> None:
        self.d(stage).mkdir(parents=True, exist_ok=True)
        (self.d(stage) / ".done.json").write_text(json.dumps({"finished": now(), **info}, indent=1, default=str))

    def env(self, extra: dict | None = None) -> dict:
        e = dict(os.environ)
        for v in THREAD_VARS:
            e[v] = "1"
        e["PYTHONIOENCODING"] = "utf-8"
        e["CBB_TRUTH"] = "verified_v1"
        if getattr(self.a, "ot_foul_carry", False):          # DEFAULT OFF: event-stream consumers carry team fouls into overtime
            e["CBB_OT_FOUL_CARRY"] = "1"
        if extra:
            e.update(extra)
        return e

    @property
    def pv(self) -> str:
        """possessions / machine version the foul-state consumers read: v4, or v4otc (--ot-foul-carry, default off)"""
        return "v4otc" if getattr(self.a, "ot_foul_carry", False) else "v4"

    def n(self, stage: str) -> int:
        """worker processes for a parallel trainer stage"""
        return self.share.get(stage, self.cores)

    def run(self, stage: str, cmd: list[str], extra_env: dict | None = None, background: bool = False,
            key: str = ""):
        cmd = [str(c) for c in cmd]
        line = " ".join(cmd)
        print(f"[{now()}] {stage}: {line}", flush=True)
        self.log.append({"stage": stage, "cmd": line, "start": now()})
        if self.a.dry_run:
            return 0
        self.d(stage).mkdir(parents=True, exist_ok=True)
        logf = open(self.d(stage) / "stage.log", "a", encoding="utf-8")
        logf.write(f"\n=== {now()} {line}\n")
        logf.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=self.env(extra_env), stdout=logf, stderr=subprocess.STDOUT)
        if background:
            self.bg[key or stage] = p
            return p
        rc = p.wait()
        logf.close()
        if rc != 0:
            raise SystemExit(f"stage {stage} FAILED rc={rc}; see {self.d(stage) / 'stage.log'}; "
                             "rerun the same command to resume")
        return rc

    def clear_partial(self, stage: str) -> None:
        if self.a.dry_run or stage in CHECKPOINTED:
            return
        p = self.d(stage)
        if p.exists() and not self.done(stage):
            log = p / "stage.log"
            keep = log.read_text(encoding="utf-8") if log.exists() else ""
            shutil.rmtree(p)
            p.mkdir(parents=True)
            if keep:
                (p / "stage.log").write_text(keep + f"\n=== {now()} cleared partial output, rerunning\n", encoding="utf-8")

    def s(self, p) -> str:
        """Repo-relative POSIX (portable between Windows and the Linux box)."""
        return Path(os.path.relpath(Path(p).resolve(), ROOT)).as_posix()

    # ------------------------------------------------------------------ paths
    def po_art(self) -> Path:
        base = self.o("po_train") / (self.stem or "")
        return base / "event_round2_s1_F2_2025"

    def fg_dir(self) -> Path:
        return self.o("fg_train") / self.stem if self.stem else self.o("fg_train")

    def rb_art(self) -> Path:
        cell = "s2_F2_O_seed0" if self.a.anchor else "s2_F2_A0B0C0_seed0"
        base = self.o("rb_train") / self.stem if self.stem else self.o("rb_train")
        return base / "artifacts" / cell

    # ----------------------------------------------------------------- stages
    def st_possessions_v4(self):
        missing = [f"{k}_{s}" for s in SEASONS for k in ("possessions", "chances")
                   if not (ROOT / f"data/processed/possessions_{self.pv}/{k}_{s}.parquet").exists()]
        if not missing:
            print(f"[{now()}] possessions_v4: present (2022-2025)", flush=True)
            return {"built": False}
        if self.pv == "v4otc":
            self.run("possessions_v4", [PY, "scripts/build_possessions_v4otc_v1.py", "--seasons", *map(str, SEASONS)])
        else:
            self.run("possessions_v4", [PY, "scripts/build_possessions_v4.py", "--seasons", *map(str, SEASONS)])
        return {"built": True, "missing_before": missing}

    def st_rotation(self):
        base = [PY, "scripts/train_rotation_v3b_s1_poss_v1.py", "--poss-version", "v4"]
        if self.a.smoke:
            return self.run("rotation", [*base, "--out-dir", self.s(self.d("rotation") / "fits"), "--fits-only",
                                         "--max-windows", "2", "--", "--test-games", "60", "--seeds", "1"],
                            background=True, key="rotation")
        if not self.a.parallel:
            return self.run("rotation", [*base, "--out-dir", self.s(self.d("rotation") / "fits"), "--fits-only"],
                            background=True, key="rotation")
        for w in self.rot_windows:          # one process per window, each in its own dir (no shared writes)
            self.run("rotation", [*base, "--out-dir", self.s(self.d("rotation") / f"w_{w}"), "--only-window", w],
                     background=True, key=f"rotation_{w}")
        return None

    def st_po_design(self):
        self.run("po_design", [PY, "scripts/build_po_design_v4_v1.py", "--poss-version", self.pv,
                               "--ratings-dir", self.ratings, "--out", self.s(self.d("po_design") / "design.parquet")])
        return {"design": self.s(self.d("po_design") / "design.parquet")}

    def st_foul_state(self):
        self.run("foul_state", [PY, "scripts/build_foul_state_v4_v1.py", "--machine", self.pv,
                                "--out-dir", self.s(self.o("foul_state")), "--workers", str(min(4, self.cores)),
                                "--overlay-design", self.s(self.d("po_design") / "design.parquet"),
                                "--compare-to",
                                "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet"])
        return {"overlay": self.s(self.o("foul_state") / "in_bonus_overlay.parquet")}

    def st_po_train(self):
        design = self.s(self.d("po_design") / "design.parquet")
        bonus = self.s(self.o("foul_state") / "in_bonus_overlay.parquet") + ":game_id,poss_index,chance_number:in_bonus"
        smoke = ["--sample-games", "40", "--test-n-estimators", "15"] if self.a.smoke else []
        if self.a.anchor:
            comb = self.d("po_design") / "design_inbonus.parquet"
            if not comb.exists():
                self.run("po_train", [PY, "scripts/build_design_overlay_v1.py", "--design", design,
                                      "--feature-table", bonus, "--out", self.s(comb)])
            cmd = [PY, "scripts/train_possession_outcome_s1_par_anchor_artifacts_v1.py", "--anchor", "O",
                   "--fold", "F2", "--season", "2025", "--design", self.s(comb),
                   "--out-root", self.s(self.o("po_train")), "--n-jobs", str(self.n('po_train')), *smoke]
        else:
            cmd = [PY, "scripts/train_possession_outcome_s1_par_v2.py", "--fold", "F2", "--season", "2025",
                   "--design", design, "--feature-table", bonus,
                   "--out-root", self.s(self.o("po_train") / self.stem) if self.stem else self.s(self.o("po_train")),
                   "--n-jobs", str(self.n('po_train')), *smoke]
        if self.e3:
            cmd += ["--team-rate-table", self.e3, "--team-rate-missing", self.a.team_rate_missing]
        self.run("po_train", cmd)
        if not self.a.dry_run and not (self.po_art() / "index.json").exists():
            raise SystemExit(f"po_train finished but {self.po_art()} has no index.json")
        return {"artifacts": self.s(self.po_art())}

    def st_clock(self):
        self.run("clock", [PY, "scripts/train_clock_chain_v1.py", "--root", self.s(self.d("clock") / "root"),
                           "--poss-version", self.pv, "--ratings-dir", self.ratings])
        return {"root": self.s(self.d("clock") / "root")}

    def st_fg_design(self):
        out = self.d("fg_design") / "design_v2_shotshooter_r.parquet"
        if self.ratings == SERVED_RATINGS:
            return {"design": FG_DESIGN, "note": "served ratings: served design"}
        self.run("fg_design", [PY, "scripts/build_design_ratings_swap_v1.py", "--design", FG_DESIGN,
                               "--ratings-dir", self.ratings, "--out", self.s(out)])
        return {"design": self.s(out)}

    def st_fg_train(self):
        design = json.loads((self.d("fg_design") / ".done.json").read_text())["design"] if not self.a.dry_run \
            else self.s(self.d("fg_design") / "design_v2_shotshooter_r.parquet")
        cmd = [PY, "scripts/train_fg_make_v4_par_v2.py", "--mode", "run", "--arms", "B1", "--no-floor",
               "--n-jobs", str(self.n('fg_train')), "--out-dir", self.s(self.o("fg_train")), "--design", design]
        if self.e3:
            cache = self.o("fg_train") / f"design_v4_extra_{self.stem}.parquet"
            if cache.exists() and not self.a.dry_run:
                cache.unlink()           # par_v1 refuses an existing cache with a table; a resume rebuilds its own
            cmd += ["--team-rate-table", self.e3, "--team-rate-missing", self.a.team_rate_missing,
                    "--extra-cache", self.s(cache)]
        self.run("fg_train", cmd)
        return {"artifacts": self.s(self.fg_dir() / "B1"), "m": self.s(self.fg_dir() / "m_fitted.json")}

    def st_rb_design(self):
        out = self.d("rb_design") / "design_round3_r.parquet"
        if self.ratings == SERVED_RATINGS:
            return {"design": RB_DESIGN, "note": "served ratings: served design"}
        self.run("rb_design", [PY, "scripts/build_design_ratings_swap_v1.py", "--design", RB_DESIGN,
                               "--ratings-dir", self.ratings, "--out", self.s(out)])
        return {"design": self.s(out)}

    def st_rb_train(self):
        design = json.loads((self.d("rb_design") / ".done.json").read_text())["design"] if not self.a.dry_run \
            else self.s(self.d("rb_design") / "design_round3_r.parquet")
        if self.a.anchor:
            cmd = [PY, "scripts/train_rebound_v3_par_anchor_artifacts_v1.py", "--anchor", "O", "--arms", "O"]
        else:
            cmd = [PY, "scripts/train_rebound_v3_par_artifacts_v1.py", "--arms", "A0B0C0"]
        cmd += ["--stage", "2", "--folds", "F2", "--n-jobs", str(self.n('rb_train')), "--out-dir", self.s(self.o("rb_train")),
                "--design", design]
        if self.e3:
            cmd += ["--team-rate-table", self.e3, "--team-rate-missing", self.a.team_rate_missing]
        env = None
        if self.a.smoke:
            cmd += ["--max-cuts", "2"]
            env = {"REB_TEST_N_ESTIMATORS": "15"}
        self.run("rb_train", cmd, extra_env=env)
        return {"artifacts": self.s(self.rb_art())}

    def join_rotation(self):
        keys = [k for k in self.bg if k.startswith("rotation")]
        if not keys:
            return
        for k in keys:
            p = self.bg[k]
            print(f"[{now()}] waiting for background {k} (pid {p.pid})", flush=True)
            rc = p.wait()
            if rc != 0:
                raise SystemExit(f"{k} FAILED rc={rc}; see {self.d('rotation') / 'stage.log'}; rerun to resume")
            del self.bg[k]
        fits = self.d("rotation") / "fits"
        if any(k != "rotation" for k in keys):          # --parallel: gather the per-window fits, then the manifest
            fits.mkdir(parents=True, exist_ok=True)
            for w in self.rot_windows:
                wd = self.d("rotation") / f"w_{w}"
                shutil.copyfile(wd / f"rotation_fit_v3_S1_{w}.json", fits / f"rotation_fit_v3_S1_{w}.json")
                for f in wd.glob("rotation_fit_v3*.json"):
                    if not (fits / f.name).exists():
                        shutil.copyfile(f, fits / f.name)
            self.run("rotation", [PY, "scripts/train_rotation_v3b_s1_poss_v1.py", "--out-dir", self.s(fits),
                                  "--manifest-only"])
        if not (fits / "rotation_r2_s1_F2_2025.json").exists():
            raise SystemExit(f"rotation finished without a manifest in {fits}")
        self.mark("rotation", {"fits": self.s(fits), "smoke": self.a.smoke,
                               "scope": "S1 windows refit on v4; first window = served static fit (v1); "
                                        "engine-input priors unchanged"})

    def base_dir(self) -> str:
        if self.a.inputs_event_layer == "v2":
            return "data/processed/models/engine_v3"
        return self.s(self.d("inputs_base") / "out" / "engine_v3_ev4")

    def st_inputs_base(self):
        """engine_v3 with the PO round-2 event block replayed on the v4 chance tables (lane D 2nd job, 2026-09-30)."""
        if self.a.inputs_event_layer == "v2":
            return {"base": self.base_dir(), "note": "engine_v3 as built (event block on possessions v2)"}
        work = self.d("inputs_base") / "out" / "work"
        n = max(1, min(self.cores, 16))      # rotation may still hold its cores
        procs = []
        for k in range(n):
            procs.append(self.run("inputs_base", [PY, "scripts/build_engine_inputs_v3_replay_evlayer_v1.py", "shard",
                                                  "--event-layer", "v4", "--work", self.s(work), "--shard", str(k),
                                                  "--n-shards", str(n)], background=True, key=f"evshard_{k}"))
        if not self.a.dry_run:
            for k in range(n):
                p = self.bg.pop(f"evshard_{k}")
                if p.wait() != 0:
                    raise SystemExit(f"inputs_base shard {k} failed; see {self.d('inputs_base') / 'stage.log'}")
        out = self.d("inputs_base") / "out" / "engine_v3_ev4"
        if out.exists() and not self.a.dry_run:
            shutil.rmtree(out)                     # a partial assemble from a cut-off run (the shards are kept)
        self.run("inputs_base", [PY, "scripts/build_engine_inputs_v3_replay_evlayer_v1.py", "assemble",
                                 "--event-layer", "v4", "--work", self.s(work), "--out", self.s(out)])
        return {"base": self.base_dir(), "census": self.s(out / "census.json")}

    def st_inputs(self):
        out = self.d("inputs") / "set"
        cmd = [PY, "scripts/build_engine_inputs_chain_v1.py", "--out", self.s(out), "--tag", self.a.tag,
               "--base-dir", self.base_dir(),
               "--ratings-dir", self.ratings, "--po-artifacts", self.s(self.po_art()),
               "--fg-artifacts", self.s(self.fg_dir() / "B1"), "--fg-m", self.s(self.fg_dir() / "m_fitted.json"),
               "--rb-artifacts", self.s(self.rb_art()), "--clock-root", self.s(self.d("clock") / "root"),
               "--rotation-dir", self.s(self.d("rotation") / "fits")]
        if self.e3:
            cmd += ["--team-rate-table", self.e3, "--team-rate-missing", self.a.team_rate_missing]
        self.run("inputs", cmd)
        info = {"input_dir": self.s(out / "inputs"), "overrides": self.s(out / "inputs/overrides.json")}
        if self.a.anchor:
            po_design = self.s(self.d("po_design") / "design_inbonus.parquet")
            rb_design = json.loads((self.d("rb_design") / ".done.json").read_text())["design"] if not self.a.dry_run else "?"
            self.run("inputs", [PY, "scripts/build_engine_anchor_offsets_v1.py", "--input-dir", info["input_dir"],
                                "--families", "po,rb", "--po-design", po_design, "--rb-design", rb_design])
            info["anchor_offsets"] = info["input_dir"] + "/anchor_offsets_F2_2025.npz"
        return info

    def st_parity(self):
        """train/serve parity of the team-rate-derived fg_make shooter feature + the rebound arm audit (2026-10-01).
        FAILS the run on a mismatch (thresholds: docs/ops/full_retrain_chain_2026-09-30.md section 3)."""
        inp = json.loads((self.d("inputs") / ".done.json").read_text()) if not self.a.dry_run else {"input_dir": "<inputs>"}
        out = self.d("parity") / "parity_fg_rb.json"
        self.run("parity", [PY, "scripts/diag_train_serve_parity_fg_v1.py", "--fg-out", self.s(self.fg_dir()),
                            "--inputs", inp["input_dir"], "--rb-arm", "A0B0C0",   # anchor O = the same features plus an init_score offset
                            "--out", self.s(out)])
        return {"report": self.s(out)}

    def st_gate(self):
        inp = json.loads((self.d("inputs") / ".done.json").read_text()) if not self.a.dry_run else {
            "input_dir": "<inputs>", "overrides": "<overrides>"}
        # ENGINE_CLOCK stays the v5b_glat_pmean KEY: the chain's clock root has the base layout, and with CK_DIR /
        # V5_PARAMS rebound to it that key serves the chain's v4 refit, which IS the adopted L2 clock (six S1 pickles
        # byte-equal to clock/r6_L2, same B1 sigma; scripts/diag_chain_clock_vs_L2_v1.py). The L2 KEY would resolve
        # <chain root>/r6_L2/... and fail: there is no second clock layer to double-apply.
        env = {"ENGINE_EVENT": "round2_s1", "ENGINE_CLOCK": "v5b_glat_pmean", "ENGINE_ROTATION": "reference",
               "ENGINE_FG3": "decision8", "CBB_TRUTH": "verified_v1", **GATE_STACKS[self.a.gate_stack]}
        if self.a.anchor:
            env["ENGINE_SEASON_ANCHOR"] = inp.get("anchor_offsets", "<offsets>")
        sample = self.a.gate_sample
        if self.a.smoke and self.a.gate_mode == "sample" and sample == SAMPLE500:
            sample = self.s(self.d("gate") / "sample30.parquet")
            if not self.a.dry_run:
                import pandas as pd
                self.d("gate").mkdir(parents=True, exist_ok=True)
                pd.read_parquet(SAMPLE500).iloc[::17].head(30).to_parquet(ROOT / sample, index=False)
        runs = []
        for off in [int(x) for x in self.a.gate_offsets.split(",") if x]:
            tag = f"fr1_{self.a.tag}_{self.a.gate_mode}_s{self.a.gate_seeds}_o{off}" + \
                ("_ev4" if self.a.inputs_event_layer == "v4" else "") + GATE_STACK_SUFFIX[self.a.gate_stack]
            if (ROOT / "results/engine_v0" / tag / "run_meta.json").exists():
                print(f"[{now()}] gate: {tag} exists, skipped", flush=True)
            else:
                if self.a.gate_mode == "sample":
                    rargs = ["--sample-file", sample, "--arm", "round2_s1", "--input-dir", inp["input_dir"],
                             "--seeds", str(self.a.gate_seeds), "--seed-offset", str(off),
                             "--workers", str(self.a.gate_workers), "--tag", tag, "--results-dir", "results/engine_v0"]
                    if self.a.gate_stack == "adopted":
                        # the sample runner's served-stack check compares the clock KEY with the default (L2); the
                        # chain pins the base-layout key that serves the chain's L2 refit (see env above): recorded
                        rargs.append("--allow-drift")
                else:
                    rargs = ["--fold", "F2", "--season", "2025", "--seeds", str(self.a.gate_seeds),
                             "--seed-offset", str(off), "--workers", str(self.a.gate_workers),
                             "--games-per-block", "60", "--seeds-per-block", "25", "--tag", tag,
                             "--results-dir", "results/engine_v0", "--input-dir", inp["input_dir"]]
                self.run("gate", [PY, "scripts/run_engine_overlay_v1.py", "--overrides", inp["overrides"],
                                  "--runner", self.a.gate_mode if self.a.gate_mode == "sample" else "full",
                                  "--", *rargs], extra_env=env)
            md = self.d("gate") / f"{tag}__verified.md"
            if not md.exists():
                self.run("gate", [PY, "scripts/eval_gates.py", "--results", f"results/engine_v0/{tag}",
                                  "--season", "2025", "--out", self.s(md)], extra_env={"CBB_TRUTH": "verified_v1"})
            runs.append({"tag": tag, "offset": off, "report": self.s(md)})
        pairs = []
        if self.a.gate_ref and runs:
            # Decision 12: the reference's own seed-offset floor draws (--gate-noise), and this arm's own
            # extra offsets when the run has them; each pairing is written next to the reports
            noises = [x for x in self.a.gate_noise.split(",") if x] + [r["report"] for r in runs[1:]]
            for k, nz in enumerate(noises or [""]):
                out = self.d("gate") / f"pair_ref_vs_{self.a.tag}_n{k}.md"
                cmd = [PY, "scripts/diag_pair_gate_reports.py", "--a", self.a.gate_ref, "--b", runs[0]["report"],
                       "--label-a", "reference", "--label-b", self.a.tag, "--out", self.s(out)]
                if nz:
                    cmd += ["--noise", nz, "--label-n", f"floor {Path(nz).stem}"]
                self.run("gate", cmd)
                pairs.append({"noise": nz, "out": self.s(out)})
        return {"runs": runs, "pairs": pairs, "sample": sample if self.a.gate_mode == "sample" else "full 5,710",
                "seeds": self.a.gate_seeds, "truth": "verified_v1"}

    # -------------------------------------------------------------- preflight
    def preflight(self) -> list[str]:
        """Every input the chain reads, with where it comes from on a fresh box (git / HF bulk key)."""
        req = [
            ("data/raw/cbbd/pbp", "HF raw (foul replay, rotation fouls, possessions_v4 rebuild)"),
            ("data/processed/possessions_v2/chances_2025.parquet", "git (possessions_v4 rebuild parity)"),
            ("data/processed/games_universe.parquet", "git"),
            ("data/processed/games_universe_v2.parquet", "git (clock latent)"),
            (f"{self.ratings}/own_ratings_2025.parquet", "git"),
            ("data/processed/ratings/own_ratings_2025.parquet", "git (inputs parity)"),
            ("data/processed/models/possession_outcome/round2/verdict.json", "git"),
            ("data/processed/models/engine/games_F2_2025.parquet", "git"),
            (FG_DESIGN, "HF model_artifacts"),
            ("data/processed/models/fg_make/events_v2_shotshooter.parquet", "HF model_artifacts"),
            ("data/processed/models/fg_make/design_v4_extra_v2.parquet", "HF model_artifacts (F_R shooter cache)"),
            ("data/processed/models/fg_make/lgbm_ladder_v2.json", "git"),
            (RB_DESIGN, "HF model_artifacts"),
            ("data/processed/models/rotation/rotation_fit_v3.json", "git"),
            ("data/processed/models/engine/rotation_r2_s1_F2_2025.json", "git"),
            ("data/processed/models/clock/v5_bakeoff/v5_bakeoff_report.json", "git"),
            ("data/processed/clock_censoring/censoring_v1_2025.parquet", "git"),
            ("data/processed/models/engine_v3/arrays_F2_2025.npz", "HF engine_inputs_v3"),
            ("data/processed/models/engine_v3/event_block_F2_2025.npz", "HF engine_inputs_v3"),
            ("data/processed/models/engine/fg_make_FGA_3_decision8_F2.joblib", "git"),
            ("data/processed/models/engine/free_throw_F2.joblib", "git"),
            ("data/processed/models/engine/rebound_F2.joblib", "git"),
            ("data/processed/models/free_throw/s1_confirm/S1_conf_aligned/F2/manifest.json",
             "HF model_artifacts (served free_throw, unchanged)"),
            (self.a.gate_sample, "git"),
        ]
        if self.e3:
            req.append((self.e3, "HF team_rate_tables"))
        if self.a.gate_ref:
            req.append((self.a.gate_ref, "git"))
        req += [(x, "git") for x in self.a.gate_noise.split(",") if x]
        missing = [f"{p}  <- {src}" for p, src in req if not (ROOT / p).exists()]
        print(f"[{now()}] preflight: {len(req) - len(missing)}/{len(req)} inputs present", flush=True)
        for m in missing:
            print(f"   MISSING {m}", flush=True)
        return missing

    # -------------------------------------------------------------------- run
    def plan(self) -> list[str]:
        only = [s for s in self.a.stages.split(",") if s] if self.a.stages else STAGES
        return [s for s in STAGES if s in only]

    def go(self) -> int:
        try:
            return self._go()
        except BaseException:
            # a failed stage must not leave this run's own background children writing into a stage dir that the
            # resume will clear: stop them (they are this process's children; their work redoes on resume)
            for k, p in list(self.bg.items()):
                if p.poll() is None:
                    print(f"[{now()}] stopping own background {k} (pid {p.pid}) after the failure", flush=True)
                    p.terminate()
                    try:
                        p.wait(timeout=60)
                    except Exception:                                  # noqa: BLE001
                        p.kill()
            raise

    def _go(self) -> int:
        stages = self.plan()
        print(f"[{now()}] full retrain chain v1: tag={self.a.tag} variant={self.a.variant} anchor={self.a.anchor or 'off'} "
              f"ratings={self.ratings} table={self.e3 or '(served features)'} cores={self.cores} smoke={self.a.smoke} "
              f"root={self.s(self.root)}", flush=True)
        missing = self.preflight()
        if self.a.preflight_only:
            return 1 if missing else 0
        if missing and not self.a.dry_run:
            raise SystemExit("preflight failed: pull the listed inputs first (scripts/hf_sync_data.py pull --dirs ...)")
        if self.a.reuse_from and not self.a.dry_run:
            rm = CHAIN / self.a.reuse_from / "chain_meta.json"
            if not rm.exists():
                raise SystemExit(f"--reuse-from {self.a.reuse_from}: no such chain run")
            ra = json.loads(rm.read_text())["args"]
            for k in ("ratings_dir", "smoke"):
                if ra.get(k) != getattr(self.a, k):
                    raise SystemExit(f"--reuse-from {self.a.reuse_from} was built with {k}={ra.get(k)!r}, "
                                     f"this run has {getattr(self.a, k)!r}")
        if not self.a.dry_run:
            self.root.mkdir(parents=True, exist_ok=True)
            meta = self.root / "chain_meta.json"
            if not meta.exists():
                try:
                    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True, timeout=30).stdout.strip()
                except Exception:                                      # noqa: BLE001 (no git in a container)
                    sha = "unknown"
                meta.write_text(json.dumps({"args": vars(self.a), "created": now(), "git": sha},
                                           indent=1, default=str))
            else:
                old = json.loads(meta.read_text())["args"]
                for k in ("variant", "anchor", "ratings_dir", "team_rate_table", "smoke", "inputs_event_layer", "ot_foul_carry"):
                    if k not in old and k in ("inputs_event_layer", "ot_foul_carry"):
                        continue          # a run started before this switch existed (laneD_1): --redo covers it
                    if old.get(k) != getattr(self.a, k):
                        raise SystemExit(f"tag {self.a.tag} was started with {k}={old.get(k)!r}; refusing to "
                                         f"resume it with {getattr(self.a, k)!r} (use a new --tag)")
        for st in [x for x in self.a.redo.split(",") if x]:
            if st not in STAGES:
                raise SystemExit(f"--redo {st}: unknown stage")
            own = self.root / st
            if own.exists() and not self.a.dry_run:
                arch = self.root / f"{st}.prev_{time.strftime('%Y%m%d_%H%M%S')}"
                own.rename(arch)
                print(f"[{now()}] --redo {st}: archived the previous output to {self.s(arch)}", flush=True)
        if self.a.parallel and not self.a.dry_run:
            # possessions_v4 and rotation first (rotation goes to the background), then the four independent
            # training branches at once, then inputs and gate
            pre = [s for s in stages if s in ("possessions_v4", "rotation")]
            for st in pre:
                self.one(st)
            branches = [[s for s in b if s in stages] for b in BRANCHES]
            import threading
            errs: list[BaseException] = []

            def _branch(b):
                try:
                    for s in b:
                        self.one(s)
                except BaseException as e:                 # noqa: BLE001  (SystemExit included)
                    errs.append(e)
            th = [threading.Thread(target=_branch, args=(b,), daemon=False) for b in branches if b]
            for t in th:
                t.start()
            for t in th:
                t.join()
            if errs:
                self.join_rotation()
                raise SystemExit(f"{len(errs)} branch(es) failed: {[str(e) for e in errs]}")
            for st in [s for s in stages if s in ("inputs_base", "inputs", "parity", "gate")]:
                self.one(st)
        else:
            for st in stages:
                self.one(st)
        self.join_rotation()
        if not self.a.dry_run:
            (self.root / "chain_log.json").write_text(json.dumps(self.log, indent=1))
        print(f"[{now()}] chain {self.a.tag}: finished the requested stages", flush=True)
        return 0

    def one(self, st: str) -> None:
        if st in ("inputs", "gate"):
            self.join_rotation()
        if self.done(st):
            print(f"[{now()}] {st}: done (skipped){' [reused from ' + self.a.reuse_from + ']' if self.d(st).parent != self.root else ''}",
                  flush=True)
            return
        self.clear_partial(st)
        t = time.time()
        info = getattr(self, f"st_{st}")()
        if st == "rotation":
            return                                   # marked when joined
        if not self.a.dry_run:
            info = info if isinstance(info, dict) else {}
            info["seconds"] = round(time.time() - t, 1)
            self.mark(st, info)
            print(f"[{now()}] {st}: DONE in {info['seconds']} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["F_R", "F_T"], required=True)
    ap.add_argument("--tag", required=True, help="output tag: data/processed/models/full_retrain_v1/<tag>/")
    ap.add_argument("--cores", type=int, default=3, help="worker processes for the parallel trainers (threads pinned to 1)")
    ap.add_argument("--anchor", choices=["", "O"], default="", help="season-drift anchor arm; default off")
    ap.add_argument("--ratings-dir", default=RATINGS_C)
    ap.add_argument("--team-rate-table", default=E3_TABLE, help="F_T only")
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise")
    ap.add_argument("--stages", default="", help="comma subset (dependency order is kept)")
    ap.add_argument("--gate-mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--gate-sample", default=SAMPLE500)
    ap.add_argument("--gate-seeds", type=int, default=25)
    ap.add_argument("--gate-offsets", default="0,1000,2000,3000,4000",
                    help="seed offsets; the first is the read, the rest are Decision-12 floor draws")
    ap.add_argument("--gate-workers", type=int, default=0, help="default --cores")
    ap.add_argument("--gate-ref", default="", help="reference eval_gates .md to pair against (optional)")
    ap.add_argument("--gate-noise", default="", help="comma list of the reference's floor-draw .md reports")
    ap.add_argument("--gate-stack", choices=sorted(GATE_STACKS), default="adopted",
                    help="loop-level switches served at the gate: adopted (served stack v2, tag suffix _sv2) or "
                         "served_v1 (last night's reads, no suffix)")
    ap.add_argument("--smoke", action="store_true", help="tiny slice of every stage (proof the chain runs)")
    ap.add_argument("--inputs-event-layer", choices=["v2", "v4"], default="v4",
                    help="event layer of the PO round-2 event block in the engine inputs (v4 = matches the retrained "
                         "PO; v2 = engine_v3 as built)")
    ap.add_argument("--ot-foul-carry", action="store_true",
                    help="DEFAULT OFF. Retrain on the OVERTIME TEAM-FOUL CARRY sibling (possessions_v4otc, foul state machine v4otc, "
                         "CBB_OT_FOUL_CARRY=1 for event-stream consumers). Needs the sibling built: scripts/build_possessions_v4otc_v1.py. "
                         "Not wired: prebuilt fg / rebound / free-throw designs (docs/tests/ot_team_foul_state_audit_2026-10-01.md)")
    ap.add_argument("--redo", default="",
                    help="comma list of THIS tag's stages to rebuild (the old output is archived, not deleted), "
                         "e.g. inputs,gate after the inputs switch")
    ap.add_argument("--reuse-from", default="",
                    help="tag of a finished run with the same ratings / smoke setting whose variant-independent stages "
                         f"({', '.join(sorted(SHARED))}) this run reads instead of rebuilding")
    ap.add_argument("--parallel", action="store_true",
                    help="box: run the four training branches and the rotation windows at once (needs >= 59 cores)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and every command; run nothing")
    ap.add_argument("--preflight-only", action="store_true", help="check every input exists, then exit")
    a = ap.parse_args()
    a.gate_workers = a.gate_workers or a.cores
    if a.smoke and a.gate_seeds == 25 and a.gate_offsets == "0,1000,2000,3000,4000":
        a.gate_seeds, a.gate_offsets = 2, "0"
    return Chain(a).go()


if __name__ == "__main__":
    raise SystemExit(main())
