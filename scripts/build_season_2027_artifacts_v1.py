#!/usr/bin/env python
"""
build_season_2027_artifacts_v1.py -- season-2027 (2026-27) keys for the live sim stage (lane F2, 2026-09-30).

NO MODEL CHOICE and NO REFIT. Each artifact below is the served fold-2 (F2_2025) artifact carried forward under the 2027 key, with its provenance
written in the artifact. The sim stage's prerequisite check passes on these; the retrain-set question (refit through 2025-26, arm C ratings, ...) stays
the PM's and is a separate bake-off dimension.

  1. data/processed/models/engine/event_round2_s1_F2_2027/index.json
       = event_round2_s1_F2_2025/index.json with season 2027 and RELATIVE file paths into the F2_2025 dir (no joblib is copied). The S1 manifest picks
         the latest refit at or before each game date, i.e. the 2025-04-01 refit (max_train_date 2025-03-xx) for every 2026-27 game. UNREFIT through 2025-26.
  2. data/processed/models/engine/names_F2_2027_v2.json
       = names_F2_2025_v2.json (team / slot name maps, usage classes, RULE CONSTANTS: dead_share, and_one, silent foul per possession, technical rate,
         usage priors, clock tempo fallback) with `meta.carried_forward_from`. Rule constants were derived on 2022-2024 and are carried, not re-derived.
  3. data/processed/models/free_throw/bonus_era.json: adds the season "2027" row (bonus 6 prior fouls, double bonus 9), marked
       derived_from "CARRIED FORWARD from 2026". state.new_state already falls back to the latest season silently; this makes the row explicit.
       The 2026-27 NCAA men's rules cycle is the SAME biennial change set as 2025-26 (docs/ops/day1_readiness_2027_2026-09-30.md section 3).

Idempotent. `--check` prints what is present without writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENG = REPO / "data/processed/models/engine"
BONUS = REPO / "data/processed/models/free_throw/bonus_era.json"
SRC_EVENT = ENG / "event_round2_s1_F2_2025"
DST_EVENT = ENG / "event_round2_s1_F2_2027"

CARRY_NOTE = ("CARRIED FORWARD, NOT REFIT, NOT RE-DERIVED: served fold-2 (train through 2023-24) artifact reused for 2026-27 (season 2027). "
              "Refit through 2025-26 is a PM decision (retrain-set dimension).")


def build_event_index() -> dict:
    idx = json.loads((SRC_EVENT / "index.json").read_text(encoding="utf-8"))
    idx = json.loads(json.dumps(idx))
    idx["season"] = 2027
    idx["carried_forward_from"] = {"dir": "event_round2_s1_F2_2025", "season": 2025, "note": CARRY_NOTE}
    for pop in idx["populations"].values():
        for sg in pop["segments"]:
            sg["file"] = f"../event_round2_s1_F2_2025/{Path(sg['file']).name}"
    return idx


def build_names() -> dict:
    t = json.loads((ENG / "names_F2_2025_v2.json").read_text(encoding="utf-8"))
    t["meta"] = {**t.get("meta", {}), "season": 2027, "carried_forward_from": "names_F2_2025_v2.json", "carry_note": CARRY_NOTE,
                 "rule_constants_status": "UNVERIFIED for 2026-27: carried from the template (derived on 2022-2024); no 2026-27 rule change affecting them is known "
                                          "(biennial 2025-26 / 2026-27 change set; points of emphasis only)."}
    return t


def build_bonus(cur: dict) -> dict:
    cur = json.loads(json.dumps(cur))
    prev = cur["by_season"]["2026"]
    cur["by_season"]["2027"] = {"bonus_prior_fouls": prev["bonus_prior_fouls"], "double_bonus_prior_fouls": prev["double_bonus_prior_fouls"],
                                "derived_from": "CARRIED FORWARD from 2026 (no 2027 data yet); NCAA 2026-27 men's rules change set equals 2025-26; UNVERIFIED by data"}
    return cur


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    state = {"event_index_2027": (DST_EVENT / "index.json").exists(), "names_2027": (ENG / "names_F2_2027_v2.json").exists(),
             "bonus_2027": "2027" in json.loads(BONUS.read_text(encoding="utf-8"))["by_season"]}
    if a.check:
        print(json.dumps(state))
        return 0
    DST_EVENT.mkdir(parents=True, exist_ok=True)
    (DST_EVENT / "index.json").write_text(json.dumps(build_event_index(), indent=1), encoding="utf-8")
    (ENG / "names_F2_2027_v2.json").write_text(json.dumps(build_names(), indent=1), encoding="utf-8")
    cur = json.loads(BONUS.read_text(encoding="utf-8"))
    if "2027" not in cur["by_season"]:
        BONUS.write_text(json.dumps(build_bonus(cur), indent=1), encoding="utf-8")
    print(json.dumps({"before": state, "wrote": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
