"""train_clock_v3c_s1.py -- ROUND 3c: the S1 schedules the closed-loop run needs.

Pre-registration: `docs/models/clock/experiments.md` section 12, committed
before this file was written.

Round 3b fitted exactly ONE S1 schedule -- `gamma_aft|P3` -- because that was
the arm its Part A nominated for the closed-loop gate. Round 3c's whole point is
that the offline stage CANNOT choose between P1, P2 and P3 (section 11.2), so
the closed-loop run needs all of them, plus the cell-based `srfloor` arm that
L26 says to prefer for the engine. This script fits the four schedules that do
not exist yet:

    gamma_aft|P1            gamma_aft|P2
    empirical_km3_srfloor|P1   empirical_km3_srfloor|P3

and writes them under `data/processed/models/clock/v3c_s1/`. It NEVER writes
into `v3b_s1/`: `gamma_aft|P3` is served to the engine straight out of round
3b's directory, so that arm is byte-identical to the one round 3b scored, and a
worker reading round 3b's artifacts cannot have them changed underneath it.

The scheme itself is round 3b's, reused: refit at every month boundary of the
test season on all prior seasons plus the test season strictly before that
boundary, with `possession_outcome.month_boundaries` as the single definition of
"monthly". No lookup tables are exported -- the engine runs the LIVE fitted
object, which is what the offline grid scored, and L26 prices binning error at
10.8 CRPS floors and +0.46 possessions.

Usage (threads capped at 4; five other workers share this machine):
    OMP_NUM_THREADS=4 .venv/Scripts/python.exe scripts/train_clock_v3c_s1.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cbb_sim.models import clock as ck  # noqa: E402
from cbb_sim.models import clock_v3 as c3  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

OUT = Path("data/processed/models/clock")
S1_DIR = OUT / "v3c_s1"
DESIGN_V2 = OUT / "design_v2.parquet"
ALL_SEASONS = [2022, 2023, 2024, 2025]
BASE_SEED = 20260910
TEST_SEASON = 2025

#: (base arm, parametrisation, engine tag). The tag is the suffix of the
#: `ENGINE_CLOCK` value the adapter accepts.
SCHEDULES: tuple[tuple[str, str, str], ...] = (
    ("gamma_aft", "P1", "gamma_P1"),
    ("gamma_aft", "P2", "gamma_P2"),
    ("empirical_km3_srfloor", "P1", "srfloor_P1"),
    ("empirical_km3_srfloor", "P3", "srfloor_P3"),
)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit_schedule(base: str, par: str, tag: str, tr: pd.DataFrame,
                 te: pd.DataFrame) -> dict:
    """One S1 schedule: a fit per month boundary, persisted with a manifest."""
    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    cuts = PO.month_boundaries(te_dates)

    # Subset the columns every fitter reads, exactly as round 3b did: the
    # monthly concatenation of a 2.6M-row design is ~1 GB per refit at full
    # width and cannot change a fitted model.
    cols = c3.s1_fit_columns("empirical_km3" if base.startswith("empirical") else "gamma_aft")
    cols = [c for c in dict.fromkeys([*cols, *ck.feature_set(c3.P_FEATURES[par])])
            if c in tr.columns]
    tr_small = tr[cols]

    months = []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None
               else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, cols]
        fit_rows = tr_small if not len(prior) else pd.concat([tr_small, prior],
                                                             ignore_index=True)
        t0 = time.time()
        arm = c3.fit_arm_v3b(base, par, fit_rows, seed=BASE_SEED)
        stamp = f"{pd.Timestamp(cut).year:04d}-{pd.Timestamp(cut).month:02d}"
        mfile = S1_DIR / f"{tag}_S1_{TEST_SEASON}_{stamp}.pkl"
        with open(mfile, "wb") as f:
            pickle.dump(arm, f)
        mx = tr_dates.max() if not before.any() else max(tr_dates.max(),
                                                         te_dates[before].max())
        months.append({
            "refit_date": str(pd.Timestamp(cut).date()),
            "valid_from": str(pd.Timestamp(cut).date()),
            "valid_to": None if nxt is None
            else str((pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()),
            "n_train": int(len(fit_rows)),
            "n_train_from_test_season": int(len(prior)),
            "n_scored": int(seg.sum()),
            "max_train_date": str(pd.Timestamp(mx).date()),
            "season": TEST_SEASON, "month": stamp,
            "model_file": str(mfile.relative_to(OUT)).replace("\\", "/"),
            "fit_seconds": round(time.time() - t0, 1),
        })
        log(f"  {tag} refit {months[-1]['refit_date']}: {len(fit_rows):,} rows "
            f"({len(prior):,} from the test season), scores {int(seg.sum()):,}, "
            f"{months[-1]['fit_seconds']:.0f}s")
        del fit_rows, prior, arm

    if not months:
        raise AssertionError(f"{tag}: S1 produced no refits")
    manifest = {
        "base_arm": base, "parametrisation": par, "tag": tag, "scheme": "S1",
        "season": TEST_SEASON, "fold": "F2", "round": "3c",
        "selection_rule": ("pick the row whose [valid_from, valid_to] contains the "
                           "game's own date; equivalently the latest refit_date at or "
                           "before it. valid_to null means the last month of the season."),
        "naming": "{tag}_S1_{season}_{YYYY-MM}.pkl, relative to "
                  "data/processed/models/clock/",
        "arm_adopted": False,
        "note": ("fitted for the round-3c Decision-10 closed-loop run "
                 "(experiments.md section 12); adoption is decided there, not here. "
                 "No lookup table is exported: the engine runs the live fitted object."),
        "months": months,
    }
    (S1_DIR / f"manifest_{tag}.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated tags to fit")
    a = ap.parse_args()

    S1_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()

    log(f"loading {DESIGN_V2}")
    design = pd.read_parquet(DESIGN_V2)
    design, cdiag = c3.attach_horn_censoring(design, ALL_SEASONS)
    c3.set_flag_inplace(design, "horn")
    tr, te = ck.fold_slices(design, "F2")
    log(f"F2 train {len(tr):,} / test {len(te):,}; horn-censored "
        f"{cdiag['censored_horn_pct']}%")
    del design

    want = {s.strip() for s in a.only.split(",") if s.strip()}
    out = {}
    for base, par, tag in SCHEDULES:
        if want and tag not in want:
            continue
        log(f"S1 schedule {base}|{par} -> {tag}")
        out[tag] = fit_schedule(base, par, tag, tr, te)

    (S1_DIR / "index.json").write_text(json.dumps({
        "round": "3c", "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "schedules": {k: {"base_arm": v["base_arm"],
                          "parametrisation": v["parametrisation"],
                          "n_months": len(v["months"])} for k, v in out.items()},
        "not_written_here": {
            "gamma_P3": "served from v3b_s1/manifest.json, byte-identical to the "
                        "schedule round 3b scored; this script never writes there"},
    }, indent=2, default=str), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
