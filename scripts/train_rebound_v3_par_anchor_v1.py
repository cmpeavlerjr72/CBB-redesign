"""Season-drift anchor `O` for the box-parallel rebound trainer (lane C, 2026-09-30).

Versioned sibling of `scripts/train_rebound_v3_par_v1.py` (lane J), which is NOT edited.
This wrapper adds ONE option, `--anchor O`, and otherwise hands every argument to J's
`main()` unchanged (`--stage`, `--folds`, `--seed`, `--team-rate-table`, `--n-jobs`,
`--out-dir`, `--max-cuts`, ...).

With `--anchor O`:
  * a new arm `O` is registered in `train_rebound_v3_round3.ARMS` as
    {"offset": True}, i.e. the round-3 A5 offset MECHANISM (LightGBM init_score on the
    OREB raw score at fit time, added back at predict time, R3.fit_arm / R3.predict_arm);
  * the offset VALUES in the fold's `_a5_off` column are replaced, after
    `R3.add_fold_columns`, by the season-drift anchor `O` from
    `cbb_sim.season_anchor.anchor_O` (as-of in-season live OREB share, day 0 = prior
    season's end level, centred on the fold's pooled train level). Computed on the fold's
    train + test rows, strictly before each row's date, so every S1 refit's pool and
    scored segment carry the as-of value;
  * any arm that reads A5's trend offset (A5, A5+...) is refused, because `_a5_off` no
    longer holds the trend.
With `--team-rate-table <E3 table>` the arm `O` is the Stage B `TO` arm (E3 team features
+ anchor O); its cell lands under `<out-dir>/<table stem>/cells/s2_<fold>_O_seed<k>.json`.
Pre-registration / ruling: `docs/models/season_drift/experiments.md` section 3.

    python scripts/train_rebound_v3_par_anchor_v1.py --anchor O --stage 2 --folds F2 \
        --arms O --team-rate-table data/processed/team_rate_features_E3_v2.parquet \
        --n-jobs 24 --out-dir data/processed/models/rebound/round3_par_TO_v1
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import train_rebound_v3_par_v1 as J  # noqa: E402  (pins threads on import)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_rebound_v3_round3 as R3  # noqa: E402
from cbb_sim import season_anchor as SA  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

ANCHOR_WHY = ("season-drift anchor O: offset = logit(as-of league live OREB share, day 0 = "
              "prior-season end) - logit(train pooled); cbb_sim.season_anchor")


def install_anchor_O() -> None:
    R3.ARMS["O"] = {"offset": True, "why": ANCHOR_WHY}
    orig = R3.add_fold_columns

    def add_fold_columns_O(tr, te):
        tr, te, meta = orig(tr, te)
        both = pd.concat([tr[["season", "game_date", "y"]], te[["season", "game_date", "y"]]],
                         ignore_index=True)
        num, den = SA.rebound_inputs(both, RB.CLASS_INDEX["OREB"], RB.CLASS_INDEX["DEAD"])
        train_seasons = sorted(int(s) for s in tr["season"].unique())
        a = SA.anchor_O(both["season"].to_numpy(), both["game_date"].to_numpy(), num, den,
                        train_seasons, "binary")
        off = a.offset()[:, 0]
        tr = tr.assign(_a5_off=off[:len(tr)])
        te = te.assign(_a5_off=off[len(tr):])
        meta = dict(meta, anchor="O", anchor_Lbar=a.meta["Lbar"],
                    anchor_prior_by_season={str(k): v for k, v in a.meta["prior_by_season"].items()},
                    note="_a5_off holds the season-drift anchor O, not the A5 trend")
        return tr, te, meta

    R3.add_fold_columns = add_fold_columns_O


def main() -> int:
    argv = sys.argv[1:]
    anchor = ""
    if "--anchor" in argv:
        i = argv.index("--anchor")
        anchor = argv[i + 1]
        del argv[i:i + 2]
    if anchor not in ("", "O"):
        raise SystemExit(f"--anchor {anchor!r}: only 'O' is defined")
    if anchor == "O":
        arms = ""
        if "--arms" in argv:
            arms = argv[argv.index("--arms") + 1]
        bad = [a for a in arms.split(",") if a and "A5" in a.split(R3.COMBO_SEP)]
        if bad or not arms:
            raise SystemExit(f"--anchor O needs explicit --arms without A5 (got {arms!r}); "
                             "_a5_off is overwritten by the anchor")
        install_anchor_O()
    sys.argv = [sys.argv[0], *argv]
    return J.main()


if __name__ == "__main__":
    raise SystemExit(main())
