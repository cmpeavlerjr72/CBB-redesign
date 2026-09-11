#!/usr/bin/env python
"""
diag_possession_outcome_conf_regime.py -- STAGE A of possession-outcome round 3
(ARCHITECTURE_DECISIONS.md Decision 9). DIAGNOSTIC ONLY: no model is selected,
no feature is adopted, nothing is fitted that round 2 did not already fit.

THE QUESTION. Round 2 adopted S1 = a CALENDAR-MONTHLY in-season walk-forward
refit (L21). Decision 9 asks whether the residual structure that survives S1 is
organised by the calendar at all, or by the CONFERENCE BOUNDARY -- the date each
team stops playing non-conference opponents. If the latter, a monthly cadence is
mis-aligned by construction and the style-rate features are reading a different
population before and after the boundary.

WHAT IT DOES.

  Step 1 (`--step score`). Re-scores the round-2 S1 WINNERS on fold 2 with the
  round-2 code path, changing nothing: `lgbm` for the `first` population,
  `cascade` for `cont`, feature set C_plus_state, scheme S1, seed 0, through
  `cbb_sim.models.possession_outcome.fit_predict_scheme`. Round 2 did not
  persist per-chance predictions, so they are regenerated rather than assumed.
  The run ASSERTS that the reproduced F2 log loss matches round 2's recorded
  value to 1e-6; if it does not, the diagnostic is not about round 2 and the
  script stops.

  Step 2 (`--step analyse`). Buckets the per-chance calibration residual
  (predicted minus realised class indicator) five ways -- calendar week; week
  relative to the OFFENCE team's first conference game; the same for the
  DEFENCE team; conference vs non-conference; week since the monthly refit --
  and asks which alignment carries more structure, against a game-level
  permutation null that controls for the differing bucket counts.

  Also: the "Mississippi Valley State" table -- per own-rating quintile, the
  mean of each raw-centred style feature in non-conference vs conference games.

OUTPUT: docs/tests/possession_outcome_conference_regime_2026-09-10.md
        data/processed/models/possession_outcome/round3/stageA/*.npz|json

RUN (4 threads, shared machine):
    OMP_NUM_THREADS=4 .venv/Scripts/python.exe \
        scripts/diag_possession_outcome_conf_regime.py --step score
    ... --step analyse
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.features.conference import (  # noqa: E402
    build_conference_flags, first_conference_game_dates,
)

OUT_DIR = _ROOT / "data/processed/models/possession_outcome/round3/stageA"
ROUND2_DIR = _ROOT / "data/processed/models/possession_outcome/round2"
OUT_MD = _ROOT / "docs/tests/possession_outcome_conference_regime_2026-09-10.md"

FEATURE_SET = "C_plus_state"
FOLD = "F2"
TEST_SEASON = 2025
#: The round-2 S1 winners, and the F2 log loss each recorded
#: (`round2/grid_results.csv`). Reproduction is asserted against these.
ROUND2_WINNERS = {
    "first": {"arm": "lgbm", "log_loss": 1.515428},
    "cont": {"arm": "cascade", "log_loss": 1.499760},
}
#: Classes the round-1/2 calibration gate scores (share >= 5%).
GATED_CLASSES = ("TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus")
#: A bucket thinner than this is reported but excluded from the variance
#: statistic and labelled UNDERPOWERED. Never presented as signal or as
#: absence of signal (CLAUDE.md, standing rule "multi-level evidence").
MIN_BUCKET_CHANCES = 5_000
N_PERM = 200
N_BOOT = 200
RNG_SEED = 20260910


# ===========================================================================
# Step 1 -- reproduce the round-2 S1 winners' per-chance predictions
# ===========================================================================
def step_score() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    design = pd.read_parquet(ROUND2_DIR / "design.parquet")
    print(f"design {design.shape} from round2 cache", flush=True)

    for pop, spec in ROUND2_WINNERS.items():
        dst = OUT_DIR / f"pred_{pop}_F2_S1.npz"
        if dst.exists():
            print(f"  {pop}: {dst.name} exists, skipping", flush=True)
            continue
        feats = PO.feature_set(FEATURE_SET, pop)
        tr, te = PO.fold_slices(design, FOLD, pop)
        t0 = time.time()
        p, meta = PO.fit_predict_scheme(spec["arm"], "S1", tr, te, feats, seed=0)
        ll = PO.log_loss(te["y"].to_numpy(), p)
        dt = time.time() - t0
        print(f"  {pop}: {spec['arm']} S1  log loss {ll:.6f} "
              f"(round 2 recorded {spec['log_loss']:.6f})  [{dt:.0f}s, {meta['n_fits']} fits]",
              flush=True)
        if abs(ll - spec["log_loss"]) > 1e-6:
            raise AssertionError(
                f"{pop}: reproduced F2 log loss {ll:.6f} != round 2's {spec['log_loss']:.6f}. "
                "The diagnostic would not be about the round-2 winner; stopping.")
        np.savez_compressed(
            dst,
            p=p.astype("float32"),
            y=te["y"].to_numpy().astype("int8"),
            game_id=te["game_id"].to_numpy(),
            game_date=te["game_date"].to_numpy().astype("datetime64[D]"),
            offense_team_id=te["offense_team_id"].to_numpy(),
            defense_team_id=te["defense_team_id"].to_numpy(),
            off_rating_off_c=te["off_rating_off_c"].to_numpy().astype("float32"),
            off_3pa_c=te["off_3pa_c"].to_numpy().astype("float32"),
            off_rim_c=te["off_rim_c"].to_numpy().astype("float32"),
            off_tov_c=te["off_tov_c"].to_numpy().astype("float32"),
            off_ftr_c=te["off_ftr_c"].to_numpy().astype("float32"),
        )
        (OUT_DIR / f"meta_{pop}_F2_S1.json").write_text(
            json.dumps({"arm": spec["arm"], "scheme": "S1", "log_loss": round(ll, 6),
                        "n_test": int(len(te)), "meta": meta,
                        "reproduces_round2": True}, indent=1, default=str))


# ===========================================================================
# Step 2 -- residual structure
# ===========================================================================
def load_preds(pop: str) -> pd.DataFrame:
    z = np.load(OUT_DIR / f"pred_{pop}_F2_S1.npz", allow_pickle=False)
    p = z["p"].astype("float64")
    d = pd.DataFrame({
        "game_id": z["game_id"], "y": z["y"],
        "game_date": pd.to_datetime(z["game_date"]),
        "offense_team_id": z["offense_team_id"],
        "defense_team_id": z["defense_team_id"],
        "off_rating_off_c": z["off_rating_off_c"],
        "off_3pa_c": z["off_3pa_c"], "off_rim_c": z["off_rim_c"],
        "off_tov_c": z["off_tov_c"], "off_ftr_c": z["off_ftr_c"],
    })
    for j, c in enumerate(PO.CLASSES):
        d[f"p_{c}"] = p[:, j]
        d[f"a_{c}"] = (d["y"].to_numpy() == j).astype("float64")
        d[f"r_{c}"] = d[f"p_{c}"] - d[f"a_{c}"]
    return d


def attach_alignments(d: pd.DataFrame, season: int) -> pd.DataFrame:
    conf = build_conference_flags([season])
    firsts = first_conference_game_dates(conf)
    d = d.merge(conf[["game_id", "is_conf_game"]], on="game_id", how="left")
    n_missing = int(d["is_conf_game"].isna().sum())
    d["is_conf_game"] = d["is_conf_game"].fillna(False).astype(bool)

    f = firsts[firsts["season"] == season][["team_id", "first_conf_date"]]
    for side, col in (("off", "offense_team_id"), ("def", "defense_team_id")):
        m = f.rename(columns={"team_id": col, "first_conf_date": f"{side}_first_conf"})
        d = d.merge(m, on=col, how="left")
        d[f"w_rel_{side}"] = np.floor(
            (d["game_date"] - d[f"{side}_first_conf"]).dt.days / 7.0)

    season_start = d["game_date"].min()
    d["w_cal"] = ((d["game_date"] - season_start).dt.days // 7).astype("int32")

    cuts = PO.month_boundaries(d["game_date"])
    idx = np.searchsorted(np.array([c.to_datetime64() for c in cuts]),
                          d["game_date"].to_numpy(), side="right") - 1
    last_cut = pd.to_datetime(pd.Series([cuts[max(i, 0)] for i in idx]).to_numpy())
    d["w_since_refit"] = np.minimum(
        ((d["game_date"].to_numpy() - last_cut.to_numpy())
         / np.timedelta64(1, "D") // 7).astype("int64"), 4)
    d.attrs["n_missing_conf_flag"] = n_missing
    return d


ALIGNMENTS = {
    "calendar_week": ("w_cal", "calendar week of the test season"),
    "conf_rel_off": ("w_rel_off", "week relative to the OFFENCE team's first conference game"),
    "conf_rel_def": ("w_rel_def", "week relative to the DEFENCE team's first conference game"),
    "conf_flag": ("is_conf_game", "conference vs non-conference game"),
    "weeks_since_refit": ("w_since_refit", "weeks since the monthly S1 refit boundary"),
}
#: The conference-relative alignments are read over the pre-registered window
#: -8..+12 weeks; calendar week is clipped to the same COUNT of buckets so the
#: raw variance figures are on comparable footing before the permutation null
#: is even applied.
CONF_REL_WINDOW = (-8, 12)


def bucket_means(d: pd.DataFrame, key: str, classes=GATED_CLASSES) -> pd.DataFrame:
    g = d.groupby(key, sort=True)
    out = g.size().rename("n").to_frame()
    out["n_games"] = g["game_id"].nunique()
    for c in classes:
        out[f"resid_{c}"] = g[f"r_{c}"].mean() * 100.0   # pp
        out[f"pred_{c}"] = g[f"p_{c}"].mean() * 100.0
        out[f"act_{c}"] = g[f"a_{c}"].mean() * 100.0
    out["underpowered"] = out["n"] < MIN_BUCKET_CHANCES
    return out.reset_index()


def decile_gap(d: pd.DataFrame, classes=GATED_CLASSES) -> float:
    """The round-1 gate quantity on a subset: largest |predicted - actual|
    over the ten predicted-probability deciles, maximised over gated classes."""
    if len(d) < 1000:
        return float("nan")
    p = np.column_stack([d[f"p_{c}"].to_numpy() for c in PO.CLASSES])
    cal = PO.decile_calibration(d["y"].to_numpy(), p)
    gated = [v["max_abs_gap_pp"] for c, v in cal.items()
             if v["share_pct"] >= 5.0 and c in classes]
    return float(max(gated)) if gated else float("nan")


def weighted_between_var(bm: pd.DataFrame, classes=GATED_CLASSES) -> dict:
    """Weighted variance of the bucket-mean residual, over powered buckets."""
    ok = bm[~bm["underpowered"]]
    if len(ok) < 2:
        return {c: float("nan") for c in classes}
    w = ok["n"].to_numpy().astype("float64")
    w = w / w.sum()
    out = {}
    for c in classes:
        m = ok[f"resid_{c}"].to_numpy()
        mu = float((w * m).sum())
        out[c] = float((w * (m - mu) ** 2).sum())
    return out


def permutation_null(d: pd.DataFrame, key: str, n_perm: int = N_PERM,
                     seed: int = RNG_SEED, classes=GATED_CLASSES) -> dict:
    """Null distribution of the between-bucket variance when the bucket label
    is shuffled BETWEEN GAMES, preserving each bucket's game count.

    Shuffling at the game level, not the chance level, is the point: chances
    inside a game share lineups, officials and pace, so a chance-level shuffle
    would understate the null and make every alignment look significant."""
    rng = np.random.default_rng(seed)
    gk = d.groupby("game_id", sort=True)
    gsize = gk.size().to_numpy().astype("float64")
    labels = gk[key].first().to_numpy()
    # x100 so the null and the bootstrap are in the same PERCENTAGE-POINT
    # units as `bucket_means`; without it the two statistics differ by 1e4
    # and the null looks vanishing next to the point estimate.
    gsum = {c: gk[f"r_{c}"].sum().to_numpy() * 100.0 for c in classes}
    powered = set(d.groupby(key).size()[lambda s: s >= MIN_BUCKET_CHANCES].index.tolist())
    keep_mask = np.isin(labels, list(powered))
    draws = {c: [] for c in classes}
    for _ in range(n_perm):
        perm = rng.permutation(len(labels))
        lab = labels[perm]
        km = np.isin(lab, list(powered))
        idx = pd.Index(lab[km])
        n_b = pd.Series(gsize[km]).groupby(idx).sum()
        w = (n_b / n_b.sum()).to_numpy()
        for c in classes:
            s_b = pd.Series(gsum[c][km]).groupby(idx).sum()
            m = (s_b / n_b).to_numpy()
            mu = float((w * m).sum())
            draws[c].append(float((w * (m - mu) ** 2).sum()))
    _ = keep_mask
    return {c: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))}
            for c, v in draws.items()}


def bootstrap_var_ci(d: pd.DataFrame, key: str, n_rep: int = N_BOOT,
                     seed: int = RNG_SEED + 1, classes=GATED_CLASSES) -> dict:
    """Game-block bootstrap CI for the between-bucket variance."""
    rng = np.random.default_rng(seed)
    gk = d.groupby("game_id", sort=True)
    gsize = gk.size().to_numpy().astype("float64")
    labels = gk[key].first().to_numpy()
    # x100 so the null and the bootstrap are in the same PERCENTAGE-POINT
    # units as `bucket_means`; without it the two statistics differ by 1e4
    # and the null looks vanishing next to the point estimate.
    gsum = {c: gk[f"r_{c}"].sum().to_numpy() * 100.0 for c in classes}
    powered = set(d.groupby(key).size()[lambda s: s >= MIN_BUCKET_CHANCES].index.tolist())
    km = np.isin(labels, list(powered))
    labels, gsize = labels[km], gsize[km]
    gsum = {c: v[km] for c, v in gsum.items()}
    n_g = len(labels)
    draws = {c: [] for c in classes}
    for _ in range(n_rep):
        pick = rng.integers(0, n_g, n_g)
        idx = pd.Index(labels[pick])
        n_b = pd.Series(gsize[pick]).groupby(idx).sum()
        w = (n_b / n_b.sum()).to_numpy()
        for c in classes:
            s_b = pd.Series(gsum[c][pick]).groupby(idx).sum()
            m = (s_b / n_b).to_numpy()
            mu = float((w * m).sum())
            draws[c].append(float((w * (m - mu) ** 2).sum()))
    return {c: {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
            for c, v in draws.items()}


def style_boundary_table(d: pd.DataFrame) -> pd.DataFrame:
    """Per own-rating quintile, the mean raw-centred style feature in
    non-conference vs conference games. Aggregated to one row per team-game on
    the OFFENCE side first, so a slow team's games do not weigh less than a
    fast team's."""
    tg = d.groupby(["game_id", "offense_team_id"], as_index=False).agg(
        is_conf_game=("is_conf_game", "first"),
        own_rating=("off_rating_off_c", "first"),
        off_3pa_c=("off_3pa_c", "first"), off_rim_c=("off_rim_c", "first"),
        off_tov_c=("off_tov_c", "first"), off_ftr_c=("off_ftr_c", "first"),
        n_chances=("y", "size"))
    team_mean = tg.groupby("offense_team_id")["own_rating"].mean()
    q = pd.qcut(team_mean, 5, labels=[1, 2, 3, 4, 5])
    tg["quintile"] = tg["offense_team_id"].map(q).astype("int8")
    rows = []
    for qi in sorted(tg["quintile"].unique()):
        s = tg[tg["quintile"] == qi]
        row = {"quintile": int(qi), "n_team_games": len(s),
               "n_teams": int(s["offense_team_id"].nunique()),
               "own_rating_mean": round(float(team_mean[q[q == qi].index].mean()), 2),
               "n_nonconf": int((~s["is_conf_game"]).sum()),
               "n_conf": int(s["is_conf_game"].sum())}
        for f in ("off_3pa_c", "off_rim_c", "off_tov_c", "off_ftr_c"):
            nc = float(s.loc[~s["is_conf_game"], f].mean())
            cf = float(s.loc[s["is_conf_game"], f].mean())
            row[f"{f}_nonconf"] = round(nc, 3)
            row[f"{f}_conf"] = round(cf, 3)
            row[f"{f}_shift"] = round(cf - nc, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    cols = cols or list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append("" if not np.isfinite(v) else f"{v:.3f}")
            elif c == "underpowered":
                vals.append("UNDERPOWERED" if bool(v) else "")
            elif isinstance(v, (bool, np.bool_)):
                vals.append("yes" if v else "no")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return out


def step_analyse() -> None:
    L: list[str] = []
    A = L.append
    A("# Possession outcome, Stage A: is the surviving residual structure "
      "organised by the calendar or by the conference boundary?")
    A("")
    A("Diagnostic for `ARCHITECTURE_DECISIONS.md` Decision 9, run 2026-09-10 by the "
      "possession-outcome worker BEFORE the round-3 pre-registration was written. "
      "No model is selected here and nothing is adopted.")
    A("")
    A("Scored population: the **round-2 S1 winners** re-scored on fold 2 with the "
      "round-2 code path -- `lgbm` (first chances) and `cascade` (continuation chances), "
      "feature set `C_plus_state`, scheme S1 (calendar-monthly), seed 0, through "
      "`cbb_sim.models.possession_outcome.fit_predict_scheme`. Round 2 did not persist "
      "per-chance predictions, so they were regenerated; the run asserts the reproduced "
      "F2 log loss equals round 2's recorded value to 1e-6 and stops otherwise.")
    A("")

    summary_rows = []
    per_pop_json: dict = {}
    for pop in ("first", "cont"):
        meta = json.loads((OUT_DIR / f"meta_{pop}_F2_S1.json").read_text())
        d = load_preds(pop)
        d = attach_alignments(d, TEST_SEASON)
        A("---")
        A("")
        A(f"## Population `{pop}` ({ROUND2_WINNERS[pop]['arm']}, S1 monthly, "
          f"n = {len(d):,} chances over {d['game_id'].nunique():,} games)")
        A("")
        A(f"Reproduced F2 log loss {meta['log_loss']:.6f} against round 2's recorded "
          f"{ROUND2_WINNERS[pop]['log_loss']:.6f} -- exact match, so this is round 2's model.")
        A(f"Games with no hoopR conference id on either side: {d.attrs['n_missing_conf_flag']:,} "
          f"chances (treated as non-conference and reported, never imputed as conference).")
        A("")

        pop_json: dict = {"n": int(len(d)), "n_games": int(d["game_id"].nunique())}
        for name, (key, label) in ALIGNMENTS.items():
            sub = d
            if name.startswith("conf_rel"):
                sub = d[d[key].between(*CONF_REL_WINDOW)].copy()
            bm = bucket_means(sub, key)
            var = weighted_between_var(bm)
            null = permutation_null(sub, key)
            ci = bootstrap_var_ci(sub, key)
            pop_json[name] = {"var": var, "null": null, "ci": ci,
                              "n_buckets": int(len(bm)),
                              "n_powered": int((~bm["underpowered"]).sum())}
            for c in GATED_CLASSES:
                summary_rows.append({
                    "population": pop, "alignment": name, "class": c,
                    "n_buckets": int(len(bm)), "n_powered": int((~bm["underpowered"]).sum()),
                    "var_pp2": var[c], "null_mean_pp2": null[c]["mean"],
                    "null_p95_pp2": null[c]["p95"],
                    "excess_pp2": var[c] - null[c]["mean"],
                    "rms_pp": float(np.sqrt(max(var[c], 0.0))),
                    "excess_rms_pp": float(np.sqrt(max(var[c] - null[c]["mean"], 0.0))),
                    "boot_lo": ci[c]["lo"], "boot_hi": ci[c]["hi"],
                    "beats_null_p95": bool(var[c] > null[c]["p95"]),
                })
            A(f"### {label} (`{name}`)")
            A("")
            cols = ["n", "n_games", "underpowered"] + \
                   [f"resid_{c}" for c in ("TOV", "FGA_rim", "FGA_jump2", "FGA_3")]
            bm2 = bm.copy()
            bm2 = bm2[[key] + cols]
            A("Bucket-mean residual, predicted minus actual, percentage points. "
              "Positive = the model over-predicts that class in that bucket.")
            A("")
            L.extend(md_table(bm2))
            A("")
        per_pop_json[pop] = pop_json

        # decile gap in the first four weeks of conference play
        A("### Per-decile calibration gap, overall vs the first four weeks of "
          "conference play (offence-team alignment)")
        A("")
        seg = {
            "overall": d,
            "non-conference games": d[~d["is_conf_game"]],
            "conference games": d[d["is_conf_game"]],
            "first 4 conf weeks (w_rel_off 0..3)": d[d["w_rel_off"].between(0, 3)],
            "conf weeks 4+ (w_rel_off >= 4)": d[d["w_rel_off"] >= 4],
            "pre-boundary (w_rel_off < 0)": d[d["w_rel_off"] < 0],
        }
        rows = [{"segment": k, "n": len(v), "n_games": int(v["game_id"].nunique()),
                 "worst_gated_decile_gap_pp": decile_gap(v)} for k, v in seg.items()]
        L.extend(md_table(pd.DataFrame(rows)))
        A("")
        pop_json["decile_gap_by_segment"] = {r["segment"]: r["worst_gated_decile_gap_pp"]
                                             for r in rows}

        if pop == "first":
            A("### The Mississippi Valley State table: raw-centred style rates at the "
              "conference boundary, by own-rating quintile")
            A("")
            A("One row per team-game on the offence side; quintiles are of the team's "
              "season-mean as-of `off_rating_off_c` (a DESCRIPTIVE stratifier, not a "
              "model feature). `shift` = conference mean minus non-conference mean, in "
              "the feature's own centred units (per-100-possession for 3PA and TOV, "
              "percentage-point share for rim and FTr).")
            A("")
            st = style_boundary_table(d)
            L.extend(md_table(st))
            A("")
            st.to_csv(OUT_DIR / "style_boundary_by_quintile.csv", index=False)
            pop_json["style_boundary"] = st.to_dict("records")

    sm = pd.DataFrame(summary_rows)
    sm.to_csv(OUT_DIR / "alignment_variance.csv", index=False)
    (OUT_DIR / "stageA_summary.json").write_text(json.dumps(per_pop_json, indent=1))

    A("---")
    A("")
    A("## Verdict: which alignment carries the residual structure")
    A("")
    A("`var` is the chance-weighted variance of the bucket-mean residual over POWERED "
      "buckets, in pp^2. Because the alignments have different bucket counts, the raw "
      "variance is not comparable across them; `null` is the same statistic under 200 "
      "shuffles of the bucket label BETWEEN GAMES (preserving bucket sizes), which is what "
      "makes them comparable, and `excess_rms` = sqrt(var - null_mean) is the reportable "
      "quantity in pp. `boot` is a 200-replicate game-block bootstrap CI on `var`.")
    A("")
    for pop in ("first", "cont"):
        A(f"### `{pop}`")
        A("")
        s = sm[sm["population"] == pop].copy()
        s = s[["alignment", "class", "n_buckets", "n_powered", "var_pp2", "null_mean_pp2",
               "null_p95_pp2", "excess_rms_pp", "boot_lo", "boot_hi", "beats_null_p95"]]
        L.extend(md_table(s))
        A("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["score", "analyse", "all"], default="all")
    a = ap.parse_args()
    if a.step in ("score", "all"):
        step_score()
    if a.step in ("analyse", "all"):
        step_analyse()


if __name__ == "__main__":
    main()
