#!/usr/bin/env python
"""
grade_daily_v1.py -- daily chain v3, stage GRADE (2026-09-30, lane F).

Next-day grading of a published slate against VERIFIED finals only (never pbp-accumulated totals): margin / total MAE and bias,
calibration, ATS / OU ROI by disagreement bucket at the flat -110 the harness uses, moneyline ROI by edge bucket at the REAL odds
stored at publish time. Rows are appended to the running ledger `results/daily/grade/ledger.parquet` (upsert on game_id, run_id,
publish_id: idempotent). A game with no verified final stays PENDING (written to the day's pending file) and is retried by the
next grade run. The publication graded is the EARLIEST one per game (the line that was actually available), unless --publish-id.

Finals sources:
  ingest  data/processed/ingest/finals_verified_{season}.parquet (daily ingestion: hoopR and CBBD agree on both scores)
  truth   REPLAY: cbb_sim.eval.reference.load_actual_games(season, verified_finals=True) (CBB_TRUTH=verified_v1), the eval harness's truth

    .venv/Scripts/python.exe scripts/grade_daily_v1.py --slate-date 2025-02-11 --season 2025 --finals truth --now 2025-02-12T14:00:00Z
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from cbb_sim.live.preseason import preseason_dir as _preseason_dir, preseason_rel as _preseason_rel  # noqa: E402,F401

from cbb_sim.live import daily as D  # noqa: E402

TRUTH_BOX = REPO / "data/processed/truth/team_game_shots_v2.parquet"


def finals_ingest(season: int, ingest_dir: Path | None = None) -> pd.DataFrame:
    p = Path(ingest_dir or REPO / "data/processed/ingest") / f"finals_verified_{int(season)}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["game_id", "home_score", "away_score", "finals_source", "verified_at"])
    f = pd.read_parquet(p)
    agree = (f["hoopr_home"] == f["cbbd_home"]) & (f["hoopr_away"] == f["cbbd_away"])
    f = f[agree & f["hoopr_home"].notna() & f["hoopr_away"].notna() & ~((f["hoopr_home"] == 0) & (f["hoopr_away"] == 0))]
    return pd.DataFrame({"game_id": f["game_id"].astype("int64"), "home_score": f["hoopr_home"].astype("int64"),
                         "away_score": f["hoopr_away"].astype("int64"), "finals_source": "ingest_verified_two_source",
                         "verified_at": pd.to_datetime(f["verified_at"], utc=True)}).drop_duplicates("game_id", keep="last")


def finals_truth(season: int) -> pd.DataFrame:
    os.environ["CBB_TRUTH"] = "verified_v1"
    from cbb_sim.eval import reference as ref
    a = ref.load_actual_games(int(season), verified_finals=True)
    return pd.DataFrame({"game_id": a["game_id"].astype("int64"), "home_score": a["home_score"].astype("int64"),
                         "away_score": a["away_score"].astype("int64"), "n_periods": a["n_periods"],
                         "finals_source": "replay_truth_verified_v1", "verified_at": pd.NaT})


def load_publications(root: Path, slate_date: str, run_id: str | None = None) -> pd.DataFrame:
    base = Path(root) / "publish" / str(slate_date)
    paths = sorted(base.glob(f"{run_id or '*'}/*/slate.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def season_start_of(season: int) -> str:
    """First D-I game date of the season (ET). 2027+: the preseason schedule; earlier: the verified games universe."""
    if int(season) >= 2027:
        g = pd.read_parquet(_preseason_dir() / "games_2027.parquet", columns=["startDate"])
        d = pd.to_datetime(g["startDate"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        return str(d.min().date())
    u = pd.read_parquet(REPO / "data/processed/games_universe.parquet", columns=["season", "game_date", "is_d1_game"])
    return str(pd.to_datetime(u.loc[(u["season"] == int(season)) & u["is_d1_game"], "game_date"]).min().date())


def tov_monitor_block(led: pd.DataFrame, root: Path, season: int, odir: Path) -> list[str]:
    """REPORT ONLY (cbb_sim.live.tov_monitor). Never raises: a monitor failure is written into the report, it must not block grading."""
    from cbb_sim.live import tov_monitor as TM
    try:
        start = season_start_of(season)
        res = TM.run_tov_monitor(led, root, season, start)
        if "table" in res:
            res["table"].to_csv(odir / "tov_monitor.csv", index=False)
        return TM.report_block(res, start)
    except Exception as e:  # noqa: BLE001
        return ["## TOV level monitor (report only)", "", f"- monitor error (grading unaffected): {type(e).__name__}: {str(e)[:200]}", ""]


def report_markdown(slate_date, tabs_day: dict, tabs_cum: dict, pending: pd.DataFrame, n_new: int, finals_source: str, now) -> str:
    def block(t: dict, title: str) -> list[str]:
        o = [f"## {title}", "", f"- games graded: {t['n_games']}  ({D.underpowered(t['n_games'])})",
             f"- margin MAE {t['margin_mae']:.3f}, bias (sim - actual) {t['margin_bias']:+.3f};  total MAE {t['total_mae']:.3f}, bias {t['total_bias']:+.3f}"]
        if t.get("n_with_spread"):
            o.append(f"- lined games {t['n_with_spread']}: model margin MAE {t['model_margin_mae_lined']:.3f} vs line {t['line_margin_mae_lined']:.3f}")
        if t.get("n_with_total"):
            o.append(f"- lined totals {t['n_with_total']}: model total MAE {t['total_mae_lined']:.3f} vs line {t['line_total_mae_lined']:.3f}")
        if "model_brier" in t:
            o.append(f"- Brier (moneyline subset n={t['n_with_ml']}): model {t['model_brier']:.4f}, de-vigged market {t['market_brier']:.4f}")
        for k, name in (("ats_table", "ATS by disagreement (pts), flat -110"), ("ou_table", "O/U by disagreement (pts), flat -110"),
                        ("ml_edge_table", "Moneyline by edge bucket at the real stored odds")):
            if k in t:
                o += ["", f"**{name}**", "", D.fmt_df(t[k])]
        for k, name in (("ats_bootstrap", "ATS"), ("ou_bootstrap", "O/U"), ("ml_bootstrap", "ML")):
            if k in t and t[k]["n_games"]:
                b = t[k]
                o.append(f"- {name} bootstrap ({b['n_games']} games): mean ROI {b['mean']:+.4f}, 95% CI [{b['lo95']:+.4f}, {b['hi95']:+.4f}]"
                         f"{'  UNDERPOWERED' if b['n_games'] < 300 else ''}")
        if "calibration" in t:
            o += ["", "**Calibration deciles (model p(home) vs de-vigged market vs actual)**", "", D.fmt_df(t["calibration"])]
        return o + [""]
    L_ = [f"# Grade, slate {slate_date}", "", f"Graded at {D.utc(now)}. Finals source: {finals_source} (verified finals only). "
          f"New ledger rows this run: {n_new}. Pending (no verified final): {len(pending)}.", ""]
    L_ += block(tabs_day, f"Slate {slate_date}") + block(tabs_cum, "Running ledger (all graded slates)")
    if len(pending):
        L_ += ["## Pending", "", ", ".join(str(int(g)) for g in pending["game_id"]), ""]
    return "\n".join(L_)


def run_grade_stage(slate_date: str, season: int, now=None, root: Path = D.DAILY_ROOT, finals: str = "ingest",
                    run_id: str | None = None, publish_id: str | None = None, finals_frame: pd.DataFrame | None = None,
                    truth_box: pd.DataFrame | None = None, n_boot: int = 1000, tov_monitor: bool = True) -> dict:
    now = D.utc(now) if now is not None else pd.Timestamp.now("UTC")
    pubs = load_publications(root, slate_date, run_id)
    if not len(pubs):
        return {"_status": "skipped", "why": f"no publication for {slate_date}"}
    pub = D.first_publication(pubs, publish_id)
    fin = finals_frame if finals_frame is not None else (finals_truth(season) if finals == "truth" else finals_ingest(season))
    fin = fin[fin["game_id"].isin(set(pub["game_id"]))]
    if not (D.utc(now) > pd.to_datetime(pub["tipoff_utc"], utc=True).min()):
        return {"_status": "skipped", "why": "grade clock is not after the first tip; nothing can be final"}
    led_new = D.settle(pub, fin, now)
    if len(led_new) and not (pd.to_datetime(led_new["tipoff_utc"], utc=True) < D.utc(now)).all():
        raise RuntimeError("a final exists for a game that has not tipped by the grade clock")
    if truth_box is None and TRUTH_BOX.exists():
        truth_box = pd.read_parquet(TRUTH_BOX)
        truth_box = truth_box[truth_box["game_id"].isin(set(led_new["game_id"]))]
    led_new = D.attach_truth_box(led_new, truth_box)
    pend = pub[~pub["game_id"].isin(set(led_new["game_id"]))][["game_id", "cbbd_game_id", "tipoff_utc", "run_id", "publish_id"]].copy()
    pend["reason"] = "no_verified_final"
    gdir = Path(root) / "grade"
    led, n_new = D.ledger_upsert(gdir / "ledger.parquet", led_new) if len(led_new) else (
        pd.read_parquet(gdir / "ledger.parquet") if (gdir / "ledger.parquet").exists() else led_new, 0)
    day = led[pd.to_datetime(led["game_date"]).dt.strftime("%Y-%m-%d") == str(slate_date)]
    t_day, t_cum = D.grade_tables(day, n_boot), D.grade_tables(led, n_boot)
    odir = gdir / str(slate_date)
    odir.mkdir(parents=True, exist_ok=True)
    pend.to_parquet(odir / "pending.parquet", index=False)
    fsrc = fin["finals_source"].iloc[0] if len(fin) else finals
    rep = report_markdown(slate_date, t_day, t_cum, pend, n_new, fsrc, now)
    if tov_monitor:
        rep = rep.rstrip(chr(10)) + chr(10) * 2 + chr(10).join(tov_monitor_block(led, Path(root), season, odir))
    (odir / "report.md").write_text(rep, encoding="utf-8")
    summary = {k: v for k, v in t_day.items() if not isinstance(v, (pd.DataFrame, dict))}
    (odir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"_status": "ok", "graded": int(len(led_new)), "new_ledger_rows": int(n_new), "pending": int(len(pend)),
            "ledger_rows": int(len(led)), "margin_mae": t_day["margin_mae"], "total_mae": t_day["total_mae"], "report": str(odir / "report.md")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slate-date", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--now", default=None)
    ap.add_argument("--root", default=str(D.DAILY_ROOT))
    ap.add_argument("--finals", choices=("ingest", "truth"), default="ingest")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--publish-id", default=None)
    ap.add_argument("--no-tov-monitor", dest="tov_monitor", action="store_false", help="skip the report-only TOV level monitor (default on)")
    a = ap.parse_args(argv)
    print(json.dumps(run_grade_stage(a.slate_date, a.season, a.now, Path(a.root), a.finals, a.run_id, a.publish_id, tov_monitor=a.tov_monitor), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
