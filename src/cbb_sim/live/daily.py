"""Daily-chain v3 core: publish frame, grade ledger, bias / CLV monitor. Pure functions over frames (no network, no engine).

Rules this module enforces (CLAUDE.md):
  * No decision-layer calibration and no adjustment of sim output: every published number is a direct summary of the per-seed
    sim rows (mean, SD, quantiles, exceedance frequencies). The monitor REPORTS and ALARMS, it never corrects.
  * Every published / graded row carries `created_at` (sim build) and `published_at`; `published_at < tipoff_utc` is asserted.
  * Grading is against VERIFIED finals only; a game without one stays pending.
  * Bets settle at the REAL posted line/odds stored at publish time; de-vig is for probability comparison only.
    Spread / total prices are not in CBBD: flat -110 (the eval harness's convention).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.eval import market as M
from cbb_sim.live import guards as G
from cbb_sim.live import lines as L

REPO = Path(__file__).resolve().parents[3]
DAILY_ROOT = REPO / "results" / "daily"
QS = (0.05, 0.25, 0.50, 0.75, 0.95)
STATS = ("fga3", "fga2_rim", "fga2_jump", "fta", "fgm3", "fgm2_rim", "fgm2_jump", "ftm")
#: sim stat -> team_game_shots_v2 event-layer truth column
TRUTH_COL = {"fga3": "ev_fga_3", "fga2_rim": "ev_fga_rim", "fga2_jump": "ev_fga_jump2", "fta": "ev_fta",
             "fgm3": "ev_fgm_3", "fgm2_rim": "ev_fgm_rim", "fgm2_jump": "ev_fgm_jump2", "ftm": "ev_ftm"}
RULE_TEXT = ("CLAUDE.md: an edge that beats the close but cannot predict line movement is presumed leaked "
             "(information that beats the close but never moves the line is leaked information). "
             "This monitor reports and alarms; it never corrects sim output.")


# ------------------------------------------------------------------------------------------------ paths / clock
def utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def sim_dir(root: Path, slate_date, run_id: str) -> Path:
    return Path(root) / "sim" / str(slate_date) / run_id


def pub_dir(root: Path, slate_date, run_id: str) -> Path:
    return Path(root) / "publish" / str(slate_date) / run_id


def default_run_id(seeds: int, seed_offset: int = 0) -> str:
    return f"s{int(seeds)}_o{int(seed_offset)}"


def split_tipped(slate: pd.DataFrame, now) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(not yet tipped, already tipped) by the injected clock. A game with no usable tip time counts as tipped (cannot be proven pregame)."""
    now = utc(now)
    tip = pd.to_datetime(slate["tipoff_utc"], utc=True)
    late = tip.isna() | ~(now < tip)
    return slate[~late].reset_index(drop=True), slate[late].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ sim summary
def sim_game_summary(games: pd.DataFrame) -> pd.DataFrame:
    """Per game_id: distribution summaries of margin (home - away) and total, home win frequency, OT rate, per-team box means.
    Direct statistics of the per-seed rows; nothing is fitted or adjusted."""
    s = games.copy()
    s["margin"] = s["home_pts"].astype("int64") - s["away_pts"].astype("int64")
    s["total"] = s["home_pts"].astype("int64") + s["away_pts"].astype("int64")
    rows = []
    for gid, g in s.groupby("game_id", sort=True):
        m, t = g["margin"].to_numpy(), g["total"].to_numpy()
        r = {"game_id": int(gid), "n_seeds": int(len(g)),
             "sim_margin_mean": float(m.mean()), "sim_margin_sd": float(m.std(ddof=1)) if len(m) > 1 else np.nan,
             "sim_total_mean": float(t.mean()), "sim_total_sd": float(t.std(ddof=1)) if len(t) > 1 else np.nan,
             "sim_home_pts_mean": float(g["home_pts"].mean()), "sim_away_pts_mean": float(g["away_pts"].mean()),
             "sim_poss_mean": float(g["possessions"].mean()),
             "p_home": float(np.where(m > 0, 1.0, np.where(m < 0, 0.0, 0.5)).mean()),
             "sim_ot_rate": float((g["n_periods"] > 2).mean())}
        for q in QS:
            r[f"sim_margin_q{int(q * 100):02d}"] = float(np.quantile(m, q))
            r[f"sim_total_q{int(q * 100):02d}"] = float(np.quantile(t, q))
        for st in STATS:
            for side in ("home", "away"):
                c = f"{side}_{st}"
                if c in g.columns:
                    r[f"sim_{c}_mean"] = float(g[c].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def line_probabilities(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Per game with a line: sim frequencies of cover / over at the posted number (pushes tracked separately)."""
    ln = lines.set_index("game_id")
    out = []
    for gid, g in games.groupby("game_id", sort=True):
        if gid not in ln.index:
            continue
        m = (g["home_pts"].astype("int64") - g["away_pts"].astype("int64")).to_numpy()
        t = (g["home_pts"].astype("int64") + g["away_pts"].astype("int64")).to_numpy()
        r = {"game_id": int(gid)}
        sp, tot = ln.at[gid, "spread"], ln.at[gid, "total"]
        if pd.notna(sp):
            x = m + sp                                  # home covers when margin + spread > 0
            r.update(p_home_cover=float((x > 0).mean()), p_away_cover=float((x < 0).mean()), p_ats_push=float((x == 0).mean()))
        if pd.notna(tot):
            r.update(p_over=float((t > tot).mean()), p_under=float((t < tot).mean()), p_ou_push=float((t == tot).mean()))
        out.append(r)
    return pd.DataFrame(out)


def _ev_110(p_win, p_loss):
    """EV per unit risked at flat -110 (win +100/110 per unit risked), pushes return the stake."""
    return p_win * (100.0 / 110.0) - p_loss


# ------------------------------------------------------------------------------------------------ publish
def build_publish(games: pd.DataFrame, slate: pd.DataFrame, lines: pd.DataFrame | None, now, run_id: str,
                  sim_created_at=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (published rows, refused rows). `slate` carries game_id, cbbd_game_id, game_date, tipoff_utc, home/away team ids,
    neutral (and optionally home_name / away_name). A game that has tipped by `now` is refused. Asserts published_at < tipoff."""
    now = utc(now)
    ok, late = split_tipped(slate, now)
    refused = late.assign(refused_reason="already_tipped_at_publish", published_at=now)
    summ = sim_game_summary(games[games["game_id"].isin(set(ok["game_id"]))])
    pub = ok.merge(summ, on="game_id", how="inner")
    if sim_created_at is None and "created_at" in games.columns:
        sim_created_at = games.groupby("game_id")["created_at"].first().reindex(pub["game_id"]).to_numpy()
    pub["created_at"] = pd.to_datetime(sim_created_at, utc=True) if sim_created_at is not None else pd.NaT
    pub["published_at"] = now
    pub["run_id"] = run_id
    if lines is not None and len(lines):
        ln = lines.rename(columns={"cbbd_game_id": "cbbd_game_id"})
        pub = pub.merge(ln, on="cbbd_game_id", how="left")
        probs = line_probabilities(games, pub[["game_id", "spread", "total"]].dropna(how="all", subset=["spread", "total"]))
        if len(probs):
            pub = pub.merge(probs, on="game_id", how="left")
    else:
        for c in L.LINE_COLS[1:]:
            pub[c] = np.nan
    for c in ("provider", "spread", "total", "home_ml", "away_ml", "spread_open", "total_open", "line_fetched_at", "line_kind",
              "ml_is_close_proxy", "p_home_cover", "p_away_cover", "p_ats_push", "p_over", "p_under", "p_ou_push"):
        if c not in pub.columns:
            pub[c] = np.nan
    pub["market_margin"] = -pub["spread"]
    pub["market_p_home_devig"], pub["ml_vig"] = L.devig(pub["home_ml"], pub["away_ml"])
    pub["ats_disagree_pts"] = pub["sim_margin_mean"] - pub["market_margin"]
    pub["ou_disagree_pts"] = pub["sim_total_mean"] - pub["total"]
    pub["ml_edge"] = pub["p_home"] - pub["market_p_home_devig"]
    pub["ats_lean"] = np.where(pub["ats_disagree_pts"].isna(), "", np.where(pub["ats_disagree_pts"] > 0, "home", np.where(pub["ats_disagree_pts"] < 0, "away", "")))
    pub["ou_lean"] = np.where(pub["ou_disagree_pts"].isna(), "", np.where(pub["ou_disagree_pts"] > 0, "over", np.where(pub["ou_disagree_pts"] < 0, "under", "")))
    pub["ml_lean"] = np.where(pub["ml_edge"].isna(), "", np.where(pub["ml_edge"] > 0, "home", np.where(pub["ml_edge"] < 0, "away", "")))
    pub["ev_ats_home_110"] = _ev_110(pub["p_home_cover"], pub["p_away_cover"])
    pub["ev_ats_away_110"] = _ev_110(pub["p_away_cover"], pub["p_home_cover"])
    pub["ev_over_110"] = _ev_110(pub["p_over"], pub["p_under"])
    pub["ev_under_110"] = _ev_110(pub["p_under"], pub["p_over"])
    pub["line_fetched_at"] = pd.to_datetime(pub["line_fetched_at"], utc=True)
    has_ln = pub["line_fetched_at"].notna()
    if (pub.loc[has_ln, "line_fetched_at"] > now).any():
        raise G.LeakGuardError("a line is stamped after the publish clock")
    G.assert_created_before_tipoff(pub.assign(created_at=pub["published_at"]))      # published_at < tipoff
    if pub["created_at"].notna().any() and (pub["created_at"] > pub["published_at"]).any():
        raise G.LeakGuardError("sim created_at is after published_at")
    pub["publish_id"] = publish_id(pub, run_id)
    return pub, refused


def publish_id(pub: pd.DataFrame, run_id: str) -> str:
    cols = ["game_id", "provider", "spread", "total", "home_ml", "away_ml", "line_kind"]
    h = hashlib.sha1(run_id.encode())
    h.update(pub[cols].astype(str).sort_values("game_id").to_csv(index=False).encode())
    return h.hexdigest()[:10]


def _nm(r, side):
    n = r.get(f"{side}_name")
    return str(n) if isinstance(n, str) and n else str(int(r[f"{side}_team_id"]))


def slate_markdown(pub: pd.DataFrame, refused: pd.DataFrame, slate_date, run_id: str, now) -> str:
    L_ = [f"# Slate {slate_date}  (run {run_id}, published {utc(now).strftime('%Y-%m-%d %H:%MZ')})", "",
          "Margin is home minus away. Spread is home-perspective (negative = home favoured). Spread / total prices are not in CBBD: flat -110 is assumed. "
          "Probabilities are raw sim frequencies, not calibrated; nothing here adjusts sim output.", "",
          "| tip (UTC) | game | sim margin (SD) | sim total (SD) | p(home) | line | prov | ATS | O/U | ML edge |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in pub.sort_values("tipoff_utc").iterrows():
        line = "none" if pd.isna(r["spread"]) and pd.isna(r["total"]) else f"{r['spread']:+.1f} / {r['total']:.1f}" if pd.notna(r["spread"]) and pd.notna(r["total"]) else "partial"
        ats = f"{r['ats_lean']} {abs(r['ats_disagree_pts']):.1f}" if r["ats_lean"] else ""
        ou = f"{r['ou_lean']} {abs(r['ou_disagree_pts']):.1f}" if r["ou_lean"] else ""
        mle = f"{r['ml_lean']} {abs(r['ml_edge']):.3f}" if r["ml_lean"] else ""
        L_.append(f"| {pd.Timestamp(r['tipoff_utc']).strftime('%m-%d %H:%M')} | {_nm(r, 'away')} @ {_nm(r, 'home')}{' (N)' if r.get('neutral') else ''} | "
                  f"{r['sim_margin_mean']:+.1f} ({r['sim_margin_sd']:.1f}) | {r['sim_total_mean']:.1f} ({r['sim_total_sd']:.1f}) | {r['p_home']:.3f} | "
                  f"{line} | {r['provider'] if isinstance(r['provider'], str) else ''} | {ats} | {ou} | {mle} |")
    if len(refused):
        L_ += ["", f"Refused (already tipped at publish time): {len(refused)} game(s): " + ", ".join(str(int(g)) for g in refused["game_id"])]
    kinds = sorted({str(k) for k in pub["line_kind"].dropna().unique()})
    L_ += ["", f"Lines: kind {kinds or 'none'}; {int(pub['spread'].notna().sum())} of {len(pub)} games have a spread, "
           f"{int(pub['home_ml'].notna().sum())} a moneyline. Line timestamps are stored in the parquet (`line_fetched_at`)."]
    return "\n".join(L_) + "\n"


# ------------------------------------------------------------------------------------------------ grade
def settle(pub: pd.DataFrame, fin: pd.DataFrame, graded_at) -> pd.DataFrame:
    """Join publish rows to verified finals and settle at the real stored lines. `fin`: game_id, home_score, away_score,
    optional n_periods / finals_source / verified_at. Rows without a final are NOT returned (they stay pending)."""
    d = pub.merge(fin, on="game_id", how="inner", suffixes=("", "_fin"))
    d["margin"] = d["home_score"].astype("int64") - d["away_score"].astype("int64")
    d["total_pts"] = d["home_score"].astype("int64") + d["away_score"].astype("int64")
    d["is_home_win"] = (d["margin"] > 0).astype(float)
    d["margin_err"] = d["sim_margin_mean"] - d["margin"]
    d["total_err"] = d["sim_total_mean"] - d["total_pts"]
    # ATS at the stored spread, flat -110: win +1, loss -1.1, push / no bet 0 and flagged
    cover = d["margin"] - d["market_margin"]                       # > 0 home covered
    side = np.sign(d["ats_disagree_pts"])
    res = np.sign(cover) * side
    d["ats_cover_margin"] = cover
    d["ats_result"] = np.where(side.isna() | (side == 0), "none", np.where(res == 1, "win", np.where(res == -1, "loss", "push")))
    d["ats_pnl"] = np.where(d["ats_result"] == "win", 1.0, np.where(d["ats_result"] == "loss", -1.1, 0.0))
    over = d["total_pts"] - d["total"]
    sideo = np.sign(d["ou_disagree_pts"])
    reso = np.sign(over) * sideo
    d["ou_cover_margin"] = over
    d["ou_result"] = np.where(sideo.isna() | (sideo == 0), "none", np.where(reso == 1, "win", np.where(reso == -1, "loss", "push")))
    d["ou_pnl"] = np.where(d["ou_result"] == "win", 1.0, np.where(d["ou_result"] == "loss", -1.1, 0.0))
    # ML at the REAL odds
    has_ml = d["home_ml"].notna() & d["away_ml"].notna() & d["ml_edge"].notna()
    bet_home = d["ml_edge"] > 0
    odds = np.where(bet_home, d["home_ml"], d["away_ml"])
    won = np.where(bet_home, d["margin"] > 0, d["margin"] < 0)
    d["ml_pnl"] = np.where(has_ml, np.where(won, M.ml_profit_if_win(np.nan_to_num(odds, nan=-110.0)), -1.0), np.nan)
    d["graded_at"] = utc(graded_at)
    return d


def attach_truth_box(d: pd.DataFrame, truth: pd.DataFrame | None) -> pd.DataFrame:
    """Actual per-team event-layer stat counts from `team_game_shots_v2` (the verified-derived truth table). Missing -> NaN."""
    d = d.copy()
    for st in STATS:
        for side in ("home", "away"):
            d[f"act_{side}_{st}"] = np.nan
    if truth is None or not len(truth):
        return d
    t = truth.set_index(["game_id", "team_id"])
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([d["game_id"].astype("int64"), d[f"{side}_team_id"].astype("int64")])
        for st, col in TRUTH_COL.items():
            if col in t.columns:
                d[f"act_{side}_{st}"] = t[col].reindex(idx).to_numpy(dtype="float64")
    return d


KEYS = ["game_id", "run_id", "publish_id"]


def ledger_upsert(ledger_path: Path, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append-with-upsert on (game_id, run_id, publish_id). Idempotent: grading the same rows twice leaves the ledger unchanged
    (existing rows win; `graded_at` of the first grading is kept)."""
    ledger_path = Path(ledger_path)
    if ledger_path.exists():
        old = pd.read_parquet(ledger_path)
        fresh = new.merge(old[KEYS], on=KEYS, how="left", indicator=True)
        fresh = fresh[fresh["_merge"] == "left_only"].drop(columns="_merge")
        out = pd.concat([old, fresh], ignore_index=True)
        n_new = len(fresh)
    else:
        out, n_new = new.reset_index(drop=True), len(new)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger_path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(ledger_path)
    return out, n_new


def first_publication(pubs: pd.DataFrame, publish_id: str | None = None) -> pd.DataFrame:
    """The publication that was actually available per game: the EARLIEST `published_at` (or a named publish_id)."""
    if publish_id:
        return pubs[pubs["publish_id"] == publish_id].copy()
    p = pubs.sort_values(["game_id", "published_at"], kind="mergesort")
    return p.drop_duplicates("game_id", keep="first").reset_index(drop=True)


def underpowered(n: int, floor: int = 100) -> str:
    return f"UNDERPOWERED (n={n} < {floor})" if n < floor else f"n={n}"


def grade_tables(led: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Aggregate scorecard over ledger rows using the eval harness's own primitives (`cbb_sim.eval.market`)."""
    out: dict = {"n_games": int(len(led))}
    out["margin_mae"] = float(led["margin_err"].abs().mean()) if len(led) else np.nan
    out["margin_bias"] = float(led["margin_err"].mean()) if len(led) else np.nan
    out["total_mae"] = float(led["total_err"].abs().mean()) if len(led) else np.nan
    out["total_bias"] = float(led["total_err"].mean()) if len(led) else np.nan
    a = led.dropna(subset=["spread"])
    out["n_with_spread"] = int(len(a))
    if len(a):
        out["model_margin_mae_lined"] = float(a["margin_err"].abs().mean())
        out["line_margin_mae_lined"] = float((a["margin"] - a["market_margin"]).abs().mean())
        disagree = a["ats_disagree_pts"].reset_index(drop=True)
        cover = a["ats_cover_margin"].reset_index(drop=True)
        out["ats_table"] = M.bucket_table(disagree, cover)
        out["ats_bootstrap"] = M.bootstrap_roi(M.decided_pnl(disagree, cover, M.ATS_THRESHOLDS[0]), n_boot)
    b = led.dropna(subset=["total"])
    out["n_with_total"] = int(len(b))
    if len(b):
        out["total_mae_lined"] = float(b["total_err"].abs().mean())
        out["line_total_mae_lined"] = float((b["total_pts"] - b["total"]).abs().mean())
        disagree = b["ou_disagree_pts"].reset_index(drop=True)
        cover = b["ou_cover_margin"].reset_index(drop=True)
        out["ou_table"] = M.bucket_table(disagree, cover)
        out["ou_bootstrap"] = M.bootstrap_roi(M.decided_pnl(disagree, cover, M.ATS_THRESHOLDS[0]), n_boot)
    c = led.dropna(subset=["home_ml", "away_ml"]).reset_index(drop=True)
    out["n_with_ml"] = int(len(c))
    if len(c):
        tab, pnl = M.ml_edge_table(c["ml_edge"], c["is_home_win"], c["home_ml"], c["away_ml"])
        out["ml_edge_table"], out["ml_bootstrap"] = tab, M.bootstrap_roi(pnl, n_boot)
        out["model_brier"] = float(((c["p_home"] - c["is_home_win"]) ** 2).mean())
        out["market_brier"] = float(((c["market_p_home_devig"] - c["is_home_win"]) ** 2).mean())
        if len(c) >= 50:
            out["calibration"] = M.calibration_deciles(c["p_home"], c["market_p_home_devig"], c["is_home_win"])
    out["brier_all_games"] = float(((led["p_home"] - led["is_home_win"]) ** 2).mean()) if len(led) else np.nan
    return out


# ------------------------------------------------------------------------------------------------ monitor
def bias_stats(x: pd.Series) -> dict:
    x = pd.Series(x).dropna().astype(float)
    n = int(len(x))
    if n < 2:
        return {"n": n, "bias": float(x.mean()) if n else np.nan, "se": np.nan, "z": np.nan}
    se = float(x.std(ddof=1) / np.sqrt(n))
    return {"n": n, "bias": float(x.mean()), "se": se, "z": float(x.mean() / se) if se > 0 else np.nan}


def rolling_bias(led: pd.DataFrame, windows_days=(14, 30, None), asof=None) -> pd.DataFrame:
    """Margin and total bias (sim minus actual) with SEs over trailing windows of slate_date (None = all)."""
    d = led.copy()
    d["slate_date"] = pd.to_datetime(d["game_date"])
    end = pd.Timestamp(asof) if asof is not None else (d["slate_date"].max() if len(d) else pd.Timestamp("1970-01-01"))
    rows = []
    for w in windows_days:
        sub = d if w is None else d[d["slate_date"] > end - pd.Timedelta(days=w)]
        for name, col in (("margin", "margin_err"), ("total", "total_err")):
            rows.append({"window": "all" if w is None else f"{w}d", "quantity": name, **bias_stats(sub[col])})
    return pd.DataFrame(rows)


def stat_bias(led: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in STATS:
        diffs = []
        for side in ("home", "away"):
            s, a = f"sim_{side}_{st}_mean", f"act_{side}_{st}"
            if s in led.columns and a in led.columns:
                diffs.append(led[s] - led[a])
        if not diffs:
            rows.append({"stat": st, "n": 0, "bias": np.nan, "se": np.nan, "z": np.nan, "note": "no sim / truth column"})
            continue
        # per team-game bias; both sides pooled, SE clustered by game (side differences averaged within a game would hide offsetting
        # errors, so pooled rows are used and the SE is computed on per-game means of the two sides: conservative)
        per_game = pd.concat(diffs, axis=1).mean(axis=1)
        pooled = pd.concat(diffs, ignore_index=True)
        b = bias_stats(per_game)
        rows.append({"stat": st, "n": b["n"], "bias": float(pooled.dropna().mean()) if pooled.notna().any() else np.nan,
                     "se": b["se"], "z": (float(pooled.dropna().mean()) / b["se"]) if b["se"] and b["se"] > 0 else np.nan,
                     "note": "" if b["n"] else "no truth rows"})
    return pd.DataFrame(rows)


def responsiveness(led: pd.DataFrame) -> pd.DataFrame:
    """Predicted margin quintile vs mean actual margin: must slope (CLAUDE.md matchup-specific rule)."""
    if len(led) < 25:
        return pd.DataFrame()
    q = pd.qcut(led["sim_margin_mean"], 5, labels=False, duplicates="drop")
    g = led.groupby(q).agg(n=("margin", "size"), sim_margin=("sim_margin_mean", "mean"), actual_margin=("margin", "mean"))
    return g.reset_index().rename(columns={"sim_margin_mean": "quintile"})


def clv_frame(led: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Per published lean: closing-line value. Publish-time line = the stored `spread` / `total` / devig ML; close = `close` frame
    (same provider). Positive = the line moved toward the lean. Points for spread / total, de-vigged probability for the ML.
    Moneyline CLV is NaN where the publish ML was only a close proxy (replay)."""
    c = close.rename(columns={"spread": "c_spread", "total": "c_total", "home_ml": "c_home_ml", "away_ml": "c_away_ml"})
    c = c[["cbbd_game_id", "provider", "c_spread", "c_total", "c_home_ml", "c_away_ml"]].drop_duplicates(["cbbd_game_id", "provider"])
    d = led.merge(c, on=["cbbd_game_id", "provider"], how="left")
    side_a = d["ats_lean"].map({"home": 1.0, "away": -1.0})
    d["clv_ats_pts"] = side_a * ((-d["c_spread"]) - d["market_margin"])
    side_o = d["ou_lean"].map({"over": 1.0, "under": -1.0})
    d["clv_ou_pts"] = side_o * (d["c_total"] - d["total"])
    cp, _ = L.devig(d["c_home_ml"], d["c_away_ml"])
    side_m = d["ml_lean"].map({"home": 1.0, "away": -1.0})
    proxy = d["ml_is_close_proxy"].fillna(False).astype(bool)
    d["clv_ml_prob"] = np.where(proxy, np.nan, side_m * (cp - d["market_p_home_devig"]))
    return d


def clv_summary(d: pd.DataFrame, col: str) -> dict:
    x = d[col].dropna()
    moved = x[x != 0]
    n = int(len(x))
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {"leans_with_close": n, "line_moved": int(len(moved)), "mean_clv": float(x.mean()) if n else np.nan, "se": se,
            "z": float(x.mean() / se) if se and se > 0 else np.nan,
            "clv_agreement": float((moved > 0).mean()) if len(moved) else np.nan}


def surprise_corr(d: pd.DataFrame, disagree_col: str, cover_col: str) -> tuple[float, int]:
    x = d[[disagree_col, cover_col]].dropna()
    if len(x) < 30 or x[disagree_col].std() == 0 or x[cover_col].std() == 0:
        return float("nan"), int(len(x))
    return float(np.corrcoef(x[disagree_col], x[cover_col])[0, 1]), int(len(x))


def monitor(led: pd.DataFrame, close: pd.DataFrame | None, tol: dict, asof=None, min_n: int = 50, z_alarm: float = 3.0) -> dict:
    """Everything the monitor reports, plus a list of alarms. Reports and alarms only; nothing is corrected."""
    alarms: list[str] = []
    rb = rolling_bias(led, asof=asof)
    for _, r in rb.iterrows():
        if r["n"] >= min_n and np.isfinite(r["z"]) and abs(r["z"]) > z_alarm:
            alarms.append(f"BIAS {r['quantity']} {r['window']}: {r['bias']:+.2f} (SE {r['se']:.2f}, z {r['z']:+.1f}, n {int(r['n'])})")
    sb = stat_bias(led)
    for _, r in sb.iterrows():
        if r["n"] >= min_n and np.isfinite(r["z"]) and abs(r["z"]) > z_alarm:
            alarms.append(f"STAT BIAS {r['stat']}: {r['bias']:+.3f} per team-game (z {r['z']:+.1f}, n {int(r['n'])})")
    out = {"rolling_bias": rb, "stat_bias": sb, "responsiveness": responsiveness(led), "alarms": alarms, "rule": RULE_TEXT,
           "n_ledger": int(len(led))}
    if close is not None and len(close) and len(led):
        d = clv_frame(led, close)
        out["clv_frame"] = d
        out["clv"] = {"ats": clv_summary(d, "clv_ats_pts"), "ou": clv_summary(d, "clv_ou_pts"), "ml": clv_summary(d, "clv_ml_prob")}
        sc, n_sc = surprise_corr(d, "ats_disagree_pts", "ats_cover_margin")
        ag = out["clv"]["ats"]["clv_agreement"]
        out["surprise_corr_ats"] = sc
        out["surprise_n"] = n_sc
        if np.isfinite(sc) and np.isfinite(ag) and out["clv"]["ats"]["line_moved"] >= 100:
            out["leak_verdict_ats"] = M.leak_verdict(sc, ag, tol)
        else:
            out["leak_verdict_ats"] = "UNDERPOWERED (needs >= 30 lined games for the correlation and >= 100 moved lines for CLV agreement)"
        if out["leak_verdict_ats"] == "LEAK-SUSPECT":
            alarms.append("LEAK-SUSPECT: surprise correlation is real but CLV agreement is near coin-flip (do not trust the edge)")
        for k in ("ats", "ou"):
            s = out["clv"][k]
            if s["leans_with_close"] >= min_n and np.isfinite(s["z"]) and s["z"] < -z_alarm:
                alarms.append(f"CLV {k}: mean {s['mean_clv']:+.3f} pts (z {s['z']:+.1f}): leans move AGAINST the published number")
    else:
        out["clv"] = None
        out["leak_verdict_ats"] = "PENDING (no closing lines yet)"
    return out


def fmt_df(df: pd.DataFrame, nd: int = 3) -> str:
    if df is None or not len(df):
        return "(empty)\n"
    try:
        return df.round(nd).to_markdown(index=False) + "\n"
    except Exception:  # tabulate missing
        return "```\n" + df.round(nd).to_string(index=False) + "\n```\n"
