#!/usr/bin/env python3
"""
Core season scorecard for the base cbb-sims-2026 model, plus honest side-by-side
comparisons against boosted_sims / improved / pace-experiment variants on the
subset of games they cover.

Run: python results_analysis.py
Writes results_out.json (all numbers) into this scratchpad dir; also prints
tables to stdout for pasting into the report.
"""
import json
import numpy as np
import pandas as pd

SCRATCH = r"C:\Users\devuser\AppData\Local\Temp\claude\C--Users-devuser-CBB-clean-sheet\f90c786b-0ad5-48b7-b88a-f8ef6f7594ed\scratchpad"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

RESULTS = {}


def american_to_prob(price):
    price = np.asarray(price, dtype=float)
    out = np.where(price < 0, (-price) / ((-price) + 100.0), 100.0 / (price + 100.0))
    return out


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def bias(a, b):
    return float(np.mean(a - b))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


# ------------------------------------------------------------------
# Load base
# ------------------------------------------------------------------
base = pd.read_parquet(f"{SCRATCH}/master_games.parquet")
d = base[base["has_final"]].copy()
d["date"] = pd.to_datetime(d["date"])
d["month"] = d["date"].dt.to_period("M").astype(str)

d["actual_margin"] = d["finalA"] - d["finalB"]
d["actual_total"] = d["finalA"] + d["finalB"]
d["model_margin"] = d["margin_p50"]
d["model_total"] = d["total_p50"]
d["y_A"] = (d["finalA"] > d["finalB"]).astype(int)

# Market fair margin for team A: A covers iff actual_margin + close_spreadA > 0
# => market breakeven margin for A = -close_spreadA
d["market_margin_close"] = -d["close_spreadA"]
d["market_margin_open"] = -d["open_spreadA"]
d["market_total_close"] = d["close_total"]
d["market_total_open"] = d["open_total"]

RESULTS["overview"] = {
    "n_games_all_rows": int(len(base)),
    "n_games_graded_with_final": int(len(d)),
    "date_min": str(d["date"].min().date()),
    "date_max": str(d["date"].max().date()),
    "n_unique_dates": int(d["date"].nunique()),
}

# ------------------------------------------------------------------
# SPREAD
# ------------------------------------------------------------------
spread_res = {}
spread_res["mae_model_vs_actual"] = mae(d["model_margin"], d["actual_margin"])
spread_res["mae_market_close_vs_actual"] = mae(d["market_margin_close"], d["actual_margin"])
spread_res["bias_model_minus_actual"] = bias(d["model_margin"], d["actual_margin"])
spread_res["bias_market_close_minus_actual"] = bias(d["market_margin_close"], d["actual_margin"])
spread_res["n"] = int(len(d))

edge = d["model_margin"] - d["market_margin_close"]
pick_A = edge > 0
push = np.isclose(d["actual_margin"].values, d["market_margin_close"].values, atol=1e-9)
a_covers = d["actual_margin"] > d["market_margin_close"]

model_win = np.where(push, "P", np.where(pick_A == a_covers, "W", "L"))
d["ats_edge"] = edge
d["ats_result"] = model_win

buckets = [1, 2, 3, 5]
ats_by_bucket = {}
for th in buckets:
    sub = d[np.abs(d["ats_edge"]) >= th]
    w = int((sub["ats_result"] == "W").sum())
    l = int((sub["ats_result"] == "L").sum())
    p = int((sub["ats_result"] == "P").sum())
    n = w + l + p
    ats_by_bucket[f">={th}pt"] = {
        "n": n, "W": w, "L": l, "P": p,
        "win_pct_excl_push": round(w / (w + l), 4) if (w + l) else None,
    }
spread_res["ats_by_disagreement_bucket"] = ats_by_bucket
# also the <1pt (no real disagreement) bucket for context
sub0 = d[np.abs(d["ats_edge"]) < 1]
w = int((sub0["ats_result"] == "W").sum()); l = int((sub0["ats_result"] == "L").sum()); p = int((sub0["ats_result"] == "P").sum())
spread_res["ats_lt_1pt_bucket"] = {"n": w + l + p, "W": w, "L": l, "P": p, "win_pct_excl_push": round(w/(w+l),4) if (w+l) else None}

RESULTS["spread"] = spread_res

# ------------------------------------------------------------------
# TOTALS
# ------------------------------------------------------------------
tot_res = {}
tot_res["mae_model_vs_actual"] = mae(d["model_total"], d["actual_total"])
tot_res["mae_market_close_vs_actual"] = mae(d["market_total_close"], d["actual_total"])
tot_res["bias_model_minus_actual"] = bias(d["model_total"], d["actual_total"])
tot_res["bias_market_close_minus_actual"] = bias(d["market_total_close"], d["actual_total"])
tot_res["n"] = int(len(d))

edge_t = d["model_total"] - d["market_total_close"]
pick_over = edge_t > 0
push_t = np.isclose(d["actual_total"].values, d["market_total_close"].values, atol=1e-9)
over_hits = d["actual_total"] > d["market_total_close"]
tot_result = np.where(push_t, "P", np.where(pick_over == over_hits, "W", "L"))
d["tot_edge"] = edge_t
d["tot_result"] = tot_result

ou_by_bucket = {}
for th in buckets:
    sub = d[np.abs(d["tot_edge"]) >= th]
    w = int((sub["tot_result"] == "W").sum())
    l = int((sub["tot_result"] == "L").sum())
    p = int((sub["tot_result"] == "P").sum())
    ou_by_bucket[f">={th}pt"] = {
        "n": w + l + p, "W": w, "L": l, "P": p,
        "win_pct_excl_push": round(w / (w + l), 4) if (w + l) else None,
    }
tot_res["ou_by_disagreement_bucket"] = ou_by_bucket

# bias by predicted-total tercile
d["total_tercile"] = pd.qcut(d["model_total"], 3, labels=["low", "mid", "high"])
tercile_bias = {}
for name, sub in d.groupby("total_tercile", observed=True):
    tercile_bias[str(name)] = {
        "n": int(len(sub)),
        "range": [float(sub["model_total"].min()), float(sub["model_total"].max())],
        "bias_model_minus_actual": bias(sub["model_total"], sub["actual_total"]),
        "mae_model": mae(sub["model_total"], sub["actual_total"]),
    }
tot_res["bias_by_tercile"] = tercile_bias

# bias by month
month_bias = {}
for name, sub in d.groupby("month"):
    month_bias[str(name)] = {
        "n": int(len(sub)),
        "bias_model_minus_actual": bias(sub["model_total"], sub["actual_total"]),
        "mae_model": mae(sub["model_total"], sub["actual_total"]),
        "bias_market_minus_actual": bias(sub["market_total_close"], sub["actual_total"]),
    }
tot_res["bias_by_month"] = month_bias

RESULTS["totals"] = tot_res

# ------------------------------------------------------------------
# WIN PROBABILITY CALIBRATION
# ------------------------------------------------------------------
wp = {}
wp["overall_brier_model"] = brier(d["A_win_prob"].values, d["y_A"].values)

# de-vigged moneyline market prob (close)
mkt = d.dropna(subset=["close_mlA", "close_mlB"]).copy()
imp_a = american_to_prob(mkt["close_mlA"].values)
imp_b = american_to_prob(mkt["close_mlB"].values)
p_a_devig = imp_a / (imp_a + imp_b)
mkt["p_a_devig"] = p_a_devig
wp["overall_brier_market_devig"] = brier(p_a_devig, mkt["y_A"].values)
wp["n_market_brier"] = int(len(mkt))
wp["overall_brier_model_on_same_subset"] = brier(mkt["A_win_prob"].values, mkt["y_A"].values)

# deciles (fixed width bins 0-1)
bins = np.linspace(0, 1, 11)
d["wp_decile"] = pd.cut(d["A_win_prob"], bins=bins, include_lowest=True)
decile_tab = []
for interval, sub in d.groupby("wp_decile", observed=True):
    if len(sub) == 0:
        continue
    decile_tab.append({
        "bucket": str(interval),
        "n": int(len(sub)),
        "avg_predicted": round(float(sub["A_win_prob"].mean()), 4),
        "actual_win_rate": round(float(sub["y_A"].mean()), 4),
    })
wp["deciles"] = decile_tab

RESULTS["win_prob_calibration"] = wp

# ------------------------------------------------------------------
# CLV: does model disagreement w/ OPEN predict the CLOSE move?
# ------------------------------------------------------------------
clv = {}
sub = d.dropna(subset=["open_spreadA", "close_spreadA"]).copy()
sub["model_edge_over_open"] = sub["model_margin"] - sub["market_margin_open"]
sub["line_move"] = sub["market_margin_close"] - sub["market_margin_open"]
clv["spread_n"] = int(len(sub))
clv["spread_corr"] = float(sub["model_edge_over_open"].corr(sub["line_move"]))
nz = sub[(sub["model_edge_over_open"] != 0) & (sub["line_move"] != 0)]
sign_agree = (np.sign(nz["model_edge_over_open"]) == np.sign(nz["line_move"])).mean()
clv["spread_sign_agreement_pct"] = round(float(sign_agree), 4)
clv["spread_sign_agreement_n"] = int(len(nz))

subt = d.dropna(subset=["open_total", "close_total"]).copy()
subt["model_edge_over_open_total"] = subt["model_total"] - subt["market_total_open"]
subt["total_move"] = subt["market_total_close"] - subt["market_total_open"]
clv["total_n"] = int(len(subt))
clv["total_corr"] = float(subt["model_edge_over_open_total"].corr(subt["total_move"]))
nzt = subt[(subt["model_edge_over_open_total"] != 0) & (subt["total_move"] != 0)]
sign_agree_t = (np.sign(nzt["model_edge_over_open_total"]) == np.sign(nzt["total_move"])).mean()
clv["total_sign_agreement_pct"] = round(float(sign_agree_t), 4)
clv["total_sign_agreement_n"] = int(len(nzt))

RESULTS["clv"] = clv

# ------------------------------------------------------------------
# Contamination check (created_at vs start_utc)
# ------------------------------------------------------------------
ca = pd.to_datetime(d["created_at"], utc=True, format="mixed", errors="coerce")
su = pd.to_datetime(d["start_utc"], utc=True, format="mixed", errors="coerce")
lead_hours = (su - ca).dt.total_seconds() / 3600
d["lead_hours"] = lead_hours
cont = {}
cont["n_with_both_ts"] = int(lead_hours.notna().sum())
cont["n_backfilled_gt3h_late"] = int((lead_hours < -3).sum())
cont["pct_backfilled_gt3h_late"] = round(100 * (lead_hours < -3).sum() / lead_hours.notna().sum(), 3)
cont["n_minor_late_0to3h"] = int(((lead_hours >= -3) & (lead_hours < 0)).sum())
cont["lead_hours_describe"] = {k: float(v) for k, v in lead_hours.describe().to_dict().items()}
backfill_dates = d.loc[lead_hours < -3, ["date"]].assign(date=lambda x: x["date"].astype(str))
cont["backfill_dates_counts"] = backfill_dates["date"].value_counts().sort_index().to_dict()
RESULTS["contamination_timing"] = cont

with open(f"{SCRATCH}/results_out.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print(json.dumps(RESULTS, indent=2, default=str))
