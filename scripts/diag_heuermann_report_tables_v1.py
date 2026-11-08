"""Renders markdown tables for docs/tests/heuermann_simv2_data_tests_2026-11-08.md from the T1-T3 CSVs (results/diag/heuermann_2026-11-08)."""
import sys
import pandas as pd, numpy as np

D = r"C:\Users\devuser\CBB-clean-sheet\results\diag\heuermann_2026-11-08"


def rd(n):
    return pd.read_csv(f"{D}/{n}")


def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pm(b, se, f=3):
    return "" if pd.isna(b) else f"{b:.{f}f} ({se:.{f}f})"


T = {}
# ---------------- T1
k = rd("data_t1_prior_strength.csv")
T["t1_k"] = md(k.assign(k_attempts=k.k_attempts.round(0))[["metric", "test_season", "prior_seasons", "k_attempts", "reliab_at_50_att", "reliab_at_200_att", "reliab_at_500_att"]])
rel = rd("data_t1_reliability.csv")
T["t1_rel"] = md(rel.pivot(index="metric", columns="season", values="split_half_r").reset_index().rename(columns=lambda c: str(c)), 2) + "\n\nSpearman-Brown full-season reliability:\n\n" + \
    md(rel.pivot(index="metric", columns="season", values="spearman_brown_full_season").reset_index().rename(columns=lambda c: str(c)), 2)
s = rd("data_t1_slopes.csv")
s["cell"] = [pm(b, e) for b, e in zip(s.slope, s.se)]
arms = ["raw_asof", "shrunk_asof", "raw_LOO_hindsight", "shrunk_LOO_hindsight"]
a = s[s.bucket_type == "all"].pivot_table(index=["metric", "season"], columns="arm", values="cell", aggfunc="first")[arms].reset_index()
n = s[s.bucket_type == "all"].pivot_table(index=["metric", "season"], values="n", aggfunc="first").reset_index()
a = a.merge(n, on=["metric", "season"])
a = a.sort_values(["metric", "season"], key=lambda c: c.map({"2024": 0, "2025": 1, "pooled": 2}) if c.name == "season" else c)
T["t1_slopes"] = md(a[["metric", "season", "n"] + arms])
for bt in ("games", "attempts"):
    g = s[(s.bucket_type == bt) & (s.season == "pooled")].pivot_table(index=["metric", "bucket"], columns="arm", values="cell", aggfunc="first")[arms[:2] + arms[2:]].reset_index()
    nn = s[(s.bucket_type == bt) & (s.season == "pooled")].pivot_table(index=["metric", "bucket"], values="n", aggfunc="first").reset_index()
    sdx = s[(s.bucket_type == bt) & (s.season == "pooled") & (s.arm == "shrunk_asof")][["metric", "bucket", "sd_x"]]
    g = g.merge(nn, on=["metric", "bucket"]).merge(sdx, on=["metric", "bucket"])
    g["_o"] = g.bucket.map(lambda x: {"1-5": 0, "6-15": 1, "16+": 2}.get(x, 0 if x.startswith("low") else 1 if x.startswith("mid") else 2))
    g = g.sort_values(["metric", "_o"])
    T["t1_" + bt] = md(g[["metric", "bucket", "n"] + arms + ["sd_x"]].rename(columns={"sd_x": "SD of shrunk rating"}))
q = rd("data_t1_quintiles.csv")
qp = q[q.season == "pooled"]
rows = []
for m, g in qp.groupby("metric"):
    for qb in ("raw_asof", "shrunk_asof"):
        gg = g[g.quintile_by == qb].sort_values("quintile")
        rows.append(dict(metric=m, quintiles_by=qb, n_per_Q=int(gg.n.iloc[0]), **{f"Q{int(r.quintile)} rating -> realised": f"{r.mean_rating:+.3f} -> {r.realised_centred:+.3f}" for r in gg.itertuples()}))
T["t1_quint"] = md(pd.DataFrame(rows))
qs = rd("data_t1_quintile_slopes.csv")
T["t1_qslope"] = md(qs[qs.season == "pooled"][["metric", "quintile_by", "quintile_mean_slope", "q5_minus_q1_realised", "q5_minus_q1_rating"]])

# ---------------- T2
sh = rd("data_t2_bonus_state_share.csv")
x = sh[sh.view == "offence_draw_rating"]
T["t2_share"] = md(x[["half", "quintile", "n_poss", "mean_rating", "share_none", "share_1and1", "share_double", "fta_per_poss"]].sort_values(["half", "quintile"]).replace({"half": {"both": "H1+H2"}}))
x = sh[(sh.view == "defence_foul_rating") & (sh.half == "both")]
T["t2_share_def"] = md(x[["quintile", "n_poss", "mean_rating", "share_none", "share_1and1", "share_double", "fta_per_poss"]])
fs = rd("data_t2_fta_per_poss_by_state.csv")
fs = fs[(fs.view == "offence_draw_rating")]
fp = fs.pivot_table(index=["half", "quintile"], columns="state", values="fta_per_poss").reset_index()
fn = fs.pivot_table(index=["half", "quintile"], columns="state", values="n_poss").reset_index()
fp = fp.merge(fn, on=["half", "quintile"], suffixes=("", "_n"))
fp = fp[["half", "quintile", "none", "1and1", "double", "none_n", "1and1_n", "double_n"]].rename(columns={"none": "FTA/poss none", "1and1": "FTA/poss 1-and-1", "double": "FTA/poss double",
                                                                                                         "none_n": "n none", "1and1_n": "n 1&1", "double_n": "n double"})
fp["half"] = fp.half.replace({"both": "H1+H2"})
T["t2_fta_state"] = md(fp.sort_values(["half", "quintile"]))
fg = rd("data_t2_fta_per_fga.csv")
T["t2_fga"] = md(fg.pivot(index="quintile", columns="view", values="fta_per_fga").reset_index())
tm = rd("data_t2_time_to_team_fouls.csv")
for v in ("offence_draw_rating", "defence_foul_rating"):
    y = tm[(tm.view == v) & tm.fouls_threshold.isin([7, 10])].copy()
    y["reach / mean min if reached / restricted mean (cap 20)"] = [f"{r.reach_share:.3f} / {r.mean_elapsed_min_if_reached:.2f} / {r.restricted_mean_min_cap20:.2f}" for r in y.itertuples()]
    p = y.pivot_table(index=["half", "quintile"], columns="fouls_threshold", values="reach / mean min if reached / restricted mean (cap 20)", aggfunc="first").reset_index()
    p.columns = ["half", "quintile", "7 fouls", "10 fouls"]
    nn = y.drop_duplicates(["half", "quintile"])[["half", "quintile", "n_def_halves"]]
    T["t2_time_" + v] = md(p.merge(nn, on=["half", "quintile"]))
sl = rd("data_t2_slopes.csv")
sl["slope (se)"] = [pm(b, e) for b, e in zip(sl.slope_per_unit, sl.se)]
def slope_tab(view, half):
    z = sl[(sl.view == view) & (sl.half == half) & sl.outcome.isin(["share_none", "share_1and1", "share_double", "fta_per_poss"])].copy()
    z["row"] = z.outcome + np.where(z.state == "all", "", " [" + z.state + "]")
    return z
rows = []
for view in ("offence_draw_rating", "defence_foul_rating"):
    for half in ("H1", "H2", "both"):
        z = slope_tab(view, half)
        base = z[(z.outcome == "fta_per_poss") & (z.state == "all")].slope_per_unit.iloc[0]
        for r in z.itertuples():
            rows.append(dict(view=view, half=half.replace("both", "H1+H2"), outcome=r.row, n=r.n, slope_per_unit=r._asdict()["_15"] if False else r.slope_per_unit, se=r.se,
                             per_SD=r.effect_per_sd, ratio_to_overall_FTA_per_poss_slope=r.slope_per_unit / base))
T["t2_slopes"] = md(pd.DataFrame(rows))
tt = sl[sl.outcome.str.startswith(("restricted", "reach")) & sl.outcome.str.contains("_7_|_10_")]
T["t2_time_slopes"] = md(tt[["view", "half", "outcome", "n", "slope_per_unit", "se", "effect_per_sd"]])
fgs = sl[sl.outcome == "fta_per_fga"]
T["t2_fga_slope"] = md(fgs[["view", "n", "slope_per_unit", "se", "sd_rating", "effect_per_sd"]])
T["t2_dec"] = md(rd("data_t2_timing_vs_state_decomposition.csv"))

# ---------------- T3
w = rd("data_t3_weekly_profile.csv")
w1 = w[(w.scope == "D1 team rows vs D1 opponents") & (w.season == "pooled")]
order = ["d0", "wk1 (d0-6)", "wk2 (d7-13)", "wk3", "wk4", "wk5", "wk6", "wk8", "wk1-2 (d0-13)", "Nov 1-14 (calendar)", "Nov 5-8 (calendar)"]
mets = ["fouls_per_poss", "fouls_per_team_game", "fta_per_poss", "poss_per_team_game", "tov_per_poss", "fg3a_share", "players_with_10plus_min", "top5_minutes_share", "top7_minutes_share", "bench_minutes_share", "new_to_team_minutes_share"]
wp = w1.pivot_table(index="cell", columns="metric", values="ratio_to_jan_feb")[mets].reindex(order)
ng = w1[w1.metric == "fouls_per_poss"].set_index("cell").n_games.reindex(order)
wp.insert(0, "n games (D1 vs D1)", ng.astype(int))
T["t3_week"] = md(wp.reset_index(), 3)
jf = w1[w1.cell == "Jan-Feb"].set_index("metric")
T["t3_jf"] = md(pd.DataFrame({"metric": mets, "Jan-Feb level (pooled)": [jf.loc[m, "level"] for m in mets], "n games Jan-Feb": [int(jf.loc[m, "n_games"]) for m in mets]}), 3)
# SE for the key rows
se = w1[w1.cell.isin(["wk1 (d0-6)", "wk1-2 (d0-13)"])].pivot_table(index="cell", columns="metric", values="se_ratio")[mets]
T["t3_se"] = md(se.reset_index(), 3)
ps = w[(w.scope == "D1 team rows vs D1 opponents") & (w.season != "pooled") & w.cell.isin(["d0", "wk1 (d0-6)", "wk2 (d7-13)", "wk3", "wk4", "wk5", "wk6", "wk1-2 (d0-13)"]) & w.metric.isin(["fouls_per_poss", "fouls_per_team_game", "poss_per_team_game", "tov_per_poss", "top7_minutes_share", "bench_minutes_share"])]
ps = ps.assign(v=[f"{r:.3f} ({e:.3f})" if e == e else f"{r:.3f}" for r, e in zip(ps.ratio_to_jan_feb, ps.se_ratio)])
pp = ps.pivot_table(index=["metric", "cell"], columns="season", values="v", aggfunc="first").reset_index()
pp["_c"] = pp.cell.map({c: i for i, c in enumerate(order)})
T["t3_season"] = md(pp.sort_values(["metric", "_c"]).drop(columns="_c"))
nn = w[(w.scope == "D1 team rows vs D1 opponents") & (w.season != "pooled") & (w.metric == "fouls_per_poss") & w.cell.isin(["d0", "wk1 (d0-6)", "wk2 (d7-13)", "wk1-2 (d0-13)", "Jan-Feb"])].pivot_table(index="cell", columns="season", values="n_games")
T["t3_nseason"] = md(nn.reset_index(), 0)
sp = rd("data_t3_wk12_splits.csv")
sp1 = sp[(sp.season == "pooled") & (sp.window == "wk1-2 (d0-13)")]
smets = ["fouls_per_poss", "fouls_per_team_game", "fta_per_poss", "poss_per_team_game", "tov_per_poss", "top7_minutes_share", "bench_minutes_share", "new_to_team_minutes_share"]
spv = sp1.pivot_table(index=["split", "cls"], columns="metric", values="ratio_to_jan_feb_overall")[smets]
spn = sp1[sp1.metric == "fouls_per_poss"].set_index(["split", "cls"]).n_games
spv.insert(0, "n team-games/rows' games", spn)
T["t3_split"] = md(spv.reset_index(), 3)
spl = sp1[sp1.metric.isin(["fouls_per_poss", "poss_per_team_game", "tov_per_poss", "top7_minutes_share"])].pivot_table(index=["split", "cls"], columns="metric", values="ratio_to_jan_feb_same_class")
spl["n Jan-Feb games (same class)"] = sp1[sp1.metric == "fouls_per_poss"].set_index(["split", "cls"]).n_games_jan_feb_same_class
lv = sp1[sp1.metric.isin(["fouls_per_poss", "poss_per_team_game"])].pivot_table(index=["split", "cls"], columns="metric", values=["level", "jan_feb_same_class_level"])
lv.columns = [f"{a_}: {b_}" for a_, b_ in lv.columns]
T["t3_split_same"] = md(spl.join(lv).reset_index(), 3)
sp0 = sp[(sp.season == "pooled") & (sp.window == "d0") & sp.metric.isin(["fouls_per_poss", "poss_per_team_game", "tov_per_poss"])].pivot_table(index=["split", "cls"], columns="metric", values="ratio_to_jan_feb_overall")
sp0["n games"] = sp[(sp.season == "pooled") & (sp.window == "d0") & (sp.metric == "poss_per_team_game")].set_index(["split", "cls"]).n_games
T["t3_split_d0"] = md(sp0.reset_index(), 3)
spseason = sp[(sp.window == "wk1-2 (d0-13)") & (sp.split == "gap_class") & (sp.season != "pooled") & (sp.metric == "fouls_per_poss")].pivot_table(index="cls", columns="season", values="ratio_to_jan_feb_overall")
T["t3_gap_season"] = md(spseason.reset_index(), 3)

if __name__ == "__main__":
    import json
    json.dump(T, open(r"C:\Users\devuser\AppData\Local\Temp\claude\e492afc7f524e11b\8ba661d1-2768-484f-a2f3-2ba26a3ef0f4\scratchpad\tables.json", "w"))
    for kk, v in T.items():
        print("\n###", kk, "\n" + v)

# extra: ratio to the whole-season level (his "season level"), per season
wa = w[(w.scope == "D1 team rows vs D1 opponents") & w.cell.isin(["Nov 1-14 (calendar)", "Nov 5-8 (calendar)", "wk1-2 (d0-13)"]) & w.metric.isin(["fouls_per_poss", "fouls_per_team_game", "poss_per_team_game"])]
wa = wa.assign(v=[f"{r:.3f} ({e:.3f})" if e == e else "" for r, e in zip(wa.ratio_to_season_all_games, wa.se_ratio_all)])
T["t3_vs_all"] = md(wa.pivot_table(index=["metric", "cell"], columns="season", values="v", aggfunc="first").reset_index())
if __name__ == "__main__":
    import json
    json.dump(T, open(r"C:\Users\devuser\AppData\Local\Temp\claude\e492afc7f524e11b\8ba661d1-2768-484f-a2f3-2ba26a3ef0f4\scratchpad\tables.json", "w"))
    print(T["t3_vs_all"])
