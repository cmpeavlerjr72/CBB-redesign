"""diag_g1_possessions_v2.py -- G1 like-for-like: engine count vs a phantom-corrected pbp count.

Lane B follow-up, 2026-09-30.  DIAGNOSTIC ONLY.  Versioned sibling of
`diag_g1_possessions_v1.py` (which stays as written).  Nothing under
`data/processed` is modified: the corrected possession table is built in
memory and only summary numbers are written to
`results/g1g5_diag/g1_lfl_report.json`.

The corrected truth removes the pbp layer's SPURIOUS possessions identified by
`diag_g1_unknown_poss_v1.py` / `diag_g1_unknown_sample_v1.py` /
`diag_g1_unknown_box_v1.py`:
  A   rebound of a missed and-one FT with no open possession (1 s phantom for the shooter)
  B1  stray DREB row after a free-throw moment
  B2s stray or out-of-order DREB row elsewhere, <= 3 s (the > 3 s part of B2 is
      a real possession whose missed shot the feed lost; sensitivities: all
      B2 removed / all B2 kept)
  C1  possession opened by a stray OREB row and closed by the other team's event
  E   the shooter's own OREB of a missed and-one FT, which the layer opens as a NEW possession
A/B/C1 phantoms: dropped, their seconds given to the next possession of the
period (whose real start they delayed).  E: merged into the preceding and-one
possession (it is that possession's continuation).  C2 and D (real
possessions whose terminal event the feed lost) and `end_period` (real horn
possessions) are kept.  Tiling is preserved exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

OUT = ROOT / "results/g1g5_diag"
TAP = OUT / "tap_full"
TAP_SEEDS = (20, 22, 24)
PREV = {0: "period_start", 1: "DREB", 2: "TOV", 3: "made_FG", 4: "made_FT", 5: "other"}
TYPES = ["period_start", "DREB", "TOV", "made_FG", "made_FT", "other"]
#: Headline phantom set.  B2 is split by duration: <= 3 s is a stray or
#: out-of-order rebound row (phantom; 70% of B2 in 2024-25), > 3 s is a missed
#: shot the feed did not log (a real possession; the rows read in section 5.1).
PHANTOM = ["A_dreb_after_missed_andone_FT", "B1_dreb_no_open_after_FT", "B2s_stray_row_le3s",
           "C1_mismatch_open_by_OREB"]


def split_b2(U: pd.DataFrame) -> pd.DataFrame:
    U = U.copy()
    U.loc[(U["cls"] == "B2_dreb_no_open_other") & (U["dur"] <= 3), "cls"] = "B2s_stray_row_le3s"
    return U


def corrected_table(p: pd.DataFrame, U: pd.DataFrame, drop_classes: list[str]) -> tuple[pd.DataFrame, dict]:
    p = p.sort_values(["game_id", "period", "poss_index"]).reset_index(drop=True)
    key = ["game_id", "period", "start_clock", "end_clock"]
    ph = U[U["cls"].isin(drop_classes)][key + ["cls"]].drop_duplicates(key)
    m = p.merge(ph, on=key, how="left")
    is_ph = (m["terminal_event"] == "unknown") & m["cls"].notna()
    g = p.groupby(["game_id", "period"])
    prev_ao = g["and_one"].shift(1).fillna(False).astype(bool)
    prev_ftm = g["ftm"].shift(1)
    prev_off = g["offense_team_id"].shift(1)
    is_e = prev_ao & (prev_ftm == 0) & (p["offense_team_id"] == prev_off) & ~is_ph.to_numpy()
    p = p.copy()
    p["dur"] = p["duration_s"].astype(float)
    p["ph"] = is_ph.to_numpy()
    p["e"] = is_e.to_numpy()
    # A/B/C1: give seconds to the next possession of the period (else the previous)
    nxt_same = (p["game_id"].shift(-1) == p["game_id"]) & (p["period"].shift(-1) == p["period"])
    add_next = np.zeros(len(p))
    idx = np.flatnonzero(p["ph"].to_numpy())
    # a run of consecutive phantoms passes its seconds down the run
    for i in idx:
        if nxt_same.iloc[i]:
            add_next[i + 1] += p.at[i, "dur"] + add_next[i]
    p["dur"] = p["dur"] + add_next
    # E: merge into the previous possession
    for i in np.flatnonzero(p["e"].to_numpy()):
        j = i - 1
        while j >= 0 and (p.at[j, "ph"] or p.at[j, "e"]):
            j -= 1
        p.at[j, "dur"] += p.at[i, "dur"]
    keep = ~(p["ph"] | p["e"])
    info = {"n_rows": int(len(p)), "n_phantom_dropped": int(p["ph"].sum()), "n_E_merged": int(p["e"].sum()),
            "phantoms_with_no_next": int((p["ph"] & ~nxt_same).sum())}
    return p[keep].reset_index(drop=True), info


def main():
    U = pd.read_parquet(OUT / "unknown_poss_classified.parquet")
    U = split_b2(U[U["season"] == 2025])
    g200 = pd.read_parquet(ROOT / "results/engine_v0/F2_2025_s200_v5b_A_full/games.parquet")
    summary, raw = G.build_grading_frame(g200, 2025)
    graded = summary["game_id"].to_numpy()
    sim200 = raw.groupby("game_id")["possessions"].mean()
    est = summary.set_index("game_id")["game_poss"]
    p0 = pd.read_parquet(ROOT / "data/processed/possessions_v2/possessions_2025.parquet")
    u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
    tb = R.load_actual_team_box(2025)
    two = tb.groupby("game_id").size()
    nt = p0.groupby("game_id")["offense_team_id"].nunique()
    pc = np.array([gid for gid in graded if gid in nt.index and nt[gid] == 2 and bool(u.at[gid, "pbp_complete"])
                   and two.get(gid, 0) == 2])
    cc = np.array([gid for gid in pc if bool(u.at[gid, "clock_complete_reg"])])

    tg = pd.concat([pd.read_parquet(TAP / f"games_{s}.parquet") for s in TAP_SEEDS], ignore_index=True)
    tp = pd.concat([pd.read_parquet(TAP / f"poss_{s}.parquet") for s in TAP_SEEDS], ignore_index=True)
    gid_of = tg.drop_duplicates("gidx").set_index("gidx")["game_id"]
    tp["game_id"] = gid_of.loc[tp["gidx"].astype(int)].to_numpy()
    tp["type"] = tp["prev_end"].astype(int).map(PREV)
    nsim = len(TAP_SEEDS) * 2
    simtap = tg.groupby("game_id")["possessions"].mean()

    # sim-side and-one quantities (as in v1), for the label views
    g4t = pd.concat([pd.read_parquet(OUT.parent / f"g4_diag/tap/trips_{s}.parquet") for s in (0, 4, 7)])
    g4c = pd.concat([pd.read_parquet(OUT.parent / f"g4_diag/tap/chances_{s}.parquet") for s in (0, 4, 7)])
    g4r = pd.concat([pd.read_parquet(OUT.parent / f"g4_diag/tap/rebs_{s}.parquet") for s in (0, 4, 7)])
    a_s = float(g4t["cls"].isin([1.0, 2.0, 3.0]).sum() / (g4c["is_first"] == 1).sum())
    o_ft = float(g4r.loc[g4r["miss_type"] == 3, "p_oreb"].mean())
    q_ft = float((raw["home_ftm"] + raw["away_ftm"]).sum() / (raw["home_fta"] + raw["away_fta"]).sum())
    m1 = a_s * q_ft
    m2 = a_s * (1 - q_ft) * (1 - o_ft)

    rep = {"subsets": {"pc": int(len(pc)), "cc": int(len(cc))}}
    for label, classes in (("headline", PHANTOM),
                           ("sensitivity_B2_all_removed", PHANTOM + ["B2_dreb_no_open_other"]),
                           ("sensitivity_B2_all_kept", [c for c in PHANTOM if c != "B2s_stray_row_le3s"])):
        pcor, info = corrected_table(p0, U, classes)
        cnt0 = p0.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
        cntc = pcor.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
        cntc = cntc.reindex(cnt0.index, fill_value=0)
        # tiling preserved?
        pr = pcor[pcor["game_id"].isin(cc) & (pcor["period"] <= 2)]
        p_cc0 = p0[p0["game_id"].isin(cc) & (p0["period"] <= 2)]
        tiling = {"sum_dur_before": float(p_cc0["duration_s"].sum()), "sum_dur_after": float(pr["dur"].sum())}
        # --- chain, like-for-like (engine count vs corrected pbp count)
        gap_lfl_all = None
        sim_pc = float(sim200.loc[pc].mean())
        cnt_pc = float(cntc.loc[pc].mean())
        gap = sim_pc - cnt_pc
        seed_term = sim_pc - float(simtap.loc[pc].mean())
        tpp = tp[tp["game_id"].isin(pc)]
        sim_ot = float((tpp["period"] >= 3).sum() / 2.0 / (len(pc) * nsim))
        pp = pcor[pcor["game_id"].isin(pc)]
        act_ot = float((pp["period"] >= 3).sum() / 2.0 / len(pc))
        sim_reg = float((tpp["period"] <= 2).sum() / 2.0 / (len(pc) * nsim))
        act_reg = float((pp["period"] <= 2).sum() / 2.0 / len(pc))
        ts = tp[tp["game_id"].isin(cc) & (tp["period"] <= 2)]
        d_s, d_a = float(ts["used"].mean()), float(pr["dur"].mean())
        N_s = len(ts) / 2.0 / (len(cc) * nsim)
        N_a = len(pr) / 2.0 / len(cc)
        S_a = float(pr["dur"].sum() / 2.0 / len(cc))
        K = -S_a / (d_s * d_a)
        slack = (1200.0 - S_a) / d_s
        sh_s = ts["type"].value_counts(normalize=True).reindex(TYPES, fill_value=0)
        sh_a = pr["start_reason"].value_counts(normalize=True).reindex(TYPES, fill_value=0)
        du_s = ts.groupby("type")["used"].mean().reindex(TYPES)
        du_a = pr.groupby("start_reason")["dur"].mean().reindex(TYPES)

        def split(shs, dus):
            comp = ((shs - sh_a) * du_a).fillna(0) * K
            law = (sh_a * (dus - du_a)).fillna(0) * K
            inter = ((shs - sh_a) * (dus - du_a)).fillna(0) * K
            return comp, law, inter

        comp, law, inter = split(sh_s, du_s)
        # training-label view: in the corrected table the missed-branch successor
        # is already DREB (as in the engine); only the made branch differs
        sh_r, du_r = sh_s.copy(), du_s.copy()
        du_r["made_FG"] = (sh_s["made_FG"] * du_s["made_FG"] + m1 * du_s["made_FT"]) / (sh_s["made_FG"] + m1)
        sh_r["made_FG"] += m1
        sh_r["made_FT"] -= m1
        comp_r, law_r, inter_r = split(sh_r, du_r)
        subset_cc = (sim_reg - act_reg) - (N_s - N_a)
        chain = {"seed sample (200-seed run - 6-seed tap)": seed_term,
                 "overtime possessions": sim_ot - act_ot,
                 "subset pc -> cc (regulation)": subset_cc,
                 "tiling slack": slack,
                 "composition (engine labels)": float(comp.sum()),
                 "law (engine labels)": float(law.sum()),
                 "interaction (engine labels)": float(inter.sum())}
        closure = gap - sum(chain.values())
        by_type = pd.DataFrame({"sim_share": sh_s, "act_share": sh_a, "sim_dur": du_s, "act_dur": du_a,
                                "comp": comp, "law": law, "inter": inter})
        rep[label] = {
            "correction": info, "tiling_cc": tiling,
            "cnt_pc_before": float(cnt0.loc[pc].mean()), "cnt_pc_corrected": cnt_pc,
            "est_pc": float(est.loc[pc].mean()), "sim_pc200": sim_pc,
            "like_for_like_gap": gap, "chain": chain, "closure": closure,
            "training_label_view": {"composition": float(comp_r.sum()), "law": float(law_r.sum()),
                                    "interaction": float(inter_r.sum()),
                                    "law_by_type": (law_r).to_dict(), "comp_by_type": comp_r.to_dict()},
            "by_type": by_type.reset_index(names="type").to_dict("records"),
            "ot": {"sim": sim_ot, "act": act_ot},
            "full_graded_bridge": {"gate_gap": float(raw["possessions"].mean() - est.mean()),
                                   "grading_definition_term_now": float(cnt0.loc[pc].mean() - est.loc[pc].mean()),
                                   "phantoms_removed_per_team_game": float(cnt0.loc[pc].mean() - cnt_pc)},
        }
    rep["and_one_engine"] = {"a_s": a_s, "q_ft": q_ft, "o_ft": o_ft, "m1_made_branch": m1, "m2_missed_branch_dreb": m2}
    (OUT / "g1_lfl_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for label in ("headline", "sensitivity_B2_all_removed", "sensitivity_B2_all_kept"):
        r = rep[label]
        print(label, json.dumps({k: r[k] for k in ("correction", "tiling_cc", "cnt_pc_before", "cnt_pc_corrected",
                                                   "est_pc", "sim_pc200", "like_for_like_gap", "chain", "closure",
                                                   "full_graded_bridge")}, indent=1, default=float))
        tv = r["training_label_view"]
        print("training-label view", {k: tv[k] for k in ("composition", "law", "interaction")})
        print(pd.DataFrame(r["by_type"]).round(4).to_string())


if __name__ == "__main__":
    main()
