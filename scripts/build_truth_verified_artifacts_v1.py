#!/usr/bin/env python
"""Lane H follow-up: versioned verified-finals siblings (never overwrites).
  1. data/processed/truth/games_universe_verified_v1.parquet  (seasons 2022-2025; unverified finals dropped,
     third-source-resolved scores applied via reference.load_actual_games(verified_finals=True) logic)
  2. data/reference/verified_v1/gate_targets_{2022..2025}.parquet + gate_targets_fold2_train.parquet
     (scripts/build_gate_reference.py run against (1))
  3. data/processed/lines/lines_close_v2_verified.parquet (+ column check)
  4. data/processed/truth/stride500_verified_v1_F2_2025.parquet (+ swap report json)
Step selected by argv[1] in {universe, lines, stride}; gate targets are built by calling build_gate_reference.py."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import reference as R
TR = ROOT / "data/processed/truth"

def universe():
    u = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
    fin = pd.read_parquet(TR / "game_finals_v2.parquet", columns=["game_id", "home_score", "away_score"]).rename(columns={"home_score": "fh", "away_score": "fa"})
    out = []
    for s in (2022, 2023, 2024, 2025):
        x = u[u.season == s]
        x = x[~x.game_id.isin(R.unverified_final_game_ids(s))].merge(fin, on="game_id", how="left")
        h = x.fh.notna()
        x.loc[h, "home_score"] = x.loc[h, "fh"].astype("int32"); x.loc[h, "away_score"] = x.loc[h, "fa"].astype("int32")
        out.append(x.drop(columns=["fh", "fa"]))
    o = pd.concat(out, ignore_index=True)
    p = TR / "games_universe_verified_v1.parquet"
    assert not p.exists(); o.to_parquet(p, index=False)
    print("wrote", p, len(o), "of", int(u.season.isin([2022,2023,2024,2025]).sum()))

def lines():
    l = pd.read_parquet(ROOT / "data/processed/lines/lines_close_v1.parquet")
    a = pd.read_parquet(TR / "truth_audit_unplayed_v1.parquet")
    a = a[~a.sealed_season]
    bad = set(a[a.cls.str.contains("nonfinal")].game_id.astype("int64"))
    # also every id the loader deems unverified (covers ids not in hoopR schedule)
    for s in (2022, 2023, 2024, 2025):
        bad |= R.unverified_final_game_ids(s)
    new = l[~l.game_id.isin(bad)]
    dropped = l[l.game_id.isin(bad)]
    p = ROOT / "data/processed/lines/lines_close_v2_verified.parquet"
    assert not p.exists(); new.to_parquet(p, index=False)
    chk = l.loc[new.index]
    same = all(((chk[c] == new[c]) | (chk[c].isna() & new[c].isna())).all() for c in l.columns)
    rep = {"rows_old": len(l), "rows_new": len(new), "rows_dropped": len(dropped), "games_dropped": int(dropped.game_id.nunique()),
           "dropped_by_season": {int(k): int(v) for k, v in dropped.groupby("season").game_id.nunique().items()},
           "columns_identical": list(l.columns) == list(new.columns), "all_remaining_rows_identical_column_for_column": bool(same),
           "dropped_with_nonnull_close_spread": int(dropped.close_spread_home.notna().sum())}
    (ROOT / "data/processed/lines/_lines_close_v2_verified_report.json").write_text(json.dumps(rep, indent=1))
    print(rep)

def stride():
    old_g = R.load_actual_games(2025)  # default truth: the order the inputs/games were built in
    SUB, N = 11, 500
    def pick(g):
        ids = np.sort(g.game_id.to_numpy())  # rule: sort by game_id, every 11th, first 500
        return ids[::SUB][:N]
    old = pick(old_g); ver_g = R.load_actual_games(2025, verified_finals=True); new = pick(ver_g)
    so, sn = set(old.tolist()), set(new.tolist())
    pd.DataFrame({"game_id": new, "rank": np.arange(len(new))}).to_parquet(TR / "stride500_verified_v1_F2_2025.parquet", index=False)
    pd.DataFrame({"game_id": old, "rank": np.arange(len(old))}).to_parquet(TR / "stride500_current_rule_reproduced_F2_2025.parquet", index=False)
    unpl = sorted(set(old_g.game_id) - set(ver_g.game_id))
    rep = {"rule": "sort game_id ascending; every 11th; first 500", "n_old": len(old), "n_new": len(new),
           "common": len(so & sn), "only_in_current": sorted(int(x) for x in so - sn), "only_in_verified": sorted(int(x) for x in sn - so),
           "unplayed_ids_removed_from_universe": [int(x) for x in unpl], "unplayed_in_current_sample": sorted(int(x) for x in so & set(unpl)),
           "unplayed_in_verified_sample": sorted(int(x) for x in sn & set(unpl))}
    pos = {int(g): i for i, g in enumerate(np.sort(old_g.game_id.to_numpy()))}
    rep["removed_positions_in_sorted_universe"] = {int(g): pos[int(g)] for g in unpl}
    (TR / "stride500_verified_v1_report.json").write_text(json.dumps(rep, indent=1))
    print({k: (v if not isinstance(v, list) else (len(v), v[:12])) for k, v in rep.items()})

if __name__ == "__main__":
    {"universe": universe, "lines": lines, "stride": stride}[sys.argv[1]]()
