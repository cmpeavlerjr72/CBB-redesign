# laneD_2 DONE as Case A (laneD_1 void), F_R then F_T, both full gate reads (AWS operator, 2026-10-01)

- laneD_1 had NOT started when laneD_2 arrived: the clone the operator built for the queue (`~/cbb4`) was already at `7d38de5`, the laneD_2 commit, so the commands below ran with the `inputs_base` stage in the chain. This is laneD_2 Case A; no laneD_1-style v2-input read exists.
- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66, us-east-2a). Engine parity on `~/cbb4` vs `parity_reference_windows_v6.json`: PASS bit-identical (02:03Z).
- Preflight: `preflight: 25/25 inputs present` (F_T), `29/29` (F_R).
- Commands exactly as written (`scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_R --tag FR_box_v1 --cores 90 --parallel --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers 90 --gate-ref ... --gate-noise <S0f1..f4>`; then F_T with `--tag FT_box_v1 --reuse-from FR_box_v1`). A second queue worker (lane A / B / C sims at 90 workers) shared the box throughout.

| variant | start | end | wall | stage timings (from the chain log) |
|---|---|---|---|---|
| F_R | 02:03:26Z | 02:24:15Z | 20 min 49 s | possessions_v4 207 s, rb_design 3 s, fg_design 5 s, po_design 8 s, clock 30 s, foul_state 26 s, fg_train 37 s, rb_train 121 s, po_train 396 s, inputs_base 7 s, inputs 2 s, gate (5,710 x 200 read + grade + 4 pair tables) 542 s |
| F_T | 02:24:15Z | 02:39:28Z | 15 min 13 s | fg_train 50 s, rb_train 113 s, po_train 374 s, inputs 3 s, gate 514 s |

Memory peak seen at the operator's checks: 85 GB used of 369 GB.

Outputs:
- Box results: `results/engine_v0/fr1_FR_box_v1_full_s200_o0_ev4/`, `results/engine_v0/fr1_FT_box_v1_full_s200_o0_ev4/`. Synced locally (scp: `games.parquet`, `run_meta.json`) and pushed to HF (`results` key, players.parquet included) right after each variant.
- `data/processed/models/full_retrain_v1/{FR_box_v1,FT_box_v1}/` (FR 1.1 GB): pushed to HF under the `model_artifacts` key after each variant. Locally synced (scp): `gate/` (`*__verified.md`, `pair_ref_vs_*_n{0..3}.md`, `stage.log`) and every `*.done.json`, `stage.log`, `*.md`, `*report*.json` under 5 MB. Pull the artifacts with `hf_sync_data.py pull --dirs model_artifacts --only 'full_retrain_v1/**'`.
- Extra (operator): Decision 12 floors against `v3full_S0_s200_o0` from `scripts/ops_pair_bootstrap_v1.py`. These are the four S0 seed-offset draws plus a 2,000-resample paired game bootstrap, with G5 by component and TOV on the count: `results/engine_v0/v3full_grade/fr1_FR_box_v1_full_s200_o0_ev4__boot.md` and `fr1_FT_box_v1_full_s200_o0_ev4__boot.md` (local).

Lines beyond the Decision 12 floor vs S0 (move, floor), reported, not interpreted:
- **F_R:**
  - G1 possessions mean -0.766 (0.010).
  - G5 total SD ratio -0.028 (0.0025). The within-game SD falls by 0.466 while the residual SD is inside the floor.
  - G5 margin SD ratio -0.002, inside the floor (0.005).
  - G5 home/away corr -0.028 (0.003).
  - G9 total bias -1.483 (0.040).
  - G9 slope +0.002, inside the floor (0.009).
  - G9 margin bias +0.071 (0.051).
  - TOV count per team-game -0.214 (0.009).
  - FTA/FGA +0.0117 (0.0004).
  - eFG% -0.0009 (0.0001).
- **F_T:**
  - G1 possessions mean -0.780 (0.010).
  - G5 total SD ratio -0.026 (0.003).
  - G5 margin SD ratio -0.0070 (0.0065).
  - G5 home/away corr -0.025 (0.004).
  - G9 total bias -1.469 (0.059).
  - G9 slope -0.0108 (0.0105).
  - TOV count -0.176 (0.018).
  - FTA/FGA +0.0109 (0.0006).
