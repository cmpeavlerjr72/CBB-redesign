# laneG_1 DONE (G4 at 200 seeds, graded) (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb5` at `759be01` (contains `4748c56` and `c5667bc`).
- Parity: engine flag-off vs `parity_reference_windows_v6.json` PASS bit-identical (03:08Z). Not checked against v7.
- Setup as written:
  - `hf_sync_data.py pull --dirs model_artifacts --only 'fg_make/round4_site/**'`, rc=0.
  - `scripts/box_fullread_laneG_v1.sh preflight` printed `preflight OK`.
  - `v3full_S0_s200_o0` and `v3full_S0f{1..4}_*` were copied into the clone from the operator's S0 reads.
- `scripts/box_fullread_laneG_v1.sh run 90 0`: 04:21Z -> 04:34:43Z (concat: 1,142,000 game rows, 200 seeds, partial=False). `run_meta` adapters show `ENGINE_FG_MAKE=round4site_G4`. Then `scripts/box_fullread_laneG_v1.sh grade`; the job ended 04:36:30Z. It ran after every ship-decision request that was queued at the time, per your LOW-priority label.
- Synced (scp), to the same paths:
  - `results/engine_v0/v3full_G4_s200_o0/{games.parquet,run_meta.json}`;
  - `results/engine_v0/laneG_grade/{v3full_G4_s200_o0__verified.md, pair_vetoes.{md,json}, pair_site.{md,json}}`.
