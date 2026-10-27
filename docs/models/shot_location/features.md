# Shot-location stage: feature inventory

Status: PRE-REGISTERED 2026-10-27; to be completed by the round-1 worker with provenance for every column it builds.

| Feature | Dtype | Source | Computation | Fallback | Notes |
|---|---|---|---|---|---|
| `band` | category | CBBD `shot_location_x/y`, folded frame | `experiments.md` 1.2 thresholds; d in ft from basket centre (52.5, 250 tenths), theta per `build_pae_shot_rating_v1.dist_angle` | none: unlocated attempts are never imputed | target of Read B; feature of Read A arms A1 |
| `cell_p` | float | prior-season `da15` cell make rate | as fg_make `experiments.md` 31.2 | class mean | Read A arm A2 |
| `off_band_c` | float per band | own located attempts before tipoff | centred shrunk log-odds, k and rho chosen on fold-1 training (1.4) | 0 (league) | leak-tested before any arm |
| `def_band_c` | float per band | located attempts faced before tipoff | same recipe | 0 | leak-tested before any arm |
| site, `C_plus_state` columns | as served | possession_outcome `features.md` | unchanged | as served | |
