# laneG_1: fg_make G4 (FE-identified site offset) vs S0, full-size paired loop. LOW PRIORITY (diagnostic)

- **Priority: LOW.** This is a diagnostic. Run it behind every ship-decision request in the queue, and skip it if box time runs out.
- **Commit:** the commit that adds this file (pushed to `main`; `git log -1 -- docs/ops/box_queue/laneG_1.md`). The engine change is `c5667bc`: one default-off registry line, with the default path bit-identical to `parity_reference_windows_v7.json` (sha `34cd58dd...`, Windows, 60 games x 5 seeds, 2 workers).
- **Lane:** G. Doc: `docs/tests/home_site_terms_2026-09-30.md`, section 9.
- **Not an adoption read.** The PM has ruled G4 off tonight's adoption list. Its offline win is marginal: 1.05 floors on fold 2, inside the floor on fold 1, and a fold-1 jumper guard miss.
- **The question.** Does removing fg_make's site excess, which is about +0.4 to +0.7 pts/game, EXPOSE the under-predicted home strength? The expectation, stated in advance:
  - the real HCA (FE) falls toward 3.06;
  - the G6 home/away margin falls about 0.4 below actual;
  - `team_part_ha_bias` stays near -0.35;
  - home-minus-away eFG% goes negative against actual.

## 0. Setup
```
git fetch && git checkout <commit below>
chmod +x scripts/*.sh
python scripts/hf_sync_data.py pull --dirs model_artifacts --only 'fg_make/round4_site/**'   # 21 files, 25 MB (verified on HF tonight)
scripts/box_fullread_laneG_v1.sh preflight      # must print "preflight OK"
```
**Inputs already on the box from tonight's session:**
- engine inputs `engine_v3_S0` with its `docker_mounts.txt`;
- the S0 reads `v3full_S0_s200_o0` and `v3full_S0f{1..4}_s200_o{1000..4000}`.

## 1. Run: about one 200-seed full read (11-16 min at 90 workers)
```
scripts/box_fullread_laneG_v1.sh run 90 0
```
- **Environment.** The script sets `ENGINE_FG_MAKE=round4site_G4`, the served pins and `CBB_TRUTH=verified_v1`. No other `ENGINE_*` variable is set.
- **Container check.** Before the first chunk, the script asserts that the G4 directory is visible inside the container with the overlay mounts.
- **Flag check.** After the concat, it asserts that `run_meta.json` records the flag.
- **Floors.** The arm itself runs at offset 0 only. Its floors are the four existing S0 seed-offset draws plus the paired game bootstrap (Decision 12).

## 2. Grade
```
scripts/box_fullread_laneG_v1.sh grade
```
It writes the following to `results/engine_v0/laneG_grade/`:
- `v3full_G4_s200_o0__verified.md`: the full gate report.
- `pair_vetoes.{md,json}` (`ops_pair_bootstrap_v1.py`): all standard lines, read as VETOES (G1, G5, G9 slope and bias, G4 pooled, OT).
- `pair_site.{md,json}` (`grade_laneG_site_loop_v1.py`): the requested lines.
  - G6 by site: home/away margin, neutral margin, and the REAL home advantage (`hca_fe`, a team-FE margin model, actual 3.06), kept separate from the listed-home neutral line (`neu_fe`, `g6_margin_neu`).
  - `team_part_ha_bias`.
  - G9 slope and margin bias, overall and by site.
  - G4 eFG% by site (home, away, neutral) and home-minus-away against actual.

## Outputs to sync back
- `results/engine_v0/v3full_G4_s200_o0/` (`results` key).
- `results/engine_v0/laneG_grade/` (`results` key).

## The one line to decide
None. Run only if box time remains after the ship-decision requests.
