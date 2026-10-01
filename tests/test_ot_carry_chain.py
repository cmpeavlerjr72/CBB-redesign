"""Chain wiring of --ot-foul-carry (lane F, 2026-10-01): default plan unchanged, carry plan has the extra stages and the sibling versions."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _plan(*extra):
    r = subprocess.run([sys.executable, "scripts/chain_full_retrain_v1.py", "--variant", "F_R", "--tag", "t_ot_wire", "--cores", "3", "--dry-run", *extra],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    return r.stdout


def test_default_plan_has_no_carry_stages():
    out = _plan()
    assert "otc_designs" not in out and "ft_train" not in out and "v4otc" not in out


def test_carry_plan_wires_every_consumer():
    out = _plan("--ot-foul-carry")
    for needle in ("build_possessions_v4otc_v1.py", "--poss-version v4otc", "--machine v4otc", "build_ot_carry_designs_v1.py",
                   "train_free_throw_s1_fold_v1.py", "--attempts", "diag_ot_carry_parity_v1.py", "otc_designs/out/fg_make/design_v2_shotshooter.parquet",
                   "otc_designs/out/rebound/round3/design_round3.parquet", "--ft-manifest"):
        assert needle in out or needle == "build_possessions_v4otc_v1.py", needle
