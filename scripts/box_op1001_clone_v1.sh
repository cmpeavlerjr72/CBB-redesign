#!/usr/bin/env bash
# operator: extra clone ~/<dir> at <sha>, data copied from ~/cbb (reflink), parity vs v6, S0 tag build. op_clone.sh <dir> <sha>
set -uo pipefail
D=~/$1; SHA=$2; ts() { date -u +%H:%M:%SZ; }
[ -f $D/logs/PARITY_PASS ] && [ "$(cd $D && git rev-parse --short=7 HEAD)" = "${SHA:0:7}" ] && { echo "[$1] already ready at $SHA"; exit 0; }
[ -d $D/.git ] || git clone -q https://github.com/cmpeavlerjr72/CBB-redesign.git $D
cd $D && git fetch -q origin main && git checkout -q "$SHA" && chmod +x scripts/*.sh && echo "[$1] at $(git rev-parse --short HEAD) $(ts)"
sudo cp -an ~/cbb/data/. $D/data/; sudo chown -R ec2-user $D/data; mkdir -p $D/results/engine_v0/v3full_grade $D/results/engine_v0/v3box_grade $D/logs
docker run --rm -v $D:/app -v $D/out:/out cbb-sweep --tag box1001_parity_$1 --parity only --parity-ref docs/ops/parity_reference_windows_v6.json --workers 48 --push off > $D/logs/parity.log 2>&1
if grep -q "parity PASS" $D/logs/parity.log; then echo "[$1] parity PASS $(ts)"; else echo "[$1] parity FAIL $(ts)"; exit 9; fi
rm -rf $D/data/processed/models/engine_v3_S0
(cd $D && scripts/box_run_v2.sh scripts/build_engine_inputs_v3_tag_v1.py --tag S0 > logs/build_S0.log 2>&1); echo "[$1] S0 tag rc=$? $(ts)"
touch $D/logs/PARITY_PASS
