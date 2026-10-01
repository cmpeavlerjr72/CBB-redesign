#!/usr/bin/env bash
# operator: extra clone ~/<dir> at <sha>, data reflink-copied from ~/cbb, parity v9 on the clone's code. clone.sh <dir> <sha>
set -uo pipefail
D=~/$1; SHA=$2; ts() { date -u +%H:%M:%SZ; }
[ -f $D/logs/PARITY_PASS ] && [ "$(cd $D && git rev-parse --short=7 HEAD)" = "${SHA:0:7}" ] && { echo "[$1] already ready at $SHA"; exit 0; }
[ -d $D/.git ] || git clone -q https://github.com/cmpeavlerjr72/CBB-redesign.git $D
cd $D && git fetch -q origin main && git checkout -q "$SHA" && chmod +x scripts/*.sh && echo "[$1] at $(git rev-parse --short HEAD) $(ts)"
sudo cp -an ~/cbb/data/. $D/data/; sudo chown -R ec2-user $D/data; mkdir -p $D/results/engine_v0/v3full_grade $D/logs $D/out
docker run --rm -v $D:/app -v $D/out:/out cbb-sweep --tag box1001d_parity_$1 --parity only --parity-ref docs/ops/parity_reference_windows_v9.json --parity-input-dir data/processed/models/engine_v3 --workers 48 --push off > $D/logs/parity.log 2>&1
if grep -q "parity PASS" $D/logs/parity.log; then echo "[$1] parity PASS $(ts)"; touch $D/logs/PARITY_PASS; else echo "[$1] parity FAIL $(ts)"; exit 9; fi
