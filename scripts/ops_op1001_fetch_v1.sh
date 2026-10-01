#!/usr/bin/env bash
# local: scp result dirs (games.parquet, run_meta.json, plus listed extra files) from ~/cbb4 into the same local paths. fetch_q.sh <relpath_dir>...
S="$(dirname "$0")"; IP=$(cat $S/ip.txt); K="-i /c/Users/devuser/.ssh/cfb-sweep-ohio.pem -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
cd /c/Users/devuser/CBB-clean-sheet
for d in "$@"; do mkdir -p "$d"; for f in $(ssh $K ec2-user@$IP "cd ~/${SRC:-cbb4}/$d && ls -p | grep -v / | grep -v '^players.parquet$'"); do scp -q $K "ec2-user@$IP:${SRC:-cbb4}/$d/$f" "$d/"; done; echo "fetched $d: $(ls $d | tr '\n' ' ')"; done
