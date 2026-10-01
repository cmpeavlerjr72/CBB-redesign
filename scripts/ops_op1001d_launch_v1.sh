#!/usr/bin/env bash
# local (operator 2026-10-01 day): launch a box and bring it to PARITY_PASS. ops_op1001d_launch_v1.sh <spot|ondemand> <SHA>
# State files in $OPDIR (default ~/op1001d): iid.txt ip.txt launch.txt. Cheapest AZ first.
set -uo pipefail; export MSYS_NO_PATHCONV=1
MODE=${1:?spot|ondemand}; SHA=${2:?sha}; S="$(cd "$(dirname "$0")" && pwd)"; R=us-east-2
OPDIR=${OPDIR:-$HOME/op1001d}; mkdir -p "$OPDIR"
L=$(AZS="us-east-2a us-east-2b us-east-2c" bash "$S/ops_launch_spot_v1.sh" $MODE | tee /dev/stderr | grep '^LAUNCHED') || { echo "NO LAUNCH"; exit 1; }
IID=$(echo $L | awk '{print $2}'); echo $IID > "$OPDIR/iid.txt"; echo "$L $MODE" >> "$OPDIR/launch.txt"
[ "$MODE" = ondemand ] && aws ec2 modify-instance-attribute --region $R --instance-id $IID --instance-initiated-shutdown-behavior Value=terminate
aws ec2 wait instance-running --region $R --instance-ids $IID
aws ec2 describe-instances --region $R --instance-ids $IID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > "$OPDIR/ip.txt"
IP=$(cat "$OPDIR/ip.txt"); echo "IP $IP"
K="-i $HOME/.ssh/cfb-sweep-ohio.pem -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
for i in $(seq 1 30); do ssh $K -o ConnectTimeout=8 ec2-user@$IP true 2>/dev/null && break; sleep 8; done
scp -q $K "$S/box_bootstrap_v1.sh" "$S/box_op1001d_setup_v1.sh" ec2-user@$IP:~/
grep "^HF_TOKEN=" "$S/../.env" | sed 's/^HF_TOKEN=/export HF_TOKEN=/' | ssh $K ec2-user@$IP 'umask 077; cat > ~/.hf_env'
ssh $K ec2-user@$IP "sudo shutdown -h 20:00 >/dev/null 2>&1; bash ~/box_bootstrap_v1.sh install >/dev/null && sg docker -c 'bash ~/box_bootstrap_v1.sh clone $SHA' >/dev/null && echo cloned"
ssh $K ec2-user@$IP "nohup sg docker -c 'bash ~/box_op1001d_setup_v1.sh $SHA' > ~/setup.out 2>&1 < /dev/null & echo setup started"
date -u +%H:%M:%SZ
