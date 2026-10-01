#!/usr/bin/env bash
# ssh helper: bx.sh <cmd...>   (IP in scratch/ip.txt)
D="$(dirname "$0")"
IP=$(cat "$D/ip.txt")
exec ssh -i /c/Users/devuser/.ssh/cfb-sweep-ohio.pem -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15 -o LogLevel=ERROR ec2-user@"$IP" "$@"
