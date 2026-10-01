#!/usr/bin/env bash
# ssh helper: bx <cmd...>  (IP in $OPDIR/ip.txt)
OPDIR=${OPDIR:-$HOME/op1001d}; IP=$(cat "$OPDIR/ip.txt")
exec ssh -i $HOME/.ssh/cfb-sweep-ohio.pem -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15 -o LogLevel=ERROR ec2-user@"$IP" "$@"
