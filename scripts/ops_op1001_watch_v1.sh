#!/usr/bin/env bash
# local poller: emits new lines of box progress files and instance state changes
S="$(dirname "$0")"; export MSYS_NO_PATHCONV=1
prev=""; pst=""
while true; do
  st=$(aws ec2 describe-instances --region us-east-2 --instance-ids "$(cat $S/iid.txt)" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  [ "$st" != "$pst" ] && echo "STATE $st $(date +%H:%M:%S)" && pst=$st
  case "$st" in shutting-down|terminated|stopped|stopping) echo "INSTANCE GONE $st"; exit 0;; esac
  cur=$(bash $S/bx.sh 'cat ~/setup.out ~/go.out ~/streamA.out ~/streamB.out ~/qA.out ~/qB.out ~/qsetup.out ~/qC.out 2>/dev/null' 2>/dev/null)
  if [ -n "$cur" ]; then comm -13 <(echo "$prev" | sort) <(echo "$cur" | sort); prev="$cur"; fi
  sleep 60
done
