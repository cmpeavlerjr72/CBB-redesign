#!/usr/bin/env bash
# operator: full chain after a (re)launch. op_go.sh <SHA_main> <SHA_queue> "<streamA arms>" "<streamB arms>"
cd ~/cbb
until [ -f logs/SETUP_DONE ]; do sleep 10; done
bash ~/op_main.sh "$1" || exit 1
nohup bash ~/op_sync.sh > /dev/null 2>&1 < /dev/null &
nohup bash ~/op_stream.sh 96 $3 > ~/streamA.out 2>&1 < /dev/null &
sleep 20
nohup bash ~/op_stream.sh 96 $4 > ~/streamB.out 2>&1 < /dev/null &
bash ~/op_q_setup.sh "$2" > ~/qsetup.out 2>&1
wait
