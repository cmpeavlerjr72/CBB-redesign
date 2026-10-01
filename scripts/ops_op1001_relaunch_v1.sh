#!/usr/bin/env bash
# local: bring a freshly launched instance up to the full chain. relaunch.sh <iid> "<streamA arms>" "<streamB arms>" <waitA> <waitB>
S="$(dirname "$0")"; export MSYS_NO_PATHCONV=1
IID=$1; echo $IID > $S/iid.txt
aws ec2 wait instance-running --region us-east-2 --instance-ids $IID
aws ec2 describe-instances --region us-east-2 --instance-ids $IID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > $S/ip.txt
IP=$(cat $S/ip.txt); echo "IP $IP"
K="-i /c/Users/devuser/.ssh/cfb-sweep-ohio.pem -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
for i in $(seq 1 20); do bash $S/bx.sh true 2>/dev/null && break; sleep 8; done
scp -q $K /c/Users/devuser/CBB-clean-sheet/scripts/box_bootstrap_v1.sh $S/box_setup.sh $S/op_main.sh $S/op_stream.sh $S/op_go.sh $S/op_sync.sh $S/op_q_setup.sh $S/op_q_laneA.sh $S/op_q_laneB.sh ec2-user@$IP:~/
grep "^HF_TOKEN=" /c/Users/devuser/CBB-clean-sheet/.env | sed 's/^HF_TOKEN=/export HF_TOKEN=/' | bash $S/bx.sh 'umask 077; cat > ~/.hf_env'
bash $S/bx.sh "sudo shutdown -h 06:50 >/dev/null 2>&1; bash ~/box_bootstrap_v1.sh install >/dev/null && sg docker -c 'bash ~/box_bootstrap_v1.sh clone f670e02' && chmod +x ~/cbb/scripts/*.sh"
bash $S/bx.sh "nohup sg docker -c 'bash ~/box_setup.sh' > ~/setup.out 2>&1 < /dev/null & nohup sg docker -c 'bash ~/op_go.sh f670e02 0ac56fd \"$2\" \"$3\"' > ~/go.out 2>&1 < /dev/null & echo chain started"
[ -n "${4:-}" ] && bash $S/bx.sh "nohup sg docker -c 'until [ -f ~/cbb2/logs/PARITY_PASS ]; do sleep 15; done; bash ~/op_q_laneB.sh $4' > ~/qB.out 2>&1 < /dev/null & echo qB armed"
[ -n "${5:-}" ] && bash $S/bx.sh "nohup sg docker -c 'until [ -f ~/cbb2/logs/PARITY_PASS ]; do sleep 15; done; bash ~/op_q_laneA.sh $5' > ~/qA.out 2>&1 < /dev/null & echo qA armed"
date "+%H:%M:%S"
