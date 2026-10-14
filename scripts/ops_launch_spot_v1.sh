#!/usr/bin/env bash
# Local: launch tonight's box (operator 2026-09-30). Tries the AZs in runbook order with the given market.
#   scripts/ops_launch_spot_v1.sh spot|ondemand
#   VOLGB=<gb> overrides the root volume (default 200); AZS="us-east-2b ..." overrides the AZ order (default 2c 2b 2a; 2a last on purpose, see aws_launch_chain.md s20).
# Prints the instance id, AZ and launch time on success (one line: LAUNCHED <id> <az> <iso>).
set -uo pipefail
export MSYS_NO_PATHCONV=1
MODE="${1:-spot}"
# Root volume GB. 200 since 2026-10-14: the full-size 5,710 x 200 paired read filled a 100 GB root during post-processing (docs/ops/aws_launch_chain.md s20).
VOLGB="${VOLGB:-200}"
R=us-east-2
AMI=$(aws ec2 describe-images --region $R --owners amazon --filters "Name=name,Values=al2023-ami-2023*-kernel-6.1-x86_64" "Name=state,Values=available" \
      --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
echo "AMI $AMI"
declare -A SUBNET=( [us-east-2c]=subnet-02b40dd48e9ed2bd9 [us-east-2b]=subnet-07dfcc25641a55194 [us-east-2a]=subnet-0f8629b0edec07b47 )
for AZ in ${AZS:-us-east-2c us-east-2b us-east-2a}; do
  if [ "$MODE" = spot ]; then
    OUT=$(aws ec2 run-instances --region $R --image-id "$AMI" --instance-type ${ITYPE:-c7a.48xlarge} --key-name cfb-sweep \
      --security-group-ids sg-05aacf67a5a55fbf7 --subnet-id "${SUBNET[$AZ]}" \
      --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
      --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${VOLGB},VolumeType=gp3,DeleteOnTermination=true}" \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cbb-box-0930}]' \
      --query 'Instances[0].InstanceId' --output text 2>&1)
  else
    OUT=$(aws ec2 run-instances --region $R --image-id "$AMI" --instance-type ${ITYPE:-c7a.48xlarge} --key-name cfb-sweep \
      --security-group-ids sg-05aacf67a5a55fbf7 --subnet-id "${SUBNET[$AZ]}" \
      --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${VOLGB},VolumeType=gp3,DeleteOnTermination=true}" \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cbb-box-0930}]' \
      --query 'Instances[0].InstanceId' --output text 2>&1)
  fi
  case "$OUT" in
    i-*) echo "LAUNCHED $OUT $AZ $(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit 0 ;;
    *) echo "$AZ: $(echo "$OUT" | tail -1 | cut -c1-160)" ;;
  esac
done
echo "NO CAPACITY in any AZ"; exit 1
