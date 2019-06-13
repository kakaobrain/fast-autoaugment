#!/usr/bin/env bash
set -euo pipefail

# A script which runs distributed parallel learning on Brain Cloud tasks with Horovod.
#
# Usage:
#
#   DATA=cifar10 BATCH=32 /data/private/bin/task-resnet200.sh
#   /data/private/bin/task-slave.sh

if [[ "$TASK_NAME" != 'task1' ]]
then
  echo "task-hvd.sh must be run at task1"
  exit 1
fi

# ------------------------------------------------------------------------------

LAUNCHER="${1:-horovod}" # pytorch|[horovod]

# ------------------------------------------------------------------------------

HOSTS=()
for i in $( seq "$NUM_TASKS" )
do
  HOSTS+=( "task$i:$TASK_RESOURCE_GPUS" )
done
HOSTS="$( echo "${HOSTS[@]}" | sed 's/ /,/g' )"

echo "Hosts: $HOSTS"

# ------------------------------------------------------------------------------

mpi () {
  mpirun \
    --allow-run-as-root \
    --mca plm_rsh_agent 'ssh -qo StrictHostKeyChecking=no' \
    --mca btl_vader_single_copy_mechanism none \
    -x http_proxy  -x HTTP_PROXY \
    -x https_proxy -x HTTPS_PROXY \
    -x no_proxy    -x NO_PROXY \
    -H "$HOSTS" "$@"
}

echo "sleep for a while..."
sleep 120
echo -n "Checking whether all tasks are ready..."
until mpi true &>/dev/null; do sleep 1; done
echo " OK"
sleep 60

# ------------------------------------------------------------------------------
# Kill slave tasks with return code 1 when an error occurs.

catch () {
  echo "Master task failed! Killing slave tasks..."
  mpi pkill -9 sleep &>/dev/null || true
}

trap catch ERR

# ------------------------------------------------------------------------------
# Run an experiment.

echo "Run an experiment..."
#echo $CONFIG
mpi bash -c 'cd /data/private/fast-autoaugment-public/FastAutoAugment
PYTHONPATH=/data/private/fast-autoaugment-public python train.py -c confs/resnet200_b4096.yaml --horovod --save ./models/resnet200_b4096_faa.pth --aug fa_reduced_imagenet'

# ------------------------------------------------------------------------------
# Stop slave tasks with exitcode 0.

mpi pkill sleep &>/dev/null || true