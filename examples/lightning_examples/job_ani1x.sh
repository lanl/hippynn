#/bin/bash


NRANKS=4
NGPUS_PER_NODE=4
NNODES=$((NRANKS/NGPUS_PER_NODE))


echo $NRANKS
echo $NGPUS_PER_NODE
echo $NNODES

python split_ani1x.py --ranks $NRANKS

srun --nodes=$NNODES --ntasks-per-node=$NGPUS_PER_NODE python ani1x_lightning_by_ranks.py




