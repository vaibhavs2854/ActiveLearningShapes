#!/bin/bash
#SBATCH  --gres=gpu:1 -p compsci-gpu --time=10-00:00:00

source /usr/xtmp/vs196/mammoproj/Env/trainenv2/bin/activate
echo "start running"
nvidia-smi

python experiment.py --random_seed=1