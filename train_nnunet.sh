#!/bin/bash
#SBATCH  --gres=gpu:p100:1 -p compsci-gpu --time=10-00:00:00

source /usr/xtmp/jly16/mammoproj/dev/bin/activate
echo "start running"
nvidia-smi

nnUNet_train 2d nnUNetTrainerV2 Task501_cbis-ddsm 0 --npz