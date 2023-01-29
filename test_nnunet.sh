#!/bin/bash
#SBATCH  --gres=gpu:p100:1 -p compsci-gpu --time=10-00:00:00

source /usr/xtmp/jly16/mammoproj/dev/bin/activate
echo "start running"
nvidia-smi

nnUNet_predict -i /usr/xtmp/jly16/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/TestA/images/ -o /usr/xtmp/jly16/mammoproj/data/output/TestA/3d/ -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task501_cbis-ddsm
