#!/bin/bash
#SBATCH  --gres=gpu:v100:1 -p compsci-gpu --time=10-00:00:00

source /usr/xtmp/jly16/mammoproj/dev/bin/activate
echo "start running"
nvidia-smi

python experiment.py --nnunet --run_id 3_03_auto_oracle_final1_nnunet --output_dir /usr/xtmp/jly16/mammoproj/nnunet_integration_tmp/AllOracleRuns  --random_seed 44