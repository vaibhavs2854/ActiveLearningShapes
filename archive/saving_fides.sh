#!/bin/bash
#SBATCH -p compsci --time=10-00:00:00

source /usr/xtmp/vs196/mammoproj/Env/trainenv2/bin/activate
echo "start running"

python saving_fides.py