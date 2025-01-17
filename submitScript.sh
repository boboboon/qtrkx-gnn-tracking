#!/bin/bash
#submit to the normal COMPUTE partition for normal CPUS
#SBATCH -p COMPUTE
#request a different amount of time to the default 12h
#SBATCH --time 60:00:00
#requesting one node
#SBATCH -N1
#requesting 1 cpu
#SBATCH -n1
#SBATCH --mail-user=zcaplcu@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -o /home/xzcaplcu/repo/qtrkx-gnn-tracking/slurms/MPSoutput.out
#SBATCH -e /home/xzcaplcu/repo/qtrkx-gnn-tracking/slurms/MPSerror.out
#SBATCH --mem-per-cpu 20G 
eval "$(/share/apps/anaconda/3-2022.05/bin/conda shell.bash hook)"
conda activate qtrkx
cd /home/xzcaplcu/repo/qtrkx-gnn-tracking/
srun python3 train.py configs/remote_MPS.yaml MPS-4

