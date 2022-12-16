#!/bin/bash
#submit to the normal COMPUTE partition for normal CPUS
#SBATCH -p COMPUTE
#request a different amount of time to the default 12h
#SBATCH --time 24:00:00
#requesting one node
#SBATCH -N1
#requesting 12 cpus
#SBATCH -n12
#SBATCH --mail-user=zcaplcu@ucl.ac.uk
#SBATCH --mail-type=ALL
eval "$(/share/apps/anaconda/3-2022.05/bin/conda shell.bash hook)"
conda activate env
cd /home/xzcaplcu/repo/qtrkx-gnn-tracking/
srun python3 fullmonty.py 
