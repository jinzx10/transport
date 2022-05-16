#!/bin/bash

#SBATCH --partition=serial,parallel,smallmem
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

dir=$HOME/projects/transport/models/CoCu_chain_clean
cd ${dir}

python imp_rks_check.py > imp_rks_check.out 2>&1

