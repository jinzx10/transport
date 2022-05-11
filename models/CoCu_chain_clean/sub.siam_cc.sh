#!/bin/bash

#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

dir=$HOME/projects/transport/models/CoCu_chain_clean
cd ${dir}

python siam_cc.py > siam_cc.out
