#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --job-name=run_dmft
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export OMP_NUM_THREADS=1

dir=$HOME/projects/transport/models/CoCu_chain

cd ${dir}

python run_dmft.py  > run_dmft.out2
