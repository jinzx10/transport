#!/bin/bash

#SBATCH --partition=smallmem
#SBATCH --output=slurm.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/gold_chain/nat_conv
cd ${dir}

python ${dir}/conv.py > conv.out
