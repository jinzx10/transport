#!/bin/bash

#SBATCH --partition=serial
#SBATCH --output=slurm.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=80G

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/gold_chain/nat_conv
cd ${dir}

suffix='220108-010318'
python ${dir}/conv.py ${suffix} > conv_${suffix}.out
