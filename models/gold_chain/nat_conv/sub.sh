#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/gold_chain/nat_conv
cd $dir

python $dir/hf.py > nat_conv.out

