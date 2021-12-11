#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=smallmem
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/gold_chain/
cd $dir

python $dir/hf.py > cu_au_conv.out

