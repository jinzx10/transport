#!/bin/bash

#SBATCH --output=fock_conv_check.out
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=30G

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/chain/
cd $dir

python $dir/fock_conv_check.py

