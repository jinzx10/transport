#!/bin/bash

#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
##SBATCH --mem=120G
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate
module load mpich/3.4.2

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7

method=`grep '^method_label = ' build_impurity.py | cut --delimiter="'" --fields=2`

dir=$HOME/projects/transport/models/CoCu_chain_clean
cd ${dir}

mpirun -np 4 python build_impurity.py  > build_impurity_${method}.out

