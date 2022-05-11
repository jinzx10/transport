#!/bin/bash

#SBATCH --partition=serial,parallel,smallmem
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
##SBATCH --mem=120G
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

dir=$HOME/projects/transport/models/CoCu_chain_clean
cd ${dir}

#python imp_rks_check.py  > imp_rks_check.out
python CoCu_set_ham_test.py --datadir=Co_def2-svp_Cu_def2-svp-bracket 

