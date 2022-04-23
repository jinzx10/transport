#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=serial,parallel,smallmem
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --job-name=CoCu_set_ham

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/CoCu_chain
datadir=${dir}/Co_svp_Cu_svp_bracket_pbe_v2

cd ${dir}
mkdir -p ${datadir}

timestamp=`date +%y%m%d-%H%M%S`
output="CoCu_set_ham_${timestamp}.out"

python ${dir}/CoCu_set_ham_v2.py --datadir=${datadir} > ${datadir}/${output}
