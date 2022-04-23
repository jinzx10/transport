#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --job-name=CoCu_set_ham

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=28
export OMP_NUM_THREADS=28

dir=$HOME/projects/transport/models/CoCu_chain2
datadir=${dir}/Co_svp_Cu_svp_bracket

cd ${dir}
mkdir -p ${datadir}

gate=0.05
output=CoCu_set_ham_gate${gate}.out

sed "s/GATE/${gate}/" ${dir}/CoCu_set_ham.py > ${dir}/CoCu_set_ham_gate${gate}.py

python ${dir}/CoCu_set_ham_gate${gate}.py --datadir=${datadir} > ${datadir}/${output}
