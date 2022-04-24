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

#-----------------------------------
use_pbe=True
do_restricted=True
left=3.6
right=3.6
gate=0

if [[ ${do_restricted} == "True" ]]; then
    method=r
else
    method=u
fi

if [[ ${use_pbe} == "True" ]]; then
    method=${method}ks
else
    method=${method}hf
fi

suffix=l${left}_r${right}_${method}_gate${gate}

script=CoCu_set_ham_${suffix}.py
output=CoCu_set_ham_${suffix}.out

sed -e"s/GATE/${gate}/" \
    -e"s/DO_RESTRICTED/${do_restricted}/" \
    -e"s/USE_PBE/${use_pbe}/" \
    -e"s/LEFT/${left}/" \
    -e"s/RIGHT/${right}/" \
    CoCu_set_ham.py > ${script}

python ${script} --datadir=${datadir} > ${output}


