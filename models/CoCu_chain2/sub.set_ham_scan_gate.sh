#!/bin/bash

#SBATCH --partition=serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=7
#SBATCH --job-name=set_ham_scan_gate
#SBATCH --array=0-50%8
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=7
export OMP_NUM_THREADS=7

dir=$HOME/projects/transport/models/CoCu_chain2
datadir=${dir}/Co_svp_Cu_svp_bracket

cd ${dir}
mkdir -p ${datadir}

#-----------------------------------
use_pbe=False
do_restricted=True
left=3.6
right=3.6

gate_list=(`seq -0.050 0.002 0.050`)
gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}

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
mv ${script} ${output} ${datadir}


