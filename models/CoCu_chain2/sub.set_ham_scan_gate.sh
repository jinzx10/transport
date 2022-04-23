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

gate_list=(`seq -0.050 0.002 0.050`)
gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}

use_pbe=True
do_restricted=True

if [[ ${do_restricted} == "True" ]]; then
    suffix=r
else
    suffix=u
fi

if [[ ${use_pbe} == "True" ]]; then
    suffix=${suffix}ks
else
    suffix=${suffix}hf
fi

suffix=${suffix}_gate${gate}


script=CoCu_set_ham_${suffix}.py
output=CoCu_set_ham_${suffix}.out

sed -e"s/GATE/${gate}/" \
    -e"s/DO_RESTRICTED/${do_restricted}/" \
    -e"s/USE_PBE/${use_pbe}/" \
    CoCu_set_ham.py > ${script}


python ${script} --datadir=${datadir} > ${output}
mv ${script} ${output} ${datadir}


