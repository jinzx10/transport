#!/bin/bash

#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --job-name=Co_mfgs_gate0
#SBATCH --array=0-20

source $HOME/.bashrc

timestamp=`date +%y%m%d-%H%M%S`

mu_list=(`seq 0.00 0.01 0.20`)

#-----------------------------------
imp_atom=Co
chem_pot=${mu_list[${SLURM_ARRAY_TASK_ID}]}
gate=0.000
#-----------------------------------

dir=$HOME/projects/transport/models/Cu_fcc_100
cd ${dir}

WORK=/scratch/global/zuxin/Cu_fcc_100/${imp_atom}/gate${gate}/mf_mu_scan/
NUM_THREADS=7

mkdir -p ${WORK}

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${NUM_THREADS}

script=imp_mf_gs_${imp_atom}_gate${gate}_mu${chem_pot}.py
output=imp_mf_gs_${imp_atom}_gate${gate}_mu${chem_pot}.out

sed -e"s/IMP_ATOM/${imp_atom}/" \
    -e"s/MODE/production/" \
    -e"s/CHEM_POT/${chem_pot}/" \
    -e"s/GATE/${gate}/" \
    -e"s/NUM_THREADS/${NUM_THREADS}/" \
    imp_mf_gs.py > ${WORK}/${script}

cd ${WORK}
python -u ${script} > ${output} 2>&1


