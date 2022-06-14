#!/bin/bash

#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=50G
#SBATCH --job-name=imp_dmrg
#SBATCH --array=0-10

source $HOME/.bashrc

module load openmpi/4.0.7
conda activate block2

timestamp=`date +%y%m%d-%H%M%S`

#gate_list=(`seq 0 -0.001 -0.1`)
mu_list=(`seq 0.0 0.02 0.2`)

opt_mu=False
#-----------------------------------
imp_atom=Co
nocc_act=6
nvir_act=11
chem_pot=${mu_list[${SLURM_ARRAY_TASK_ID}]}

#gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}
gate=0.000
#-----------------------------------

dir=$HOME/projects/transport/models/Cu_fcc_100
cd ${dir}

WORK=/scratch/global/zuxin/imp_dmrg/${imp_atom}/gate${gate}/
SCRATCH=${WORK}/tmp/mu${chem_pot}/
SAVEDIR=${WORK}/save/mu${chem_pot}/
PREFIX=/home/zuxin/
NUM_THREADS=14

mkdir -p ${WORK} ${SCRATCH} ${SAVEDIR}

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${NUM_THREADS}

script=imp_dmrg_${imp_atom}_gate${gate}_mu${chem_pot}.py
output=imp_dmrg_${imp_atom}_gate${gate}_mu${chem_pot}.out

sed -e"s/IMP_ATOM/${imp_atom}/" \
    -e"s/MODE/production/" \
    -e"s/CHEM_POT/${chem_pot}/" \
    -e"s/GATE/${gate}/" \
    -e"s/NOCC_ACT/${nocc_act}/" \
    -e"s/NVIR_ACT/${nvir_act}/" \
    -e"s:SCRATCH:${SCRATCH}:" \
    -e"s:SAVEDIR:${SAVEDIR}:" \
    -e"s:PREFIX:${PREFIX}:" \
    -e"s/NUM_THREADS/${NUM_THREADS}/" \
    -e"s/OPT_MU/${opt_mu}/" \
    imp_dmrg.py > ${WORK}/${script}

cd ${WORK}
python -u ${script} > ${output} 2>&1


