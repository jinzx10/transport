#!/bin/bash

#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=50G
#SBATCH --job-name=imp_dmrg
#SBATCH --array=0-100:10

source $HOME/.bashrc

module load openmpi/4.0.7
conda activate block2

timestamp=`date +%y%m%d-%H%M%S`
gate_list=(`seq 0 -0.001 -0.1`)
#-----------------------------------
imp_atom=Fe
nocc_act=5
nvir_act=6
chem_pot=0.06
gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}
#-----------------------------------

dir=$HOME/projects/transport/models/Cu_fcc_100
cd ${dir}

WORK=/scratch/global/zuxin/imp_dmrg/${imp_atom}/gate${gate}
SCRATCH=${WORK}/tmp
SAVEDIR=${WORK}/save
NUM_THREADS=14

mkdir -p ${WORK} ${SCRATCH} ${SAVEDIR}

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${NUM_THREADS}

script=imp_dmrg_${imp_atom}_gate${gate}.py
output=imp_dmrg_${imp_atom}_gate${gate}.out

sed -e"s/IMP_ATOM/${imp_atom}/" \
    -e"s/MODE/production/" \
    -e"s/CHEM_POT/${chem_pot}/" \
    -e"s/GATE/${gate}/" \
    -e"s/NOCC_ACT/${nocc_act}/" \
    -e"s/NVIR_ACT/${nvir_act}/" \
    -e"s:SCRATCH:${SCRATCH}:" \
    -e"s:SAVEDIR:${SAVEDIR}:" \
    -e"s:NUM_THREADS:${NUM_THREADS}:" \
    imp_dmrg.py > ${WORK}/${script}

cd ${WORK}
python -u ${script} > ${output} 2>&1


