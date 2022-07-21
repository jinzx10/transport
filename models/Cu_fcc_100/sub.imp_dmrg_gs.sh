#!/bin/bash

#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --job-name=Co-gs-gate0
#SBATCH --array=9-9

source $HOME/.bashrc

module load openmpi/4.0.7
conda activate block2

timestamp=`date +%y%m%d-%H%M%S`

mu_list=(`seq 0.10 0.01 0.20`)

#-----------------------------------
imp_atom=Co
nocc_act=6
nvir_act=9
chem_pot=${mu_list[${SLURM_ARRAY_TASK_ID}]}

gate=0.000
#-----------------------------------

dir=$HOME/projects/transport/models/Cu_fcc_100
cd ${dir}

WORK=/scratch/global/zuxin/Cu_fcc_100/${imp_atom}/gate${gate}/dmrg_mu${chem_pot}/
SCRATCH=${WORK}/tmp/
SAVEDIR=${WORK}/save/
PREFIX=/home/zuxin/
NUM_THREADS=9

mkdir -p ${WORK} ${SCRATCH} ${SAVEDIR}

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${NUM_THREADS}

script=imp_dmrg_gs_${imp_atom}_gate${gate}_mu${chem_pot}.py
output=imp_dmrg_gs_${imp_atom}_gate${gate}_mu${chem_pot}.out

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
    imp_dmrg_gs.py > ${WORK}/${script}

cd ${WORK}
python -u ${script} > ${output} 2>&1


