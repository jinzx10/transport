#!/bin/bash

#SBATCH --partition=parallel,serial,smallmem
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=110G
#SBATCH --job-name=imp_dmrg
#SBATCH --array=0-10%5

source $HOME/.bashrc

module load openmpi/4.0.7
conda activate block2

timestamp=`date +%y%m%d-%H%M%S`
chem_pot_list=(`seq 0 0.02 0.2`)
#-----------------------------------
imp_atom=Fe
nocc_act=5
nvir_act=6
chem_pot=${chem_pot_list[${SLURM_ARRAY_TASK_ID}]}
#-----------------------------------

dir=$HOME/projects/transport/models/Cu_fcc_100
cd ${dir}

WORK=/scratch/global/zuxin/imp_dmrg/${imp_atom}_${chem_pot}
SCRATCH=${WORK}/tmp
SAVEDIR=${WORK}/save
NUM_THREADS=28

mkdir -p ${WORK} ${SCRATCH} ${SAVEDIR}

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${NUM_THREADS}

script=imp_dmrg_${imp_atom}.py
output=imp_dmrg_${imp_atom}.out

sed -e"s/IMP_ATOM/${imp_atom}/" \
    -e"s/MODE/${production}/" \
    -e"s/CHEM_POT/${chem_pot}/" \
    -e"s/NOCC_ACT/${nocc_act}/" \
    -e"s/NVIR_ACT/${nvir_act}/" \
    -e"s:SCRATCH:${SCRATCH}:" \
    -e"s:SAVEDIR:${SAVEDIR}:" \
    -e"s:NUM_THREADS:${NUM_THREADS}:" \
    imp_dmrg.py > ${WORK}/${script}

cd ${WORK}
python -u ${script} > ${output} 2>&1


