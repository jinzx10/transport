#!/bin/bash

#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=gf-Co-0.17
#SBATCH --array=0-4

source $HOME/.bashrc

module load openmpi/4.0.7
conda activate block2

timestamp=`date +%y%m%d-%H%M%S`

#   Fe
#   gate         -0.1    -0.05     0       0.05     0.1
# optimized mu   0.1422  0.1536  0.1544   0.1575   0.1601

#   Co
#   gate         -0.1    -0.05     0       0.05     0.1
# optimized mu   0.1847  0.1829  0.1829   0.1829   0.1859

#   Ni
#   gate         -0.1    -0.05     0       0.05     0.1
# optimized mu   0.0941  0.0842  0.0652   0.0638   0.0655

#-----------------------------------
imp_atom=Co
nocc_act=6
nvir_act=9
chem_pot=0.17

gate=0.000

segs=(0 1 2 3 4)
segment=${segs[${SLURM_ARRAY_TASK_ID}]}
suffix=seg${segment}

#-----------------------------------

dir=$HOME/projects/transport/models/Cu_fcc_100
cd ${dir}

WORK=/scratch/global/zuxin/Cu_fcc_100/${imp_atom}/gate${gate}/dmrg_mu${chem_pot}/
SCRATCH=${WORK}/tmp_${suffix}/
SAVEDIR=${WORK}/save_${suffix}/
PREFIX=/home/zuxin/
NUM_THREADS=28

mkdir -p ${WORK} ${SCRATCH} ${SAVEDIR}

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${NUM_THREADS}

script=gf_imp_dmrg_${imp_atom}_gate${gate}_mu${chem_pot}_${suffix}.py
output=gf_imp_dmrg_${imp_atom}_gate${gate}_mu${chem_pot}_${suffix}.out

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
    -e"s/SEGMENT/${segment}/" \
    imp_dmrg_gf.py > ${WORK}/${script}

cd ${WORK}
python -u ${script} > ${output} 2>&1


