#!/bin/bash

#SBATCH --partition=serial,parallel,smallmem
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=pauling001

source $HOME/.bashrc

module load openmpi/4.0.7

conda activate block2

timestamp=`date +%y%m%d-%H%M%S`

dir=$HOME/projects/transport/models/siam/
cd ${dir}

WORK=/scratch/global/zuxin/siam_dmrg_noscdm${timestamp}/
SCRATCH=${WORK}/tmp
SAVEDIR=${WORK}/save

mkdir -p ${WORK} ${SCRATCH} ${SAVEDIR}

NUM_THREADS=28

export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=1

sed -e"s:SCRATCH:${SCRATCH}:" \
    -e"s:SAVEDIR:${SAVEDIR}:" \
    -e"s:NUM_THREADS:${NUM_THREADS}:" \
    siam_dmrg.py > ${WORK}/siam_dmrg.py

cd ${WORK}

python -u siam_dmrg.py > siam_dmrg.out
