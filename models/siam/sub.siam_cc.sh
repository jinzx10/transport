#!/bin/bash

#SBATCH --partition=serial,parallel,smallmem
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

timestamp=`date +%y%m%d-%H%M%S`

export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

dir=$HOME/projects/transport/models/siam

WORK=/scratch/global/zuxin/siam_cc_${timestamp}/

mkdir -p ${WORK}
cp ${dir}/siam_cc.py ${WORK}

cd ${WORK}
python siam_cc.py > siam_cc.out
