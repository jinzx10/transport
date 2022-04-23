#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=serial,parallel,smallmem
##SBATCH --partition=smallmem
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --job-name=Cu_mf

export MKL_NUM_THREADS=28
export OMP_NUM_THREADS=28

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/CoCu_chain
datadir=${dir}/Cu_svp_bracket_pbe_v2

cd ${dir}
mkdir -p ${datadir}

timestamp=`date +%y%m%d-%H%M%S`
output="Cu_mf_${timestamp}.out"

python ${dir}/Cu_mf_v2.py --datadir=${datadir} > ${datadir}/${output}
