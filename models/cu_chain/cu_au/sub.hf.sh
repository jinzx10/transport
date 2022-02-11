#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --job-name=cu_au_hf

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/gold_chain/cu_au
cd ${dir}

timestamp=`date +%y%m%d-%H%M%S`
savedir="data-${timestamp}"
output="cu_au_${timestamp}.out"

echo `date`
echo "savedir=${savedir}"

mkdir -p ${savedir}
python ${dir}/hf.py --savedir=${savedir} > ${output}
