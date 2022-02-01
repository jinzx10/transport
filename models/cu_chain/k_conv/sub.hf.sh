#!/bin/bash

#SBATCH --partition=smallmem
#SBATCH --output=slurm.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=100G

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/cu_chain/k_conv
cd ${dir}

timestamp=`date +%y%m%d-%H%M%S`
savedir="data-${timestamp}"
output="k_conv_${timestamp}.out"

echo `date`
echo "savedir=${savedir}"

mkdir -p ${savedir}
python ${dir}/hf.py --savedir=${savedir} > ${output}

