#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=smallmem
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/gold_chain/k_conv
cd ${dir}

timestamp=`date +%Y%m%d%H%M%S`
savedir="data-${timestamp}"
output="k_conv_${timestamp}.out"

echo `date`
echo "savedir=${savedir}"

mkdir -p ${savedir}
python ${dir}/hf.py --savedir=${savedir} > ${output}

