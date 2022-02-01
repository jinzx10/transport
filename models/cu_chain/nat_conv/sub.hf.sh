#!/bin/bash

#SBATCH --partition=serial
#SBATCH --output=slurm.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --job-name=au_chain_hf

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/cu_chain/nat_conv
cd ${dir}

timestamp=`date +%y%m%d-%H%M%S`
savedir="data-${timestamp}"
output="nat_conv_${timestamp}.out"

echo `date`
echo "savedir=${savedir}"

mkdir -p ${savedir}
python ${dir}/hf.py --savedir=${savedir} --smearing=0.05 > ${output}

