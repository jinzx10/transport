#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/gold_chain/cu_au
cd ${dir}

#timestamp=`date +%y%m%d-%H%M%S`
#savedir="data-${timestamp}"
#output="nat_conv_${timestamp}.out"
#
#echo `date`
#echo "savedir=${savedir}"
#
#mkdir -p ${savedir}
#python ${dir}/hf.py --savedir=${savedir} > ${output}

python ${dir}/conv.py > conv.out
