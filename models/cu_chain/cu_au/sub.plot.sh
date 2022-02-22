#!/bin/bash

#SBATCH --partition=smallmem
#SBATCH --output=slurm.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --job-name=cu_au_iao

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/cu_chain/cu_au
cd ${dir}

timestamp=`date +%y%m%d-%H%M%S`
output="plot_${timestamp}.out"

data_suffix="220215-021345"

echo `date`

mkdir -p data-${data_suffix}/plot-${timestamp}
python ${dir}/plot.py ${data_suffix} ${timestamp} > ${output}

