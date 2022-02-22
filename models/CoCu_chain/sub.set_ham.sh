#!/bin/bash

#SBATCH --output=slurm.out
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --job-name=CoCu_set_ham

source $HOME/.bashrc
conda activate

dir=$HOME/projects/transport/models/CoCu_chain
cd ${dir}

timestamp=`date +%y%m%d-%H%M%S`
output="CoCu_set_ham_${timestamp}.out"

echo `date`

python ${dir}/CoCu_set_ham.py > ${output}
