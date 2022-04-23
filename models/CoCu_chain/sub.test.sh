#!/bin/bash

#SBATCH --output=test.slurm.out
#SBATCH --partition=serial
#SBATCH --time=120:00:00
##SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
##SBATCH --mem=240G
#SBATCH --job-name=test
#SBATCH --array=0-7%4

mu=(`seq 0.060 0.001 0.090`)
sleep 30
echo 'id = ' $SLURM_ARRAY_TASK_ID ', mu = '  ${mu[${SLURM_ARRAY_TASK_ID}]} >> test.array.out
