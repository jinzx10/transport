#!/bin/bash

#SBATCH --output=slurm.dmft50.out
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
##SBATCH --mem=120G
#SBATCH --job-name=run_dmft_50_cc
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7

dir=$HOME/projects/transport/models/CoCu_chain
cd ${dir}

nb=30 # number of bath ENERGIES
mu=-0.161393
disc=log
solver=cc
delta=0.01
base=1.5 # base for log discretization
nbpe=1
casno=cc
gate=-0.07
wl0=-0.15
wh0=0.45
calc_occ_only=False

suffix=gate${gate//.}_nb${nb}_mu${mu//.}_${disc}_${solver}_delta${delta//.}_base${base//.}_nbpe${nbpe}

sed -e"s/CHEMICAL_POTENTIAL/${mu}/" \
    -e"s/NUM_BATH_ENERGY/${nb}/" \
    -e"s/DISC_TYPE/${disc}/" \
    -e"s/SOLVER_TYPE/${solver}/" \
    -e"s/DELTA/${delta}/" \
    -e"s/LOG_DISC_BASE/${base}/" \
    -e"s/NUM_BATH_PER_ENERGY/${nbpe}/" \
    -e"s/LDOS_FILE_NAME/ldos_${suffix}.dat/" \
    -e"s/GATE/${gate}/" \
    -e"s/WL_MU/${wl0}/" \
    -e"s/WH_MU/${wh0}/" \
    -e"s/CALC_OCC_ONLY/${calc_occ_only}/" \
    run_dmft.py > run_dmft_${suffix}.py

if [[ ${solver} == "dmrg" ]]; then
    module unload mpich
    module load openmpi
    sed -i -e"s/DO_CAS/True/"  -e"s/CASNO/${casno}/" run_dmft_${suffix}.py
else
    sed -i -e"s/DO_CAS/False/"  -e"s/CASNO/${casno}/" run_dmft_${suffix}.py
fi

mpirun -np 4 python run_dmft_${suffix}.py  > run_dmft_${suffix}.out
#python run_dmft_${suffix}.py  > run_dmft_${suffix}.out

