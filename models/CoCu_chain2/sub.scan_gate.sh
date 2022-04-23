#!/bin/bash

#SBATCH --partition=serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=7
#SBATCH --job-name=scan_gate
#SBATCH --array=0-40%8
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=7

dir=$HOME/projects/transport/models/CoCu_chain
cd ${dir}

# ensure fixed width
ID=${SLURM_ARRAY_TASK_ID}
if [ ${#ID} == 1 ]; then
    ID=0${ID}
fi

gate_list=(`seq -w -0.090 0.001 -0.050`)
gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}

mu=-0.161393
nb=40 # number of bath ENERGIES
disc=log
solver=cc
delta=0.01
base=1.5 # base for log discretization
nbpe=1
casno=ci
wl0=-0.15
wh0=0.45
calc_occ_only=True

# suffix without gate
suffix=mu${mu//.}_nb${nb}_${disc}_${solver}_delta${delta//.}_base${base//.}_nbpe${nbpe} 

outdir=${dir}/scan_gate_${suffix}
mkdir -p ${outdir}

suffix=gate${gate//.}_${suffix}

sed -e"s/CHEMICAL_POTENTIAL/${mu}/" \
    -e"s/NUM_BATH_ENERGY/${nb}/" \
    -e"s/DISC_TYPE/${disc}/" \
    -e"s/SOLVER_TYPE/${solver}/" \
    -e"s/DELTA/${delta}/" \
    -e"s/LOG_DISC_BASE/${base}/" \
    -e"s/NUM_BATH_PER_ENERGY/${nbpe}/" \
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


python run_dmft_${suffix}.py  > ${outdir}/run_dmft_${suffix}.out
rm run_dmft_${suffix}.py



