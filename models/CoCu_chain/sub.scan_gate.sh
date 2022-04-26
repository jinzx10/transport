#!/bin/bash

#SBATCH --partition=serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=7
#SBATCH --job-name=scan_gate
#SBATCH --array=0-20%8
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7

dir=$HOME/projects/transport/models/CoCu_chain
cd ${dir}

gate_list=(`seq -0.1 0.01 0.1`)

mu=-0.161393
nb=30 # number of bath ENERGIES
disc=log
solver=cc
delta=0.01
base=1.5 # base for log discretization
nbpe=1
do_cas=True
casno=ci
gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}
wl0=-0.15
wh0=0.45
eri_scale=1.0
calc_occ_only=False

method=${solver}
if [[ ${do_cas} == "True" ]]; then
    method=${method}_casno${casno}
fi

# suffix without gate
suffix=eri${eri_scale}_nb${nb}_mu${mu}_${disc}_${method}_delta${delta}_base${base}_nbpe${nbpe}

outdir=${dir}/scan_gate_${suffix}
mkdir -p ${outdir}

suffix=gate${gate}_${suffix}

sed -e"s/CHEMICAL_POTENTIAL/${mu}/" \
    -e"s/NUM_BATH_ENERGY/${nb}/" \
    -e"s/DISC_TYPE/${disc}/" \
    -e"s/SOLVER_TYPE/${solver}/" \
    -e"s/DELTA/${delta}/" \
    -e"s/LDOS_FILE_NAME/ldos_${suffix}.dat/" \
    -e"s/LOG_DISC_BASE/${base}/" \
    -e"s/NUM_BATH_PER_ENERGY/${nbpe}/" \
    -e"s/GATE/${gate}/" \
    -e"s/WL_MU/${wl0}/" \
    -e"s/WH_MU/${wh0}/" \
    -e"s/CALC_OCC_ONLY/${calc_occ_only}/" \
    -e"s/ERI_SCALE/${eri_scale}/" \
    -e"s/DO_CAS/${do_cas}/" \
    -e"s/CASNO/${casno}/" \
    run_dmft.py > run_dmft_${suffix}.py

python run_dmft_${suffix}.py  > ${outdir}/run_dmft_${suffix}.out
mv run_dmft_${suffix}.py ${outdir}



