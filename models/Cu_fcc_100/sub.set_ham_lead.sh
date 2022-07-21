#!/bin/bash

#SBATCH --partition=parallel,serial,smallmem
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --job-name=set_ham_lead

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=14
export OMP_NUM_THREADS=14

dir=$HOME/projects/transport/models/Cu_fcc_100

#-----------------------------------
use_dft=False
xcfun=pbe0
do_restricted=True

latconst=3.6
num_layer=8

use_smearing=False
smearing_sigma=0

load_mf=False
load_solver_label='smearing0'
#-----------------------------------

if [[ ${do_restricted} == "True" ]]; then
    method=r
else
    method=u
fi

if [[ ${use_dft} == "True" ]]; then
    method=${method}ks_${xcfun}
else
    method=${method}hf
fi

if [[ ${use_smearing} == "True" ]]; then
    method=${method}_smearing${smearing_sigma}
else
    method=${method}_newton
fi

Cu_basis=`grep 'Cu_basis = ' set_ham_lead.py | cut --delimiter="'" --fields=2`

suffix=a${latconst}_n${num_layer}_${method}_${Cu_basis}

script=set_ham_lead_${suffix}.py
output=set_ham_lead_${suffix}.out

datadir=${dir}/Cu

cd ${dir}
mkdir -p ${datadir}

sed -e"s/TEST/production/" \
    -e"s/DO_RESTRICTED/${do_restricted}/" \
    -e"s/USE_DFT/${use_dft}/" \
    -e"s/XCFUN/${xcfun}/" \
    -e"s/LATCONST/${latconst}/" \
    -e"s/NUM_LAYER/${num_layer}/" \
    -e"s/USE_SMEARING/${use_smearing}/" \
    -e"s/SMEARING_SIGMA/${smearing_sigma}/" \
    -e"s/LOAD_MF/${load_mf}/" \
    -e"s/LOAD_SOLVER_LABEL/${load_solver_label}/" \
    set_ham_lead.py > ${script}

python -u ${script} --datadir=${datadir} > ${output} 2>&1
mv ${script} ${output} ${datadir}


