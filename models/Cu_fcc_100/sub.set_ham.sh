#!/bin/bash

#SBATCH --partition=parallel,serial,smallmem
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --job-name=set_ham

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=28
export OMP_NUM_THREADS=28

dir=$HOME/projects/transport/models/Cu_fcc_100

Co_basis=`grep 'Co_basis = ' set_ham_Co.py | cut --delimiter="'" --fields=2`
Cu_basis=`grep 'Cu_basis = ' set_ham_Co.py | cut --delimiter="'" --fields=2`

#datadir=${dir}/Co_${Co_basis}_Cu_${Cu_basis}
datadir=${dir}/data

cd ${dir}
mkdir -p ${datadir}

#-----------------------------------
use_dft=True
xcfun=pbe0
do_restricted=True

left=1.8
right=1.8
spacing=3.6

use_smearing=False
smearing_sigma=0.05

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

suffix=a${spacing}_l${left}_r${right}_${method}_${Co_basis}_${Cu_basis}

script=set_ham_Co_${suffix}.py
output=set_ham_Co_${suffix}.out

sed -e"s/DO_RESTRICTED/${do_restricted}/" \
    -e"s/USE_DFT/${use_dft}/" \
    -e"s/XCFUN/${xcfun}/" \
    -e"s/LEFT/${left}/" \
    -e"s/RIGHT/${right}/" \
    -e"s/SPACING/${spacing}/" \
    -e"s/NAT_CU/${nat_cu}/" \
    -e"s/USE_SMEARING/${use_smearing}/" \
    -e"s/SMEARING_SIGMA/${smearing_sigma}/" \
    set_ham_Co.py > ${script}

python ${script} --datadir=${datadir} > ${output} 2>&1
mv ${script} ${output} ${datadir}


