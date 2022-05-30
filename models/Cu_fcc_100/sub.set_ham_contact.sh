#!/bin/bash

#SBATCH --partition=parallel,serial,smallmem
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=50G
#SBATCH --job-name=set_ham_contact

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=14
export OMP_NUM_THREADS=14

dir=$HOME/projects/transport/models/Cu_fcc_100

#-----------------------------------
imp_atom=Co
use_dft=True
xcfun=pbe0
do_restricted=True

nl=4 # change to 2 for Ni/Fe and 4 for Co
nr=3
left=1.8
right=1.8
latconst=3.6

use_smearing=False
smearing_sigma=0
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

imp_basis=`grep 'imp_basis = ' set_ham_contact.py | cut --delimiter="'" --fields=2`
Cu_basis=`grep 'Cu_basis = ' set_ham_contact.py | cut --delimiter="'" --fields=2`

suffix=a${latconst}_nl${nl}_nr${nr}_l${left}_r${right}_${method}_${imp_basis}_${Cu_basis}

script=set_ham_contact_${imp_atom}_${suffix}.py
output=set_ham_contact_${imp_atom}_${suffix}.out

datadir=${dir}/${imp_atom}

cd ${dir}
mkdir -p ${datadir}

sed -e"s/TEST/production/" \
    -e"s/IMP_ATOM/${imp_atom}/" \
    -e"s/DO_RESTRICTED/${do_restricted}/" \
    -e"s/USE_DFT/${use_dft}/" \
    -e"s/XCFUN/${xcfun}/" \
    -e"s/NUM_LEFT/${nl}/" \
    -e"s/NUM_RIGHT/${nr}/" \
    -e"s/LEFT/${left}/" \
    -e"s/RIGHT/${right}/" \
    -e"s/LATCONST/${latconst}/" \
    -e"s/USE_SMEARING/${use_smearing}/" \
    -e"s/SMEARING_SIGMA/${smearing_sigma}/" \
    set_ham_contact.py > ${script}

python -u ${script} --datadir=${datadir} > ${output} 2>&1
mv ${script} ${output} ${datadir}


