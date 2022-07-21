#!/bin/bash

#SBATCH --partition=smallmem
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --job-name=homo_lumo
#SBATCH --array=0-200

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

dir=$HOME/projects/transport/models/Cu_fcc_100
cd ${dir}

#-----------------------------------
imp_basis=`grep 'imp_basis = ' get_homo_lumo_energy.py | cut --delimiter="'" --fields=2`
Cu_basis=`grep 'Cu_basis = ' get_homo_lumo_energy.py | cut --delimiter="'" --fields=2`

#-----------------------------------
imp_atom=Ni
use_dft=True
xcfun=pbe0
do_restricted=True

if [[ ${imp_atom} == "Co" ]]; then
    nl=4
else
    nl=2
fi

nr=3
left=1.8
right=1.8
latconst=3.6

cell=${imp_atom}_${imp_basis}_Cu_${Cu_basis}_nl${nl}_nr${nr}_l${left}_r${right}_a${latconst}

#-----------------------------------
gate_list=(`seq -0.1 0.001 0.1`)
gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}

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

#-----------------------------------
mf_load_fname=${imp_atom}/${cell}_${method}_gate${gate}.chk

echo 'mf_load_fname' ${mf_load_fname}

suffix=a${latconst}_nl${nl}_nr${nr}_l${left}_r${right}_${method}_gate${gate}_${imp_basis}_${Cu_basis}

script=get_homo_lumo_energy_${imp_atom}_${suffix}.py
output=get_homo_lumo_energy_${imp_atom}_${suffix}.out

datadir=${dir}/${imp_atom}

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
    -e"s/GATE/${gate}/" \
    -e"s:MF_LOAD_FNAME:${mf_load_fname}:" \
    get_homo_lumo_energy.py > ${script}

echo 'imp atom = ' ${imp_atom}
echo 'mf_load_fname = ' ${mf_load_fname}
echo 'suffix = ' ${suffix}

python -u ${script} --datadir=${datadir} > ${output} 2>&1
mv ${script} ${output} ${datadir}


