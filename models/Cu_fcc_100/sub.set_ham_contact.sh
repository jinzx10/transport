#!/bin/bash

#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=50G
#SBATCH --job-name=set_ham_contact
##SBATCH --array=51-100%1

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=14
export OMP_NUM_THREADS=14

dir=$HOME/projects/transport/models/Cu_fcc_100
cd ${dir}

#-----------------------------------
imp_basis=`grep 'imp_basis = ' set_ham_contact.py | cut --delimiter="'" --fields=2`
Cu_basis=`grep 'Cu_basis = ' set_ham_contact.py | cut --delimiter="'" --fields=2`

#-----------------------------------
imp_atom=Co
use_dft=False
xcfun=pbe
do_restricted=True
plot_lo=True

left=1.8
right=1.8
latconst=3.6

cell=${imp_atom}_${imp_basis}_Cu_${Cu_basis}_l${left}_r${right}_a${latconst}

#-----------------------------------
#gate_list=(`seq 0 0.001 0.2`)
#gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}
#
#idx_last=`bc <<< "${SLURM_ARRAY_TASK_ID}-1"`
#gate_last=${gate_list[${idx_last}]}
#
#echo 'idx_last = ' ${idx_last}
#echo 'gate_last = ' ${gate_last}
gate=0.00

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
#mf_load_fname=${imp_atom}/${cell}_${method}_gate${gate_last}.chk
mf_load_fname="${imp_atom}/${cell}_${method}_gate0.00.chk"

echo 'mf_load_fname' ${mf_load_fname}

suffix=a${latconst}_l${left}_r${right}_${method}_gate${gate}_${imp_basis}_${Cu_basis}

script=set_ham_contact_${imp_atom}_${suffix}.py
output=set_ham_contact_${imp_atom}_${suffix}.out

datadir=${dir}/${imp_atom}

mkdir -p ${datadir}

sed -e"s/MODE/production/" \
    -e"s/IMP_ATOM/${imp_atom}/" \
    -e"s/DO_RESTRICTED/${do_restricted}/" \
    -e"s/USE_DFT/${use_dft}/" \
    -e"s/XCFUN/${xcfun}/" \
    -e"s/LEFT/${left}/" \
    -e"s/RIGHT/${right}/" \
    -e"s/LATCONST/${latconst}/" \
    -e"s/GATE/${gate}/" \
    -e"s/PLOT_LO/${plot_lo}/" \
    -e"s:MF_LOAD_FNAME:${mf_load_fname}:" \
    set_ham_contact.py > ${script}

echo 'imp atom = ' ${imp_atom}
echo 'mf_load_fname = ' ${mf_load_fname}
echo 'suffix = ' ${suffix}

python -u ${script} --datadir=${datadir} > ${output} 2>&1
mv ${script} ${output} ${datadir}


