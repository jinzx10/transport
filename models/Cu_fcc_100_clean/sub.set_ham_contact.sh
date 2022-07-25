#!/bin/bash

#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --job-name=set_ham_contact
##SBATCH --array=51-100%1

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=8

dir=$HOME/projects/transport/models/Cu_fcc_100_clean/
work=$HOME/projects/transport/models/Cu_fcc_100_clean/

mkdir -p ${work}
cd ${dir}

#------------ cell ----------------
imp_atom=Co

left=1.8
right=1.8
latconst=3.6

imp_basis=`grep 'imp_basis = ' set_ham_contact.py | cut --delimiter="'" --fields=2`
Cu_basis=`grep 'Cu_basis = ' set_ham_contact.py | cut --delimiter="'" --fields=2`

cell=${imp_atom}_${imp_basis}_Cu_${Cu_basis}_l${left}_r${right}_a${latconst}

#------------- method ----------------
xcfun=pbe0
method=rks_${xcfun}

#------------- k sampling ----------------
kmesh=311

#------------- gate ----------------
gate=0.00
#gate_list=(`seq 0 0.001 0.2`)
#gate=${gate_list[${SLURM_ARRAY_TASK_ID}]}
#
#idx_last=`bc <<< "${SLURM_ARRAY_TASK_ID}-1"`
#gate_last=${gate_list[${idx_last}]}
#
#echo 'idx_last = ' ${idx_last}
#echo 'gate_last = ' ${gate_last}

#-------------- labels ----------------
labels=${cell}_k${kmesh}_${method}_gate${gate}


#-------------- job control ----------------
plot_lo=True
mf_load_fname=

script=set_ham_contact_${labels}.py
output=set_ham_contact_${labels}.out

#-------------- data directory ----------------
datadir=${dir}/${imp_atom}/
mkdir -p ${datadir}

#-------------- summary ----------------
echo 'labels = ' ${labels}
echo 'plot_lo = ' ${plot_lo}
echo 'mf_load_fname = ' ${mf_load_fname}
echo 'datadir = ' ${datadir}

#-------------- generate script ----------------
sed -e"s/MODE/production/" \
    -e"s/IMP_ATOM/${imp_atom}/" \
    -e"s/XCFUN/${xcfun}/" \
    -e"s/LEFT/${left}/" \
    -e"s/RIGHT/${right}/" \
    -e"s/LATCONST/${latconst}/" \
    -e"s/GATE/${gate}/" \
    -e"s/KMESH/${kmesh}/" \
    -e"s/PLOT_LO/${plot_lo}/" \
    -e"s:MF_LOAD_FNAME:${mf_load_fname}:" \
    -e"s:DATADIR:${datadir}:" \
    set_ham_contact.py > ${work}/${script}

#-------------- run ----------------
cd ${work}
python -u ${script} > ${output} 2>&1
mv ${script} ${output} ${datadir}


