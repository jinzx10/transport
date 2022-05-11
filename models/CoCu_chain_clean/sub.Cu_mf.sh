#!/bin/bash

#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --job-name=Cu_mf
#SBATCH --exclude=pauling001

source $HOME/.bashrc
conda activate

export MKL_NUM_THREADS=28
export OMP_NUM_THREADS=28

dir=$HOME/projects/transport/models/CoCu_chain_clean
datadir=${dir}/Cu_def2-svp-bracket/

cd ${dir}
mkdir -p ${datadir}

#-----------------------------------
use_pbe=False
nat=16
spacing=2.55

if [[ ${use_pbe} == "True" ]]; then
    method=rks
else
    method=rhf
fi

suffix=nat${nat}_a${spacing}_${method}

script=Cu_mf_${suffix}.py
output=Cu_mf_${suffix}.out

sed -e"s/USE_PBE/${use_pbe}/" \
    -e"s/NAT/${nat}/" \
    -e"s/SPACING/${spacing}/" \
    Cu_mf.py > ${script}

python ${script} --datadir=${datadir} > ${output}
mv ${script} ${output} ${datadir}
