#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=120:00:00   # walltime
#SBATCH --ntasks=32   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=5800M   # memory per CPU core
#SBATCH --mail-user=zuxinjin@caltech.edu   # email address

source ${HOME}/.bashrc
conda activate

timestamp=`date +%y%m%d-%H%M%S`


dir=${HOME}/projects/transport/models/cu_chain/cu_au


cd ${SCRATCH}
WORK=${SCRATCH}/cu_au_${timestamp}


mkdir -p ${WORK}
cd ${WORK}
cp ${dir}/hf.py ${WORK}

savedir="data-${timestamp}"
output="cu_au_${timestamp}.out"

echo `date`
echo "savedir=${savedir}"

mkdir -p ${savedir}
python ${dir}/hf.py --savedir=${savedir} --smearing=0.05 > ${output}

