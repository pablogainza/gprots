#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=40:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000
#SBATCH --array=1-47

echo STARTING AT `date`
echo ${SLURM_ARRAY_TASK_ID}
echo FINISHED at `date`

source /work/upcorreia/bin/load_masif_environment_gpu.sh
i=1
while read p; do
    if [ $(( i % 1001 )) == ${SLURM_ARRAY_TASK_ID} ]; then
#        python3 -W ignore train_classifier_activation_3cat.py $p gnaoAc
#        python3 -W ignore train_classifier_activation_3cat.py $p gnaqAc
        python3 -W ignore train_classifier_activation_3cat.py $p gna15Ac
#       python3 -W ignore train_classifier_activation_3cat.py $p gnas2Ac
#python3 -W ignore train_classifier_activation_3cat.py $p gnas13Ac
    fi
    i=$((i+1))
done < lists/gpcr_list.txt


