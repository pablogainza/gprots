#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=40:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000
#SBATCH --array=1-20

echo STARTING AT `date`
echo ${SLURM_ARRAY_TASK_ID}

source /work/upcorreia/bin/load_masif_environment_gpu.sh
python3 -W ignore train_classifier_activation.py gnaoAct ${SLURM_ARRAY_TASK_ID}
python3 -W ignore train_classifier_activation.py gnaqAct ${SLURM_ARRAY_TASK_ID}
python3 -W ignore train_classifier_activation.py gna15Act ${SLURM_ARRAY_TASK_ID}
python3 -W ignore train_classifier_activation.py gnas2Act ${SLURM_ARRAY_TASK_ID}
python3 -W ignore train_classifier_activation.py gnas13Act ${SLURM_ARRAY_TASK_ID}

echo FINISHED at `date`
