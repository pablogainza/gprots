#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=40:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000
#SBATCH --array=40-49

echo STARTING AT `date`
echo ${SLURM_ARRAY_TASK_ID}

source /work/upcorreia/bin/load_masif_environment_gpu.sh
# use_DNN = False, pretrain=False, human_only=False
#python3 -W ignore train_classifier.py gnaoAmp ${SLURM_ARRAY_TASK_ID} 0 0 0
#python3 -W ignore train_classifier.py gnaqAmp ${SLURM_ARRAY_TASK_ID} 0 0 0
#python3 -W ignore train_classifier.py gna15Amp ${SLURM_ARRAY_TASK_ID} 0 0 0
#python3 -W ignore train_classifier.py gnas2Amp ${SLURM_ARRAY_TASK_ID} 0 0 0
#python3 -W ignore train_classifier.py gnas13Amp ${SLURM_ARRAY_TASK_ID} 0 0 0

# use_DNN = True, pretrain=False, human_only=False
#python3 -W ignore train_classifier.py gnaoAmp ${SLURM_ARRAY_TASK_ID} 1 0 0
#python3 -W ignore train_classifier.py gnaqAmp ${SLURM_ARRAY_TASK_ID} 1 0 0
#python3 -W ignore train_classifier.py gna15Amp ${SLURM_ARRAY_TASK_ID} 1 0 0
#python3 -W ignore train_classifier.py gnas2Amp ${SLURM_ARRAY_TASK_ID} 1 0 0
#python3 -W ignore train_classifier.py gnas13Amp ${SLURM_ARRAY_TASK_ID} 1 0 0

# use_DNN = True, pretrain=True, human_only=True
#python3 -W ignore train_classifier.py gnaoAmp ${SLURM_ARRAY_TASK_ID} 1 1 1
#python3 -W ignore train_classifier.py gnaqAmp ${SLURM_ARRAY_TASK_ID} 1 1 1
#python3 -W ignore train_classifier.py gna15Amp ${SLURM_ARRAY_TASK_ID} 1 1 1
#python3 -W ignore train_classifier.py gnas2Amp ${SLURM_ARRAY_TASK_ID} 1 1 1
#python3 -W ignore train_classifier.py gnas13Amp ${SLURM_ARRAY_TASK_ID} 1 1 1

# use_DNN = True, pretrain=True, human_only=False
python3 -W ignore train_classifier_uncertainty.py gnaoAmp ${SLURM_ARRAY_TASK_ID} 1 1 0
python3 -W ignore train_classifier_uncertainty.py gnaqAmp ${SLURM_ARRAY_TASK_ID} 1 1 0
python3 -W ignore train_classifier_uncertainty.py gna15Amp ${SLURM_ARRAY_TASK_ID} 1 1 0
python3 -W ignore train_classifier_uncertainty.py gnas2Amp ${SLURM_ARRAY_TASK_ID} 1 1 0
python3 -W ignore train_classifier_uncertainty.py gnas13Amp ${SLURM_ARRAY_TASK_ID} 1 1 0

echo FINISHED at `date`
