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
python3 -W ignore train_classifier.py $p gnaoAmp
python3 -W ignore train_classifier.py $p gnaqAmp
python3 -W ignore train_classifier.py $p gna15Amp
python3 -W ignore train_classifier.py $p gnas2Amp
python3 -W ignore train_classifier.py $p gnas13Amp

