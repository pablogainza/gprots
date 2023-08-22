#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 16000
#SBATCH --array=1

echo STARTING AT `date`
echo ${SLURM_ARRAY_TASK_ID}

source ~/lpdi_fs/masif/pytorch_gpu_gcc54/bin/activate
./01-make_all_embeddings.sh
# use_DNN = False, pretrain=False, human_only=False
