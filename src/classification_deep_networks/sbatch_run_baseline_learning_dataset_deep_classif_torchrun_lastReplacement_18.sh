#!/bin/bash
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --array=3-9                       # <-- run k jobs with IDs low, low+1, low+2,..., high (e.g. 1-3 will run jobs with IDs 1, 2, and 3)
#SBATCH --mail-type=ALL           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --open-mode=append
#SBATCH -o ./logs/classification/CIFAR100/baseline/ResNet152_pretrained_noPoolStrideConv1_lastReplacement/augment_True/bs_128/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR/T_max_240/run_%a/output.txt        # stdout goes to output.txt
#SBATCH -e ./logs/classification/CIFAR100/baseline/ResNet152_pretrained_noPoolStrideConv1_lastReplacement/augment_True/bs_128/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR/T_max_240/run_%a/output.txt        # stderr goes to err_job.txt

RUN_NUMBER=${SLURM_ARRAY_TASK_ID}

torchrun --standalone --nproc_per_node=gpu baseline_learning_dataset_deep_classif_torchrun_lastReplacement.py --dataset_name CIFAR100 --train True --ignore_checkpoint False --augment True --model_name ResNet152 --pretrained True --no_pooling_or_stride_conv1 True --no_change_deep_layers False --batch_size 128 --lr 0.0001 --epochs 240 --optimizer Adam --lr_scheduler CosineAnnealingLR --step_size_lr_scheduler 30 --multistep_lr_scheduler 60-120-160 --gamma_lr_scheduler 0.2 --T_max 240 --run_number ${RUN_NUMBER}