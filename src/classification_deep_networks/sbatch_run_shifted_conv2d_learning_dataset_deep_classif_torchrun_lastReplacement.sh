#!/bin/bash
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --open-mode=append
#SBATCH -o ./logs/classification/CIFAR100/shifted_conv2d/ResNet152_pretrained_noPoolStrideConv1_lastReplacement/augment_True/k_None__bs_128/eps_1e-06/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR/T_max_240/output.txt        # stdout goes to output.txt
#SBATCH -e ./logs/classification/CIFAR100/shifted_conv2d/ResNet152_pretrained_noPoolStrideConv1_lastReplacement/augment_True/k_None__bs_128/eps_1e-06/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR/T_max_240/output.txt         # stderr goes to err_job.txt

torchrun --standalone --nproc_per_node=gpu shifted_conv2d_learning_dataset_deep_classif_torchrun_lastReplacement.py --dataset_name CIFAR100 --ker_fixed False --k None --eps 1e-06 --train True --ignore_checkpoint False --augment True --model_name ResNet152 --pretrained True --no_pooling_or_stride_conv1 True --batch_size 128 --lr 0.0001 --epochs 240 --optimizer Adam --lr_scheduler CosineAnnealingLR --step_size_lr_scheduler 30 --multistep_lr_scheduler 60-120-160 --gamma_lr_scheduler 0.1 --T_max 240 --run_number 0