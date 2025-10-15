#!/bin/bash
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --array=7-9                       # <-- run k jobs with IDs low, low+1, low+2,..., high (e.g. 1-3 will run jobs with IDs 1, 2, and 3)
#SBATCH --mail-type=ALL           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --open-mode=append
#SBATCH -o ./logs/classification/CIFAR100/unit_tangent_ball/ResNet152_pretrained_noPoolStrideConv1_lastReplacement/augment_True/intermediate_strat_conv_eigvec_eigval_sigmoid_lambda_sep_scale_scales_min_max_0.1_1.5/sampling_strat_onion_peeling_grid/k_None__bs_128/eps_1e-06_epsL_0.01_epsw_1.0/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR_lambda_reg_Mw_5000.0/T_max_240/run_%a/output.txt        # stdout goes to output.txt
#SBATCH -e ./logs/classification/CIFAR100/unit_tangent_ball/ResNet152_pretrained_noPoolStrideConv1_lastReplacement/augment_True/intermediate_strat_conv_eigvec_eigval_sigmoid_lambda_sep_scale_scales_min_max_0.1_1.5/sampling_strat_onion_peeling_grid/k_None__bs_128/eps_1e-06_epsL_0.01_epsw_1.0/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR_lambda_reg_Mw_5000.0/T_max_240/run_%a/output.txt         # stderr goes to err_job.txt

RUN_NUMBER=${SLURM_ARRAY_TASK_ID}

torchrun --standalone --nproc_per_node=gpu unit_tangent_ball_learning_dataset_deep_classif_torchrun_lastReplacement.py --dataset_name CIFAR100 --ker_fixed False --k None --strat_w sigmoid_norm_detach --eps_w 1.0 --eps_L 0.01 --eps 1e-06 --scale_min 0.1 --scale_max 1.5 --intermediate_strat conv_eigvec_eigval_sigmoid_lambda_sep_scale --sampling_strat onion_peeling_grid --lambda_reg_Mw 5000.0 --train True --ignore_checkpoint False --augment True --model_name ResNet152 --pretrained True --no_pooling_or_stride_conv1 True --batch_size 128 --lr 0.0001 --epochs 240 --optimizer Adam --lr_scheduler CosineAnnealingLR --step_size_lr_scheduler 30 --multistep_lr_scheduler 60-120-160 --gamma_lr_scheduler 0.2 --T_max 240 --run_number ${RUN_NUMBER}