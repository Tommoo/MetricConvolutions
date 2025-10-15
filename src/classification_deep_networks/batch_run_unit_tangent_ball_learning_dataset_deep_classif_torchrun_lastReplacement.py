import subprocess
# import argparse
import pathlib
import os
import itertools


# parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', type=int)
# parser.add_argument('--gpus', type=int)
# args = parser.parse_args()

nb_runs = 1
dataset_name_list = ['CIFAR100']                    # ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
ker_fixed_list = ['False']                          # ['True', 'False']
k_list = ['None']                                   # ['None'] can also be a number e.g. 5, 11
strat_w_list = ['sigmoid_norm_detach']              # ['sigmoid_norm', 'sigmoid_norm_detach', 'norm_clip_detach']
eps_w_list = [0.5]                                  # [1.0]
eps_L_list = [0.01]                                 # [0.01]
eps_list = [1e-6]                                   # [1e-6]
scale_min_list = [0.1]                              # [0.1, 0.01]
scale_max_list = [1.5]                              # [1.5, 1.9]
intermediate_strat_list = ['conv_eigvec_eigval_sigmoid_lambda_sep_scale']#['conv_eigvec_eigval_sigmoid_lambda_sep_scale']  # for ker_fixed, sep_scale was unstable
                                                    # ['vanilla_conv', 'vanilla_conv_LDLT', 'conv_eigvec_eigval',
                                                    #  'conv_eigvec_eigval_sigmoid_lambda_sep_scale',
                                                    #  'conv_eigvec_eigval_sigmoid_lambda_sep_scale_det_ratio']
sampling_strat_list = ['onion_peeling_grid']        # ['polar_grid', 'onion_peeling_grid']
lambda_reg_Mw_list = [5000.]                     # [0.0]
train_list = ['True']                               # ['True']
ignore_checkpoint_list = ['False']                  # ['False', 'True']
augment_list = ['True']                            # ['False', 'True']
model_name_list = ['ResNet18']                      # ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
pretrained_list = ['True']                          # ['True', 'False'])
no_pooling_or_stride_conv1_list = ['True']          # ['True', 'False']
batch_size_list = [128]                             # [128]
lr_list = [0.0001]                                  # [0.1, 0.001, 0.0001]  # 0.1 is for SGD, 0.001 is for Adam
epochs_list = [240]                                 # [120]
optimizer_list = ['Adam']                           # ['SGD', 'Adam']
lr_scheduler_list = ['CosineAnnealingLR']           # ['StepLR', 'MultiStepLR', 'CosineAnnealingLR']  # StepLR is for SGD, CosineAnnealingLR is for Adam
step_size_lr_scheduler_list = [30]                  # [30] # For StepLR
multistep_lr_scheduler_list = ['60-120-160']        # ['60-120-160'] # For MultiStepLR  | Note: Dash, no spacing
gamma_lr_scheduler_list = [0.2]                     # [0.1] # 0.1 For StepLR, 0.2 For MultiStepLR
T_max_list = [240]                                  # [240] # For CosineAnnealingLR


if True in ignore_checkpoint_list:
    import warnings
    warnings.warn('ignore_checkpoint_list contains True. This will overwrite previous checkpoints.')


processes = []
for dataset_name, \
    ker_fixed, \
    k, \
    strat_w, \
    eps_w, \
    eps_L, \
    eps, \
    scale_min, \
    scale_max, \
    intermediate_strat, \
    sampling_strat, \
    lambda_reg_Mw, \
    train, \
    ignore_checkpoint, \
    augment, \
    model_name, \
    pretrained, \
    no_pooling_or_stride_conv1, \
    batch_size, \
    lr, \
    epochs, \
    optimizer, \
    lr_scheduler, \
    step_size_lr_scheduler, \
    multistep_lr_scheduler, \
    gamma_lr_scheduler, \
    T_max, \
    run_number \
        in itertools.product(
            dataset_name_list,
            ker_fixed_list,
            k_list,
            strat_w_list,
            eps_w_list,
            eps_L_list,
            eps_list,
            scale_min_list,
            scale_max_list,
            intermediate_strat_list,
            sampling_strat_list,
            lambda_reg_Mw_list,
            train_list,
            ignore_checkpoint_list,
            augment_list,
            model_name_list,
            pretrained_list,
            no_pooling_or_stride_conv1_list,
            batch_size_list,
            lr_list,
            epochs_list,
            optimizer_list,
            lr_scheduler_list,
            step_size_lr_scheduler_list,
            multistep_lr_scheduler_list,
            gamma_lr_scheduler_list,
            T_max_list,
            range(nb_runs)
        ):

    # TODO: Remove this
    if dataset_name in ['MNIST', 'FashionMNIST'] and augment == 'True':
        continue
    if dataset_name in ['CIFAR10', 'CIFAR100'] and augment == 'False':
        continue

    # Filter out some combinations
    if optimizer == 'SGD':
        if not lr == 0.1:
            continue
        if not lr_scheduler == 'StepLR' and not lr_scheduler == 'MultiStepLR':
            continue
    if optimizer == 'Adam':
        if lr not in [0.001, 0.0001]:
            continue
        if not lr_scheduler == 'CosineAnnealingLR':
            continue
    if eps_w == 1.0 and len(strat_w_list) > 1:
        if not strat_w == strat_w_list[0]:
            continue

    print(
        'dataset_name', dataset_name,
        'ker_fixed', ker_fixed,
        'k', k,
        'strat_w', strat_w,
        'eps_w', eps_w,
        'eps_L', eps_L,
        'eps', eps,
        'scale_min', scale_min,
        'scale_max', scale_max,
        'intermediate_strat', intermediate_strat,
        'sampling_strat', sampling_strat,
        'lambda_reg_Mw', lambda_reg_Mw,
        'train', train,
        'ignore_checkpoint', ignore_checkpoint,
        'augment', augment,
        'model_name', model_name,
        'pretrained', pretrained,
        'no_pooling_or_stride_conv1', no_pooling_or_stride_conv1,
        'batch_size', batch_size,
        'lr', lr,
        'epochs', epochs,
        'optimizer', optimizer,
        'lr_scheduler', lr_scheduler,
        'step_size_lr_scheduler', step_size_lr_scheduler,
        'multistep_lr_scheduler', multistep_lr_scheduler,
        'gamma_lr_scheduler', gamma_lr_scheduler,
        'T_max', T_max,
        'run_number', run_number
    )

    str_pretrained = '_pretrained' if pretrained == 'True' else ''
    str_no_pooling_or_stride_conv1 = '_noPoolStrideConv1' if no_pooling_or_stride_conv1 == 'True' else ''
    if lr_scheduler == 'StepLR':
        str_scheduler_args = 'step_size_lr_scheduler_' + str(step_size_lr_scheduler) + '_gamma_lr_scheduler_' + str(gamma_lr_scheduler)
    elif lr_scheduler == 'MultiStepLR':
        str_scheduler_args = 'multistep_lr_scheduler_' + multistep_lr_scheduler + '_gamma_lr_scheduler_' + str(gamma_lr_scheduler)
    else:
        str_scheduler_args = 'T_max_' + str(T_max)
    str_strat_w = '_strat_w_' + strat_w if not eps_w == 1.0 else ''
    str_scales = '_scales_min_max_' + str(scale_min) + '_' + str(scale_max)  \
        if intermediate_strat in ['conv_eigvec_eigval_sigmoid_lambda_sep_scale',
                                  'conv_eigvec_eigval_sigmoid_lambda_sep_scale_det_ratio'] else ''
    str_lambda_reg = '_lambda_reg_Mw_' + str(lambda_reg_Mw) if lambda_reg_Mw > 0 else ''

    dir_log = os.path.join(
        'logs',
        'classification',
        dataset_name,
        'unit_tangent_ball_ker_fixed' if ker_fixed == 'True' else 'unit_tangent_ball',
        model_name + str_pretrained + str_no_pooling_or_stride_conv1 + '_lastReplacement',
        'augment_' + str(augment),
        'intermediate_strat_'+intermediate_strat + str_scales,
        'sampling_strat_'+sampling_strat,
        'k_'+str(k)+'__bs_'+str(batch_size),
        'eps_' + str(eps) + '_epsL_' + str(eps_L) + '_epsw_' + str(eps_w) + str_strat_w,
        'lr_' + str(lr) + '_optimizer_' + optimizer + '_lr_scheduler_' + lr_scheduler + str_lambda_reg,
        str_scheduler_args,
    )
    if run_number > 0:
        dir_log = os.path.join(dir_log, 'run_' + str(run_number))
    if not pathlib.Path(dir_log).exists():
        pathlib.Path(dir_log).mkdir(parents=True, exist_ok=True)

    f = open(
        f"{dir_log}/output.txt",
        # "w"  # w overwrites previous file
        "a+"  # a appends to previous file  # a+ also allows reading
    )

    f_partition_args = ''

    print(dir_log)

    # print(
    #     f' torchrun --standalone --nproc_per_node=gpu unit_tangent_ball_learning_dataset_deep_classif_torchrun_lastReplacement.py'
    #     f' --dataset_name {dataset_name}'
    #     f' --ker_fixed {ker_fixed}'
    #     f' --k {k}'
    #     f' --strat_w {strat_w}'
    #     f' --eps_w {eps_w}'
    #     f' --eps_L {eps_L}'
    #     f' --eps {eps}'
    #     f' --scale_min {scale_min}'
    #     f' --scale_max {scale_max}'
    #     f' --intermediate_strat {intermediate_strat}'
    #     f' --sampling_strat {sampling_strat}'
    #     f' --lambda_reg_Mw {lambda_reg_Mw}'
    #     f' --train {train}'
    #     f' --ignore_checkpoint {ignore_checkpoint}'
    #     f' --augment {augment}'
    #     f' --model_name {model_name}'
    #     f' --pretrained {pretrained}'
    #     f' --no_pooling_or_stride_conv1 {no_pooling_or_stride_conv1}'
    #     f' --batch_size {batch_size}'
    #     f' --lr {lr}'
    #     f' --epochs {epochs}'
    #     f' --optimizer {optimizer}'
    #     f' --lr_scheduler {lr_scheduler}'
    #     f' --step_size_lr_scheduler {step_size_lr_scheduler}'
    #     f' --multistep_lr_scheduler {multistep_lr_scheduler}'
    #     f' --gamma_lr_scheduler {gamma_lr_scheduler}'
    #     f' --T_max {T_max}'
    #     f' --run_number {run_number}'
    # )

    p = subprocess.Popen(
        # f'srun -c {args.cpus} --gres=gpu:{args.gpus}'
        f'srun -c 32 --gres=gpu:1'
        f' -v'  # verbose
        f' -u'  # For debugging  # Right now some jobs are stuck on startup without this, unknown bug
        f'{f_partition_args}'
        f' torchrun --standalone --nproc_per_node=gpu unit_tangent_ball_learning_dataset_deep_classif_torchrun_lastReplacement.py'
        f' --dataset_name {dataset_name}'
        f' --ker_fixed {ker_fixed}'
        f' --k {k}'
        f' --strat_w {strat_w}'
        f' --eps_w {eps_w}'
        f' --eps_L {eps_L}'
        f' --eps {eps}'
        f' --scale_min {scale_min}'
        f' --scale_max {scale_max}'
        f' --intermediate_strat {intermediate_strat}'
        f' --sampling_strat {sampling_strat}'
        f' --lambda_reg_Mw {lambda_reg_Mw}'
        f' --train {train}'
        f' --ignore_checkpoint {ignore_checkpoint}'
        f' --augment {augment}'
        f' --model_name {model_name}'
        f' --pretrained {pretrained}'
        f' --no_pooling_or_stride_conv1 {no_pooling_or_stride_conv1}'
        f' --batch_size {batch_size}'
        f' --lr {lr}'
        f' --epochs {epochs}'
        f' --optimizer {optimizer}'
        f' --lr_scheduler {lr_scheduler}'
        f' --step_size_lr_scheduler {step_size_lr_scheduler}'
        f' --multistep_lr_scheduler {multistep_lr_scheduler}'
        f' --gamma_lr_scheduler {gamma_lr_scheduler}'
        f' --T_max {T_max}'
        f' --run_number {run_number}',
        shell=True,
        stdout=f,
        stderr=f
    )
    processes.append(p)

for p in processes:
    p.wait()
