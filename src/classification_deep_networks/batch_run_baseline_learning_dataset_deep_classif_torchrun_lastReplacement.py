import subprocess
# import argparse
import pathlib
import os
import itertools


# parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', type=int)
# parser.add_argument('--gpus', type=int)
# args = parser.parse_args()

nb_runs = 1                                         # 1 is the default
dataset_name_list = ['CIFAR100']                    # ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
train_list = ['True']                               # ['True']
ignore_checkpoint_list = ['False']                  # ['False', 'True']
augment_list = ['True']                             # ['False', 'True']
model_name_list = ['ResNet18']                      # ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
pretrained_list = ['True']                          # ['True', 'False'])
no_pooling_or_stride_conv1_list = ['True']         # ['True', 'False']
no_change_deep_layers_list = ['False']              # ['True', 'False']
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
    train, \
    ignore_checkpoint, \
    augment, \
    model_name, \
    pretrained, \
    no_pooling_or_stride_conv1, \
    no_change_deep_layers, \
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
            train_list,
            ignore_checkpoint_list,
            augment_list,
            model_name_list,
            pretrained_list,
            no_pooling_or_stride_conv1_list,
            no_change_deep_layers_list,
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

    print(
        'dataset_name', dataset_name,
        'train', train,
        'ignore_checkpoint', ignore_checkpoint,
        'augment', augment,
        'model_name', model_name,
        'pretrained', pretrained,
        'no_pooling_or_stride_conv1', no_pooling_or_stride_conv1,
        'no_change_deep_layers', no_change_deep_layers,
        'batch_size', batch_size,
        'lr', lr,
        'epochs', epochs,
        'optimizer', optimizer,
        'lr_scheduler', lr_scheduler,
        'step_size_lr_scheduler', step_size_lr_scheduler,
        'multistep_lr_scheduler', multistep_lr_scheduler,
        'gamma_lr_scheduler', gamma_lr_scheduler,
        'T_max', T_max,
        'run_', run_number
    )

    str_pretrained = '_pretrained' if pretrained == 'True' else ''
    str_no_pooling_or_stride_conv1 = '_noPoolStrideConv1' if no_pooling_or_stride_conv1 == 'True' else ''
    str_no_change_deep_layers = '_noChangeDeepLayers' if no_change_deep_layers == 'True' else ''
    if lr_scheduler == 'StepLR':
        str_scheduler_args = 'step_size_lr_scheduler_' + str(step_size_lr_scheduler) + '_gamma_lr_scheduler_' + str(gamma_lr_scheduler)
    elif lr_scheduler == 'MultiStepLR':
        str_scheduler_args = 'multistep_lr_scheduler_' + multistep_lr_scheduler + '_gamma_lr_scheduler_' + str(gamma_lr_scheduler)
    else:
        str_scheduler_args = 'T_max_' + str(T_max)
    dir_log = os.path.join(
        'logs',
        'classification',
        dataset_name,
        'baseline',
        model_name + str_pretrained + str_no_pooling_or_stride_conv1 + str_no_change_deep_layers + '_lastReplacement',
        'augment_' + str(augment),
        'bs_' + str(batch_size),
        'lr_' + str(lr) + '_optimizer_' + optimizer + '_lr_scheduler_' + lr_scheduler,
        str_scheduler_args,
    )
    if run_number > 0:
        dir_log = os.path.join(dir_log, 'run_' + str(run_number))
    if not pathlib.Path(dir_log).exists():
        pathlib.Path(dir_log).mkdir(parents=True, exist_ok=True)

    f = open(
        f"{dir_log}/output.txt",
        #"w"  # w overwrites previous file
        "a+"  # a appends to previous file  # a+ also allows reading
    )

    f_partition_args = ''

    print(dir_log)

    # print(
    #     f' torchrun'
    #     f' --standalone --nproc_per_node=gpu baseline_learning_dataset_deep_classif_torchrun_lastReplacement.py'
    #     f' --dataset_name {dataset_name}'
    #     f' --train {train}'
    #     f' --ignore_checkpoint {ignore_checkpoint}'
    #     f' --augment {augment}'
    #     f' --model_name {model_name}'
    #     f' --pretrained {pretrained}'
    #     f' --no_pooling_or_stride_conv1 {no_pooling_or_stride_conv1}'
    #     f' --no_change_deep_layers {no_change_deep_layers}'
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
        f'srun -c 32 --gres=gpu:1'  # --gres=gpu:8 -w plato2'
        f' -v'  # verbose
        f' -u'  # For debugging  # Right now some jobs are stuck on startup without this, unknown bug
        f'{f_partition_args}'
        # f' python -m torch.distributed.run'
        f' torchrun'
        f' --standalone --nproc_per_node=gpu baseline_learning_dataset_deep_classif_torchrun_lastReplacement.py'
        f' --dataset_name {dataset_name}'
        f' --train {train}'
        f' --ignore_checkpoint {ignore_checkpoint}'
        f' --augment {augment}'
        f' --model_name {model_name}'
        f' --pretrained {pretrained}'
        f' --no_pooling_or_stride_conv1 {no_pooling_or_stride_conv1}'
        f' --no_change_deep_layers {no_change_deep_layers}'
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
