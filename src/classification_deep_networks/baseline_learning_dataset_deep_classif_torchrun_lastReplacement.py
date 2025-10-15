
# Migrate code to torchrun and ddp multi-gpu single node training

import torch
import torchvision
import os
import pathlib
import warnings
import argparse


from utils_torchrun import (adapt_model_classifier,
                            none_or_int, none_or_float, none_or_str, import_model_architecture, prepare_dataset,
                            get_dataset_params, train_cnn_classif_several_epochs)


def ddp_setup():
    torch.distributed.init_process_group(backend='nccl')


def convert_model_baseline(model_name, model, no_pooling_or_stride_conv1=False, no_change_deep_layers=False):
    # For imagenet, no_change_deep_layers=True reinitialises modifies layers, so results will be different from
    # pretrained on Imagenet

    device = next(model.parameters()).device

    # Change of stride different in Dilated Resnet and deform conv paper (although not the first).

    # If we only do layer4 (resnet) like in deformable convolution, then for small image e.g. 32x32, the input of layer4
    # is 2x2, which is too small for our method to be useful. In deformable conv, they only change layer4, which is ok
    # because the pascalVOC images are resized to min 360, giving the features of the conv in layer4 to be of size
    # 23x23, which is ok.
    # Instead, we do a hybrid with More Deformable Convolution, where we change layer2, layer3 and layer4. But we
    # do not add the mask modulation concept from More Deformable Convolution.
    # We also do not change the stride of the first conv of layer2 and layer3, perhaps like in More Deformable
    # Convolution as it is not mentioned. In dilated resnet, a change in an earlier layer impacts the dilation of later
    # layers, not the case here. Also, in dilated resnet, they change from layer3 only.

    # TODO: VGG? See, update, and fix icml version

    if model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:

        if no_pooling_or_stride_conv1:
            # For small init resolution, the 7x7 initial conv stride 2 followed by maxpool looses too much information
            # We change the first conv to 3x3 stride 1 (padding 1) and remove the maxpool
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #     model.conv1 = Conv2dTangentBall(3, 64, 3, stride=1, dilation=1,
        #                     ker_fixed=ker_fixed, bias=False,
        #                     eps_L=eps_L, strat_w=strat_w, eps_w=eps_w, eps=eps,
        #                     scale_max=scale_max, scale_min=scale_min,
        #                     sampling_strat=sampling_strat, intermediate_strat=intermediate_strat,
        #                     ker_init=None,
        #                     device=device
        #                 )
            model.maxpool = torch.nn.Identity()
        # else:
        #     # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #     model.conv1 = Conv2dTangentBall(
        #         3, 64, 7, stride=2, dilation=1,
        #         ker_fixed=ker_fixed, bias=False,
        #         eps_L=eps_L, strat_w=strat_w, eps_w=eps_w, eps=eps,
        #         scale_max=scale_max, scale_min=scale_min,
        #         sampling_strat=sampling_strat, intermediate_strat=intermediate_strat,
        #         ker_init=model.conv1.weight,
        #         device=device
        #     )

        if no_change_deep_layers:
            return model

        # layers_changed = [model.layer1, model.layer2, model.layer3, model.layer4]
        layers_changed = [model.layer4]

        if model_name in ['ResNet18', 'ResNet34']:
            # Could make it user defined by a list of to_change =  [False, True, True, True], and then loop in to_change
            # if to_change[layer_idx]: then change layer else pass
            for layer_idx, layer in enumerate(layers_changed):
                for basicblock_idx, basicblock in enumerate(layer):
                    # 3x3 conv is given by basicblock.conv1 and basicblock.conv2
                    conv1 = basicblock.conv1
                    conv2 = basicblock.conv2

                    for i in range(2):
                        conv = conv1 if i == 0 else conv2
                        k = conv.kernel_size[0]
                        stride = 1 if (layer_idx == len(layers_changed) - 1 and basicblock_idx == 0) else conv.stride[0]
                        dilation = 2 if (layer_idx == len(layers_changed) - 1 and basicblock_idx > 0) else conv.dilation[0]
                        bias = conv.bias is not None  # No bias in default resnet

                        # Downsampling happens at layers conv3_1, conv4_1, conv5_1 in original resnet paper (conv2 = layer1)
                        if (layer_idx == len(layers_changed) - 1 and basicblock_idx == 0 and i == 0):
                            basicblock.downsample[0] = \
                                torch.nn.Conv2d(
                                    conv1.in_channels, conv2.out_channels,
                                    kernel_size=basicblock.downsample[0].kernel_size,  # 1x1 in default resnet
                                    stride=1,  # Changed from 2 to 1. No downsampling is done in our method (keep the 1x1)
                                    bias=basicblock.downsample[0].bias  # False in default resnet
                                )
                                # In original resnet paper, downsampling on cifar is identity (Option A) rather than 1x1
                                # using 0-padding. They also don't use resnet18 by more custom architectures (and prefer
                                # a choice closer to resnet101).

                        basicblock_conv = torch.nn.Conv2d(
                            conv.in_channels, conv.out_channels,
                            kernel_size=k, stride=stride, dilation=dilation,
                            bias=bias, padding=conv.padding[0]*dilation,
                            device=device
                        )
                        if i == 0:
                            basicblock.conv1 = basicblock_conv
                        else:
                            basicblock.conv2 = basicblock_conv

        else:  # ResNet50, ResNet101, ResNet152
            for layer_idx, layer in enumerate(layers_changed):
                for bottleneck_idx, bottleneck in enumerate(layer):
                    # 3x3 conv is given by bottleneck.conv2
                    conv2 = bottleneck.conv2
                    k = conv2.kernel_size[0]
                    stride = 1 if (layer_idx == len(layers_changed)-1 and bottleneck_idx == 0) else conv2.stride[0]
                    dilation = 2 if (layer_idx == len(layers_changed)-1 and bottleneck_idx > 0) else conv2.dilation[0]
                    bias = conv2.bias is not None  # No bias in default resnet

                    # Downsampling happens at layers conv3_1, conv4_1, conv5_1 in original resnet paper (conv2 = layer1)
                    if (layer_idx == len(layers_changed) - 1 and bottleneck_idx == 0):
                        bottleneck.downsample[0] = \
                            torch.nn.Conv2d(
                                bottleneck.conv1.in_channels, bottleneck.conv3.out_channels,
                                kernel_size=bottleneck.downsample[0].kernel_size,  # 1x1 in default resnet
                                stride=1,  # Changed from 2 to 1. No downsampling is done in our method (keep the 1x1)
                                bias=bottleneck.downsample[0].bias  # False in default resnet
                            )

                    bottleneck.conv2 = torch.nn.Conv2d(
                        conv2.in_channels, conv2.out_channels,
                        kernel_size=k, stride=stride, dilation=dilation,
                        bias=bias, padding=conv2.padding[0]*dilation,
                        device=device
                    )

    else:
        raise ValueError('conversion not implemented for model_name ' + model_name)
    return model


def main_baseline():

    # torch.autograd.set_detect_anomaly(True)  # Debugging
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    #
    # # use CUDA_LAUNCH_BLOCKING=1 to debug CUDA code
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    computer = 'newton'  # 'local' or 'newton'

    if computer == 'local':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if dev == 'cpu':
            warnings.warn('No GPU available. Using CPU')
            dataset_root = '/datasets'
        else:
            dataset_root = '/datasets'

        args = argparse.Namespace()
        args.run_number = 0                         # 0 is the default
        args.dataset_name = 'CIFAR10'               # 'MNIST', 'FashionMNIST', 'CIFAR10', 'ImageNet', (maybe: 'STL10')
        args.train = True                           # True, False
        args.ignore_checkpoint = False              # True, False
        args.augment = False                        # True, False
        args.model_name = 'ResNet18'                # 'ResNet18', 'ResNet50', 'ResNet152'
        args.pretrained = True                      # True, False  # TODO: maybe train vanilla resnet on dataset with fixes (classif + small dataset) for initialisations?
        args.no_pooling_or_stride_conv1 = True      # False, True
        args.no_change_deep_layers = False          # False, True
        args.batch_size = 8                       # 128
        args.epochs = 120                           # 120
        args.lr = 0.0001                            # 0.1
        args.optimizer = 'Adam'                     # 'SGD', 'Adam'
        args.lr_scheduler = 'CosineAnnealingLR'     # 'StepLR', 'CosineAnnealingLR', 'MultiStepLR'
        args.step_size_lr_scheduler = 30            # 30, None
        args.multistep_lr_scheduler = '60,120,160'  # '60,120,160', None
        args.gamma_lr_scheduler = 0.1               # 0.1, None
        args.T_max = 240                            # 240, None

        # TODO: warmup?

    elif computer == 'newton':
        dev = 'cuda'

        parser = argparse.ArgumentParser()
        parser.add_argument('--run_number', type=int)
        parser.add_argument('--dataset_name', type=str, choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'])
        parser.add_argument('--train', type=str, choices=['True', 'False'])
        parser.add_argument('--ignore_checkpoint', type=str, choices=['True', 'False'])
        parser.add_argument('--augment', type=str, choices=['True', 'False'])
        parser.add_argument('--model_name', type=str, choices=['VGG16', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'])
        parser.add_argument('--pretrained', type=str, choices=['True', 'False'])
        parser.add_argument('--no_pooling_or_stride_conv1', type=str, choices=['True', 'False'])
        parser.add_argument('--no_change_deep_layers', type=str, choices=['True', 'False'])
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--epochs', type=int, default=120)
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'])
        parser.add_argument('--lr_scheduler', type=str, choices=['StepLR', 'MultiStepLR', 'CosineAnnealingLR'])
        parser.add_argument('--step_size_lr_scheduler', type=none_or_int, default=None)
        parser.add_argument('--multistep_lr_scheduler', type=none_or_str, default=None)
        parser.add_argument('--gamma_lr_scheduler', type=none_or_float, default=None)
        parser.add_argument('--T_max', type=none_or_int, default=None)

        args = parser.parse_args()
        args.train = args.train == 'True'
        args.ignore_checkpoint = args.ignore_checkpoint == 'True'
        args.pretrained = args.pretrained == 'True'
        args.no_pooling_or_stride_conv1 = args.no_pooling_or_stride_conv1 == 'True'
        args.no_change_deep_layers = args.no_change_deep_layers == 'True'
        args.augment = args.augment == 'True'
        if args.multistep_lr_scheduler is not None:
            args.multistep_lr_scheduler = args.multistep_lr_scheduler.split('-')
            args.multistep_lr_scheduler = [int(step) for step in args.multistep_lr_scheduler]

        if args.dataset_name == 'MNIST' or args.dataset_name == 'FashionMNIST':
            dataset_root = '../../Datasets'
        elif args.dataset_name == 'CIFAR10' or args.dataset_name == 'CIFAR100':# or args.dataset_name == 'ImageNet':
            dataset_root = '/datasets'
        elif args.dataset_name == 'ImageNet':
            dataset_root = '../../Datasets'
        else:
            raise ValueError('dataset_name not recognized')

    else:
        raise ValueError('computer not recognized')

    if not os.environ.keys().__contains__("LOCAL_RANK") or (os.environ.keys().__contains__("LOCAL_RANK") and dev == 0):
        print(args)

    if os.environ.keys().__contains__("LOCAL_RANK"):
        ddp_setup()
        dev = int(os.environ['LOCAL_RANK'])
        device = dev
    else:
        device = torch.device(dev)

    torch.manual_seed(42)
    if dev == 'cuda' or isinstance(dev, int):
        torch.cuda.manual_seed(42)

    run_number = args.run_number
    dataset_name = args.dataset_name
    model_name = args.model_name
    pretrained = args.pretrained
    batch_size = args.batch_size
    retrain = args.train
    ignore_checkpoint = args.ignore_checkpoint
    lr = args.lr  # We choose to take the lr strategy that is the same on every dataset and model (similar to competition)
    optimizer = args.optimizer
    lr_scheduler = args.lr_scheduler
    epochs = args.epochs
    step_size_lr_scheduler = args.step_size_lr_scheduler
    multistep_lr_scheduler = args.multistep_lr_scheduler
    gamma_lr_scheduler = args.gamma_lr_scheduler
    T_max = args.T_max
    no_pooling_or_stride_conv1 = args.no_pooling_or_stride_conv1
    no_change_deep_layers = args.no_change_deep_layers
    augment = args.augment

    ##################### Train hyperparameters #####################

    if not os.environ.keys().__contains__("LOCAL_RANK"):
        num_workers = 0 if computer == 'local' else 4
    else:
        num_workers = 4

    ##################### Res preparation #####################

    str_pretrained = '_pretrained' if pretrained else ''
    str_no_pooling_or_stride_conv1 = '_noPoolStrideConv1' if no_pooling_or_stride_conv1 else ''
    str_no_change_deep_layers = '_noChangeDeepLayers' if no_change_deep_layers else ''
    if lr_scheduler == 'StepLR':
        str_scheduler_args = 'step_size_lr_scheduler_' + str(step_size_lr_scheduler) + \
                             '_gamma_lr_scheduler_' + str(gamma_lr_scheduler)
    elif lr_scheduler == 'MultiStepLR':
        str_scheduler_args = 'multistep_lr_scheduler_' + '-'.join([str(step) for step in multistep_lr_scheduler]) + \
                             '_gamma_lr_scheduler_' + str(gamma_lr_scheduler)
    else:
        str_scheduler_args = 'T_max_' + str(T_max)

    dir_res = os.path.join(
        'res',
        'classification',
        dataset_name,
        'baseline',
        model_name + str_pretrained + str_no_pooling_or_stride_conv1 + str_no_change_deep_layers + '_lastReplacement',
        'augment_' + str(augment),
        'bs_'+str(batch_size),
        'lr_' + str(lr) + '_optimizer_' + optimizer + '_lr_scheduler_' + lr_scheduler,
        str_scheduler_args,
    )
    if run_number > 0:
        dir_res = os.path.join(dir_res, 'run_' + str(run_number))
    pathlib.Path(dir_res).mkdir(parents=True, exist_ok=True)

    dir_checkpoint = os.path.join(dir_res, 'checkpoint')
    pathlib.Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()

    model = import_model_architecture(model_name, pretrained)
    model = model.to(device)
    # Change classification layer in model
    model = adapt_model_classifier(model_name, model, dataset_name).to(device)  # device just in case
    model = convert_model_baseline(
        model_name, model,
        no_pooling_or_stride_conv1=no_pooling_or_stride_conv1,
        no_change_deep_layers=no_change_deep_layers
    ).to(device)  # device just in case

    if os.environ.keys().__contains__("LOCAL_RANK"):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    ##################### Train #####################
    if retrain:

        if not os.environ.keys().__contains__("LOCAL_RANK") or (os.environ.keys().__contains__("LOCAL_RANK") and dev == 0):
            warnings.warn('Retrain activated. Previous results will be overwritten')

        ##################### Data preparation #####################

        mean, std, size_resize, crop_size, inp_size = get_dataset_params(dataset_name)

        resize = torchvision.transforms.Resize(
            (size_resize, size_resize)) if size_resize is not None else torch.nn.Identity()
        crop = torchvision.transforms.CenterCrop(crop_size) if crop_size is not None else torch.nn.Identity()
        normalize = torchvision.transforms.Normalize(mean, std)

        true_inp_size = size_resize if size_resize is not None else inp_size
        if augment:
            augments = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(true_inp_size),
                torchvision.transforms.RandomHorizontalFlip(),
            ])
        else:
            augments = torchvision.transforms.Compose([])

        # Write transform_train for default Imagenet CNN classification
        transform_train = torchvision.transforms.Compose([
            resize,
            crop,
            augments,
            torchvision.transforms.ToTensor(),
            normalize
        ])

        transform_val = torchvision.transforms.Compose([
            resize,
            crop,
            torchvision.transforms.ToTensor(),
            normalize
        ])


        dataset_train, dataset_val = prepare_dataset(dataset_name, dataset_root, transform_train, transform_val,
                                                     computer=computer)

        if dataset_name == 'ImageNet' or dataset_name == 'Imagenette'\
                or not model_name == 'ResNet18':  # High resolution images, need to do gradient accumulation, or large models
            num_batch_accumulate = 1  # 2
            if not batch_size % num_batch_accumulate == 0:
                raise ValueError('batch_size should be divisible by num_batch_accumulate for ImageNet')
            local_batch_size = int(batch_size / num_batch_accumulate)
        else:
            num_batch_accumulate = 1
            local_batch_size = batch_size

        if not os.environ.keys().__contains__("LOCAL_RANK"):
            dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=local_batch_size, shuffle=True,
                                                           num_workers=num_workers, drop_last=False)
            dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=local_batch_size, shuffle=False,
                                                         num_workers=num_workers, drop_last=False)
        else:
            dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=local_batch_size // int(os.environ["WORLD_SIZE"]),
                                                           shuffle=False,
                                                           num_workers=num_workers, drop_last=False, pin_memory=True,
                                                           sampler=torch.utils.data.distributed.DistributedSampler(
                                                               dataset_train, shuffle=True))
            dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=local_batch_size // int(os.environ["WORLD_SIZE"]),
                                                         shuffle=False,
                                                         num_workers=num_workers, drop_last=False, pin_memory=True,
                                                         sampler=torch.utils.data.distributed.DistributedSampler(
                                                             dataset_val, shuffle=False))

        ##################### Learning preparation #####################
        if not os.environ.keys().__contains__("LOCAL_RANK"):
            model_params = model.parameters()
        else:
            model_params = model.module.parameters()
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model_params),  # Only optimise parameters that require grad
                lr=lr, momentum=0.9, weight_decay=1e-4)  #0.1/0.01 is a common choice for sgd
            # momentum=0.9 and weight_decay=1e-4 are common values in comparable papers
            # (e.g. anisotropic conv or deform conv or dilated resnet)
        elif optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model_params),  # Only optimise parameters that require grad
                lr=lr, weight_decay=1e-4  #1e-3 to 3e-5 are good usually good for adam
            )
        else:
            raise ValueError('optimizer not recognized')

        if lr_scheduler == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size_lr_scheduler, gamma=gamma_lr_scheduler)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, multistep_lr_scheduler, gamma=gamma_lr_scheduler)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        else:
            raise ValueError('lr_scheduler not recognized')


        # clamp gradient norm after the whole pass before optimizer.step() ?
        # for p in model_params:
        #     p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        if ignore_checkpoint or not os.path.exists(os.path.join(dir_checkpoint, 'checkpoint.pth')):
            if not os.environ.keys().__contains__("LOCAL_RANK") or (os.environ.keys().__contains__("LOCAL_RANK") and dev == 0):
                if ignore_checkpoint:
                    warnings.warn('Ignore checkpoint activated. Previous checkpoint will be overwritten')
                else:
                    warnings.warn('Checkpoint not found. Starting from scratch')
            checkpoint = {}
            checkpoint['epoch'] = None
            checkpoint['loss_tracker_train'] = []
            checkpoint['loss_tracker_val'] = []
            checkpoint['last_epoch_and_nan'] = None
            checkpoint['model_state_dict'] = None
            checkpoint['optimizer_state_dict'] = None
            checkpoint['learning_rate_scheduler_dict'] = None
            checkpoint['acc_tracker_train'] = []
            checkpoint['acc_tracker_val'] = []

        else:
            if not os.environ.keys().__contains__("LOCAL_RANK"):
                checkpoint = torch.load(os.path.join(dir_checkpoint, 'checkpoint.pth'), map_location=device)
            else:
                checkpoint = torch.load(os.path.join(dir_checkpoint, 'checkpoint.pth'))
            if not os.environ.keys().__contains__("LOCAL_RANK"):
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['learning_rate_scheduler_dict'])
            if not os.environ.keys().__contains__("LOCAL_RANK") or (os.environ.keys().__contains__("LOCAL_RANK") and dev == 0):
                warnings.warn('Checkpoint found. Resuming training')

        start_epoch = checkpoint['epoch'] + 1 if checkpoint['epoch'] is not None else 0

        model, checkpoint = train_cnn_classif_several_epochs(epochs, start_epoch, dataset_train, dataloader_train,
                                                             dataset_val, dataloader_val, model, criterion, optimizer,
                                                             lr_scheduler, dir_res, dir_checkpoint, checkpoint,
                                                             num_batch_accumulate=num_batch_accumulate,
                                                             )

        ##################### Save results #####################
        if not os.environ.keys().__contains__("LOCAL_RANK") or (os.environ.keys().__contains__("LOCAL_RANK") and dev == 0):
            torch.save(checkpoint, os.path.join(dir_checkpoint, 'checkpoint.pth'))
            torch.save(checkpoint['model_state_dict'], os.path.join(dir_res, 'model.pth'))
            torch.save(checkpoint['loss_tracker_train'], os.path.join(dir_res, 'loss_tracker_train.pth'))
            torch.save(checkpoint['loss_tracker_val'], os.path.join(dir_res, 'loss_tracker_val.pth'))
            torch.save(checkpoint['acc_tracker_train'], os.path.join(dir_res, 'acc_tracker_train.pth'))
            torch.save(checkpoint['acc_tracker_val'], os.path.join(dir_res, 'acc_tracker_val.pth'))
            torch.save(checkpoint['last_epoch_and_nan'], os.path.join(dir_res, 'last_epoch_and_nan.pth'))


    ##################### Plot results #####################

    plot_results = True
    if plot_results:
        if not os.environ.keys().__contains__("LOCAL_RANK") or (os.environ.keys().__contains__("LOCAL_RANK") and dev == 0):
            warnings.warn('Plotting training results from previous saved run')

            # if not os.environ.keys().__contains__("LOCAL_RANK"):
            #     model.load_state_dict(torch.load(os.path.join(dir_res, 'model.pth'), map_location=dev))
            # else:
            #     model.module.load_state_dict(torch.load(os.path.join(dir_res, 'model.pth'), map_location=dev))
            loss_tracker_train = torch.load(os.path.join(dir_res, 'loss_tracker_train.pth'))
            loss_tracker_val = torch.load(os.path.join(dir_res, 'loss_tracker_val.pth'))
            last_epoch_and_nan = torch.load(os.path.join(dir_res, 'last_epoch_and_nan.pth'))
            acc_tracker_train = torch.load(os.path.join(dir_res, 'acc_tracker_train.pth'))
            acc_tracker_val = torch.load(os.path.join(dir_res, 'acc_tracker_val.pth'))

            # print values of trackers at the last epoch
            last_epoch = last_epoch_and_nan[0]-1 if last_epoch_and_nan[1] else last_epoch_and_nan[0]

            # Test if first element of list is a list
            # Early runs on non imagenet data did not compute top5 accuracy
            if isinstance(acc_tracker_train[0], list):
                acc5_tracker_train = [acc[1] for acc in acc_tracker_train]
                acc5_tracker_val = [acc[1] for acc in acc_tracker_val]
                acc_tracker_train = [acc[0] for acc in acc_tracker_train]
                acc_tracker_val = [acc[0] for acc in acc_tracker_val]
                acc5_tracker_train_str = '\tacc5_tracker_train: ' + str(acc5_tracker_train[last_epoch])
                acc5_tracker_val_str = '\tacc5_tracker_val: ' + str(acc5_tracker_val[last_epoch])
                flag_acc5 = True
            else:
                acc5_tracker_train_str = ''
                acc5_tracker_val_str = ''
                flag_acc5 = False

            delta_loss_normalised = (torch.tensor(loss_tracker_val) - torch.tensor(loss_tracker_train)) / torch.tensor(loss_tracker_train)

            # print values of trackers at the last epoch
            print()
            print('Model: ', model_name, ' (', dataset_name, ')')
            print()
            print('last_epoch: ', last_epoch)
            print('last_epoch_and_nan: ', last_epoch_and_nan)
            print()
            print('loss_tracker_train: ', loss_tracker_train[last_epoch],
                  '\tacc_tracker_train: ', acc_tracker_train[last_epoch],
                  acc5_tracker_train_str
                  )
            print('loss_tracker_val: ', loss_tracker_val[last_epoch],
                  '\tacc_tracker_val: ', acc_tracker_val[last_epoch],
                  acc5_tracker_val_str)
            print()
            print('\tdelta_loss_normalised\t', delta_loss_normalised[last_epoch].item())

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(loss_tracker_train, label='train')
            ax.plot(loss_tracker_val, label='val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            if last_epoch_and_nan[1]:
                ax.axvline(last_epoch, color='red', linestyle='--') #, label='nan')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(dir_res, 'loss.png'))

            fig, ax = plt.subplots()
            ax.plot(acc_tracker_train, label='train')
            ax.plot(acc_tracker_val, label='val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Acc')
            if last_epoch_and_nan[1]:
                ax.axvline(last_epoch, color='red', linestyle='--') #, label='nan')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(dir_res, 'acc.png'))

            if flag_acc5:
                fig, ax = plt.subplots()
                ax.plot(acc5_tracker_train, label='train')
                ax.plot(acc5_tracker_val, label='val')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Acc')
                if last_epoch_and_nan[1]:
                    ax.axvline(last_epoch, color='red', linestyle='--') #, label='nan')
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(dir_res, 'acc5.png'))

            warnings.warn('Plotting training results from previous saved run done.')

            if computer == 'local':
                plt.show()

        if os.environ.keys().__contains__("LOCAL_RANK"):
            torch.distributed.destroy_process_group()

        return


    ##################### End #####################


if __name__ == '__main__':
    main_baseline()
