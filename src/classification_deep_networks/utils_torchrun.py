
import torch
import torchvision
import warnings
import math
import os
import time


def none_or_int(x):
    if x == 'None':
        return None
    return int(x)


def none_or_float(x):
    if x == 'None':
        return None
    return float(x)


def none_or_str(x):
    if x == 'None':
        return None
    return str(x)


def F_randers(v, M, w):
    # ASSUMING deform_groups = 1  (same metric for each channel at a given pixel)
    # M = batch, 4, rows, cols
    # v = batch, 2, rows, cols
    # w = batch, 2, rows, cols

    # see torch.einsum for batch matrix multiplication. It is not the same as torch.bmm
    M_2x2 = torch.reshape(M, (M.shape[0], 2, 2, *M.shape[-2:]))
    norm_M_v = torch.sqrt(torch.einsum('bjrc,bijrc,birc->brc', v, M_2x2, v))
    drift_w_v = torch.einsum('birc,birc->brc', w, v)
    return norm_M_v + drift_w_v

def F_randers_batch_v(v, M, w):
    # ASSUMING deform_groups = 1  (same metric for each channel at a given pixel)
    # M = batch, 4, rows, cols
    # v = batch, n, 2, rows, cols
    # w = batch, 2, rows, cols

    v_b = torch.reshape(v, (v.shape[0]*v.shape[1], *v.shape[2:]))
    M_b = M.repeat(v.shape[1], 1, 1, 1)
    w_b = w.repeat(v.shape[1], 1, 1, 1)

    return F_randers(v_b, M_b, w_b).reshape(v.shape[0], v.shape[1], *v.shape[-2:])


def sample_unit_ball_tangent(im, M, w, eps=1e-6, kh=5, kw=5, sampling_strat='polar_grid'):
    # Returns y_s_theta: batch, kh, kw, 2, rows, cols

    # M and w use the x,y convention!

    device = im.device
    out_shape_pix = M.shape[-2:]  # M and w are already downsampled if the conv2D convMw generating them is strided

    if sampling_strat == 'polar_grid':
        # Naive polar sparse grid sampling strategy
        n_theta = math.ceil(math.sqrt(kh * kw))
        # Number of samples: n_theta ** 2 + 1. We can add 1 extra point for the centre
        theta = torch.arange(0, 2 * torch.pi - eps, 2 * torch.pi / n_theta).to(device)
        u_theta = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        u_theta = u_theta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        u_theta = torch.tile(u_theta, (1, 1, 1, *out_shape_pix))
        u_theta = torch.tile(u_theta, (im.shape[0], 1, 1, 1, 1))  # batch, n_theta, 2, rows, cols

        # M and w are already downsampled if the conv2D convMw generating them is strided

        # Compute y_theta (not just boundary - sparse grid sampling strategy)
        F_randers_u_theta = F_randers_batch_v(u_theta, M, w)  # batch, n_theta, rows, cols
        if F_randers_u_theta.min() < 0:
            warnings.warn('Warning: negative F_randers_u_theta')

        y_theta = (1 / (F_randers_u_theta + eps)).unsqueeze(2) * u_theta  # batch, n_theta, 2, rows, cols

        y_s_theta = y_theta.unsqueeze(1).repeat(1, n_theta, 1, 1, 1, 1)  # batch, n_theta, n_theta, 2, rows, cols
        s_interp = torch.arange(0, 1 - eps, 1 / n_theta).to(device) + 1 / n_theta  # n_theta
        # +eps to make sure we don't hit exactly the right value, as recommended by pytorch doc
        y_s_theta = (torch.permute(y_s_theta, (0, 2, 3, 4, 5, 1)) * s_interp).permute(0, 5, 1, 2, 3,
                                                                                      4)  # batch, n_theta, n_theta, 2, rows, cols
        # multiplication with last dimension of same size
    elif sampling_strat == 'onion_peeling_grid':  # like standard conv sampling
        u_theta = []
        s_interp = []
        for k_peel in range(int((kh - 1) / 2) + 1):
            if k_peel == 0:
                # n_theta = 1
                theta = torch.tensor([0]).to(device)
                u_theta.append(torch.stack([torch.cos(theta), torch.sin(theta)], dim=1).to(device))
                s_interp.append(torch.tensor([0.]).to(device))
            else:
                # Onion square layer of width k has 4*(k-1) points
                # Onion square layer of width k is the layer number (k-1)/2
                n_theta = 4 * (2 * k_peel)
                theta = torch.arange(0, 2 * torch.pi - eps, 2 * torch.pi / n_theta).to(device)
                u_theta.append(torch.stack([torch.cos(theta), torch.sin(theta)], dim=1).to(device))
                s_interp.append(k_peel / ((kh - 1) / 2) * torch.ones(n_theta).to(device))
            # Flatten u_theta
        u_theta_torch = torch.zeros(im.shape[0], sum([u.shape[0] for u in u_theta]), 2, *out_shape_pix).to(device)
        s_interp_torch = torch.zeros(sum([len(s) for s in s_interp])).to(device)
        u_theta_torch_idx = 0
        s_interp_torch_idx = 0
        for i in range(len(u_theta)):
            u_theta_torch[:, u_theta_torch_idx:u_theta_torch_idx + u_theta[i].shape[0], :, :, :] = u_theta[i].unsqueeze(
                0).unsqueeze(-1).unsqueeze(-1).repeat(im.shape[0], 1, 1, 1, 1)
            u_theta_torch_idx += u_theta[i].shape[0]
            s_interp_torch[s_interp_torch_idx:s_interp_torch_idx + len(s_interp[i])] = s_interp[i]
            s_interp_torch_idx += len(s_interp[i])
        u_theta = u_theta_torch  # batch, kh*kh, 2, rows, cols
        s_interp = s_interp_torch  # kh*kw
        F_randers_u_theta = F_randers_batch_v(u_theta, M, w)  # batch, kh*kw, rows, cols
        if F_randers_u_theta.min() < 0:
            warnings.warn('Warning: negative F_randers_u_theta')
        y_theta = (1 / (F_randers_u_theta + eps)).unsqueeze(2) * u_theta  # batch, kh*kw, 2, rows, cols
        y_s_theta = (torch.permute(y_theta, (0, 2, 3, 4, 1)) * s_interp).permute(0, 4, 1, 2,
                                                                                 3)  # batch, kh*kw, 2, rows, cols
        y_s_theta = torch.reshape(y_s_theta,
                                  (y_s_theta.shape[0], kh, kw, *y_s_theta.shape[-3:]))  # batch, kh, kw, 2, rows, cols
    else:
        raise ValueError('sampling_strat not recognized or implemented yet')

    return y_s_theta


def compute_offsets_ball(y_s_theta, kh, kw, dilation, device):
    # Dilation is not used by our UTB to compute sample locations. We nevertheless put it here to show how our offsets
    # differ form those of deform_conv2d with a non unit offset

    # We can implement the custom interpolation using the built-in deform_conv2d by computing offsets
    kernel_grid = torch.stack(
        torch.meshgrid(torch.arange(-dilation*(kw//2), dilation*(kw//2)+1, step=dilation),
                       torch.arange(-dilation*(kh//2), dilation*(kh//2)+1, step=dilation), indexing='ij'),
        dim=-1).float().to(device)  # i,j ordering, shape is kh, kw, 2
    offsets_ball = (y_s_theta.permute((0,4,5,1,2,3)).flip(-1) - kernel_grid).permute((0,3,4,5,1,2))  # Uses i,j ordering
    # Correct ordering is khkw2 of deform_conv2d, but the official doc misleads into thinking 2khkw
    # ordering = 'khkw2'
    offsets_ball = torch.reshape(offsets_ball, (offsets_ball.shape[0], -1, *offsets_ball.shape[-2:]))  # 1 deform_group
    return offsets_ball


def adapt_model_classifier(model_name, model, dataset_name=None, num_classes=None):

    if num_classes is None and dataset_name is None:
        raise ValueError('num_classes and dataset_name cannot be both None')

    if num_classes is None:
        if dataset_name == 'MNIST':
            num_classes = 10
        elif dataset_name == 'FashionMNIST':
            num_classes = 10
        elif dataset_name == 'CIFAR10':
            num_classes = 10
        elif dataset_name == 'CIFAR100':
            num_classes = 100
        elif dataset_name == 'STL10':
            num_classes = 10
        elif dataset_name == 'ImageNet':
            num_classes = 1000
        elif dataset_name == 'Imagenette':
            num_classes = 10
        else:
            raise ValueError('dataset_name not recognized')

    if dataset_name == 'ImageNet':
        return model  # Do no modifications (keep pretrained classification weights)

    # TODO: VGG? See, update, and fix icml version

    if model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError('model_name not recognized')

    return model


def train_cnn_classif_single_epoch(dataset_train, dataloader_train, optimizer, model, criterion, epoch, epochs,
                                   lambda_reg=0., reg_hooks=None, num_batch_accumulate=1):
    if not os.environ.keys().__contains__('LOCAL_RANK'):
        device = next(model.parameters()).device
        model.train()
    else:
        device = next(model.module.parameters()).device
        model.module.train()

    t = time.time()

    loss_train = 0.
    loss_ce_train = 0.
    correct = 0
    correct_top5 = 0
    count = 0
    batch_count_accumulate = 0
    optimizer.zero_grad()
    for i, (data, label) in enumerate(dataloader_train):
        data, label = data.to(device), label.to(device)
        if data.shape[1] == 1:
            data = data.repeat(1, 3, 1, 1)
        if reg_hooks is not None:
            reg_hooks.clear()
        out = model(data)
        loss = criterion(out, label)
        loss_ce = loss.clone().detach()
        if lambda_reg != 0:
                loss_reg = torch.mean(
                        torch.stack([
                            torch.mean(torch.abs(reg_hooks.list[j_reg]))
                            for j_reg in range(len(reg_hooks.list))]
                        )
                    )  # L1 regularisation
                loss += lambda_reg * loss_reg

        if loss.isnan().any():
            print('Loss is nan')
            return None, None

        loss = loss / num_batch_accumulate
        loss_ce = loss_ce / num_batch_accumulate

        loss.backward()
        if batch_count_accumulate < num_batch_accumulate - 1 and not i == len(dataloader_train) - 1:
            batch_count_accumulate += 1
        else:
            batch_count_accumulate = 0
            optimizer.step()
            optimizer.zero_grad()

        predicted = torch.max(out, 1).indices.detach()
        correct_batch = (predicted == label).sum().item()
        correct_top5_batch = (torch.topk(out, 5).indices == label.unsqueeze(1)).sum().item()
        if not os.environ.keys().__contains__('LOCAL_RANK'):
            loss_train += loss.item()
            loss_ce_train += loss_ce.item()
            count += data.shape[0]
            correct += correct_batch
            correct_top5 += correct_top5_batch

        # all_reduce if necessary: count, correct_batch, correct_top5_batch, correct, correct_top5, loss_ce, loss_reg, loss
        if os.environ.keys().__contains__('LOCAL_RANK'):
            count_batch = torch.scalar_tensor(data.shape[0]).to(device)
            correct_batch = torch.scalar_tensor(correct_batch).to(device)
            correct_top5_batch = torch.scalar_tensor(correct_top5_batch).to(device)
            loss_ce = torch.scalar_tensor(loss_ce).to(device)
            loss = torch.scalar_tensor(loss.item()).to(device)
            # if int(os.environ['LOCAL_RANK']) == 0:
            #     print()
            #     print('before', int(os.environ['LOCAL_RANK']), count_batch, correct_batch, correct_top5_batch, count, correct, correct_top5, loss_ce, loss_ce_train)
            # torch.distributed.barrier()
            # if int(os.environ['LOCAL_RANK']) == 1:
            #     print('before', int(os.environ['LOCAL_RANK']), count_batch, correct_batch, correct_top5_batch, count, correct, correct_top5, loss_ce, loss_ce_train)
            # torch.distributed.barrier()
            torch.distributed.all_reduce(count_batch)
            torch.distributed.all_reduce(correct_batch)
            torch.distributed.all_reduce(correct_top5_batch)
            torch.distributed.all_reduce(loss_ce)
            torch.distributed.all_reduce(loss)
            count_batch = count_batch.item()
            correct_batch = correct_batch.item()
            correct_top5_batch = correct_top5_batch.item()
            count += count_batch
            correct += correct_batch
            correct_top5 += correct_top5_batch
            loss_ce = loss_ce.item()
            loss = loss.item()
            loss_train += loss
            loss_ce_train += loss_ce
            # if int(os.environ['LOCAL_RANK']) == 0:
            #     print('after', int(os.environ['LOCAL_RANK']), count_batch, correct_batch, correct_top5_batch, count, correct, correct_top5, loss_ce, loss_ce_train)
            # torch.distributed.barrier()
            # if int(os.environ['LOCAL_RANK']) == 1:
            #     print('after', int(os.environ['LOCAL_RANK']), count, correct_batch, correct_top5_batch, correct, correct_top5, loss_ce, loss_ce_train)
            # torch.distributed.barrier()
            if lambda_reg != 0:
                # loss_reg = torch.scalar_tensor(loss_reg).to(device)
                torch.distributed.all_reduce(loss_reg)
                loss_reg = loss_reg.item()

        if not os.environ.keys().__contains__('LOCAL_RANK') \
                or (os.environ.keys().__contains__('LOCAL_RANK') and int(os.environ['LOCAL_RANK']) == 0):
            # in print, end='\r' bugs in pycharm run mode and shows nothing. Instead, put the \r at the beginning and end ''
            str_reg = ''
            if lambda_reg != 0:
                str_reg = ' Loss regs: {:.8f}'.format(loss_reg / count_batch) + ','
            print(
                '\rEpoch: {}/{}, Batch: {}/{}, Loss ce batch: {:.8f}, Loss ce: {:.8f},{} Acc batch: {}, Acc: {:.8f}, Acc5 batch: {}, Acc5: {:.8f}'.format(
                    epoch, epochs, i, len(dataloader_train), loss_ce / count_batch, loss_ce_train / count, str_reg,
                    correct_batch / count_batch, correct / count, correct_top5_batch / count_batch, correct_top5 / count
                ),
                end='') #Can cause no printing in run on pycharm
    if not os.environ.keys().__contains__('LOCAL_RANK') \
            or (os.environ.keys().__contains__('LOCAL_RANK') and int(os.environ['LOCAL_RANK']) == 0):
        print('')
        print('Time for epoch: ', time.time() - t)
        torch.cuda.reset_peak_memory_stats(device=None)
        print(f"gpu used {torch.cuda.max_memory_allocated(device=None) / (1024 ** 2)} memory")  # in MB
    loss_train /= len(dataset_train)
    loss_ce_train /= len(dataset_train)
    acc_train = correct / len(dataset_train)
    acc_top5_train = correct_top5 / len(dataset_train)
    acc_train = [acc_train, acc_top5_train]
    return loss_train, acc_train


def val_cnn_classif(dataset_val, dataloader_val, model, criterion, lambda_reg=0., reg_hooks=None, num_batch_accumulate=1):

    if not os.environ.keys().__contains__('LOCAL_RANK'):
        device = next(model.parameters()).device
        model.eval()
    else:
        device = next(model.module.parameters()).device
        model.module.eval()

    with torch.no_grad():
        loss_val = 0.
        loss_ce_val = 0.
        loss_reg_val = 0. if lambda_reg != 0 else None
        correct = 0
        correct_top5 = 0
        for i, (data, label) in enumerate(dataloader_val):
            data, label = data.to(device), label.to(device)
            if reg_hooks is not None:
                reg_hooks.clear()
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
            out = model(data)
            loss = criterion(out, label)
            loss_ce = loss.clone().detach()
            if lambda_reg != 0:
                loss_val_reg_batch = torch.mean(
                    torch.stack([
                        torch.mean(torch.abs(reg_hooks.list[j_reg]))
                        for j_reg in range(len(reg_hooks.list))]
                    )
                )  # L1 regularisation
                loss += lambda_reg * loss_val_reg_batch
                loss_reg_val += loss_val_reg_batch.item()
            if loss.isnan().any():
                print('Val Loss is nan')
                return None, None

            loss = loss / num_batch_accumulate
            loss_ce = loss_ce / num_batch_accumulate

            loss_val += loss.item()
            loss_ce_val += loss_ce.item()
            predicted = torch.max(out, 1).indices
            correct_batch = (predicted == label).sum().item()
            correct_top5_batch = (torch.topk(out, 5).indices == label.unsqueeze(1)).sum().item()
            correct += correct_batch
            correct_top5 += correct_top5_batch

        # all_reduce if necessary: correct_batch, correct_top5_batch, correct, correct_top5, loss_ce, loss_reg, loss
        if os.environ.keys().__contains__('LOCAL_RANK'):
            correct = torch.tensor(correct).to(device)
            correct_top5 = torch.tensor(correct_top5).to(device)
            loss_ce_val = torch.tensor(loss_ce_val).to(device)
            torch.distributed.all_reduce(correct)
            torch.distributed.all_reduce(correct_top5)
            torch.distributed.all_reduce(loss_ce_val)
            correct = correct.item()
            correct_top5 = correct_top5.item()
            loss_ce_val = loss_ce_val.item()
            if lambda_reg != 0:
                loss_reg_val = torch.tensor(loss_reg_val).to(device)
                torch.distributed.all_reduce(loss_reg_val)
                loss_reg_val = loss_reg_val.item()

        loss_val /= len(dataset_val)
        loss_ce_val /= len(dataset_val)
        acc_val = correct / len(dataset_val)
        acc_top5_val = correct_top5 / len(dataset_val)
        acc_val = [acc_val, acc_top5_val]
        if lambda_reg != 0:
            loss_reg_val /= len(dataset_val)

        str_val_reg = ''
        if lambda_reg != 0.:
            str_val_reg = ' Loss regs: {:.8f}'.format(loss_reg_val) + ','
        str_print_val = 'Loss_ce_val: {:.8f},{} Acc_ce_val: {:.8f}, Acc5_val: {:.8f}'.format(loss_ce_val, str_val_reg, acc_val[0], acc_val[1])
    return loss_val, acc_val, str_print_val


def import_model_architecture(model_name, pretrained):
    if model_name == 'ResNet18':
        weights = None if not pretrained else torchvision.models.ResNet18_Weights.DEFAULT
        return torchvision.models.resnet18(weights=weights)
    elif model_name == 'ResNet34':
        weights = None if not pretrained else torchvision.models.ResNet34_Weights.DEFAULT
        return torchvision.models.resnet34(weights=weights)
    elif model_name == 'ResNet50':
        weights = None if not pretrained else torchvision.models.ResNet50_Weights.DEFAULT
        return torchvision.models.resnet50(weights=weights)
    elif model_name == 'ResNet101':
        weights = None if not pretrained else torchvision.models.ResNet101_Weights.DEFAULT
        return torchvision.models.resnet101(weights=weights)
    elif model_name == 'ResNet152':
        weights = None if not pretrained else torchvision.models.ResNet152_Weights.DEFAULT
        return torchvision.models.resnet152(weights=weights)
    else:
        raise ValueError('model_name not recognized')


def prepare_dataset(dataset_name, dataset_root, transform_train, transform_val, computer=None):
    if dataset_name == 'MNIST':
        dataset_train = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform_train)
        dataset_val = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform_val)
    elif dataset_name == 'FashionMNIST':
        dataset_train = torchvision.datasets.FashionMNIST(root=dataset_root, train=True, download=True, transform=transform_train)
        dataset_val = torchvision.datasets.FashionMNIST(root=dataset_root, train=False, download=True, transform=transform_val)
    elif dataset_name == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform_train)
        dataset_val = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform_val)
    elif dataset_name == 'CIFAR100':
        dataset_train = torchvision.datasets.CIFAR100(root=dataset_root, train=True, download=True, transform=transform_train)
        dataset_val = torchvision.datasets.CIFAR100(root=dataset_root, train=False, download=True, transform=transform_val)
    elif dataset_name == 'ImageNet':
        if computer == 'mac':  # Will not work on mac with the new change to computer == 'local'
            dataset_train = torchvision.datasets.ImageNet(root=dataset_root, split='train', download=True, transform=transform_train)
            dataset_val = torchvision.datasets.ImageNet(root=dataset_root, split='val', download=True, transform=transform_val)
        else:
            dataset_train = torchvision.datasets.ImageNet(root=dataset_root, split='train', transform=transform_train)
            dataset_val = torchvision.datasets.ImageNet(root=dataset_root, split='val', transform=transform_val)
    elif dataset_name == 'Imagenette':
        dataset_train = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_root, 'imagenette2-160', 'train'), transform=transform_train)
        dataset_val = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_root, 'imagenette2-160', 'val'), transform=transform_val)
    else:
        raise ValueError('dataset_name not recognized')
    return dataset_train, dataset_val


def get_dataset_params(dataset_name):
    if dataset_name == 'MNIST':
        mean, std = (0.1307,), (0.3081,)
        size_resize = None  # if not model_name[:3] == 'VGG' else 32  # VGG needs at least 32x32
        crop_size = None
        inp_size = 28
    elif dataset_name == 'FashionMNIST':
        mean, std = (0.2860,), (0.3530,)
        size_resize = None  # if not model_name[:3] == 'VGG' else 32  # VGG needs at least 32x32
        crop_size = None
        inp_size = 28
    elif dataset_name == 'CIFAR10':
        mean, std = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
        size_resize = None
        crop_size = None
        inp_size = 32
    elif dataset_name == 'CIFAR100':
        mean, std = (0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)
        size_resize = None
        crop_size = None
        inp_size = 32
    elif dataset_name == 'ImageNet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        size_resize = 256
        crop_size = 224
        inp_size = 224
    elif dataset_name == 'Imagenette':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        size_resize = 256
        crop_size = 224
        inp_size = 224
    else:
        mean, std = None, None
        size_resize = None
        crop_size = None
        inp_size = None
    return mean, std, size_resize, crop_size, inp_size


def train_cnn_classif_several_epochs(
        epochs, start_epoch, dataset_train, dataloader_train, dataset_val, dataloader_val,
        model, criterion, optimizer, lr_scheduler, dir_res, dir_checkpoint, checkpoint,
        lambda_reg=0., reg_hooks=None, num_batch_accumulate=1):
    for epoch in range(start_epoch, epochs):
        loss_train, acc_train = train_cnn_classif_single_epoch(
            dataset_train, dataloader_train, optimizer, model, criterion, epoch, epochs, lambda_reg, reg_hooks, num_batch_accumulate)
        loss_val, acc_val, str_print_val = val_cnn_classif(dataset_val, dataloader_val, model, criterion, lambda_reg, reg_hooks, num_batch_accumulate)
        if not os.environ.keys().__contains__('LOCAL_RANK') \
                or (os.environ.keys().__contains__('LOCAL_RANK') and int(os.environ['LOCAL_RANK']) == 0):
            print('Epoch: {}/{}, {}'.format(epoch, epochs, str_print_val))

        if loss_train is None or loss_val is None:
            checkpoint['last_epoch_and_nan'] = [epoch, True]
            break  # Does not save new checkpoint giving nan loss

        lr_scheduler.step()

        # update and overwrite checkpoint
        checkpoint['epoch'] = epoch
        checkpoint['last_epoch_and_nan'] = [epoch, False]
        checkpoint['loss_tracker_train'].append(loss_train)
        checkpoint['loss_tracker_val'].append(loss_val)
        checkpoint['acc_tracker_train'].append(acc_train)
        checkpoint['acc_tracker_val'].append(acc_val)
        if not os.environ.keys().__contains__('LOCAL_RANK'):
            checkpoint['model_state_dict'] = model.state_dict()
        else:
            checkpoint['model_state_dict'] = model.module.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['learning_rate_scheduler_dict'] = lr_scheduler.state_dict()
        if not os.environ.keys().__contains__('LOCAL_RANK') \
                or (os.environ.keys().__contains__('LOCAL_RANK') and int(os.environ['LOCAL_RANK']) == 0):
            torch.save(checkpoint, os.path.join(dir_checkpoint, 'checkpoint.pth'))
        ######################################
        # if epoch in [20, 50, 100]:  # TODO: Remove this
        #     torch.save(checkpoint, os.path.join(dir_checkpoint, 'checkpoint_{}.pth'.format(epoch)))
    return model, checkpoint


# List of Hooks of intermediate outputs. After forward call of a module with a registered hook will append its output
class OutputHook:
    def __init__(self, num_lists=1):
        self.num_lists = num_lists
        self.lists = [[] for _ in range(num_lists)]

    def __call__(self, module, input, output):
        for i in range(self.num_lists):
            self.lists[i].append(output[i])

    def clear(self):
        self.lists = [[] for _ in range(self.num_lists)]


# List of Hooks of intermediate outputs. After forward call of a module with a registered hook will append its output
class WeightHook:
    def __init__(self):
        self.list = []

    def __call__(self, module, input, output):
        for key in module._parameters:
            if module._parameters[key] is not None:
                self.list.append(module._parameters[key])

    def clear(self):
        self.list = []

