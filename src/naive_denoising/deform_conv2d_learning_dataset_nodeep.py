
import torch
import torchvision
import os
from PIL import Image
import pathlib
import warnings

import torch_lr_finder  # https://github.com/davidtvs/pytorch-lr-finder

# train = torchvision.datasets.VOCSegmentation(dataset_root, download=True, image_set='train')
# trainval = torchvision.datasets.VOCSegmentation(dataset_root, download=True, image_set='trainval')
# val = torchvision.datasets.VOCSegmentation(dataset_root, download=True, image_set='val')


class CustomDeformModelForLrFinder(torch.nn.Module):
    def __init__(self, conv_offsets, ker):
        super(CustomDeformModelForLrFinder, self).__init__()
        self.conv_offsets = conv_offsets
        self.ker = ker

    def forward(self, input):
        offsets = self.conv_offsets(input)
        return offsets, input, self.ker


class CustomOffsetCriterion(torch.nn.Module):
    def __init__(self, criterion):
        super(CustomOffsetCriterion, self).__init__()
        self.criterion = criterion

    def forward(self, offsets_and_input_and_ker, target):
        offsets, input, ker = offsets_and_input_and_ker
        k = ker.shape[-1]
        image_noisy_deformed = torchvision.ops.deform_conv2d(
            input, offset=offsets, weight=ker, dilation=1, padding=int(k // 2)
        )
        loss = self.criterion(image_noisy_deformed, target)
        return loss

class Dataset_noise(torch.utils.data.Dataset):
    def __init__(self, dataset_path, sigma, bw=True, transform=None, target_transform=None):
        # Transform is needed to get images of same shape into batch. If single image, no need for transform
        super(Dataset_noise, self).__init__()
        self.dataset_path = dataset_path
        self.sigma = float(sigma)
        self.bw = bw
        self.path_gt = os.path.join(self.dataset_path, 'gt')
        self.path_noisy = os.path.join(self.dataset_path, 'sigma_{}'.format(self.sigma))
        self.images_gt = os.listdir(self.path_gt)
        self.images_noisy = os.listdir(self.path_noisy)
        self.transform = transform if transform is not None else torchvision.transforms.ToTensor()
        self.target_transform = target_transform if target_transform is not None else torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        image_gt = Image.open(os.path.join(self.path_gt, self.images_gt[index]))
        image_noisy = Image.open(os.path.join(self.path_noisy, self.images_noisy[index]))
        if self.bw:
            image_gt = image_gt.convert('L')
            image_noisy = image_noisy.convert('L')
        else:
            image_gt = image_gt.convert('RGB')
            image_noisy = image_noisy.convert('RGB')
        image_noisy = self.transform(image_noisy)
        image_gt = self.target_transform(image_gt)
        return image_noisy, image_gt

    def __len__(self):
        return len(self.images_gt)


def prepare_noise_datasets(dataset_root, sigma, bw=True, size_resize=256):

    # Adding noise to images is not the same whether we add to rgb then convert to bw or add to bw directly
    # We resize images before adding noise as resize after noise filters it out due to interpolation

    dataset_names = ['BSDS300', 'PascalVOC2012']  # 'BSDS300', 'standard_test_images', 'PascalVOC2012'
    # TODO: Include 'standard_test_images' and fix tiff images

    color = 'bw' if bw else 'color'

    for dataset_name in dataset_names:

        if dataset_name == 'BSDS300':  # train only on train, test only on val (do not hyperparameter tune on val), no test
            # Images are stored in train, val, test folders (with test belonging to BSDS500 and so excluded)
            # Hack: use image folder to create the noisy dataset
            if not bw:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((size_resize, size_resize))
                ])
            else:

                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.Resize((size_resize, size_resize))
                ])
            dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_root, 'BSR', 'BSDS500', 'data', 'images'),
                                                       transform=transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            # pathlib create folder for saving images
            for set_ in ['train', 'val']:
                pathlib.Path(
                    os.path.join(dataset_root, 'noisy_datasets', 'BSDS300', color, set_, 'sigma_{}'.format(sigma))
                ).mkdir(parents=True, exist_ok=True)
                pathlib.Path(
                    os.path.join(dataset_root, 'noisy_datasets', 'BSDS300', color, set_, 'gt')
                ).mkdir(parents=True, exist_ok=True)

            for i, (image, y) in enumerate(dataloader):

                set = dataset.classes[y]
                if set == 'test':
                    continue

                image = image.squeeze()
                image_noisy = image + sigma * torch.randn(image.shape)
                image_noisy = torch.clamp(image_noisy, 0, 1)

                torchvision.utils.save_image(
                    image_noisy,
                    os.path.join(dataset_root, 'noisy_datasets', 'BSDS300', color, set, 'sigma_{}'.format(sigma), 'image_'+str(i)+'.png')
                )
                torchvision.utils.save_image(
                    image,
                    os.path.join(dataset_root, 'noisy_datasets', 'BSDS300', color, set, 'gt', 'image_'+str(i)+'.png')
                )

        elif dataset_name == 'standard_test_images':
            dataset_path = os.path.join(dataset_root, 'standard_test_images')

            # pathlib create folder for saving images
            for set_ in ['train', 'val']:
                pathlib.Path(
                    os.path.join(dataset_root, 'noisy_datasets', 'standard_test_images', color, set_, 'sigma_{}'.format(sigma))
                ).mkdir(parents=True, exist_ok=True)
                pathlib.Path(
                    os.path.join(dataset_root, 'noisy_datasets', 'standard_test_images', color, set_, 'gt')
                ).mkdir(parents=True, exist_ok=True)

            color_code = {'bw': 'L', 'color': 'RGB'}

            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((size_resize, size_resize))
            ])

            files = os.listdir(dataset_path)
            # Enumerate files
            for i, file in enumerate(files):
                image = Image.open(os.path.join(dataset_path, file)).convert(color_code[color])
                image = transform(image)
                image_noisy = image + sigma * torch.randn(image.shape)
                image_noisy = torch.clamp(image_noisy, 0, 1)
                if i <= len(files) * 0.75:
                    set = 'train'
                else:
                    set = 'val'
                torchvision.utils.save_image(
                    image_noisy,
                    os.path.join(dataset_root, 'noisy_datasets', 'standard_test_images', color, set, 'sigma_{}'.format(sigma), 'image_'+str(i)+'.png')
                )
                torchvision.utils.save_image(
                    image,
                    os.path.join(dataset_root, 'noisy_datasets', 'standard_test_images', color, set, 'gt', 'image_'+str(i)+'.png')
                )
        elif dataset_name == 'PascalVOC2012':
            train_segmentation = torchvision.datasets.VOCSegmentation(dataset_root, download=True, image_set='train')
            val_segmentation = torchvision.datasets.VOCSegmentation(dataset_root, download=True, image_set='val')

            # pathlib create folder for saving images
            for set_ in ['train', 'val']:
                pathlib.Path(
                    os.path.join(dataset_root, 'noisy_datasets', 'PascalVOC2012', color, set_, 'sigma_{}'.format(sigma))
                ).mkdir(parents=True, exist_ok=True)
                pathlib.Path(
                    os.path.join(dataset_root, 'noisy_datasets', 'PascalVOC2012', color, set_, 'gt')
                ).mkdir(parents=True, exist_ok=True)

            if not bw:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((size_resize, size_resize))
                ])
            else:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.Resize((size_resize, size_resize))
                ])

            # Enumerate on images
            for set, set_segmentation in [('train', train_segmentation), ('val', val_segmentation)]:
                for i, (image, _) in enumerate(set_segmentation):
                    image = transform(image)
                    image_noisy = image + sigma * torch.randn(image.shape)
                    image_noisy = torch.clamp(image_noisy, 0, 1)
                    torchvision.utils.save_image(image_noisy, os.path.join(dataset_root, 'noisy_datasets', 'PascalVOC2012', color, set, 'sigma_{}'.format(sigma), 'image_'+str(i)+'.png'))
                    torchvision.utils.save_image(image, os.path.join(dataset_root, 'noisy_datasets', 'PascalVOC2012', color, set, 'gt', 'image_'+str(i)+'.png'))
        else:
            raise ValueError('dataset_name not recognized')

    return


def train_single_epoch(dataset_train, dataloader_train, optimizer, conv_offsets, criterion, epoch, epochs, ker, k, device):
    conv_offsets.train()
    loss_train = 0.
    for i, (image_noisy, image_gt) in enumerate(dataloader_train):
        image_noisy, image_gt = image_noisy.to(device), image_gt.to(device)
        optimizer.zero_grad()
        offsets = conv_offsets(image_noisy)
        image_noisy_deformed = torchvision.ops.deform_conv2d(
            image_noisy, offset=offsets, weight=ker, dilation=1, padding=int(k//2)
        )
        loss = criterion(image_noisy_deformed, image_gt)
        if loss.isnan().any():
            print('Loss is nan')
            return None
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        print('Epoch: {}/{}, Batch: {}/{}, Loss batch: {}, Loss: {}'.format(
            epoch, epochs, i, len(dataloader_train), loss, loss_train / len(dataset_train)), end='\r')

    print('')
    loss_train /= len(dataset_train)
    return loss_train


def val(dataset_val, dataloader_val, conv_offsets, criterion, ker, k, device):
    conv_offsets.eval()
    with torch.no_grad():
        loss_val = 0.
        for i, (image_noisy, image_gt) in enumerate(dataloader_val):
            image_noisy, image_gt = image_noisy.to(device), image_gt.to(device)
            offsets = conv_offsets(image_noisy)
            image_noisy_deformed = torchvision.ops.deform_conv2d(
                image_noisy, offset=offsets, weight=ker, dilation=1, padding=int(k // 2)
            )
            loss = criterion(image_noisy_deformed, image_gt)
            loss_val += loss.item()

        loss_val /= len(dataset_val)
    return loss_val


def main_deform():

    dev = 'cpu'  # 'cpu' or 'cuda'
    device = torch.device(dev)

    if dev == 'cpu' or not torch.cuda.is_available():
        import warnings
        warnings.warn('Using cpu for training. This will be slow. If you have a cuda device, set dev to "cuda"')

    dataset_root = '../Datasets'

    sigma = 0.5  # 0.1, 0.3, 0.5
    bw = True
    color = 'bw' if bw else 'color'

    ##################### Prepare noisy datasets: do this only once #####################

    prepare_data = False  # If True, prepare noisy datasets and exit
    if prepare_data:
        warnings.warn('Data is being prepared. No training will be done.')

        torch.manual_seed(0)
        if dev == 'cuda':
            torch.cuda.manual_seed(0)

        prepare_noise_datasets(dataset_root, sigma, bw=bw)

        warnings.warn('Data has been prepared. Set prepare_data to False to avoid doing it again. Exiting...')
        return

    ##################### Train hyperparameters #####################

    torch.manual_seed(42)
    if dev == 'cuda':
        torch.cuda.manual_seed(42)

    dataset_name = 'PascalVOC2012'  # 'BSDS300', 'PascalVOC2012', # TODO:  'standard_test_images'

    k = 31  # 5, 11, 31
    kh, kw = k, k

    ker_fixed = False  # True, False

    if k == 5 or k == 11:
        batch_size = 32
    elif k == 31:
        batch_size = 4
    else:
        batch_size = 32  # Default value

    if dataset_name == 'BSDS300':
        if sigma == 0.1 and k == 11 and bw and batch_size == 32 and ker_fixed:
            lr = 3.4e3  # Provided by lr_finder
        elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and ker_fixed:  # Weren't able to use higher batch size due to mem issues
            lr = 1.0e4
        elif sigma == 0.1 and k == 5 and bw and batch_size == 32 and ker_fixed:
            lr = 5.7e2
        elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and ker_fixed:
            lr = 3.2e2
        elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and ker_fixed:
            lr = 1.5e3
        elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and ker_fixed:
            lr = 1.0e4
        elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and ker_fixed:
            lr = 2.5e2
        elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and ker_fixed:
            lr = 1.5e3
        elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and ker_fixed:
            lr = 1.0e4
        elif sigma == 0.1 and k == 5 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and not ker_fixed:  # Weren't able to use higher batch size due to mem issues
            lr = 6.6e-4
        elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and not ker_fixed:
            lr = 6.6e-4
        elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and not ker_fixed:
            lr = 6.6e-4
        else:
            lr = None  # Will generate an error if used for training. But not if used for finding lr
    elif dataset_name == 'PascalVOC2012':
        if sigma == 0.1 and k == 5 and bw and batch_size == 32 and ker_fixed:
            lr = 4.1e2
        elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and ker_fixed:
            lr = 4.0e3
        elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and ker_fixed:
            lr = 4.0e3
        elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and ker_fixed:
            lr = 3.0e2
        elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and ker_fixed:
            lr = 4.0e3
        elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and ker_fixed:
            lr = 4.0e3
        elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and ker_fixed:
            lr = 4.1e2
        elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and ker_fixed:
            lr = 1.1e3
        elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and ker_fixed:
            lr = 4.3e3
        elif sigma == 0.1 and k == 5 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and not ker_fixed:
            lr = 1.0e-3
        elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and not ker_fixed:
            lr = 1.0e-3
        elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and not ker_fixed:
            lr = 1.0e-2
        elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and not ker_fixed:
            lr = 1.0e-3
        else:
            lr = None
    else:
        lr = None
    epochs = 100

    size_resize = 256  # 256, None; if None, no resize

    if size_resize is not None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size_resize, size_resize)),
            torchvision.transforms.ToTensor()
        ])
    else:
        transform = torchvision.transforms.ToTensor()

    ##################### Prepare train objects #####################

    dir_res = os.path.join(
        './res/denoising/deform_conv2d_ker_fixed_'+str(ker_fixed)+'/', dataset_name, color, 'sigma_{}'.format(sigma),
        'k_'+str(k)+'__batchsize_'+str(batch_size)
    )
    pathlib.Path(dir_res).mkdir(parents=True, exist_ok=True)

    dataset_train = Dataset_noise(
        os.path.join(dataset_root, 'noisy_datasets', dataset_name, color, 'train'), sigma,
        bw=bw, transform=transform, target_transform=transform
    )
    dataset_val = Dataset_noise(
        os.path.join(dataset_root, 'noisy_datasets', dataset_name, color, 'val'), sigma,
        bw=bw, transform=transform, target_transform=transform
    )

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    criterion = torch.nn.MSELoss()

    conv_offsets = torch.nn.Conv2d(1, kh * kw * 2, k, padding=int(k // 2)).to(device)
    conv_offsets.weight.data.fill_(0)  # Initializes offsests to 0
    conv_offsets.weight.requires_grad = True

    ker = torch.ones(1, 1, kh, kw).to(device) / (kh * kw)
    if not ker_fixed:
        ker.requires_grad = True


    ##################### Find lr #####################

    ##################### Grid sarch #####################
    find_lr = False  # If True, lr_finder is run and no training is done
    if find_lr:
        warnings.warn('Finding learning rate activated. Full training will not be done')

        lr_min = 1e-7
        lr_max = 1e7
        num_iter = 100  # number of tested lr values

        method = 'fastai'  # 'smith' or 'fastai'

        if method == 'smith':
            step_mode = 'linear'
        elif method == 'fastai':
            step_mode = 'exp'
            dataloader_val = None  # Use only train loss
        else:
            raise ValueError('method not recognized for lr_finder')

        model = CustomDeformModelForLrFinder(conv_offsets, ker)

        if ker_fixed:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)
        else:
            optimizer = torch.optim.SGD(list(conv_offsets.parameters()) + [ker], lr=lr_min)  # Min learning rate
        lr_finder = torch_lr_finder.LRFinder(model, optimizer, CustomOffsetCriterion(criterion), device=device)
        # Here the lr_finder does only iteration per test. Otherwise use accumulation_steps>1 in lr_finder.range_test
        lr_finder.range_test(dataloader_train, val_loader=dataloader_val, end_lr=lr_max, num_iter=num_iter, step_mode=step_mode)
        torch.save(lr_finder, os.path.join(dir_res, 'lr_finder_'+method+'.pth'))

        warnings.warn('Finding learning rate activated. Choose an lr and rerun. Exiting...')
        return
    ##################### Viewing lr grid search #####################
    view_find_lr = False
    if view_find_lr:
        warnings.warn('Finding learning rate visually activated. Full training will not be done')

        import matplotlib.pyplot as plt

        method = 'fastai'  # 'smith' or 'fastai'

        lr_finder = torch.load(os.path.join(dir_res, 'lr_finder_'+method+'.pth'), map_location=dev)
        fig, ax = plt.subplots()
        ax, lr = lr_finder.plot(ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(dir_res, 'lr_finder_'+method+'.png'))
        torch.save(torch.scalar_tensor(lr), os.path.join(dir_res, 'lr_suggested_'+method+'.pth'))

        plt.show()

        warnings.warn('Finding learning rate visually activated. Choose an lr and rerun. Exiting...')
        return


    ##################### Train #####################

    if ker_fixed:
        optimizer = torch.optim.SGD(conv_offsets.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(list(conv_offsets.parameters()) + [ker], lr=lr)

    retrain = False
    if retrain:
        warnings.warn('Retrain activated. Previous results will be overwritten')

        loss_tracker_train = []
        loss_tracker_val = []
        last_epoch_and_nan = [epochs-1, False]

        for epoch in range(epochs):
            loss_train = train_single_epoch(dataset_train, dataloader_train, optimizer, conv_offsets, criterion, epoch, epochs, ker, k, device)
            if loss_train is None:
                last_epoch_and_nan = [epoch, True]
                break
            loss_tracker_train.append(loss_train)
            torch.save(conv_offsets.state_dict(), os.path.join(dir_res, 'conv_offsets_lr_'+str(lr)+'.pth'))
            torch.save(loss_tracker_train, os.path.join(dir_res, 'loss_tracker_train_lr_'+str(lr)+'.pth'))
            loss_val = val(dataset_val, dataloader_val, conv_offsets, criterion, ker, k, device)
            loss_tracker_val.append(loss_val)
            print('Epoch: {}/{}, Loss_val: {}'.format(epoch, epochs, loss_val))

        ##################### Save results #####################

        torch.save(conv_offsets.state_dict(), os.path.join(dir_res, 'conv_offsets_lr_'+str(lr)+'.pth'))
        torch.save(loss_tracker_train, os.path.join(dir_res, 'loss_tracker_train_lr_'+str(lr)+'.pth'))
        torch.save(loss_tracker_val, os.path.join(dir_res, 'loss_tracker_val_lr_'+str(lr)+'.pth'))
        torch.save(last_epoch_and_nan, os.path.join(dir_res, 'last_epoch_and_nan_lr_'+str(lr)+'.pth'))

    ##################### Plot results #####################

    plot_results = True
    if plot_results:
        warnings.warn('Plotting training results from previous saved run')

        conv_offsets.load_state_dict(torch.load(os.path.join(dir_res, 'conv_offsets_lr_'+str(lr)+'.pth'), map_location=dev))
        loss_tracker_train = torch.load(os.path.join(dir_res, 'loss_tracker_train_lr_'+str(lr)+'.pth'))
        loss_tracker_val = torch.load(os.path.join(dir_res, 'loss_tracker_val_lr_'+str(lr)+'.pth'))
        last_epoch_and_nan = torch.load(os.path.join(dir_res, 'last_epoch_and_nan_lr_'+str(lr)+'.pth'))


        # print values of trackers at the last epoch
        # print('loss_tracker_train: ', loss_tracker_train[-1])
        if torch.isnan(torch.scalar_tensor(loss_tracker_val[-1])).item():
            loss_tracker_val_idx = -2
        else:
            loss_tracker_val_idx = -1
        print('loss_tracker_val: ', loss_tracker_val[loss_tracker_val_idx])
        print('last_epoch_and_nan: ', last_epoch_and_nan)

        delta_loss_normalised = (torch.tensor(loss_tracker_val) - torch.tensor(loss_tracker_train)) / torch.tensor(loss_tracker_train)

        # print values of trackers at the last epoch
        print('sigma:\t', sigma)
        print('k:\t', k)
        print()
        # print('\tloss_tracker_unit_tangent_ball (Train):\t', loss_tracker_train[loss_tracker_val_idx])
        print('\tloss_tracker_unit_tangent_ball (Test):\t', loss_tracker_val[loss_tracker_val_idx])
        print('\tdelta_loss_normalised_unit_tangent_ball\t', delta_loss_normalised[loss_tracker_val_idx].item())


        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(loss_tracker_train, label='train')
        ax.plot(loss_tracker_val, label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        if last_epoch_and_nan[1]:
            ax.axvline(last_epoch_and_nan[0], color='red', linestyle='--', label='nan')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(dir_res, 'loss_lr_'+str(lr)+'.png'))
        plt.show()

        warnings.warn('Plotting training results from previous saved run done. Exiting...')
        return

    ##################### End #####################

    return


if __name__ == '__main__':
    main_deform()
