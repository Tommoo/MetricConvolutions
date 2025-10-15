
import torch
import torchvision
import os
from PIL import Image
import pathlib
import warnings
import argparse

import torch_lr_finder  # https://github.com/davidtvs/pytorch-lr-finder

from singleim_learning_nonan import blur_randers_ball_tangent

# train = torchvision.datasets.VOCSegmentation(dataset_root, download=True, image_set='train')
# trainval = torchvision.datasets.VOCSegmentation(dataset_root, download=True, image_set='trainval')
# val = torchvision.datasets.VOCSegmentation(dataset_root, download=True, image_set='val')


class CustomRandersModelForLrFinder(torch.nn.Module):
    def __init__(self, conv_Mw, k, ker, sample_centre, ker_samp_ctr, eps, eps_L, eps_w, ker_fixed):
        super(CustomRandersModelForLrFinder, self).__init__()
        self.conv_Mw = conv_Mw
        self.ker = ker if (type(ker) == torch.nn.parameter.Parameter or ker_fixed) else torch.nn.parameter.Parameter(ker)
        self.ker_samp_ctr = ker_samp_ctr
        self.k = k
        self.sample_centre = sample_centre
        self.eps = eps
        self.eps_L = eps_L
        self.eps_w = eps_w

    def forward(self, input):
        Mw = self.conv_Mw(input)
        L_params = Mw[:, :3, :, :].permute((0,2,3,1))  # batch, rows, cols, 3
        w = Mw[:, -2:, :, :]  # batch, 2, rows, cols

        L_params_tuned = torch.zeros_like(L_params)
        L_params_tuned[:, :, :, torch.tensor([0, 2])] = torch.abs(L_params[:, :, :, torch.tensor([0, 2])]) + self.eps_L
        L_params_tuned[:, :, :, torch.tensor([1])] = L_params[:, :, :, torch.tensor([1])]

        L = torch.cat([L_params_tuned, torch.zeros((*L_params_tuned.shape[:-1], 1)).to(input.device)], dim=-1)[:, :, :,
            torch.tensor([0, 3, 1, 2])].reshape((*L_params_tuned.shape[:3], 2, 2))
        M = (L @ L.permute((0, 1, 2, 4, 3))).permute((0, 3, 4, 1, 2)).reshape((L.shape[0], 4, *L.shape[1:3]))

        M_inv = torch.inverse(
            M.permute((0, 2, 3, 1)).reshape((M.shape[0], *M.shape[-2:], 2, 2)) \
            + self.eps * torch.eye(2).to(M.device).unsqueeze(0).unsqueeze(1).unsqueeze(1).tile((1, *M.shape[-2:], 1, 1))
        )  # batch, rows, cols, 2, 2

        # Tune w for norm < 1-eps_w
        norm_w_Minv = torch.sqrt( \
            w.permute(0, 2, 3, 1).unsqueeze(-2) \
            @ M_inv
            @ w.permute((0, 2, 3, 1)).unsqueeze(-1)
            + self.eps  # at 0 we get nan in the gradient due to sqrt
        ).squeeze(-1).permute((0, 3, 1, 2))
        norm_w_Minv_tuned = (torch.sigmoid(norm_w_Minv) - 1 / 2) * 2 * (1 - self.eps_w)
        w_tuned = w * norm_w_Minv_tuned / (norm_w_Minv + self.eps)

        im_blur_unit_tangent_ball = blur_randers_ball_tangent(
            input, M, w_tuned, eps=self.eps, kh=self.k, kw=self.k, sample_centre=self.sample_centre, ker=self.ker,
            ker_samp_ctr=self.ker_samp_ctr
        )

        return im_blur_unit_tangent_ball


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

    # Adding noise to images is not the same whether we add to rgb then convert to bw or add to bw directly?
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

def train_single_epoch(dataset_train, dataloader_train, optimizer, conv_Mw, criterion, epoch, epochs,
                       ker, k, sample_centre, ker_samp_ctr, device, eps, eps_L, eps_w):
    conv_Mw.train()
    loss_train = 0.
    for i, (image_noisy, image_gt) in enumerate(dataloader_train):
        image_noisy, image_gt = image_noisy.to(device), image_gt.to(device)
        optimizer.zero_grad()
        Mw = conv_Mw(image_noisy)
        L_params = Mw[:, :3, :, :].permute((0,2,3,1))  # batch, rows, cols, 3
        w = Mw[:, -2:, :, :]  # batch, 2, rows, cols

        # Tune L_params for non singularity of L: abs of diagonal and shift by small eps_L
        L_params_tuned = torch.zeros_like(L_params)
        L_params_tuned[:,:,:, torch.tensor([0,2])] = torch.abs(L_params[:,:,:,torch.tensor([0,2])]) + eps_L
        L_params_tuned[:,:,:, torch.tensor([1])] = L_params[:,:,:,torch.tensor([1])]

        L = torch.cat([L_params_tuned, torch.zeros((*L_params_tuned.shape[:-1], 1)).to(device)], dim=-1)[:,:,:,
            torch.tensor([0,3,1,2])].reshape((*L_params_tuned.shape[:3], 2, 2))
        M = (L @ L.permute((0, 1, 2, 4, 3))).permute((0, 3, 4, 1, 2)).reshape((L.shape[0], 4, *L.shape[1:3]))

        M_inv = torch.inverse(
                M.permute((0,2,3,1)).reshape((M.shape[0],*M.shape[-2:], 2, 2)) \
                + eps * torch.eye(2).to(M.device).unsqueeze(0).unsqueeze(1).unsqueeze(1).tile((1, *M.shape[-2:], 1, 1))
            )  # batch, rows, cols, 2, 2

        # Tune w for norm < 1-eps_w
        norm_w_Minv = torch.sqrt( \
            w.permute(0,2,3,1).unsqueeze(-2) \
            @  M_inv
            @ w.permute((0,2,3,1)).unsqueeze(-1)
            + eps  # at 0 we get nan in the gradient due to sqrt
        ).squeeze(-1).permute((0,3,1,2))
        norm_w_Minv_tuned = (torch.sigmoid(norm_w_Minv) - 1/2) * 2 * (1 - eps_w)
        w_tuned = w * norm_w_Minv_tuned / (norm_w_Minv + eps)

        im_blur_unit_tangent_ball = blur_randers_ball_tangent(
            image_noisy, M, w_tuned, eps=eps, kh=k, kw=k, sample_centre=sample_centre, ker=ker, ker_samp_ctr=ker_samp_ctr
        )

        loss = criterion(im_blur_unit_tangent_ball, image_gt)
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


def val(dataset_val, dataloader_val, conv_Mw, criterion, ker, k, sample_centre, ker_samp_ctr, device, eps, eps_L, eps_w):
    conv_Mw.eval()
    with torch.no_grad():
        loss_val = 0.
        for i, (image_noisy, image_gt) in enumerate(dataloader_val):
            image_noisy, image_gt = image_noisy.to(device), image_gt.to(device)
            Mw = conv_Mw(image_noisy)
            L_params = Mw[:, :3, :, :].permute((0, 2, 3, 1))  # batch, rows, cols, 3
            w = Mw[:, -2:, :, :]  # batch, 2, rows, cols

            # Tune L_params for non singularity of L: abs of diagonal and shift by small eps_L
            L_params_tuned = torch.zeros_like(L_params)
            L_params_tuned[:, :, :, torch.tensor([0, 2])] = torch.abs(L_params[:, :, :, torch.tensor([0, 2])]) + eps_L
            L_params_tuned[:, :, :, torch.tensor([1])] = L_params[:, :, :, torch.tensor([1])]

            L = torch.cat([L_params_tuned, torch.zeros((*L_params_tuned.shape[:-1], 1)).to(device)], dim=-1)[:, :, :,
                torch.tensor([0, 3, 1, 2])].reshape((*L_params_tuned.shape[:3], 2, 2))
            M = (L @ L.permute((0, 1, 2, 4, 3))).permute((0, 3, 4, 1, 2)).reshape((L.shape[0], 4, *L.shape[1:3]))

            M_inv = torch.inverse(
                M.permute((0, 2, 3, 1)).reshape((M.shape[0], *M.shape[-2:], 2, 2)) \
                + eps * torch.eye(2).to(M.device).unsqueeze(0).unsqueeze(1).unsqueeze(1).tile((1, *M.shape[-2:], 1, 1))
            )  # batch, rows, cols, 2, 2

            # Tune w for norm < 1-eps_w
            norm_w_Minv = torch.sqrt( \
                w.permute(0, 2, 3, 1).unsqueeze(-2) \
                @ M_inv
                @ w.permute((0, 2, 3, 1)).unsqueeze(-1)
                + eps  # at 0 we get nan in the gradient due to sqrt
            ).squeeze(-1).permute((0, 3, 1, 2))
            norm_w_Minv_tuned = (torch.sigmoid(norm_w_Minv) - 1 / 2) * 2 * (1 - eps_w)
            w_tuned = w * norm_w_Minv_tuned / (norm_w_Minv + eps)

            im_blur_unit_tangent_ball = blur_randers_ball_tangent(
                image_noisy, M, w_tuned, eps=eps, kh=k, kw=k, sample_centre=sample_centre, ker=ker, ker_samp_ctr=ker_samp_ctr
            )

            loss = criterion(im_blur_unit_tangent_ball, image_gt)
            loss_val += loss.item()

        loss_val /= len(dataset_val)
    return loss_val


def main_unit_tangent_ball():

    script_type = 'args'  # 'local_dirty' or 'args'

    if script_type == 'local_dirty':  # Not for production, just for quick testing, manually tweak all parameters
        dev = 'cpu'
        dataset_root = '../Datasets'
        args = None
    elif script_type == 'args'  # Recommended setting. See python batch file for use cases
        dev = 'cuda'
        dataset_root = '../Datasets'

        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', type=str, choices=['BSDS300', 'PascalVOC2012'])
        parser.add_argument('--bw', type=str, choices=['True', 'False'])
        parser.add_argument('--ker_fixed', type=str, choices=['True', 'False'])
        parser.add_argument('--sample_centre', type=str, choices=['True', 'False'])
        parser.add_argument('--k', type=int)
        parser.add_argument('--sigma', type=float)
        parser.add_argument('--eps_w', type=float)

        args = parser.parse_args()
        args.bw = args.bw == 'True'
        args.ker_fixed = args.ker_fixed == 'True'
        args.sample_centre = args.sample_centre == 'True'
    else:
        raise ValueError('script_type not recognized')

    device = torch.device(dev)

    sigma = 0.1 if args is None else args.sigma         # 0.1, 0.3, 0.5
    bw = True if args is None else args.bw            # True
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

    dataset_name = 'PascalVOC2012' if args is None else args.dataset_name  # 'BSDS300', 'PascalVOC2012'
    # TODO:  'standard_test_images'

    sample_centre = False if args is None else args.sample_centre  # False

    k = 31 if args is None else args.k                              # 5, 11, 31
    # kh, kw = k, k

    ker_fixed = True if args is None else args.ker_fixed          # True, False

    eps = 1e-6
    eps_L = 0.01
    eps_w = 0.9 if args is None else args.eps_w                    # must be < 1

    if k == 5 or k == 11:
        batch_size = 32
    elif k == 31:
        batch_size = 4
    else:
        batch_size = 32  # Default value

    if eps_w == 0.1:
        if dataset_name == 'BSDS300':
            if sigma == 0.1 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 3.1e0
            elif sigma == 0.1 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 6.0e0
            elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 0.7e0
            elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 1.0e0
            elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 1.0e-1
            elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 1.0e-1
            elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 6.1e-1
            elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 3.2e-1
            elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 4.5e-2
            elif sigma == 0.1 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and not ker_fixed:
                lr = 1.0e-3
            elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-1
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
                lr = None  # Will generate an error if used for training. But not if used for finding lr
        elif dataset_name == 'PascalVOC2012':
            if sigma == 0.1 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 1.2e1
            elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 4.3e0
            elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 5.0e-2
            elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 1.5e0
            elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 1.0e-1
            elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 5.0e-2
            elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 4.3e-1
            elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.1 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-1
            elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2#1.0e-4
            elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and not ker_fixed:
                lr = 1.0e-3#1.0e-4
            elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and not ker_fixed:
                lr = 1.0e-3#1.0e-4
            elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and not ker_fixed:
                lr = 1.0e-3#1.0e-4
            else:
                lr = None
        else:
            lr = None
    elif eps_w == 0.9:
        if dataset_name == 'BSDS300':
            if sigma == 0.1 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 6.1e0
            elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 1.1e0
            elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 1.0e0
            elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 1.0e0
            elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 1.1e0
            elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 1.0e-1
            elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 1.0e0
            elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 3.8e-1
            elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 3.1e-2
            elif sigma == 0.1 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 5.0e-2
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
        elif dataset_name == 'PascalVOC2012':
            if sigma == 0.1 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 1.4e1
            elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 1.5e0
            elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 1.0e0
            elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 4.0e0
            elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 5.0e-1
            elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 1.0e-1
            elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and ker_fixed:
                lr = 8.1e0
            elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and ker_fixed:
                lr = 3.8e-1
            elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.1 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.1 and k == 11 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.1 and k == 31 and bw and batch_size == 4 and not ker_fixed:
                lr = 1.0e-3
            elif sigma == 0.3 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-1
            elif sigma == 0.3 and k == 11 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.3 and k == 31 and bw and batch_size == 4 and not ker_fixed:
                lr = 1.0e-3
            elif sigma == 0.5 and k == 5 and bw and batch_size == 32 and not ker_fixed:
                lr = 1.0e-2
            elif sigma == 0.5 and k == 11 and bw and batch_size == 32 and not ker_fixed:
                lr = 5.0e-3
            elif sigma == 0.5 and k == 31 and bw and batch_size == 4 and not ker_fixed:
                lr = 1.0e-3
            else:
                lr = None
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
        './res/denoising/unit_tangent_ball_ker_fixed_'+str(ker_fixed)+'/', dataset_name, color, 'sigma_{}'.format(sigma),
        'k_'+str(k)+'__batchsize_'+str(batch_size),
        'eps_' + str(eps) + '_eps_L_' + str(eps_L) + '_eps_w_' + str(eps_w)
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

    conv_Mw = torch.nn.Conv2d(1, 5, k, padding=int(k // 2)).to(device)
    warnings.warn('Learning (M,w) with conv and using (M,w) for unit ball. Might be best to learn (M,w) and use'
                  '(M*,w*) for unit ball? See linear vs invert dependence on image (and gradients)')
    if not sample_centre:
        normalisation_conv_init = 1. / (k ** 2)
    else:
        normalisation_conv_init = 1. / ((k ** 2) + 1)
        raise ValueError('sample centre method not implemented yet')
    conv_Mw.weight.data[0].fill_(normalisation_conv_init)
    conv_Mw.weight.data[1].fill_(0)
    conv_Mw.weight.data[2].fill_(normalisation_conv_init)
    conv_Mw.weight.data[3].fill_(0)
    conv_Mw.weight.data[4].fill_(0)
    conv_Mw.weight.requires_grad = True

    normalisation_ker_init = 1. / (k ** 2) if not sample_centre else 1. / ((k ** 2) + 1)
    ker = torch.ones(1, 1, k, k).to(device) * normalisation_ker_init
    if not ker_fixed:
        ker.requires_grad = True
    else:
        ker.requires_grad = False
    if sample_centre:
        ker_samp_ctr = torch.scalar_tensor(normalisation_ker_init).to(device)
    else:
        ker_samp_ctr = None



    ##################### Find lr #####################

    ##################### Grid sarch #####################
    find_lr = False  # If True, lr_finder is run and no training is done
    if find_lr:
        warnings.warn('Finding learning rate activated. Full training will not be done')

        lr_min = 1e-10
        lr_max = 1e2
        num_iter = 100  # number of tested lr values

        method = 'fastai'  # 'smith' or 'fastai'

        if method == 'smith':
            step_mode = 'linear'
        elif method == 'fastai':
            step_mode = 'exp'
            dataloader_val = None  # Use only train loss
        else:
            raise ValueError('method not recognized for lr_finder')

        model = CustomRandersModelForLrFinder(conv_Mw, k, ker, sample_centre, ker_samp_ctr, eps, eps_L, eps_w, ker_fixed)

        if ker_fixed:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)  # Min learning rate
        elif not sample_centre:
            optimizer = torch.optim.SGD(list(model.parameters()) + [ker], lr=lr_min)
        elif sample_centre:
            optimizer = torch.optim.SGD(list(model.parameters()) + [ker] + [ker_samp_ctr], lr=lr_min)
        else:  # Should not happen
            raise ValueError('Error in ker_fixed and sample_centre arguments')
        lr_finder = torch_lr_finder.LRFinder(model, optimizer, criterion, device=device)
        # Here the lr_finder does only iteration per test. Otherwise use accumulation_steps>1 in lr_finder.range_test
        lr_finder.range_test(dataloader_train, val_loader=dataloader_val, end_lr=lr_max, num_iter=num_iter, step_mode=step_mode,
                             diverge_th=1.0e10, accumulation_steps=1)
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
        ax, lr = lr_finder.plot(ax=ax, skip_end=0)
        fig.tight_layout()
        fig.savefig(os.path.join(dir_res, 'lr_finder_'+method+'.png'))
        torch.save(torch.scalar_tensor(lr), os.path.join(dir_res, 'lr_suggested_'+method+'.pth'))

        plt.show()

        warnings.warn('Finding learning rate visually activated. Choose an lr and rerun. Exiting...')
        return


    ##################### Train #####################

    if ker_fixed:
        optimizer = torch.optim.SGD(conv_Mw.parameters(), lr=lr)
    elif not sample_centre:
        optimizer = torch.optim.SGD(list(conv_Mw.parameters()) + [ker], lr=lr)
    elif sample_centre:
        optimizer = torch.optim.SGD(list(conv_Mw.parameters()) + [ker] + [ker_samp_ctr], lr=lr)
    else: # Should not happen
        raise ValueError('Error in ker_fixed and sample_centre arguments')

    retrain = True
    if retrain:
        warnings.warn('Retrain activated. Previous results will be overwritten')

        loss_tracker_train = []
        loss_tracker_val = []
        last_epoch_and_nan = [epochs-1, False]

        for epoch in range(epochs):
            loss_train = train_single_epoch(dataset_train, dataloader_train, optimizer, conv_Mw, criterion,
                                            epoch, epochs, ker, k, sample_centre, ker_samp_ctr, device,
                                            eps, eps_L, eps_w)
            if loss_train is None:
                last_epoch_and_nan = [epoch, True]
                break
            loss_tracker_train.append(loss_train)
            torch.save(conv_Mw.state_dict(), os.path.join(dir_res, 'conv_offsets_lr_'+str(lr)+'.pth'))
            torch.save(loss_tracker_train, os.path.join(dir_res, 'loss_tracker_train_lr_'+str(lr)+'.pth'))
            loss_val = val(dataset_val, dataloader_val, conv_Mw, criterion, ker, k, sample_centre, ker_samp_ctr, device,
                           eps, eps_L, eps_w)
            loss_tracker_val.append(loss_val)
            print('Epoch: {}/{}, Loss_val: {}'.format(epoch, epochs, loss_val))

        ##################### Save results #####################

        torch.save(conv_Mw.state_dict(), os.path.join(dir_res, 'conv_offsets_lr_'+str(lr)+'.pth'))
        torch.save(loss_tracker_train, os.path.join(dir_res, 'loss_tracker_train_lr_'+str(lr)+'.pth'))
        torch.save(loss_tracker_val, os.path.join(dir_res, 'loss_tracker_val_lr_'+str(lr)+'.pth'))
        torch.save(last_epoch_and_nan, os.path.join(dir_res, 'last_epoch_and_nan_lr_'+str(lr)+'.pth'))

    ##################### Plot results #####################

    plot_results = False
    if plot_results:
        warnings.warn('Plotting training results from previous saved run')

        conv_Mw.load_state_dict(torch.load(os.path.join(dir_res, 'conv_offsets_lr_'+str(lr)+'.pth'), map_location=dev))
        loss_tracker_train = torch.load(os.path.join(dir_res, 'loss_tracker_train_lr_'+str(lr)+'.pth'))
        loss_tracker_val = torch.load(os.path.join(dir_res, 'loss_tracker_val_lr_'+str(lr)+'.pth'))
        last_epoch_and_nan = torch.load(os.path.join(dir_res, 'last_epoch_and_nan_lr_'+str(lr)+'.pth'))

        # print values of trackers at the last epoch
        # print('loss_tracker_train: ', loss_tracker_train[-1])
        if torch.isnan(torch.scalar_tensor(loss_tracker_val[-1])).item():
            print('loss_tracker_val: ', loss_tracker_val[-2])
        else:
            print('loss_tracker_val: ', loss_tracker_val[-1])
        print('last_epoch_and_nan: ', last_epoch_and_nan)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(loss_tracker_train, label='train')
        ax.plot(loss_tracker_val, label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        if last_epoch_and_nan[1]:
            ax.axvline(last_epoch_and_nan[0]-1, color='red', linestyle='--') #, label='nan')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(dir_res, 'loss_lr_'+str(lr)+'.png'))
        plt.show()

        warnings.warn('Plotting training results from previous saved run done. Exiting...')
        return

    ##################### End #####################

    return


if __name__ == '__main__':
    main_unit_tangent_ball()
