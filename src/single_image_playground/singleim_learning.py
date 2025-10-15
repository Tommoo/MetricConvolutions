# In this file, we will learn the shape of unit tangent balls on a single image
# Train and test are on the same image, but with two independent noise realizations
# The kernel weights are uniform and not learned
# Only the shape of the kernels are learned
# The goal is to see if our theory can adapt to learning and if we can learn a good representation.
# Beware of overfitting


import torch
import torchvision
import math
import matplotlib.pyplot as plt
import warnings
from unit_tangent_ball_playground import F_randers_batch_v
import os
import pathlib
import torch_lr_finder


class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, im, im_gt):
        super(SingleImageDataset, self).__init__()
        self.im = im.squeeze(0)
        self.im_gt = im_gt.squeeze(0)

    def __getitem__(self, index):
        return self.im, self.im_gt

    def __len__(self):
        return 1


class CustomDeformModelForLrFinder(torch.nn.Module):
    def __init__(self, offsets, k, sample_centre, dilation, ker):
        super(CustomDeformModelForLrFinder, self).__init__()
        self.offsets = torch.nn.Parameter(offsets)
        self.k = k
        self.sample_centre = sample_centre
        self.dilation = dilation
        self.weight = ker

    def forward(self, input):
        im_blur_deformed = torchvision.ops.deform_conv2d(
            input, offset=self.offsets, weight=self.weight, dilation=self.dilation, padding=self.dilation * (self.k // 2)
        )
        return im_blur_deformed


class CustomRandModelForLrFinder(torch.nn.Module):
    def __init__(self, k, sample_centre, L_params, w, eps, eps_L, eps_w):
        super(CustomRandModelForLrFinder, self).__init__()
        self.k = k
        self.sample_centre = sample_centre
        self.eps = eps
        self.L_params = torch.nn.Parameter(L_params)
        self.w = torch.nn.Parameter(w)
        self.eps_L = eps_L
        self.eps_w = eps_w

    def forward(self, input):
        # Tune L_params for non singularity of L: abs of diagonal and shift by small eps_L
        L_params_tuned = torch.zeros_like(self.L_params)
        L_params_tuned[:, :, :, torch.tensor([0, 2])] = torch.abs(self.L_params[:, :, :, torch.tensor([0, 2])]) + self.eps_L
        L_params_tuned[:, :, :, torch.tensor([1])] = self.L_params[:, :, :, torch.tensor([1])]

        L = torch.cat([L_params_tuned, torch.zeros((*L_params_tuned.shape[:-1], 1)).to(input.device)], dim=-1)[:, :, :,
            torch.tensor([0, 3, 1, 2])].reshape((*L_params_tuned.shape[:3], 2, 2))
        M = (L @ L.permute((0, 1, 2, 4, 3))).permute((0, 3, 4, 1, 2)).reshape((L.shape[0], 4, *L.shape[1:3]))

        M_inv = torch.inverse(
            M.permute((0, 2, 3, 1)).reshape((M.shape[0], *M.shape[-2:], 2, 2)) \
            + self.eps * torch.eye(2).to(M.device).unsqueeze(0).unsqueeze(1).unsqueeze(1).tile((1, *M.shape[-2:], 1, 1))
        )  # batch, rows, cols, 2, 2

        # Tune w for norm < 1-eps_w
        norm_w_Minv = torch.sqrt( \
            self.w.permute(0, 2, 3, 1).unsqueeze(-2) \
            @ M_inv
            @ self.w.permute((0, 2, 3, 1)).unsqueeze(-1)
            + self.eps  # at 0 we get nan in the gradient due to sqrt
        ).squeeze(-1).permute((0, 3, 1, 2))
        norm_w_Minv_tuned = (torch.sigmoid(norm_w_Minv) - 1 / 2) * 2 * (1 - self.eps_w)
        w_tuned = self.w * norm_w_Minv_tuned / (norm_w_Minv + self.eps)

        im_blur_unit_tangent_ball = blur_randers_ball_tangent(input, M, w_tuned, eps=self.eps, kh=self.k, kw=self.k,
                                                              sample_centre=self.sample_centre)

        return im_blur_unit_tangent_ball


def learn_deform_single_im_fixed_weights(im, im_gt, im_test=None, weight=None, dilation=1, lr=1e8, n_iter=10):

    device = im.device

    if weight is None:
        kw, kh = 11, 11
        weight = torch.ones(*im.shape[:2], kw, kh).to(device)
        weight = weight / weight.sum((-2,-1), keepdim=True)

    kw, kh = weight.shape[-2:]

    weight.requires_grad = False
    offsets = torch.zeros(1, 2*1*kh*kw, *im.shape[-2:]).to(device)
    offsets.requires_grad = True

    optimizer = torch.optim.SGD([offsets], lr=lr)

    im_blur_deformed = torch.zeros_like(im)

    nb_loss_tracked = 1 if im_test is None else 2
    loss_tracker = torch.zeros(n_iter, nb_loss_tracked)

    for iter in range(n_iter):
        print('iter: ', iter, end=' | ')
        optimizer.zero_grad()
        im_blur_deformed = torchvision.ops.deform_conv2d(im, offset=offsets, weight=weight, dilation=dilation, padding=dilation*(kw//2))
        loss = torch.mean((im_blur_deformed - im_gt) ** 2)
        loss.backward()
        optimizer.step()

        loss_tracker[iter, 0] = loss.item()

        print('loss: ', loss.item(), end=' | ')

        if im_test is not None:
            with torch.no_grad():
                im_blur_deformed_test = torchvision.ops.deform_conv2d(im_test, offset=offsets, weight=weight, dilation=dilation, padding=dilation*(kw//2))
                loss_test = torch.mean((im_blur_deformed_test - im_gt) ** 2)
                loss_tracker[iter, 1] = loss_test.item()
                print('loss test: ', loss_test.item(), end='')
        print()

    return im_blur_deformed.detach(), offsets.detach(), loss_tracker


def init_randers_metric(im, direc='ortho', anisotropic_scale=100, eps=1e-6, tau=0.):

    if not direc in [None, 'grad', 'ortho']:
        raise NotImplementedError('Only isotropic, grad, and ortho initialisations are implemented for now')

    device = im.device

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], requires_grad=False).float().unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], requires_grad=False).float().unsqueeze(0).unsqueeze(0).to(device)
    # Blur im with sobel_x and sobel_y kernels and then stack the results into im_grad variable
    im_grad_x = torch.nn.functional.conv2d(im, sobel_x, padding='same')
    im_grad_y = torch.nn.functional.conv2d(im, sobel_y, padding='same')
    im_grad = torch.cat([im_grad_x, im_grad_y], dim=1)

    if direc in [None, 'grad', 'ortho']:
        M = torch.eye(2)  # Per channel then needs deform_groups>1, otherwise deform_groups=1
        M = M / 10  # 10
        # Eig val of M should be < 1 to have a unit ball bigger than 1 pixel due to inversion in scaling
        M = M.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat((1, 1, 1, *im.shape[-2:]))
    else:
        raise NotImplementedError('Only isotropic, grad, and ortho initialisations are implemented for now')

    M = M.to(device)

    if direc in ['grad', 'ortho']:
        if direc == 'grad':
            i_scale = 0
        elif direc == 'ortho':
            i_scale = 1
        norm_grad = torch.sqrt(torch.sum(im_grad ** 2, dim=1))
        norm_grad_normed = norm_grad / norm_grad.max()
        M[0, i_scale, i_scale, :, :] = M[0, i_scale, i_scale, :, :] / (1 + anisotropic_scale * norm_grad_normed)
        M[0, int(1 - i_scale), int(1 - i_scale), :, :] = M[0, int(1 - i_scale), int(1 - i_scale), :, :] * (1 + anisotropic_scale * norm_grad_normed)
        R_grad = torch.stack([torch.cat([im_grad_x + eps, im_grad_y + eps], dim=1),
                              torch.cat([-im_grad_y - eps, im_grad_x + eps], dim=1)], dim=2).to(device)
        R_grad = R_grad / (norm_grad.unsqueeze(1).unsqueeze(1) + eps)
        M = (R_grad.permute((0, 3, 4, 1, 2)) @ M.permute((0, 3, 4, 1, 2)) @ R_grad.permute((0, 3, 4, 2, 1))).permute((0, 3, 4, 1, 2))
    M = torch.reshape(M, (M.shape[0], 4, *M.shape[-2:]))
    # w should be as large as possible for maximum anisotropic deformation, with norm_M-1(w) < 1
    if tau != 0:
        w = im_grad / torch.sqrt((im_grad ** 2).sum(1)).max()  # Along image gradient
        if direc == 'grad':
            pass
        elif direc == 'ortho':
            w_ortho = torch.zeros_like(w)
            w_ortho[:, 0, :, :] = -w[:, 1, :, :]
            w_ortho[:, 1, :, :] = w[:, 0, :, :]
            w = w_ortho  # Along orthogonal of image gradient
        M_inv = torch.inverse(torch.reshape(M, (M.shape[0], 2, 2, *M.shape[-2:])).permute((0, 3, 4, 1, 2)))  # batch, rows, cols, 2, 2
        norm_M_inv_w = torch.einsum('brcj,brcij,brci->brc', w.permute((0, 2, 3, 1)), M_inv, w.permute(0, 2, 3, 1)) # batch, rows, cols
        w = w * tau / (torch.sqrt(norm_M_inv_w.unsqueeze(1)) + eps)
    else:
        w = torch.zeros(im.shape[0], 2, *im.shape[-2:]).to(device)

    # M and w use the x,y convention!

    return M, w


def blur_randers_ball_tangent(im, M, w, eps=1e-6, kh=5, kw=5, sample_centre=False, ker=None, ker_samp_ctr=None):

    # M: batch, 4, rows, cols
    # w: batch, 2, rows, cols

    # M and w use the x,y convention!

    device = im.device

    # Naive polar sparse grid sampling strategy
    n_theta = math.ceil(math.sqrt(kh * kw))
    # Number of samples: n_theta ** 2 + 1. We can add 1 extra point for the centre
    theta = torch.arange(0, 2*torch.pi-eps, 2*torch.pi / n_theta).to(device)
    u_theta = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    u_theta = u_theta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    u_theta = torch.tile(u_theta, (1,1,1,*im.shape[-2:]))
    u_theta = torch.tile(u_theta, (im.shape[0], 1, 1, 1, 1))

    # Compute y_theta (not just boundary - sparse grid sampling strategy)
    F_randers_u_theta = F_randers_batch_v(u_theta, M, w)
    if F_randers_u_theta.min() < 0:
        warnings.warn('Warning: negative F_randers_u_theta')
    y_theta = (1 / (F_randers_u_theta + eps)).unsqueeze(2) * u_theta

    y_s_theta = y_theta.unsqueeze(1).repeat(1, n_theta, 1, 1, 1, 1)
    s_interp = torch.arange(0, 1-eps, 1/n_theta).to(device) + 1/n_theta  # +eps to make sure we don't hit exactly the right value, as recommended by pytorch doc
    y_s_theta = (torch.permute(y_s_theta, (0, 2, 3, 4, 5, 1)) * s_interp).permute(0, 5, 1, 2, 3, 4)  # multiplication with last dimension of same size
    # We can implement the custom interpolation using the built-in deform_conv2d by computing offsets
    kernel_grid = torch.stack(
        torch.meshgrid(torch.arange(-(n_theta//2), n_theta//2+1),
                       torch.arange(-(n_theta//2), n_theta//2+1), indexing='ij'),
        dim=-1).float().to(device)  # i,j ordering, shape is kh, kw, 2
    offsets_ball = (y_s_theta.permute((0,4,5,1,2,3)).flip(-1) - kernel_grid).permute((0,3,4,5,1,2))  # Uses i,j ordering
    # Correct ordering is khkw2 of deform_conv2d, but the official doc misleads into thinking 2khkw
    # ordering = 'khkw2'
    offsets_ball = torch.reshape(offsets_ball, (offsets_ball.shape[0], -1, *offsets_ball.shape[-2:]))  # 1 deform_group


    if ker is None:
        if sample_centre and ker_samp_ctr is None:
            normalisation_ker_ball = (n_theta**2) + 1  # +1 for centre
            ker_ball_centre = torch.ones_like(im) / normalisation_ker_ball
        elif sample_centre:
            ker_ball_centre = ker_samp_ctr
        else:
            normalisation_ker_ball = (n_theta**2)
            ker_ball_centre = torch.zeros_like(im)  # should be scalar 0, but changes nothing here (weights are shared)
        ker_ball = torch.ones(1, 1, n_theta, n_theta) / normalisation_ker_ball
        ker_ball = ker_ball.to(device)
        ker_ball.requires_grad = False
    else:
        ker_ball = ker
        if sample_centre and ker_samp_ctr is None:
            normalisation_ker_ball = (n_theta**2) + 1  # +1 for centre
            ker_ball_centre = torch.ones_like(im) / normalisation_ker_ball
        elif sample_centre:
            ker_ball_centre = ker_samp_ctr
        else:
            ker_ball_centre = torch.zeros_like(im)  # should be scalar 0, but changes nothing here (weights are shared)
    im_blur_deformed_ball = torchvision.ops.deform_conv2d(im, offset=offsets_ball, weight=ker_ball, dilation=1, padding=n_theta//2)
    im_blur_deformed_ball = im_blur_deformed_ball + im * ker_ball_centre
    return im_blur_deformed_ball


def learn_unit_tangent_ball_single_im_fixed_weights(
        im, im_gt, im_test=None, weight=None, direc='ortho', tau=0., anisotropic_scale=10,
        lr=1e8, n_iter=10, sample_centre=False,
        dir_res=None, # For saving
        eps=1e-6,
        eps_L=0.01,  # We will be scaling by 1 / (eps_L ** 2)
        eps_w=0.1  # must be < 1, close to 0 allows asymmetry, close to 1 forces symmetry
):

    device = im.device

    if weight is None:
        kw, kh = 11, 11
        weight = torch.ones(*im.shape[:2], kw, kh).to(device)

    kw, kh = weight.shape[-2:]

    weight.requires_grad = False

    M, w = init_randers_metric(im, direc=direc, anisotropic_scale=anisotropic_scale, eps=eps, tau=tau)
    L = torch.linalg.cholesky(M.permute((0,2,3,1)).reshape((M.shape[0],*M.shape[-2:], 2, 2)), upper=False)
    L_params = L.reshape((*L.shape[:3], 4))[:,:,:, torch.tensor([0, 2, 3])]
    L_params.requires_grad = True
    w.requires_grad = True

    optimizer = torch.optim.SGD([L_params, w], lr=lr)

    im_blur_unit_tangent_ball = torch.zeros_like(im)

    nb_loss_tracked = 1 if im_test is None else 2
    loss_tracker = torch.zeros(n_iter, nb_loss_tracked)

    for iter in range(n_iter):
        print('iter: ', iter, end=' | ')
        optimizer.zero_grad()

        # Tune L_params for non singularity of L: abs of diagonal and shift by small eps_L
        L_params_tuned = torch.zeros_like(L_params)
        L_params_tuned[:, :, :, torch.tensor([0,2])] = torch.abs(L_params[:, :, :, torch.tensor([0, 2])]) + eps_L
        L_params_tuned[:, :, :, torch.tensor([1])] = L_params[:, :, :, torch.tensor([1])]

        L = torch.cat([L_params_tuned, torch.zeros((*L_params_tuned.shape[:-1], 1)).to(device)], dim=-1)[:, :, :, torch.tensor([0, 3, 1, 2])].reshape((*L_params_tuned.shape[:3], 2, 2))
        M = (L @ L.permute((0, 1, 2, 4, 3))).permute((0, 3, 4, 1, 2)).reshape(M.shape)

        M_inv = torch.inverse(
                M.permute((0, 2, 3, 1)).reshape((M.shape[0],*M.shape[-2:], 2, 2)) \
                + eps * torch.eye(2).to(M.device).unsqueeze(0).unsqueeze(1).unsqueeze(1).tile((1, *M.shape[-2:], 1, 1))
            )  # batch, rows, cols, 2, 2

        # Tune w for norm < 1-eps_w
        norm_w_Minv = torch.sqrt( \
            w.permute(0, 2, 3, 1).unsqueeze(-2) \
            @  M_inv
            @ w.permute((0, 2, 3, 1)).unsqueeze(-1)
            + eps  # at 0 we get nan in the gradient due to sqrt
        ).squeeze(-1).permute((0, 3, 1, 2))
        norm_w_Minv_tuned = (torch.sigmoid(norm_w_Minv) - 1/2) * 2 * (1 - eps_w)
        w_tuned = w * norm_w_Minv_tuned / (norm_w_Minv + eps)

        im_blur_unit_tangent_ball = blur_randers_ball_tangent(im, M, w_tuned, eps=eps, kh=kh, kw=kw, sample_centre=sample_centre)

        loss = torch.mean((im_blur_unit_tangent_ball - im_gt) ** 2)
        loss.backward()
        optimizer.step()

        loss_tracker[iter, 0] = loss.item()

        print('loss: ', loss.item(), end=' | ')

        if im_test is not None:
            with torch.no_grad():
                im_blur_unit_tangent_ball_test = blur_randers_ball_tangent(im_test, M, w_tuned, eps=eps, kh=kh, kw=kw, sample_centre=sample_centre)
                loss_test = torch.mean((im_blur_unit_tangent_ball_test - im_gt) ** 2)
                loss_tracker[iter, 1] = loss_test.item()

                if not torch.isnan(loss).any():  # In case there are nan, we save latest non nan
                    torch.save(M.detach().cpu(), os.path.join(dir_res, 'M_lr_'+str(float(lr))+'.pt'))
                    torch.save(w.detach().cpu(), os.path.join(dir_res, 'w_lr_'+str(float(lr))+'.pt'))
                    torch.save(im_blur_unit_tangent_ball.detach().cpu(),
                               os.path.join(dir_res, 'im_blur_unit_tangent_ball_lr_'+str(float(lr))+'.pt'))
                    torch.save(im_blur_unit_tangent_ball_test.detach().cpu(),
                               os.path.join(dir_res, 'im_blur_unit_tangent_ball_other_lr_'+str(float(lr))+'.pt'))
                print('loss test: ', loss_test.item(), end='')

                if torch.isnan(loss).any():
                    break
        print()

    return im_blur_unit_tangent_ball.detach(), (M.detach(), w_tuned.detach()), loss_tracker


def main():

    dev = 'cuda'  # 'cpu' or 'cuda'
    device = torch.device(dev)

    if dev == 'cpu' or not torch.cuda.is_available():
        import warnings
        warnings.warn('Using CPU. Consider using GPU for speedup if learning the convolutions')

    torch.manual_seed(42)
    if dev == 'cuda':
        torch.cuda.manual_seed(42)

    # Load data
    im_gt = torchvision.io.read_image('cameraman.png').to(device)
    sigma = 0.5  # 0.1, 0.3, 0.5

    im_gt = im_gt.float() / 255.0
    im_gt = im_gt.unsqueeze(0)

    im = im_gt + torch.randn_like(im_gt) * sigma
    im_other = im_gt + torch.randn_like(im_gt) * sigma

    psnr = 10 * torch.log10(1 / torch.mean((im - im_gt) ** 2))
    psnr_other = 10 * torch.log10(1 / torch.mean((im_other - im_gt) ** 2))

    eps = 1e-6
    eps_L = 0.01  # We will be scaling by 1 / (eps_L ** 2)
    eps_w = 0.9  # must be < 1      # 0.9, 0.1

    k = 121  # 5, 11, 31, 51, 121  # Size of the convolution. Feel free to play with thie parameter!
    kw, kh = k, k
    ker = torch.ones(1, 1, kh, kw).to(device) / (kw*kh)
    dilation = 1

    # TODO: Learnable weights ?

    # Found best lr for each configuration
    if sigma == 0.1 and k == 5:
        lr_deform = 4.6e6
    elif sigma == 0.1 and k == 11:
        lr_deform = 2.4e7
    elif sigma == 0.1 and k == 31:
        lr_deform = 1.9e8
    elif sigma == 0.1 and k == 51:
        lr_deform = 6.1e8
    elif sigma == 0.1 and k == 121:
        lr_deform = 1.1e9
    elif sigma == 0.3 and k == 5:
        lr_deform = 1.5e6
    elif sigma == 0.3 and k == 11:
        lr_deform = 7.4e6
    elif sigma == 0.3 and k == 31:
        lr_deform = 6.0e7
    elif sigma == 0.3 and k == 51:
        lr_deform = 1.9e8
    elif sigma == 0.3 and k == 121:
        lr_deform = 9.8e8
    elif sigma == 0.5 and k == 5:
        lr_deform = 5.7e5
    elif sigma == 0.5 and k == 11:
        lr_deform = 2.9e6
    elif sigma == 0.5 and k == 31:
        lr_deform = 2.4e7
    elif sigma == 0.5 and k == 51:
        lr_deform = 7.6e7
    elif sigma == 0.5 and k == 121:
        lr_deform = 4.9e8
    else:
        lr_deform = None

    if eps_w == 0.9:
        if sigma == 0.1 and k == 5:
            lr_rand = 3.7e5
        elif sigma == 0.1 and k == 11:
            lr_rand = 4.0e5
        elif sigma == 0.1 and k == 31:
            lr_rand = 4.5e5
        elif sigma == 0.1 and k == 51:
            lr_rand = 5.7e5
        elif sigma == 0.1 and k == 121:
            lr_rand = 5.7e5
        elif sigma == 0.3 and k == 5:
            lr_rand = 1.0e4
        elif sigma == 0.3 and k == 11:
            lr_rand = 1.0e4
        elif sigma == 0.3 and k == 31:
            lr_rand = 1.0e4
        elif sigma == 0.3 and k == 51:
            lr_rand = 1.2e5
        elif sigma == 0.3 and k == 121:
            lr_rand = 1.4e5
        elif sigma == 0.5 and k == 5:
            lr_rand = 1.0e3
        elif sigma == 0.5 and k == 11:
            lr_rand = 5.0e3
        elif sigma == 0.5 and k == 31:
            lr_rand = 1.0e4
        elif sigma == 0.5 and k == 51:
            lr_rand = 1.0e4
        elif sigma == 0.5 and k == 121:
            lr_rand = 1.0e4
        else:
            lr_rand = None
    elif eps_w == 0.1:
        if sigma == 0.1 and k == 5:
            lr_rand = 2.5e5
        elif sigma == 0.1 and k == 11:
            lr_rand = 3.0e5
        elif sigma == 0.1 and k == 31:
            lr_rand = 4.0e5
        elif sigma == 0.1 and k == 51:
            lr_rand = 4.0e5
        elif sigma == 0.1 and k == 121:
            lr_rand = 6.0e5
        elif sigma == 0.3 and k == 5:
            lr_rand = 1.0e4
        elif sigma == 0.3 and k == 11:
            lr_rand = 1.0e4
        elif sigma == 0.3 and k == 31:
            lr_rand = 1.0e4
        elif sigma == 0.3 and k == 51:
            lr_rand = 1.0e4
        elif sigma == 0.3 and k == 121:
            lr_rand = 1.1e5
        elif sigma == 0.5 and k == 5:
            lr_rand = 2.0e3
        elif sigma == 0.5 and k == 11:
            lr_rand = 5.0e3
        elif sigma == 0.5 and k == 31:
            lr_rand = 1.0e4
        elif sigma == 0.5 and k == 51:
            lr_rand = 1.0e4
        elif sigma == 0.5 and k == 121:
            lr_rand = 1.0e4
        else:
            lr_rand = None
    else:
        lr_rand = None

    n_iter_deform = 100  # 100
    n_iter_rand = 100  # 100

    # Initial params of randers metric
    sample_centre = False
    tau = 0.
    direc = 'ortho'  # 'ortho'   # 'ortho', 'grad', None
    anisotropic_scale = 10  # 10

    dir_res_def = os.path.join('res', 'denoising_singleim', 'cameraman_learning', 'sigma_'+str(sigma),'k_'+str(k))
    dir_res_rand = os.path.join('res', 'denoising_singleim', 'cameraman_learning', 'sigma_'+str(sigma),'k_'+str(k),
                                'eps_'+str(eps)+'_eps_L_'+str(eps_L)+'_eps_w_'+str(eps_w))
    pathlib.Path(dir_res_def).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dir_res_rand).mkdir(parents=True, exist_ok=True)

    ##################################### Find lr #####################################
    searching_for_lr = False
    if searching_for_lr:
        warnings.warn('Searching learning rate activated. Full training will not be done')
        ##################################### Run lr finder #####################################
        find_lr = False  # If True, lr_finder is run and no training is done
        if find_lr:
            warnings.warn('Finding learning rate activated. Full training will not be done')

            lr_min = 1e0
            lr_max = 1e10
            num_iter = 100  # number of tested lr values

            method = 'fastai'  # 'smith' or 'fastai'

            dataset_train = SingleImageDataset(im, im_gt)
            dataset_val = SingleImageDataset(im_gt + torch.randn_like(im_gt) * sigma, im_gt)
            dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
            dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

            if method == 'smith':
                step_mode = 'linear'
            elif method == 'fastai':
                step_mode = 'exp'
                dataloader_val = None  # Use only train loss
            else:
                raise ValueError('method not recognized for lr_finder')

            criterion = torch.nn.MSELoss()

            offsets = torch.zeros(1, 2*1*kh*kw, *im.shape[-2:]).to(device)
            offsets.requires_grad = True
            model = CustomDeformModelForLrFinder(offsets, k, sample_centre, dilation, ker)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)  # Min learning rate
            lr_finder = torch_lr_finder.LRFinder(model, optimizer, criterion,
                                                 device=device)
            # Here the lr_finder does only iteration per test. Otherwise use accumulation_steps>1 in lr_finder.range_test
            lr_finder.range_test(dataloader_train, val_loader=dataloader_val, end_lr=lr_max, num_iter=num_iter,
                                 step_mode=step_mode)
            if k == 31 or k == 51 or k == 121:
                # lr_finder saves model and other unnecessary variables. But this becomes expensive for deform_conv2D
                # with high k (2.7GB for k=51, 15GB for k=121). So we remove them in those cases
                lr_finder.optimizer = None
                lr_finder.model = None
                lr_finder.memory_cache = None
                lr_finder.state_cacher = None
            torch.save(lr_finder, os.path.join(dir_res_def, 'lr_finder_deform_' + method + '.pth'))

            M, w = init_randers_metric(im, direc=direc, anisotropic_scale=anisotropic_scale, eps=eps, tau=tau)
            L = torch.linalg.cholesky(M.permute((0, 2, 3, 1)).reshape((M.shape[0], *M.shape[-2:], 2, 2)), upper=False)
            L_params = L.reshape((*L.shape[:3], 4))[:, :, :, torch.tensor([0, 2, 3])]
            L_params.requires_grad = True
            w.requires_grad = True
            model = CustomRandModelForLrFinder(k, sample_centre, L_params, w, eps, eps_L, eps_w)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)  # Min learning rate
            lr_finder = torch_lr_finder.LRFinder(model, optimizer, criterion,
                                                 device=device)
            # Here the lr_finder does only iteration per test. Otherwise use accumulation_steps>1 in lr_finder.range_test
            lr_finder.range_test(dataloader_train, val_loader=dataloader_val, end_lr=lr_max, num_iter=num_iter,
                                 step_mode=step_mode)
            torch.save(lr_finder, os.path.join(dir_res_rand, 'lr_finder_rand_' + method + '.pth'))

            warnings.warn('Finding learning rate activated. Choose an lr and rerun. Exiting...')
            return
        ##################### Viewing lr grid search #####################
        view_find_lr = True
        if view_find_lr:
            warnings.warn('Finding learning rate visually activated. Full training will not be done')

            method = 'fastai'  # 'smith' or 'fastai'

            print('Deform conv2D...')

            lr_finder = torch.load(os.path.join(dir_res_def, 'lr_finder_deform_' + method + '.pth'), map_location=dev)
            fig, ax = plt.subplots()
            ax, lr = lr_finder.plot(ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(dir_res_def, 'lr_finder_deform_' + method + '.png'))
            torch.save(torch.scalar_tensor(lr), os.path.join(dir_res_def, 'lr_suggested_deform_' + method + '.pth'))

            print('Randers UTB...')

            lr_finder = torch.load(os.path.join(dir_res_rand, 'lr_finder_rand_' + method + '.pth'), map_location=dev)
            fig, ax = plt.subplots()
            ax, lr = lr_finder.plot(ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(dir_res_rand, 'lr_finder_rand_' + method + '.png'))
            torch.save(torch.scalar_tensor(lr), os.path.join(dir_res_rand, 'lr_suggested_rand_' + method + '.pth'))

            plt.show()

            warnings.warn('Finding learning rate visually activated. Choose an lr and rerun. Exiting...')
            return


    ##################################### Train: Rerun or load #####################################
    rerun = False
    if not rerun:
        # offsets = torch.load(os.path.join(dir_res_def, 'offsets_lr_'+str(float(lr_deform))+'.pt'), map_location=torch.device(device))
        loss_tracker_deformed = \
            torch.load(
                os.path.join(dir_res_def, 'loss_tracker_deformed_lr_'+str(float(lr_deform))+'.pt'),
                map_location=torch.device(device))
        loss_tracker_unit_tangent_ball = \
            torch.load(
                os.path.join(dir_res_rand, 'loss_tracker_unit_tangent_ball_lr_'+str(float(lr_rand))+'.pt'),
                map_location=torch.device(device))
        if torch.isnan(loss_tracker_unit_tangent_ball).any():
            i_nan = -1
            for i in range(loss_tracker_unit_tangent_ball.shape[0]):
                if torch.isnan(loss_tracker_unit_tangent_ball[i,:]).any():
                    i_nan = i
                    break
            loss_tracker_unit_tangent_ball = loss_tracker_unit_tangent_ball[:i_nan,:]
        M = torch.load(
                os.path.join(dir_res_rand, 'M_lr_'+str(float(lr_rand))+'.pt'),
                map_location=torch.device(device)).detach()
        w = torch.load(
            os.path.join(dir_res_rand, 'w_lr_'+str(float(lr_rand))+'.pt'),
            map_location=torch.device(device)).detach()
        im_blur_deformed = torch.load(
            os.path.join(dir_res_def, 'im_blur_deformed_lr_'+str(float(lr_deform))+'.pt'),
            map_location=torch.device(device))
        im_blur_deformed_other = torch.load(
            os.path.join(dir_res_def, 'im_blur_deformed_other_lr_'+str(float(lr_deform))+'.pt'),
            map_location=torch.device(device))
        im_blur_unit_tangent_ball = torch.load(
            os.path.join(dir_res_rand, 'im_blur_unit_tangent_ball_lr_'+str(float(lr_rand))+'.pt'),
            map_location=torch.device(device))
        im_blur_unit_tangent_ball_other = torch.load(
            os.path.join(dir_res_rand, 'im_blur_unit_tangent_ball_other_lr_'+str(float(lr_rand))+'.pt'),
            map_location=torch.device(device))
    else:
        im_blur_deformed, offsets, loss_tracker_deformed = \
            learn_deform_single_im_fixed_weights(im, im_gt, im_test=im_other,
                                                 weight=ker, dilation=dilation, lr=lr_deform, n_iter=n_iter_deform)
        im_blur_deformed_other = \
            torchvision.ops.deform_conv2d(im_other, offset=offsets, weight=ker, dilation=dilation, padding=dilation*(kw//2))

        im_blur_unit_tangent_ball, (M, w), loss_tracker_unit_tangent_ball = \
            learn_unit_tangent_ball_single_im_fixed_weights(im, im_gt, im_test=im_other, weight=ker, direc=direc, tau=tau,
                                                            anisotropic_scale=anisotropic_scale,
                                                            lr=lr_rand, n_iter=n_iter_rand,
                                                            sample_centre=sample_centre,
                                                            dir_res=dir_res_rand,
                                                            eps=eps, eps_L=eps_L, eps_w=eps_w)
        im_blur_unit_tangent_ball_other = \
            blur_randers_ball_tangent(im_other, M, w, eps=eps, kh=kh, kw=kw, sample_centre=sample_centre)

    psnr_deformed = 10 * torch.log10(1 / torch.mean((im_blur_deformed - im_gt) ** 2))
    psnr_deformed_other = 10 * torch.log10(1 / torch.mean((im_blur_deformed_other - im_gt) ** 2))
    psnr_unit_tangent_ball = 10 * torch.log10(1 / torch.mean((im_blur_unit_tangent_ball - im_gt) ** 2))
    psnr_unit_tangent_ball_other = 10 * torch.log10(1 / torch.mean((im_blur_unit_tangent_ball_other - im_gt) ** 2))

    ##################################### Save results #####################################
    save_results = False
    if save_results:
        torch.save(loss_tracker_deformed,
                   os.path.join(dir_res_def, 'loss_tracker_deformed_lr_'+str(float(lr_deform))+'.pt'))
        torch.save(loss_tracker_unit_tangent_ball,
                   os.path.join(dir_res_rand, 'loss_tracker_unit_tangent_ball_lr_'+str(float(lr_rand))+'.pt'))
        # torch.save(offsets, 'offsets_lr_'+str(float(lr_deform))+'.pt')  # This can be expensive for high k
        torch.save(im_blur_deformed.cpu(),
                   os.path.join(dir_res_def, 'im_blur_deformed_lr_'+str(float(lr_deform))+'.pt'))
        torch.save(im_blur_deformed_other.cpu(),
                   os.path.join(dir_res_def, 'im_blur_deformed_other_lr_'+str(float(lr_deform))+'.pt'))
        if not loss_tracker_unit_tangent_ball.isnan().any():  # Otherwise already been saved
            torch.save(M, os.path.join(dir_res_rand, 'M_lr_'+str(float(lr_rand))+'.pt'))
            torch.save(w, os.path.join(dir_res_rand, 'w_lr_'+str(float(lr_rand))+'.pt'))
            torch.save(im_blur_unit_tangent_ball.cpu(),
                       os.path.join(dir_res_rand, 'im_blur_unit_tangent_ball_lr_'+str(float(lr_rand))+'.pt'))
            torch.save(im_blur_unit_tangent_ball_other.cpu(),
                       os.path.join(dir_res_rand, 'im_blur_unit_tangent_ball_other_lr_'+str(float(lr_rand))+'.pt'))


    ##################################### Plot results #####################################
    plot_results = True
    if plot_results:
        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        ax[0,0].imshow(im_gt.squeeze().cpu(), cmap='gray')
        ax[0,1].imshow(im.squeeze().cpu(), cmap='gray')
        ax[1,1].imshow(im_other.squeeze().cpu(), cmap='gray')
        ax[0,2].imshow(im_blur_deformed.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[1,2].imshow(im_blur_deformed_other.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[0,3].imshow(im_blur_unit_tangent_ball.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[1,3].imshow(im_blur_unit_tangent_ball_other.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[0,0].set_title('GT')
        ax[0,1].set_title("Noisy {:.3f}".format(psnr.item()))
        ax[1,1].set_title("Noisy other {:.3f}".format(psnr_other.item()))
        ax[0,2].set_title('Def {:.3f}'.format(psnr_deformed.item()))
        ax[1,2].set_title('Def {:.3f}'.format(psnr_deformed_other.item()))
        ax[0,3].set_title('UTB {:.3f}'.format(psnr_unit_tangent_ball.item()))
        ax[1,3].set_title('UTB other {:.3f}'.format(psnr_unit_tangent_ball_other.item()))
        ax[1,0].plot(loss_tracker_deformed[:,0].cpu().numpy(), label='Train (Def)')
        ax[1,0].plot(loss_tracker_deformed[:,1].cpu().numpy(), label='Test (Def)')
        ax[1,0].plot(loss_tracker_unit_tangent_ball[:, 0].cpu().numpy(), label='Train (UTB)')
        ax[1,0].plot(loss_tracker_unit_tangent_ball[:, 1].cpu().numpy(), label='Test (UTB)')
        ax[1,0].set_title('Loss')
        ax[1,0].legend()
        for i in range(2):
            for j in range(4):
                if i == 1 and j == 0:
                    continue
                ax[i,j].axis('off')
        plt.tight_layout()
        plt.savefig(
            os.path.join(dir_res_rand,
                         'results_cameraman_learn_deform_vs_tangent'
                         '_lr_def_'+str(float(lr_deform))+'_lr_utb_'+str(float(lr_rand))+'.png'),
            bbox_inches='tight')#, pad_inches=0)

        delta_loss_normalised_deformed = \
            (loss_tracker_deformed[:, 1] - loss_tracker_deformed[:, 0]) / loss_tracker_deformed[:, 0]
        delta_loss_normalised_unit_tangent_ball = \
            ((loss_tracker_unit_tangent_ball[:, 1] - loss_tracker_unit_tangent_ball[:, 0])
             / loss_tracker_unit_tangent_ball[:, 0])

        # print values of trackers at the last epoch
        print('sigma:\t', sigma)
        print('k:\t', k)
        print()
        print('loss_tracker_deformed (Train):\t', loss_tracker_deformed[-1, 0].item())
        print('loss_tracker_deformed (Test):\t', loss_tracker_deformed[-1, 1].item())
        print('delta_loss_normalised_deformed\t', delta_loss_normalised_deformed[-1].item())
        print()
        print('eps_w:\t', eps_w)
        print('\tloss_tracker_unit_tangent_ball (Train):\t', loss_tracker_unit_tangent_ball[-1, 0].item())
        print('\tloss_tracker_unit_tangent_ball (Test):\t', loss_tracker_unit_tangent_ball[-1, 1].item())
        print('\tdelta_loss_normalised_unit_tangent_ball\t', delta_loss_normalised_unit_tangent_ball[-1].item())


        # ALL RESULTS

        if eps_w == 0.1:
            warnings.warn('Do plot and save global results please use eps_w=0.9. Stopping here.')
            return

        if eps_w == 0.1:
            if sigma == 0.1 and k == 5:
                lr_rand_otherw = 3.7e5
            elif sigma == 0.1 and k == 11:
                lr_rand_otherw = 4.0e5
            elif sigma == 0.1 and k == 31:
                lr_rand_otherw = 4.5e5
            elif sigma == 0.1 and k == 51:
                lr_rand_otherw = 5.7e5
            elif sigma == 0.1 and k == 121:
                lr_rand_otherw = 5.7e5
            elif sigma == 0.3 and k == 5:
                lr_rand_otherw = 1.0e4  # 5.6e4
            elif sigma == 0.3 and k == 11:
                lr_rand_otherw = 1.0e4  # 7.1e4
            elif sigma == 0.3 and k == 31:
                lr_rand_otherw = 1.0e4  # 1.1e5
            elif sigma == 0.3 and k == 51:
                lr_rand_otherw = 1.2e5
            elif sigma == 0.3 and k == 121:
                lr_rand_otherw = 1.4e5
            elif sigma == 0.5 and k == 5:
                lr_rand_otherw = 1.0e3  # 1.8e4
            elif sigma == 0.5 and k == 11:
                lr_rand_otherw = 5.0e3  # 2.8e4
            elif sigma == 0.5 and k == 31:
                lr_rand_otherw = 1.0e4  # 4.4e4
            elif sigma == 0.5 and k == 51:
                lr_rand_otherw = 1.0e4  # 4.4e4
            elif sigma == 0.5 and k == 121:
                lr_rand_otherw = 1.0e4  # 7.1e4
            else:
                lr_rand_otherw = None
        elif eps_w == 0.9:
            if sigma == 0.1 and k == 5:
                lr_rand_otherw = 2.5e5
            elif sigma == 0.1 and k == 11:
                lr_rand_otherw = 3.0e5
            elif sigma == 0.1 and k == 31:
                lr_rand_otherw = 4.0e5
            elif sigma == 0.1 and k == 51:
                lr_rand_otherw = 4.0e5
            elif sigma == 0.1 and k == 121:
                lr_rand_otherw = 6.0e5
            elif sigma == 0.3 and k == 5:
                lr_rand_otherw = 1.0e4  # 3.4e4
            elif sigma == 0.3 and k == 11:
                lr_rand_otherw = 1.0e4  # 5.0e4
            elif sigma == 0.3 and k == 31:
                lr_rand_otherw = 1.0e4  # 1.0e5
            elif sigma == 0.3 and k == 51:
                lr_rand_otherw = 1.0e4  # 1.1e5
            elif sigma == 0.3 and k == 121:
                lr_rand_otherw = 1.1e5
            elif sigma == 0.5 and k == 5:
                lr_rand_otherw = 2.0e3  # 1.4e4
            elif sigma == 0.5 and k == 11:
                lr_rand_otherw = 5.0e3  # 2.0e4
            elif sigma == 0.5 and k == 31:
                lr_rand_otherw = 1.0e4  # 3.0e4
            elif sigma == 0.5 and k == 51:
                lr_rand_otherw = 1.0e4  # 2.5e4
            elif sigma == 0.5 and k == 121:
                lr_rand_otherw = 1.0e4  # 5.0e4
            else:
                lr_rand_otherw = None
        else:
            lr_rand_otherw = None

        eps_w_otherw = 0.1 if eps_w == 0.9 else 0.9
        dir_res_rand_otherw = os.path.join('res', 'denoising_singleim', 'cameraman_learning', 'sigma_'+str(sigma),'k_'+str(k),
                                'eps_'+str(eps)+'_eps_L_'+str(eps_L)+'_eps_w_'+str(eps_w_otherw))
        loss_tracker_unit_tangent_ball_otherw = \
            torch.load(
                os.path.join(dir_res_rand_otherw, 'loss_tracker_unit_tangent_ball_lr_'+str(float(lr_rand_otherw))+'.pt'),
                map_location=torch.device(device))
        if torch.isnan(loss_tracker_unit_tangent_ball_otherw).any():
            i_nan = -1
            for i in range(loss_tracker_unit_tangent_ball_otherw.shape[0]):
                if torch.isnan(loss_tracker_unit_tangent_ball_otherw[i,:]).any():
                    i_nan = i
                    break
            loss_tracker_unit_tangent_ball_otherw = loss_tracker_unit_tangent_ball_otherw[:i_nan, :]
        im_blur_unit_tangent_ball_otherw = torch.load(
            os.path.join(dir_res_rand_otherw, 'im_blur_unit_tangent_ball_lr_'+str(float(lr_rand_otherw))+'.pt'),
            map_location=torch.device(device))
        im_blur_unit_tangent_ball_other_otherw = torch.load(
            os.path.join(dir_res_rand_otherw, 'im_blur_unit_tangent_ball_other_lr_'+str(float(lr_rand_otherw))+'.pt'),
            map_location=torch.device(device))

        psnr_unit_tangent_ball_otherw = 10 * torch.log10(1 / torch.mean((im_blur_unit_tangent_ball_otherw - im_gt) ** 2))
        psnr_unit_tangent_ball_other_otherw = 10 * torch.log10(1 / torch.mean((im_blur_unit_tangent_ball_other_otherw - im_gt) ** 2))





        fig, ax = plt.subplots(2, 5, figsize=(20, 8))
        ax[0,0].imshow(im_gt.squeeze().cpu(), cmap='gray')
        ax[0,1].imshow(im.squeeze().cpu(), cmap='gray')
        ax[1,1].imshow(im_other.squeeze().cpu(), cmap='gray')
        ax[0,2].imshow(im_blur_deformed.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[1,2].imshow(im_blur_deformed_other.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[0,3].imshow(im_blur_unit_tangent_ball.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[1,3].imshow(im_blur_unit_tangent_ball_other.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[0,4].imshow(im_blur_unit_tangent_ball_otherw.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[1,4].imshow(im_blur_unit_tangent_ball_other_otherw.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[0,0].set_title('GT')
        ax[0,1].set_title("Noisy train {:.3f}".format(psnr.item()))
        ax[1,1].set_title("Noisy test {:.3f}".format(psnr_other.item()))
        ax[0,2].set_title('Def train {:.3f}'.format(psnr_deformed.item()))
        ax[1,2].set_title('Def test {:.3f}'.format(psnr_deformed_other.item()))
        ax[0,3].set_title('UTB train (eps_w='+str(eps_w)+') {:.3f}'.format(psnr_unit_tangent_ball.item()))
        ax[1,3].set_title('UTB test (eps_w='+str(eps_w)+') {:.3f}'.format(psnr_unit_tangent_ball_other.item()))
        ax[0,4].set_title('UTB train (eps_w='+str(eps_w_otherw)+') {:.3f}'.format(psnr_unit_tangent_ball_otherw.item()))
        ax[1,4].set_title('UTB test (eps_w='+str(eps_w_otherw)+') {:.3f}'.format(psnr_unit_tangent_ball_other_otherw.item()))
        ax[1,0].plot(loss_tracker_deformed[:,0].cpu().numpy(), label='Train (Def)')
        ax[1,0].plot(loss_tracker_deformed[:,1].cpu().numpy(), label='Test (Def)')
        ax[1,0].plot(loss_tracker_unit_tangent_ball[:, 0].cpu().numpy(), label='Train (UTB) eps_w='+str(eps_w), linestyle='--')
        ax[1,0].plot(loss_tracker_unit_tangent_ball[:, 1].cpu().numpy(), label='Test (UTB) eps_w='+str(eps_w), linestyle='--')
        ax[1,0].plot(loss_tracker_unit_tangent_ball_otherw[:, 0].cpu().numpy(), label='Train (UTB) eps_w='+str(eps_w_otherw), linestyle='-.')
        ax[1,0].plot(loss_tracker_unit_tangent_ball_otherw[:, 1].cpu().numpy(), label='Test (UTB ) eps_w='+str(eps_w_otherw), linestyle='-.')
        ax[1,0].set_title('Loss')
        ax[1,0].legend()
        for i in range(2):
            for j in range(5):
                if i == 1 and j == 0:
                    continue
                ax[i,j].axis('off')
        plt.tight_layout()
        dir_res_ = os.path.join('res', 'denoising_singleim', 'cameraman_learning', 'sigma_'+str(sigma),'k_'+str(k))
        plt.savefig(
            os.path.join(dir_res_,
                         'results_cameraman_learn_deform_vs_tangent'
                         + '_sigma_'+str(sigma)+'_k_'+str(k) + '_eps_w_'+str(eps_w)+'_eps_w_otherw_'+str(eps_w_otherw)
                         + '.png'),
            bbox_inches='tight')#, pad_inches=0)

    if dev == 'cpu':
        plt.show()


if __name__ == '__main__':
    main()