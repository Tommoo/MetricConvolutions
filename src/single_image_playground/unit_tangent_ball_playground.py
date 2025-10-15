
# Here we consider tangential unit balls


import torch
import torchvision
import math
import matplotlib.pyplot as plt
import warnings


def F_randers(v, M, w):
    # ASSUMING deform_groups = 1  (same metric for each channel at a given pixel)
    # M = batch, 4, rows, cols
    # v = batch, 2, rows, cols
    # w = batch, 2, rows, cols

    # TODO: Would have been wiser to change convention in the ordering, and then batch matrix multiplication could have
    #  simply been done with @ symbol instead of einsum. Would have to rewrite the whole code though...

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


def blur_randers_ball_tangent(im, M=None, w=None, tau=None, eps=1e-6, kh=5, kw=5, direc='grad', sample_centre=True):

    if not direc in ['grad', 'ortho']:
        raise ValueError('direc must be grad or ortho')

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], requires_grad=False).float().unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], requires_grad=False).float().unsqueeze(0).unsqueeze(0)
    # In demo figure we use kernel of 5 (otherwise would need to get 1 pixel away from the edge and it works but it
    # harms the visualisation for the paper)
    # sobel_x = torch.tensor([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]], requires_grad=False).float().unsqueeze(0).unsqueeze(0)
    # sobel_y = torch.tensor([[-2, -2, -4, -2, -2], [-1, -1, -2, -1, -1], [0, 0, 0, 0, 0], [1, 1, 2, 1, 1], [2, 2, 4, 2, 2]], requires_grad=False).float().unsqueeze(0).unsqueeze(0)

    # Blur im with sobel_x and sobel_y kernels and then stack the results into im_grad variable
    im_grad_x = torch.nn.functional.conv2d(im, sobel_x, padding='same')
    im_grad_y = torch.nn.functional.conv2d(im, sobel_y, padding='same')
    im_grad = torch.cat([im_grad_x, im_grad_y], dim=1)
    if M is None:
        M = torch.eye(2)  # Per channel then needs deform_groups>1, otherwise deform_groups=1
        M = M / 10  # Eig val of M should be < 1 to have a unit ball bigger than 1 pixel due to inversion in scaling
        M = M.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat((1, 1, 1, *im.shape[-2:]))
    elif M in ['grad','grad_orth']:
        if M == 'grad':
            i_scale = 0
        elif M == 'grad_orth':
            i_scale = 1
        else:
            raise ValueError('M must be None, grad or grad_orth')
        anisotropic_scale = 100  # 10 in demo figure
        M = torch.eye(2)
        M = M / 10
        M = M.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat(1, 1, 1, *im.shape[-2:])
        norm_grad = torch.sqrt(torch.sum(im_grad ** 2, dim=1))
        norm_grad_normed = norm_grad / norm_grad.max()
        M[0, i_scale, i_scale, :, :] = M[0, i_scale, i_scale, :, :] / (1 + anisotropic_scale * norm_grad_normed)
        M[0, int(1 - i_scale), int(1 - i_scale), :, :] = M[0, int(1 - i_scale), int(1 - i_scale), :, :] * (1 + anisotropic_scale * norm_grad_normed)
        R_grad = torch.stack([torch.cat([im_grad_x + eps, im_grad_y + eps], dim=1),
                              torch.cat([-im_grad_y - eps, im_grad_x + eps], dim=1)], dim=2)
        R_grad = R_grad / (norm_grad.unsqueeze(1).unsqueeze(1) + eps)
        M = (R_grad.permute((0,3,4,1,2)) @ M.permute((0,3,4,1,2)) @ R_grad.permute((0,3,4,2,1))).permute((0,3,4,1,2))
    M_flat = torch.reshape(M, (M.shape[0], 4, *M.shape[-2:]))
    if w is None:
        w = im_grad / torch.sqrt((im_grad ** 2).sum(1)).max()  # Along image gradient
        if direc == 'grad':
            pass
        elif direc == 'ortho':
            w_ortho = torch.zeros_like(w)
            w_ortho[:, 0, :, :] = -w[:, 1, :, :]
            w_ortho[:, 1, :, :] = w[:, 0, :, :]
            w = w_ortho  # Along orthogonal of image gradient
        # tau < 1 / sqrt(norm_M-1(w')) is the condition for positivity of F_randers
        M_inv = torch.inverse(torch.reshape(M_flat, (M_flat.shape[0], 2, 2, *M_flat.shape[-2:])).permute((0, 3, 4, 1, 2)))  # batch, rows, cols, 2, 2
        norm_M_inv_w = torch.einsum('brcj,brcij,brci->brc', w.permute((0, 2, 3, 1)), M_inv, w.permute(0, 2, 3, 1)) # batch, rows, cols
        w = w * tau / (torch.sqrt(norm_M_inv_w.unsqueeze(1)) + eps)

        # w should be as large as possible for maximum anisotropic deformation, with norm_M-1(w) < 1

    # M and w use the x,y convention!

    # Naive polar grid sampling instead of better onion peeling strategy
    n_theta = math.ceil(math.sqrt(kh * kw))
    # Number of samples: n_theta ** 2 + 1. We add 1 extra point for the centre, but not necessary (old design)
    theta = torch.arange(0, 2*torch.pi-eps, 2*torch.pi / n_theta)
    u_theta = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    u_theta = u_theta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    u_theta = torch.tile(u_theta, (1,1,1,*im.shape[-2:]))
    u_theta = torch.tile(u_theta, (im.shape[0], 1, 1, 1,1))

    # Compute y_theta (not just boundary - sparse grid sampling strategy)
    F_randers_u_theta = F_randers_batch_v(u_theta, M_flat, w)
    if F_randers_u_theta.min() < 0:
        warnings.warn('Warning: negative F_randers_u_theta')
    y_theta = (1 / (F_randers_u_theta + eps)).unsqueeze(2) * u_theta

    y_s_theta = y_theta.unsqueeze(1).repeat(1, n_theta, 1, 1, 1, 1)
    s_interp = torch.arange(0, 1-eps, 1/n_theta) + 1/n_theta  # +eps to make sure we don't hit exactly the right value, as recommended by pytorch doc
    y_s_theta = (torch.permute(y_s_theta, (0, 2, 3, 4, 5, 1)) * s_interp).permute(0, 5, 1, 2, 3, 4)  # multiplication with last dimension of same size
    # We can implement the custom interpolation using the built-in deform_conv2d by computing offsets
    kernel_grid = torch.stack(
        torch.meshgrid(torch.arange(-(n_theta//2), n_theta//2+1),
                       torch.arange(-(n_theta//2), n_theta//2+1), indexing='ij'),
        dim=-1).float()  # i,j ordering, shape is kh, kw, 2
    offsets_ball = (y_s_theta.permute((0,4,5,1,2,3)).flip(-1) - kernel_grid).permute((0,3,4,5,1,2))  # Uses i,j ordering
    # Correct ordering is khkw2 of deform_conv2d, but the official doc misleads into thinking 2khkw
    ordering = 'khkw2'  # Correct ordering
    # ordering = '2khkw'  # Incorrect ordering
    if ordering == 'khkw2':
        pass
    elif ordering == '2khkw':
        offsets_ball = torch.permute(offsets_ball, (0, 3, 1, 2, 4, 5))
    offsets_ball = torch.reshape(offsets_ball, (offsets_ball.shape[0], -1, *offsets_ball.shape[-2:]))  # 1 deform_group

    # Plot kernel positions
    if ordering == '2khkw':  # Wrong ordering
        ker_pos = torch.reshape(offsets_ball, (offsets_ball.shape[0], 2, kh, kw, *im.shape[-2:])).permute((0, 2, 3, 1, 4, 5))
    elif ordering == 'khkw2':  # Correct ordering
        ker_pos = torch.reshape(offsets_ball, (offsets_ball.shape[0], kh, kw, 2, *im.shape[-2:])).permute((0, 1, 2, 3, 4, 5))
    pix_i_debug, pix_j_debug = 83, 50  # 36, 112 | 84, 50 | 192, 178 | 52, 222 || (83, 50 | 35, 111 with 5x5 sobel)
    ker_pos = ker_pos[0, :, :, :, pix_i_debug, pix_j_debug] + kernel_grid
    ker_pos_x, ker_pos_y = ker_pos[:,:,1].flatten() + pix_j_debug, ker_pos[:,:,0].flatten() + pix_i_debug
    plt.imshow(im[0].squeeze().cpu().numpy(), cmap='gray')

    t = torch.linspace(0, 2 * torch.pi, 1000)
    u_t = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    u_t = u_t.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    u_t = torch.tile(u_t, (1, 1, 1, *im.shape[-2:]))
    u_t = torch.tile(u_t, (im.shape[0], 1, 1, 1, 1))
    F_randers_u_t = F_randers_batch_v(u_t, M_flat, w)
    y_t = (1 / (F_randers_u_t + eps)).unsqueeze(2) * u_t
    y_t_debug = y_t.squeeze()[:, :, pix_i_debug, pix_j_debug]
    plt.plot(y_t_debug[:, 0] + pix_j_debug, y_t_debug[:, 1] + pix_i_debug, label='{:.1f}'.format(1-tau),
             linewidth=5)  #eps_w = 1-tau
    plt.scatter(pix_j_debug, pix_i_debug, marker='+', color='red')

    #plt.scatter(ker_pos_x, ker_pos_y)#, label=str(tau))
    plt.axis('equal')
    plt.axis('off')
    plt.legend(fontsize=18)

    if sample_centre:
        normalisation_ker_ball = (n_theta**2) + 1  # +1 for centre
        ker_ball_centre = torch.ones_like(im) / normalisation_ker_ball
    else:
        normalisation_ker_ball = (n_theta**2)
        ker_ball_centre = torch.zeros_like(im)
    ker_ball = torch.ones(1, 1, n_theta, n_theta) / normalisation_ker_ball
    im_blur_deformed_ball = torchvision.ops.deform_conv2d(im, offset=offsets_ball, weight=ker_ball, dilation=1, padding=n_theta//2)
    im_blur_deformed_ball = im_blur_deformed_ball + im * ker_ball_centre
    return im_blur_deformed_ball


def main():

    torch.manual_seed(42)

    # Load data
    im_gt = torchvision.io.read_image('cameraman.png')
    #im_gt = torchvision.transforms.Resize((30,30))(im_gt.cpu()).to(im_gt.device)  # TODO: Remove this line
    sigma = 0.1

    im_gt = im_gt.float() / 255.0
    im_gt = im_gt.unsqueeze(0)

    im = im_gt + torch.randn_like(im_gt) * sigma

    psnr = 10 * torch.log10(1 / torch.mean((im - im_gt) ** 2))

    eps = 1e-6
    kw, kh = 11, 11
    # kw, kh = 5, 5
    ker = torch.ones(1, 1, kh, kw) / (kw*kh)
    dilation = 3
    dilation_def = 1
    # offsets = torch.rand(1, 2*1*kh*kw, *im.shape[-2:]) * (dilation * kw * 2) - (dilation * kw)
    offsets = torch.rand(1, 2*1*kh*kw, *im.shape[-2:]) * (dilation_def * kw * 2) - (dilation_def * kw)

    # Test with interpolation of kw,kh grid inside square of size nrxnr. Interpolation removes noise (2 conv and not just 1)
    nr = 5
    offsets_centered_nrxnr_khxkw = \
        (torch.stack(
            torch.meshgrid((nr//2) * torch.arange(-(kh//2), kh//2+1) / (kh//2), (nr//2) * torch.arange(-(kw//2), kw//2+1) / (kw//2), indexing='ij'),
        dim=-1).float())
    kernel_grid_nrxnr_khxkw = \
        (torch.stack(
            torch.meshgrid(torch.arange(-(kh//2), kh//2+1), torch.arange(-(kw//2), kw//2+1), indexing='ij'),
        dim=-1).float())  # i,j ordering, shape is kh, kw, 2
    offsets_nrxnr_khxkw = offsets_centered_nrxnr_khxkw - kernel_grid_nrxnr_khxkw
    offsets_nrxnr_khxkw = torch.reshape(offsets_nrxnr_khxkw, (1, 2*1*kh*kw))
    offsets_nrxnr_khxkw = offsets_nrxnr_khxkw.unsqueeze(-1).unsqueeze(-1).tile(1, 1, *im.shape[-2:])

    im_blur = torch.nn.functional.conv2d(im, ker, padding='same')
    im_blur_dilated = torch.nn.functional.conv2d(im, ker, padding='same', dilation=dilation)
    im_blur_deformed = torchvision.ops.deform_conv2d(im, offset=offsets, weight=ker, dilation=1, padding=kw//2)

    im_blur_nrxnr_kwxkh = torchvision.ops.deform_conv2d(im, offset=offsets_nrxnr_khxkw, weight=ker, dilation=1, padding=kw//2)

    psnr_blur = 10 * torch.log10(1 / torch.mean((im_blur - im_gt) ** 2))
    psnr_blur_dilated = 10 * torch.log10(1 / torch.mean((im_blur_dilated - im_gt) ** 2))
    psnr_blur_deformed = 10 * torch.log10(1 / torch.mean((im_blur_deformed - im_gt) ** 2))

    psnr_blur_nrxnr_kwxkh = 10 * torch.log10(1 / torch.mean((im_blur_nrxnr_kwxkh - im_gt) ** 2))

    plt.figure()
    im_blur_deformed_ball_0 = blur_randers_ball_tangent(im, tau=0., eps=eps, kh=kh, kw=kw)
    im_blur_deformed_ball_05 = blur_randers_ball_tangent(im, tau=0.5, eps=eps, kh=kh, kw=kw)
    im_blur_deformed_ball_09 = blur_randers_ball_tangent(im, tau=0.9, eps=eps, kh=kh, kw=kw)
    plt.figure()
    im_blur_deformed_ball_0_ortho = blur_randers_ball_tangent(im, tau=0., eps=eps, kh=kh, kw=kw, direc='ortho')
    im_blur_deformed_ball_05_ortho = blur_randers_ball_tangent(im, tau=0.5, eps=eps, kh=kh, kw=kw, direc='ortho')
    im_blur_deformed_ball_09_ortho = blur_randers_ball_tangent(im, tau=0.9, eps=eps, kh=kh, kw=kw, direc='ortho')

    psnr_blur_deformed_ball_0 = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_0 - im_gt) ** 2))
    psnr_blur_deformed_ball_05 = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_05 - im_gt) ** 2))
    psnr_blur_deformed_ball_09 = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_09 - im_gt) ** 2))
    psnr_blur_deformed_ball_0_ortho = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_0_ortho - im_gt) ** 2))
    psnr_blur_deformed_ball_05_ortho = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_05_ortho - im_gt) ** 2))
    psnr_blur_deformed_ball_09_ortho = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_09_ortho - im_gt) ** 2))

    plt.figure()
    im_blur_deformed_ball_0_Mg = blur_randers_ball_tangent(im, M='grad', tau=0., eps=eps, kh=kh, kw=kw)
    im_blur_deformed_ball_05_Mg = blur_randers_ball_tangent(im, M='grad', tau=0.5, eps=eps, kh=kh, kw=kw)
    im_blur_deformed_ball_09_Mg = blur_randers_ball_tangent(im, M='grad', tau=0.9, eps=eps, kh=kh, kw=kw)
    plt.figure()
    im_blur_deformed_ball_0_ortho_Mg = blur_randers_ball_tangent(im, M='grad', tau=0., eps=eps, kh=kh, kw=kw,
                                                                 direc='ortho')
    im_blur_deformed_ball_05_ortho_Mg = blur_randers_ball_tangent(im, M='grad', tau=0.5, eps=eps, kh=kh, kw=kw,
                                                                  direc='ortho')
    im_blur_deformed_ball_09_ortho_Mg = blur_randers_ball_tangent(im, M='grad', tau=0.9, eps=eps, kh=kh, kw=kw,
                                                                  direc='ortho')

    psnr_blur_deformed_ball_0_Mg = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_0_Mg - im_gt) ** 2))
    psnr_blur_deformed_ball_05_Mg = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_05_Mg - im_gt) ** 2))
    psnr_blur_deformed_ball_09_Mg = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_09_Mg - im_gt) ** 2))
    psnr_blur_deformed_ball_0_ortho_Mg = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_0_ortho_Mg - im_gt) ** 2))
    psnr_blur_deformed_ball_05_ortho_Mg = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_05_ortho_Mg - im_gt) ** 2))
    psnr_blur_deformed_ball_09_ortho_Mg = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_09_ortho_Mg - im_gt) ** 2))

    plt.figure()
    im_blur_deformed_ball_0_Mgo = blur_randers_ball_tangent(im, M='grad_orth', tau=0., eps=eps, kh=kh, kw=kw)
    im_blur_deformed_ball_05_Mgo = blur_randers_ball_tangent(im, M='grad_orth', tau=0.5, eps=eps, kh=kh, kw=kw)
    im_blur_deformed_ball_09_Mgo = blur_randers_ball_tangent(im, M='grad_orth', tau=0.9, eps=eps, kh=kh, kw=kw)
    plt.figure()
    im_blur_deformed_ball_0_ortho_Mgo = blur_randers_ball_tangent(im, M='grad_orth', tau=0., eps=eps, kh=kh, kw=kw,
                                                                  direc='ortho')
    im_blur_deformed_ball_05_ortho_Mgo = blur_randers_ball_tangent(im, M='grad_orth', tau=0.5, eps=eps, kh=kh, kw=kw,
                                                                   direc='ortho')
    im_blur_deformed_ball_09_ortho_Mgo = blur_randers_ball_tangent(im, M='grad_orth', tau=0.9, eps=eps, kh=kh, kw=kw,
                                                                   direc='ortho')

    psnr_blur_deformed_ball_0_Mgo = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_0_Mgo - im_gt) ** 2))
    psnr_blur_deformed_ball_05_Mgo = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_05_Mgo - im_gt) ** 2))
    psnr_blur_deformed_ball_09_Mgo = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_09_Mgo - im_gt) ** 2))
    psnr_blur_deformed_ball_0_ortho_Mgo = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_0_ortho_Mgo - im_gt) ** 2))
    psnr_blur_deformed_ball_05_ortho_Mgo = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_05_ortho_Mgo - im_gt) ** 2))
    psnr_blur_deformed_ball_09_ortho_Mgo = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_09_ortho_Mgo - im_gt) ** 2))


    im_blur_deformed_ball_0_Mg_geodesic = torchvision.io.read_image('cameraman_conv_unit_geodesic_grad_0_Mg_newton.png')[0].unsqueeze(0) / 255  # Produced by geodesic unit ball
    im_blur_deformed_ball_05_Mg_geodesic = torchvision.io.read_image('cameraman_conv_unit_geodesic_grad_05_Mg_newton.png')[0].unsqueeze(0) / 255  # Produced by geodesic unit ball
    im_blur_deformed_ball_09_Mg_geodesic = torchvision.io.read_image('cameraman_conv_unit_geodesic_grad_09_Mg_newton.png')[0].unsqueeze(0) / 255  # Produced by geodesic unit ball
    # for im_blur in [im_blur_deformed_ball_0_Mg_geodesic, im_blur_deformed_ball_05_Mg_geodesic, im_blur_deformed_ball_09_Mg_geodesic]:
    #     im_blur.data = torchvision.transforms.Resize((30,30))(im_blur.data)
    psnr_blur_deformed_ball_0_Mg_geodesic = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_0_Mg_geodesic - im_gt) ** 2))
    psnr_blur_deformed_ball_05_Mg_geodesic = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_05_Mg_geodesic - im_gt) ** 2))
    psnr_blur_deformed_ball_09_Mg_geodesic = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_09_Mg_geodesic - im_gt) ** 2))


    # Write code to plot im, im_blur, im_blur_dilated, im_blur_deformed in matplotlib subplots
    fig, axs = plt.subplots(7, 7, figsize=(10, 10))
    axs[0,0].imshow(im.squeeze().numpy(), cmap='gray')
    axs[0,1].imshow(im_blur.squeeze().numpy(), cmap='gray')
    axs[0,2].imshow(im_blur_dilated.squeeze().numpy(), cmap='gray')
    axs[0,3].imshow(im_blur_deformed.squeeze().numpy(), cmap='gray')
    axs[0,4].imshow(im_blur_deformed_ball_0.squeeze().numpy(), cmap='gray')
    axs[0,5].imshow(im_blur_deformed_ball_05.squeeze().numpy(), cmap='gray')
    axs[0,6].imshow(im_blur_deformed_ball_09.squeeze().numpy(), cmap='gray')
    axs[1,1].imshow(im_blur_nrxnr_kwxkh.squeeze().numpy(), cmap='gray')
    axs[1,4].imshow(im_blur_deformed_ball_0_ortho.squeeze().numpy(), cmap='gray')
    axs[1,5].imshow(im_blur_deformed_ball_05_ortho.squeeze().numpy(), cmap='gray')
    axs[1,6].imshow(im_blur_deformed_ball_09_ortho.squeeze().numpy(), cmap='gray')
    axs[2,4].imshow(im_blur_deformed_ball_0_Mg.squeeze().numpy(), cmap='gray')
    axs[2,5].imshow(im_blur_deformed_ball_05_Mg.squeeze().numpy(), cmap='gray')
    axs[2,6].imshow(im_blur_deformed_ball_09_Mg.squeeze().numpy(), cmap='gray')
    axs[3,4].imshow(im_blur_deformed_ball_0_ortho_Mg.squeeze().numpy(), cmap='gray')
    axs[3,5].imshow(im_blur_deformed_ball_05_ortho_Mg.squeeze().numpy(), cmap='gray')
    axs[3,6].imshow(im_blur_deformed_ball_09_ortho_Mg.squeeze().numpy(), cmap='gray')
    axs[4,4].imshow(im_blur_deformed_ball_0_Mgo.squeeze().numpy(), cmap='gray')
    axs[4,5].imshow(im_blur_deformed_ball_05_Mgo.squeeze().numpy(), cmap='gray')
    axs[4,6].imshow(im_blur_deformed_ball_09_Mgo.squeeze().numpy(), cmap='gray')
    axs[5,4].imshow(im_blur_deformed_ball_0_ortho_Mgo.squeeze().numpy(), cmap='gray')
    axs[5,5].imshow(im_blur_deformed_ball_05_ortho_Mgo.squeeze().numpy(), cmap='gray')
    axs[5,6].imshow(im_blur_deformed_ball_09_ortho_Mgo.squeeze().numpy(), cmap='gray')
    axs[6,4].imshow(im_blur_deformed_ball_0_Mg_geodesic.squeeze().numpy(), cmap='gray')
    axs[6,5].imshow(im_blur_deformed_ball_05_Mg_geodesic.squeeze().numpy(), cmap='gray')
    axs[6,6].imshow(im_blur_deformed_ball_09_Mg_geodesic.squeeze().numpy(), cmap='gray')
    axs[0,0].set_title("{:.3f}".format(psnr.item()))
    axs[0,1].set_title("{:.3f}".format(psnr_blur.item()))
    axs[0,2].set_title("{:.3f}".format(psnr_blur_dilated.item()))
    axs[0,3].set_title("{:.3f}".format(psnr_blur_deformed.item()))
    axs[0,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0.item()))
    axs[0,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05.item()))
    axs[0,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09.item()))
    axs[1,1].set_title("{:.3f}".format(psnr_blur_nrxnr_kwxkh.item()))
    axs[1,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0_ortho.item()))
    axs[1,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05_ortho.item()))
    axs[1,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09_ortho.item()))
    axs[2,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0_Mg.item()))
    axs[2,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05_Mg.item()))
    axs[2,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09_Mg.item()))
    axs[3,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0_ortho_Mg.item()))
    axs[3,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05_ortho_Mg.item()))
    axs[3,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09_ortho_Mg.item()))
    axs[4,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0_Mgo.item()))
    axs[4,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05_Mgo.item()))
    axs[4,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09_Mgo.item()))
    axs[5,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0_ortho_Mgo.item()))
    axs[5,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05_ortho_Mgo.item()))
    axs[5,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09_ortho_Mgo.item()))
    axs[6,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0_Mg_geodesic.item()))
    axs[6,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05_Mg_geodesic.item()))
    axs[6,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09_Mg_geodesic.item()))
    for ii in range(7):
        for jj in range(7):
            axs[ii,jj].axis('off')
    plt.tight_layout()




    fig, axs = plt.subplots(2, 7, figsize=(10, 3.5))
    axs[0,0].imshow(im.squeeze().numpy(), cmap='gray')
    axs[0,1].imshow(im_blur.squeeze().numpy(), cmap='gray')
    axs[0,2].imshow(im_blur_dilated.squeeze().numpy(), cmap='gray')
    axs[0,3].imshow(im_blur_deformed.squeeze().numpy(), cmap='gray')
    axs[0,4].imshow(im_blur_deformed_ball_0_ortho_Mgo.squeeze().numpy(), cmap='gray')
    axs[0,5].imshow(im_blur_deformed_ball_05_ortho_Mgo.squeeze().numpy(), cmap='gray')
    axs[0,6].imshow(im_blur_deformed_ball_09_ortho_Mgo.squeeze().numpy(), cmap='gray')
    axs[1,1].imshow(im_blur_nrxnr_kwxkh.squeeze().numpy(), cmap='gray')
    axs[1,4].imshow(im_blur_deformed_ball_0_Mg_geodesic.squeeze().numpy(), cmap='gray')
    axs[1,5].imshow(im_blur_deformed_ball_05_Mg_geodesic.squeeze().numpy(), cmap='gray')
    axs[1,6].imshow(im_blur_deformed_ball_09_Mg_geodesic.squeeze().numpy(), cmap='gray')
    axs[0,0].set_title("{:.3f}".format(psnr.item()))
    axs[0,1].set_title("{:.3f}".format(psnr_blur.item()))
    axs[0,2].set_title("{:.3f}".format(psnr_blur_dilated.item()))
    axs[0,3].set_title("{:.3f}".format(psnr_blur_deformed.item()))
    axs[0,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0_ortho_Mgo.item()))
    axs[0,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05_ortho_Mgo.item()))
    axs[0,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09_ortho_Mgo.item()))
    axs[1,1].set_title("{:.3f}".format(psnr_blur_nrxnr_kwxkh.item()))
    axs[1,4].set_title("{:.3f}".format(psnr_blur_deformed_ball_0_Mg_geodesic.item()))
    axs[1,5].set_title("{:.3f}".format(psnr_blur_deformed_ball_05_Mg_geodesic.item()))
    axs[1,6].set_title("{:.3f}".format(psnr_blur_deformed_ball_09_Mg_geodesic.item()))
    for ii in range(2):
        for jj in range(7):
            axs[ii,jj].axis('off')
    plt.tight_layout()
    fig.savefig('cameraman_blur_no_learn_res.png', bbox_inches='tight')


    fig, axs = plt.subplots(1,1)
    axs.imshow(im.squeeze().numpy(), cmap='gray')
    axs.scatter([112], [36], label='36_112')
    axs.scatter([50], [84], label='84_50')
    axs.scatter([178], [192], label='192_178')
    axs.scatter([222], [52], label='52_222')
    axs.legend()
    axs.axis('off')

    plt.show()





    return

if __name__ == '__main__':
    main()
    plt.show()
