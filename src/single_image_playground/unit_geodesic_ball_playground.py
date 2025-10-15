
import torch
import torchvision
import matplotlib.pyplot as plt
from unit_tangent_ball_playground import F_randers_batch_v
import math
import warnings
import time


def blur_randers_ball_geodesic(im, M=None, w=None, tau=None, eps=1e-6, kh=5, kw=5, direc='grad', sample_centre=True):

    if not direc in ['grad', 'ortho', 'minus_ortho']:
        raise ValueError('direc must be grad or ortho')

    device = im.device

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], requires_grad=False).float().unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], requires_grad=False).float().unsqueeze(0).unsqueeze(0).to(device)

    # Blur im with sobel_x and sobel_y kernels and then stack the results into im_grad variable
    im_grad_x = torch.nn.functional.conv2d(im, sobel_x, padding='same')
    im_grad_y = torch.nn.functional.conv2d(im, sobel_y, padding='same')
    im_grad = torch.cat([im_grad_x, im_grad_y], dim=1)
    print('Computing M...')
    if M is None:
        M = torch.eye(2).to(device)  # Per channel then needs deform_groups>1, otherwise deform_groups=1
        M = M / 10  # Eig val of M should be < 1 to have a unit ball bigger than 1 pixel due to inversion in scaling
        M = M.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat((1, 1, 1, *im.shape[-2:]))
    elif M in ['grad','grad_orth']:
        if M == 'grad':
            i_scale = 0
        elif M == 'grad_orth':
            i_scale = 1
        else:
            raise ValueError('M must be None, grad or grad_orth')
        anisotropic_scale = 10  # 10
        M = torch.eye(2).to(device)
        M = M #/ 1  # iota = 1 | iota = 1 in figure full res
        M = M.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat(1, 1, 1, *im.shape[-2:])
        norm_grad = torch.sqrt(torch.sum(im_grad ** 2, dim=1))
        norm_grad_normed = norm_grad / norm_grad.max()
        M[0, i_scale, i_scale, :, :] = M[0, i_scale, i_scale, :, :] / (1 + anisotropic_scale * norm_grad_normed)
        M[0, int(1-i_scale), int(1-i_scale), :, :] = M[0, int(1-i_scale), int(1-i_scale), :, :] * (1 + anisotropic_scale * norm_grad_normed)
        R_grad = torch.stack([torch.cat([im_grad_x + eps, im_grad_y + eps], dim=1),torch.cat([-im_grad_y - eps, im_grad_x + eps], dim=1)], dim=2)
        R_grad = R_grad / (norm_grad.unsqueeze(1).unsqueeze(1) + eps)
        M = (R_grad.permute((0,3,4,1,2)) @ M.permute((0,3,4,1,2)) @ R_grad.permute((0,3,4,2,1))).permute((0,3,4,1,2))
    M_flat = torch.reshape(M, (M.shape[0], 4, *M.shape[-2:]))
    print('Computing w...')
    if w is None:
        w = im_grad / torch.sqrt((im_grad ** 2).sum(1)).max()  # Along image gradient
        if direc == 'grad':
            pass
        elif direc == 'ortho' or 'minus_ortho':
            w_ortho = torch.zeros_like(w)
            w_ortho[:, 0, :, :] = -w[:, 1, :, :]
            w_ortho[:, 1, :, :] = w[:, 0, :, :]
            w = w_ortho  # Along orthogonal of image gradient
            if direc == 'minus_ortho':
                w = -w
        # tau < 1 / sqrt(norm_M-1(w')) is the condition for positivity of F_randers
        M_inv = torch.inverse(torch.reshape(M_flat, (M_flat.shape[0], 2, 2, *M_flat.shape[-2:])).permute((0, 3, 4, 1, 2)))  # batch, rows, cols, 2, 2
        norm_M_inv_w = torch.einsum('brcj,brcij,brci->brc', w.permute((0, 2, 3, 1)), M_inv, w.permute(0, 2, 3, 1)) # batch, rows, cols
        w = w * tau / (torch.sqrt(norm_M_inv_w.unsqueeze(1)) + eps)

        # w should be as large as possible for maximum anisotropic deformation, with norm_M-1(w) < 1

    print('Computing dual...')
    norm_M_inv_w = torch.einsum('brcj,brcij,brci->brc', w.permute((0, 2, 3, 1)), M_inv, w.permute(0, 2, 3, 1))  # batch, rows, cols
    alpha = 1 - norm_M_inv_w
    M_inv_w = M_inv @ w.permute((0, 2, 3, 1)).unsqueeze(-1)  # batch, rows, cols, 2, 1
    M_star = (1/((alpha.unsqueeze(-1).unsqueeze(-1) ** 2) + eps)) * (alpha.unsqueeze(-1).unsqueeze(-1) * M_inv + M_inv_w @ M_inv_w.permute((0, 1, 2, 4, 3))) # batch, rows, cols, 2, 2
    w_star = - (1/(alpha.unsqueeze(-1).unsqueeze(-1) + eps)) * M_inv_w  # batch, rows, cols, 2
    M_star = M_star.permute((0, 3, 4, 1, 2)) # batch, 2, 2, rows, cols
    M_star = M_star.reshape((M_star.shape[0], 4, *M_star.shape[-2:])) # batch, 4, rows, cols
    w_star = w_star.squeeze(-1).permute((0, 3, 1, 2)) # batch, 2, rows, cols

    print('Computing F*...')

    delta_t = 0.1  # 0.01 | 0.1 in figure full res
    t_heat = 0.5  # 0.1 | 0.5 in figure full res
    grid_size = 11  # 11
    scale_init = 2.  # 2 | 2. in figure full res
    delta_t_stencils = 0.1  # 0.01 | 0.1 in figure full res
    t_stencils = 1.  # 0.1 | 1. in figure full res

    v_grid = torch.stack(
        torch.meshgrid(
            torch.arange(-(grid_size // 2), (grid_size // 2)+1),
            torch.arange(-(grid_size // 2), (grid_size // 2)+1)), dim=2).reshape((-1, 2)).float().to(device) # n, 2
    v_grid = v_grid.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).tile((1, 1, 1, *im.shape[-2:]))  # 1, n, 2, rows, cols
    F_star_randers_grid = F_randers_batch_v(v_grid, M_star, w_star)
    # F_star_heat_ker = (1/(delta_t)) * torch.exp(-F_star_randers_grid / (4 * delta_t)) # 1, n, rows, cols  # BUG: This line is incorrect. Paper results obtained with this line
    F_star_heat_ker = (1/(delta_t)) * torch.exp(-(F_star_randers_grid ** 2) / (4 * delta_t)) # 1, n, rows, cols  # BUG: This line is correct. Paper results not obtained with this line
    F_star_heat_ker = F_star_heat_ker / torch.sum(F_star_heat_ker, dim=1)  # Normalise properly

    n_theta = math.ceil(math.sqrt(kh * kw))
    if sample_centre:
        normalisation_ker_ball = (n_theta**2) + 1  # +1 for centre
        ker_ball_centre = 1. / normalisation_ker_ball
    else:
        normalisation_ker_ball = (n_theta**2)
        ker_ball_centre = 0.
    ker_ball = torch.ones(1, 1, n_theta, n_theta).to(device) / normalisation_ker_ball

    print('Starting pixel loop...')

    t = time.time()

    # If pix debug are not None, then we only look at what is going on at those pixels and plot UGB figures
    pix_i_debug, pix_j_debug = None, None  # None, None | 36, 112 | 83, 50 for whole res ugb figures
    if pix_i_debug is not None and pix_j_debug is not None:
        print('Debugging pixel: ', pix_i_debug, pix_j_debug)

    im_blur_unit_ball_manifold = torch.zeros_like(im)
    for pix_i in range(im.shape[-2]):
        print(pix_i)

        if pix_i_debug is not None and not pix_i == pix_i_debug:
            continue

        for pix_j in range(im.shape[-1]):

            if pix_j_debug is not None and not pix_j == pix_j_debug:
                continue

            if pix_i == pix_i_debug and pix_j == pix_j_debug:
                print('here')

            im_dirac = torch.zeros_like(im)
            im_dirac[:, :, pix_i, pix_j] = 1

            im_diffused = im_dirac

            im_diffused = im_diffused.permute((0,1,3,2)) # x,y convention, not i,j

            t_curr = 0
            for t_iter in range(int(math.floor(t_heat / delta_t))):
                im_unfolded = torch.nn.functional.unfold(im_diffused, kernel_size=(grid_size, grid_size), padding=grid_size//2, stride=1, dilation=1)   # batch, grid_size**2, rows*cols
                im_unfolded_diffused = im_unfolded * F_star_heat_ker.reshape((*F_star_heat_ker.shape[:2], -1)) # batch, grid_size**2, rows*cols
                im_diffused = torch.sum(im_unfolded_diffused, dim=1).reshape((-1, *im.shape[1:])) # batch, 1, rows, cols
                if torch.isnan(im_diffused).any():  # t_iter=43
                    print('here')
                t_curr += delta_t

            im_diffused = im_diffused.permute((0,1,3,2)) # i,j convention, not x,y

            im_diffused_grad = torch.cat(
                [torch.nn.functional.conv2d(im_diffused, sobel_x, padding='same'),
                 torch.nn.functional.conv2d(im_diffused, sobel_y, padding='same')], dim=1)

            im_diffused_min_grad_normed = - im_diffused_grad / (torch.sqrt(torch.sum(im_diffused_grad ** 2, dim=1, keepdim=True)) + eps)

            if pix_i == pix_i_debug and pix_j == pix_j_debug:
                fig, axs = plt.subplots(1, 4)
                axs[0].imshow(im_diffused.squeeze().cpu(), cmap='gray')
                axs[1].imshow(im_diffused_min_grad_normed.squeeze()[0].cpu(), cmap='gray')
                axs[2].imshow(im_diffused_min_grad_normed.squeeze()[1].cpu(), cmap='gray')
                axs[3].imshow((torch.sum(im_diffused_min_grad_normed ** 2, dim=1).squeeze().cpu() > 1e-4) * im.squeeze().cpu(), cmap='gray')
                axs[0].scatter([pix_j], [pix_i], color='r')
                axs[1].scatter([pix_j], [pix_i], color='r')
                axs[2].scatter([pix_j], [pix_i], color='r')
                axs[3].scatter([pix_j], [pix_i], color='r')
                fig, axs = plt.subplots(1,1)
                axs.imshow(im.squeeze().cpu())
                axs.quiver(im_diffused_min_grad_normed[0,0].cpu(), im_diffused_min_grad_normed[0,1].cpu(), angles='xy')
                axs.scatter([pix_j], [pix_i], color='r')

            # Manually diffuse unit ball stencil
            # Start with Randers unit ball stencil
            # Scale it to be a less than unit ball stencil
            # Diffuse it for a fix amount of time (t_heat?)

            if pix_i_debug is not None and pix_j_debug is not None:
                print('Computing final dense unit ball only. If you do not want this please remove this section of code')
                n_theta = 1000

            # Get unit tangent ball stencil
            # Number of samples: n_theta ** 2 (+ 1). We add 1 extra point for the centre
            theta = torch.arange(0, 2 * torch.pi - eps, 2 * torch.pi / n_theta).to(device)
            u_theta = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
            u_theta = u_theta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            # Compute y_theta (not just boundary - sparse grid sampling strategy)
            F_randers_u_theta = F_randers_batch_v(u_theta, M_flat[:,:,pix_i,pix_j].unsqueeze(-1).unsqueeze(-1), w[:,:,pix_i,pix_j].unsqueeze(-1).unsqueeze(-1))
            if F_randers_u_theta.min() < 0:
                warnings.warn('Warning: negative F_randers_u_theta')
            y_theta = (1 / (F_randers_u_theta + eps)).unsqueeze(2) * u_theta

            y_s_theta = y_theta.unsqueeze(1).repeat(1, n_theta, 1, 1, 1, 1)
            s_interp = torch.arange(0, 1 - eps, 1 / n_theta).to(device) + 1 / n_theta  # +eps to make sure we don't hit exactly the right value, as recommended by pytorch doc
            y_s_theta = (torch.permute(y_s_theta, (0, 2, 3, 4, 5, 1)) * s_interp).permute(0, 5, 1, 2, 3, 4)  # multiplication with last dimension of same size
            # y_s_theta: batch, n_theta, n_theta, 2, 1, 1

            # Scale unit tangent ball stencil to be less than unit ball stencil
            y_s_theta = scale_init * y_s_theta

            # Centre scaled tangent ball stencil to pixel position
            y_s_theta = y_s_theta.squeeze(-1).squeeze(-1).flip(-1) + torch.tensor([pix_i, pix_j]).to(device)  # i,j convention, not x,y; batch, n_theta, n_theta, 2

            im_size = torch.tensor(im.shape[-2:]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)  # i,j convention, not x,y
            im_size_half_xy = im_size.flip(-1) / 2  # x,y convention, not i,j

            if pix_i == pix_i_debug and pix_j == pix_j_debug:
                y_s_theta_all = torch.zeros_like(y_s_theta).unsqueeze(1).tile((1, int(math.floor(t_stencils / delta_t_stencils)) + 1, 1, 1, 1))
                y_s_theta_all[:, 0, :, :, :] = y_s_theta
                iter = 0

            # Diffuse tangent ball stencil to manifold ball stencil
            t_curr = 0
            for t_iter in range(int(math.floor(t_stencils / delta_t_stencils))):
                grad_interpolated = torch.nn.functional.grid_sample(
                    im_diffused_min_grad_normed,
                    (y_s_theta.flip(-1) - im_size_half_xy)  /
                    im_size_half_xy,
                    mode='bilinear', align_corners=True
                )  # x,y convention, not i,j
                y_s_theta = y_s_theta + delta_t_stencils * grad_interpolated.permute((0, 2, 3, 1)).flip(-1)
                t_curr += delta_t_stencils
                if pix_i == pix_i_debug and pix_j == pix_j_debug:
                    iter += 1
                    print(iter, y_s_theta_all.shape[1], int(math.floor(t_stencils / delta_t_stencils)) + 1)
                    y_s_theta_all[:,iter,:,:,:] = y_s_theta

            if pix_i == pix_i_debug and pix_j == pix_j_debug:

                # Prints intermediate stencils. Ok if not showing continuous balls, i.e. n_theta not changed to 1000
                # fig, axs = plt.subplots(1, 1)
                # axs.imshow(im.squeeze().cpu(), cmap='gray')
                # for ii in range(y_s_theta_all.shape[1]):
                #     if not ii % 5 == 0:
                #         continue
                #     axs.scatter(y_s_theta_all[0,ii,:,:,1].flatten().cpu(), y_s_theta_all[0,ii,:,:,0].flatten().cpu(), label=str(ii*delta_t))
                # axs.scatter([pix_j], [pix_i], color='r')
                # axs.legend()
                # plt.tight_layout()

                # If we want to show all the geodesic dists in the unit ball
                # fig_final, axs_final = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[10,1]))
                # axs_final[0].imshow(im.squeeze().cpu(), cmap='gray')
                # y_s_theta_all_final = y_s_theta_all[:, -1, :, :, :].squeeze()
                # cm = matplotlib.colormaps['inferno'].resampled(len(s_interp))
                # bn = matplotlib.colors.BoundaryNorm(s_interp.cpu(), len(s_interp))
                # for s_idx, s in enumerate(s_interp):
                #     print('s_idx...', s_idx)
                #     axs_final[0].plot(y_s_theta_all_final[s_idx, :, 1].cpu(), y_s_theta_all_final[s_idx, :, 0].cpu(), color=cm(s_idx))
                # axs_final[0].scatter([pix_j], [pix_i], color='r', marker='+', zorder=1000000)  # zorder to bring to front
                # axs_final[0].axis('off')
                # fig_final.colorbar(matplotlib.cm.ScalarMappable(norm=bn, cmap=cm), cax=axs_final[1])
                # plt.tight_layout()

                # Only the unit circle
                # fig_final_circle, axs_final_circle = plt.subplots(1, 1)
                axs_final_circle.imshow(im.squeeze().cpu(), cmap='gray')
                y_s_theta_all_final = y_s_theta_all[:, -1, :, :, :].squeeze()
                y_circle_final = y_s_theta_all_final[-1, :, :]
                axs_final_circle.plot(y_circle_final[:, 1].cpu(), y_circle_final[:, 0].cpu(),
                                      label='{:.1f}'.format(1-tau), linewidth=5)
                axs_final_circle.scatter([pix_j], [pix_i], color='r', marker='+')  # zorder to bring to front
                axs_final_circle.legend(fontsize=18)#, loc='lower right')
                axs_final_circle.axis('off')
                plt.tight_layout()

            if pix_i == pix_i_debug and pix_j == pix_j_debug:
                # We changed n_theta to 1000 for debugging or plotting purposes
                normalisation_ker_ball = (n_theta ** 2) + 1  if sample_centre else (n_theta ** 2)
                ker_ball = torch.ones(1, 1, n_theta, n_theta).to(device) / normalisation_ker_ball
                warnings.warn('We are debugging and convolving only at a single pixel. The psnr score is not relevant!')

            # Sample from manifold ball stencil
            im_blur_unit_ball_manifold[:, :, pix_i, pix_j] = torch.sum(torch.nn.functional.grid_sample(
                im,
                (y_s_theta.flip(-1) - im_size_half_xy) / im_size_half_xy,
                mode='bilinear', align_corners=True
            ) * ker_ball) + im[:, :, pix_i, pix_j] * ker_ball_centre

        print(time.time() - t)
        t = time.time()

    return im_blur_unit_ball_manifold


def main():

    # For reproducing UGB plots of figure 5 in the paper, please adapt the code above based on the comments, e.g. use
    # hyperparameters mentioned by figure full res. They correspond to what is written in the supplementary material.

    dev = 'cuda'  # 'cpu' or 'cuda'
    device = torch.device(dev)

    if dev == 'cpu' or not torch.cuda.is_available():
        import warnings
        warnings.warn('Using CPU. This will be slow. If you have a GPU, please set dev = "cuda"')

    torch.manual_seed(42)

    import matplotlib
    matplotlib.use('Qt5Agg')

    # Load data
    im_gt = torchvision.io.read_image('cameraman.png').to(device)

    sigma = 0.1

    im_gt = im_gt.float() / 255.0
    im_gt = im_gt.unsqueeze(0)

    im = im_gt + torch.randn_like(im_gt) * sigma

    psnr = 10 * torch.log10(1 / torch.mean((im - im_gt) ** 2))

    eps = 1e-6
    kw, kh = 11, 11

    global fig_final_circle, axs_final_circle  # For plotting superimposed unit circles. Remove if undesired
    fig_final_circle, axs_final_circle = plt.subplots(1, 1)

    direc = 'ortho'  # 'ortho'
    M = 'grad_orth'  # 'grad_orth'

    im_blur_deformed_ball_0_Mg = blur_randers_ball_geodesic(im, M=M, tau=0., eps=eps, kh=kh, kw=kw, direc=direc)
    psnr_blur = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_0_Mg - im_gt) ** 2))
    print('PSNR 0: ', psnr_blur.item())
    torchvision.utils.save_image(im_blur_deformed_ball_0_Mg.squeeze().cpu(),
                                 'cameraman_conv_unit_geodesic_'+direc+'_0_M_'+M+'.png'
                                 )

    im_blur_deformed_ball_05_Mg = blur_randers_ball_geodesic(im, M=M, tau=0.5, eps=eps, kh=kh, kw=kw, direc=direc)
    psnr_blur = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_05_Mg - im_gt) ** 2))
    print('PSNR 05: ', psnr_blur.item())
    torchvision.utils.save_image(im_blur_deformed_ball_05_Mg.squeeze().cpu(),
                                 'cameraman_conv_unit_geodesic_' + direc + '_05_M_'+M+'.png'
                                 )

    im_blur_deformed_ball_09_Mg = blur_randers_ball_geodesic(im, M=M, tau=0.9, eps=eps, kh=kh, kw=kw, direc=direc)
    psnr_blur = 10 * torch.log10(1 / torch.mean((im_blur_deformed_ball_09_Mg - im_gt) ** 2))
    print('PSNR 09: ', psnr_blur.item())
    torchvision.utils.save_image(im_blur_deformed_ball_09_Mg.squeeze().cpu(),
                                 'cameraman_conv_unit_geodesic_' + direc + '_09_M_'+M+'.png'
                                 )


if __name__ == '__main__':
    main()
    plt.show(block=False)
    plt.show()