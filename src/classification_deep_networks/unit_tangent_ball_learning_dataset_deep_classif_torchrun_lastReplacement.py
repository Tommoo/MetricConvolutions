
# Migrate code to torchrun and ddp multi-gpu single node training

import torch
import torchvision
import os
import pathlib
import warnings
import argparse

from utils_torchrun import (sample_unit_ball_tangent, compute_offsets_ball, adapt_model_classifier,
                            none_or_int, none_or_float, none_or_str, import_model_architecture, prepare_dataset,
                            get_dataset_params, train_cnn_classif_several_epochs, WeightHook)


def ddp_setup():
    torch.distributed.init_process_group(backend='nccl')


def blur_randers_ball_tangent(im, M, w, eps=1e-6, kh=5, kw=5, ker=None, bias=None, stride=1, dilation=1,
                              sampling_strat='polar_grid'):
    # TODO: implement modulation ?
    # TODO: implement other sampling (like standard conv sampling)

    # In deformable convolution (and in the original paper), the deformation is the same accross all channels.

    # M: batch, 4, rows, cols
    # w: batch, 2, rows, cols
    # M and w use the x,y convention!

    device = im.device

    y_s_theta = sample_unit_ball_tangent(im, M, w, eps=eps, kh=kh, kw=kw, sampling_strat=sampling_strat)
    offsets_ball = compute_offsets_ball(y_s_theta, kh, kw, dilation, device)
    im_blur_deformed_ball = torchvision.ops.deform_conv2d(
        im, offset=offsets_ball, weight=ker,
        stride=stride, dilation=dilation, padding=int(dilation*(kh//2)),
        bias=bias
    )
    return im_blur_deformed_ball


class ComputeMAndwFromMw(torch.nn.Module):
    def __init__(self, intermediate_strat_code, strat_w_code, eps_L=0.01, eps=1e-6, eps_w=1.0, scale_max=2., scale_min=0.1):
        super().__init__()
        self.intermediate_strat_code = intermediate_strat_code
        self.strat_w_code = strat_w_code
        self.eps_L = eps_L
        self.eps = eps
        self.eps_w = eps_w
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.avg_scale = (scale_max + scale_min) / 2
        self.half_scale_range = (scale_max - scale_min) / 2

    def forward(self, Mw, inp):

        device = inp.device

        if self.intermediate_strat_code == 0:  # LL^T
            # Mw = intermediate_Mw(inp)  # batch, 5, rows, cols
            L_params = Mw[:, :3, :, :].permute((0, 2, 3, 1))  # batch, rows, cols, 3
            w = Mw[:, -2:, :, :]  # batch, 2, rows, cols

            # Tune L_params for non singularity of L: abs of diagonal and shift by small eps_L
            L_params_tuned = torch.zeros_like(L_params)
            L_params_tuned[:, :, :, torch.tensor([0, 2])] = torch.abs(L_params[:, :, :, torch.tensor([0, 2])]) + self.eps_L
            L_params_tuned[:, :, :, torch.tensor([1])] = L_params[:, :, :, torch.tensor([1])]

            L = torch.cat([L_params_tuned, torch.zeros((*L_params_tuned.shape[:-1], 1)).to(device)], dim=-1)[:, :, :,
                torch.tensor([0, 3, 1, 2])].reshape((*L_params_tuned.shape[:3], 2, 2))

            M = (L @ L.permute((0, 1, 2, 4, 3))).permute((0, 3, 4, 1, 2)).reshape((L.shape[0], 4, *L.shape[1:3]))
        elif self.intermediate_strat_code == 1:  # LDL^T
            # Mw = intermediate_Mw(inp)  # batch, 6, rows, cols
            L_params = Mw[:, :1, :, :].permute((0, 2, 3, 1))  # batch, rows, cols, 1
            L_params = torch.cat([L_params, torch.zeros((*L_params.shape[:-1], 3)).to(device)], dim=-1)
            L_params = L_params[:, :, :, torch.tensor([1, 2, 0, 3])]
            L_params = L_params.reshape((*L_params.shape[:3], 2, 2))  # batch, rows, cols, 2, 2
            L_params = L_params + torch.eye(2).to(L_params.device).unsqueeze(0).unsqueeze(1).unsqueeze(1)
            D_params = Mw[:, 1:3, :, :].permute((0, 2, 3, 1))  # batch, rows, cols, 2
            D_params_tuned = torch.abs(D_params) + self.eps_L
            D_params_tuned = torch.cat([D_params_tuned, torch.zeros((*D_params_tuned.shape[:-1], 2)).to(device)], dim=-1)
            D_params_tuned = D_params_tuned[:, :, :, torch.tensor([0, 2, 3, 1])]
            D_params_tuned = D_params_tuned.reshape((*D_params_tuned.shape[:3], 2, 2))  # batch, rows, cols, 2, 2
            M = (L_params @ D_params_tuned @ L_params.permute((0, 1, 2, 4, 3))).permute((0, 3, 4, 1, 2)).reshape((L_params.shape[0], 4, *L_params.shape[1:3]))
            w = Mw[:, -2:, :, :]  # batch, 2, rows, cols
        elif self.intermediate_strat_code in [2, 3, 4]:  # eigvec eigval
            # Mw = intermediate_Mw(inp)  # batch, 6, rows, cols
            eigvec_params = Mw[:, :2, :, :].permute((0, 2, 3, 1))  # batch, rows, cols, 2
            # normalise
            eigvec_params = (eigvec_params + self.eps) / (torch.norm(eigvec_params, dim=-1, keepdim=True) + self.eps)  # +eps in num since for 0 gives still 0
            eigvec_params_orth = torch.zeros_like(eigvec_params)
            eigvec_params_orth[:, :, :, 0] = -eigvec_params[:, :, :, 1]
            eigvec_params_orth[:, :, :, 1] = eigvec_params[:, :, :, 0]
            eigvec_params = torch.cat([eigvec_params, eigvec_params_orth], dim=-1)
            eigvec_params = eigvec_params[:, :, :, torch.tensor([0, 2, 1, 3])].reshape((*eigvec_params.shape[:3], 2, 2))  # batch, rows, cols, 2, 2
            if self.intermediate_strat_code == 2:  # eigvec eigval
                eigval_params = Mw[:, 2:4, :, :].permute((0, 2, 3, 1))  # batch, rows, cols, 2
                eigval_params = torch.abs(eigval_params) + self.eps_L
            elif self.intermediate_strat_code in [3, 4]:  # sigmoid_lambda_sep_scale, sigmoid_lambda_sep_scale_det_ratio
                eigval_params = Mw[:, 2:4, :, :].permute((0, 2, 3, 1))  # batch, rows, cols, 2

                ###################################### FREEZE: DEBUG
                # eigval_params = torch.zeros_like(eigval_params)
                ######################################

                eigval_params = 2 * torch.sigmoid(eigval_params)# + self.eps_L  # 2*sigmoid = 1+2*(sigmoid-0.5)
                if self.intermediate_strat_code == 4:  # sigmoid_lambda_sep_scale_det_ratio
                    eigval_params[:, 1, :, :] = 1 / (eigval_params[:, 1, :, :] + self.eps)  # Forces ratio to be a constant up to scale
                eigval_scale_params = Mw[:, 4, :, :].unsqueeze(-1)  # batch, rows, cols, 1

                ###################################### FREEZE: DEBUG
                # eigval_scale_params = torch.zeros_like(eigval_scale_params)
                ######################################

                eigval_scale_params = self.avg_scale + 2 * (torch.sigmoid(eigval_scale_params) - 0.5) * self.half_scale_range# + self.eps_L
                # Belongs to [scale_min, scale_max] + eps_L

                eigval_params = eigval_params * eigval_scale_params
            else:
                raise ValueError('intermediate_strat not recognized or implemented yet')
            eigval_params = torch.cat([eigval_params, torch.zeros((*eigval_params.shape[:-1], 2)).to(device)], dim=-1)
            eigval_params = eigval_params[:, :, :, torch.tensor([0, 2, 3, 1])]
            eigval_params = eigval_params.reshape((*eigval_params.shape[:3], 2, 2))  # batch, rows, cols, 2, 2
            M = (eigvec_params @ eigval_params @ eigvec_params.permute((0, 1, 2, 4, 3))).permute((0, 3, 4, 1, 2)).reshape((eigvec_params.shape[0], 4, *eigvec_params.shape[1:3]))
            w = Mw[:, -2:, :, :]  # batch, 2, rows, cols

        else:
            raise ValueError('intermediate_strat not recognized or implemented yet')

        # TODO: If this does not work (e.g. nan), then maybe a softer constraint on w?
        #  Ideas:
        #    - implicit (regularisation on norm L2 of omega)
        #    - hard-coded (omega output of bounded function eg sigmoid)
        if not float(self.eps_w) == 1.:
            if self.strat_w_code in [0, 1, 2]:  # sigmoid_norm, sigmoid_norm_detach

                # M_inv = torch.inverse(
                #     M.permute((0, 2, 3, 1)).reshape((M.shape[0], *M.shape[-2:], 2, 2)) \
                #     + self.eps * torch.eye(2).to(M.device).unsqueeze(0).unsqueeze(1).unsqueeze(1).tile((1, *M.shape[-2:], 1, 1))
                # )  # batch, rows, cols, 2, 2

                # TODO: The inverse operation leads to nan values in the backward pass. Workaround?
                M_reshaped = M.permute((0, 2, 3, 1)).reshape((M.shape[0], *M.shape[-2:], 2, 2))
                # Compute explicit invert manually of 2x2 matrix
                # PROBLEM: when big numbers are involved yet the matrix is close to singular, det can produce negative numbers
                # which creates a wrong negative matrix. Compute det explicitly rather than invoke det algorithm search (minors)
                M_inv = torch.zeros_like(M_reshaped)
                M_reshaped_plus_eps = M_reshaped + self.eps * torch.eye(2).to(M.device).unsqueeze(0).unsqueeze(1).unsqueeze(1).repeat((M.shape[0], *M.shape[-2:], 1, 1))
                M_inv[:, :, :, 0, 0] = M_reshaped_plus_eps[:, :, :, 1, 1]
                M_inv[:, :, :, 1, 1] = M_reshaped_plus_eps[:, :, :, 0, 0]
                M_inv[:, :, :, 0, 1] = -M_reshaped_plus_eps[:, :, :, 0, 1]
                M_inv[:, :, :, 1, 0] = -M_reshaped_plus_eps[:, :, :, 1, 0]
                det_M_inv = \
                    (M_reshaped_plus_eps[:, :, :, 0, 0] * M_reshaped_plus_eps[:, :, :, 1, 1]
                     - M_reshaped_plus_eps[:, :, :, 0, 1] * M_reshaped_plus_eps[:, :, :, 1, 0]
                     ).unsqueeze(-1).unsqueeze(-1)
                M_inv = M_inv / (det_M_inv + self.eps)

                # print(torch.mean(torch.abs(M_inv)))
                # if torch.mean(torch.abs(M_inv)) == torch.inf:
                #     print('wtf')

                # Tune w for norm < 1-eps_w
                norm_w_Minv = torch.sqrt(
                    w.permute(0, 2, 3, 1).unsqueeze(-2) \
                    @ M_inv
                    @ w.permute((0, 2, 3, 1)).unsqueeze(-1)
                    + self.eps  # at 0 we get nan in the gradient due to sqrt
                ).squeeze(-1).permute((0, 3, 1, 2))

                if self.strat_w_code in [0, 1]:  # sigmoid_norm
                    norm_w_Minv_tuned = (torch.sigmoid(norm_w_Minv) - 1 / 2) * 2 * (1 - self.eps_w)
                    factor = norm_w_Minv_tuned / (norm_w_Minv + self.eps)
                    if self.strat_w_code == 1:  # sigmoid_norm_detach
                        factor = factor.detach()
                    w_tuned = w * factor
                elif self.strat_w_code == 2:  # norm_clip_detach
                    norm_w_Minv_clamped = torch.clamp(norm_w_Minv, max=1 - self.eps_w)
                    factor = norm_w_Minv_clamped / (norm_w_Minv + self.eps)
                    factor = factor.detach()
                    w_tuned = w * factor
                else:
                    raise ValueError('strat_w not recognized or implemented yet')
            else:
                raise ValueError('strat_w not recognized or implemented yet')
        else:
            # The asymetric component of the Randers is not used, i.e. Riemannian metric is used
            w_tuned = torch.zeros_like(w)

        return M, w_tuned


class Conv2dTangentBall(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k, stride=1, dilation=1,
                 ker_fixed=False,
                 bias=True,
                 strat_w='sigmoid_norm',  # 'sigmoid_norm', 'sigmoid_norm_detach', 'norm_clip_detach'
                 eps_L=0.01, eps_w=0.1, eps=1e-6,
                 scale_max=2., scale_min=0.1,
                 sampling_strat='polar_grid',  # 'polar_grid', 'onion_peeling_grid'
                 intermediate_strat='vanilla_conv',
                 modulation=False,
                 device='cpu'):
        super(Conv2dTangentBall, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k  # int
        self.stride = stride  # int or tuple
        self.dilation = dilation  # int
        self.ker_fixed = ker_fixed
        self.device = device
        self.strat_w = strat_w
        self.eps_L = eps_L
        self.eps_w = eps_w
        self.eps = eps
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.avg_scale = (scale_max + scale_min) / 2
        self.half_scale_range = (scale_max - scale_min) / 2
        self.sampling_strat = sampling_strat  # 'polar_grid', 'onion_peeling_grid'
        self.intermediate_strat = intermediate_strat  # 'vanilla_conv' or TODO?

        self.modulation = modulation  # TODO?
        if self.modulation:
            # TODO: implement modulation?
            raise ValueError('modulation not implemented yet')

        # For simplicity, we only do 1 group. Using more than 1 group is not described in the deformable conv papers.
        # Only when looking at the code of the authors, we see that they use groups > 1 to do deformable convolutions.
        # self.groups = groups  # 1

        self.intermediate_strat_codes = {  # For speeding up calculations
            'vanilla_conv': 0,
            'vanilla_conv_LDLT': 1,
            'conv_eigvec_eigval': 2,
            'conv_eigvec_eigval_sigmoid_lambda_sep_scale': 3,
            'conv_eigvec_eigval_sigmoid_lambda_sep_scale_det_ratio': 4,
        }
        # TODO: provide self.forward_intermediate function to avoid using these dangerous string/int comparisons in forward
        self.intermediate_strat_code = self.intermediate_strat_codes[self.intermediate_strat]

        self.strat_w_codes = {  # For speeding up calculations
            'sigmoid_norm': 0,
            'sigmoid_norm_detach': 1,
            'norm_clip_detach': 2,
        }
        self.strat_w_code = self.strat_w_codes[self.strat_w]

        if self.intermediate_strat in ['vanilla_conv', 'vanilla_conv_LDLT']:
            self.intermediate_Mw = torch.nn.Conv2d(
                in_channels,
                5,  # The input channels are merged by conv summation (not a metric per channel); all output channels share same offsets as in deform_conv2d
                k, stride=stride, padding=int((k // 2)*dilation), dilation=dilation,
            ).to(device)
            self.intermediate_Mw.weight.requires_grad = True
            self.intermediate_Mw.bias.requires_grad = True
        elif self.intermediate_strat in ['conv_eigvec_eigval']:
            # For M, predict R_theta, lambda_1, lambda_2. To predict R_theta, we can theoretically just predict theta,
            # but it is better to predict cos_theta and sin_theta to avoid the 2pi periodicity. We still need to enforce
            # the norm of the eigenvectors to be 1. We can do this by predicting the cos_theta and sin_theta and then
            # normalising the vector.
            # 1) Instead of Cholesky from conv, change. Learn the eigvec (maybe conv) and eigval.
            #    - Eigval must be positive
            #    - Eigvec must be orthonormal, maybe regularisation? Or just learn the first direc and then do ortho
            #    - Then, M = eigvec @ eigval @ eigvec^T
            #    - First eigval maybe linear in input (so conv)
            #    - Second eigval better not linear in input (so not conv, maybe do a transform of input)
            self.intermediate_Mw = torch.nn.Conv2d(
                in_channels,
                6,  k, stride=stride, padding=int((k // 2)*dilation), dilation=dilation,
            ).to(device)
            self.intermediate_Mw.weight.requires_grad = True
            self.intermediate_Mw.bias.requires_grad = True
        elif self.intermediate_strat in ['conv_eigvec_eigval_sigmoid_lambda_sep_scale',
                                         'conv_eigvec_eigval_sigmoid_lambda_sep_scale_det_ratio']:
            # Similar to conv_eigvec_eigval but lambda_1 and lambda_2 are predicted separately and then multiplied by a
            #  scale factor. The scale factor is predicted by a conv2d layer.
            self.intermediate_Mw = torch.nn.Conv2d(
                in_channels,
                7,  k, stride=stride, padding=int((k // 2)*dilation), dilation=dilation,
            ).to(device)
            self.intermediate_Mw.weight.requires_grad = True
            self.intermediate_Mw.bias.requires_grad = True
        else:
            # Implement strat such that self.intermediate_Mw has a weight and bias fields?

            # Options:
            # 2) Normalise data fed to intermediate_Mw by removing the mean? Want to be sensitive to the grad rather than
            #  actual values. Mean removal should just be done for the intermediate_Mw, not the final metric utb conv?
            #  the idea is that offsets should not depend on average intensity but rather on gradient. Note that this
            #  idea is not used in deform_conv2d but it might benefit them too
            # 3) If nan from M^-1 for non zero omega, then change strat for omega
            #     - Reg on norm2 omega?
            #     - Add explicit constraint on omega (e.g. sigmoid)?
            #     - Can be combined on constraints (implicit via reg or explicit) on M by controlling smallest eigval etc

            raise ValueError('intermediate_strat not recognized or implemented yet')

        self.compute_M_w_from_Mw = ComputeMAndwFromMw(
            intermediate_strat_code=self.intermediate_strat_code, strat_w_code=self.strat_w_code,
            eps_L=self.eps_L, eps=self.eps, eps_w=self.eps_w, scale_max=self.scale_max, scale_min=self.scale_min
        )

        self.ker = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, k, k).to(device)) if not ker_fixed else None
        # Problem: optimizer might still update ker even if requires_grad=False if using momentum or weight_decay if it
        # is a parameter fed to the optimizer

        self.ker_samp_ctr = None

        self.bias = bias  # Output bias after the convolution (not convMw), will become a parameter if True or None
        # In previous experiments (no deep), we used torchvision.ops.deform_conv2d, which does not have a bias by
        # default. Note that the layer torchvision.ops.DeformConv2d does have a bias by default.

        self.init_params()

    def init_params(self):
        normalisation_conv_init = 1. / ((self.k ** 2) * self.in_channels)
        if self.intermediate_strat in ['vanilla_conv']:
            self.intermediate_Mw.weight.data[0].fill_(normalisation_conv_init)
            self.intermediate_Mw.weight.data[1].fill_(0 + self.eps)
            self.intermediate_Mw.weight.data[2].fill_(normalisation_conv_init)
            self.intermediate_Mw.weight.data[3].fill_(0 + self.eps)
            self.intermediate_Mw.weight.data[4].fill_(0 + self.eps)
            self.intermediate_Mw.weight.requires_grad = True
            self.intermediate_Mw.bias.data.fill_(0)
        elif self.intermediate_strat in ['vanilla_conv_LDLT']:
            self.intermediate_Mw.weight.data[0].fill_(0 + self.eps)
            self.intermediate_Mw.weight.data[1].fill_(normalisation_conv_init)
            self.intermediate_Mw.weight.data[2].fill_(normalisation_conv_init)
            self.intermediate_Mw.weight.data[3].fill_(0 + self.eps)
            self.intermediate_Mw.weight.data[4].fill_(0 + self.eps)
            self.intermediate_Mw.weight.requires_grad = True
            self.intermediate_Mw.bias.data.fill_(0)
        elif self.intermediate_strat in ['conv_eigvec_eigval']:
            # No obvious fancy initialisation
            self.intermediate_Mw.weight.requires_grad = True
            self.intermediate_Mw.bias.data.fill_(0)
        elif self.intermediate_strat in ['conv_eigvec_eigval_sigmoid_lambda_sep_scale',
                                         'conv_eigvec_eigval_sigmoid_lambda_sep_scale_det_ratio']:
            for ii in range(2, 7):
                # sigmoid approach gives that for 0 init is isotrop riemman with dist 1 pix
                self.intermediate_Mw.weight.data[ii].fill_(0 + self.eps)
            self.intermediate_Mw.weight.requires_grad = True
            self.intermediate_Mw.bias.data.fill_(0)
        else:
            raise ValueError('intermediate_strat not recognized or implemented yet')

        # In pytorch, deform_conv2d like conv2d sums accross input channels. Thus a 3x3 conv is actually in_channels*3x3
        # To avoid having numbers explode, we should divide by in_channels.
        normalisation_ker_init = 1. / ((self.k ** 2) * self.in_channels)
        if not self.ker_fixed:
            self.ker.data.fill_(normalisation_ker_init)
            self.ker.requires_grad = True
        else:
            self.ker = torch.ones(self.out_channels, self.in_channels, self.k, self.k).to(self.device) * normalisation_ker_init
            self.ker.requires_grad = False

        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels).to(self.device)) if self.bias else None
        if self.bias is not None:
            self.bias.data.fill_(0)
            self.bias.requires_grad = True
        else:
            self.bias = torch.zeros(self.out_channels).to(self.device)
            self.bias.requires_grad = False

    def forward(self, inp):

        Mw = self.intermediate_Mw(inp)
        M, w_tuned = self.compute_M_w_from_Mw(Mw, inp)
        out = blur_randers_ball_tangent(
            inp, M, w_tuned, eps=self.eps, kh=self.k, kw=self.k, ker=self.ker,
            bias=self.bias, stride=self.stride, dilation=self.dilation,
            sampling_strat=self.sampling_strat
        )

        # print(torch.mean(torch.abs(M)), torch.mean(torch.abs(w_tuned)))

        return out


def convert_model_to_tangent_ball(model_name, model,
                                  k=None,
                                  ker_fixed=False, strat_w='sigmoid_norm',
                                  eps_L=0.01, eps_w=0.1, eps=1e-6,
                                  scale_max=2., scale_min=0.1,
                                  sampling_strat='polar_grid', intermediate_strat='vanilla_conv',
                                  no_pooling_or_stride_conv1=False):

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
                        if k is None:
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

                        basicblock_conv = Conv2dTangentBall(
                            conv.in_channels, conv.out_channels, k, stride=stride, dilation=dilation,
                            ker_fixed=ker_fixed, bias=bias,
                            eps_L=eps_L, strat_w=strat_w, eps_w=eps_w, eps=eps,
                            scale_max=scale_max, scale_min=scale_min,
                            sampling_strat=sampling_strat, intermediate_strat=intermediate_strat,
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
                    if k is None:
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

                    bottleneck.conv2 = Conv2dTangentBall(
                        conv2.in_channels, conv2.out_channels, k, stride=stride, dilation=dilation,
                        ker_fixed=ker_fixed, bias=bias,
                        eps_L=eps_L, strat_w=strat_w, eps_w=eps_w, eps=eps,
                        scale_max=scale_max, scale_min=scale_min,
                        sampling_strat=sampling_strat, intermediate_strat=intermediate_strat,
                        device=device
                    )

    else:
        raise ValueError('conversion not implemented for model_name ' + model_name)
    return model


def main_unit_tangent_ball():

    # torch.autograd.set_detect_anomaly(True)  # Debugging
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    #
    # # use CUDA_LAUNCH_BLOCKING=1 to debug CUDA code
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # TODO: multigpu ddp training for imagenet
    # Imagenette ? Small dataset but big images

    computer = 'newton'  # 'local' or 'newton'

    if computer == 'local':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if dev == 'cpu':
            warnings.warn('No GPU available. Using CPU')
            dataset_root = '/datasets'
        else:
            dataset_root = '/datasets'

        args = argparse.Namespace()
        args.run_number = 0
        args.dataset_name = 'CIFAR10'               # 'MNIST', 'FashionMNIST', 'CIFAR10', 'ImageNet', (maybe: 'STL10')
        args.ker_fixed = False                      # True, False
        args.k = None                               # None or int, (None defaults to 3)
        args.strat_w = 'sigmoid_norm_detach'           # 'sigmoid_norm', 'sigmoid_norm_detach', 'norm_clip_detach'
        args.eps_w = 0.5                            # must be <= 1  # 0.1, 0.9, 1.
        args.eps_L = 0.01                           # 0.01
        args.eps = 1e-6                             # 1e-6
        args.scale_min = 0.1                 # None, 0.1  # Ellipse largest size is 10 for 0.1 (inv dependency)
        args.scale_max = 1.5                  # None, 1.  # Ellipse smallest size is 0.5 for 1 (inv dependency)
        # For 0 conv init gives ellipse of (scale_min+scale_max)/2, = 1 for 0.1-1.9, = 0.8 for 0.1-1.5
        args.intermediate_strat = 'conv_eigvec_eigval_sigmoid_lambda_sep_scale'   # 'vanilla_conv', 'vanilla_conv_LDLT', 'conv_eigvec_eigval',
                                                                                   # 'conv_eigvec_eigval_sigmoid_lambda_sep_scale',
                                                                                   # 'conv_eigvec_eigval_sigmoid_lambda_sep_scale_det_ratio'
        args.sampling_strat = 'onion_peeling_grid'          # 'polar_grid', 'onion_peeling_grid'
        args.lambda_reg_Mw = 1.0                     # 0.0
        args.train = True                           # True, False
        args.ignore_checkpoint = False              # True, False
        args.augment = True                         # True, False
        args.model_name = 'ResNet18'                # 'ResNet18', 'ResNet50', 'ResNet152'
        args.pretrained = True                      # True, False  # TODO: maybe train vanilla resnet on dataset with fixes (classif + small dataset) for initialisations?
        args.no_pooling_or_stride_conv1 = True      # False, True
        args.batch_size = 128                       # 128
        args.epochs = 240                           # 120
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
        parser.add_argument('--ker_fixed', type=str, choices=['True', 'False'])
        parser.add_argument('--k', type=none_or_int, default=None)
        parser.add_argument('--strat_w', type=str, choices=['sigmoid_norm', 'sigmoid_norm_detach', 'norm_clip_detach'])
        parser.add_argument('--eps_w', type=float)
        parser.add_argument('--eps_L', type=float)
        parser.add_argument('--eps', type=float)
        parser.add_argument('--scale_min', type=none_or_float, default=None)
        parser.add_argument('--scale_max', type=none_or_float, default=None)
        parser.add_argument('--intermediate_strat', type=str,
                            choices=['vanilla_conv', 'vanilla_conv_LDLT', 'conv_eigvec_eigval',
                                     'conv_eigvec_eigval_sigmoid_lambda_sep_scale',
                                     'conv_eigvec_eigval_sigmoid_lambda_sep_scale_det_ratio'])
        parser.add_argument('--sampling_strat', type=str, choices=['polar_grid', 'onion_peeling_grid'])
        parser.add_argument('--lambda_reg_Mw', type=float)
        parser.add_argument('--train', type=str, choices=['True', 'False'])
        parser.add_argument('--ignore_checkpoint', type=str, choices=['True', 'False'])
        parser.add_argument('--augment', type=str, choices=['True', 'False'])
        parser.add_argument('--model_name', type=str, choices=['VGG16', 'ResNet18', 'ResNet50', 'ResNet152'])
        parser.add_argument('--pretrained', type=str, choices=['True', 'False'])
        parser.add_argument('--no_pooling_or_stride_conv1', type=str, choices=['True', 'False'])
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
        args.ker_fixed = args.ker_fixed == 'True'
        args.train = args.train == 'True'
        args.ignore_checkpoint = args.ignore_checkpoint == 'True'
        args.pretrained = args.pretrained == 'True'
        args.no_pooling_or_stride_conv1 = args.no_pooling_or_stride_conv1 == 'True'
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

    # torch.manual_seed(42)
    # if dev == 'cuda' or isinstance(dev, int):
    #     torch.cuda.manual_seed(42)

    run_number = args.run_number
    dataset_name = args.dataset_name
    k = args.k
    ker_fixed = args.ker_fixed
    model_name = args.model_name
    pretrained = args.pretrained
    batch_size = args.batch_size
    strat_w = args.strat_w
    eps_w = args.eps_w
    retrain = args.train
    ignore_checkpoint = args.ignore_checkpoint
    sampling_strat = args.sampling_strat
    intermediate_strat = args.intermediate_strat
    lambda_reg_Mw = args.lambda_reg_Mw
    lr = args.lr  # We choose to take the lr strategy that is the same on every dataset and model (similar to competition)
    optimizer = args.optimizer
    lr_scheduler = args.lr_scheduler
    epochs = args.epochs
    step_size_lr_scheduler = args.step_size_lr_scheduler
    multistep_lr_scheduler = args.multistep_lr_scheduler
    gamma_lr_scheduler = args.gamma_lr_scheduler
    T_max = args.T_max
    no_pooling_or_stride_conv1 = args.no_pooling_or_stride_conv1
    augment = args.augment
    eps = args.eps
    eps_L = args.eps_L
    scale_min = args.scale_min
    scale_max = args.scale_max

    ##################### Train hyperparameters #####################

    if not os.environ.keys().__contains__("LOCAL_RANK"):
        num_workers = 0 if computer == 'local' else 4
    else:
        num_workers = 4

    ##################### Res preparation #####################

    str_pretrained = '_pretrained' if pretrained else ''
    str_no_pooling_or_stride_conv1 = '_noPoolStrideConv1' if no_pooling_or_stride_conv1 else ''
    if lr_scheduler == 'StepLR':
        str_scheduler_args = 'step_size_lr_scheduler_' + str(step_size_lr_scheduler) + \
                             '_gamma_lr_scheduler_' + str(gamma_lr_scheduler)
    elif lr_scheduler == 'MultiStepLR':
        str_scheduler_args = 'multistep_lr_scheduler_' + '-'.join([str(step) for step in multistep_lr_scheduler]) + \
                             '_gamma_lr_scheduler_' + str(gamma_lr_scheduler)
    else:
        str_scheduler_args = 'T_max_' + str(T_max)
    str_strat_w = '_strat_w_' + strat_w if not eps_w == 1.0 else ''
    str_scales = '_scales_min_max_' + str(scale_min) + '_' + str(scale_max)  \
        if intermediate_strat in ['conv_eigvec_eigval_sigmoid_lambda_sep_scale',
                                  'conv_eigvec_eigval_sigmoid_lambda_sep_scale_det_ratio'] else ''
    str_lambda_reg = '_lambda_reg_Mw_' + str(lambda_reg_Mw) if lambda_reg_Mw > 0 else ''

    dir_res = os.path.join(
        'res',
        'classification',
        dataset_name,
        'unit_tangent_ball_ker_fixed' if ker_fixed else 'unit_tangent_ball',
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
        dir_res = os.path.join(dir_res, 'run_' + str(run_number))
    pathlib.Path(dir_res).mkdir(parents=True, exist_ok=True)

    dir_checkpoint = os.path.join(dir_res, 'checkpoint')
    pathlib.Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()

    model = import_model_architecture(model_name, pretrained)
    model = model.to(device)
    # Change classification layer in model
    model = adapt_model_classifier(model_name, model, dataset_name).to(device)  # device just in case
    model = convert_model_to_tangent_ball(
        model_name, model, k=k, ker_fixed=ker_fixed,
        eps_L=eps_L, strat_w=strat_w, eps_w=eps_w, eps=eps,
        scale_max=scale_max, scale_min=scale_min,
        sampling_strat=sampling_strat, intermediate_strat=intermediate_strat,
        no_pooling_or_stride_conv1=no_pooling_or_stride_conv1
    ).to(device)  # device just in case

    if lambda_reg_Mw == 0.:
        reg_hooks = None
    else:
        reg_hooks = WeightHook()
        for layer in model.children():
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer.children():
                    if isinstance(sublayer, torchvision.models.resnet.BasicBlock) or \
                            isinstance(sublayer, torchvision.models.resnet.Bottleneck):
                        for subsublayer in sublayer.children():
                            if isinstance(subsublayer, Conv2dTangentBall):
                                subsublayer.intermediate_Mw.register_forward_hook(reg_hooks)

        # Do not forget to use reg_hooks.clear() before each batch or epoch

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
                                                             lambda_reg=lambda_reg_Mw, reg_hooks=reg_hooks,
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
                ax.axvline(last_epoch, color='red', linestyle='--')  # , label='nan')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(dir_res, 'loss.png'))

            fig, ax = plt.subplots()
            ax.plot(acc_tracker_train, label='train')
            ax.plot(acc_tracker_val, label='val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Acc')
            if last_epoch_and_nan[1]:
                ax.axvline(last_epoch, color='red', linestyle='--')  # , label='nan')
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
                    ax.axvline(last_epoch, color='red', linestyle='--')  # , label='nan')
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
    main_unit_tangent_ball()
