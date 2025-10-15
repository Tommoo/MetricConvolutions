import cv2
import torch
import torchvision
import os
import pathlib
from utils_torchrun import import_model_architecture, adapt_model_classifier, get_dataset_params, prepare_dataset
from baseline_learning_dataset_deep_classif import convert_model_baseline
from deform_conv2d_learning_dataset_deep_classif import convert_model_to_deform_conv
from shifted_conv2d_learning_dataset_deep_classif import convert_model_to_shift_conv
from unit_tangent_ball_learning_dataset_deep_classif import convert_model_to_tangent_ball
import matplotlib.pyplot as plt
import matplotlib
from pytorch_grad_cam.grad_cam import GradCAM
# from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def main():
    dataset_root = '/datasets'
    dir_res_gradcam = 'res/gradcam/CIFAR100/ResNet18_pretrained_noPoolStrideConv1/'

    path_model_baseline = 'res/classification/CIFAR100/baseline/ResNet18_pretrained_noPoolStrideConv1/augment_True/' \
                          'bs_128/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR/T_max_240/' \
                          'model.pth'
    path_model_deform = 'res/classification/CIFAR100/deform_conv2d/ResNet18_pretrained_noPoolStrideConv1/augment_True/' \
                        'k_None__bs_128/eps_1e-06/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR/T_max_240/' \
                        'model.pth'
    path_model_shifted = 'res/classification/CIFAR100/shifted_conv2d/ResNet18_pretrained_noPoolStrideConv1/augment_True/' \
                         'k_None__bs_128/eps_1e-06/lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR/T_max_240/' \
                         'model.pth'
    path_model_utb = 'res/classification/CIFAR100/unit_tangent_ball/ResNet18_pretrained_noPoolStrideConv1/augment_True/' \
                     'intermediate_strat_conv_eigvec_eigval_sigmoid_lambda_sep_scale_scales_min_max_0.1_1.5/' \
                     'sampling_strat_onion_peeling_grid/k_None__bs_128/eps_1e-06_epsL_0.01_epsw_0.5_strat_w_sigmoid_norm_detach/' \
                     'lr_0.0001_optimizer_Adam_lr_scheduler_CosineAnnealingLR_lambda_reg_Mw_5000.0/T_max_240/run_1/' \
                     'model.pth'

    pathlib.Path(dir_res_gradcam).mkdir(parents=True, exist_ok=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    print(f'Using {device} device')

    matplotlib.use('TkAgg')
    # matplotlib.use('Qt5Agg')  # For pytorch debug interactive mode. Does not work with cv2: pip uninstall opencv-python

    path_models = [path_model_baseline, path_model_deform, path_model_shifted, path_model_utb]
    convert_model_list = [convert_model_baseline, convert_model_to_deform_conv, convert_model_to_shift_conv, convert_model_to_tangent_ball]
    args_convert_model_list = [
        {'no_pooling_or_stride_conv1': True, 'no_change_deep_layers': False},
        {'k': None, 'ker_fixed': False, 'eps': 1e-6, 'no_pooling_or_stride_conv1': True},
        {'k': None, 'ker_fixed': False, 'eps': 1e-6, 'no_pooling_or_stride_conv1': True},
        {
            'k': None, 'ker_fixed': False, 'eps_L': 0.01, 'strat_w': 'sigmoid_norm_detach', 'eps_w': 0.5, 'eps': 1e-6,
            'scale_max': 1.5, 'scale_min': 0.1, 'sampling_strat': 'onion_peeling_grid',
            'intermediate_strat': 'conv_eigvec_eigval_sigmoid_lambda_sep_scale', 'no_pooling_or_stride_conv1': True
        }
    ]
    model_names = ['baseline', 'deform', 'shifted', 'utb']
    models = []
    for model_idx in range(4):
        model = import_model_architecture('ResNet18', pretrained=True)
        # Change classification layer in model
        model = adapt_model_classifier('ResNet18', model, 'CIFAR100')
        convert_model_fn = convert_model_list[model_idx]
        args_convert_model_fn = args_convert_model_list[model_idx]
        model = convert_model_fn(
            'ResNet18', model,
            **args_convert_model_fn
        )
        model.load_state_dict(torch.load(path_models[model_idx], map_location='cpu'))
        model.eval()
        models.append(model)

    # model_baseline = models[0]
    # model_deform = models[1]
    # model_shifted = models[2]
    # model_utb = models[3]

    # Load cifar100 dataset
    mean, std, size_resize, crop_size, inp_size = get_dataset_params('CIFAR100')
    resize = torchvision.transforms.Resize(
        (size_resize, size_resize)) if size_resize is not None else torch.nn.Identity()
    crop = torchvision.transforms.CenterCrop(crop_size) if crop_size is not None else torch.nn.Identity()
    normalize = torchvision.transforms.Normalize(mean, std)
    transform_val = torchvision.transforms.Compose([
        resize,
        crop,
        torchvision.transforms.ToTensor(),
        normalize
    ])
    # Identity so that we can plot images without normalization
    # Do not forget to apply normalisation after loading the images before feeding them to the models
    _, dataset_val = prepare_dataset('CIFAR100', dataset_root, torch.nn.Identity(), torch.nn.Identity(), computer='local')

    ii = 0
    firstk = 5
    selected_classes = ['dolphin', 'lamp', 'tank']
    first_ims = [[] for _ in range(len(selected_classes))]
    select_labels = [dataset_val.classes.index(class_name) for class_name in selected_classes]
    while True:
        if ii >= len(dataset_val):
            break
        if dataset_val[ii][1] in select_labels and len(first_ims[select_labels.index(dataset_val[ii][1])]) < firstk:
            first_ims[select_labels.index(dataset_val[ii][1])].append(dataset_val[ii][0])
        if all([len(first_im) == firstk for first_im in first_ims]):
            break
        ii += 1

    # Create list of figure handles. In the same figure, show the original image and the 4 grad cam images
    fig_axs_list = [[] for _ in range(len(selected_classes))]
    for index_label in range(len(selected_classes)):
        for index_im in range(firstk):
            fig, axs = plt.subplots(1, 1 + len(models))
            fig_axs_list[index_label].append((fig, axs))

    # Compute grad cams
    for index_model, model in enumerate(models):
        target_layers = [model.layer4[-1]]
        for index_label in range(len(selected_classes)):
            label = dataset_val.classes.index(selected_classes[index_label])
            targets = [ClassifierOutputTarget(label)]
            for index_im in range(firstk):
                print('Model:', index_model, 'Label:', index_label, 'Image:', index_im)
                raw_im = first_ims[index_label][index_im]
                im = torchvision.transforms.ToTensor()(raw_im)
                im_transf = transform_val(raw_im).unsqueeze(0)
                im_transf = im_transf.to(device)

                cam = GradCAM(model=model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=im_transf, targets=targets)
                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(im.permute((1, 2, 0)).numpy(), grayscale_cam, use_rgb=True)

                # save im and its gradcam
                cv2.imwrite(os.path.join(dir_res_gradcam, f'im_{selected_classes[index_label]}_{index_im}.png'),
                            cv2.cvtColor(im.permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(dir_res_gradcam,
                                         f'gradcam_{model_names[index_model]}_{selected_classes[index_label]}_{index_im}.png'),
                            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

                fig, axs = fig_axs_list[index_label][index_im]
                axs[0].imshow(im.permute(1, 2, 0))
                axs[0].axis('off')
                axs[0].set_title(selected_classes[index_label])
                axs[index_model + 1].imshow(visualization)
                axs[index_model + 1].axis('off')
                plt.show(block=False)





    plt.show()



if __name__ == '__main__':
    main()