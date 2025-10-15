# Summary explanation


The main file is unit_tangent_ball_learning_dataset_deep_classif_torchrun. As the name suggests, we ran it with torchrun. We provide examples of how to use it, either with the python batch_run[...].sh files or the proper batch files sbatch_run[...].sh

Regarding the organisation:

- The batch_run[...].py were used to conveniently launch a lot of simulateous jobs exploring many configurations. They were used in the paper but are not recommended as they use the srun instead of the sbatch slurm command.

- The sbatch_run[...].sh examples are more recommended and perfect for learning just on manually selected configurations.

- The following terminologies correspond to specific types of adaptive convolutions
    - "baseline" is standard convolution.
    - "deform_conv2d" is deformable convolution.
    - "shifted_conv2d" is shifted convolution, a.k.a. entire deformable convolution.
    - "unit_tangent_ball" is our Randers metric convolution.

- The terminology "lastReplacement" corresponds to replacing only the last layers of the CNNs (Table 11 of the supplementary).

- The terminology "Riemann" corresponds to our metric convolutions using Riemannian, i.e. Randers with no asymmetry, rather than asymmetric Finsler metrics (Table 11 of the supplementary)

- The pytorch_grad_cam library is from https://github.com/jacobgil/pytorch-grad-cam