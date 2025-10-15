import subprocess
# import argparse
import pathlib

if not pathlib.Path('logs').exists():
    pathlib.Path('logs').mkdir()

# parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', type=int)
# parser.add_argument('--gpus', type=int)
# args = parser.parse_args()

dataset_name_list = ['PascalVOC2012']  # ['BSDS300', 'PascalVOC2012']
bw_list = [True]  # [False, True]
ker_fixed_list = [True]  # [False, True]
sample_centre_list = [False]  # [False, True]
k_list = [31]  # [5, 11, 31]
sigma_list = [0.1]  # [0.1, 0.3, 0.5]
eps_w_list = [0.9]  # [0.1, 0.9]

processes = []
for dataset_name in dataset_name_list:
    for bw in bw_list:
        for ker_fixed in ker_fixed_list:
            for sample_centre in sample_centre_list:
                for k in k_list:
                    for sigma in sigma_list:
                        for eps_w in eps_w_list:
                            print(
                                'dataset_name', dataset_name,
                                'bw', bw,
                                'ker_fixed', ker_fixed,
                                'sample_centre', sample_centre,
                                'k', k,
                                'sigma', sigma,
                                'eps_w', eps_w,
                            )
                            f = open(
                                f"logs/output-"
                                f"{dataset_name}-"
                                f"{bw}-"
                                f"{ker_fixed}-"
                                f"{sample_centre}-"
                                f"{k}-"
                                f"{sigma}-"
                                f"{eps_w}-"
                                f".txt",
                                "w"
                            )

                            # for k<=11 you can run everything on a laptop cpu
                            # for larger k you really should go for a gpu.
                            # You might need a partition argument for your slurm setup
                            f_partition_args = ''

                            p = subprocess.Popen(
                                # f'srun -c {args.cpus} --gres=gpu:{args.gpus}'
                                f'srun -c 4 --gres=gpu:1'
                                f'{f_partition_args}'
                                f' python unit_tangent_ball_learning_dataset_nodeep_nonan.py'
                                f' --dataset_name {dataset_name}'
                                f' --bw {bw}'
                                f' --ker_fixed {ker_fixed}'
                                f' --sample_centre {sample_centre}'
                                f' --k {k}'
                                f' --sigma {sigma}'
                                f' --eps_w {eps_w}',
                                shell=True,
                                stdout=f,
                                stderr=f
                            )
                            processes.append(p)

for p in processes:
    p.wait()
