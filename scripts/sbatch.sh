#!/bin/bash
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH -p defq # Place job into 'defq' queue
#SBATCH --time=4:00:00
#SBATCH --task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1 # Reserve 1 GPUs for this job

python main.py -g 0 -6 --config config/main_scene_my_general.yaml \
    log.run_name="$1" \
    data.nvs_root="$2" \
    data.resume_ckpt_fp="$3"