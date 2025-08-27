import argparse
import itertools
import multiprocessing
import os
import wandb
import time

import argparse
import itertools
import os
import wandb
import torch
import numpy as np

# from dataset.base import load_frames

from util.util import load_config, parse_model, set_random_seed, str2list, split_list

""" 
def gen_pose_all(model, config, scenes, ids):
    pass
def gen_nvs_all(model, config, scenes, ids):
    pass
def gen_nvs_my_finetune(model, config, scenes, ids):
    pass

def main(config, mode, gpu_ids):
    def worker(config, mode, scenes, ids, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        model = parse_model(config.model)
        if mode[0]:
            gen_pose_all(model, config, scenes, ids)
        if mode[1]:
            gen_nvs_all(model, config, scenes, ids)
        if mode[2]:
            gen_nvs_my_finetune(model, config, scenes, ids)
    config, conf_dict = config

    perm = list(itertools.combinations(range(3), 2))
    ids = [",".join(map(str, p)) for p in perm]
    gpu_ids = str2list(gpu_ids)
    scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))[0:5]

    curr_time = time.localtime(time.time())
    mon, mday, hours = curr_time.tm_mon, curr_time.tm_mday, curr_time.tm_hour
    mins = curr_time.tm_min + curr_time.tm_sec / 60
    

    # split scenes and multi-process
    scenes = split_list(scenes, len(gpu_ids))
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = multiprocessing.Process(target=worker, args=(config, mode, scenes[i], ids, gpu_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join() 
"""
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--nvs", action="store_true")
    parser.add_argument("--my_nvs", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    set_random_seed(config[0].seed)
    config, conf_dict = config
    print(config is None)
    gpu_ids = str2list(args.gpu_ids)
    scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))[0:5]

    curr_time = time.localtime(time.time())
    mon, mday, hours = curr_time.tm_mon, curr_time.tm_mday, curr_time.tm_hour
    mins = curr_time.tm_min + curr_time.tm_sec / 60
    wb_run = wandb.init(dir="./wandb/finetune",entity="kevin-shih",project="iFusion-Adv",)
    wb_run.finish()
    # main(config, [args.pose, args.nvs, args.my_nvs], args.gpu_ids)