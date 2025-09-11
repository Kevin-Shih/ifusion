import argparse
import itertools
import multiprocessing
import os
import wandb
import time
import json
import numpy as np

from dataset.base import load_frames
from ifusion import finetune, inference, optimize_pose, my_finetune, my_finetune_general, inference_all
from util.util import load_config, parse_model, set_random_seed, str2list, split_list
from rich import print

def set_default_latlon(config):
    id = str2list(config.data.id)[0]
    latlon = load_frames(
        config.data.image_dir, config.data.gt_transform_fp, verbose=False
    )[2]
    default_latlon = latlon[id].tolist()
    config.pose.default_theta = default_latlon[0]
    config.pose.default_azimuth = default_latlon[1]
    config.pose.default_radius = default_latlon[2]


def gen_pose_all(model, config, scenes, ids):
    for scene in scenes:
        transform_dict = {"camera_angle_x": np.deg2rad(49.1), "frames": []}
        for id in ids:
            print(f"[INFO] Optimizing pose {scene}:{id}")
            config.data.scene = scene
            config.data.id = id
            set_default_latlon(config)
            optimize_pose(model, transform_dict, **config.pose)
        if config.pose.scene_transform_fp :
            with open(config.pose.scene_transform_fp, "w") as f:
                json.dump(transform_dict, f, indent=4)


def gen_nvs_all(model, config, scenes, ids):
    for scene in scenes:
        for id in ids:
            if config.finetune:
                print(f"[INFO] Fine-tuning \'{scene}\':{id}")
                config.data.scene = scene
                config.data.id = id
                finetune(model, **config.finetune)
                inference(model, **config.inference)
            else:
                print(f"[INFO] Inference \'{scene}\':{id}")
                config.data.scene = scene
                config.data.id = id
                inference(model, **config.inference)

def gen_nvs_my_finetune(model, config, scenes, ids):
    # scenes = [scenes[0], scenes[2]]
    # ids = [ids[0], ids[3]]
    # for scene, id in zip(scenes, ids):
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Fine-tuning {scene}")
            config.data.scene = scene
            config.data.id = id
            my_finetune(model, **config.finetune)
            inference(model, **config.inference)

def gen_nvs_my_finetune_general(model, config, scenes, ids):
    print(f"[INFO] Fine-tuning (Generalizable)")
    config.data.lora_ckpt_fp = f'{config.data.nvs_root_dir}/{config.data.name}/lora.ckpt'
    my_finetune_general(model, config, scenes, ids)
    for scene in scenes:
        for id in ids:
            config.data.scene = scene
            config.data.id = id
            inference_all(model, **config.inference)

def gen_nvs_from_ckpt(model, config, scenes, ids):
    # print(f"[INFO] Fine-tuning (Generalizable)")
    # config.data.lora_ckpt_fp = f'{config.data.nvs_root_dir}/{config.data.name}/lora.ckpt'
    config.data.lora_ckpt_fp = f'{config.data.nvs_root_dir}/{config.data.name}/lora/lora_1000.ckpt'
    model.inject_lora(
        ckpt_fp=config.inference.lora_ckpt_fp,
        rank=config.inference.lora_rank,
        target_replace_module=config.inference.lora_target_replace_module,
    )
    for scene in scenes:
        for id in ids:
            config.data.scene = scene
            config.data.id = id
            inference(model, **config.inference)

def main(config, mode, gpu_ids):
    def worker(config, mode, scenes, ids, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        model = parse_model(config.model)
        if mode[0]:
            gen_pose_all(model, config, scenes, ids)
        if mode[1]:
            gen_nvs_all(model, config, scenes, ids=["0,1"])
        elif mode[2]:
            gen_nvs_my_finetune(model, config, scenes, ids=["0,1"])
        elif mode[3]:
            gen_nvs_my_finetune_general(model, config, scenes, ids=["0,1"])
        elif mode[4]:
            gen_nvs_from_ckpt(model, config, scenes, ids=["0,1"])
    config, conf_dict = config

    perm = list(itertools.permutations(range(5), 2))
    # perm = list(itertools.combinations(range(5), 2))
    # perm = list(itertools.combinations(range(3), 2))
    ids = [",".join(map(str, p)) for p in perm]
    gpu_ids = str2list(gpu_ids)
    scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))[0:]
    print(f"[INFO] Found {len(scenes)} scenes")

    curr_time = time.localtime(time.time())
    mon, mday, hours = curr_time.tm_mon, curr_time.tm_mday, curr_time.tm_hour
    mins = curr_time.tm_min + curr_time.tm_sec / 60
    if config.log is None:
        wb_run = None
    elif config.log.run_path:
        print(f'[INFO] Resuming wandb run from \'{config.log.run_path}\'. Ignoring group_name and run_name arguments.')
        wb_run = wandb.Api().run(f'kevin-shih/iFusion-Adv/{config.log.run_path.split("/")[-1]}')
        print(f'Confirm Resuming from \'{wb_run.name}\', id:\'{wb_run.id}\'? [Y/N]')
        user_input = input()
        if user_input.lower() in ('y', 'yes'):
            wb_run = wandb.init(
                dir="../wandb/eval",
                entity="kevin-shih",
                project="iFusion-Adv",
                id=f"{config.log.run_path.split('/')[-1]}",
                resume="must",
                settings=wandb.Settings(x_disable_stats=True, x_save_requirements=False),
                config={
                    'mode':{
                        'pose':  wb_run.config.mode.pose or mode[0],
                        'zero123_nvs':  wb_run.config.mode.zero123_nvs or (mode[1] and config.finetune is None),
                        'iFusion_finetune': wb_run.config.mode.iFusion_finetune or (mode[1] and config.finetune is not None),
                        'my_finetune': wb_run.config.mode.my_finetune or mode[2],
                    },
                    **conf_dict,
                },
            )
        else:
            print(f'Canceled: Abort execution.')
            exit(0)
    else:
        wb_run = wandb.init(
            dir="./wandb/finetune",
            entity="kevin-shih",
            project="iFusion-Adv",
            group= f'{config.log.group_name}',
            name= f'{config.log.run_name}',
            settings=wandb.Settings(x_disable_stats=True, x_save_requirements=False),
            config={
                    "start_date": f'{mon:02d}-{mday:02d}',
                    "start_time": f'{hours:02d}-{mins:04.1f}',
                    'mode':{
                        'pose':  mode[0],
                        'zero123_nvs':  mode[1] and config.finetune is None,
                        'iFusion_finetune':  mode[1] and config.finetune is not None,
                        'my_finetune':  mode[2],
                    },
                    **conf_dict,
            },
        )

    # split scenes and multi-process
    scenes = split_list(scenes, len(gpu_ids))
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = multiprocessing.Process(target=worker, args=(config, mode, scenes[i], ids, gpu_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    if wb_run:
        wb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    parser.add_argument('-p', "--pose", action="store_true")
    parser.add_argument('-1', '-n', "--nvs", action="store_true")
    parser.add_argument('-2', "--my_nvs", action="store_true")
    parser.add_argument('-3', "--my_nvs_general", action="store_true")
    parser.add_argument('-4', "--my_lora_nvs", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    set_random_seed(config[0].seed)
    main(config, [args.pose, args.nvs, args.my_nvs, args.my_nvs_general, args.my_lora_nvs], args.gpu_ids)