import argparse
import itertools
import multiprocessing
import os
import wandb
import json
import numpy as np

from dataset.base import load_frames
from ifusion import finetune, inference, optimize_pose, my_finetune, my_finetune_general, inference_for_consist
from util.util import load_config, parse_model, set_random_seed, str2list, split_list, start_wabdb
from rich import print


def set_default_latlon(config):
    id = str2list(config.data.id)[0]
    latlon = load_frames(config.data.image_dir, config.data.gt_transform_fp, verbose=False)[2]
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
        if config.pose.scene_transform_fp:
            with open(config.pose.scene_transform_fp, "w") as f:
                json.dump(transform_dict, f, indent=4)


def gen_nvs_all(mode, model, config, scenes, ids):
    for scene in scenes:
        for id in ids:
            config.data.scene = scene
            config.data.id = id
            if mode[0]:
                print(f"[INFO] Inference Zero123 \'{scene}\':{id}")
            elif mode[1]:
                print(f"[INFO] Fine-tuning \'{scene}\':{id}")
                finetune(model, reuse_lora=True, **config.finetune)
            else:
                print(f"[INFO] Inference \'{scene}\':{id}")
                model.inject_lora(
                    ckpt_fp=config.inference.lora_ckpt_fp,
                    rank=config.inference.lora_rank,
                    target_replace_module=config.inference.lora_target_replace_module,
                )
            inference(model, reuse_lora=True, **config.inference)
            inference_for_consist(model, reuse_lora=True, **config.inference)
            if not mode[0]:
                model.remove_lora()


def gen_nvs_my_finetune(mode, model, config, scenes, ids):
    for scene in scenes:
        for id in ids:
            config.data.scene = scene
            config.data.id = id
            if mode[0]:
                print(f"[INFO] Fine-tuning \'{scene}\'")
                my_finetune(model, reuse_lora=True, **config.finetune)
            else:
                print(f"[INFO] Inference \'{scene}\'")
                model.inject_lora(
                    ckpt_fp=config.inference.lora_ckpt_fp,
                    rank=config.inference.lora_rank,
                    target_replace_module=config.inference.lora_target_replace_module,
                )
            inference(model, reuse_lora=True, **config.inference)
            inference_for_consist(model, reuse_lora=True, **config.inference)
            model.remove_lora()


def gen_nvs_my_finetune_general(mode, model, config, scenes, ids, wb_run):
    if mode[0]:
        print(f"[INFO] Fine-tuning (Generalizable)")
        my_finetune_general(model, True, config, scenes, ids, wb_run)
    else:
        print(f"[INFO] Inference (Generalizable)")
        model.inject_lora(
            ckpt_fp=config.inference.lora_ckpt_fp,
            rank=config.inference.lora_rank,
            target_replace_module=config.inference.lora_target_replace_module,
        )
    for scene in (scenes if config.data.name != 'Objaverse' else scenes[-7818:]):
        for id in ids:
            config.data.scene = scene
            config.data.id = id
            inference(model, reuse_lora=True, **config.inference)
            inference_for_consist(model, reuse_lora=True, **config.inference)
    model.remove_lora()


def main(config, mode, gpu_ids):

    def worker(config, mode, scenes, ids, gpu_id, wb_run):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        model = parse_model(config.model)
        if mode[0]:
            gen_pose_all(model, config, scenes, ids)
        if mode[1] or mode[2] or mode[3]: # Zero123, ifusion, ifusion(inference only)
            gen_nvs_all(mode[1:4], model, config, scenes, ids=["0,1"])
        elif mode[4] or mode[5]:          # my, my(inference only)
            gen_nvs_my_finetune(mode[4:6], model, config, scenes, ids=["0,1"])
        elif mode[6] or mode[7]:          # my_general, my_general(inference only)
            gen_nvs_my_finetune_general(mode[6:8], model, config, scenes, ids=["0,1"], wb_run=wb_run)

    config, conf_dict = config
    gpu_ids = str2list(gpu_ids)
    if config.data.name == 'Objaverse':
        perm = list(itertools.permutations(range(12), 2))
        ids = [",".join(map(str, p)) for p in perm]
        with open(f"{config.data.root_dir}/{config.data.name}/my_valid_paths.json") as f:
            scenes: list = sorted(json.load(f))
    else:
        perm = list(itertools.permutations(range(5), 2))
        # perm = list(itertools.combinations(range(5), 2))
        ids = [",".join(map(str, p)) for p in perm]
        scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))[0:]
        config.inference.test_transform_fp = None # reassure should be null except generalizable on Objaverse
    wb_run = start_wabdb(config, conf_dict, mode)
    print(f"[INFO] Found {len(scenes)} scenes")

    # split scenes and multi-process
    scenes = split_list(scenes, len(gpu_ids))
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = multiprocessing.Process(target=worker, args=(config, mode, scenes[i], ids, gpu_id, wb_run))
        processes.append(p)
        p.start()
    # print(f"[INFO] Started {len(processes)} processes")
    for p in processes:
        p.join()

    if wb_run:
        wb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, default="config/main.yaml")
    parser.add_argument('-p', "--pose", action="store_true")
    parser.add_argument('-1', "--zero123", action="store_true")
    parser.add_argument('-2', "--nvs", action="store_true")
    parser.add_argument('-3', "--infer", action="store_true")
    parser.add_argument('-4', "--m_nvs", action="store_true")
    parser.add_argument('-5', "--m_infer", action="store_true")
    parser.add_argument('-6', "--mgen_nvs", action="store_true")
    parser.add_argument('-7', "--mgen_infer", action="store_true")
    parser.add_argument('-g', "--gpu_ids", type=str, default="0")
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    set_random_seed(config[0].seed)
    main(
        config,
        [args.pose, args.zero123, args.nvs, args.infer, args.m_nvs, args.m_infer, args.mgen_nvs, args.mgen_infer],
        args.gpu_ids
    )
