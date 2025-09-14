import importlib
import random
from inspect import isfunction
import cv2
import wandb
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from typing import Optional

str2list = lambda x: list(map(int, x.split(",")))


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    return conf, OmegaConf.to_container(conf, resolve=True)


def parse_optimizer(config, params):
    optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler(config, optim):
    scheduler = getattr(torch.optim.lr_scheduler, config.name)(optim, **config.args)
    return scheduler


def parse_model(config):
    models = importlib.import_module("model." + config.name)
    model = getattr(models, config.name[0].upper() + config.name[1:])(**config.args)
    return model


def load_image(fp, resize=True, to_clip=True, verbose=True, device="cuda"):
    if verbose:
        print(f"[INFO] Loading image {fp}")

    image = np.array(Image.open(fp))
    if image.shape[-1] == 4:
        image[image[..., -1] < 128] = [255] * 4
        image = image[..., :3]

    if resize:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).contiguous().to(device)
    image = image.permute(2, 0, 1).unsqueeze(0)

    if to_clip:
        image = image * 2 - 1
    return image


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def start_wabdb(config, conf_dict, mode, eval=False) -> Optional[wandb.Run]:
    curr_time = time.localtime(time.time())
    mon, mday, hours = curr_time.tm_mon, curr_time.tm_mday, curr_time.tm_hour
    mins = curr_time.tm_min + curr_time.tm_sec / 60
    if config.log is None:
        wb_run = None
    elif config.log.run_path:
        print(f'[INFO] Resuming wandb run from \'{config.log.run_path}\'. Ignoring group_name and run_name arguments.')
        wb_api_run = wandb.Api().run(f'kevin-shih/iFusion-Adv/{config.log.run_path.split("/")[-1]}')
        print(f'Confirm Resuming from \'{wb_api_run.name}\', id:\'{wb_api_run.id}\'? [Y/N]')
        user_input = input()
        if user_input.lower() in ('y', 'yes'):
            wb_run = wandb.init(
                dir="../wandb",
                entity="kevin-shih",
                project="iFusion-Adv",
                id=f"{config.log.run_path.split('/')[-1]}",
                resume="must",
                settings=wandb.Settings(x_disable_stats=True, x_save_requirements=False),
                config={
                    'mode': {
                        'pose':
                            wb_api_run.config.mode.pose or mode[0],
                        'zero123_nvs':
                            wb_api_run.config.mode.zero123_nvs or (mode[1] and config.finetune is None),
                        'iFusion_finetune':
                            wb_api_run.config.mode.iFusion_finetune or (mode[1] and config.finetune is not None),
                        'my_finetune':
                            wb_api_run.config.mode.my_finetune or mode[2],
                    },
                },
            )
        else:
            print(f'Canceled: Abort execution.')
            exit(0)
    else:
        wb_run = wandb.init(
            dir="./wandb",
            entity="kevin-shih",
            project="iFusion-Adv",
            group=f'{config.log.group_name}',
            name=f'{config.log.run_name}',
            settings=wandb.Settings(x_disable_stats=True, x_save_requirements=False),
            config={
                "start_date": f'{mon:02d}-{mday:02d}',
                "start_time": f'{hours:02d}-{mins:04.1f}',
                'mode': {
                    'pose': mode[0],
                    'zero123_nvs': mode[1] and config.finetune is None,
                    'iFusion_finetune': mode[1] and config.finetune is not None,
                    'my_finetune': mode[2],
                },
                **conf_dict,
            },
        )
    return wb_run
