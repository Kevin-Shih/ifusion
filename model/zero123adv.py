import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from einops import rearrange
from omegaconf import OmegaConf

from ldm.lora import (
    inject_trainable_lora_extended,
    monkeypatch_remove_lora,
    save_lora_weight,
)
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import load_model_from_config
from util.pose import make_T
from util.typing import *
from util.util import default
from model.zero123 import Zero123

class Zero123adv(Zero123, nn.Module):
    def configure(self) -> None:
        print("[INFO] Loading Zero123adv...")

        self.pretrained_config = OmegaConf.load(self.config.pretrained_config)
        self.weights_dtype = torch.float32
        self.model: LatentDiffusion = load_model_from_config(
            self.pretrained_config,
            self.config.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.config.vram_O,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.pretrained_config.model.params.timesteps
        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.pretrained_config.model.params.linear_start,
            self.pretrained_config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps(
            min_step_percent=self.config.min_step_percent,
            max_step_percent=self.config.max_step_percent,
        )

        print("[INFO] Loaded Zero123adv")

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, batch, ddpm_step=None, bs=None, noise=None, uncond=0.05, scale=3, ddim_eta=1, max_ddim_steps=50):
        # region get_input()
        image_target = batch["image_target"]
        if len(image_target.shape) == 3:
            image_target = image_target[..., None]
        image_target = image_target.to(self.model.device, memory_format=torch.contiguous_format).float()
        T1 = batch["T1"].to(self.model.device, memory_format=torch.contiguous_format).float()
        T2 = batch["T2"].to(self.model.device, memory_format=torch.contiguous_format).float()
        if bs is not None:
            image_target = image_target[:bs]
            T1 = T1[:bs]
            T2 = T2[:bs]

        encoder_posterior = self.model.encode_first_stage(image_target)
        target_latent = self.model.get_first_stage_encoding(encoder_posterior).detach()
        image_cond1 = batch["image_cond1"].to(self.model.device)
        image_cond2 = batch["image_cond2"].to(self.model.device)
        if bs is not None:
            image_cond1 = image_cond1[:bs]
            image_cond2 = image_cond2[:bs]
        cond1 = {}
        cond2 = {}

        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        random = torch.rand(image_target.size(0), device=self.model.device)
        prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1" )
        # print(f'[INFO] image_target shape: {batch["image_target"].shape}, T1 shape: {batch["T1"].shape}, target_latent shape: {target_latent.shape}')

        # target_latent.shape: [8, 4, 64, 64]; c.shape: [8, 1, 768] b c3 h32 w32
        # print('=========== xc shape ===========', xc.shape)
        with torch.enable_grad():
            clip_emb1 = self.model.get_learned_conditioning(image_cond1).detach()
            clip_emb2 = self.model.get_learned_conditioning(image_cond2).detach()
            null_prompt = self.model.get_learned_conditioning([""]).detach()
            cond1["c_crossattn"] = [
                self.model.cc_projection(
                    torch.cat(
                        [
                            torch.where(prompt_mask, null_prompt, clip_emb1),
                            T1[:, None, :],
                        ],
                        dim=-1,
                    )
                )
            ]
            cond2["c_crossattn"] = [
                self.model.cc_projection(
                    torch.cat(
                        [
                            torch.where(prompt_mask, null_prompt, clip_emb2),
                            T2[:, None, :],
                        ],
                        dim=-1,
                    )
                )
            ]
        cond1["c_concat"] = [input_mask * self.model.encode_first_stage(image_cond1).mode().detach()]
        cond2["c_concat"] = [input_mask * self.model.encode_first_stage(image_cond2).mode().detach()]
        # endregion
        ddpm_step = default(ddpm_step, torch.randint(low=0, high=self.model.num_timesteps, size=(1,)).item())
        # ddpm_step = 50
        ddim_step = (ddpm_step * max_ddim_steps) // self.model.num_timesteps
        t = torch.full((target_latent.shape[0],), ddpm_step, device=self.model.device).long()

        noise = default(noise, lambda: torch.randn_like(target_latent))
        x_noisy = self.model.q_sample(x_start=target_latent, t=t, noise=noise)
        noise_pred1 = self.model.apply_model(x_noisy, t, cond1)
        noise_pred2 = self.model.apply_model(x_noisy, t, cond2)

        self.scheduler.set_timesteps(max_ddim_steps)
        latent1 = self.scheduler.step(noise_pred1, ddim_step, target_latent, eta=ddim_eta)["pred_original_sample"]
        latent2 = self.scheduler.step(noise_pred2, ddim_step, target_latent, eta=ddim_eta)["pred_original_sample"]
        # latent_loss = torch.nn.functional.mse_loss(latent1, latent2, reduction='none').mean([1, 2, 3])
        # image1 = self.decode_latent(latent1)
        # image2 = self.decode_latent(latent2)
        return noise_pred1, noise_pred2, noise, latent1, latent2