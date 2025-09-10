import json
from glob import glob
import os

import numpy as np
import torch
from einops import rearrange
from liegroups.torch import SE3
from tqdm import trange

from dataset.finetune import FinetuneIterableDataset, MyFinetuneIterableDataset
from dataset.inference import MultiImageInferenceDataset, SingleImageInferenceDataset
from util.pose import latlon2mat, make_T, mat2latlon
from util.typing import *
from util.util import load_image, parse_optimizer, parse_scheduler, str2list
from util.viz import plot_image
from PIL import Image

def dualway_optimize_pose_loop(
    model,
    image_cond: Float[Tensor, "2 3 256 256"],
    image_target: Float[Tensor, "2 3 256 256"],
    T: Float[Tensor, "4 4"],
    default_radius: float,
    search_radius_range: float,
    use_step_ratio: bool,
    args,
    **kwargs,
):
    # init xi in se(3)
    xi = torch.randn(6) * 1e-6
    xi.requires_grad_()
    optimizer = parse_optimizer(args.optimizer, [xi])
    scheduler = parse_scheduler(args.scheduler, optimizer)

    total_loss = 0.0
    with trange(args.max_step,ncols=140) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # se(3) -> SE(3)
            T_delta = SE3.exp(xi).as_matrix()
            T_ = T @ T_delta

            latlon = mat2latlon(T_).squeeze()
            theta, azimuth = latlon[0], latlon[1]
            distance = (
                torch.sin(torch.norm(T_[:3, 3]) - default_radius) * search_radius_range
                # torch.clamp(torch.norm(T_[:3, 3]) - default_radius,min= -search_radius_range, max= search_radius_range)
                # torch.norm(T_[:3, 3]) - default_radius
            ) # distance is the offset from the default radius, but why sin and times scale?

            # idx = [0, 1] if torch.rand(1) < 0.5 else [1, 0] # on every step randomly choose train r2q or q2r. Why not both?
            idx = [0, 1]  # on every step randomly choose train r2q or q2r. Why not both?
            batch = {
                "image_cond": image_cond[idx],
                "image_target": image_target[idx],
                "T": torch.stack(
                    (
                        make_T(theta, azimuth, distance),
                        make_T(-theta, -azimuth, -distance),
                    )
                )[idx].to(model.device),
            }

            if use_step_ratio:
                loss = model(batch, step_ratio=step / args.max_step)
            else:
                loss = model(batch)
            total_loss = total_loss + loss
            # region inv_batch
            idx = [1, 0]
            batch = {
                "image_cond": image_cond[idx],
                "image_target": image_target[idx],
                "T": torch.stack(
                    (
                        make_T(theta, azimuth, distance),
                        make_T(-theta, -azimuth, -distance),
                    )
                )[idx].to(model.device),
            }

            if use_step_ratio:
                inv_loss = model(batch, step_ratio=step / args.max_step)
            else:
                inv_loss = model(batch)
            # endregion
            total_loss = total_loss + inv_loss

            pbar.set_description(
                f"lr: {scheduler.get_last_lr()[0]:.3f}, total_loss: {total_loss:.3f}, loss: {loss.item():.2f}, theta: {theta.rad2deg().item():.2f}, azimuth: {azimuth.rad2deg().item():.2f}, distance: {distance.item():.2f}"
            )
            (loss+inv_loss).backward()
            optimizer.step()
            scheduler.step(total_loss)

    return total_loss, theta, azimuth, distance

def original_optimize_pose_loop(
    model,
    image_cond: Float[Tensor, "2 3 256 256"],
    image_target: Float[Tensor, "2 3 256 256"],
    T: Float[Tensor, "4 4"],
    default_radius: float,
    search_radius_range: float,
    use_step_ratio: bool,
    args,
    **kwargs,
):
    # init xi in se(3)
    xi = torch.randn(6) * 1e-6
    xi.requires_grad_()
    optimizer = parse_optimizer(args.optimizer, [xi])
    scheduler = parse_scheduler(args.scheduler, optimizer)

    total_loss = 0.0
    with trange(args.max_step) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # se(3) -> SE(3)
            T_delta = SE3.exp(xi).as_matrix()
            T_ = T @ T_delta

            latlon = mat2latlon(T_).squeeze()
            theta, azimuth = latlon[0], latlon[1]
            distance = (
                torch.sin(torch.norm(T_[:3, 3]) - default_radius) * search_radius_range
            )

            idx = [0, 1] if torch.rand(1) < 0.5 else [1, 0]
            batch = {
                "image_cond": image_cond[idx],
                "image_target": image_target[idx],
                "T": torch.stack(
                    (
                        make_T(theta, azimuth, distance),
                        make_T(-theta, -azimuth, -distance),
                    )
                )[idx].to(model.device),
            }

            if use_step_ratio:
                loss = model(batch, step_ratio=step / args.max_step)
            else:
                loss = model(batch)

            total_loss += loss

            pbar.set_description(
                f"step: {step}, total_loss: {total_loss:.4f}, loss: {loss.item():.2f}, theta: {theta.rad2deg().item():.2f}, azimuth: {azimuth.rad2deg().item():.2f}, distance: {distance.item():.2f}"
            )

            loss.backward()
            optimizer.step()
            scheduler.step(total_loss)

    return total_loss, theta, azimuth, distance

def optimize_pose_pair(
    model,
    ref_image: Float[Tensor, "1 3 256 256"],
    qry_image: Float[Tensor, "1 3 256 256"],
    init_latlon: List[List],
    **kwargs,
):
    image_cond = torch.cat((ref_image, qry_image)).to(model.device)
    image_target = torch.cat((qry_image, ref_image)).to(model.device)
    init_T = latlon2mat(torch.tensor(init_latlon))
    results = []
    if kwargs.get('use_dualway', False):
        print("[INFO] Using dual-way optimization")
        optimize_pose_loop_fn = dualway_optimize_pose_loop
    else:
        print("[INFO] Using original optimization")
        optimize_pose_loop_fn = original_optimize_pose_loop

    for T in init_T:
        total_loss, theta, azimuth, distance = optimize_pose_loop_fn(
            model,
            image_cond=image_cond,
            image_target=image_target,
            T=T,
            **kwargs,
        )

        results.append(
            (
                total_loss.item(),
                theta.rad2deg().item(),
                azimuth.rad2deg().item(),
                distance.item(),
            )
        )

    results = torch.tensor(results)
    best_idx = torch.argmin(results[:, 0])
    pred_pose = results[best_idx][1:]
    pred_loss = results[best_idx][0]
    print(
        f"[INFO] Best pose: theta: {pred_pose[0]:.2f}, azimuth: {pred_pose[1]:.2f}, distance: {pred_pose[2]:.2f}"
    )

    return pred_pose, pred_loss.item()


def optimize_pose(
    model,
    transform_dict: Dict,
    image_dir: str,
    transform_fp: str,
    demo_fp: str,
    id: str,
    default_latlon: List[float] = [0, 0, 1],
    **kwargs,
):
    image_fps = sorted(glob(image_dir + "/*.png") + glob(image_dir + "/*.jpg"))
    image_fps = [fp for fp in image_fps if fp != demo_fp]

    id = list(range(len(image_fps))) if id == "all" else str2list(id)
    ref_image = load_image(image_fps[id[0]])
    qry_images = [load_image(image_fps[i]) for i in id[1:]]

    out_dict = {"camera_angle_x": np.deg2rad(49.1), "frames": []}
    out_dict["frames"].append(
        {
            "file_path": image_fps[0].replace(image_dir + "/", ""),
            "transform_matrix": latlon2mat(torch.tensor([default_latlon])).squeeze(0).tolist(),
            "latlon": list(default_latlon),
        }
    )
    if len(transform_dict["frames"]) == 0:
        transform_dict["frames"].append(
        {
            "file_path": image_fps[0].replace(image_dir + "/", ""),
            "transform_matrix": latlon2mat(torch.tensor([default_latlon])).squeeze(0).tolist(),
            "latlon": list(default_latlon),
            "loss": 0.0,
        }
    )
    for qry_idx, qry_image in zip(id[1:], qry_images):
        assert ref_image.shape == qry_image.shape
        pose, loss = optimize_pose_pair(
            model=model, ref_image=ref_image, qry_image=qry_image, **kwargs
        )
        pose = np.add(default_latlon, pose.unsqueeze(0))
        out_dict["frames"].append(
            {
                "file_path": image_fps[qry_idx].replace(image_dir + "/", ""),
                "transform_matrix": latlon2mat(pose.clone()).squeeze(0).tolist(),
                "latlon": pose.squeeze().tolist(),
            }
        )
        if qry_idx < len(transform_dict["frames"]):
            if qry_idx != 0 and transform_dict["frames"][qry_idx]["loss"] > loss:
                transform_dict["frames"][qry_idx] = {
                    "file_path": image_fps[qry_idx].replace(image_dir + "/", ""),
                    "transform_matrix": latlon2mat(pose.clone()).squeeze(0).tolist(),
                    "latlon": pose.squeeze().tolist(),
                    "loss": loss,
                }
        else:
            transform_dict["frames"].append(
                {
                    "file_path": image_fps[qry_idx].replace(image_dir + "/", ""),
                    "transform_matrix": latlon2mat(pose.clone()).squeeze(0).tolist(),
                    "latlon": pose.squeeze().tolist(),
                    "loss": loss,
                }
            )

    # save poses to json
    os.makedirs(os.path.dirname(transform_fp), exist_ok=True)
    with open(transform_fp, "w") as f:
        json.dump(out_dict, f, indent=4)
    return transform_dict


def finetune(
    model,
    image_dir: str,
    transform_fp: str,
    lora_ckpt_fp: str,
    lora_rank: int,
    lora_target_replace_module: List[str],
    args,
):
    model.inject_lora(
        rank=lora_rank,
        target_replace_module=lora_target_replace_module,
    )

    train_dataset = FinetuneIterableDataset(image_dir, transform_fp)
    train_loader = train_dataset.loader(args.batch_size)
    optimizer = parse_optimizer(args.optimizer, model.require_grad_params)
    scheduler = parse_scheduler(args.scheduler, optimizer)

    train_loader = iter(train_loader)
    with trange(args.max_step, ncols=140) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            batch = next(train_loader)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(batch)

            pbar.set_description(f"step: {step}, loss: {loss.item():.4f}")
            loss.backward()

            optimizer.step()
            scheduler.step()

    os.makedirs(os.path.dirname(lora_ckpt_fp), exist_ok=True)
    model.save_lora(lora_ckpt_fp)
    model.remove_lora()

def my_finetune(
    model,
    scenes: List[str],
    ids: List[str],
    image_dir: str,
    transform_fp: str,
    lora_ckpt_fp: str,
    lora_rank: int,
    lora_target_replace_module: List[str],
    args,
):
    model.inject_lora(
        rank=lora_rank,
        target_replace_module=lora_target_replace_module,
    )
    
    # rand scenes index -> batch size scene
    # rand ids index -> 1 for all scene (2 index, 1 for ref., 1 for target)
    # run model from another ids (same scene, same target) to get the loss
    # scene_idxs = torch.randint(0, len(scenes), (args.batch_size,), device=model.device)
    # img_idxs, next_img_idxs = torch.randint(0, len(ids), (2), device=model.device)
    # next_img_idxs[1] = img_idxs[1]  # make sure the second image has the same idx for all scenes

    train_dataset = MyFinetuneIterableDataset(image_dir, transform_fp)
    train_loader = train_dataset.loader(args.batch_size)
    optimizer = parse_optimizer(args.optimizer, model.require_grad_params)
    scheduler = parse_scheduler(args.scheduler, optimizer)
    
    # from met3r import MEt3R
    # met3r_eval = MEt3R(
    #     img_size=256, # Default to 256, set to `None` to use the input resolution on the fly!
    #     use_norm=True, # Default to True 
    #     backbone="mast3r", # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
    #     feature_backbone="dino16", # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
    #     feature_backbone_weights="mhamilton723/FeatUp", # Default
    #     upsampler="featup", # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
    #     distance="cosine", # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
    #     freeze=True, # Default to True
    # ).cuda()
    train_loader = iter(train_loader)
    with trange(args.max_step, ncols=160) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            batch = next(train_loader)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            noise_a, noise_b, noise, nvs_latent_a, nvs_latent_b = model(batch)
            consist_loss    = torch.nn.functional.mse_loss(noise_a, noise_b, reduction='mean')
            noise_pred_loss = torch.nn.functional.mse_loss(noise_a, noise, reduction='mean') + torch.nn.functional.mse_loss(noise_b, noise, reduction='mean')
            # region      commented      
            # if step%5==0:
            #     image1 = model.decode_latent(nvs_latent_a).detach()[0]
            #     image2 = model.decode_latent(nvs_latent_b).detach()[0]
            #     print(f"[INFO] image1: {image1.shape}")
            #     Image.fromarray((image1.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).save(f'{exp_dir}/decode_img1_{step}.png')
            #     Image.fromarray((image2.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).save(f'{exp_dir}/decode_img2_{step}.png')

            # score, mask, score_map,_ = met3r_eval(
            #     images=inputs, 
            #     return_overlap_mask=True, # Default 
            #     return_score_map=True, # Default 
            #     return_projections=True # Default 
            # )
            # loss += torch.nn.functional.mse_loss(nvs_latent_b, noise, reduction='mean')
            # loss = inconsistency(nvs_latent_a, nvs_latent_b) # l2 or met3r
            # endregion
            consist_loss    = args.consist_loss_ratio * consist_loss
            noise_pred_loss = args.pred_loss_ratio * noise_pred_loss
            loss = consist_loss + noise_pred_loss
            pbar.set_description(f"step: {step}, loss: {loss.item():.4f}, c_loss: {consist_loss.item():.4f}, p_loss: {noise_pred_loss.item():.4f}")
            loss.backward()

            optimizer.step()
            scheduler.step()
            # scheduler.step(loss)

    os.makedirs(os.path.dirname(lora_ckpt_fp), exist_ok=True)
    model.save_lora(lora_ckpt_fp)
    model.remove_lora()

def inference(
    model,
    image_dir: str,
    transform_fp: str,
    test_transform_fp: str,
    lora_ckpt_fp: str,
    demo_fp: str,
    lora_rank: int,
    lora_target_replace_module: List[str],
    use_single_view: bool,
    use_multi_view_condition: bool,
    n_views: int,
    theta: float,
    radius: float,
    args,
):
    if not use_single_view and lora_ckpt_fp:
        model.inject_lora(
            ckpt_fp=lora_ckpt_fp,
            rank=lora_rank,
            target_replace_module=lora_target_replace_module,
        )

    if not use_single_view and use_multi_view_condition:
        test_dataset = MultiImageInferenceDataset
        generate_fn = model.generate_from_tensor_multi_cond
    else:
        test_dataset = SingleImageInferenceDataset
        generate_fn = model.generate_from_tensor

    test_dataset = test_dataset(
        image_dir=image_dir, transform_fp=transform_fp, test_transform_fp=test_transform_fp, n_views=n_views, theta=theta, radius=radius
    )
    test_loader = test_dataset.loader(args.batch_size)
    for batch in test_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = generate_fn(
            image=batch["image_cond"],
            theta=batch["theta"],
            azimuth=batch["azimuth"],
            distance=batch["distance"],
        )

    if lora_ckpt_fp:
        model.remove_lora()

    # plot_image(out, fp=demo_fp)
    out = rearrange(out, "b c h w -> c h (b w)")
    plot_image(out, fp=demo_fp)
    print(f"[INFO] Saved image to {demo_fp}")

    return out