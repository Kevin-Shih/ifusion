import json
import os
import wandb
import torch
import numpy as np
from einops import rearrange
from tqdm import trange
from glob import glob
from liegroups.torch import SE3
from rich import print

from dataset.finetune import FinetuneIterableDataset, MyFinetuneIterableDataset, MyFinetuneAllSceneIterableDataset
from dataset.inference import MultiImageInferenceDataset, SingleImageInferenceDataset
from util.pose import latlon2mat, make_T, mat2latlon
from util.typing import *
from util.util import load_image, parse_optimizer, parse_scheduler, str2list
from util.viz import plot_image


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
    with trange(args.max_step, ncols=130) as pbar:
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
            )                                                                           # distance is the offset from the default radius, but why sin and times scale?

            # idx = [0, 1] if torch.rand(1) < 0.5 else [1, 0] # on every step randomly choose. Why not both?
            idx = [0, 1]
            batch = {
                "image_cond":
                    image_cond[idx],
                "image_target":
                    image_target[idx],
                "T":
                    torch.stack((
                        make_T(theta, azimuth, distance),
                        make_T(-theta, -azimuth, -distance),
                    ))[idx].to(model.device),
            }

            if use_step_ratio:
                loss = model(batch, step_ratio=step / args.max_step)
            else:
                loss = model(batch)
            total_loss = total_loss + loss
            # region inv_batch
            idx = [1, 0]
            batch = {
                "image_cond":
                    image_cond[idx],
                "image_target":
                    image_target[idx],
                "T":
                    torch.stack((
                        make_T(theta, azimuth, distance),
                        make_T(-theta, -azimuth, -distance),
                    ))[idx].to(model.device),
            }

            if use_step_ratio:
                inv_loss = model(batch, step_ratio=step / args.max_step)
            else:
                inv_loss = model(batch)
            # endregion
            total_loss = total_loss + inv_loss

            pbar.set_description(
                f"total_loss: {total_loss:.3f}, loss: {loss.item():.2f}, theta: {theta.rad2deg().item():.2f}, azimuth: {azimuth.rad2deg().item():.2f}, distance: {distance.item():.2f}"
            )
            (loss + inv_loss).backward()
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
    with trange(args.max_step, ncols=130) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # se(3) -> SE(3)
            T_delta = SE3.exp(xi).as_matrix()
            T_ = T @ T_delta

            latlon = mat2latlon(T_).squeeze()
            theta, azimuth = latlon[0], latlon[1]
            distance = (torch.sin(torch.norm(T_[:3, 3]) - default_radius) * search_radius_range)

            idx = [0, 1] if torch.rand(1) < 0.5 else [1, 0]
            batch = {
                "image_cond":
                    image_cond[idx],
                "image_target":
                    image_target[idx],
                "T":
                    torch.stack((
                        make_T(theta, azimuth, distance),
                        make_T(-theta, -azimuth, -distance),
                    ))[idx].to(model.device),
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

        results.append((
            total_loss.item(),
            theta.rad2deg().item(),
            azimuth.rad2deg().item(),
            distance.item(),
        ))

    results = torch.tensor(results)
    best_idx = torch.argmin(results[:, 0])
    pred_pose = results[best_idx][1:]
    pred_loss = results[best_idx][0]
    print(f"[INFO] Best pose: theta: {pred_pose[0]:.2f}, azimuth: {pred_pose[1]:.2f}, distance: {pred_pose[2]:.2f}")

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
    out_dict["frames"].append({
        "file_path": image_fps[0].replace(image_dir + "/", ""),
        "transform_matrix": latlon2mat(torch.tensor([default_latlon])).squeeze(0).tolist(),
        "latlon": list(default_latlon),
    })
    if len(transform_dict["frames"]) == 0:
        transform_dict["frames"].append({
            "file_path": image_fps[0].replace(image_dir + "/", ""),
            "transform_matrix": latlon2mat(torch.tensor([default_latlon])).squeeze(0).tolist(),
            "latlon": list(default_latlon),
            "loss": 0.0,
        })
    for qry_idx, qry_image in zip(id[1:], qry_images):
        assert ref_image.shape == qry_image.shape
        pose, loss = optimize_pose_pair(model=model, ref_image=ref_image, qry_image=qry_image, **kwargs)
        pose = np.add(default_latlon, pose.unsqueeze(0))
        out_dict["frames"].append({
            "file_path": image_fps[qry_idx].replace(image_dir + "/", ""),
            "transform_matrix": latlon2mat(pose.clone()).squeeze(0).tolist(),
            "latlon": pose.squeeze().tolist(),
        })
        if qry_idx < len(transform_dict["frames"]):
            if qry_idx != 0 and transform_dict["frames"][qry_idx]["loss"] > loss:
                transform_dict["frames"][qry_idx] = {
                    "file_path": image_fps[qry_idx].replace(image_dir + "/", ""),
                    "transform_matrix": latlon2mat(pose.clone()).squeeze(0).tolist(),
                    "latlon": pose.squeeze().tolist(),
                    "loss": loss,
                }
        else:
            transform_dict["frames"].append({
                "file_path": image_fps[qry_idx].replace(image_dir + "/", ""),
                "transform_matrix": latlon2mat(pose.clone()).squeeze(0).tolist(),
                "latlon": pose.squeeze().tolist(),
                "loss": loss,
            })

    # save poses to json
    os.makedirs(os.path.dirname(transform_fp), exist_ok=True)
    with open(transform_fp, "w") as f:
        json.dump(out_dict, f, indent=4)
    return transform_dict


def finetune(
    model,
    reuse_lora: bool,
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
    with trange(args.max_step, ncols=130) as pbar:
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
    if not reuse_lora:
        model.remove_lora()


def my_finetune(
    model,
    reuse_lora: bool,
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

    train_dataset = MyFinetuneIterableDataset(image_dir, transform_fp)
    train_loader = train_dataset.loader(args.batch_size)
    optimizer = parse_optimizer(args.optimizer, model.require_grad_params)
    scheduler = parse_scheduler(args.scheduler, optimizer)
    train_loader = iter(train_loader)
    ddpm_steps = args.get('ddpm_steps', None)
    if ddpm_steps and len(ddpm_steps) == 2:
        print('[Info] DDIM step range:', ddpm_steps[0], ddpm_steps[1])
        ddpm_step_fn = lambda step: int(ddpm_steps[0] - step * (ddpm_steps[0] - ddpm_steps[1]) / args.max_step)
    else:
        print('[Info] DDIM step:', ddpm_steps)
        ddpm_step_fn = lambda step: int(ddpm_steps[0])

    grad_accumelation = args.get('gradient_accumelation', 1)
    with trange(args.max_step, ncols=130) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            for _ in range(grad_accumelation):
                batch = next(train_loader)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                noise_a, noise_b, noise, nvs_latent_a, nvs_latent_b = model(batch, ddpm_step=ddpm_step_fn(step))
                consist_loss = torch.nn.functional.mse_loss(noise_a, noise_b, reduction='mean')                                   # or cosine?
                noise_pred_loss = torch.nn.functional.mse_loss(
                    noise_a, noise, reduction='mean'
                ) + torch.nn.functional.mse_loss(
                    noise_b, noise, reduction='mean'
                )
                loss = (
                    consist_loss * args.consist_loss_ratio + noise_pred_loss * args.pred_loss_ratio
                ) / grad_accumelation
                loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(
                f"step: {step}, loss: {loss.item():.4f}, c_loss: {consist_loss.item():.4f}, p_loss: {noise_pred_loss.item():.4f}"
            )

    os.makedirs(os.path.dirname(lora_ckpt_fp), exist_ok=True)
    model.save_lora(lora_ckpt_fp)
    if not reuse_lora:
        model.remove_lora()


def my_finetune_general(
    model,
    reuse_lora: bool,
    config: Dict,
    scenes: List[str],
    ids: List[str],
    wb_run: wandb.Run,
):
    if not wb_run:
        print("[ERROR] wandb logging is requiered for gerneralizable finetuning.")
        exit(1)
    args: dict = config.finetune.args
    model.inject_lora(
        ckpt_fp=args.get('resume_ckpt_fp', None),
        rank=config.finetune.lora_rank,
        target_replace_module=config.finetune.lora_target_replace_module,
    )

    # rand scenes index -> batch size scene
    # rand ids index -> 1 for all scene (2 index, 1 for ref., 1 for target)
    # run model from another ids (same scene, same target) to get the loss
    # scene_idxs = torch.randint(0, len(scenes), (args.batch_size,), device=model.device)
    # img_idxs, next_img_idxs = torch.randint(0, len(ids), (2), device=model.device)
    # next_img_idxs[1] = img_idxs[1]  # make sure the second image has the same idx for all scenes
    if config.data.name == 'Objaverse':
        train_scenes = scenes[:-7818]
        eval_scenes = scenes[-7818::10]
    else:
        train_scenes = scenes
        eval_scenes = scenes[::10]
    train_dataset = MyFinetuneAllSceneIterableDataset(len(train_scenes))
    for scene in train_scenes:
        for id in ids:
            config.data.scene = scene
            config.data.id = id
            # print(f"[INFO] Adding scene {scene}, id {id}")
            train_dataset.add_scenes(config.finetune.image_dir, config.finetune.transform_fp, verbose=False)

    train_loader = train_dataset.loader(args.batch_size)
    optimizer = parse_optimizer(args.optimizer, model.require_grad_params)
    scheduler = parse_scheduler(args.scheduler, optimizer)
    ddpm_milestones = args.scheduler.args.milestones
    ddpm_steps = args.get('ddpm_steps', None)
    # if ddpm_steps and len(ddpm_steps) <= len(ddpm_milestones):
    #     ddpm_steps.extend(ddpm_steps[-1] * (len(ddpm_milestones) + 1 - len(ddpm_steps)))
    print('[Info] DDIM step range:', ddpm_steps)
    ddpm_steps_iter = iter(ddpm_steps)
    ddpm_step = next(ddpm_steps_iter)

    train_loader = iter(train_loader)
    grad_accumelation = args.get('gradient_accumelation', 1)
    eval_steps = args.get('eval_steps', 50)
    with trange(args.max_step, ncols=130) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            for _ in range(grad_accumelation):
                batch = next(train_loader)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                noise_a, noise_b, noise, nvs_latent_a, nvs_latent_b = model(batch, ddpm_step=ddpm_step)
                consist_loss = torch.nn.functional.mse_loss(noise_a, noise_b, reduction='mean')
                noise_pred_loss = torch.nn.functional.mse_loss(
                    noise_a, noise, reduction='mean'
                ) + torch.nn.functional.mse_loss(
                    noise_b, noise, reduction='mean'
                )
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
                loss = (
                    consist_loss * args.consist_loss_ratio + noise_pred_loss * args.pred_loss_ratio
                ) / grad_accumelation
                loss.backward()
            optimizer.step()
            scheduler.step()
            # scheduler.step(loss)

            pbar.set_description(
                f"ddpm_step: {ddpm_step}, loss: {loss.item()*grad_accumelation:.4f}, c_loss: {consist_loss.item():.4f}, p_loss: {noise_pred_loss.item():.4f}"
            )
            wb_run.log({
                "train/loss": loss.item(),
                "train/consist_loss": consist_loss.item(),
                "train/noise_pred_loss": noise_pred_loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
            },
                       step=step + 1)

            if (step + 1) % eval_steps == 0:
                lora_ckpt_fp = f'{config.data.nvs_root}/{config.data.name}/lora/lora_{step + 1}.ckpt'
                metric = ckpt_infer_and_eval(model, config, eval_scenes, ids, lora_ckpt_fp=lora_ckpt_fp)
                PSNR_mean = np.mean(metric[:, 0])
                SSIM_mean = np.mean(metric[:, 1])
                LPIPS_mean = np.mean(metric[:, 2])
                wb_run.log({
                    "eval/PSNR": PSNR_mean,
                    "eval/SSIM": SSIM_mean,
                    "eval/LPIPS": LPIPS_mean,
                }, step=step + 1)
            if step in ddpm_milestones:
                ddpm_step = next(ddpm_steps_iter, ddpm_steps[-1])
    model.save_lora(config.data.lora_ckpt_fp)
    if not reuse_lora:
        model.remove_lora()


def ckpt_infer_and_eval(model, config, scenes, ids, lora_ckpt_fp):
    os.makedirs(os.path.dirname(lora_ckpt_fp), exist_ok=True)
    print()
    model.save_lora(lora_ckpt_fp)
    # print(f"[INFO] Evaluating 1/10 scenes")
    metric = []
    # config.data.lora_ckpt_fp = lora_ckpt_fp
    from eval import eval_nvs
    for scene in scenes:                                          # infer 1/10 scenes
        for id in ids:
            config.data.scene = scene
            config.data.id = id
            inference(model, reuse_lora=True, **config.inference) # inference if is objaverse should use testtransformfp
            metric.append(eval_nvs(**config.data))
    metric = np.array(metric)
    return metric


def inference(
    model,
    reuse_lora: bool,
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
    if lora_ckpt_fp and not reuse_lora:
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
        image_dir=image_dir,
        transform_fp=transform_fp,
        test_transform_fp=test_transform_fp,
        n_views=n_views,
        theta=theta,
        radius=radius
    )
    test_loader = test_dataset.loader(args.batch_size)
    for batch in test_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out_colmap = generate_fn(
            image=batch["image_cond"],
            theta=batch["theta"],
            azimuth=batch["azimuth"],
            distance=batch["distance"],
        )

    if lora_ckpt_fp and not reuse_lora:
        model.remove_lora()

    os.makedirs(os.path.dirname(demo_fp), exist_ok=True)
    out = rearrange(out_colmap[::2], "b c h w -> c h (b w)")
    plot_image(out, fp=demo_fp)
    out_colmap = rearrange(out_colmap, "b c h w -> c h (b w)")
    plot_image(out_colmap, fp=demo_fp.replace('.png', '_colmap.png'))
    print(f"[INFO] Saved image to {demo_fp} and {os.path.basename(demo_fp).replace('.png', '_colmap.png')}")
    return out


def inference_for_consist(
    model,
    reuse_lora: bool,
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
    if lora_ckpt_fp and not reuse_lora:
        model.inject_lora(
            ckpt_fp=lora_ckpt_fp,
            rank=lora_rank,
            target_replace_module=lora_target_replace_module,
        )

    generate_fn = model.generate_from_tensor
    # from view 0
    test_dataset = SingleImageInferenceDataset(
        image_dir=image_dir,
        transform_fp=transform_fp,
        test_transform_fp=test_transform_fp,
        n_views=n_views,
        theta=theta,
        radius=radius,
        image_idx=0
    )
    test_loader = test_dataset.loader(args.batch_size)
    for batch in test_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out_0_16view = generate_fn(
            image=batch["image_cond"],
            theta=batch["theta"],
            azimuth=batch["azimuth"],
            distance=batch["distance"],
        )
    # from view 1
    test_dataset = SingleImageInferenceDataset(
        image_dir=image_dir,
        transform_fp=transform_fp,
        test_transform_fp=test_transform_fp,
        n_views=n_views,
        theta=theta,
        radius=radius,
        image_idx=1
    )
    test_loader = test_dataset.loader(args.batch_size)
    for batch in test_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out_1_16view = generate_fn(
            image=batch["image_cond"],
            theta=batch["theta"],
            azimuth=batch["azimuth"],
            distance=batch["distance"],
        )

    if lora_ckpt_fp and not reuse_lora:
        model.remove_lora()

    os.makedirs(os.path.dirname(demo_fp), exist_ok=True)
    out_0 = rearrange(out_0_16view[::2], "b c h w -> c h (b w)")
    plot_image(out_0, fp=demo_fp.replace('.png', '_0.png'))
    out_0_16view = rearrange(out_0_16view, "b c h w -> c h (b w)")
    plot_image(out_0_16view, fp=demo_fp.replace('.png', '_0_16view.png'))

    out_1 = rearrange(out_1_16view[::2], "b c h w -> c h (b w)")
    plot_image(out_1, fp=demo_fp.replace('.png', '_1.png'))
    out_1_16view = rearrange(out_1_16view, "b c h w -> c h (b w)")
    plot_image(out_1_16view, fp=demo_fp.replace('.png', '_1_16view.png'))
    print(f"[INFO] Saved image to {demo_fp.replace('.png', '_0.png')}")
    return out_0_16view, out_1_16view
