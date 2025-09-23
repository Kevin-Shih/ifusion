import argparse
import itertools
import os
import wandb
import torch
import json
import numpy as np

from dataset.base import load_frames
from util.util import load_config, load_image, set_random_seed, str2list, start_wabdb
from util.pose import mat2latlon
from util.colmap import patch_match_with_known_poses

from rich import print
from met3r import MEt3R
from PIL import Image
from pytorch3d import io
from typing import Optional
from einops import rearrange


def eval_pose(scene_transform_fp, transform_fp, gt_transform_fp, image_dir, id, **kwargs):
    from util.criterion import pose_err_fn
    gt_camtoworlds = load_frames(image_dir, gt_transform_fp, verbose=False)[1]
    if scene_transform_fp:
        camtoworlds = load_frames(image_dir, scene_transform_fp, verbose=False, return_images=True)[1]
    else:
        camtoworlds = load_frames(image_dir, transform_fp, verbose=False, return_images=True)[1]
        gt_camtoworlds = gt_camtoworlds[str2list(id)]
    pose_err = [pose_err_fn(pred, gt) for pred, gt in zip(camtoworlds[1:], gt_camtoworlds[1:])]
    pose_err = np.array(pose_err)
    print(f"Rot. error: {np.median(pose_err[:, 0]):.2f}, Trans. error: {np.median(pose_err[:, 1]):.2f}")
    return pose_err


def eval_nvs(demo_fp, test_image_dir, test_transform_fp, **kwargs) -> tuple[float, float, float]:
    from util.criterion import lpips_fn, psnr_fn, ssim_fn
    pred = load_image(demo_fp, resize=False, to_clip=False, verbose=False)
    pred = torch.cat(torch.chunk(pred, 8, dim=-1))
    gt = load_frames(test_image_dir, test_transform_fp, to_clip=False, verbose=False)[0]
    gt = gt.to(pred.device)

    psnr = psnr_fn(pred, gt).item()
    ssim = ssim_fn(pred, gt).item()
    lpips = lpips_fn(pred, gt).item()
    return psnr, ssim, lpips


def eval_consistency(met3r_eval, nvs_dir, demo_fp, **kwargs):
    # RGB range must be in [-1, 1], input shape B, k=2, c=3, 256, 256
    imgs = load_image(demo_fp, resize=False, verbose=False)
    imgs = torch.cat(torch.chunk(imgs, 8, dim=-1))
    inputs = []
    for i in range(imgs.shape[0] - 1):
        inputs.append(torch.Tensor(imgs[i:i + 2]))
    inputs = torch.stack(inputs).cuda()
    inputs = inputs.clip(-1, 1)
    score, mask, score_map, pcloud = met3r_eval(
        images=inputs,
        return_overlap_mask=True,
        return_score_map=True,
        return_projections=False,                # Default
        return_point_clouds=True
    )

    # pcloud : (B-1) * 2 point clouds. Each image pair has 2 point clouds (left and right)
    eval_out_dir = f'{nvs_dir}/eval/'
    os.makedirs(eval_out_dir, exist_ok=True)
    for i in range(score_map.shape[0]):
        # Image.fromarray((score_map[i].clamp(0,1).cpu().numpy() * 255).astype(np.uint8)).save(f'{eval_out_dir}/score_map_{i},{i+1}.png')
        score_rgb = torch.stack([score_map[i]] * 3, dim=-1).clamp(0, 1)
        mask_rgb = torch.stack([mask[i] + 0.3, mask[i] + 0.3, torch.full_like(mask[i], 1)], dim=-1).clamp(0, 1)
        masked_score_map = (score_rgb * mask_rgb * 255).cpu().numpy()

        Image.fromarray(masked_score_map.astype(np.uint8)).save(f'{eval_out_dir}/score_map_masked_{i},{i+1}.png')
        if met3r_eval.distance == 'cosine': # if choose cosine distance in met3r, pointclouds has no color
            io.save_ply(f'{eval_out_dir}/pointcloud_{i},{i+1}_l.ply', pcloud.points_list()[i * 2])
            io.save_ply(f'{eval_out_dir}/pointcloud_{i},{i+1}_r.ply', pcloud.points_list()[i * 2 + 1])
        else:
            io.IO().save_pointcloud(pcloud[i * 2], f'{eval_out_dir}/pointcloud_{i},{i+1}_l.ply')
            io.IO().save_pointcloud(pcloud[i * 2 + 1], f'{eval_out_dir}/pointcloud_{i},{i+1}_r.ply')

    np.set_printoptions(precision=3, suppress=None, floatmode='fixed')
    score: np.ndarray = score.cpu().numpy()
    print(f'consistency score: {score}')
    print(f'median: {np.median(score):.3f}, mean: {score.mean():.3f}')
    return score   #.mean()


def eval_colmap(demo_fp: str, colmap_proj_root: str, colmap_path='colmap'):
    colmap_proj_dir = os.path.join(colmap_proj_root, 'colmap')
    # if not os.path.exists(os.path.join(colmap_proj_dir, 'points.ply')):
    in_image_fp = demo_fp.replace('.png', '_colmap.png')
    colmap_demo_imgs = np.array(Image.open(in_image_fp))
    patch_match_with_known_poses(colmap_demo_imgs, colmap_proj_dir, colmap_path=colmap_path)
    if os.path.exists(os.path.join(colmap_proj_dir, 'points.ply')):
        vert, _ = io.load_ply(f'{colmap_proj_dir}/points.ply',)
    else:
        vert = []
    return len(vert)


def eval_pose_all(config, scenes, ids, wb_run: Optional[wandb.Run]):
    metric = []
    metric_dict = {}
    for scene in scenes:
        if config.data.scene_transform_fp:
            print(f"[INFO] Evaluating pose \'{scene}\' with scene_transform_fp")
            config.data.scene = scene
            pose_err = eval_pose(**config.data)
            metric.append(pose_err)
            metric_dict[scene] = pose_err
        else:
            metric_dict[scene] = {}
            for id in ids:
                print(f"[INFO] Evaluating pose \'{scene}\':{id}")
                config.data.scene = scene
                config.data.id = id
                pose_err = eval_pose(**config.data)
                metric.append(pose_err)
                metric_dict[scene][id] = pose_err
    # print(f"[INFO] metric shape: {len(metric)} x {metric[0].shape}")
    metric = np.concatenate(metric, axis=0)
    # print(f"[INFO] metric shape after:  {metric.shape}")
    np.savez(f"{config.data.exp_root}/pose_{config.data.name}.npz", metric)
    with open(f"{config.data.exp_root}/pose_{config.data.name}.json", "w") as f:
        json.dump(metric_dict, f, indent=4)
    rot_p25, rot_p50, rot_p75 = np.percentile(metric[:, 0], [25, 50, 75])
    trans_p25, trans_p50, trans_p75 = np.percentile(metric[:, 1], [25, 50, 75])
    Recall5 = sum(metric[:, 0] <= 5) / len(metric)
    Recall15 = sum(metric[:, 0] <= 15) / len(metric)
    Recall30 = sum(metric[:, 0] <= 30) / len(metric)
    if wb_run:
        wb_run.summary.update({
            'Pose/Recall(<=5)': Recall5,
            'Pose/Recall(<=15)': Recall15,
            'Pose/Recall(<=30)': Recall30,
            'Pose/Rot. error (p25)': rot_p25,
            'Pose/Rot. error (median)': rot_p50,
            'Pose/Rot. error (p75)': rot_p75,
            'Pose/Trans. error (p25)': trans_p25,
            'Pose/Trans. error (median)': trans_p50,
            'Pose/Trans. error (p75)': trans_p75
        })
    # NOTE: report the median error and recall < 5 degree
    print(
        f"Rot. error: {rot_p50:.3f}, Trans. error: {trans_p50:.3f}, Recall <=5: {Recall5:.3f}, Recall <=15: {Recall15:.3f}, Recall <=30: {Recall30:.3f}"
    )


def eval_nvs_all(config, scenes, ids, wb_run: Optional[wandb.Run]):
    metric = []
    metric_dict = {}
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Evaluating nvs \'{scene}\':{id}")
            config.data.scene = scene
            config.data.id = id
            nvs_err = eval_nvs(**config.data)
            metric.append(nvs_err)
            metric_dict[scene] = {
                'PSNR': nvs_err[0],
                'SSIM': nvs_err[1],
                'LPIPS': nvs_err[2],
            }
    metric = np.array(metric)
    np.savez(f"{config.data.nvs_root}/nvs_{config.data.name}.npz", metric)
    with open(f"{config.data.nvs_root}/nvs_{config.data.name}.json", "w") as f:
        json.dump(metric_dict, f, indent=4)
    if wb_run:
        PSNR_p25, PSNR_p50, PSNR_p75 = np.percentile(metric[:, 0], [25, 50, 75])
        SSIM_p25, SSIM_p50, SSIM_p75 = np.percentile(metric[:, 1], [25, 50, 75])
        LPIPS_p25, LPIPS_p50, LPIPS_p75 = np.percentile(metric[:, 2], [25, 50, 75])
        PSNR_mean, SSIM_mean, LPIPS_mean = np.mean(metric, axis=0)
        wb_run.summary.update({
            'NVS/PSNR (p25)': PSNR_p25,
            'NVS/PSNR (median)': PSNR_p50,
            'NVS/PSNR (p75)': PSNR_p75,
            'NVS/SSIM (p25)': SSIM_p25,
            'NVS/SSIM (median)': SSIM_p50,
            'NVS/SSIM (p75)': SSIM_p75,
            'NVS/LPIPS (p25)': LPIPS_p25,
            'NVS/LPIPS (median)': LPIPS_p50,
            'NVS/LPIPS (p75)': LPIPS_p75,
            'NVS/PSNR  (mean)': PSNR_mean,
            'NVS/SSIM  (mean)': SSIM_mean,
            'NVS/LPIPS (mean)': LPIPS_mean,
        })
    print(f"PSNR: {metric[:, 0].mean():.3f}, SSIM: {metric[:, 1].mean():.3f}, LPIPS: {metric[:, 2].mean():.3f}")


def eval_consistency_all(config, scenes, ids, wb_run: Optional[wandb.Run]):
    met3r_eval = MEt3R(
        img_size=256,                                   # Default to 256, set to `None` to use the input resolution on the fly!
        use_norm=True,                                  # Default to True
        backbone="mast3r",                              # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
        feature_backbone="dino16",                      # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
        feature_backbone_weights="mhamilton723/FeatUp", # Default
        upsampler="featup",                             # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
        distance="cosine",                              # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
        freeze=True,                                    # Default to True
    ).cuda()

    metric = []
    metric_dict = {}
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Evaluating consistency \'{scene}\':{id}")
            config.data.scene = scene
            config.data.id = id
            consistency_score = eval_consistency(met3r_eval, **config.data)
            metric.append(consistency_score)
            metric_dict[scene] = consistency_score.squeeze().tolist()
    metric = np.concatenate(metric, axis=0)
    np.savez(f"{config.data.nvs_root}/consistency_{config.data.name}.npz", metric)
    with open(f"{config.data.nvs_root}/consistency_{config.data.name}.json", "w") as f:
        json.dump(metric_dict, f, indent=4)
    MEt3R_mean = np.mean(metric[:])
    if wb_run:
        MEt3R_p25, MEt3R_p50, MEt3R_p75 = np.percentile(metric[:], [25, 50, 75])
        wb_run.summary.update({
            'Consistentcy/Recall(<=0.1)': sum(metric[:] <= 0.1) / len(metric),
            'Consistentcy/Recall(<=0.2)': sum(metric[:] <= 0.2) / len(metric),
            'Consistentcy/Recall(<=0.3)': sum(metric[:] <= 0.3) / len(metric),
            'Consistentcy/MEt3R error (p25)': MEt3R_p25,
            'Consistentcy/MEt3R error (median)': MEt3R_p50,
            'Consistentcy/MEt3R error (p75)': MEt3R_p75,
            'Consistentcy/MEt3R error (mean)': MEt3R_mean,
        })
    print(f"Consistency score: {MEt3R_mean:.3f}, Recall: {sum(metric[:] <= 0.1) / len(metric):.3f}")


def eval_colmap_all(config, scenes, ids, wb_run: Optional[wandb.Run]):
    colmap_metric = []
    metric_dict = {}
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Evaluating colmap \'{scene}\':{id}")
            config.data.scene = scene
            config.data.id = id
            colmap_points = eval_colmap(config.inference.demo_fp, config.data.nvs_dir)
            colmap_metric.append(colmap_points)
            metric_dict[scene] = colmap_points
    # colmap_metric = np.concatenate(colmap_metric, axis=0)
    colmap_metric = np.array(colmap_metric)
    print(colmap_metric.shape)
    np.savez(f"{config.data.nvs_root}/colmap_{config.data.name}.npz", colmap_metric)
    with open(f"{config.data.exp_root}/colmap_{config.data.name}.json", "w") as f:
        json.dump(metric_dict, f, indent=4)
    colmap_mean = np.mean(colmap_metric[:])
    if wb_run:
        colmap_p25, colmap_p50, colmap_p75 = np.percentile(colmap_metric[:], [25, 50, 75])
        wb_run.summary.update({
            'COLMAP/colmap points (p25)': colmap_p25,
            'COLMAP/colmap points (median)': colmap_p50,
            'COLMAP/colmap points (p75)': colmap_p75,
            'COLMAP/colmap points (mean)': colmap_mean,
        })
    print(f"COLMAP points: {colmap_mean:.3f}")


def main(config, mode):
    config, conf_dict = config
    wb_run = start_wabdb(config, conf_dict, mode, eval=True)
    perm = list(itertools.permutations(range(5), 2))
    # perm = list(itertools.combinations(range(5), 2))
    # perm = list(itertools.combinations(range(3), 2))
    ids = [",".join(map(str, p)) for p in perm]
    scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))[0:]
    print(f"[INFO] Found {len(scenes)} scenes")
    if mode[0]:
        eval_pose_all(config, scenes, ids=ids, wb_run=wb_run)
    if mode[1]:
        eval_nvs_all(config, scenes, ids=["0,1"], wb_run=wb_run)
    if mode[2]:
        eval_consistency_all(config, scenes, ids=["0,1"], wb_run=wb_run)
    if mode[3]:
        eval_colmap_all(config, scenes, ids=["0,1"], wb_run=wb_run)
    if wb_run:
        wb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    parser.add_argument('-p', "--pose", action="store_true")
    parser.add_argument('-n', "--nvs", action="store_true")
    parser.add_argument('-c', "--consistency", action="store_true")
    parser.add_argument('-3', "--colmap", action="store_true")
    parser.add_argument('-g', '--gpu_id', type=str, default='4')
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)

    set_random_seed(config[0].seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(config, [args.pose, args.nvs, args.consistency, args.colmap])
