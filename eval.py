import argparse
import itertools
import os

import numpy as np
import torch

from dataset.base import load_frames

from util.criterion import lpips_fn, pose_err_fn, psnr_fn, ssim_fn
from util.util import load_config, load_image, set_random_seed, str2list
from util.pose import mat2latlon
from met3r import MEt3R
from PIL import Image

def eval_consistency(met3r_eval, exp_dir, id, **kwargs):
    # Prepare inputs of shape (batch, views, channels, height, width): views must be 2
    # RGB range must be in [-1, 1]
    # print(f'img shape:{len(img)},{img[0].shape}') 2, 3, 256, 256
    imgs = []
    for fp in os.listdir(exp_dir):
        if 'demo_' in fp:
            imgs.append(load_image(os.path.join(exp_dir, fp),verbose=False))
    imgs = torch.cat(imgs)
    inputs = []
    # print(f'imgs lens : {imgs.shape[0] - 1}')
    for i in range(imgs.shape[0] - 1):
        inputs.append(torch.Tensor(imgs[i : i + 2]))
    inputs= torch.stack(inputs).cuda()
    # fp_a = os.path.join(kwargs['image_dir'], '0.png')
    # fp_b = os.path.join(kwargs['image_dir'], '0.png')
    # a = load_image(fp_a,verbose=False)
    # b = load_image(fp_b,verbose=False)
    # print(f"Image A: {fp_a}\nImage B: {fp_b}")
    # inputs = torch.cat((a, b)).cuda().unsqueeze(0)
    # inputs = torch.randn((8, 2, 3, 256, 256)).cuda()
    inputs = inputs.clip(-1, 1)
    score, mask, score_map = met3r_eval(
        images=inputs, 
        return_overlap_mask=True, # Default 
        return_score_map=True, # Default 
        return_projections=False # Default 
    )
    # mask = mask.cpu()
    # score_map = score_map.cpu()
    # Image.fromarray(((inputs[0, 0].permute(1, 2, 0) +1).cpu().numpy() / 2 * 255).astype(np.uint8)).save(f'{exp_dir}/input_{id[0]}.png')
    # Image.fromarray(((inputs[0, 1].permute(1, 2, 0) +1).cpu().numpy() / 2 * 255).astype(np.uint8)).save(f'{exp_dir}/input_{id[2]}.png')
    # print(f'score map range : {score_map[0].min()} to {score_map[0].max()}')
    # print(f'mask range : {mask[0].min()} to {mask[0].max()}')
    for i in range(score_map.shape[0]):
        temp = torch.stack([score_map[i], score_map[i], score_map[i]], axis=-1).clamp(0,1) *  torch.stack([mask[i]+0.3, mask[i]+0.3, torch.full_like(mask[i], 0.8)], axis=-1).clamp(0,1)
        Image.fromarray((score_map[i].clamp(0,1).cpu().numpy() * 255).astype(np.uint8)).save(f'{exp_dir}/score_map_{i},{i+1}.png')
        Image.fromarray((temp * 255).cpu().numpy().astype(np.uint8)).save(f'{exp_dir}/score_map_masked_{i},{i+1}.png')
        Image.fromarray((mask[i].clamp(0,1).cpu().numpy() * 255).astype(np.uint8)).save(f'{exp_dir}/mask_{i},{i+1}.png')
        # Image.fromarray(((projections[i].cpu().numpy() +1) / 2 * 255).astype(np.uint8)).save(f'{exp_dir}/projections_{i},{i+1}.png')
    np.set_printoptions(precision=3, suppress=None, floatmode='fixed')
    print(f'consistency score: {score.cpu().numpy()}')
    print(f'median: {score.median().item():.3f}, mean: {score.mean().item():.3f}')
    return score.mean().item()#, score.median().item()

def eval_pose(transform_fp, gt_transform_fp, image_dir, exp_dir, id, **kwargs):
    camtoworlds = load_frames(image_dir, transform_fp, verbose=False, return_images=True)[1]
    # print(f'given_c2w 0 = {mat2latlon(camtoworlds[0], in_deg=True, return_radius=True)[0]}')
    # print(f'est_c2w   1 = {mat2latlon(camtoworlds[1], in_deg=True, return_radius=True)[0]}')
    gt_camtoworlds = load_frames(image_dir, gt_transform_fp, verbose=False)[1]
    
    gt_camtoworlds = gt_camtoworlds[str2list(id)]
    # print(f'gt_c2w    0 = {mat2latlon(gt_camtoworlds[0], in_deg=True, return_radius=True)[0]}')
    # print(f'gt_c2w    1 = {mat2latlon(gt_camtoworlds[1], in_deg=True, return_radius=True)[0]}')

    pose_err = [pose_err_fn(pred, gt) for pred, gt in zip(camtoworlds[1:], gt_camtoworlds[1:])]
    pose_err = np.array(pose_err).mean(axis=0)
    print(f"Rot. error: {pose_err[0]:.2f}, Trans. error: {pose_err[1]:.2f}")
    return pose_err

def eval_nvs(demo_fp, test_image_dir, test_transform_fp, **kwargs):
    pred = load_image(demo_fp, resize=False, to_clip=False)
    pred = torch.cat(torch.chunk(pred, 8, dim=-1))
    gt = load_frames(test_image_dir, test_transform_fp, to_clip=False)[0]
    gt = gt.to(pred.device)

    psnr = psnr_fn(pred, gt).item()
    ssim = ssim_fn(pred, gt).item()
    lpips = lpips_fn(pred, gt).item()
    return psnr, ssim, lpips

def eval_pose_all(config, scenes, ids):
    metric = []
    scenes = [scenes[0], scenes[2]]
    ids = [ids[0], ids[3]]
    # for scene in scenes:
    #     for id in ids:
    for scene, id in zip(scenes, ids):
            print(f"[INFO] Evaluating pose {scene}:{id}")
            config.data.scene = scene
            config.data.id = id
            pose_err = eval_pose(**config.data)

            metric.append(pose_err)
    metric = np.array(metric)
    np.savez(f"{config.data.exp_root_dir}/pose_{config.data.name}.npz", metric)

    # NOTE: report the median error and recall < 5 degree
    print(f"Rot. error: {np.median(metric[:, 0]):.3f}, Trans. error: {np.median(metric[:, 1]):.3f}, Recall <=5: {sum(metric[:, 0] <= 5) / len(metric)}, Recall <=15: {sum(metric[:, 0] <= 15) / len(metric)}, Recall <=30: {sum(metric[:, 0] <= 30) / len(metric)}")

def eval_consistency_all(config, scenes, ids):
    met3r_eval = MEt3R(
        img_size=256, # Default to 256, set to `None` to use the input resolution on the fly!
        use_norm=True, # Default to True 
        backbone="mast3r", # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
        feature_backbone="dino16", # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
        feature_backbone_weights="mhamilton723/FeatUp", # Default
        upsampler="featup", # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
        distance="cosine", # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
        freeze=True, # Default to True
    ).cuda()
    consistency_metric = []
    scenes = [scenes[0], scenes[2]]
    ids = [ids[0], ids[3]]
    # for scene in scenes:
    #     for id in ids:
    for scene, id in zip(scenes, ids):
            print(f"[INFO] Evaluating consistency {scene}:{id}")
            config.data.scene = scene
            config.data.id = id
            consistency_score = eval_consistency(met3r_eval, **config.data)

            consistency_metric.append(consistency_score)
    consistency_metric = np.array(consistency_metric)
    # np.savez(f"{config.data.exp_root_dir}/pose_{config.data.name}.npz", metric)

    print(f"Consistency score: {np.mean(consistency_metric[:]):.3f}, Recall: {sum(consistency_metric[:] <= 0.1) / len(consistency_metric):.3f}")


def eval_nvs_all(config, scenes, ids):
    metric = []
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Evaluating nvs {scene}:{id}")
            config.data.scene = scene
            config.data.id = id
            metric.append(eval_nvs(**config.data))
    metric = np.array(metric)
    np.savez(f"{config.data.exp_root_dir}/nvs_{config.data.name}.npz", metric)
    print(
        f"PSNR: {metric[:, 0].mean()}, SSIM: {metric[:, 1].mean()}, LPIPS: {metric[:, 2].mean()}"
    )


def main(config, mode):
    perm = list(itertools.combinations(range(5), 2))
    ids = [",".join(map(str, p)) for p in perm]
    scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))[0:3]
    print(f"[INFO] Found {len(scenes)} scenes: {scenes}")
    if mode[0]:
        eval_pose_all(config, scenes, ids)
    if mode[1]:
        eval_nvs_all(config, scenes, ids=["0,1"])
    if mode[2]:
        eval_consistency_all(config, scenes, ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    parser.add_argument("--consistency", action="store_true")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--nvs", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="4")
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    
    set_random_seed(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    main(config, [args.pose, args.nvs, args.consistency])
