import argparse
import itertools
import os
import wandb
import torch
import numpy as np

from dataset.base import load_frames
from util.criterion import lpips_fn, pose_err_fn, psnr_fn, ssim_fn
from util.util import load_config, load_image, set_random_seed, str2list
from util.pose import mat2latlon

from rich import print
from met3r import MEt3R
from PIL import Image

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

def eval_nvs(demo_fp, test_image_dir, test_transform_fp, **kwargs)->tuple[float,float,float]:
    pred = load_image(demo_fp, resize=False, to_clip=False)
    #print(f'[INFO 1] Loaded demo image {demo_fp} with shape {pred.shape}')
    # print(f'[INFO 2] chunked demo image shape {torch.chunk(pred, 8, dim=-1)[0].shape}')
    pred = torch.cat(torch.chunk(pred, 8, dim=-1))
    # print(f'[INFO 3] chunk and cat demo image shape {pred.shape}')
    gt = load_frames(test_image_dir, test_transform_fp, to_clip=False)[0]
    gt = gt.to(pred.device)

    psnr = psnr_fn(pred, gt).item()
    ssim = ssim_fn(pred, gt).item()
    lpips = lpips_fn(pred, gt).item()
    return psnr, ssim, lpips

def eval_consistency(met3r_eval, exp_dir, id, **kwargs):
    # Prepare inputs of shape (batch, views, channels, height, width): views must be 2
    # RGB range must be in [-1, 1]
    # print(f'img shape:{len(img)},{img[0].shape}') 2, 3, 256, 256
    # imgs = []
    # for fp in os.listdir(exp_dir):
    #     if 'demo_' in fp:
    #         imgs.append(load_image(os.path.join(exp_dir, fp),verbose=False))
    # imgs = torch.cat(imgs)
    # inputs = []
    # # print(f'imgs lens : {imgs.shape[0] - 1}')
    # for i in range(imgs.shape[0] - 1):
    #     inputs.append(torch.Tensor(imgs[i : i + 2]))
    # inputs= torch.stack(inputs).cuda()
    fp_a = '/eva_data0/kevinshih/zero123-adv/ifusion/exp_dualway_ifusion/GSO/3D_Dollhouse_Sofa/0,4/demo_1.png'
    fp_b = '/eva_data0/kevinshih/zero123-adv/ifusion/exp_dualway_ifusion/GSO/3D_Dollhouse_Sofa/0,4/demo_2.png'
    a = load_image(fp_a,verbose=False)
    b = load_image(fp_b,verbose=False)
    print(f"Image A: {fp_a}\nImage B: {fp_b}")
    inputs = torch.cat((a, b)).cuda().unsqueeze(0)
    # inputs = torch.randn((8, 2, 3, 256, 256)).cuda()
    inputs = inputs.clip(-1, 1)
    score, mask, score_map, pcloud = met3r_eval(
        images=inputs, 
        return_overlap_mask = True,
        return_score_map = True,
        return_projections = False, # Default 
        return_point_clouds = True
    )
    # pcloud : (B-1) * 2 point clouds. Each image pair has 2 point clouds (left and right)
    # mask = mask.cpu()
    # score_map = score_map.cpu()
    # Image.fromarray(((inputs[0, 0].permute(1, 2, 0) +1).cpu().numpy() / 2 * 255).astype(np.uint8)).save(f'{exp_dir}/input_{id[0]}.png')
    # Image.fromarray(((inputs[0, 1].permute(1, 2, 0) +1).cpu().numpy() / 2 * 255).astype(np.uint8)).save(f'{exp_dir}/input_{id[2]}.png')
    # print(f'score map range : {score_map[0].min()} to {score_map[0].max()}')
    # print(f'mask range : {mask[0].min()} to {mask[0].max()}')
    # from pytorch3d.io import IO
    from pytorch3d import io
    # from pytorch3d.structures import Pointclouds
    temp = torch.stack([score_map[0], score_map[0], score_map[0]], axis=-1).clamp(0,1) *  torch.stack([mask[0]+0.3, mask[0]+0.3, torch.full_like(mask[0], 0.8)], axis=-1).clamp(0,1)
    Image.fromarray((score_map[0].clamp(0,1).cpu().numpy() * 255).astype(np.uint8)).save(f'{exp_dir}/score_map_1,2.png')
    Image.fromarray((temp * 255).cpu().numpy().astype(np.uint8)).save(f'{exp_dir}/score_map_masked_1,2.png')
    Image.fromarray((mask[0].clamp(0,1).cpu().numpy() * 255).astype(np.uint8)).save(f'{exp_dir}/mask_1,2.png')
    # Image.fromarray(((projections[i].cpu().numpy() +1) / 2 * 255).astype(np.uint8)).save(f'{exp_dir}/projections_{i},{i+1}.png')
    # print(f'verts:{pcloud.points_list()[0].shape}, feats:{pcloud.features_list()[0].shape}')
    if met3r_eval.distance == 'cosine':
        # no colors is met3r choose cosine distance as metric
        print(f'[INFO] Save pointclouds without color')
        io.save_ply(f'{exp_dir}/pointcloud_1,2_l.ply', pcloud.points_list()[0])
        io.save_ply(f'{exp_dir}/pointcloud_1,2_r.ply', pcloud.points_list()[1])
    else:
        # temp = Pointclouds(points=[pcloud.points_list()[i*2]], features=[(pcloud.features_list()[i*2]*255).clamp(0,255).to(torch.uint8)])
        # io.IO().save_pointcloud(temp, f'{exp_dir}/pointcloud_{i},{i+1}_l.ply')
        io.IO().save_pointcloud(pcloud[0], f'{exp_dir}/pointcloud_1,2_l.ply')
        io.IO().save_pointcloud(pcloud[1], f'{exp_dir}/pointcloud_1,2_r.ply')
            
    np.set_printoptions(precision=3, suppress=None, floatmode='fixed')
    print(f'consistency score: {score.cpu().numpy()}')
    print(f'median: {score.median().item():.3f}, mean: {score.mean().item():.3f}')
    return score.mean().item()#, score.median().item()

def eval_pose_all(config, scenes, ids, wb_run):
    metric = []
    # scenes = [scenes[0], scenes[2]]
    # ids = [ids[0], ids[3]]
    # for scene, id in zip(scenes, ids):
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Evaluating pose \'{scene}\':{id}")
            config.data.scene = scene
            config.data.id = id
            pose_err = eval_pose(**config.data)

            metric.append(pose_err)
    metric = np.array(metric)
    np.savez(f"{config.data.exp_root_dir}/pose_{config.data.name}.npz", metric)
    if wb_run:
        rot_p25, rot_p50, rot_p75 = np.percentile(metric[:, 0], [25, 50, 75])
        trans_p25, trans_p50, trans_p75 = np.percentile(metric[:, 1], [25, 50, 75])
        wb_run.summary['Pose/Recall(<=5)']  = sum(metric[:, 0] <=  5) / len(metric)
        wb_run.summary['Pose/Recall(<=15)'] = sum(metric[:, 0] <= 15) / len(metric)
        wb_run.summary['Pose/Recall(<=30)'] = sum(metric[:, 0] <= 30) / len(metric)
        wb_run.summary['Pose/Rot. error (p25)']      = rot_p25
        wb_run.summary['Pose/Rot. error (median)']   = rot_p50
        wb_run.summary['Pose/Rot. error (p75)']      = rot_p75
        wb_run.summary['Pose/Trans. error (p25)']    = trans_p25
        wb_run.summary['Pose/Trans. error (median)'] = trans_p50
        wb_run.summary['Pose/Trans. error (p75)']    = trans_p75
    # NOTE: report the median error and recall < 5 degree
    print(f"Rot. error: {np.median(metric[:, 0]):.3f}, Trans. error: {np.median(metric[:, 1]):.3f}, Recall <=5: {sum(metric[:, 0] <= 5) / len(metric):.3f}, Recall <=15: {sum(metric[:, 0] <= 15) / len(metric):.3f}, Recall <=30: {sum(metric[:, 0] <= 30) / len(metric):.3f}")

def eval_nvs_all(config, scenes, ids, wb_run):
    metric = []
    # scenes = [scenes[0], scenes[2]]
    # ids = [ids[0], ids[3]]
    # for scene, id in zip(scenes, ids):
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Evaluating nvs \'{scene}\':{id}")
            config.data.scene = scene
            config.data.id = id
            metric.append(eval_nvs(**config.data))
    metric = np.array(metric)
    np.savez(f"{config.data.exp_root_dir}/nvs_{config.data.name}.npz", metric)
    if wb_run:
        PSNR_p25, PSNR_p50, PSNR_p75    = np.percentile(metric[:, 0], [25, 50, 75])
        SSIM_p25, SSIM_p50, SSIM_p75    = np.percentile(metric[:, 1], [25, 50, 75])
        LPIPS_p25, LPIPS_p50, LPIPS_p75 = np.percentile(metric[:, 2], [25, 50, 75])
        PSNR_mean  = np.mean(metric[:, 0])
        SSIM_mean  = np.mean(metric[:, 1])
        LPIPS_mean = np.mean(metric[:, 2])
        wb_run.summary['NVS/PSNR (p25)']     = PSNR_p25
        wb_run.summary['NVS/PSNR (median)']  = PSNR_p50
        wb_run.summary['NVS/PSNR (p75)']     = PSNR_p75
        wb_run.summary['NVS/SSIM (p25)']     = SSIM_p25
        wb_run.summary['NVS/SSIM (median)']  = SSIM_p50
        wb_run.summary['NVS/SSIM (p75)']     = SSIM_p75
        wb_run.summary['NVS/LPIPS (p25)']    = LPIPS_p25
        wb_run.summary['NVS/LPIPS (median)'] = LPIPS_p50
        wb_run.summary['NVS/LPIPS (p75)']    = LPIPS_p75
        wb_run.summary['NVS/PSNR  (mean)']   = PSNR_mean
        wb_run.summary['NVS/SSIM  (mean)']   = SSIM_mean
        wb_run.summary['NVS/LPIPS (mean)']   = LPIPS_mean
    print(
        f"PSNR: {metric[:, 0].mean()}, SSIM: {metric[:, 1].mean()}, LPIPS: {metric[:, 2].mean()}"
    )

def eval_consistency_all(config, scenes, ids, wb_run):
    met3r_eval = MEt3R(
        img_size=256, # Default to 256, set to `None` to use the input resolution on the fly!
        use_norm=True, # Default to True 
        backbone="dust3r", # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
        feature_backbone="dino16", # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
        feature_backbone_weights="mhamilton723/FeatUp", # Default
        upsampler="featup", # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
        distance="mse", # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
        freeze=True, # Default to True
    ).cuda()
    consistency_metric = []
    scenes = [scenes[2]]
    ids = [ids[3]]
    for scene, id in zip(scenes, ids):
    # for scene in scenes:
        # for id in ids:
            print(f"[INFO] Evaluating consistency \'{scene}\':{id}")
            config.data.scene = scene
            config.data.id = id
            consistency_score = eval_consistency(met3r_eval, **config.data)

            consistency_metric.append(consistency_score)
    consistency_metric = np.array(consistency_metric)
    # np.savez(f"{config.data.exp_root_dir}/pose_{config.data.name}.npz", metric)
    MEt3R_mean = np.mean(consistency_metric[:])
    if wb_run:
        MEt3R_p25, MEt3R_p50, MEt3R_p75 = np.percentile(consistency_metric[:], [25, 50, 75])
        wb_run.summary['Consistentcy/Recall(<=0.1)'] = sum(consistency_metric[:] <= 0.1) / len(consistency_metric)
        wb_run.summary['Consistentcy/Recall(<=0.2)'] = sum(consistency_metric[:] <= 0.2) / len(consistency_metric)
        wb_run.summary['Consistentcy/Recall(<=0.3)'] = sum(consistency_metric[:] <= 0.3) / len(consistency_metric)
        wb_run.summary['Consistentcy/MEt3R error (p25)']      = MEt3R_p25
        wb_run.summary['Consistentcy/MEt3R error (median)']   = MEt3R_p50
        wb_run.summary['Consistentcy/MEt3R error (p75)']      = MEt3R_p75
        wb_run.summary['Consistentcy/MEt3R error (mean)']     = MEt3R_mean
    print(f"Consistency score: {MEt3R_mean:.3f}, Recall: {sum(consistency_metric[:] <= 0.1) / len(consistency_metric):.3f}")

def main(config, mode):
    config, conf_dict = config
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
                settings=wandb.Settings(x_disable_stats=True),
                id=f"{config.log.run_path.split('/')[-1]}",
                resume="must",
            )
        else:
            print(f'Canceled: Abort execution.')
            exit(0)
    else:
        wb_run = wandb.init(
            dir="./wandb/eval",
            entity="kevin-shih",
            project="iFusion-Adv",
            group= f'{config.log.group_name}',
            name= f'{config.log.run_name}',
            settings=wandb.Settings(x_disable_stats=True),
            config={
                    **conf_dict,
            },
        )
    perm = list(itertools.permutations(range(5), 2))
    ids = [",".join(map(str, p)) for p in perm]
    scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))[0:]
    print(f"[INFO] Found {len(scenes)} scenes: {scenes}")
    if mode[0]:
        eval_pose_all(config, scenes, ids=ids, wb_run=wb_run)
    if mode[1]:
        eval_nvs_all(config, scenes, ids=ids, wb_run=wb_run)
    if mode[2]:
        eval_consistency_all(config, scenes, ids=ids, wb_run=wb_run)
    if wb_run:
        wb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    parser.add_argument('-c', "--consistency", action="store_true")
    parser.add_argument('-p', "--pose", action="store_true")
    parser.add_argument('-n', "--nvs", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="4")
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    
    set_random_seed(config[0].seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    main(config, [args.pose, args.nvs, args.consistency])
