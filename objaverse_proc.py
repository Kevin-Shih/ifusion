# Dataset name ---scene------ train ---- rgb --- 5 views png
#                     |              |
#                     |              -- transform.json
#                     |
#                     |
#                     ------- test  ---- rgb --- 8 views png
#                                    |
#                                    -- transform.json

import json
import os
import torch
import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from util.pose import mat2latlon, latlon2mat


def procees_data(step, dir, pbar):
    pbar.set_description(f'dir for {"train" if step < 772000 else "test"}')
    scene_dir = os.path.join("rendering/Objaverse/", dir)
    sub_dir = os.path.join(scene_dir, 'train') if step < 772000 else os.path.join(scene_dir, 'test')
    img_dir = os.path.join(sub_dir, 'rgb')
    os.makedirs(img_dir, exist_ok=True)
    transforms_dict = {"camera_angle_x": 0.8569566627292158, "frames": []}
    for file in sorted(os.listdir(scene_dir)):
        if file.endswith('.npy'):
            idx = int(file[:3])
            transform = np.load(f"{scene_dir}/{idx:03d}.npy")
            R, T = transform[:3, :3], transform[:, -1]
            T_cond = -R.T @ T
            T_cond = T_cond[None, :]
            # ptsnew = np.hstack((T_cond, np.zeros(T_cond.shape)))
            xy = T_cond[:, 0]**2 + T_cond[:, 1]**2
            z = np.sqrt(xy + T_cond[:, 2]**2)
            theta = np.arctan2(np.sqrt(xy), T_cond[:, 2]) # for elevation angle defined from Z-axis down
            azimuth = np.arctan2(T_cond[:, 1], T_cond[:, 0])
            latlon = torch.tensor(np.array([[np.rad2deg(theta) - 90, np.rad2deg(azimuth) + 90, z]]))

            transforms_dict['frames'].append({
                "file_path": f"{idx}.png",
                "transform_matrix": latlon2mat(latlon.clone()).squeeze(0).tolist(),
                "latlon": latlon.squeeze().tolist()
            })
        if file.endswith('.png'):
            idx = int(file[:3])
            os.rename(f"{scene_dir}/{idx:03d}.png", f"{img_dir}/{idx}.png")
    with open(os.path.join(sub_dir, 'transform.json'), "w") as transforms_f:
        json.dump(transforms_dict, transforms_f, indent=4)
    return f'{"train" if step < 772000 else "test"}'


def test_imgs(valid_path):
    with tqdm.trange(len(valid_path), ncols=100) as pbar:
        for step in pbar:
            pbar.set_description(f'dir for {"train" if step < 772000 else "test"}')
            yield len([
                file for file in os.listdir(os.path.join("rendering/Objaverse/", valid_path[step]))
                if file.endswith('.npy')
            ]), valid_path[step]


if __name__ == "__main__":
    with open(f"rendering/Objaverse/my_valid_paths.json") as f:
        valid_path: list = json.load(f)
    # 772000 as train 870 as eval
    train_or_not = []
    with tqdm.trange(len(valid_path[:]), ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(procees_data, step, valid_path[step], pbar) for step in pbar]
            for future in as_completed(futures):
                pbar.update(1)
                train_or_not.append(future.result())
        print(train_or_not[-5:])
    test = np.zeros(16, dtype=np.int32)
    dirs = test_imgs(valid_path)
    print(type(valid_path), len(valid_path))
    remove_list = []
    for dir in dirs:
        test[dir[0]] += 1
        if dir[0] < 12:
            remove_list.append(dir[1])
    # for it in remove_list:
    #     valid_path.remove(it)
    # print(type(valid_path), len(valid_path))
    # with open(f"rendering/Objaverse/my_valid_paths.json", mode='w') as f:
    #     json.dump(valid_path, f)
    for i in range(1, 16):
        if test[i] > 0:
            print(f'Dir with {i:2d} images = {test[i]:6d}', end=', ')
